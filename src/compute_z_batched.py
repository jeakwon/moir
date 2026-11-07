"""Batched compute_z — optimise N requests' delta vectors simultaneously.

Mathematically equivalent to calling ``compute_z()`` N times in sequence,
but runs all N×P prompts through a single forward/backward pass per
optimisation step.

Optimisations over naive batching:
  - ln_f @ lm_w computed ONCE for all prompts (not per-request)
  - NLL computed only at target-token positions (not full seq × vocab)
  - KL computed once (vectorised), no duplicate
  - TraceDict registered once (not per step)
  - Per-request loss aggregation via scatter_add (no Python loop)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.util import nethook
from src.models.memit.memit_hparams import MEMITHyperParams
from src.models.memit.compute_z import find_fact_lookup_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_z_batched(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    batch_size: int = 32,
    early_stopping: bool = True,
    step_callback: Optional[Callable] = None,
) -> List[torch.Tensor]:
    """Compute z* (target + delta) for *N* requests in parallel.

    Args:
        early_stopping: If True (default), stop optimising requests whose
            loss falls below threshold. Set False for per-step analysis.
        step_callback: If provided, called after each step with signature
            ``callback(step, deltas, target_inits, per_req_loss, total_loss)``.
            All tensors are detached. Useful for snapshotting z* per step.
    """
    device = f"cuda:{hparams.device}"
    N = len(requests)
    if N == 0:
        return []

    results: List[torch.Tensor] = [None] * N  # type: ignore[assignment]

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_requests = requests[batch_start:batch_end]
        batch_results = _compute_z_batch(
            model, tok, batch_requests, hparams, layer, context_templates,
            device,
            early_stopping=early_stopping,
            step_callback=step_callback,
        )
        for local_i, z in enumerate(batch_results):
            results[batch_start + local_i] = z.detach().cpu()
        del batch_results
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Core batched optimisation (optimised inner loop)
# ---------------------------------------------------------------------------

def _compute_z_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    device: str,
    early_stopping: bool = True,
    step_callback: Optional[Callable] = None,
) -> List[torch.Tensor]:
    """Optimise delta vectors for a single batch of requests."""

    B = len(requests)

    # ---- model constants (cached on device) -----------------------------
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    lm_w_dev = lm_w.to(device)
    lm_b_dev = lm_b.to(device)

    hidden_dim = (model.config.n_embd
                  if hasattr(model.config, 'n_embd')
                  else model.config.hidden_size)

    # ---- per-request target token ids -----------------------------------
    target_ids_list: List[torch.Tensor] = []
    for req in requests:
        tids = tok.encode(
            req["target_new"], return_tensors="pt",
            add_special_tokens=False,
        ).to(device)[0]
        if tids[0] == tok.bos_token_id or tids[0] == tok.unk_token_id:
            tids = tids[1:]
        target_ids_list.append(tids)

    # ---- build flat prompt list -----------------------------------------
    all_prompts: List[str] = []
    prompt_to_req: List[int] = []
    prompt_is_kl: List[bool] = []
    n_rewriting_per_req: List[int] = []
    n_kl_per_req: List[int] = []

    for ri, req in enumerate(requests):
        tids = target_ids_list[ri]
        rw_prompts = [
            context.format(req["prompt"]).replace("{}", req["subject"], 1)
            + tok.decode(tids[:-1])
            for ctx_types in context_templates
            for context in ctx_types
        ]
        kl_prompts_raw = ["{} is a"]
        kl_prompts = [p.replace("{}", req["subject"], 1) for p in kl_prompts_raw]

        n_rw = len(rw_prompts)
        n_kl = len(kl_prompts)

        all_prompts.extend(rw_prompts + kl_prompts)
        prompt_to_req.extend([ri] * (n_rw + n_kl))
        prompt_is_kl.extend([False] * n_rw + [True] * n_kl)
        n_rewriting_per_req.append(n_rw)
        n_kl_per_req.append(n_kl)

    total_prompts = len(all_prompts)

    # ---- tokenise -------------------------------------------------------
    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    seq_len = input_tok["input_ids"].shape[1]

    # ---- rewriting targets (NLL mask) -----------------------------------
    rewriting_targets = torch.full(
        (total_prompts, seq_len), -100, dtype=torch.long, device=device,
    )
    for pi in range(total_prompts):
        if prompt_is_kl[pi]:
            continue
        ri = prompt_to_req[pi]
        tids = target_ids_list[ri]
        ex_len = input_tok["attention_mask"][pi].sum()
        rewriting_targets[pi, ex_len - len(tids): ex_len] = tids

    # ---- lookup indices -------------------------------------------------
    all_template_prompts: List[str] = []
    all_subjects: List[str] = []
    for ri, req in enumerate(requests):
        tids = target_ids_list[ri]
        rw_templates = [
            context.format(req["prompt"]) + tok.decode(tids[:-1])
            for ctx_types in context_templates
            for context in ctx_types
        ]
        kl_templates = ["{} is a"]
        all_template_prompts.extend(rw_templates + kl_templates)
        all_subjects.extend([req["subject"]] * (n_rewriting_per_req[ri] + n_kl_per_req[ri]))

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, subject, tok, hparams.fact_token, verbose=False,
        )
        for prompt, subject in zip(all_template_prompts, all_subjects)
    ]

    # ---- group prompt indices by request --------------------------------
    rw_indices_per_req: List[List[int]] = [[] for _ in range(B)]
    kl_indices_per_req: List[List[int]] = [[] for _ in range(B)]
    for pi in range(total_prompts):
        ri = prompt_to_req[pi]
        if prompt_is_kl[pi]:
            kl_indices_per_req[ri].append(pi)
        else:
            rw_indices_per_req[ri].append(pi)

    # ---- pre-compute vectorised indices (ONCE) --------------------------
    prompt_to_req_t = torch.tensor(prompt_to_req, dtype=torch.long, device=device)
    lookup_idxs_t = torch.tensor(lookup_idxs, dtype=torch.long, device=device)

    # Rewriting prompt indices
    rw_global_idxs = torch.tensor(
        [pi for pi in range(total_prompts) if not prompt_is_kl[pi]],
        dtype=torch.long, device=device,
    )
    # KL prompt indices
    kl_global_idxs = torch.tensor(
        [pi for pi in range(total_prompts) if prompt_is_kl[pi]],
        dtype=torch.long, device=device,
    )

    total_rw = len(rw_global_idxs)
    total_kl = len(kl_global_idxs)

    rw_to_req = prompt_to_req_t[rw_global_idxs]   # (total_rw,)
    kl_to_req = prompt_to_req_t[kl_global_idxs]   # (total_kl,)

    # Target token length per rewriting prompt (for NLL normalisation)
    target_lens_per_rw = torch.tensor(
        [len(target_ids_list[prompt_to_req[pi.item()]]) for pi in rw_global_idxs],
        dtype=torch.float, device=device,
    )

    # Number of rewriting prompts per request (for per-req averaging)
    n_rw_per_req = torch.zeros(B, dtype=torch.float, device=device)
    for ri in range(B):
        n_rw_per_req[ri] = len(rw_indices_per_req[ri])

    # Pre-extract rewriting targets and find target-token positions
    all_rw_targets = rewriting_targets[rw_global_idxs]        # (total_rw, seq_len)
    # Sparse target positions: (rw_local_idx, seq_pos) where target != -100
    tgt_rw_idx, tgt_seq_pos = torch.where(all_rw_targets != -100)
    tgt_token_ids = all_rw_targets[tgt_rw_idx, tgt_seq_pos]   # target token IDs
    # Map sparse target → request index
    tgt_to_rw = tgt_rw_idx                                     # which rw prompt
    tgt_to_req = rw_to_req[tgt_rw_idx]                         # which request
    tgt_lens = target_lens_per_rw[tgt_rw_idx]                  # target len for norm

    # KL lookup positions
    kl_lookup_pos = lookup_idxs_t[kl_global_idxs]  # (total_kl,)

    # ---- optimisation setup ---------------------------------------------
    loss_layer = max(hparams.v_loss_layer, layer)

    deltas = torch.zeros((B, hidden_dim), requires_grad=True, device=device)
    opt = torch.optim.Adam([deltas], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    target_inits: List[torch.Tensor | None] = [None] * B
    target_init_norms: torch.Tensor | None = None
    kl_distr_init: torch.Tensor | None = None

    converged = torch.zeros(B, dtype=torch.bool, device=device)

    # ---- edit hook -------------------------------------------------------
    def edit_output_fn(cur_out, cur_layer):
        def _unwrap(output):
            if isinstance(output, torch.Tensor):
                return output, None
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise ValueError("Empty layer output.")
                return output[0], output
            raise TypeError(f"Unsupported output type {type(output)}")

        def _rewrap(updated, original):
            if original is None:
                return updated
            new_out = list(original)
            new_out[0] = updated
            return type(original)(new_out) if isinstance(original, tuple) else new_out

        if cur_layer == hparams.layer_module_tmp.format(layer):
            layer_output, container = _unwrap(cur_out)

            # Record target_init (first forward only)
            for ri in range(B):
                if target_inits[ri] is None:
                    first_pi = rw_indices_per_req[ri][0]
                    target_inits[ri] = (
                        layer_output[first_pi, lookup_idxs[first_pi]]
                        .detach().clone()
                    )

            # Vectorised delta injection
            if len(lookup_idxs_t) == layer_output.shape[0]:
                layer_output[
                    torch.arange(total_prompts, device=device),
                    lookup_idxs_t,
                ] += deltas[prompt_to_req_t]
            else:
                for pi in range(total_prompts):
                    ri = prompt_to_req[pi]
                    idx = lookup_idxs[pi]
                    layer_output[idx, pi, :] += deltas[ri]

            return _rewrap(layer_output, container)

        return cur_out

    # ---- optimisation loop (hooks registered ONCE) ----------------------
    loss_layer_name = hparams.layer_module_tmp.format(loss_layer)

    with nethook.TraceDict(
        module=model,
        layers=[
            hparams.layer_module_tmp.format(layer),
            loss_layer_name,
        ],
        retain_input=False,
        retain_output=True,
        edit_output=edit_output_fn,
    ) as tr:
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            # ---- single forward pass ------------------------------------
            logits = model(**input_tok, use_cache=False).logits

            # Cache target_init_norms after first forward
            if it == 0:
                target_init_norms = torch.stack(
                    [t.norm() for t in target_inits]
                )

            # ---- loss layer output --------------------------------------
            loss_layer_out = tr[loss_layer_name].output
            if isinstance(loss_layer_out, (list, tuple)):
                output = loss_layer_out[0]
            else:
                output = loss_layer_out
            if output.shape[0] != total_prompts:
                output = output.transpose(0, 1)

            # ---- VECTORISED NLL loss ------------------------------------
            # Extract hidden states only at target-token positions
            # (total_targets,) positions instead of (total_rw × seq_len)
            hidden_at_targets = output[
                rw_global_idxs[tgt_rw_idx], tgt_seq_pos
            ]  # (n_targets, hidden)
            logits_at_targets = (
                ln_f(hidden_at_targets) @ lm_w_dev + lm_b_dev
            )  # (n_targets, vocab)
            log_probs_at_targets = F.log_softmax(
                logits_at_targets, dim=1,
            )  # (n_targets, vocab)
            nll_per_target = -log_probs_at_targets[
                torch.arange(len(tgt_token_ids), device=device),
                tgt_token_ids,
            ].float()  # (n_targets,) — cast to float32 for scatter_add

            # Aggregate: sum per rw-prompt, normalise by target length
            nll_per_rw = torch.zeros(total_rw, device=device)
            nll_per_rw.scatter_add_(0, tgt_rw_idx, nll_per_target)
            nll_per_rw = nll_per_rw / target_lens_per_rw

            # Aggregate: mean per request
            nll_per_req = torch.zeros(B, device=device)
            nll_per_req.scatter_add_(0, rw_to_req, nll_per_rw)
            nll_per_req = nll_per_req / n_rw_per_req

            # ---- VECTORISED KL loss (computed ONCE) ---------------------
            kl_logits = logits[
                kl_global_idxs, kl_lookup_pos
            ]  # (total_kl, vocab)
            kl_log_probs = F.log_softmax(kl_logits, dim=1)

            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

            # Per-KL-prompt divergence (each request has 1 KL prompt)
            kl_per_prompt = F.kl_div(
                kl_distr_init, kl_log_probs,
                log_target=True, reduction="none",
            ).sum(1).float()  # (total_kl,) — cast to float32 for scatter_add

            kl_per_req = torch.zeros(B, device=device)
            kl_per_req.scatter_add_(0, kl_to_req, kl_per_prompt)
            kl_per_req = hparams.kl_factor * kl_per_req

            # ---- VECTORISED weight decay --------------------------------
            delta_norms = torch.norm(deltas, dim=1)
            wd_per_req = hparams.v_weight_decay * (
                delta_norms / target_init_norms ** 2
            )

            # ---- total loss (mask converged if early_stopping) ----------
            per_req_loss = nll_per_req + kl_per_req + wd_per_req
            if early_stopping:
                per_req_loss = torch.where(
                    converged, torch.zeros_like(per_req_loss), per_req_loss,
                )
            total_loss = per_req_loss.sum()

            # Print
            if it % 5 == 0 or it == hparams.v_num_grad_steps - 1:
                n_active = (~converged).sum().item()
                print(
                    f"  [batched compute_z] step {it:3d} | "
                    f"total_loss {total_loss.item():.4f} | "
                    f"active {n_active}/{B} | "
                    f"max_req_loss {per_req_loss[~converged].max().item():.4f}"
                    if n_active > 0 else
                    f"  [batched compute_z] step {it:3d} | all converged"
                )

            # Step callback (for per-step analysis)
            if step_callback is not None:
                step_callback(it, deltas.detach(), target_inits,
                              per_req_loss.detach(), total_loss.item())

            # Early stopping
            if early_stopping:
                newly_converged = per_req_loss < 5e-2
                converged = converged | newly_converged
                if converged.all():
                    print(f"  [batched compute_z] all {B} requests converged at step {it}")
                    break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backward + step
            total_loss.backward()
            opt.step()

            # Vectorised norm clamping
            with torch.no_grad():
                max_norms = hparams.clamp_norm_factor * target_init_norms
                d_norms = torch.norm(deltas, dim=1)
                need_clamp = d_norms > max_norms
                if need_clamp.any():
                    scale = torch.where(
                        need_clamp,
                        max_norms / d_norms,
                        torch.ones_like(d_norms),
                    )
                    deltas.data.mul_(scale.unsqueeze(1))

    # ---- collect results ------------------------------------------------
    results: List[torch.Tensor] = []
    for ri in range(B):
        target = target_inits[ri] + deltas[ri].detach()
        print(
            f"  req {ri}: init_norm {target_inits[ri].norm():.2f} | "
            f"delta_norm {deltas[ri].norm():.2f} | "
            f"target_norm {target.norm():.2f}"
        )
        results.append(target)

    return results
