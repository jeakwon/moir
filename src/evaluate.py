"""BatchEvaluator — batched pre-edit evaluation for EasyEdit requests.

Replicates the logic of EasyEdit's `test_prediction_acc()` (teacher-forcing,
token_em metric) but processes all probes across all requests in large batches
instead of one request at a time.

Metrics computed per probe (single forward pass):
  - token_em:         hard accuracy (argmax == target)
  - target_log_prob:  mean log P(target tokens | prompt)
  - ppl:              perplexity = exp(-target_log_prob)
  - gt_first_log_prob: log P(first token of ground_truth | prompt)
                       → used for ES/EM computation

Usage:
    evaluator = BatchEvaluator(model, tok, device=0, max_tokens=4096)
    results = evaluator.evaluate(requests)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Probe descriptor
# ---------------------------------------------------------------------------

@dataclass
class _Probe:
    """Single evaluation probe — will be batched with others."""
    idx: int                  # original index in flat probe list
    request_idx: int          # which request this belongs to
    kind: str                 # "rewrite" | "rephrase" | "locality" | "portability"
    sub_key: str | None       # e.g. "neighborhood" for locality
    prompt: str
    target: str
    ground_truth: str | None = None   # original answer (for ES/EM)
    locality: bool = False    # if True, return raw token ids instead of acc
    concat_len: int = 0       # token length of "prompt + ' ' + target" (for sorting)
    topk: int = 0             # 0=skip topk, >0=compute topk_acc with this k


# ---------------------------------------------------------------------------
# Per-probe result — all metrics from a single forward pass
# ---------------------------------------------------------------------------

@dataclass
class _ProbeResult:
    """All metrics computed for a single probe."""
    token_em: float = 0.0           # hard accuracy
    target_log_prob: float = 0.0    # mean log P(target tokens)
    ppl: float = 0.0                # exp(-target_log_prob)
    gt_first_log_prob: float | None = None  # log P(first ground_truth token | prompt)
    tgt_first_log_prob: float = 0.0 # log P(first target token | prompt)
    locality_tokens: list[int] = field(default_factory=list)  # raw argmax (locality only)
    # topk fields (UniEdit-style metrics)
    topk_acc: float | None = None           # fraction of target tokens in top-k
    topk_each_in: list[bool] | None = None  # per-token in/out of top-k
    topk_strs: list[list[str]] | None = None  # decoded top-k strings per position


# ---------------------------------------------------------------------------
# N-gram entropy (fluency metric, same formula as EasyEdit)
# ---------------------------------------------------------------------------

def _compute_n_gram_entropy(
    sentence: str,
    ns: list[int] | None = None,
    weights: list[float] | None = None,
) -> float:
    """Weighted n-gram Shannon entropy (EasyEdit's fluency metric).

    Default: weighted mean of 2-gram and 3-gram entropy.
    """
    import nltk

    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]

    tokens = nltk.word_tokenize(sentence)
    if len(tokens) < max(ns):
        return 0.0

    entropy_list = []
    for n in ns:
        ngrams = list(nltk.ngrams(tokens, n))
        if not ngrams:
            entropy_list.append(0.0)
            continue
        fdist = nltk.FreqDist(ngrams)
        freqs = np.array([freq for _, freq in fdist.items()], dtype=np.float64)
        freqs = freqs / freqs.sum()
        entropy_list.append(float(np.sum(-freqs * np.log(freqs) / np.log(2))))

    weighted = np.array(entropy_list) * np.array(weights)
    return float(np.mean(weighted))


# ---------------------------------------------------------------------------
# Batching utilities
# ---------------------------------------------------------------------------

def _make_batches(
    order: list[int],
    concat_lens: list[int],
    batch_size: int,
    max_tokens: int | None,
) -> list[list[int]]:
    """Split sorted indices into batches.

    Uses max_tokens budget when set (adaptive), falls back to fixed batch_size.
    """
    batches: list[list[int]] = []
    if max_tokens is not None and max_tokens > 0:
        cur: list[int] = []
        cur_max_len = 0
        for k, idx in enumerate(order):
            seq_len = concat_lens[k]
            new_max_len = max(cur_max_len, seq_len)
            new_total = (len(cur) + 1) * new_max_len
            if cur and new_total > max_tokens:
                batches.append(cur)
                cur = [idx]
                cur_max_len = seq_len
            else:
                cur.append(idx)
                cur_max_len = new_max_len
        if cur:
            batches.append(cur)
    else:
        for i in range(0, len(order), batch_size):
            batches.append(order[i : i + batch_size])
    return batches


# ---------------------------------------------------------------------------
# Core batched prediction — unified, all metrics
# ---------------------------------------------------------------------------

def _batched_prediction_unified(
    model: Module,
    tok: PreTrainedTokenizerBase,
    probes: list[_Probe],
    device: int | str | torch.device,
    batch_size: int = 64,
    max_tokens: int | None = None,
    max_length: int = 512,
    sort_by_length: bool = True,
) -> list[_ProbeResult]:
    """Run teacher-forcing evaluation on all probes in batches.

    Returns a _ProbeResult per probe with all metrics computed from
    a single forward pass.
    """
    if not probes:
        return []

    device_str = f"cuda:{device}" if isinstance(device, int) else str(device)
    n = len(probes)
    results: list[_ProbeResult] = [None] * n  # type: ignore[assignment]

    # Sort by concat length for tighter packing
    if sort_by_length:
        order = sorted(range(n), key=lambda i: probes[i].concat_len)
    else:
        order = list(range(n))

    concat_lens_sorted = [probes[i].concat_len for i in order]
    batches = _make_batches(order, concat_lens_sorted, batch_size, max_tokens)

    # Pre-compute per-prompt token lengths
    prompt_real_lens_map = {i: len(tok.encode(probes[i].prompt)) for i in order}

    # Pre-tokenize ground_truth first tokens (for ES/EM)
    gt_first_tok_map: dict[int, int | None] = {}
    for i in order:
        gt = probes[i].ground_truth
        if gt is not None:
            gt_ids = tok.encode(" " + gt, add_special_tokens=False)
            gt_first_tok_map[i] = gt_ids[0] if gt_ids else None
        else:
            gt_first_tok_map[i] = None

    for batch_idxs in batches:
        bs = len(batch_idxs)

        batch_prompts = [probes[i].prompt for i in batch_idxs]
        batch_targets = [probes[i].target for i in batch_idxs]
        batch_prompt_lens = [prompt_real_lens_map[i] for i in batch_idxs]

        prompt_target = [p + " " + t for p, t in zip(batch_prompts, batch_targets)]
        max_pt_len = max(len(tok.encode(pt)) for pt in prompt_target) + 1

        # Right-padding: real tokens start at position 0, pads go to the right.
        # For causal LMs, right-side pads never affect real tokens (causal mask),
        # so position embeddings and attention are identical to single-item inference.
        before_padding_side = tok.padding_side
        tok.padding_side = "right"

        prompt_target_tok = tok(
            prompt_target,
            padding=True,
            truncation=True,
            max_length=max(max_length, max_pt_len),
            return_tensors="pt",
        ).to(device_str)

        tok.padding_side = before_padding_side

        # With right-padding, real content starts at position 0.
        # prompt_len[i] = prompt token count, end[i] = total real token count.
        prompt_len = batch_prompt_lens
        content_end = [
            int((ids != tok.pad_token_id).sum())
            for ids in prompt_target_tok["input_ids"].cpu()
        ]

        with torch.no_grad():
            outputs = model(**prompt_target_tok)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            # --- token_em (hard accuracy) ---
            answers = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels = prompt_target_tok["input_ids"].detach().cpu().numpy()

            # --- soft metrics: log probabilities ---
            log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
            input_ids_cpu = prompt_target_tok["input_ids"].detach().cpu()

        if answers.ndim == 1:
            answers = answers[np.newaxis, :]
            labels = labels[np.newaxis, :]
            log_probs = log_probs.unsqueeze(0)
            input_ids_cpu = input_ids_cpu.unsqueeze(0)

        for i in range(bs):
            pl = prompt_len[i]
            end = content_end[i]
            orig_idx = batch_idxs[i]
            pr = _ProbeResult()

            # -- token_em --
            ans_slice = answers[i, pl - 1 : end - 1].tolist()
            lab_slice = labels[i, pl : end].tolist()

            if probes[orig_idx].locality:
                pr.locality_tokens = ans_slice
                pr.token_em = 0.0
            else:
                acc = float(np.mean(np.equal(ans_slice, lab_slice)))
                pr.token_em = 0.0 if np.isnan(acc) else acc

            # -- target_log_prob: mean log P(target_tok | context) --
            # At positions [pl-1 .. end-2], logits predict tokens [pl .. end-1]
            target_positions = list(range(pl, end))
            if target_positions:
                target_ids = input_ids_cpu[i, target_positions]  # actual target tokens
                # Gather log probs at positions [pl-1 .. end-2] for those target tokens
                pred_positions = list(range(pl - 1, end - 1))
                gathered = log_probs[i, pred_positions, :]
                tgt_lps = gathered[
                    torch.arange(len(target_ids)), target_ids
                ].numpy()
                pr.target_log_prob = float(np.mean(tgt_lps))
                pr.ppl = float(np.exp(-pr.target_log_prob))
            else:
                pr.target_log_prob = 0.0
                pr.ppl = 1.0

            # -- first-token log probs (for ES/EM) --
            # At position pl-1, model predicts the first target token
            if pl - 1 < log_probs.shape[1] and pl < end:
                first_logprobs = log_probs[i, pl - 1, :]  # (vocab,)

                # P(first target token | prompt)
                first_tgt_id = input_ids_cpu[i, pl].item()
                pr.tgt_first_log_prob = float(first_logprobs[first_tgt_id])

                # P(first ground_truth token | prompt) — for ES/EM
                gt_tok = gt_first_tok_map[orig_idx]
                if gt_tok is not None:
                    pr.gt_first_log_prob = float(first_logprobs[gt_tok])

            # -- topk metrics (UniEdit-style) --
            probe_topk = probes[orig_idx].topk
            if probe_topk > 0 and target_positions:
                pred_positions = list(range(pl - 1, end - 1))
                pred_lp = log_probs[i, pred_positions, :]   # (n_target, vocab)
                topk_ids = torch.topk(pred_lp, probe_topk, dim=-1).indices  # (n_target, k)
                target_ids = input_ids_cpu[i, target_positions]              # (n_target,)
                each_in = [
                    bool(target_ids[j].item() in topk_ids[j])
                    for j in range(len(target_ids))
                ]
                pr.topk_acc = sum(each_in) / len(each_in)
                pr.topk_each_in = each_in
                # Decode topk strings (locality needs for pre/post comparison,
                # generality needs for post_edit_predict_tokens)
                pr.topk_strs = [
                    [tok.decode([t]) for t in row]
                    for row in topk_ids.tolist()
                ]

            results[orig_idx] = pr

    return results


# ---------------------------------------------------------------------------
# BatchEvaluator
# ---------------------------------------------------------------------------

class BatchEvaluator:
    """Evaluate many EasyEdit requests in batched forward passes.

    Parameters
    ----------
    model : torch.nn.Module
        Language model (e.g. GPT-J, LLaMA).
    tok : PreTrainedTokenizerBase
        Matching tokenizer. Must have pad_token set.
    device : int | str
        CUDA device.
    eval_batch_size : int
        Mini-batch size for forward passes (default 64).
    max_tokens : int | None
        Token budget per batch for adaptive batching (default None = fixed).
    max_length : int
        Max token length for tokenization (default 512).
    sort_by_length : bool
        Sort probes by sequence length before batching (default True).
    unified : bool
        Process all probe types in a single pass (default True).
    """

    def __init__(
        self,
        model: Module,
        tok: PreTrainedTokenizerBase,
        device: int | str = 0,
        eval_batch_size: int = 64,
        max_tokens: int | None = None,
        max_length: int = 512,
        sort_by_length: bool = True,
        unified: bool = True,
    ) -> None:
        self.model = model
        self.tok = tok
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.sort_by_length = sort_by_length
        self.unified = unified

        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Evaluate all requests."""
        probes = self._collect_probes(requests)
        for p in probes:
            p.concat_len = len(self.tok.encode(p.prompt + " " + p.target))

        if self.unified:
            return self._evaluate_unified(requests, probes)
        else:
            return self._evaluate_separate(requests, probes)

    def evaluate_fluency(
        self,
        requests: list[dict[str, Any]],
        generation_prompts: list[list[str]] | None = None,
        max_new_tokens: int = 100,
        batch_size: int = 16,
    ) -> list[float]:
        """Compute fluency via batched generation + n-gram entropy.

        Follows MEMIT/ROME convention: for each request, generate text from
        all generation_prompts, concatenate, then compute weighted 2/3-gram
        entropy on the concatenated text.

        Parameters
        ----------
        requests : list[dict]
            Each must have a "prompt" key.
        generation_prompts : list[list[str]] | None
            Per-request list of generation prompts (e.g. CounterFact's
            generation_prompts). If None, falls back to request["prompt"].
        max_new_tokens : int
            Max tokens to generate per prompt.
        batch_size : int
            Generation batch size.

        Returns
        -------
        list[float]
            Per-request n-gram entropy values.
        """

        device_str = (
            f"cuda:{self.device}" if isinstance(self.device, int)
            else str(self.device)
        )

        # Build flat prompt list with mapping back to request index
        flat_prompts: list[str] = []
        prompt_to_req: list[int] = []
        for i, req in enumerate(requests):
            if generation_prompts and generation_prompts[i]:
                for gp in generation_prompts[i]:
                    flat_prompts.append(gp)
                    prompt_to_req.append(i)
            else:
                flat_prompts.append(req["prompt"])
                prompt_to_req.append(i)

        # Batched generation
        all_texts: list[str] = []
        before_padding_side = self.tok.padding_side
        self.tok.padding_side = "left"

        for i in range(0, len(flat_prompts), batch_size):
            batch = flat_prompts[i : i + batch_size]
            inputs = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(device_str)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )

            for j, out in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].ne(
                    self.tok.pad_token_id
                ).sum()
                gen_text = self.tok.decode(
                    out[prompt_len:], skip_special_tokens=True
                )
                all_texts.append(gen_text)

        self.tok.padding_side = before_padding_side

        # Group texts by request, concatenate, compute entropy
        from collections import defaultdict
        req_texts: dict[int, list[str]] = defaultdict(list)
        for text, req_idx in zip(all_texts, prompt_to_req):
            req_texts[req_idx].append(text)

        entropies: list[float] = []
        for i in range(len(requests)):
            combined = " ".join(req_texts.get(i, []))
            entropies.append(_compute_n_gram_entropy(combined))

        return entropies

    def evaluate_uniedit(
        self,
        test_cases: list[dict[str, Any]],
        default_topk: int = 5,
    ) -> list[dict[str, Any]]:
        """Evaluate UniEdit test cases in batched forward passes.

        Converts UniEdit's hierarchical probes (reliability/generality/locality
        with related_hops) into a flat probe list, runs them through the batched
        prediction engine, and reconstructs the structured results.

        Call this twice (pre-edit and post-edit) and then use
        ``compute_locality_acc(pre_results, post_results)`` to compute the
        locality preservation metric.

        Parameters
        ----------
        test_cases : list[dict]
            UniEdit format: each has 'requests', 'generality', 'locality'.
        default_topk : int
            Default k for top-k checks (overridden per-probe by target_check_topk).

        Returns
        -------
        list[dict]
            Structured results with reliability acc, generality topk_acc,
            locality topk_strs/argmax_tokens.
        """
        from src.data_uniedit import (
            flatten_uniedit_probes,
            unflatten_uniedit_results,
        )

        flat_probes, origins = flatten_uniedit_probes(test_cases, default_topk)

        # Convert to _Probe objects
        probes: list[_Probe] = []
        for i, fp in enumerate(flat_probes):
            probes.append(_Probe(
                idx=i,
                request_idx=0,  # not used for UniEdit unflatten
                kind="uniedit",
                sub_key=None,
                prompt=fp["prompt"],
                target=fp["target"],
                locality=fp["is_locality"],
                topk=fp["topk"],
            ))

        # Pre-compute concat lengths
        for p in probes:
            p.concat_len = len(self.tok.encode(p.prompt + " " + p.target))

        # Run batched prediction
        results = _batched_prediction_unified(
            self.model, self.tok, probes, self.device,
            batch_size=self.eval_batch_size,
            max_tokens=self.max_tokens,
            max_length=self.max_length,
            sort_by_length=self.sort_by_length,
        )

        return unflatten_uniedit_results(test_cases, origins, results, self.tok)

    def _evaluate_unified(
        self, requests: list[dict[str, Any]], probes: list[_Probe],
    ) -> list[dict[str, Any]]:
        """Single pass for all probe types."""
        results = _batched_prediction_unified(
            self.model, self.tok, probes, self.device,
            batch_size=self.eval_batch_size,
            max_tokens=self.max_tokens,
            max_length=self.max_length,
            sort_by_length=self.sort_by_length,
        )
        return self._unflatten(requests, probes, results)

    def _evaluate_separate(
        self, requests: list[dict[str, Any]], probes: list[_Probe],
    ) -> list[dict[str, Any]]:
        """Separate passes for acc probes and locality probes."""
        acc_probes = [p for p in probes if not p.locality]
        loc_probes = [p for p in probes if p.locality]

        acc_results = _batched_prediction_unified(
            self.model, self.tok, acc_probes, self.device,
            batch_size=self.eval_batch_size,
            max_tokens=self.max_tokens,
            max_length=self.max_length,
            sort_by_length=self.sort_by_length,
        ) if acc_probes else []

        loc_results = _batched_prediction_unified(
            self.model, self.tok, loc_probes, self.device,
            batch_size=self.eval_batch_size,
            max_tokens=self.max_tokens,
            max_length=self.max_length,
            sort_by_length=self.sort_by_length,
        ) if loc_probes else []

        all_probes = acc_probes + loc_probes
        all_results = list(acc_results) + list(loc_results)
        return self._unflatten(requests, all_probes, all_results)

    # ------------------------------------------------------------------
    # Internal: collect probes
    # ------------------------------------------------------------------

    def _collect_probes(
        self, requests: list[dict[str, Any]]
    ) -> list[_Probe]:
        """Gather all (prompt, target) pairs from all requests."""
        probes: list[_Probe] = []
        idx = 0

        for i, req in enumerate(requests):
            prompt = req["prompt"]
            target_new = req["target_new"]
            ground_truth = req.get("ground_truth")

            # Rewrite
            probes.append(_Probe(
                idx=idx, request_idx=i, kind="rewrite", sub_key=None,
                prompt=prompt, target=target_new, ground_truth=ground_truth,
            ))
            idx += 1

            # Rephrase
            rephrase = req.get("rephrase_prompt")
            if rephrase is not None:
                probes.append(_Probe(
                    idx=idx, request_idx=i, kind="rephrase", sub_key=None,
                    prompt=rephrase, target=target_new, ground_truth=ground_truth,
                ))
                idx += 1

            # Locality (raw tokens)
            locality = req.get("locality", {})
            if locality:
                for loc_key, loc_data in locality.items():
                    loc_prompts = loc_data["prompt"]
                    loc_gt = loc_data["ground_truth"]
                    if isinstance(loc_prompts, str):
                        loc_prompts, loc_gt = [loc_prompts], [loc_gt]
                    for lp, lg in zip(loc_prompts, loc_gt):
                        probes.append(_Probe(
                            idx=idx, request_idx=i, kind="locality",
                            sub_key=loc_key, prompt=lp, target=lg,
                            locality=True,
                        ))
                        idx += 1

            # Portability (accuracy)
            portability = req.get("portability", {})
            if portability:
                for port_key, port_data in portability.items():
                    port_prompts = port_data["prompt"]
                    port_gt = port_data["ground_truth"]
                    if isinstance(port_prompts, str):
                        port_prompts, port_gt = [port_prompts], [port_gt]
                    for pp, pg in zip(port_prompts, port_gt):
                        probes.append(_Probe(
                            idx=idx, request_idx=i, kind="portability",
                            sub_key=port_key, prompt=pp, target=pg,
                        ))
                        idx += 1

        return probes

    # ------------------------------------------------------------------
    # Internal: unflatten results
    # ------------------------------------------------------------------

    def _unflatten(
        self,
        requests: list[dict[str, Any]],
        probes: list[_Probe],
        results: list[_ProbeResult],
    ) -> list[dict[str, Any]]:
        """Reassemble flat results into per-request dicts."""
        out: list[dict[str, Any]] = [
            {"locality": {}, "portability": {}}
            for _ in requests
        ]

        for probe, pr in zip(probes, results):
            r = out[probe.request_idx]

            if probe.kind == "rewrite":
                r["rewrite_acc"] = [pr.token_em]
                r["rewrite_log_prob"] = [pr.target_log_prob]
                r["rewrite_ppl"] = [pr.ppl]
                r["rewrite_tgt_first_lp"] = [pr.tgt_first_log_prob]
                if pr.gt_first_log_prob is not None:
                    r["rewrite_gt_first_lp"] = [pr.gt_first_log_prob]
                    # ES: P(new) > P(old) at first token
                    r["rewrite_es"] = [1.0 if pr.tgt_first_log_prob > pr.gt_first_log_prob else 0.0]
                    # EM: P(new) - P(old) in probability space
                    r["rewrite_em"] = [
                        float(np.exp(pr.tgt_first_log_prob) - np.exp(pr.gt_first_log_prob))
                    ]

            elif probe.kind == "rephrase":
                r["rephrase_acc"] = [pr.token_em]
                r["rephrase_log_prob"] = [pr.target_log_prob]
                r["rephrase_ppl"] = [pr.ppl]
                if pr.gt_first_log_prob is not None:
                    r["rephrase_es"] = [1.0 if pr.tgt_first_log_prob > pr.gt_first_log_prob else 0.0]

            elif probe.kind == "locality":
                key = f"{probe.sub_key}_output"
                r["locality"].setdefault(key, []).append(pr.locality_tokens)
                # Soft locality: log prob of ground truth (should stay high after edit)
                lp_key = f"{probe.sub_key}_log_prob"
                r["locality"].setdefault(lp_key, []).append(pr.target_log_prob)

            elif probe.kind == "portability":
                key = f"{probe.sub_key}_acc"
                r["portability"].setdefault(key, []).append(pr.token_em)
                lp_key = f"{probe.sub_key}_log_prob"
                r["portability"].setdefault(lp_key, []).append(pr.target_log_prob)

        return out
