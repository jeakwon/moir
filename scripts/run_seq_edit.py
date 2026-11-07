"""Sequential editing: apply edits one-by-one, checkpoint at specified positions.

Saves state_dict + BatchEval metrics at each checkpoint.
HumanEval/lm-eval can be run separately on saved state_dicts.

Usage:
    python scripts/run_seq_edit.py \
        --alg memit \
        --hparams hparams/MEMIT/olmo2-7b-instruct.yaml \
        --n_edits 10000 \
        --mom2_dataset wikitext \
        --checkpoints 10,20,50,100,200,500,1000,2000,5000,10000 \
        --output_dir results/seq_edit/memit_olmo2i_wikitext
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import BatchEvaluator


def load_zsre(n, path="data/zsre_mend_eval.json"):
    """ZsRE in the same request shape as format_request().

    ZsRE has no portability probes, so Portability is undefined and the edit
    score is a four-metric harmonic mean rather than the five used for CounterFact.
    """
    with open(path) as f:
        raw = json.load(f)
    requests = []
    for r in raw[:n]:
        req = {
            "prompt": r["src"],
            "subject": r["subject"],
            "target_new": r["pred"],
            "ground_truth": r["alt"],
        }
        if r.get("rephrase"):
            req["rephrase_prompt"] = r["rephrase"]
        if r.get("loc") and r.get("loc_ans"):
            req["locality"] = {"zsre_loc": {"prompt": [r["loc"]],
                                            "ground_truth": [r["loc_ans"]]}}
        requests.append(req)
    return requests


def load_counterfact(n):
    with open("data/counterfact.json") as f:
        data = json.load(f)
    return data[:n]


def format_request(sample):
    rw = sample["requested_rewrite"]
    prompt = rw["prompt"].format(rw["subject"])
    target_new = rw["target_new"]["str"]
    target_true = rw["target_true"]["str"]
    req = {
        "prompt": prompt,
        "subject": rw["subject"],
        "target_new": target_new,
        "ground_truth": target_true,
    }
    if sample.get("paraphrase_prompts"):
        req["rephrase_prompt"] = sample["paraphrase_prompts"][0]
    if sample.get("neighborhood_prompts"):
        req["locality"] = {
            "neighborhood": {
                "prompt": sample["neighborhood_prompts"],
                "ground_truth": [target_true] * len(sample["neighborhood_prompts"]),
            }
        }
    return req


def aggregate_metrics(eval_results):
    collectors = {}
    for r in eval_results:
        for key, val in r.items():
            if key in ("locality", "portability"):
                for sk, sv in val.items():
                    fk = f"{key}.{sk}"
                    if isinstance(sv, list) and sv and isinstance(sv[0], (int, float)):
                        collectors.setdefault(fk, []).extend(sv)
            elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
                collectors.setdefault(key, []).extend(val)
    return {k: float(np.mean(v)) for k, v in collectors.items()}


def apply_rect_to_state_dict(edited_sd, orig_weights, keep_ratio=0.4):
    """Apply RECT to state_dict (returns new dict, does NOT modify model)."""
    result = {}
    for name, w_orig in orig_weights.items():
        w_edited = edited_sd[name].cpu().float()
        w_orig_cpu = w_orig.cpu().float()
        delta_w = w_edited - w_orig_cpu
        rel_change = torch.abs(delta_w) / (torch.abs(w_orig_cpu) + 1e-10)
        num_keep = max(int(rel_change.numel() * keep_ratio), 1)
        num_zero = rel_change.numel() - num_keep
        threshold = torch.kthvalue(rel_change.flatten(), num_zero).values
        mask = rel_change >= threshold
        delta_rect = delta_w * mask.float()
        result[name] = (w_orig_cpu + delta_rect).to(edited_sd[name].dtype)
    return result


def apply_prune_to_state_dict(edited_sd, orig_weights, alpha=None, rank=100):
    """Apply PRUNE to state_dict (returns new dict, does NOT modify model)."""
    import math
    if alpha is None:
        alpha = math.e
    result = {}
    for name, w_orig in orig_weights.items():
        w_edited = edited_sd[name].cpu().float()
        w_orig_cpu = w_orig.cpu().float()
        delta = w_edited - w_orig_cpu
        U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
        S_orig = torch.linalg.svdvals(w_orig_cpu)
        sigma_max = S_orig[0].item()
        r = min(rank, len(S))
        S_new = S[:r].clone()
        for i in range(r):
            if S_new[i].item() >= sigma_max:
                S_new[i] = math.log(S_new[i].item(), alpha) - math.log(sigma_max, alpha) + sigma_max
        delta_new = U[:, :r] @ torch.diag(S_new) @ Vt[:r, :]
        result[name] = (w_orig_cpu + delta_new).to(edited_sd[name].dtype)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, required=True, choices=["memit", "memit_cached", "alphaedit"])
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--n_edits", type=int, default=10000)
    parser.add_argument("--mom2_dataset", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="counterfact",
                        choices=["counterfact", "zsre"])
    parser.add_argument("--c0_dir", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default="10,20,50,100,200,500,1000,2000,5000,10000")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_identity_c0", action="store_true",
                        help="Use identity matrix instead of C₀ (RoSE)")
    parser.add_argument("--regularizer", type=str, default=None, choices=["rect", "prune"])
    parser.add_argument("--rect_keep_ratio", type=float, default=0.4)
    parser.add_argument("--prune_alpha", type=str, default="e")
    parser.add_argument("--prune_rank", type=int, default=100)
    args = parser.parse_args()

    # Parse prune_alpha
    import math
    prune_alpha = math.e if args.prune_alpha == "e" else float(args.prune_alpha)

    checkpoints = sorted(set(int(x) for x in args.checkpoints.split(",")))
    checkpoints = [c for c in checkpoints if c <= args.n_edits]

    # Load algorithm
    if args.alg in ("memit", "memit_cached"):
        from src.models.memit.memit_hparams import MEMITHyperParams
        from src.models.memit.memit_main import apply_memit_to_model
        hparams = MEMITHyperParams.from_hparams(args.hparams)
        apply_fn = apply_memit_to_model
    else:
        from src.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
        from src.models.alphaedit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cache_c, set_cache_c
        hparams = AlphaEditHyperParams.from_hparams(args.hparams)
        apply_fn = apply_AlphaEdit_to_model

    hparams.device = 0
    if args.mom2_dataset:
        hparams.mom2_dataset = args.mom2_dataset
    if args.use_identity_c0 and args.alg in ("memit", "memit_cached"):
        hparams.use_identity_c0 = True
    if args.c0_dir and args.alg == "alphaedit":
        hparams.c0_dir = args.c0_dir

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Algorithm: {args.alg}")
    print(f"Model: {hparams.model_name}")
    print(f"C₀: {args.mom2_dataset or args.c0_dir or hparams.mom2_dataset}")
    print(f"Sequential edits: {args.n_edits}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Output: {out_dir}")
    print("=" * 70)

    # Load model
    print("Loading model...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(f"cuda:{hparams.device}")
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load data
    if args.dataset == "zsre":
        requests = load_zsre(args.n_edits)
    else:
        data = load_counterfact(args.n_edits)
        requests = [format_request(s) for s in data]
    print(f"  {len(requests)} edit requests from {args.dataset}")
    evaluator = BatchEvaluator(model, tok, device=hparams.device)

    # Resume: check for existing checkpoints
    resume_from = 0
    t_elapsed_prior = 0.0
    existing_ckpts = []
    for c in checkpoints:
        ckpt_dir = out_dir / f"ckpt_{c}"
        sd_path = ckpt_dir / "state_dict.pt"
        er_path = ckpt_dir / "edit_result.json"
        if sd_path.exists() and er_path.exists():
            existing_ckpts.append(c)

    # Also check rolling latest
    latest_meta = out_dir / "latest" / "meta.json"
    if latest_meta.exists():
        lm = json.load(open(latest_meta))
        latest_n = lm["n_edits"]
        if not existing_ckpts or latest_n > max(existing_ckpts):
            existing_ckpts.append(latest_n)

    if existing_ckpts:
        resume_from = max(existing_ckpts)
        # Determine which dir to load from
        ckpt_sd_path = out_dir / f"ckpt_{resume_from}" / "state_dict.pt"
        latest_sd_path = out_dir / "latest" / "state_dict.pt"
        if ckpt_sd_path.exists():
            resume_dir = out_dir / f"ckpt_{resume_from}"
        else:
            resume_dir = out_dir / "latest"
        resume_sd = torch.load(str(resume_dir / "state_dict.pt"), map_location="cpu")
        sd = model.state_dict()
        for k, v in resume_sd.items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd)
        del resume_sd, sd

        # Restore AlphaEdit cache_c
        if args.alg == "alphaedit":
            cache_c_path = out_dir / "latest" / "cache_c.pt"
            if cache_c_path.exists():
                set_cache_c(torch.load(str(cache_c_path)))
                print(f"Restored AlphaEdit cache_c from {cache_c_path}")
            else:
                print("WARNING: No AlphaEdit cache_c found, starting fresh")

        # Load prior elapsed time
        er_path = resume_dir / "edit_result.json"
        meta_path = resume_dir / "meta.json"
        if er_path.exists():
            t_elapsed_prior = json.load(open(er_path))["metrics"].get("t_elapsed", 0.0)
        elif meta_path.exists():
            t_elapsed_prior = json.load(open(meta_path)).get("t_elapsed", 0.0)

        # Load prior checkpoint results
        checkpoint_results_prior = []
        for c in sorted(existing_ckpts):
            er = out_dir / f"ckpt_{c}" / "edit_result.json"
            if er.exists():
                checkpoint_results_prior.append(json.load(open(er))["metrics"])

        print(f"Resumed from ckpt_{resume_from} (elapsed={t_elapsed_prior:.0f}s)")
    else:
        checkpoint_results_prior = []

    # Sequential editing
    original_weights = None
    checkpoint_set = set(checkpoints)
    edit_times = []
    checkpoint_results = checkpoint_results_prior

    # Initialize cache_c for memit_cached
    seq_cache_c = None
    if args.alg == "memit_cached":
        from src.util import nethook
        w = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        key_dim = w.shape[1]
        del w
        seq_cache_c = torch.zeros((len(hparams.layers), key_dim, key_dim))
        # If resuming, cache_c should reflect prior edits — recompute from checkpoint keys
        # For simplicity, start fresh (approximation; exact would need saved cache_c)
        if resume_from > 0:
            cache_c_path = out_dir / "latest" / "cache_c.pt"
            if cache_c_path.exists():
                seq_cache_c = torch.load(str(cache_c_path))
                print(f"Loaded cache_c from {cache_c_path}")
            else:
                print(f"WARNING: No saved cache_c, starting fresh (results may differ from continuous run)")
        print(f"Initialized cache_c: {seq_cache_c.shape}")

    print(f"\nStarting sequential editing ({args.n_edits} edits, resume_from={resume_from})...")
    t_start = time.time()

    for i in range(resume_from, args.n_edits):
        t0 = time.time()
        kwargs = {}
        if seq_cache_c is not None:
            kwargs["cache_c"] = seq_cache_c
        model, wc_i = apply_fn(
            model, tok, [requests[i]], hparams,
            copy=False, return_orig_weights=True, **kwargs,
        )
        edit_times.append(time.time() - t0)

        if original_weights is None:
            original_weights = wc_i
        else:
            del wc_i

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  Edit {i+1}/{args.n_edits} ({edit_times[-1]:.1f}s, total {elapsed:.0f}s)")

        # Rolling checkpoint every 100 edits (overwrite)
        if (i + 1) % 100 == 0:
            latest_dir = out_dir / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            sd = model.state_dict()
            deltas_latest = {key: sd[key].cpu() for key in original_weights}
            torch.save(deltas_latest, latest_dir / "state_dict.pt")
            # Save cache_c (memit_cached or alphaedit)
            if seq_cache_c is not None:
                torch.save(seq_cache_c, latest_dir / "cache_c.pt")
            elif args.alg == "alphaedit":
                ae_cc = get_cache_c()
                if ae_cc is not None:
                    torch.save(ae_cc, latest_dir / "cache_c.pt")
            json.dump({"n_edits": i + 1, "t_elapsed": t_elapsed_prior + time.time() - t_start},
                      open(latest_dir / "meta.json", "w"))
            del deltas_latest

        # Checkpoint
        if (i + 1) in checkpoint_set:
            n_done = i + 1
            ckpt_dir = out_dir / f"ckpt_{n_done}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save state_dict
            deltas = {}
            sd = model.state_dict()
            for key in original_weights:
                deltas[key] = sd[key].cpu()
            torch.save(deltas, ckpt_dir / "state_dict.pt")

            # Apply regularizer (save separately, do NOT modify model)
            if args.regularizer:
                if args.regularizer == "rect":
                    reg_sd = apply_rect_to_state_dict(deltas, original_weights, args.rect_keep_ratio)
                else:
                    reg_sd = apply_prune_to_state_dict(deltas, original_weights, prune_alpha, args.prune_rank)
                torch.save(reg_sd, ckpt_dir / "state_dict_reg.pt")
                print(f"  Saved regularized ({args.regularizer}) state_dict")
                del reg_sd

            # BatchEval (quick)
            t0 = time.time()
            eval_results = evaluator.evaluate(requests[:n_done])
            t_eval = time.time() - t0
            metrics = aggregate_metrics(eval_results)
            metrics["n_edits"] = n_done
            metrics["t_elapsed"] = t_elapsed_prior + time.time() - t_start
            metrics["t_eval"] = t_eval
            metrics["t_per_edit"] = (t_elapsed_prior + time.time() - t_start - t_eval) / n_done

            with open(ckpt_dir / "edit_result.json", "w") as f:
                json.dump({
                    "alg": args.alg,
                    "model": hparams.model_name,
                    "c0": args.mom2_dataset or args.c0_dir or hparams.mom2_dataset,
                    "n_edits": n_done,
                    "mode": "sequential",
                    "metrics": metrics,
                }, f, indent=2)

            checkpoint_results.append(metrics)
            es = metrics.get("rewrite_es", "?")
            print(f"  >>> Checkpoint {n_done}: ES={es:.3f} elapsed={metrics['t_elapsed']:.0f}s")

            del deltas
            gc.collect()
            torch.cuda.empty_cache()

    t_total = t_elapsed_prior + time.time() - t_start

    # Save summary
    summary = {
        "alg": args.alg,
        "model": hparams.model_name,
        "c0": args.mom2_dataset or args.c0_dir or hparams.mom2_dataset,
        "n_edits": args.n_edits,
        "resumed_from": resume_from,
        "mode": "sequential",
        "checkpoints": checkpoint_results,
        "t_total": t_total,
        "t_per_edit": t_total / args.n_edits,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone: {t_total:.0f}s ({t_total/args.n_edits:.1f}s/edit)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
