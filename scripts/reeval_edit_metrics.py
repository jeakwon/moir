"""
Re-evaluate edit metrics from saved state_dict using BatchEvaluator.
Saves both per-request raw results and aggregated summary.

Usage:
    python scripts/reeval_edit_metrics.py \
        --model_name allenai/OLMo-2-1124-7B-Instruct \
        --edited_weights results/.../state_dict.pt \
        --n_edits 2000 \
        --output results/.../edit_result_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import BatchEvaluator


def load_counterfact(n=2000):
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

    # Portability: generation_prompts (consistency) + attribute_prompts
    portability = {}
    if sample.get("generation_prompts"):
        portability["generation"] = {
            "prompt": sample["generation_prompts"],
            "ground_truth": [target_new] * len(sample["generation_prompts"]),
        }
    if sample.get("attribute_prompts"):
        portability["attribute"] = {
            "prompt": sample["attribute_prompts"],
            "ground_truth": [target_new] * len(sample["attribute_prompts"]),
        }
    if portability:
        req["portability"] = portability

    return req


def load_zsre(n=2000, path="data/zsre_mend_eval.json"):
    """Load ZsRE in the same request format as format_request().

    ZsRE has no portability probes, so Portability is undefined here and the
    edit score is a 4-metric harmonic mean (Efficacy, Generalization,
    Specificity, Fluency).
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
            req["locality"] = {
                "zsre_loc": {
                    "prompt": [r["loc"]],
                    "ground_truth": [r["loc_ans"]],
                }
            }
        requests.append(req)
    return requests


def aggregate_metrics(eval_results):
    n = len(eval_results)
    if n == 0:
        return {}

    collectors = {}
    for r in eval_results:
        for key, val in r.items():
            if key in ("locality", "portability"):
                for sub_key, sub_vals in val.items():
                    full_key = f"{key}.{sub_key}"
                    if isinstance(sub_vals, list) and sub_vals and isinstance(sub_vals[0], (int, float)):
                        collectors.setdefault(full_key, []).extend(sub_vals)
            elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
                collectors.setdefault(key, []).extend(val)

    agg = {}
    for key, vals in collectors.items():
        agg[key] = float(np.mean(vals))

    agg["n_samples"] = n
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--edited_weights", type=str, required=True)
    parser.add_argument("--n_edits", type=int, default=2000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--eval_batch_tokens", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="counterfact",
                        choices=["counterfact", "zsre"],
                        help="Edit benchmark. zsre yields no Portability.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Weights: {args.edited_weights}")
    print(f"N edits: {args.n_edits}")
    print("=" * 70)

    # Load model (baseline first)
    print("Loading model...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load data
    print(f"\nLoading {args.dataset} ({args.n_edits} samples)...")
    if args.dataset == "zsre":
        requests = load_zsre(args.n_edits)
    else:
        data = load_counterfact(args.n_edits)
        requests = [format_request(s) for s in data]
    print(f"  {len(requests)} edit requests")

    evaluator = BatchEvaluator(
        model, tok, device=0,
        max_tokens=args.eval_batch_tokens,
    )

    # 1. Baseline eval (pre-edit) — for NS computation
    print(f"\n>>> Baseline eval (pre-edit)...")
    t0 = time.time()
    baseline_results = evaluator.evaluate(requests)
    print(f"  Baseline eval: {time.time() - t0:.1f}s")

    # 2. Apply edited weights
    print(f"\n>>> Applying edited weights...")
    delta = torch.load(args.edited_weights, map_location="cpu")
    sd = model.state_dict()
    for k, v in delta.items():
        if k in sd:
            sd[k] = v
    model.load_state_dict(sd)
    del delta, sd
    torch.cuda.empty_cache()
    print(f"  Applied {len(torch.load(args.edited_weights, map_location='cpu'))} weight updates")

    # 3. Post-edit eval
    print(f"\n>>> Post-edit eval...")
    t0 = time.time()
    post_results = evaluator.evaluate(requests)
    post_eval_time = time.time() - t0
    print(f"  Post-edit eval: {post_eval_time:.1f}s")

    # 4. Fluency
    print(f"\n>>> Fluency eval...")
    t0 = time.time()
    fluency_scores = evaluator.evaluate_fluency(requests)
    fluency_time = time.time() - t0
    print(f"  Fluency eval: {fluency_time:.1f}s")
    print(f"  Mean fluency (n-gram entropy): {np.mean(fluency_scores):.4f}")

    # 5. Compute NS (Neighborhood Specificity)
    # Compare pre-edit vs post-edit locality argmax tokens
    ns_scores = []
    for base_r, post_r in zip(baseline_results, post_results):
        base_loc = base_r.get("locality", {})
        post_loc = post_r.get("locality", {})
        for key in base_loc:
            if key.endswith("_output"):
                base_tokens = base_loc[key]
                post_tokens = post_loc.get(key, [])
                for bt, pt in zip(base_tokens, post_tokens):
                    if bt and pt:
                        match = sum(a == b for a, b in zip(bt, pt)) / max(len(bt), 1)
                        ns_scores.append(match)
    ns = float(np.mean(ns_scores)) if ns_scores else None

    # Aggregate post-edit metrics
    metrics = aggregate_metrics(post_results)
    metrics["neighborhood_specificity"] = ns
    metrics["fluency"] = float(np.mean(fluency_scores))

    print(f"\n{'='*60}")
    print("Aggregated metrics:")
    for k, v in sorted(metrics.items()):
        if k != "n_samples" and v is not None:
            print(f"  {k}: {v:.4f}")
    print(f"  n_samples: {metrics['n_samples']}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": args.model_name,
        "edited_weights": args.edited_weights,
        "n_edits": len(requests),
        "metrics": metrics,
        "per_request_post": post_results,
        "per_request_baseline": baseline_results,
        "fluency_per_request": fluency_scores,
        "eval_time_seconds": post_eval_time,
        "fluency_time_seconds": fluency_time,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
