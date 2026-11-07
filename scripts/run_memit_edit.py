"""
Run MEMIT editing with custom C₀ and evaluate ES/PA/NS via BatchEvaluator.

Usage:
    python scripts/run_memit_edit.py \
        --hparams hparams/MEMIT/olmo2-7b-instruct.yaml \
        --mom2_dataset wiki \
        --n_edits 100 \
        --output results/edit_olmo2i_wiki.json
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

from src.models.memit.memit_hparams import MEMITHyperParams
from src.models.memit.memit_main import apply_memit_to_model, get_cov, COV_CACHE
from src.evaluate import BatchEvaluator


def load_counterfact(n=2000):
    """Load CounterFact samples."""
    with open("data/counterfact.json") as f:
        data = json.load(f)
    return data[:n]


def load_zsre(n=2000, path="data/zsre_mend_eval.json"):
    """Load ZsRE (zsre_mend_eval.json) in the same request format as format_request().

    Field mapping: src -> prompt, pred -> target_new, alt -> ground_truth,
    rephrase -> rephrase_prompt, loc/loc_ans -> locality. ZsRE carries no
    portability probes, so Portability is undefined for this dataset and the
    edit score is a 4-metric harmonic mean.
    """
    with open(path) as f:
        raw = json.load(f)
    raw = raw[:n]

    requests = []
    for r in raw:
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


def format_request(sample):
    """Convert CounterFact sample to MEMIT request + evaluation format."""
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

    # Paraphrase (PA) — use first paraphrase prompt
    if sample.get("paraphrase_prompts"):
        req["rephrase_prompt"] = sample["paraphrase_prompts"][0]

    # Locality (NS) — neighborhood prompts with ground_truth = target_true
    if sample.get("neighborhood_prompts"):
        req["locality"] = {
            "neighborhood": {
                "prompt": sample["neighborhood_prompts"],
                "ground_truth": [target_true] * len(sample["neighborhood_prompts"]),
            }
        }

    return req


def inject_c0(model, hparams, c0_dir):
    """Pre-fill COV_CACHE with precomputed C₀ matrices."""
    c0_dir = Path(c0_dir)
    model_name = model.config._name_or_path.replace("/", "_")

    for layer in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer)
        key = (model_name, layer_name)

        candidates = [
            c0_dir / f"layer_{layer}.npz",
            c0_dir / f"{layer_name.replace('.', '_')}.npz",
        ]
        for chunk_dir in sorted(c0_dir.glob("chunk_*")):
            safe_name = layer_name.replace(".", "_")
            candidates.append(chunk_dir / f"{safe_name}.npz")

        found = None
        for c in candidates:
            if c.exists():
                found = c
                break

        if found is None:
            print(f"WARNING: No C₀ found for {layer_name}, will compute on-the-fly")
            continue

        data = np.load(str(found))
        if "c0" in data:
            c0 = torch.from_numpy(data["c0"].astype(np.float32))
        elif "mom2" in data:
            mom2 = data["mom2"].astype(np.float64)
            count = int(data["count"])
            c0 = torch.from_numpy((mom2 / count).astype(np.float32))
        else:
            print(f"WARNING: Unknown format in {found}")
            continue

        COV_CACHE[key] = c0
        print(f"  Injected C₀ for {layer_name} from {found} (shape={c0.shape})")

    print(f"COV_CACHE: {len(COV_CACHE)} layers loaded")


def aggregate_metrics(eval_results):
    """Compute aggregate metrics from BatchEvaluator results.

    Collects ALL numeric fields from per-request results.
    """
    n = len(eval_results)
    if n == 0:
        return {}

    # Collect all scalar/list metrics across requests
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

    # Compute mean for each metric
    agg = {}
    for key, vals in collectors.items():
        agg[key] = float(np.mean(vals))

    agg["n_samples"] = n
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--n_edits", type=int, default=2000)
    parser.add_argument("--c0_dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_weights", default=None)
    parser.add_argument("--mom2_n_samples", type=int, default=None)
    parser.add_argument("--mom2_dataset", type=str, default=None)
    parser.add_argument("--eval_batch_tokens", type=int, default=4096,
                        help="Token budget per eval batch")
    parser.add_argument("--mom2_update_weight", type=float, default=None,
                        help="Override lambda (mom2_update_weight)")
    parser.add_argument("--dataset", type=str, default="counterfact",
                        choices=["counterfact", "zsre"],
                        help="Edit benchmark. zsre has no portability probes.")
    parser.add_argument("--use_identity_c0", action="store_true",
                        help="Use identity matrix instead of C₀ (RoSE)")
    args = parser.parse_args()

    hparams = MEMITHyperParams.from_hparams(args.hparams)
    hparams.device = 0
    if args.mom2_n_samples is not None:
        hparams.mom2_n_samples = args.mom2_n_samples
    if args.mom2_dataset is not None:
        hparams.mom2_dataset = args.mom2_dataset
    if args.mom2_update_weight is not None:
        hparams.mom2_update_weight = args.mom2_update_weight
    if args.use_identity_c0:
        hparams.use_identity_c0 = True

    c0_source = "identity (RoSE)" if args.use_identity_c0 else (
        args.c0_dir or f"{hparams.mom2_dataset} (on-the-fly, n={hparams.mom2_n_samples})"
    )

    print("=" * 70)
    print(f"Model: {hparams.model_name}")
    print(f"Layers: {hparams.layers}")
    print(f"C₀: {c0_source}")
    print(f"Lambda: {hparams.mom2_update_weight}")
    print(f"Edits: {args.n_edits}")
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

    # Prepare C₀
    if args.use_identity_c0:
        print("\nUsing identity matrix (RoSE) — skipping C₀ computation")
    elif args.c0_dir:
        print(f"\nInjecting C₀ from {args.c0_dir}...")
        inject_c0(model, hparams, args.c0_dir)
    else:
        print("\nPre-computing C₀ covariance...")
        t1 = time.time()
        _ = get_cov(
            model, tok,
            hparams.rewrite_module_tmp.format(hparams.layers[0]),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            hparams=hparams,
        )
        del _
        torch.cuda.empty_cache()
        print(f"C₀ cached for {len(hparams.layers)} layers in {time.time() - t1:.1f}s")

    # Load data
    print(f"\nLoading {args.dataset} ({args.n_edits} samples)...")
    if args.dataset == "zsre":
        requests = load_zsre(args.n_edits)
    else:
        data = load_counterfact(args.n_edits)
        requests = [format_request(s) for s in data]
    print(f"  {len(requests)} edit requests")

    # Apply MEMIT
    print(f"\nApplying MEMIT ({len(requests)} edits)...")
    t0 = time.time()
    model, weights_copy = apply_memit_to_model(
        model, tok, requests, hparams, return_orig_weights=True,
    )
    edit_time = time.time() - t0
    print(f"Editing complete in {edit_time:.1f}s")

    # Save weight deltas
    if args.save_weights:
        print(f"\nSaving edited weights to {args.save_weights}...")
        deltas = {}
        current_sd = model.state_dict()
        for key in weights_copy:
            deltas[key] = current_sd[key].cpu()
        Path(args.save_weights).parent.mkdir(parents=True, exist_ok=True)
        torch.save(deltas, args.save_weights)
        print(f"  Saved {len(deltas)} weight tensors")

    # Evaluate with BatchEvaluator (ES/PA/NS)
    print(f"\nEvaluating edit quality (BatchEvaluator)...")
    t0 = time.time()
    evaluator = BatchEvaluator(
        model, tok, device=hparams.device,
        max_tokens=args.eval_batch_tokens,
    )
    eval_results = evaluator.evaluate(requests)
    eval_time = time.time() - t0

    metrics = aggregate_metrics(eval_results)
    print(f"  Edit Success (ES): {metrics.get('edit_success', 'N/A')}")
    print(f"  Rewrite Acc:       {metrics.get('rewrite_acc', 'N/A')}")
    print(f"  Paraphrase Acc:    {metrics.get('paraphrase_acc', 'N/A')}")
    print(f"  Paraphrase ES:     {metrics.get('paraphrase_es', 'N/A')}")
    print(f"  Locality LogProb:  {metrics.get('locality_log_prob', 'N/A')}")
    print(f"  Eval time: {eval_time:.1f}s")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": hparams.model_name,
        "layers": hparams.layers,
        "c0_source": c0_source,
        "n_edits": len(requests),
        "metrics": metrics,
        "edit_time_seconds": edit_time,
        "eval_time_seconds": eval_time,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
