"""
Run AlphaEdit batch editing (all requests in one call) with a chosen C0 and evaluate the
edits with BatchEvaluator.

This is the driver of the paper's AlphaEdit batch runs, which was inline Python in the
experiment launchers: same request construction, fp16 model, single
apply_AlphaEdit_to_model call, saved weights and edit_result.json keys.

Usage (from the repository root):
    python scripts/run_alphaedit_edit.py \
        --hparams hparams/AlphaEdit/qwen3-8b-t005.yaml \
        --mom2_dataset memoir-rand1 \
        --n_edits 20000 \
        --output_dir results/batch_edit/alphaedit_t005_20000/Qwen3-8B/memoir-rand1
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
from src.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
from src.models.alphaedit.AlphaEdit_main import apply_AlphaEdit_to_model


def build_requests(records):
    requests = []
    for s in records:
        rw = s["requested_rewrite"]
        req = {
            "prompt": rw["prompt"].format(rw["subject"]),
            "subject": rw["subject"],
            "target_new": rw["target_new"]["str"],
            "ground_truth": rw["target_true"]["str"],
        }
        if s.get("paraphrase_prompts"):
            req["rephrase_prompt"] = s["paraphrase_prompts"][0]
        if s.get("neighborhood_prompts"):
            req["locality"] = {
                "neighborhood": {
                    "prompt": s["neighborhood_prompts"],
                    "ground_truth": [rw["target_true"]["str"]] * len(s["neighborhood_prompts"]),
                }
            }
        requests.append(req)
    return requests


def aggregate(eval_results):
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
    metrics = {k: float(np.mean(v)) for k, v in collectors.items()}
    metrics["n_samples"] = len(eval_results)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AlphaEdit batch editing with a chosen C0")
    parser.add_argument("--hparams", type=str, required=True,
                        help="hparams/AlphaEdit/<model>[-t<tau>].yaml (sets the model, layers and threshold)")
    parser.add_argument("--mom2_dataset", type=str, required=True,
                        help="C0 source key in configs/datasets.py: memoir-rand1 (MOIR) or wikipedia (C_Wiki)")
    parser.add_argument("--n_edits", type=int, required=True, help="first N CounterFact records, in file order")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--counterfact", type=str, default="data/counterfact.json")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    hparams = AlphaEditHyperParams.from_hparams(args.hparams)
    hparams.device = args.device
    hparams.mom2_dataset = args.mom2_dataset

    print(f"Model: {hparams.model_name}")
    print(f"C0: {args.mom2_dataset}")
    print(f"Threshold: {hparams.nullspace_threshold}")

    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(f"cuda:{hparams.device}")
    model.eval()

    records = json.load(open(args.counterfact))[: args.n_edits]
    requests = build_requests(records)

    print(f"Editing {len(requests)} facts...")
    t0 = time.time()
    model, weights_copy = apply_AlphaEdit_to_model(model, tok, requests, hparams, return_orig_weights=True)
    edit_time = time.time() - t0
    print(f"Edit done in {edit_time:.1f}s")

    # The edited down_proj weights (full tensors, not differences), as in the paper runs.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    torch.save({key: sd[key].cpu() for key in weights_copy}, out_dir / "state_dict.pt")

    evaluator = BatchEvaluator(model, tok, device=hparams.device)
    metrics = aggregate(evaluator.evaluate(requests))

    output = {
        "model": hparams.model_name,
        "alg": "AlphaEdit",
        "c0": args.mom2_dataset,
        "n_edits": args.n_edits,
        "nullspace_threshold": hparams.nullspace_threshold,
        "metrics": metrics,
        "edit_time_seconds": edit_time,
    }
    json.dump(output, open(out_dir / "edit_result.json", "w"), indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
