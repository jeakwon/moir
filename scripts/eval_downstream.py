"""
Evaluate edited model on downstream benchmarks using lm-eval-harness.

Measures performance delta (Δ) after MEMIT editing to quantify locality preservation.
Supports: MMLU, GSM8K, HellaSwag, WinoGrande, TriviaQA, NQ, ARC.

Usage:
    # Evaluate pre-edit baseline
    python scripts/eval_downstream.py \
        --model_name allenai/OLMo-2-1124-7B \
        --tasks mmlu,gsm8k,hellaswag \
        --output results/baseline_olmo2.json

    # Evaluate post-edit model (provide edited state_dict)
    python scripts/eval_downstream.py \
        --model_name allenai/OLMo-2-1124-7B \
        --edited_weights results/edited_wikitext_c0.pt \
        --tasks mmlu,gsm8k,hellaswag \
        --output results/post_edit_wikitext.json

    # Quick smoke test (few-shot reduced)
    python scripts/eval_downstream.py \
        --model_name allenai/OLMo-2-1124-7B \
        --tasks mmlu,gsm8k \
        --limit 100 \
        --output results/smoke_test.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--edited_weights", type=str, default=None,
                        help="Path to edited model state_dict (.pt)")
    parser.add_argument("--tasks", type=str, default="mmlu,gsm8k,hellaswag,winogrande,arc_challenge",
                        help="Comma-separated task names")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (for debugging)")
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Override few-shot count (default: task-specific)")
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Wrap prompts in the model's chat template. Needed for "
                             "instruction-following tasks (e.g. ifeval) on instruct models.")
    args = parser.parse_args()

    # Task mapping to lm-eval task names
    TASK_MAP = {
        "mmlu": "mmlu",
        "gsm8k": "gsm8k",
        "hellaswag": "hellaswag",
        "winogrande": "winogrande",
        "triviaqa": "triviaqa",
        "nq": "nq_open",
        "arc_easy": "arc_easy",
        "arc_challenge": "arc_challenge",
    }

    tasks = []
    for t in args.tasks.split(","):
        t = t.strip()
        if t in TASK_MAP:
            tasks.append(TASK_MAP[t])
        else:
            tasks.append(t)  # pass through for custom tasks

    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Tasks: {tasks}")
    print(f"Edited weights: {args.edited_weights or 'None (baseline)'}")
    print(f"Limit: {args.limit or 'full'}")
    print("=" * 70)

    try:
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception as e:
        print(f"ERROR importing lm-eval: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load model
    print("Loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Apply edited weights if provided
    if args.edited_weights:
        print(f"Loading edited weights from {args.edited_weights}...")
        state_dict_delta = torch.load(args.edited_weights, map_location="cpu")
        current_sd = model.state_dict()
        for key, value in state_dict_delta.items():
            if key in current_sd:
                current_sd[key] = value
            else:
                print(f"  WARNING: {key} not in model, skipping")
        model.load_state_dict(current_sd)
        print(f"  Applied {len(state_dict_delta)} weight updates")

    model = model.to(args.device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Wrap for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    # Run evaluation
    print(f"\nRunning evaluation on {len(tasks)} tasks...")
    t0 = time.time()

    eval_kwargs = {
        "model": lm,
        "tasks": tasks,
    }
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    if args.apply_chat_template:
        eval_kwargs["apply_chat_template"] = True

    results = evaluator.simple_evaluate(**eval_kwargs)

    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s")

    # Extract scores
    scores = {}
    for task_name, task_results in results["results"].items():
        scores[task_name] = {}
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                scores[task_name][metric] = value

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    for task_name, task_scores in scores.items():
        main_metric = None
        for key in ["acc,none", "acc_norm,none", "exact_match,none", "em,none"]:
            if key in task_scores:
                main_metric = (key, task_scores[key])
                break
        if main_metric:
            print(f"  {task_name:20s} {main_metric[0]:20s} = {main_metric[1]:.4f}")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": args.model_name,
        "edited_weights": args.edited_weights,
        "tasks": tasks,
        "limit": args.limit,
        "scores": scores,
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
