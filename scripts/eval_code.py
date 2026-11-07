"""
Evaluate edited model on code generation benchmarks (HumanEval, MBPP).

Generates code completions then executes them to compute pass@k.

Usage:
    # HumanEval
    python scripts/eval_code.py \
        --model_name allenai/OLMo-2-1124-7B \
        --benchmark humaneval \
        --output results/humaneval_baseline.json

    # MBPP
    python scripts/eval_code.py \
        --model_name allenai/OLMo-2-1124-7B \
        --benchmark mbpp \
        --output results/mbpp_baseline.json

    # With edited weights
    python scripts/eval_code.py \
        --model_name allenai/OLMo-2-1124-7B \
        --edited_weights results/edited_wikitext_c0.pt \
        --benchmark humaneval \
        --output results/humaneval_wikitext.json
"""

import argparse
import json
import tempfile
import os
import sys
import time
from pathlib import Path

import torch


def load_model(model_name, edited_weights=None, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    if edited_weights:
        print(f"Loading edited weights from {edited_weights}...")
        delta = torch.load(edited_weights, map_location="cpu")
        sd = model.state_dict()
        for k, v in delta.items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd)
        print(f"  Applied {len(delta)} updates")

    model = model.to(device).eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt, max_new_tokens=512,
                        temperature=0.2, top_p=0.95, device="cuda:0"):
    """Generate a single code completion."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output[0][input_ids.shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    return completion


def generate_completions_batched(model, tokenizer, prompts, max_new_tokens=512,
                                 temperature=0.2, top_p=0.95, device="cuda:0",
                                 batch_size=16):
    """Generate completions for multiple prompts in batches."""
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    completions = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            generated = output[input_len:]
            completion = tokenizer.decode(generated, skip_special_tokens=True)
            completions.append(completion)

    tokenizer.padding_side = old_padding_side
    return completions


def stop_at_completion(completion, stop_tokens=None):
    """Truncate completion at stop tokens."""
    if stop_tokens is None:
        stop_tokens = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]
    for stop in stop_tokens:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]
    return completion


def eval_humaneval(model, tokenizer, n_samples=1, device="cuda:0",
                   batch_size=16):
    """Evaluate on HumanEval benchmark."""
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    problems = read_problems()
    print(f"HumanEval: {len(problems)} problems, {n_samples} sample(s) each, batch_size={batch_size}")

    task_ids = list(problems.keys())
    prompts = [problems[tid]["prompt"] for tid in task_ids]

    # Repeat prompts for n_samples
    all_task_ids = [tid for tid in task_ids for _ in range(n_samples)]
    all_prompts = [p for p in prompts for _ in range(n_samples)]

    # Batched generation
    all_completions = generate_completions_batched(
        model, tokenizer, all_prompts, max_new_tokens=512,
        device=device, batch_size=batch_size,
    )

    samples = []
    for task_id, completion in zip(all_task_ids, all_completions):
        completion = stop_at_completion(completion)
        samples.append({"task_id": task_id, "completion": completion})

    print(f"  Generated {len(samples)} completions")

    # Write samples and evaluate
    # One samples file per run (a fixed /tmp path let concurrent runs overwrite each other).
    fd, tmp_path = tempfile.mkstemp(prefix="humaneval_samples_", suffix=".jsonl")
    os.close(fd)
    write_jsonl(tmp_path, samples)
    results = evaluate_functional_correctness(tmp_path, k=[1])

    return {
        "pass@1": results["pass@1"],
        "n_problems": len(problems),
        "n_samples": n_samples,
    }


def eval_mbpp(model, tokenizer, n_samples=1, device="cuda:0"):
    """Evaluate on MBPP benchmark."""
    from datasets import load_dataset

    ds = load_dataset("mbpp", "sanitized", split="test")
    print(f"MBPP (sanitized): {len(ds)} problems, {n_samples} sample(s) each")

    passed = 0
    total = 0

    for i, sample in enumerate(ds):
        prompt = sample["prompt"] + "\n"
        # Add function signature hint if available
        if sample.get("code"):
            # Extract function signature from reference
            lines = sample["code"].strip().split("\n")
            for line in lines:
                if line.strip().startswith("def "):
                    prompt += line + "\n"
                    break

        for _ in range(n_samples):
            completion = generate_completion(
                model, tokenizer, prompt, max_new_tokens=512, device=device,
            )
            completion = stop_at_completion(completion)

            # Test execution
            full_code = prompt + completion
            test_code = "\n".join(sample["test_list"])
            try:
                exec_globals = {}
                exec(full_code, exec_globals)
                exec(test_code, exec_globals)
                passed += 1
            except Exception:
                pass
            total += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} problems, pass@1 so far: {passed/total:.3f}")

    return {
        "pass@1": passed / total if total > 0 else 0,
        "passed": passed,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--edited_weights", type=str, default=None)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["humaneval", "mbpp", "both"])
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem for pass@k")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Edited: {args.edited_weights or 'baseline'}")
    print("=" * 70)

    model, tokenizer = load_model(args.model_name, args.edited_weights, args.device)

    t0 = time.time()
    results = {}

    if args.benchmark in ("humaneval", "both"):
        print("\n--- HumanEval ---")
        results["humaneval"] = eval_humaneval(
            model, tokenizer, n_samples=args.n_samples, device=args.device,
            batch_size=args.batch_size,
        )
        print(f"HumanEval pass@1: {results['humaneval']['pass@1']:.4f}")

    if args.benchmark in ("mbpp", "both"):
        print("\n--- MBPP ---")
        results["mbpp"] = eval_mbpp(
            model, tokenizer, n_samples=args.n_samples, device=args.device,
        )
        print(f"MBPP pass@1: {results['mbpp']['pass@1']:.4f}")

    elapsed = time.time() - t0

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model_name,
        "edited_weights": args.edited_weights,
        "results": results,
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
