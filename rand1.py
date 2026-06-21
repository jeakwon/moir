import os, json, random, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--ndocs", type=int, default=10000)
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_dir", default=".")
p.add_argument("--use_flashinfer", action="store_true")
a = p.parse_args()

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if a.use_flashinfer else "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

Path(a.out_dir).mkdir(parents=True, exist_ok=True)
out = str(Path(a.out_dir) / (a.model.replace("/", "__") + ".jsonl"))
V = AutoTokenizer.from_pretrained(a.model).vocab_size
rng = random.Random(a.seed)
prompts = [{"prompt_token_ids": [rng.randint(0, V - 1)]} for _ in range(a.ndocs)]

llm = LLM(model=a.model, dtype="float16", max_model_len=a.max_len + 8)
sp = SamplingParams(max_tokens=a.max_len, min_tokens=a.max_len,
                    temperature=1.0, top_p=0.95, seed=a.seed)
outs = llm.generate(prompts, sp)

with open(out, "w") as f:
    for o in outs:
        f.write(json.dumps({
            "prompt_token_ids": list(o.prompt_token_ids),
            "token_ids": list(o.outputs[0].token_ids),
        }) + "\n")

print(f"saved {len(outs)} sequences x {a.max_len} tokens -> {out}")
