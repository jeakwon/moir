import os, json, random, argparse, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--ndocs", type=int, default=10000)
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_dir", default=".")
p.add_argument("--chunk_size", type=int, default=500)
p.add_argument("--dtype", default="float16",
               choices=["auto", "float16", "bfloat16", "float32"])
p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
p.add_argument("--max_model_len", type=int, default=None)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--top_k", type=int, default=-1)
p.add_argument("--enforce_eager", action="store_true")
p.add_argument("--use_flashinfer", action="store_true")
p.add_argument("--verbose", action="store_true")
a = p.parse_args()

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if a.use_flashinfer else "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def log(*args):
    if a.verbose:
        print(*args, flush=True)

Path(a.out_dir).mkdir(parents=True, exist_ok=True)
out = str(Path(a.out_dir) / (a.model.replace("/", "__") + ".jsonl"))
max_model_len = a.max_model_len if a.max_model_len is not None else a.max_len + 8

V = AutoTokenizer.from_pretrained(a.model).vocab_size
rng = random.Random(a.seed)
prompts = [{"prompt_token_ids": [rng.randint(0, V - 1)]} for _ in range(a.ndocs)]
chunks = [prompts[i:i + a.chunk_size] for i in range(0, len(prompts), a.chunk_size)]

log(f"model={a.model} ndocs={a.ndocs} max_len={a.max_len} dtype={a.dtype} "
    f"temp={a.temperature} top_p={a.top_p} top_k={a.top_k} "
    f"max_model_len={max_model_len} chunks={len(chunks)}x{a.chunk_size} out={out}")

llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max_model_len,
          gpu_memory_utilization=a.gpu_memory_utilization,
          enforce_eager=a.enforce_eager)
sp = SamplingParams(max_tokens=a.max_len, min_tokens=a.max_len,
                    temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                    seed=a.seed)

bar = None
if a.verbose:
    from tqdm.auto import tqdm
    bar = tqdm(total=a.ndocs, unit="seq")

saved = 0
t0 = time.perf_counter()
with open(out, "w") as f:
    for ci, chunk in enumerate(chunks, 1):
        outs = llm.generate(chunk, sp, use_tqdm=False)
        for o in outs:
            f.write(json.dumps({
                "prompt_token_ids": list(o.prompt_token_ids),
                "token_ids": list(o.outputs[0].token_ids),
            }) + "\n")
        f.flush()
        saved += len(outs)
        if bar is not None:
            bar.update(len(outs))
            bar.set_postfix(chunk=f"{ci}/{len(chunks)}")
if bar is not None:
    bar.close()

dt = time.perf_counter() - t0
toks = saved * a.max_len
print(f"saved {saved} sequences x {a.max_len} tokens -> {out}  "
      f"({dt:.1f}s, {toks / dt:.0f} tok/s)")
