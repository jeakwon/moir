"""
Layer statistics computation for MEMIT.

Computes E[hh^T] (second moment of hidden states) at specified layers.

Pipeline:
    1. _ensure_jsonl_cache()  — any source → JSONL cache (one-time)
    2. _jsonl_pretokenize()   — JSONL → batch tokenize → sorted tensor list
    3. _gpu_batch_iter()      — tensors → padded GPU batches
    4. forward loop           — TraceDict(stop=True) → SecondMoment accumulation
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.util.nethook import Trace, TraceDict, set_requires_grad
from src.util.runningstats import (
    CombinedStat, Mean, NormMean, SecondMoment,
    load_cached_state, save_cached_state,
)

from .tok_dataset import dict_to_, flatten_masked_batch


STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

from configs.datasets import DATASET_REGISTRY, LEGACY_JSONL_DIR


# ---------------------------------------------------------------------------
# Step 1: Ensure JSONL cache
# ---------------------------------------------------------------------------

def _ensure_jsonl_cache(ds_name, sample_size,
                        model=None, tokenizer=None, device="cuda:0",
                        model_name=None, stats_dir=None):
    """Convert any DATASET_REGISTRY source to a cached JSONL file.

    Selfgen → model-specific: <stats_dir>/<model>/jsonl/<ds>_<size>.jsonl
    Non-selfgen → shared:     <stats_dir>/shared/jsonl/<ds>_<size>.jsonl

    Supports resume: if a partial cache exists, appends only remaining docs.
    Returns path to the JSONL file.
    """
    cfg = DATASET_REGISTRY.get(ds_name)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(DATASET_REGISTRY)}")

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", ds_name)
    base = Path(stats_dir) if stats_dir else Path(LEGACY_JSONL_DIR).parent / "stats"

    if cfg["type"] == "selfgen":
        # Selfgen: model-specific (text differs per model)
        jsonl_dir = base / model_name / "jsonl"
    else:
        # Non-selfgen: shared (text is model-agnostic)
        jsonl_dir = base / "shared" / "jsonl"

    jsonl_dir.mkdir(parents=True, exist_ok=True)
    cache_path = jsonl_dir / f"{safe_name}_{sample_size}.jsonl"

    # Legacy fallback for non-selfgen
    if not cache_path.exists() and cfg["type"] != "selfgen":
        legacy_path = Path(LEGACY_JSONL_DIR) / f"{safe_name}_{sample_size}.jsonl"
        if legacy_path.exists():
            cache_path = legacy_path

    # Quick check before lock — full cache hit
    existing_lines = 0
    if cache_path.exists():
        existing_lines = sum(1 for _ in open(cache_path))
        if existing_lines >= sample_size:
            print(f"JSONL cache hit: {cache_path} ({existing_lines} docs)")
            return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # File lock to prevent concurrent writes from multiple SLURM tasks
    lock_path = cache_path.with_suffix(".lock")
    import fcntl
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Waiting for another task to finish writing {cache_path}...")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # blocking wait

    # Re-check after acquiring lock (another process may have completed/resumed)
    if cache_path.exists():
        existing_lines = sum(1 for _ in open(cache_path))
        if existing_lines >= sample_size:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            lock_path.unlink(missing_ok=True)
            print(f"JSONL cache hit: {cache_path} ({existing_lines} docs)")
            return str(cache_path)

    remaining = sample_size - existing_lines
    if existing_lines > 0:
        print(f"Resuming JSONL cache: {cache_path} "
              f"({existing_lines}/{sample_size}, {remaining} remaining)")

    if cfg["type"] == "hf":
        from datasets import load_dataset
        print(f"{'Resuming' if existing_lines else 'Downloading'} "
              f"{ds_name} → {cache_path} ({remaining} docs)...")
        ds = load_dataset(cfg["path"], cfg.get("name"), split="train", streaming=True)
        field = cfg["field"]
        n = 0
        skipped = 0
        with open(cache_path, "a") as f:
            for item in ds:
                text = item[field]
                if not (isinstance(text, str) and len(text.strip()) > 50):
                    continue
                if skipped < existing_lines:
                    skipped += 1
                    continue
                f.write(json.dumps({"text": text}) + "\n")
                n += 1
                if n % 10000 == 0:
                    print(f"  {existing_lines + n}/{sample_size}")
                if n >= remaining:
                    break
        print(f"Wrote {n} docs to {cache_path} (total: {existing_lines + n})")

    elif cfg["type"] == "selfgen":
        init_mode = cfg["init"]
        gen_length = cfg.get("gen_length", 256)
        num_prefix = cfg.get("num_prefix_tokens", 0)
        gen_bs = 64
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Frequency-weighted seeding needs a unigram distribution over the vocab.
        freq_probs = None
        if init_mode == "freq":
            freq_file = cfg.get("freq_file")
            if freq_file:
                fd = np.load(str(freq_file).format(model=model_name))
                counts = fd["counts"].astype(np.float64)[: tokenizer.vocab_size]
                freq_probs = torch.from_numpy(counts / counts.sum()).to(
                    device=device, dtype=torch.float32
                )
                print(f"  freq seed distribution from {freq_file} "
                      f"({int((counts > 0).sum())}/{tokenizer.vocab_size} tokens observed)")

        print(f"{'Resuming' if existing_lines else 'Generating'} "
              f"selfgen text: {init_mode}, {remaining} remaining")
        n = 0
        with open(cache_path, "a") as f:
            while n < remaining:
                cur_bs = min(gen_bs, remaining - n)
                if init_mode == "bos":
                    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
                    prefix = torch.full((cur_bs, 1), bos, dtype=torch.long, device=device)
                elif init_mode == "random":
                    prefix = torch.randint(0, tokenizer.vocab_size, (cur_bs, num_prefix), device=device)
                elif init_mode == "freq":
                    # Seed drawn from the empirical unigram distribution rather than
                    # uniformly over vocabulary IDs. Ablation for the seeding question:
                    # uniform sampling puts most of its mass on tokens that natural
                    # text almost never uses.
                    if freq_probs is None:
                        raise ValueError(
                            f"selfgen init 'freq' needs freq_file; registry entry for "
                            f"{ds_name} must set 'freq_file' (see scripts/compute_token_freq.py)"
                        )
                    prefix = torch.multinomial(
                        freq_probs, cur_bs * num_prefix, replacement=True
                    ).view(cur_bs, num_prefix).to(device=device, dtype=torch.long)
                else:
                    raise ValueError(f"Unknown selfgen init: {init_mode}")

                with torch.inference_mode():
                    output = model.generate(
                        prefix, max_new_tokens=gen_length,
                        do_sample=True, temperature=1.0, top_p=1.0,
                        pad_token_id=pad_id,
                    )
                for i in range(cur_bs):
                    text = tokenizer.decode(output[i].tolist(), skip_special_tokens=True)
                    f.write(json.dumps({"text": text}) + "\n")
                n += cur_bs
                if n <= gen_bs or n % 1024 == 0:
                    print(f"  Generated {existing_lines + n}/{sample_size}")
        print(f"Selfgen saved: {cache_path} (total: {existing_lines + n})")

    else:
        raise ValueError(f"Cannot cache dataset type: {cfg['type']}")

    # Release lock
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()
    lock_path.unlink(missing_ok=True)

    return str(cache_path)


# ---------------------------------------------------------------------------
# Step 2: Batch tokenize from JSONL
# ---------------------------------------------------------------------------

def _jsonl_pretokenize(jsonl_path, tokenizer, maxlen, sample_size):
    """Read JSONL, batch-tokenize, return sorted list of token tensors."""
    t0 = time.time()
    print(f"Pre-tokenizing {sample_size} samples from {jsonl_path} (maxlen={maxlen})...")

    all_chunks = []
    tok_bs = 1000
    buffer = []

    with open(jsonl_path) as f:
        for line in f:
            if len(all_chunks) >= sample_size:
                break
            text = json.loads(line).get("text", "")
            if len(text.strip()) > 50:
                buffer.append(text)

            if len(buffer) >= tok_bs:
                encs = tokenizer(buffer, truncation=True, max_length=maxlen,
                                 padding=False, return_tensors=None)
                for ids in encs["input_ids"]:
                    if len(ids) >= 32:
                        all_chunks.append(torch.tensor(ids, dtype=torch.long))
                        if len(all_chunks) >= sample_size:
                            break
                buffer = []

    # Flush remaining
    if buffer and len(all_chunks) < sample_size:
        encs = tokenizer(buffer, truncation=True, max_length=maxlen,
                         padding=False, return_tensors=None)
        for ids in encs["input_ids"]:
            if len(ids) >= 32:
                all_chunks.append(torch.tensor(ids, dtype=torch.long))
                if len(all_chunks) >= sample_size:
                    break

    # Sort by length descending for efficient batching
    all_chunks.sort(key=lambda x: -len(x))
    print(f"Tokenized {len(all_chunks)} samples in {time.time()-t0:.1f}s")
    return all_chunks


# ---------------------------------------------------------------------------
# Step 3: GPU batch iterator
# ---------------------------------------------------------------------------

def _gpu_batch_iter(all_chunks, batch_tokens, device):
    """Yield padded batches built directly on GPU from pre-tokenized chunks."""
    i = 0
    while i < len(all_chunks):
        batch_seqs = []
        cur_max_len = len(all_chunks[i])

        while i < len(all_chunks):
            seq_len = len(all_chunks[i])
            new_max = max(cur_max_len, seq_len)
            if batch_seqs and new_max * (len(batch_seqs) + 1) > batch_tokens:
                break
            batch_seqs.append(all_chunks[i])
            cur_max_len = new_max
            i += 1

        bsz = len(batch_seqs)
        padded_ids = torch.zeros(bsz, cur_max_len, dtype=torch.long, device=device)
        attn_mask = torch.zeros(bsz, cur_max_len, dtype=torch.long, device=device)
        for j, seq in enumerate(batch_seqs):
            padded_ids[j, :len(seq)] = seq
            attn_mask[j, :len(seq)] = 1

        yield [{"input_ids": padded_ids, "attention_mask": attn_mask}]


def _random_iter(vocab_size, maxlen, batch_tokens, sample_size, seed=42):
    """Generate batches of random token IDs (no JSONL needed)."""
    rng = np.random.RandomState(seed)
    bs = max(1, batch_tokens // maxlen)
    n = 0
    while n < sample_size:
        cur_bs = min(bs, sample_size - n)
        ids = torch.from_numpy(rng.randint(0, vocab_size, size=(cur_bs, maxlen))).long()
        mask = torch.ones_like(ids)
        yield [{"input_ids": ids, "attention_mask": mask}]
        n += cur_bs


# ---------------------------------------------------------------------------
# GPU memory optimization
# ---------------------------------------------------------------------------

def _offload_unused_layers(model, layer_names):
    """Move layers beyond max target to CPU to free GPU memory.

    Returns a restore callable that moves them back.
    """
    max_idx = -1
    container_path = None
    for name in layer_names:
        m = re.match(r'^(.*?)\.(\d+)\.', name)
        if m:
            container_path = m.group(1)
            max_idx = max(max_idx, int(m.group(2)))

    if max_idx < 0 or container_path is None:
        return lambda: None

    obj = model
    for part in container_path.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return lambda: None
    layer_container = obj

    n_total = len(layer_container)
    if max_idx + 1 >= n_total:
        return lambda: None

    offloaded = []
    freed_bytes = 0
    for i in range(max_idx + 1, n_total):
        orig_device = next(layer_container[i].parameters()).device
        for p in layer_container[i].parameters():
            freed_bytes += p.numel() * p.element_size()
        layer_container[i].to("cpu")
        offloaded.append((i, orig_device))

    torch.cuda.empty_cache()
    print(f"Offloaded layers {max_idx+1}-{n_total-1} to CPU "
          f"({freed_bytes / 1e9:.1f}GB freed)")

    def restore():
        for i, dev in offloaded:
            layer_container[i].to(dev)

    return restore


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def _get_npos(model):
    """Get max sequence length from model config."""
    for attr in ("n_positions", "max_sequence_length",
                 "max_position_embeddings", "seq_length"):
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise NotImplementedError("Cannot determine max sequence length")


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
    streaming=False,
):
    """
    Compute cached stats for one or more layers.

    Pipeline: JSONL cache → batch tokenize → GPU batches → forward → save.
    """
    # --- normalise inputs ---
    single = isinstance(layer_name, str)
    layer_names = [layer_name] if single else list(layer_name)

    device = f"cuda:{hparams.device}" if hparams else "cuda:0"
    precision = precision or "float32"
    dtype = getattr(torch, precision)

    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    maxlen = min(_get_npos(model), 2048)
    if batch_tokens is None:
        batch_tokens = 12288

    # --- cache path helper ---
    stats_dir = Path(stats_dir)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    safe_ds_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", ds_name)

    def _cache_path(ln):
        collect_tag = "-".join(sorted(to_collect))
        fname = f"{ln}_{precision}_{collect_tag}{size_suffix}.npz"
        # New: <stats_dir>/<model>/c0/<ds>/<file>.npz
        new_path = stats_dir / model_name / "c0" / safe_ds_name / fname
        # Legacy: <stats_dir>/<model>/<ds>_stats/<file>.npz
        legacy_path = stats_dir / model_name / f"{safe_ds_name}_stats" / fname
        if not new_path.exists() and legacy_path.exists():
            return legacy_path
        return new_path

    # --- check disk cache per layer ---
    results = {}
    to_compute = []

    for ln in layer_names:
        cp = _cache_path(ln)
        if cp.exists() and not force_recompute:
            stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
            cached = load_cached_state(str(cp), {"sample_size": sample_size})
            if cached is not None:
                stat.load_state_dict(cached)
                results[ln] = stat
                print(f"Loaded cached {cp}")
                continue
        to_compute.append(ln)

    # --- compute missing layers ---
    if to_compute:
        effective_samples = sample_size or 100000
        print(f"Computing stats for {len(to_compute)} layer(s) over {ds_name} "
              f"(samples={effective_samples})")

        layer_stat = {
            ln: CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
            for ln in to_compute
        }

        # --- prepare data (before offload — selfgen needs full model) ---
        cfg = DATASET_REGISTRY.get(ds_name)

        if cfg is not None and cfg["type"] == "random":
            jsonl_path = None
        else:
            # Step 1: ensure JSONL cache (selfgen uses model.generate)
            jsonl_path = _ensure_jsonl_cache(
                ds_name, effective_samples,
                model=model, tokenizer=tokenizer, device=device,
                model_name=model_name, stats_dir=str(stats_dir),
            )

        # --- offload unused layers (after JSONL cache is ready) ---
        restore_fn = _offload_unused_layers(model, to_compute)

        # --- tokenize + GPU batches ---
        if jsonl_path is None:
            batch_iter = _random_iter(
                tokenizer.vocab_size, maxlen, batch_tokens, effective_samples)
        else:
            # Step 2: batch tokenize
            all_chunks = _jsonl_pretokenize(jsonl_path, tokenizer, maxlen, effective_samples)
            # Step 3: GPU batch iterator
            batch_iter = _gpu_batch_iter(all_chunks, batch_tokens, device)

        # --- forward loop ---
        n_tokens = 0
        if progress is None:
            progress = lambda x, **kw: x

        with torch.inference_mode():
            for batch_group in progress(batch_iter):
                for batch in batch_group:
                    batch = dict_to_(batch, device)

                    if len(to_compute) > 1:
                        with TraceDict(model, to_compute,
                                       retain_input=True, retain_output=False,
                                       stop=True) as td:
                            model(**batch)
                        for ln in to_compute:
                            feats = flatten_masked_batch(td[ln].input,
                                                         batch["attention_mask"])
                            feats = feats.to(dtype=dtype)
                            layer_stat[ln].add(feats)
                    else:
                        ln = to_compute[0]
                        with Trace(model, ln,
                                   retain_input=True, retain_output=False,
                                   stop=True) as tr:
                            model(**batch)
                        feats = flatten_masked_batch(tr.input,
                                                     batch["attention_mask"])
                        feats = feats.to(dtype=dtype)
                        layer_stat[ln].add(feats)

                    n_tokens += batch["attention_mask"].sum().item()

        print(f"Processed {n_tokens:,} tokens")

        # --- restore offloaded layers ---
        restore_fn()

        # --- save to disk ---
        for ln in to_compute:
            stat = layer_stat[ln]
            stat.to_(device="cpu")
            cp = _cache_path(ln)
            cp.parent.mkdir(parents=True, exist_ok=True)
            save_cached_state(str(cp), stat, {"sample_size": sample_size})
            results[ln] = stat
            print(f"Saved {cp}")

    # --- return ---
    if single:
        return results[layer_names[0]]
    return results
