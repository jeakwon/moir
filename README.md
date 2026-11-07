# MOIR

Code for **MOIR: Let the Model Direct Its Own Story for Robust Cross-Domain Knowledge Editing**
(NeurIPS 2026).

Covariance-based editors such as MEMIT and AlphaEdit protect existing knowledge through a
preservation covariance C0 = E[k k^T] of the keys entering each edited MLP layer. C0 is
usually estimated from Wikipedia. MOIR estimates it from text the model generates itself,
each sequence seeded with a single random vocabulary token. It is a drop-in replacement for
C0 and changes nothing else in the editor.

## Naming

MOIR was called MEMOIR during development, so code and file names still say `memoir`:

- **MOIR** is the C0 source `memoir-rand1` in `configs/datasets.py` (one random seed token).
- `memoir-bos` (begin-of-sequence seed) and `memoir-rand2` to `memoir-rand4` (two to four
  random seed tokens) are the ablations of the paper.
- **C_Wiki** is the C0 source `wikipedia`.

This is unrelated to MEMOIR by Wang et al. (NeurIPS 2025), a lifelong-editing method.

## Layout

```
configs/datasets.py         C0 sources (registry keys -> corpus and cache names)
hparams/MEMIT/              MEMIT settings for the three models
hparams/AlphaEdit/          AlphaEdit settings; -t001 / -t0005 / -t005 set the null-space threshold
src/models/rome/layer_stats.py   self-generation of the MOIR corpus and C0 estimation
src/models/{memit,alphaedit}/    the editors (C0 enters here)
src/evaluate.py             edit metrics (BatchEvaluator)
scripts/run_memit_edit.py      MEMIT batch editing (builds C0 on first use)
scripts/run_alphaedit_edit.py  AlphaEdit batch editing
scripts/run_seq_edit.py        sequential editing, one edit per step, with checkpoints
scripts/reeval_edit_metrics.py edit metrics of saved weights (Efficacy, Generalization,
                               Specificity, Fluency, Portability)
scripts/eval_downstream.py     MMLU, GSM8K, HellaSwag, WinoGrande, ARC-Challenge (lm-eval)
scripts/eval_code.py           HumanEval
scripts/download_data.sh       CounterFact (and zsRE) with sha256 checks
results/                    pre-edit reference scores of the three models
rand1.py                    a faster vLLM generator (see Notes)
```

## Setup

```
conda create -n moir python=3.10 && conda activate moir
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m nltk.downloader punkt
bash scripts/download_data.sh
```

- **Llama-3.1 is gated.** Log in with `huggingface-cli login` before the first download, and
  never commit a token.
- **HumanEval executes model-generated code.** Run it in a sandbox. Check that
  `exec(check_program, exec_globals)` in `human_eval/execution.py` is not commented out:
  without it nothing runs and every sample counts as passed. A working setup reproduces the
  pre-edit pass@1 in `results/eval_code_*_baseline.json` up to sampling noise (.390 OLMo-2,
  .610 Llama-3.1, .622 Qwen3).
- **Paths are relative.** Run every command from the repository root. The hparams files point
  to `./data/stats` for all cached corpora, covariances and projections.
- **Offline runs.** With `HF_HUB_OFFLINE=1`, lm-eval only finds datasets that `datasets` has
  already prepared in its cache; downloaded hub files alone are not enough. Load each task's
  dataset once with network access, e.g.
  `python -c "from datasets import load_dataset; load_dataset('allenai/ai2_arc', 'ARC-Challenge')"`.

## Running MOIR

Models and edit layers (set in the hparams files):

| model | hparams | edit layers |
|---|---|---|
| allenai/OLMo-2-1124-7B-Instruct | `olmo2-7b-instruct` | 9-14 |
| meta-llama/Llama-3.1-8B-Instruct | `llama3-8b-instruct` | 3-8 |
| Qwen/Qwen3-8B | `qwen3-8b` | 12-17 |

### 1. Build C_MOIR

The first run that asks for `memoir-rand1` builds it, so start with a small MEMIT run:

```
python scripts/run_memit_edit.py --hparams hparams/MEMIT/qwen3-8b.yaml \
    --mom2_dataset memoir-rand1 --n_edits 10 --output /tmp/build_c0.json
```

This takes several GPU-hours per model, once. It does two things:
- **Self-generation.** It writes 100,000 sequences to
  `data/stats/<model>/jsonl/memoir-rand1_100000.jsonl`. Each sequence is seeded with one
  token drawn uniformly from the vocabulary, then sampled with Hugging Face `generate` in
  batches of 64: up to 256 new tokens, `temperature=1.0`, `top_p=1.0`. `top_k` is not set, so
  the model's generation config applies.
- **C0 estimation.** It accumulates C0 at the input of each edited `mlp.down_proj` in float32,
  then writes it to `data/stats/<model>/c0/memoir-rand1/`. Documents of 50 characters or fewer
  and token sequences shorter than 32 are skipped.

Later runs reuse both files. The cache key is the C0 name and the sample count, not the
generation settings.

Build each (model, C0) pair in one job before you start runs that use it. Two jobs writing
the same corpus file corrupt it, and a corpus with fewer than 100,000 lines is topped up
without a warning.

C_Wiki is built the same way with `--mom2_dataset wikipedia`. It reads the Hugging Face
`wikipedia` dataset, config `20200501.en`. That is a dataset script, and `datasets>=4.0` cannot
load dataset scripts. Use an older `datasets` release for this one step. Alternatively,
provide `data/stats/shared/jsonl/wikipedia_100000.jsonl`: the first 100,000 articles with more
than 50 characters, one `{"text": ...}` object per line.

### 2. Edit

Requests are the first N CounterFact records, in file order.

```
# MEMIT, batch (lambda = 15,000)
python scripts/run_memit_edit.py --hparams hparams/MEMIT/qwen3-8b.yaml \
    --mom2_dataset memoir-rand1 --n_edits 5000 \
    --output results/batch_edit/memit_5000/Qwen3-8B/memoir-rand1/edit_result.json \
    --save_weights results/batch_edit/memit_5000/Qwen3-8B/memoir-rand1/state_dict.pt

# AlphaEdit, batch
python scripts/run_alphaedit_edit.py --hparams hparams/AlphaEdit/qwen3-8b-t005.yaml \
    --mom2_dataset memoir-rand1 --n_edits 20000 \
    --output_dir results/batch_edit/alphaedit_t005_20000/Qwen3-8B/memoir-rand1

# Sequential (memit_cached or alphaedit); checkpoints at 10, 20, 50, ..., 10000 edits
python scripts/run_seq_edit.py --alg alphaedit --hparams hparams/AlphaEdit/qwen3-8b-t005.yaml \
    --mom2_dataset memoir-rand1 --n_edits 5000 \
    --output_dir results/seq_edit/alphaedit_qwen3_memoir-rand1
```

- **AlphaEdit thresholds.** The paper's AlphaEdit runs use the per-model null-space
  thresholds: `olmo2-7b-instruct-t001` (1e-3), `llama3-8b-instruct-t0005` (5e-4) and
  `qwen3-8b-t005` (5e-3). The files without a suffix use 0.02.
- **Projector cache.** The projector is computed from C0 on first use and cached under
  `data/stats/<model>/null_projection/`.
- **Saved weights.** Each run saves the edited `down_proj` weights (`state_dict.pt`) and quick
  edit metrics (`edit_result.json`).
- **Sequential runs** resume from `latest/` after an interruption.

### 3. Evaluate

```
M=Qwen/Qwen3-8B
W=results/batch_edit/memit_5000/Qwen3-8B/memoir-rand1/state_dict.pt
python scripts/reeval_edit_metrics.py --model_name $M --edited_weights $W --n_edits 5000 \
    --output ${W%/*}/edit_metrics_full.json
python scripts/eval_downstream.py --model_name $M --edited_weights $W --output ${W%/*}/lm_eval.json
python scripts/eval_code.py --model_name $M --edited_weights $W --benchmark humaneval \
    --output ${W%/*}/humaneval.json
```

Omit `--edited_weights` to evaluate the unedited model.

How each score is computed:
- **Downstream tasks** use lm-eval 0.4.11 task defaults (GSM8K 5-shot, the rest 0-shot) and
  no chat template. The paper reports MMLU `acc`, GSM8K `exact_match,strict-match`, HellaSwag
  `acc_norm`, WinoGrande `acc` and ARC-Challenge `acc_norm`.
- **HumanEval** draws one sample per problem (temperature 0.2, top_p 0.95) and reports pass@1.
- **Fluency** in the paper is normalised by the unedited model's mean over the first 2,000
  CounterFact prompts (`results/baseline_fluency*.json`).
- **Aggregate scores.** S_e is the harmonic mean of the five edit metrics, and S_p is the
  harmonic mean of the six preservation scores.

### Resources

Every stage fits on one GPU with 40 GB of memory; plan for at least 64 GB of host memory.
Approximate GPU time on one A100, per run:

| stage | time |
|---|---|
| MEMIT batch, 1K to 5K edits | 0.2-1.1 h |
| AlphaEdit batch, 5K to 20K edits | 2.4-18 h, depending on the model |
| sequential editing, 10K edits | 17-35 h, depending on the model |
| `eval_downstream.py`, per model | 0.7-2.3 h |
| `eval_code.py`, per model | about 5 min |

## Notes

- **No seeds.** Self-generation, the MEMIT/AlphaEdit context templates and HumanEval
  sampling are unseeded, so a fresh C_MOIR differs from the one in the paper, and repeated
  runs differ slightly.
- **rand1.py** generates seed-token text with vLLM, which is faster. It is not the generator
  of the paper's corpora (`layer_stats.py` above). Its defaults differ: 10,000 documents,
  `top_p` 0.95, and always 256 new tokens. Its output format is not read by the C0 code.
- **Other options.** `run_memit_edit.py` and `run_seq_edit.py` also accept
  `--dataset zsre` (run `scripts/download_data.sh --zsre` first). `run_seq_edit.py` also has
  optional RECT/PRUNE post-processing (`--regularizer`).

## License

MIT (`LICENSE`). The editing code builds on ROME, MEMIT, AlphaEdit and EasyEdit, all MIT; see
`THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@inproceedings{kwon2026moir,
  title     = {{MOIR}: Let the Model Direct Its Own Story for Robust Cross-Domain Knowledge Editing},
  author    = {Kwon, Jea and Kim, Jiwon and Kim, Dong-Kyum and Cha, Meeyoung},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
