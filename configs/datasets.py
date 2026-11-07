"""
Central registry of datasets for C₀ computation.

Used by layer_stats.py and mix_c0.py.
"""

DATASET_REGISTRY = {
    # Dolma domains (OLMo-2 training data)
    "wiki":       {"type": "hf", "path": "allenai/olmo-mix-1124", "name": "wiki", "field": "text"},
    "math":       {"type": "hf", "path": "allenai/dolmino-mix-1124", "name": "math", "field": "text"},
    "code":       {"type": "hf", "path": "allenai/olmo-mix-1124", "name": "starcoder", "field": "text"},
    "sci":        {"type": "hf", "path": "allenai/dolmino-mix-1124", "name": "pes2o", "field": "text"},
    "web":        {"type": "hf", "path": "allenai/olmo-mix-1124", "name": "dclm", "field": "text"},

    # Wikipedia / WikiText
    "wikipedia":  {"type": "hf", "path": "wikipedia", "name": "20200501.en", "field": "text"},
    "wikitext":   {"type": "hf", "path": "wikitext", "name": "wikitext-103-raw-v1", "field": "text"},

    # Self-generation
    "memoir-bos":   {"type": "selfgen", "init": "bos", "gen_length": 256},
    "memoir-rand1": {"type": "selfgen", "init": "random", "num_prefix_tokens": 1, "gen_length": 256},
    "memoir-rand2": {"type": "selfgen", "init": "random", "num_prefix_tokens": 2, "gen_length": 256},
    "memoir-rand3": {"type": "selfgen", "init": "random", "num_prefix_tokens": 3, "gen_length": 256},
    "memoir-rand4": {"type": "selfgen", "init": "random", "num_prefix_tokens": 4, "gen_length": 256},

    # Frequency-weighted seed: same as memoir-rand1 but the single seed token is
    # drawn from the empirical unigram distribution instead of uniformly over
    # vocabulary IDs. Isolates the effect of the uniform-sampling choice.
    "memoir-freq1": {"type": "selfgen", "init": "freq", "num_prefix_tokens": 1, "gen_length": 256,
                     "freq_file": "data/stats/{model}/token_freq_wikipedia.npz"},

    # Random tokens
    "random":     {"type": "random"},
}

# Legacy JSONL cache directory (for backward compat with non-selfgen)
LEGACY_JSONL_DIR = "data/c0_cache"
