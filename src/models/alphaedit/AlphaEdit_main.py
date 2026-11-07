import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.rome.layer_stats import layer_stats
from src.util import nethook
from src.util.nethook import upd_matrix_match_shape
from src.util.generate import generate_fast
from src.util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P = None
P_loaded = False
cache_c = None
cache_c_new = False


def get_cache_c():
    """Get the current cache_c tensor (for saving)."""
    return cache_c


def set_cache_c(tensor):
    """Set cache_c from a saved tensor (for resume)."""
    global cache_c, cache_c_new
    cache_c = tensor
    cache_c_new = True


def _load_c0(c0_dir: str, layers: list) -> torch.Tensor:
    """Load pre-computed C₀ matrices from npz files."""
    c0_list = []
    for layer in layers:
        path = os.path.join(c0_dir, f"layer_{layer}.npz")
        data = np.load(path)
        c0_list.append(torch.from_numpy(data["c0"]).float())
    return torch.stack(c0_list, dim=0)

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    reset_cache=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :param reset_cache: If true, will reset cache_c_new to False, forcing re-initialization of cache_c.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new
    
    # Reset cache if requested
    if reset_cache:
        cache_c_new = False

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    # Calculate the null-space projection matrix P
    # Saved per-layer under <stats_dir>/<model>/null_projection/<c0_name>/<layer>.npz
    if not P_loaded:
        import re
        model_short = hparams.model_name.rsplit("/")[-1]
        ds_name = hparams.mom2_dataset if not (hparams.c0_dir and os.path.isdir(hparams.c0_dir)) else Path(hparams.c0_dir).name
        safe_ds = re.sub(r"[^a-zA-Z0-9_\-]", "_", ds_name)
        # Include threshold in cache path for non-default values
        p_subdir = safe_ds
        if hparams.nullspace_threshold != 0.02:
            p_subdir = f"{safe_ds}_tau{hparams.nullspace_threshold}"
        p_dir = Path(hparams.stats_dir) / model_short / "null_projection" / p_subdir
        p_dir.mkdir(parents=True, exist_ok=True)

        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "gpt2-xl" in hparams.model_name.lower():
            dim = W_out.shape[0]
        else:
            dim = W_out.shape[1]
        del W_out
        P = torch.zeros((len(hparams.layers), dim, dim), device="cpu")

        all_cached = True
        for i, layer in enumerate(hparams.layers):
            layer_name = hparams.rewrite_module_tmp.format(layer)
            p_path = p_dir / f"{layer_name.replace('.', '_')}_null_proj.npz"
            if p_path.exists():
                data = np.load(str(p_path))
                P[i,:,:] = torch.from_numpy(data["P"]).float()
                print(f"  Loaded P for {layer_name} from {p_path}")
            else:
                all_cached = False
                print(f"  Computing P for {layer_name}...")
                P[i,:,:] = get_project(model, tok, layer, hparams)
                np.savez(str(p_path), P=P[i,:,:].numpy())
                print(f"  Saved P to {p_path}")

        if all_cached:
            print(f"All P loaded from {p_dir}")
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        if hparams.c0_dir and os.path.isdir(hparams.c0_dir):
            cache_c = _load_c0(hparams.c0_dir, hparams.layers)
            print(f"Loaded C₀ from {hparams.c0_dir} (norm={cache_c.norm():.2f})")
        else:
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            if "llama" in hparams.model_name.lower() or "olmo" in hparams.model_name.lower() or "qwen" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower():
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            elif "gpt2-xl" in hparams.model_name.lower():
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            del W_out
        cache_c_new = True
    
    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(f"cuda:{hparams.device}")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        # Cast all operands to float32 to avoid dtype mismatch with fp16 models
        dev = f"cuda:{hparams.device}"
        P_i = P[i,:,:].to(device=dev, dtype=torch.float)
        layer_ks_f = layer_ks.to(device=dev, dtype=torch.float)
        cache_c_i = cache_c[i,:,:].to(device=dev, dtype=torch.float)
        resid_f = resid.to(device=dev, dtype=torch.float)
        upd_matrix = torch.linalg.solve(
                P_i @ (layer_ks_f @ layer_ks_f.T + cache_c_i) + hparams.L2*torch.eye(layer_ks_f.shape[0], dtype=torch.float, device=dev),
                P_i @ layer_ks_f @ resid_f.T
        )

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T