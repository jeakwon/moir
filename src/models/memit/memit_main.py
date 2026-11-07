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
from .memit_hparams import MEMITHyperParams
from src.compute_z_batched import compute_z_batched

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    cache_c = kwargs.get("cache_c", None)
    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template, cache_c=cache_c)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(f"cuda:{hparams.device}"), val_mat.to(f"cuda:{hparams.device}")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
    cache_c: Optional[torch.Tensor] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Enable gradient checkpointing to reduce activation memory
    _grad_ckpt_was_enabled = getattr(model, "gradient_checkpointing", False)
    if not _grad_ckpt_was_enabled and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for MEMIT")

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

    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
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

    print(f"Computing z* for {len(requests)} requests (batched)...")
    z_list = compute_z_batched(
        model, tok, requests, hparams, z_layer, context_templates,
        batch_size=min(len(requests), 16),
    )
    zs = torch.stack(z_list, dim=1).to(f"cuda:{hparams.device}")

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
            track='out'
        ).T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix (or use identity for RoSE)
        if getattr(hparams, 'use_identity_c0', False):
            w = nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            )
            key_dim = w.shape[1]
            cov = torch.eye(key_dim, device=f"cuda:{hparams.device}")
        else:
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
            )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        # Add cache_c if provided (accumulated key info from previous edits)
        lhs = hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T
        if cache_c is not None:
            lhs = lhs + cache_c[i].to(lhs.device).double()

        adj_k = torch.linalg.solve(lhs, layer_ks)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Update cache_c with post-edit keys (if tracking)
    if cache_c is not None:
        context_templates = get_context_templates(model, tok)
        for i, layer in enumerate(hparams.layers):
            post_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            cache_c[i] += post_ks.cpu() @ post_ks.cpu().T
            del post_ks

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    # Restore gradient checkpointing state
    if not _grad_ckpt_was_enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

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

    On first call, computes ALL target layers (hparams.layers) in a single
    forward pass via TraceDict, then caches them in COV_CACHE.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in COV_CACHE or force_recompute:
        # Compute all target layers at once for efficiency
        all_layer_names = [
            hparams.rewrite_module_tmp.format(l) for l in hparams.layers
        ]
        missing = [
            ln for ln in all_layer_names
            if (model_name, ln) not in COV_CACHE or force_recompute
        ]

        if missing:
            print(f"Retrieving covariance for {model_name}: "
                  f"{len(missing)} layer(s) in one pass.")
            stats = layer_stats(
                model,
                tok,
                missing,
                hparams.stats_dir,
                mom2_dataset,
                to_collect=["mom2"],
                sample_size=mom2_n_samples,
                precision=mom2_dtype,
                batch_tokens=getattr(hparams, "mom2_batch_tokens", None),
                hparams=hparams,
                force_recompute=force_recompute,
                streaming=getattr(hparams, "mom2_streaming", False),
            )
            # layer_stats returns dict when given a list
            if isinstance(stats, dict):
                for ln, stat in stats.items():
                    COV_CACHE[(model_name, ln)] = stat.mom2.moment().float().to("cpu")
            else:
                COV_CACHE[(model_name, missing[0])] = stats.mom2.moment().float().to("cpu")
            del stats
            torch.cuda.empty_cache()

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}"))
        if inv
        else COV_CACHE[key].to(f"cuda:{hparams.device}")
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
            "Update matrix computed by MEMIT does not match original weight shape. "
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
