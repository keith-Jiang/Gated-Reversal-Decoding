"""
Unified single-GPU inference for Greedy, COIECD, AdaCAD, CoCoA, CoCoA_test,
SimpleInterp, and ARR methods.

Reads the dual-process JSONL format (same as CAD/AdaCAD/CoCoA) and runs
inference on a single GPU. Supports resume via append mode.

Usage:
    # Greedy baseline
    python -m methods.inference --method greedy \
        --model meta-llama/Meta-Llama-3-8B \
        --input_path data/nq_swap.jsonl \
        --output_path results/greedy/nq_swap.jsonl \
        --max_new_tokens 32

    # COIECD
    python -m methods.inference --method coiecd \
        --model meta-llama/Meta-Llama-3-8B \
        --input_path data/nq_swap.jsonl \
        --output_path results/coiecd/nq_swap.jsonl \
        --max_new_tokens 32 --alpha 1.0

    # Simple Interpolation (α=0.75)
    python -m methods.inference --method simple_interp \
        --model meta-llama/Meta-Llama-3-8B \
        --input_path data/nq_swap.jsonl \
        --output_path results/simple_interp/nq_swap.jsonl \
        --max_new_tokens 32 --interp_alpha 0.75
"""

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dual_process_data(input_path):
    """Load dual-process JSONL and group by input_index.

    Returns list of dicts: [{0: row_ctx, 1: row_prior}, ...]
    """
    groups = defaultdict(dict)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            groups[row["input_index"]][row["assigned_process"]] = row
    return [groups[k] for k in sorted(groups.keys())]


def load_done_indices(output_path):
    """Read already-completed indices for resume support."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["input_index"])
    return done


# ---------------------------------------------------------------------------
# Tokenization helpers
#
# `context_string` is a model-agnostic plain-text prompt (see
# benchmarks/prompt_templates.py). All models, including instruct/chat-tuned
# checkpoints, are evaluated as raw causal-LM continuations for consistency
# with the original CAD/AdaCAD protocol.
# ---------------------------------------------------------------------------

def build_batch_inputs(tokenizer, prompts, max_ctx_len, device):
    """Tokenize a batch of raw prompts."""
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, padding=True, return_tensors="pt",
            truncation=True, max_length=max_ctx_len,
            add_special_tokens=True,
        ).to(device)
    finally:
        tokenizer.padding_side = orig_padding_side
    return inputs


def build_single_input(tokenizer, prompt, max_ctx_len, device):
    """Tokenize a single raw prompt."""
    return tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=max_ctx_len,
        add_special_tokens=True,
    ).to(device)


def get_eos_token_ids(model, tokenizer):
    """Collect all token ids that should terminate generation.

    Merges `model.generation_config.eos_token_id` (which may be an int or a
    list, e.g. Gemma2.5-Instruct's `<|im_end|>` + `<|endoftext|>`) with
    `tokenizer.eos_token_id`.
    """
    ids = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gen_eos is not None:
        if isinstance(gen_eos, (list, tuple)):
            ids.update(gen_eos)
        else:
            ids.add(gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def create_decode_cache(model, input_ids, max_new_tokens):
    """Allocate enough static cache space for Gemma 2 token-by-token decoding."""
    if getattr(model.config, "model_type", None) != "gemma2":
        return None

    from transformers.cache_utils import HybridCache

    return HybridCache(
        model.config,
        max_batch_size=input_ids.shape[0],
        max_cache_len=input_ids.shape[1] + max_new_tokens,
        device=input_ids.device,
        dtype=model.dtype,
    )


def get_cache_position(input_ids, cache):
    """Gemma 2 needs explicit positions when using its fixed-size HybridCache."""
    if cache is None:
        return None
    return torch.arange(input_ids.shape[1], device=input_ids.device)


def get_decode_model_inputs(input_ids, attention_mask, past_kv, cache_position):
    """Request only the final distribution during custom token-by-token decoding."""
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_kv,
        "use_cache": True,
        "return_dict": True,
    }
    if cache_position is not None:
        inputs["cache_position"] = cache_position
    # Current decoder-only models support this argument and otherwise
    # materialize [batch, prompt_len, vocab] on the first step. That can add
    # gigabytes for long prompts although every custom decoder uses only -1.
    inputs["logits_to_keep"] = 1
    return inputs


# ---------------------------------------------------------------------------
# Greedy decoding (single forward, uses HF generate)
# ---------------------------------------------------------------------------

def greedy_generate(model, tokenizer, prompt, max_new_tokens, max_ctx_len):
    inputs = build_single_input(
        tokenizer, prompt, max_ctx_len, model.device,
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    generated = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Greedy decoding without context (single forward, uses prior-only prompt)
# ---------------------------------------------------------------------------

def greedy_no_ctx_generate(model, tokenizer, prior_prompt, max_new_tokens, max_ctx_len):
    inputs = build_single_input(
        tokenizer, prior_prompt, max_ctx_len, model.device,
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    generated = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# COIECD decoding (token-by-token with entropy constraint)
# ---------------------------------------------------------------------------

def coiecd_generate(model, tokenizer, ctx_prompt, prior_prompt,
                    max_new_tokens, max_ctx_len, alpha=1.0, threshold_ratio=4):
    from methods.coiecd import COIECDDecoding
    method = COIECDDecoding(alpha=alpha, threshold_ratio=threshold_ratio)

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Simple Interpolation decoding (token-by-token, τ = α fixed)
# ---------------------------------------------------------------------------

def simple_interp_generate(model, tokenizer, ctx_prompt, prior_prompt,
                           max_new_tokens, max_ctx_len, interp_alpha=0.75):
    from methods.simple_interp import SimpleInterpDecoding
    method = SimpleInterpDecoding(alpha=interp_alpha)

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CoCoA_test decoding (paper-faithful, token-by-token adaptive gating)
# ---------------------------------------------------------------------------

def cad_generate(model, tokenizer, ctx_prompt, prior_prompt,
                 max_new_tokens, max_ctx_len, alpha=1.0):
    from methods.cad import CADDecoding
    method = CADDecoding(alpha=alpha)

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# AdaCAD decoding (token-by-token, α_t = JSD, dual independent forward)
# ---------------------------------------------------------------------------

def adacad_generate(model, tokenizer, ctx_prompt, prior_prompt,
                    max_new_tokens, max_ctx_len, warmup_beta=0.0):
    from methods.adacad import AdaCADDecoding
    method = AdaCADDecoding(warmup_beta=warmup_beta)

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()



# ---------------------------------------------------------------------------
# CoCoA decoding (token-by-token, simplified single-GPU)
# ---------------------------------------------------------------------------

def cocoa_generate(model, tokenizer, ctx_prompt, prior_prompt,
                   max_new_tokens, max_ctx_len,
                   global_alpha=0.5, gamma=1.0, lambda_pm=100.0):
    from methods.cocoa import CoCoADecoding
    method = CoCoADecoding(global_alpha=global_alpha, gamma=gamma, lambda_pm=lambda_pm)

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CoCoA_test decoding (paper-faithful, token-by-token adaptive gating)
# ---------------------------------------------------------------------------

def cocoa_test_generate(model, tokenizer, ctx_prompt, prior_prompt,
                        max_new_tokens, max_ctx_len,
                        alpha_renyi=0.5, z=5.0, gamma=1.0, delta=1e-8):
    from methods.cocoa_test import CoCoATestDecoding
    method = CoCoATestDecoding(
        alpha_renyi=alpha_renyi, z=z, gamma=gamma, delta=delta,
    )

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Adaptive Regime Routing (ARR): two-stage regime gate + conflict strength
# ---------------------------------------------------------------------------

def arr_c_js_generate(model, tokenizer, ctx_prompt, prior_prompt,
                          max_new_tokens, max_ctx_len,
                          route_warmup_beta=0.0):
    from methods.arr_c_js import ARRCJsDecoding
    method = ARRCJsDecoding(warmup_beta=route_warmup_beta)
    return arr_generate_with_method(
        model, tokenizer, ctx_prompt, prior_prompt,
        max_new_tokens, max_ctx_len, method,
    )




def arr_c_kl_generate(model, tokenizer, ctx_prompt, prior_prompt,
                      max_new_tokens, max_ctx_len,
                      route_warmup_beta=0.0):
    from methods.arr_c_kl import ARRCKlDecoding
    method = ARRCKlDecoding(warmup_beta=route_warmup_beta)
    return arr_generate_with_method(
        model, tokenizer, ctx_prompt, prior_prompt,
        max_new_tokens, max_ctx_len, method,
    )


def arr_c_discrete_generate(model, tokenizer, ctx_prompt, prior_prompt,
                            max_new_tokens, max_ctx_len,
                            route_warmup_beta=0.0):
    from methods.arr_c_discrete import ARRCDiscreteDecoding
    method = ARRCDiscreteDecoding(warmup_beta=route_warmup_beta)
    return arr_generate_with_method(
        model, tokenizer, ctx_prompt, prior_prompt,
        max_new_tokens, max_ctx_len, method,
    )


def arr_generate_with_method(model, tokenizer, ctx_prompt, prior_prompt,
                             max_new_tokens, max_ctx_len, method):

    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        merged = method.get_next_token_logits(logits[0:1], logits[1:2])
        next_token = torch.argmax(merged, dim=-1)

        if next_token.item() in eos_ids:
            break
        generated_ids.append(next_token.item())

        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# GRD: zero-calibration first-conflict gate + tau* reversal strength
# ---------------------------------------------------------------------------


def grd_generate(model, tokenizer, ctx_prompt, prior_prompt,
                 max_new_tokens, max_ctx_len, grd_lambda=0.75,
                 grd_gate_mode="full"):
    """Gated Reversal Decoding via methods.grd.GRDDecoding."""
    from methods.grd import GRDDecoding

    method = GRDDecoding(grd_lambda=grd_lambda, gate_mode=grd_gate_mode)
    inputs = build_batch_inputs(
        tokenizer, [ctx_prompt, prior_prompt], max_ctx_len, model.device,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_kv = create_decode_cache(model, input_ids, max_new_tokens)
    cache_position = get_cache_position(input_ids, past_kv)
    generated_ids = []
    eos_ids = get_eos_token_ids(model, tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(**get_decode_model_inputs(
                input_ids, attention_mask, past_kv, cache_position,
            ))
            past_kv = out.past_key_values

        logits = out.logits[:, -1, :]
        next_token_id = method.select_next_token(logits[0], logits[1])

        if next_token_id in eos_ids:
            break
        generated_ids.append(next_token_id)

        next_token = torch.tensor([next_token_id], dtype=torch.long, device=model.device)
        n = input_ids.shape[0]
        input_ids = next_token.unsqueeze(0).expand(n, -1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(n, 1, dtype=torch.long, device=model.device)],
            dim=1,
        )
        if cache_position is not None:
            cache_position = cache_position[-1:] + 1

    tokenizer.padding_side = "right"
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def main():
    parser = argparse.ArgumentParser(
        description="Single-GPU inference for greedy and contrastive decoding methods",
    )
    parser.add_argument("--method", type=str, required=True,
                        choices=["greedy", "greedy_no_ctx",
                                 "cad", "adacad", "cocoa",
                                 "coiecd", "simple_interp",
                                 "cocoa_test",
                                 "arr_c_js", "arr_c_kl",
                                 "arr_c_discrete", "grd"])
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model id or local path")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_ctx_len", type=int, default=4064,
                        help="Max input context length (truncation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=list(DTYPE_MAP.keys()))
    # COIECD-specific
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="COIECD / CAD alpha parameter")
    parser.add_argument("--threshold_ratio", type=float, default=4,
                        help="COIECD entropy threshold ratio")
    # AdaCAD-specific
    parser.add_argument("--warmup_beta", type=float, default=0.0,
                        help="AdaCAD warmup beta (minimum alpha)")
    # SimpleInterp-specific
    parser.add_argument("--interp_alpha", type=float, default=0.75,
                        help="SimpleInterp alpha (ctx weight, prior weight = 1-alpha)")
    # CoCoA-specific (simplified)
    parser.add_argument("--cocoa_global_alpha", type=float, default=0.5,
                        help="CoCoA global mixing alpha")
    parser.add_argument("--cocoa_lambda_pm", type=float, default=100.0,
                        help="CoCoA z-score perturbation weight")
    # CoCoA_test-specific (paper-faithful)
    parser.add_argument("--cocoa_alpha_renyi", type=float, default=0.5,
                        help="CoCoA_test Rényi divergence order α")
    parser.add_argument("--cocoa_z", type=float, default=5.0,
                        help="CoCoA_test contextual peakedness weight z")
    parser.add_argument("--cocoa_gamma", type=float, default=1.0,
                        help="CoCoA / CoCoA_test gamma weight")
    parser.add_argument("--cocoa_delta", type=float, default=1e-8,
                        help="CoCoA_test numerical stability constant δ")
    # ARR-specific
    parser.add_argument("--route_warmup_beta", type=float, default=0.0)
    # GRD-specific
    parser.add_argument("--grd_lambda", type=float, default=0.75,
                        help="GRD commitment depth on prior-trusted conflict tokens; "
                             "tau = (1-grd_lambda) * tau_star")
    parser.add_argument("--grd_gate_mode", type=str, default="full",
                        choices=["full", "conf_only", "ent_only"],
                        help="GRD first-conflict gate signal selection")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Resume support
    done = load_done_indices(args.output_path)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE_MAP[args.dtype],
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"Loading data: {args.input_path}")
    samples = load_dual_process_data(args.input_path)
    pending = [s for s in samples if s.get(0, s.get(1, {})).get("input_index") not in done]
    print(f"Total: {len(samples)}, done: {len(done)}, pending: {len(pending)}")

    with open(args.output_path, "a", encoding="utf-8") as fout:
        for group in tqdm(pending, desc=f"{args.method}"):
            ctx_row = group.get(0, {})
            prior_row = group.get(1, {})
            idx = ctx_row.get("input_index", prior_row.get("input_index"))

            ctx_prompt = ctx_row.get("context_string", "")
            prior_prompt = prior_row.get("context_string", "")

            if args.method == "greedy":
                prediction = greedy_generate(
                    model, tokenizer, ctx_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                )
            elif args.method == "greedy_no_ctx":
                prediction = greedy_no_ctx_generate(
                    model, tokenizer, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                )
            elif args.method == "cad":
                prediction = cad_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    alpha=args.alpha,
                )
            elif args.method == "adacad":
                prediction = adacad_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    warmup_beta=args.warmup_beta,
                )
            elif args.method == "cocoa":
                prediction = cocoa_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    global_alpha=args.cocoa_global_alpha,
                    gamma=args.cocoa_gamma,
                    lambda_pm=args.cocoa_lambda_pm,
                )
            elif args.method == "coiecd":
                prediction = coiecd_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    alpha=args.alpha, threshold_ratio=args.threshold_ratio,
                )
            elif args.method == "simple_interp":
                prediction = simple_interp_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    interp_alpha=args.interp_alpha,
                )
            elif args.method == "cocoa_test":
                prediction = cocoa_test_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    alpha_renyi=args.cocoa_alpha_renyi,
                    z=args.cocoa_z,
                    gamma=args.cocoa_gamma,
                    delta=args.cocoa_delta,
                )
            elif args.method == "arr_c_js":
                prediction = arr_c_js_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    route_warmup_beta=args.route_warmup_beta,
                )
            elif args.method == "arr_c_kl":
                prediction = arr_c_kl_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    route_warmup_beta=args.route_warmup_beta,
                )
            elif args.method == "arr_c_discrete":
                prediction = arr_c_discrete_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    route_warmup_beta=args.route_warmup_beta,
                )
            elif args.method == "grd":
                prediction = grd_generate(
                    model, tokenizer, ctx_prompt, prior_prompt,
                    args.max_new_tokens, args.max_ctx_len,
                    grd_lambda=args.grd_lambda,
                    grd_gate_mode=args.grd_gate_mode,
                )
            else:
                raise ValueError(f"Unsupported method: {args.method}")

            result = {
                "input_index": idx,
                "benchmark": ctx_row.get("benchmark", ""),
                "task_type": ctx_row.get("task_type", ""),
                "gold_answers": ctx_row.get("gold_answers", ""),
                "article": ctx_row.get("article", ctx_row.get("context", "")),
                "prediction": prediction,
                "method": args.method,
                "grd_gate_mode": args.grd_gate_mode if args.method == "grd" else None,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    total = len(done) + len(pending)
    print(f"Done. {total} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
