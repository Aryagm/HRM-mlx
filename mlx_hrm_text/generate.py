from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu


def sample_next(logits: mx.array, temperature: float) -> int:
    logits = logits.astype(mx.float32)
    if temperature <= 1e-6:
        token = mx.argmax(logits, axis=-1)
    else:
        scaled = logits / temperature
        token = mx.random.categorical(scaled)
    mx.eval(token)
    return int(token.item())


def generate(
    model: HrmTextForCausalLM,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    eos_token_id: int | None,
) -> str:
    encoded = tokenizer(prompt, return_tensors="np", return_attention_mask=False, add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    if len(prompt_ids) == 0:
        raise ValueError("Prompt produced no tokens.")

    max_length = len(prompt_ids) + max_tokens if getattr(model, "use_static_cache", False) else None
    cache = model.make_cache(max_length=max_length)
    input_ids = mx.array(prompt_ids, dtype=mx.int32)
    logits = model.prefill(input_ids, cache)
    next_id = sample_next(logits[0], temperature)

    generated: list[int] = []
    position = len(prompt_ids)
    for _ in range(max_tokens):
        if eos_token_id is not None and next_id == eos_token_id:
            break
        generated.append(next_id)
        logits = model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
        next_id = sample_next(logits[0], temperature)
        position += 1

    return tokenizer.decode(generated, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an MLX HRM-Text checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing config.json and model.safetensors.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--static-cache", action="store_true", help="Preallocate the KV cache instead of growing it dynamically.")
    parser.add_argument("--h-cycles", type=int, default=None, help="Override H cycles for faster approximate inference.")
    parser.add_argument("--l-cycles", type=int, default=None, help="Override L cycles for faster approximate inference.")
    parser.add_argument("--quantize-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--quantize-group-size", type=int, default=64)
    parser.add_argument("--quantize-mode", choices=("affine", "mxfp4", "nvfp4", "mxfp8"), default="affine")
    parser.add_argument("--metal-swiglu", action="store_true", help="Use a custom Metal kernel for SwiGLU activation.")
    args = parser.parse_args()

    set_metal_swiglu(args.metal_swiglu)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    model = HrmTextForCausalLM.from_pretrained(args.model_dir, dtype=args.dtype)
    if args.quantize_bits is not None:
        import mlx.nn as nn

        nn.quantize(model, bits=args.quantize_bits, group_size=args.quantize_group_size, mode=args.quantize_mode)
    if args.h_cycles is not None:
        model.model.H_cycles = args.h_cycles
    if args.l_cycles is not None:
        model.model.L_cycles = args.l_cycles
    model.use_static_cache = args.static_cache
    mx.eval(model.parameters())

    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
