from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu


def run_once(model: HrmTextForCausalLM, prompt_tokens: int, decode_tokens: int) -> tuple[float, float]:
    input_ids = mx.arange(prompt_tokens, dtype=mx.int32) % model.config.vocab_size
    max_length = prompt_tokens + decode_tokens if getattr(model, "use_static_cache", False) else None
    cache = model.make_cache(max_length=max_length)

    start = time.perf_counter()
    logits = model.prefill(input_ids, cache)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - start

    next_id = mx.argmax(logits[0], axis=-1).astype(mx.int32)
    mx.eval(next_id)

    start = time.perf_counter()
    for idx in range(decode_tokens):
        logits = model.decode_one(next_id, prompt_tokens + idx, cache)
        next_id = mx.argmax(logits[0], axis=-1).astype(mx.int32)
        mx.eval(next_id)
    decode_seconds = time.perf_counter() - start

    return prefill_seconds, decode_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX HRM-Text prefill and decode throughput.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
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

    for _ in range(args.warmup):
        run_once(model, args.prompt_tokens, args.decode_tokens)

    prefill_times = []
    decode_times = []
    for _ in range(args.repeats):
        prefill_seconds, decode_seconds = run_once(model, args.prompt_tokens, args.decode_tokens)
        prefill_times.append(prefill_seconds)
        decode_times.append(decode_seconds)

    prefill = sum(prefill_times) / len(prefill_times)
    decode = sum(decode_times) / len(decode_times)
    print(f"prompt_tokens: {args.prompt_tokens}")
    print(f"decode_tokens: {args.decode_tokens}")
    print(f"prefill_seconds: {prefill:.4f}")
    print(f"prefill_tok_s: {args.prompt_tokens / prefill:.2f}")
    print(f"decode_seconds: {decode:.4f}")
    print(f"decode_tok_s: {args.decode_tokens / decode:.2f}")


if __name__ == "__main__":
    main()
