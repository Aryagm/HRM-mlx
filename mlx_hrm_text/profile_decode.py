from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu, set_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MLX HRM-Text decode operation groups.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--detail", action="store_true", help="Profile individual attention/MLP sub-ops. Adds more sync overhead.")
    parser.add_argument("--metal-swiglu", action="store_true", help="Use a custom Metal kernel for SwiGLU activation.")
    args = parser.parse_args()

    set_metal_swiglu(args.metal_swiglu)
    model = HrmTextForCausalLM.from_pretrained(args.model_dir, dtype=args.dtype)
    mx.eval(model.parameters())

    input_ids = mx.arange(args.prompt_tokens, dtype=mx.int32) % model.config.vocab_size
    cache = model.make_cache()
    logits = model.prefill(input_ids, cache)
    token = mx.argmax(logits[0], axis=-1).astype(mx.int32)
    mx.eval(token)

    profile: dict[str, list[float | int]] = {}
    set_profile(profile, detail=args.detail)
    for idx in range(args.decode_tokens):
        logits = model.decode_one(token, args.prompt_tokens + idx, cache)
        token = mx.argmax(logits[0], axis=-1).astype(mx.int32)
        mx.eval(token)
    set_profile(None)

    rows = []
    total = 0.0
    for name, (seconds, calls) in profile.items():
        seconds = float(seconds)
        calls = int(calls)
        total += seconds
        rows.append((name, seconds, calls, seconds / max(calls, 1)))

    rows.sort(key=lambda row: row[1], reverse=True)
    print(f"decode_tokens: {args.decode_tokens}")
    print(f"profiled_seconds: {total:.4f}")
    print("name,total_s,calls,avg_ms,share")
    for name, seconds, calls, avg in rows:
        share = 100.0 * seconds / total if total else 0.0
        print(f"{name},{seconds:.6f},{calls},{avg * 1000:.3f},{share:.1f}%")


if __name__ == "__main__":
    main()
