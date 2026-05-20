#!/usr/bin/env python3
"""Capture real generation outputs for the marketing comparison video."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = ROOT / "marketing" / "assets" / "captures"
PROMPT_TEXT = (
    "Differentiate f(x) = x^2 / ln(x). Write a detailed solution with quotient-rule "
    "setup, derivative substitution, simplification, a product-rule cross-check, "
    "and a domain note. Put the final derivative in \\boxed{}."
)
HRM_PROMPT = f"<|im_start|><|quad_end|><|object_ref_end|>{PROMPT_TEXT}<|im_end|>"
FINAL_RE = re.compile(r"(final answer|### final|\\boxed|boxed\{)", re.IGNORECASE)


def looks_complete(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 900 or FINAL_RE.search(stripped) is None:
        return False
    return stripped.endswith((".", "!", "?", "}", "]"))


def write_capture(filename: str, name: str, tps: float, text: str) -> None:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = CAPTURE_DIR / filename
    path.write_text(json.dumps({"name": name, "tps": tps, "text": text.strip()}, indent=2) + "\n")
    print(f"Saved {path}")


def sample_torch_next(logits, temperature: float, generator):
    import torch

    logits = logits.float()
    if temperature <= 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def generate_torch(
    model_dir: Path,
    *,
    device_name: str,
    dtype_name: str,
    safety_max_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    import torch
    import transformers.utils.generic as transformers_generic
    from transformers import AutoTokenizer

    if not hasattr(transformers_generic, "split_attention_implementation"):
        transformers_generic.split_attention_implementation = lambda value: (None, value)

    sys.path.insert(0, str(ROOT))
    from benchmarks.benchmark_hf import dtype_from_name, load_packed_checkpoint_as_hf

    device = torch.device(device_name)
    dtype = dtype_from_name(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    model = load_packed_checkpoint_as_hf(model_dir, dtype).to(device).eval()

    encoded = tokenizer(HRM_PROMPT, return_tensors="pt", return_attention_mask=False, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    # Keep this aligned with benchmarks/benchmark_hf.py. The local generated HF files use
    # a newer PrefixLM mask API than the installed Transformers package, so PyTorch captures
    # use the causal path unless the environment is updated.
    token_type_ids = None
    generator = torch.Generator(device=device.type).manual_seed(seed)

    generated = []
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = out.past_key_values
        next_id = sample_torch_next(out.logits[:, -1, :], temperature, generator)

        for step in range(safety_max_tokens):
            token_id = int(next_id.item())
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            if step > 80 and step % 16 == 0:
                text = tokenizer.decode(generated, skip_special_tokens=False)
                if looks_complete(text):
                    break
            out = model(
                input_ids=next_id.to(device),
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = out.past_key_values
            next_id = sample_torch_next(out.logits[:, -1, :], temperature, generator)

    return tokenizer.decode(generated, skip_special_tokens=False)


def generate_mlx(
    model_dir: Path,
    *,
    safety_max_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    import mlx.core as mx
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from mlx_hrm_text.generate import sample_next
    from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu

    mx.random.seed(seed)
    set_metal_swiglu(True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    model = HrmTextForCausalLM.from_pretrained(model_dir, dtype="bfloat16")
    mx.eval(model.parameters())

    encoded = tokenizer(HRM_PROMPT, return_tensors="np", return_attention_mask=False, add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    cache = model.make_cache(max_length=None)
    logits = model.prefill(mx.array(prompt_ids, dtype=mx.int32), cache)
    next_id = sample_next(logits[0], temperature)

    generated: list[int] = []
    position = len(prompt_ids)
    for step in range(safety_max_tokens):
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        if step > 80 and step % 16 == 0:
            text = tokenizer.decode(generated, skip_special_tokens=False)
            if looks_complete(text):
                break
        logits = model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
        next_id = sample_next(logits[0], temperature)
        position += 1

    return tokenizer.decode(generated, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("cpu", "mps", "mlx", "video", "all"), default="video")
    parser.add_argument("--hf-model-dir", type=Path, default=ROOT / "exports" / "hrm-text-1b-hf")
    parser.add_argument("--mlx-model-dir", type=Path, default=ROOT / "exports" / "hrm-text-1b-mlx-mxfp4")
    parser.add_argument(
        "--safety-max-tokens",
        type=int,
        default=2048,
        help="Internal guard against runaway generations; normal captures stop on a final-answer marker.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.runtime in ("cpu", "all"):
        text = generate_torch(
            args.hf_model_dir,
            device_name="cpu",
            dtype_name="float32",
            safety_max_tokens=args.safety_max_tokens,
            temperature=args.temperature,
            seed=101,
        )
        write_capture("pytorch_cpu.json", "PyTorch CPU", 5.15, text)

    if args.runtime in ("mps", "video", "all"):
        text = generate_torch(
            args.hf_model_dir,
            device_name="mps",
            dtype_name="bfloat16",
            safety_max_tokens=args.safety_max_tokens,
            temperature=args.temperature,
            seed=202,
        )
        write_capture("pytorch_mps.json", "PyTorch MPS", 21.99, text)

    if args.runtime in ("mlx", "video", "all"):
        text = generate_mlx(
            args.mlx_model_dir,
            safety_max_tokens=args.safety_max_tokens,
            temperature=args.temperature,
            seed=303,
        )
        write_capture("hrm_mlx_fast.json", "HRM-mlx", 55.96, text)


if __name__ == "__main__":
    main()
