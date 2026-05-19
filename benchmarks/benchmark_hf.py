from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


def load_packed_checkpoint_as_hf(model_dir: Path, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=True)

    raw = load_file(model_dir / "model.safetensors", device="cpu")
    state = {}
    h = config.num_attention_heads
    hd = config.head_dim
    hidden = h * hd
    inter = config.intermediate_size

    for key, value in raw.items():
        value = value.to(dtype=dtype) if value.is_floating_point() else value
        if ".attn.gqkv_proj.weight" in key:
            prefix = key.replace(".attn.gqkv_proj.weight", ".self_attn")
            gate, query, key_w, value_w = value.split((hidden, hidden, hidden, hidden), dim=0)
            state[f"{prefix}.gate_proj.weight"] = gate
            state[f"{prefix}.q_proj.weight"] = query
            state[f"{prefix}.k_proj.weight"] = key_w
            state[f"{prefix}.v_proj.weight"] = value_w
        elif ".attn.o_proj.weight" in key:
            state[key.replace(".attn.o_proj.weight", ".self_attn.o_proj.weight")] = value
        elif ".mlp.gate_up_proj.weight" in key:
            prefix = key.replace(".mlp.gate_up_proj.weight", ".mlp")
            gate, up = value.split((inter, inter), dim=0)
            state[f"{prefix}.gate_proj.weight"] = gate
            state[f"{prefix}.up_proj.weight"] = up
        else:
            state[key] = value

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"HF remap failed: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


def run_once(model, device: torch.device, prompt_tokens: int, decode_tokens: int, prefix_lm: bool) -> tuple[float, float]:
    input_ids = (torch.arange(prompt_tokens, dtype=torch.long, device=device) % model.config.vocab_size).unsqueeze(0)
    token_type_ids = torch.ones_like(input_ids) if prefix_lm else None

    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    out = model(input_ids=input_ids, token_type_ids=token_type_ids, use_cache=True, logits_to_keep=1)
    if device.type == "mps":
        torch.mps.synchronize()
    prefill_seconds = time.perf_counter() - start

    past_key_values = out.past_key_values
    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(decode_tokens):
        out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True, logits_to_keep=1)
        past_key_values = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    if device.type == "mps":
        torch.mps.synchronize()
    decode_seconds = time.perf_counter() - start

    return prefill_seconds, decode_seconds


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HRM-Text through Transformers/PyTorch.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefix-lm", action="store_true", help="Pass token_type_ids for PrefixLM prefill.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this PyTorch build.")
    dtype = dtype_from_name(args.dtype, device)

    model = load_packed_checkpoint_as_hf(args.model_dir, dtype).to(device).eval()

    with torch.no_grad():
        for _ in range(args.warmup):
            run_once(model, device, args.prompt_tokens, args.decode_tokens, args.prefix_lm)

        prefill_times = []
        decode_times = []
        for _ in range(args.repeats):
            prefill_seconds, decode_seconds = run_once(model, device, args.prompt_tokens, args.decode_tokens, args.prefix_lm)
            prefill_times.append(prefill_seconds)
            decode_times.append(decode_seconds)

    prefill = sum(prefill_times) / len(prefill_times)
    decode = sum(decode_times) / len(decode_times)
    print(f"device: {device.type}")
    print(f"dtype: {dtype}")
    print(f"prompt_tokens: {args.prompt_tokens}")
    print(f"decode_tokens: {args.decode_tokens}")
    print(f"prefill_seconds: {prefill:.4f}")
    print(f"prefill_tok_s: {args.prompt_tokens / prefill:.2f}")
    print(f"decode_seconds: {decode:.4f}")
    print(f"decode_tok_s: {args.decode_tokens / decode:.2f}")


if __name__ == "__main__":
    main()
