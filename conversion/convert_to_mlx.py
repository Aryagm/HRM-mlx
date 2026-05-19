from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}


def mlx_dtype(name: str):
    return {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }[name]


def is_floating_array(array: mx.array) -> bool:
    return array.dtype in (mx.float16, mx.float32, mx.bfloat16)


def copy_tokenizer_files(source: Path, out_dir: Path) -> None:
    for file in source.iterdir():
        if file.name in TOKENIZER_FILES and file.is_file():
            shutil.copy2(file, out_dir / file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an HRM-Text HF export to an MLX checkpoint.")
    parser.add_argument("--hf-dir", type=Path, required=True, help="Directory containing config.json and model.safetensors.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    config_path = args.hf_dir / "config.json"
    weight_path = args.hf_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {args.hf_dir}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {args.hf_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    config["mlx_format_version"] = 1
    config["mlx_dtype"] = args.dtype
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    dtype = mlx_dtype(args.dtype)
    weights = mx.load(str(weight_path))
    converted = {name: value.astype(dtype) if is_floating_array(value) else value for name, value in weights.items()}
    mx.save_safetensors(str(args.out_dir / "model.safetensors"), converted)

    copy_tokenizer_files(args.hf_dir, args.out_dir)
    print(f"[convert-mlx] wrote {len(converted)} tensors to {args.out_dir}")


if __name__ == "__main__":
    main()
