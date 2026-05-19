from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_hrm_text.model import HrmTextForCausalLM


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}


def copy_tokenizer_files(source: Path, out_dir: Path) -> None:
    for file in source.iterdir():
        if file.name in TOKENIZER_FILES and file.is_file():
            shutil.copy2(file, out_dir / file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist a quantized MLX HRM-Text checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", choices=("affine", "mxfp4", "nvfp4", "mxfp8"), default="affine")
    args = parser.parse_args()

    model = HrmTextForCausalLM.from_pretrained(args.model_dir, dtype=args.dtype)
    nn.quantize(model, bits=args.bits, group_size=args.group_size, mode=args.mode)
    mx.eval(model.parameters())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model_dir / "config.json").read_text())
    config["mlx_format_version"] = 1
    config["mlx_dtype"] = args.dtype
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out_dir / "quantization.json").write_text(
        json.dumps(
            {
                "bits": args.bits,
                "group_size": args.group_size,
                "mode": args.mode,
            },
            indent=2,
        )
        + "\n"
    )
    model.save_weights(str(args.out_dir / "model.safetensors"))
    copy_tokenizer_files(args.model_dir, args.out_dir)
    print(f"[quantize-mlx] wrote {args.bits}-bit checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
