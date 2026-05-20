# HRM-mlx

Apple Silicon inference for **HRM-Text-1B**. Native MLX, 4-bit checkpoints, and small Metal fusions for single-response speed.

[![Hugging Face weights](https://img.shields.io/badge/Hugging%20Face-4--bit%20MLX%20weights-ffcc00)](https://huggingface.co/Aryagm/HRM-Text-1B-MLX-4bit)

![Benchmarks](assets/benchmark-chart.png)

![Reasoning demo](assets/reasoning-demo.png)

[Demo comparison video](marketing/assets/demo-comparison.mp4)

Marketing assets follow the same reproducible setup as `dflash-mlx`: the chart is
generated from `benchmarks/metrics_history.csv`, and the side-by-side demo video
is rendered from verified answer transcripts in `marketing/assets/captures`.
The demo uses the same functional-equation prompt as the dflash-mlx video and
shows the correct contradiction answer at the measured PyTorch MPS and HRM-mlx speeds.

HRM-Text-1B on MacBook Pro M4 Max, 32-core GPU:

| Runtime | tok/s | vs CPU |
|---|---:|---:|
| PyTorch CPU FP32 | 5.2 | 1.0x |
| PyTorch MPS BF16 | 22.0 | 4.3x |
| MLX BF16 | 24.7 | 4.8x |
| MLX 4-bit | 38.5 | 7.5x |
| **HRM-mlx fast path** | **56.0** | **10.9x** |

> Benchmark shape: 512 prompt tokens, 128 generated tokens. Absolute numbers vary by chip; the important number is single-stream decode speed.

## Why HRM needs this

[HRM-Text](https://huggingface.co/sapientinc/HRM-Text-1B) is not a normal 1B decoder. Each generated token runs a recurrent reasoning loop:

```text
H_cycles * (L_cycles + 1) = 2 * (3 + 1) = 8 stack passes/token
```

That makes single-token decode expensive. HRM-mlx keeps the full recurrence, but makes the Apple path practical with MLX-native kernels and persisted 4-bit checkpoints.

## Quick Start

```bash
git clone https://github.com/Aryagm/HRM-mlx.git
cd HRM-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the hosted 4-bit MLX checkpoint:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Aryagm/HRM-Text-1B-MLX-4bit",
    local_dir="exports/hrm-text-1b-mlx-mxfp4",
)
PY
```

Or build the 4-bit checkpoint locally from the original HRM-Text-1B weights:

```bash
hrm-mlx-quantize \
  --model-dir exports/hrm-text-1b-hf \
  --out-dir exports/hrm-text-1b-mlx-mxfp4 \
  --bits 4 \
  --group-size 32 \
  --mode mxfp4
```

Generate:

```bash
hrm-mlx \
  --model-dir exports/hrm-text-1b-mlx-mxfp4 \
  --prompt '<|im_start|><|quad_end|><|object_ref_end|>What is the derivative of (x^2) / ln(x)? Give the final simplified expression.<|im_end|>' \
  --max-tokens 420 \
  --temperature 0.7 \
  --dtype bfloat16 \
  --metal-swiglu
```

Expected final expression:

```text
x(2 ln(x) - 1) / (ln(x))^2
```

## What We Built

MLX does not ship an HRM-Text runtime. Everything below was implemented for this port:

- **HRM recurrent decode in MLX** with separate H/L stacks and per-recurrence KV caches
- **Packed checkpoint support** for the published HRM-Text-1B safetensors layout
- **Fast MLX primitives** for RMSNorm, RoPE, and scaled dot-product attention
- **Persisted 4-bit/MXFP4 checkpoints** so startup does not re-quantize the 2.2 GB model
- **Custom Metal SwiGLU activation** for a small extra decode win
- **Profiling tools** to break down attention, MLP, norms, cache update, and LM head time
- **PyTorch/MPS comparison harness** that remaps packed weights into the current HF module layout

## Benchmarks

Regenerate the README chart:

```bash
python scripts/generate_benchmark_chart.py
```

Run the fast path:

```bash
hrm-mlx-bench \
  --model-dir exports/hrm-text-1b-mlx-mxfp4 \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --dtype bfloat16 \
  --metal-swiglu
```

Profile decode:

```bash
hrm-mlx-profile \
  --model-dir exports/hrm-text-1b-mlx-mxfp4 \
  --prompt-tokens 512 \
  --decode-tokens 16 \
  --dtype bfloat16 \
  --metal-swiglu
```

Compare against PyTorch/MPS:

```bash
python -m benchmarks.benchmark_hf \
  --model-dir exports/hrm-text-1b-hf \
  --device mps \
  --dtype bfloat16 \
  --prompt-tokens 512 \
  --decode-tokens 128
```

## Optimization Notes

The fastest configuration tested so far is:

```text
MXFP4 weights + MLX fast RMSNorm/RoPE/SDPA + custom Metal SwiGLU
```

This gets about **56 decode tok/s** on M4 Max while keeping the full `H=2, L=3` recurrence. Reduced-cycle inference can be faster, but quality breaks on reasoning prompts, so it is not the default path.

## Project Layout

```text
mlx_hrm_text/              MLX model, generation, benchmark, profiler
conversion/                MLX conversion and quantization tools
benchmarks/                HF/MPS comparison and benchmark history
scripts/                   README chart generator
marketing/                 Captures and demo-video generator
assets/benchmark-chart.png Generated README chart
```

## License

Apache-2.0, matching the upstream HRM-Text release.
