from __future__ import annotations

import argparse
import json
import re
import sys
import time
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import transformers.utils.generic as transformers_generic
from transformers import AutoTokenizer

from mlx_hrm_text.generate import sample_next
from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu


if not hasattr(transformers_generic, "split_attention_implementation"):
    transformers_generic.split_attention_implementation = lambda value: (None, value)


PROMPT_PREFIX = "<|im_start|><|quad_end|><|object_ref_end|>"
PROMPT_SUFFIX = "<|im_end|>"


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str
    checker: str
    max_tokens: int


@dataclass
class TimeResult:
    runtime: str
    prompt: str
    correct: bool
    answer_tokens: int | None
    answer_seconds: float | None
    total_tokens: int
    total_seconds: float
    stopped_on_eos: bool
    text_tail: str


PROMPTS = [
    PromptCase(
        name="derivative",
        prompt="What is the derivative of (x^2) / ln(x)? Give the final simplified expression.",
        checker=r"(x\s*\(\s*2\s*\\?ln\s*\(?x\)?\s*-\s*1\s*\)\s*/\s*\(?\\?ln\s*\(?x\)?\)?\^?2|x\s*\(\s*2\s*\\?ln\s*\(?x\)?\s*-\s*1\s*\).*\\?ln\s*\(?x\)?.*2)",
        max_tokens=520,
    ),
    PromptCase(
        name="linear_equation",
        prompt="Solve 3x + 7 = 22. Show the work and give the final value of x.",
        checker=r"(\\boxed\{?\s*5\s*\}?|x\s*=\s*5|solution is.*5)",
        max_tokens=220,
    ),
    PromptCase(
        name="modular_arithmetic",
        prompt="Compute 17^23 mod 5. Explain briefly and give the final answer.",
        checker=r"(\\boxed\{?\s*3\s*\}?|final answer.*3|equiv\s*3\s*\\?mod\s*5)",
        max_tokens=520,
    ),
]


def format_prompt(prompt: str) -> str:
    if "<|im_start|>" in prompt:
        return prompt
    return f"{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}"


def normalize_text(text: str) -> str:
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("[", "").replace("]", "")
    return text


def is_correct(text: str, checker: str) -> bool:
    return re.search(checker, normalize_text(text), re.IGNORECASE | re.DOTALL) is not None


def run_mlx_case(
    runtime: str,
    model: HrmTextForCausalLM,
    tokenizer,
    case: PromptCase,
    *,
    check_every: int,
) -> TimeResult:
    encoded = tokenizer(format_prompt(case.prompt), return_tensors="np", return_attention_mask=False, add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    cache = model.make_cache(max_length=None)

    start = time.perf_counter()
    logits = model.prefill(mx.array(prompt_ids, dtype=mx.int32), cache)
    next_id = sample_next(logits[0], 0.0)

    generated: list[int] = []
    answer_tokens = None
    answer_seconds = None
    stopped_on_eos = False
    position = len(prompt_ids)

    for _ in range(case.max_tokens):
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            stopped_on_eos = True
            break
        generated.append(next_id)
        should_check = len(generated) % check_every == 0
        if answer_tokens is None and should_check and is_correct(tokenizer.decode(generated, skip_special_tokens=False), case.checker):
            answer_tokens = len(generated)
            answer_seconds = time.perf_counter() - start

        logits = model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
        next_id = sample_next(logits[0], 0.0)
        position += 1

    total_seconds = time.perf_counter() - start
    text = tokenizer.decode(generated, skip_special_tokens=False)
    if answer_tokens is None and is_correct(text, case.checker):
        answer_tokens = len(generated)
        answer_seconds = total_seconds
    return TimeResult(
        runtime=runtime,
        prompt=case.name,
        correct=answer_tokens is not None,
        answer_tokens=answer_tokens,
        answer_seconds=answer_seconds,
        total_tokens=len(generated),
        total_seconds=total_seconds,
        stopped_on_eos=stopped_on_eos,
        text_tail=text[-500:],
    )


def run_torch_case(
    runtime: str,
    model,
    tokenizer,
    case: PromptCase,
    *,
    device,
    check_every: int,
) -> TimeResult:
    import torch

    encoded = tokenizer(format_prompt(case.prompt), return_tensors="pt", return_attention_mask=False, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    generated: list[int] = []
    answer_tokens = None
    answer_seconds = None
    stopped_on_eos = False

    with torch.no_grad():
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        out = model(input_ids=input_ids, token_type_ids=None, use_cache=True, logits_to_keep=1)
        if device.type == "mps":
            torch.mps.synchronize()
        past_key_values = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(case.max_tokens):
            token_id = int(next_id.item())
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                stopped_on_eos = True
                break
            generated.append(token_id)
            should_check = len(generated) % check_every == 0
            if answer_tokens is None and should_check and is_correct(tokenizer.decode(generated, skip_special_tokens=False), case.checker):
                if device.type == "mps":
                    torch.mps.synchronize()
                answer_tokens = len(generated)
                answer_seconds = time.perf_counter() - start

            out = model(input_ids=next_id.to(device), past_key_values=past_key_values, use_cache=True, logits_to_keep=1)
            if device.type == "mps":
                torch.mps.synchronize()
            past_key_values = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        if device.type == "mps":
            torch.mps.synchronize()
        total_seconds = time.perf_counter() - start

    text = tokenizer.decode(generated, skip_special_tokens=False)
    if answer_tokens is None and is_correct(text, case.checker):
        answer_tokens = len(generated)
        answer_seconds = total_seconds
    return TimeResult(
        runtime=runtime,
        prompt=case.name,
        correct=answer_tokens is not None,
        answer_tokens=answer_tokens,
        answer_seconds=answer_seconds,
        total_tokens=len(generated),
        total_seconds=total_seconds,
        stopped_on_eos=stopped_on_eos,
        text_tail=text[-500:],
    )


def load_torch_model(model_dir: Path, device_name: str, dtype_name: str):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.benchmark_hf import dtype_from_name, load_packed_checkpoint_as_hf

    device = torch.device(device_name)
    dtype = dtype_from_name(dtype_name, device)
    model = load_packed_checkpoint_as_hf(model_dir, dtype).to(device).eval()
    return model, device


def print_table(results: list[TimeResult]) -> None:
    by_prompt: dict[str, list[TimeResult]] = {}
    for result in results:
        by_prompt.setdefault(result.prompt, []).append(result)

    for prompt, rows in by_prompt.items():
        baseline = next((row for row in rows if row.runtime == "PyTorch CPU"), None)
        base_answer = baseline.answer_seconds if baseline else None
        print(f"\n## {prompt}")
        print("runtime,correct,answer_s,total_s,answer_tokens,total_tokens,answer_speedup_vs_cpu")
        for row in rows:
            ratio = None
            if base_answer and row.answer_seconds:
                ratio = base_answer / row.answer_seconds
            answer_s = f"{row.answer_seconds:.3f}" if row.answer_seconds is not None else "NA"
            speedup = f"{ratio:.2f}x" if ratio is not None else "NA"
            print(
                f"{row.runtime},{row.correct},{answer_s},{row.total_seconds:.3f},"
                f"{row.answer_tokens},{row.total_tokens},{speedup}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure wall-clock time to first correct answer for HRM-Text runtimes.")
    parser.add_argument("--hf-model-dir", type=Path, required=True)
    parser.add_argument("--q4-model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--check-every", type=int, default=8, help="Decode/check correctness every N generated tokens.")
    parser.add_argument(
        "--runtime",
        choices=("cpu", "mps", "mlx-bf16", "mlx-q4", "all"),
        action="append",
        default=None,
        help="Runtime to run. Can be passed multiple times. Default: all.",
    )
    args = parser.parse_args()

    results: list[TimeResult] = []
    runtimes = set(args.runtime or ["all"])
    run_all = "all" in runtimes

    if (run_all or "cpu" in runtimes) and not args.skip_cpu:
        print("Loading PyTorch CPU...")
        cpu_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, use_fast=True, trust_remote_code=True)
        cpu_model, cpu_device = load_torch_model(args.hf_model_dir, "cpu", "float32")
        for case in PROMPTS:
            print(f"Running CPU {case.name}...")
            results.append(run_torch_case("PyTorch CPU", cpu_model, cpu_tokenizer, case, device=cpu_device, check_every=args.check_every))
        del cpu_model
        gc.collect()

    if run_all or "mps" in runtimes:
        print("Loading PyTorch MPS...")
        mps_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, use_fast=True, trust_remote_code=True)
        mps_model, mps_device = load_torch_model(args.hf_model_dir, "mps", "bfloat16")
        for case in PROMPTS:
            print(f"Running MPS {case.name}...")
            results.append(run_torch_case("PyTorch MPS", mps_model, mps_tokenizer, case, device=mps_device, check_every=args.check_every))
        del mps_model
        gc.collect()

    if run_all or "mlx-bf16" in runtimes:
        print("Loading MLX BF16 fast path...")
        mlx_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, use_fast=True, trust_remote_code=True)
        set_metal_swiglu(True)
        bf16_model = HrmTextForCausalLM.from_pretrained(args.hf_model_dir, dtype="bfloat16")
        mx.eval(bf16_model.parameters())
        for case in PROMPTS:
            print(f"Running MLX BF16 {case.name}...")
            results.append(run_mlx_case("MLX BF16 fast", bf16_model, mlx_tokenizer, case, check_every=args.check_every))
        del bf16_model
        gc.collect()

    if run_all or "mlx-q4" in runtimes:
        print("Loading MLX 4-bit fast path...")
        q4_tokenizer = AutoTokenizer.from_pretrained(args.q4_model_dir, use_fast=True, trust_remote_code=True)
        set_metal_swiglu(True)
        q4_model = HrmTextForCausalLM.from_pretrained(args.q4_model_dir, dtype="bfloat16")
        mx.eval(q4_model.parameters())
        for case in PROMPTS:
            print(f"Running MLX 4-bit {case.name}...")
            results.append(run_mlx_case("MLX 4-bit fast", q4_model, q4_tokenizer, case, check_every=args.check_every))

    print_table(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
