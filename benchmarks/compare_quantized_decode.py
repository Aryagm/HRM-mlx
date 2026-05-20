from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import transformers.utils.generic as transformers_generic
from transformers import AutoTokenizer

from mlx_hrm_text.generate import sample_next
from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu


PROMPT_PREFIX = "<|im_start|><|quad_end|><|object_ref_end|>"
PROMPT_SUFFIX = "<|im_end|>"

DEFAULT_PROMPTS = [
    "What is the derivative of (x^2) / ln(x)? Give the final simplified expression.",
    (
        "Differentiate f(x) = x^2 / ln(x). Write a detailed solution with quotient-rule "
        "setup, derivative substitution, simplification, a product-rule cross-check, "
        "and a domain note. Put the final derivative in boxed braces."
    ),
    "Solve 3x + 7 = 22. Show the work and give the final value of x.",
    "Compute 17^23 mod 5. Explain briefly and give the final answer.",
]

if not hasattr(transformers_generic, "split_attention_implementation"):
    transformers_generic.split_attention_implementation = lambda value: (None, value)


@dataclass
class TokenInfo:
    token_id: int
    token: str
    score: float


@dataclass
class StepRecord:
    selected_id: int
    selected_text: str
    eos_rank: int | None
    eos_score: float | None
    top5: list[TokenInfo]


@dataclass
class RunRecord:
    text: str
    token_ids: list[int]
    steps: list[StepRecord]
    stopped_on_eos: bool


def format_prompt(prompt: str) -> str:
    if "<|im_start|>" in prompt:
        return prompt
    return f"{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}"


def topk_info(logits: mx.array, tokenizer, eos_token_id: int | None, k: int = 5) -> tuple[list[TokenInfo], int | None, float | None]:
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    scores = np.array(logits)
    top_ids = np.argsort(scores)[-k:][::-1].tolist()
    top_scores = scores[top_ids].tolist()
    top5 = [
        TokenInfo(
            token_id=token_id,
            token=tokenizer.decode([token_id], skip_special_tokens=False),
            score=score,
        )
        for token_id, score in zip(top_ids, top_scores, strict=True)
    ]

    if eos_token_id is None:
        return top5, None, None

    eos_score = float(scores[eos_token_id])
    eos_rank = int(np.sum(scores > eos_score) + 1)
    return top5, eos_rank, eos_score


def load_model(model_dir: Path, dtype: str, metal_swiglu: bool) -> HrmTextForCausalLM:
    set_metal_swiglu(metal_swiglu)
    model = HrmTextForCausalLM.from_pretrained(model_dir, dtype=dtype)
    mx.eval(model.parameters())
    return model


def run_greedy(
    model: HrmTextForCausalLM,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
) -> RunRecord:
    encoded = tokenizer(format_prompt(prompt), return_tensors="np", return_attention_mask=False, add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    cache = model.make_cache(max_length=None)
    logits = model.prefill(mx.array(prompt_ids, dtype=mx.int32), cache)

    steps: list[StepRecord] = []
    generated: list[int] = []
    position = len(prompt_ids)
    stopped_on_eos = False

    for _ in range(max_tokens):
        step_logits = logits[0]
        next_id = sample_next(step_logits, 0.0)
        top5, eos_rank, eos_score = topk_info(step_logits, tokenizer, tokenizer.eos_token_id)
        steps.append(
            StepRecord(
                selected_id=next_id,
                selected_text=tokenizer.decode([next_id], skip_special_tokens=False),
                eos_rank=eos_rank,
                eos_score=eos_score,
                top5=top5,
            )
        )

        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            stopped_on_eos = True
            break
        generated.append(next_id)
        logits = model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
        position += 1

    return RunRecord(
        text=tokenizer.decode(generated, skip_special_tokens=False),
        token_ids=generated,
        steps=steps,
        stopped_on_eos=stopped_on_eos,
    )


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def summarize_prompt(prompt: str, bf16: RunRecord, q4: RunRecord) -> dict:
    divergence = first_divergence(bf16.token_ids, q4.token_ids)
    result = {
        "prompt": prompt,
        "bf16_tokens": len(bf16.token_ids),
        "q4_tokens": len(q4.token_ids),
        "bf16_stopped_on_eos": bf16.stopped_on_eos,
        "q4_stopped_on_eos": q4.stopped_on_eos,
        "first_divergence": divergence,
        "bf16_tail": bf16.text[-500:],
        "q4_tail": q4.text[-500:],
    }
    if divergence is not None:
        if divergence < len(bf16.steps):
            result["bf16_at_divergence"] = asdict(bf16.steps[divergence])
        if divergence < len(q4.steps):
            result["q4_at_divergence"] = asdict(q4.steps[divergence])
    if bf16.stopped_on_eos:
        stop_idx = len(bf16.token_ids)
        result["bf16_eos_step"] = stop_idx
        if stop_idx < len(bf16.steps):
            result["bf16_stop_choice"] = asdict(bf16.steps[stop_idx])
        if stop_idx < len(q4.steps):
            result["q4_at_bf16_eos_step"] = asdict(q4.steps[stop_idx])
    if q4.stopped_on_eos:
        stop_idx = len(q4.token_ids)
        result["q4_eos_step"] = stop_idx
        if stop_idx < len(q4.steps):
            result["q4_stop_choice"] = asdict(q4.steps[stop_idx])
        if stop_idx < len(bf16.steps):
            result["bf16_at_q4_eos_step"] = asdict(bf16.steps[stop_idx])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLX BF16 and quantized HRM-Text greedy decode behavior.")
    parser.add_argument("--bf16-model-dir", type=Path, required=True)
    parser.add_argument("--q4-model-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=420)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--metal-swiglu", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--prompt", action="append", default=None, help="Prompt to compare. Can be passed multiple times.")
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(args.bf16_model_dir, use_fast=True, trust_remote_code=True)
    bf16_model = load_model(args.bf16_model_dir, args.dtype, args.metal_swiglu)
    q4_model = load_model(args.q4_model_dir, args.dtype, args.metal_swiglu)

    summaries = []
    for prompt in prompts:
        print(f"\n=== {prompt[:100]} ===")
        bf16 = run_greedy(bf16_model, tokenizer, prompt, max_tokens=args.max_tokens)
        q4 = run_greedy(q4_model, tokenizer, prompt, max_tokens=args.max_tokens)
        summary = summarize_prompt(prompt, bf16, q4)
        summaries.append(summary)

        print(f"BF16 tokens: {summary['bf16_tokens']}  EOS: {summary['bf16_stopped_on_eos']}")
        print(f"Q4 tokens:   {summary['q4_tokens']}  EOS: {summary['q4_stopped_on_eos']}")
        print(f"First divergence: {summary['first_divergence']}")
        if summary["first_divergence"] is not None:
            bf16_step = summary.get("bf16_at_divergence")
            q4_step = summary.get("q4_at_divergence")
            if bf16_step:
                print(
                    "BF16 chose:",
                    repr(bf16_step["selected_text"]),
                    "eos_rank:",
                    bf16_step["eos_rank"],
                    "top5:",
                    [(item["token"], round(item["score"], 3)) for item in bf16_step["top5"]],
                )
            if q4_step:
                print(
                    "Q4 chose:",
                    repr(q4_step["selected_text"]),
                    "eos_rank:",
                    q4_step["eos_rank"],
                    "top5:",
                    [(item["token"], round(item["score"], 3)) for item in q4_step["top5"]],
                )
        if "bf16_eos_step" in summary:
            print("BF16 EOS step:", summary["bf16_eos_step"])
            q4_same_step = summary.get("q4_at_bf16_eos_step")
            if q4_same_step:
                print(
                    "Q4 at BF16 EOS step chose:",
                    repr(q4_same_step["selected_text"]),
                    "eos_rank:",
                    q4_same_step["eos_rank"],
                    "top5:",
                    [(item["token"], round(item["score"], 3)) for item in q4_same_step["top5"]],
                )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
