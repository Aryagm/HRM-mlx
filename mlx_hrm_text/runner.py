from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from mlx_hrm_text.generate import sample_next
from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu


DEFAULT_MODEL_REPO = "Aryagm/HRM-Text-1B-MLX-4bit"
PROMPT_PREFIX = "<|im_start|><|quad_end|><|object_ref_end|>"
PROMPT_SUFFIX = "<|im_end|>"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    finished: bool
    seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class StreamEvent:
    delta: str
    text: str
    token_id: int | None
    generated_tokens: int
    finished: bool
    seconds: float
    tokens_per_second: float


class HRMTextGenerator:
    """Programmatic HRM-mlx generator.

    If ``model_dir`` is omitted, the hosted 4-bit MLX checkpoint is downloaded
    from Hugging Face on first use and reused from the local cache after that.
    Plain prompts are wrapped in the prompt format used by HRM-Text.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        repo_id: str = DEFAULT_MODEL_REPO,
        dtype: str = "bfloat16",
        temperature: float = 0.0,
        static_cache: bool = False,
        metal_swiglu: bool = True,
        h_cycles: int | None = None,
        l_cycles: int | None = None,
        wrap_prompt: bool = True,
    ) -> None:
        if model_dir is None:
            model_dir = snapshot_download(repo_id=repo_id)

        self.model_dir = Path(model_dir)
        self.temperature = temperature
        self.wrap_prompt = wrap_prompt

        set_metal_swiglu(metal_swiglu)
        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True, trust_remote_code=True)
        finally:
            transformers_logging.set_verbosity(previous_verbosity)
        self.model = HrmTextForCausalLM.from_pretrained(self.model_dir, dtype=dtype)
        if h_cycles is not None:
            self.model.model.H_cycles = h_cycles
        if l_cycles is not None:
            self.model.model.L_cycles = l_cycles
        self.model.use_static_cache = static_cache
        mx.eval(self.model.parameters())

    def format_prompt(self, prompt: str) -> str:
        if not self.wrap_prompt or "<|im_start|>" in prompt:
            return prompt
        return f"{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}"

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float | None = None,
        eos_token_id: int | None = None,
    ) -> GenerationResult:
        text = ""
        token_ids: list[int] = []
        prompt_tokens = self._count_prompt_tokens(prompt)
        finished = False
        seconds = 0.0
        tokens_per_second = 0.0

        for event in self.stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        ):
            text = event.text
            seconds = event.seconds
            tokens_per_second = event.tokens_per_second
            if event.token_id is not None:
                token_ids.append(event.token_id)
            finished = event.finished

        return GenerationResult(
            text=text,
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(token_ids),
            finished=finished,
            seconds=seconds,
            tokens_per_second=tokens_per_second,
        )

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float | None = None,
        eos_token_id: int | None = None,
    ) -> Iterator[StreamEvent]:
        temperature = self.temperature if temperature is None else temperature
        eos_token_id = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        prompt_ids = self._encode_prompt(prompt)
        if not prompt_ids:
            raise ValueError("Prompt produced no tokens.")

        max_length = len(prompt_ids) + max_new_tokens if getattr(self.model, "use_static_cache", False) else None
        cache = self.model.make_cache(max_length=max_length)
        input_ids = mx.array(prompt_ids, dtype=mx.int32)

        start = perf_counter()
        logits = self.model.prefill(input_ids, cache)
        next_id = sample_next(logits[0], temperature)

        generated: list[int] = []
        text = ""
        position = len(prompt_ids)

        for _ in range(max_new_tokens):
            if eos_token_id is not None and next_id == eos_token_id:
                break

            generated.append(next_id)
            new_text = self.tokenizer.decode(generated, skip_special_tokens=False)
            delta = new_text[len(text) :]
            text = new_text
            elapsed = perf_counter() - start
            yield StreamEvent(
                delta=delta,
                text=text,
                token_id=next_id,
                generated_tokens=len(generated),
                finished=False,
                seconds=elapsed,
                tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
            )

            logits = self.model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
            next_id = sample_next(logits[0], temperature)
            position += 1

        elapsed = perf_counter() - start
        yield StreamEvent(
            delta="",
            text=text,
            token_id=None,
            generated_tokens=len(generated),
            finished=True,
            seconds=elapsed,
            tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
        )

    def _encode_prompt(self, prompt: str) -> list[int]:
        encoded = self.tokenizer(
            self.format_prompt(prompt),
            return_tensors="np",
            return_attention_mask=False,
            add_special_tokens=False,
        )
        return encoded["input_ids"][0].tolist()

    def _count_prompt_tokens(self, prompt: str) -> int:
        return len(self._encode_prompt(prompt))


HrmTextGenerator = HRMTextGenerator
