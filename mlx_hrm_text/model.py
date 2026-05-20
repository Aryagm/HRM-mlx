from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn


PROFILE: dict[str, list[float | int]] | None = None
PROFILE_DETAIL = False
USE_METAL_SWIGLU = False


def set_profile(profile: dict[str, list[float | int]] | None, *, detail: bool = False) -> None:
    global PROFILE, PROFILE_DETAIL
    PROFILE = profile
    PROFILE_DETAIL = detail


def set_metal_swiglu(enabled: bool) -> None:
    global USE_METAL_SWIGLU
    USE_METAL_SWIGLU = enabled


def _profile_eval(name: str, *arrays: mx.array, detail: bool = False) -> None:
    if PROFILE is None:
        return
    if detail and not PROFILE_DETAIL:
        return
    start = time.perf_counter()
    mx.eval(*arrays)
    elapsed = time.perf_counter() - start
    total, count = PROFILE.setdefault(name, [0.0, 0])
    PROFILE[name] = [float(total) + elapsed, int(count) + 1]


def _find_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _rotate_half(x: mx.array) -> mx.array:
    mid = x.shape[-1] // 2
    return mx.concatenate([-x[..., mid:], x[..., :mid]], axis=-1)


def _rms_norm(x: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _make_swiglu_kernel():
    if not mx.metal.is_available():
        return None
    source = """
        uint elem = thread_position_in_grid.x;
        float gate_v = static_cast<float>(gate[elem]);
        float up_v = static_cast<float>(up[elem]);
        float sig = 1.0f / (1.0f + metal::exp(-gate_v));
        out[elem] = static_cast<T>(gate_v * sig * up_v);
    """
    return mx.fast.metal_kernel(
        name="swiglu_activation",
        input_names=["gate", "up"],
        output_names=["out"],
        source=source,
    )


SWIGLU_KERNEL = _make_swiglu_kernel()


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    if USE_METAL_SWIGLU and SWIGLU_KERNEL is not None and mx.default_device() == mx.gpu:
        output = SWIGLU_KERNEL(
            inputs=[gate, up],
            template=[("T", gate.dtype)],
            grid=(gate.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[gate.shape],
            output_dtypes=[gate.dtype],
        )
        if isinstance(output, (list, tuple)):
            return output[0]
        return output
    return _silu(gate) * up


def _dtype_from_name(name: str | None):
    if name is None:
        return None
    mapping = {
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
        "float16": mx.float16,
        "fp16": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Use bfloat16, float16, or float32.") from exc


@dataclass
class HrmTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    H_cycles: int
    L_cycles: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    prefix_lm: bool = True
    embedding_scale: float = 1.0
    model_type: str = "hrm_text"
    num_key_value_heads: int | None = None
    num_layers_per_stack: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HrmTextConfig":
        hidden_size = int(data["hidden_size"])
        expansion = float(data.get("expansion", 4.0))
        intermediate_size = int(
            data.get(
                "intermediate_size",
                _find_multiple(round(expansion * hidden_size * 2 / 3), 256),
            )
        )
        num_hidden_layers = int(data.get("num_hidden_layers", data.get("n_layers")))
        num_heads = int(data.get("num_attention_heads", data.get("num_heads")))
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=int(data.get("num_key_value_heads", num_heads)),
            max_position_embeddings=int(data.get("max_position_embeddings", data.get("max_seq_len"))),
            H_cycles=int(data["H_cycles"]),
            L_cycles=int(data["L_cycles"]),
            rms_norm_eps=float(data.get("rms_norm_eps", data.get("norm_eps", 1e-6))),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            prefix_lm=bool(data.get("prefix_lm", True)),
            embedding_scale=float(data.get("embedding_scale", 1.0)),
            model_type=str(data.get("model_type", "hrm_text")),
            num_layers_per_stack=(
                int(data["num_layers_per_stack"]) if data.get("num_layers_per_stack") is not None else None
            ),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def recurrent_layers(self) -> int:
        return self.num_layers_per_stack or self.num_hidden_layers


class ScaledEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dims: int, scale: float):
        super().__init__()
        self.weight = mx.zeros((vocab_size, dims))
        self.scale = scale

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.scale * self.weight[input_ids]


class RotaryEmbedding:
    def __init__(self, dim: int, base: float):
        self.dim = dim
        self.base = base

    def __call__(self, position_ids: mx.array, dtype) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        freqs = mx.expand_dims(position_ids.astype(mx.float32), -1) * inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


class KVCache:
    def __init__(self, max_length: int | None = None):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.max_length = max_length
        self.offset = 0

    @property
    def length(self) -> int:
        return self.offset if self.max_length is not None else (0 if self.keys is None else self.keys.shape[2])

    def update(self, key: mx.array, value: mx.array) -> tuple[mx.array, mx.array]:
        if self.max_length is not None:
            if self.offset + key.shape[2] > self.max_length:
                raise ValueError(f"KV cache capacity exceeded: {self.offset + key.shape[2]} > {self.max_length}")
            if self.keys is None:
                shape = (key.shape[0], key.shape[1], self.max_length, key.shape[3])
                self.keys = mx.zeros(shape, dtype=key.dtype)
                self.values = mx.zeros(shape, dtype=value.dtype)
            self.keys = mx.slice_update(self.keys, key, start_indices=mx.array(self.offset), axes=(2,))
            self.values = mx.slice_update(self.values, value, start_indices=mx.array(self.offset), axes=(2,))
            self.offset += key.shape[2]
            assert self.values is not None
            return (
                mx.slice(self.keys, start_indices=mx.array(0), axes=(2,), slice_size=(key.shape[0], key.shape[1], self.offset, key.shape[3])),
                mx.slice(self.values, start_indices=mx.array(0), axes=(2,), slice_size=(value.shape[0], value.shape[1], self.offset, value.shape[3])),
            )

        if self.keys is None:
            self.keys = key
            self.values = value
        else:
            self.keys = mx.concatenate([self.keys, key], axis=2)
            self.values = mx.concatenate([self.values, value], axis=2)
        return self.keys, self.values


class Attention(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.prefix_lm = config.prefix_lm
        self.rope_theta = config.rope_theta

        qkv_heads = 2 * self.num_heads + 2 * self.num_key_value_heads
        self.gqkv_proj = nn.Linear(config.hidden_size, qkv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        position_offset: int | mx.array | None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch, seq_len, _ = hidden_states.shape
        gqkv = self.gqkv_proj(hidden_states)
        _profile_eval("attn.qkv_proj", gqkv, detail=True)
        gqkv = mx.reshape(
            gqkv,
            (batch, seq_len, 2 * self.num_heads + 2 * self.num_key_value_heads, self.head_dim),
        )

        h = self.num_heads
        kvh = self.num_key_value_heads
        gate = gqkv[:, :, :h, :]
        query = gqkv[:, :, h : 2 * h, :]
        key = gqkv[:, :, 2 * h : 2 * h + kvh, :]
        value = gqkv[:, :, 2 * h + kvh :, :]

        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        if position_offset is not None:
            query = mx.fast.rope(
                query,
                self.head_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=position_offset,
            )
            key = mx.fast.rope(
                key,
                self.head_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=position_offset,
            )
            _profile_eval("attn.rope", query, key, detail=True)
        if cache is not None:
            value = mx.transpose(value, (0, 2, 1, 3))
            key, value = cache.update(key, value)
            _profile_eval("attn.cache_update", key, value)
            mask = None
        else:
            value = mx.transpose(value, (0, 2, 1, 3))
            mask = None if self.prefix_lm else "causal"

        attn = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale, mask=mask)
        _profile_eval("attn.sdpa", attn, detail=True)
        attn = mx.transpose(attn, (0, 2, 1, 3))
        attn = mx.sigmoid(gate) * attn
        _profile_eval("attn.gate", attn, detail=True)
        out = self.o_proj(mx.reshape(attn, (batch, seq_len, self.num_heads * self.head_dim)))
        _profile_eval("attn.o_proj", out, detail=True)
        return out


class SwiGLU(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        _profile_eval("mlp.gate_up_proj", gate_up, detail=True)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = _swiglu(gate, up)
        _profile_eval("mlp.swiglu", hidden, detail=True)
        out = self.down_proj(hidden)
        _profile_eval("mlp.down_proj", out, detail=True)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = SwiGLU(config)
        self.norm_eps = config.rms_norm_eps

    def __call__(self, x: mx.array, position_offset: int | mx.array | None, cache: KVCache | None = None) -> mx.array:
        attn_input = _rms_norm(x, self.norm_eps)
        _profile_eval("norm.attn", attn_input)
        attn_out = self.attn(attn_input, position_offset=position_offset, cache=cache)
        _profile_eval("block.attn", attn_out)
        x = x + attn_out
        mlp_input = _rms_norm(x, self.norm_eps)
        _profile_eval("norm.mlp", mlp_input)
        mlp_out = self.mlp(mlp_input)
        _profile_eval("block.mlp", mlp_out)
        return x + mlp_out


class TransformerModule(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.layers = [TransformerBlock(config) for _ in range(config.recurrent_layers)]
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.norm_eps = config.rms_norm_eps

    def make_cache(self, max_length: int | None = None) -> list[KVCache]:
        return [KVCache(max_length=max_length) for _ in self.layers]

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,
        input_injection: mx.array,
        position_offset: int | mx.array | None = None,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        x = x + input_injection
        for idx, layer in enumerate(self.layers):
            x = layer(x, position_offset=position_offset, cache=None if cache is None else cache[idx])
        x = _rms_norm(x, self.norm_eps)
        _profile_eval("norm.final", x)
        return x


class HrmTextBackbone(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.embed_tokens = ScaledEmbedding(config.vocab_size, config.hidden_size, config.embedding_scale)
        self.H_module = TransformerModule(config)
        self.L_module = TransformerModule(config)
        self.z_L_init = mx.zeros((config.hidden_size,))
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

    def make_cache(self, max_length: int | None = None) -> dict[str, list[list[KVCache]]]:
        return {
            "H": [self.H_module.make_cache(max_length=max_length) for _ in range(self.H_cycles)],
            "L": [self.L_module.make_cache(max_length=max_length) for _ in range(self.H_cycles * self.L_cycles)],
        }

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: dict[str, list[list[KVCache]]] | None = None,
    ) -> mx.array:
        z_h = self.embed_tokens(input_ids)
        z_l = mx.broadcast_to(self.z_L_init.astype(z_h.dtype), z_h.shape)
        position_offset = position_ids[:, 0] if position_ids.ndim == 2 else position_ids[0]

        for h_idx in range(self.H_cycles):
            for l_idx in range(h_idx * self.L_cycles, (h_idx + 1) * self.L_cycles):
                z_l = self.L_module(
                    z_l,
                    position_ids=position_ids,
                    input_injection=z_h,
                    position_offset=position_offset,
                    cache=None if cache is None else cache["L"][l_idx],
                )
            z_h = self.H_module(
                z_h,
                position_ids=position_ids,
                input_injection=z_l,
                position_offset=position_offset,
                cache=None if cache is None else cache["H"][h_idx],
            )
        return z_h


class HrmTextForCausalLM(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.config = config
        self.model = HrmTextBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: str | None = None,
        strict: bool = True,
    ) -> "HrmTextForCausalLM":
        import json
        import mlx.nn as nn

        model_dir = Path(model_dir)
        config = HrmTextConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
        model = cls(config)
        quantization_path = model_dir / "quantization.json"
        if quantization_path.exists():
            quantization = json.loads(quantization_path.read_text())
            nn.quantize(
                model,
                bits=int(quantization["bits"]),
                group_size=int(quantization["group_size"]),
                mode=quantization.get("mode", "affine"),
            )
        weights = model_dir / "model.safetensors"
        if not weights.exists():
            weights = model_dir / "weights.safetensors"
        model.load_weights(str(weights), strict=strict)
        target_dtype = _dtype_from_name(dtype)
        if target_dtype is not None:
            model.set_dtype(target_dtype)
        return model

    def make_cache(self, max_length: int | None = None) -> dict[str, list[list[KVCache]]]:
        return self.model.make_cache(max_length=max_length)

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array | None = None,
        cache: dict[str, list[list[KVCache]]] | None = None,
    ) -> mx.array:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if position_ids is None:
            position_ids = mx.arange(input_ids.shape[1])[None, :]
        elif position_ids.ndim == 1:
            position_ids = position_ids[None, :]

        hidden = self.model(input_ids, position_ids=position_ids, cache=cache)
        logits = self.lm_head(hidden)
        _profile_eval("lm_head", logits)
        return logits

    def prefill(self, input_ids: mx.array, cache: dict[str, list[list[KVCache]]] | None = None) -> mx.array:
        if cache is None:
            cache = self.make_cache()
        logits = self(input_ids[None, :] if input_ids.ndim == 1 else input_ids, cache=cache)
        return logits[:, -1, :]

    def decode_one(self, input_ids: mx.array, position: int, cache: dict[str, list[list[KVCache]]]) -> mx.array:
        if input_ids.ndim == 0:
            input_ids = input_ids[None, None]
        elif input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        position_ids = mx.full(input_ids.shape, position, dtype=mx.int32)
        logits = self(input_ids, position_ids=position_ids, cache=cache)
        return logits[:, -1, :]
