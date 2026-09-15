"""Analytical KV cache calculator: any HuggingFace config -> predicted bytes.

The whole model is one line:

    kv_bytes = 2 * num_hidden_layers * num_key_value_heads * head_dim
               * dtype_bytes * seq_len * batch

The factor 2 is K and V, not "two bytes". Everything interesting is in getting
the four structural terms out of a config that may not state them.

**`num_key_value_heads` is the trap.** Under grouped-query attention the query
projection has `num_attention_heads` heads but the key/value projections have
`num_key_value_heads`, and only K and V are cached. Using the query head count
overstates the cache by exactly the GQA ratio: 7x for Qwen2.5-7B (28 query heads
against 4 KV heads), 5x for Qwen2.5-32B, 1x for a true MHA model, and up to the
full head count for MQA. A calculator that is "only" wrong on GQA models is wrong
on essentially every model shipped since 2023, so :func:`kv_bytes` refuses to
fall back to `num_attention_heads` silently: the fallback happens only when the
config omits `num_key_value_heads` entirely, which is the documented HF spelling
of MHA, and it is recorded in :attr:`ModelKVShape.kv_heads_source`.

Three further details separate a correct answer from a plausible one:

* **`head_dim` is not always `hidden_size // num_attention_heads`.** Newer configs
  state it explicitly and some models (Llama-3.2-1B, Gemma) use a value that does
  not divide out. Prefer the explicit field; derive only as a fallback, and record
  which happened.
* **Sliding-window attention caps the cached span.** A window model holds
  `min(seq_len, window)` tokens per layer, not `seq_len`. Qwen2.5 carries a
  `sliding_window` key with `use_sliding_window: false`, so the key alone does not
  mean the window is active; both must be checked.
* **Compressed-latent attention (MLA, DeepSeek-V2/V3) does not obey this formula
  at all.** It caches a single low-rank latent per token instead of per-head K and
  V. Rather than return a confidently wrong number, :func:`shape_from_config`
  raises :class:`UnsupportedArchitecture`.

Nothing here touches a GPU; it is pure arithmetic over a config file, which is
exactly why it can be unit tested against hand-computed values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "DTYPE_BYTES",
    "ModelKVShape",
    "UnsupportedArchitecture",
    "block_padded_tokens",
    "kv_bytes",
    "kv_bytes_per_token",
    "max_concurrent_sequences",
    "max_tokens_in_budget",
    "shape_from_config",
]

# Bytes per stored KV element. vLLM's `--kv-cache-dtype` accepts the fp8
# spellings; `auto` means "same as the model's compute dtype", which is resolved
# against the config's `torch_dtype` rather than assumed to be fp16.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
}

# Config keys that mean the model does not cache per-head K and V at all.
_MLA_KEYS = ("kv_lora_rank", "q_lora_rank")
_SSM_KEYS = ("mamba_d_state", "ssm_cfg", "layers_block_type", "conv_kernel")


class UnsupportedArchitecture(ValueError):
    """The 2*L*H_kv*D formula does not describe this model's cache.

    Raised rather than returning a number. An architecture whose cache is
    compressed (MLA) or stateful (Mamba/SSM) would otherwise get an answer that
    looks fine and is wrong by an order of magnitude.
    """


@dataclass(frozen=True)
class ModelKVShape:
    """The four structural terms of the KV cache, plus how each was obtained.

    The provenance fields are not decoration. `head_dim` and `num_key_value_heads`
    are the two values most often derived rather than read, and a validation
    against a measured engine is only interpretable if it says which of the two
    happened.
    """

    model_id: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    model_type: str | None = None
    architecture: str | None = None
    hidden_size: int | None = None
    max_position_embeddings: int | None = None
    torch_dtype: str | None = None

    # Effective sliding window in tokens, or None for full causal attention.
    sliding_window: int | None = None

    head_dim_source: Literal["config", "derived"] = "config"
    kv_heads_source: Literal["config", "mha_fallback"] = "config"

    # Ground-truth identifiers, copied from the source, never typed.
    revision: str | None = None
    config_path: str | None = None

    @property
    def gqa_ratio(self) -> float:
        """Query heads per KV head. 1.0 is MHA; `num_attention_heads` is MQA."""
        return self.num_attention_heads / self.num_key_value_heads

    @property
    def attention_kind(self) -> Literal["MHA", "GQA", "MQA"]:
        if self.num_key_value_heads == 1:
            return "MQA"
        if self.num_key_value_heads == self.num_attention_heads:
            return "MHA"
        return "GQA"

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        d["gqa_ratio"] = self.gqa_ratio
        d["attention_kind"] = self.attention_kind
        return d


def _require_int(config: dict[str, Any], *names: str) -> int:
    for n in names:
        v = config.get(n)
        if isinstance(v, int) and v > 0:
            return v
    raise UnsupportedArchitecture(
        f"config states none of {names!r}; cannot determine KV cache shape"
    )


def shape_from_config(
    config: dict[str, Any],
    *,
    model_id: str,
    revision: str | None = None,
    config_path: str | None = None,
) -> ModelKVShape:
    """Extract the KV cache shape from a raw HuggingFace ``config.json`` dict.

    Raises :class:`UnsupportedArchitecture` when the model does not cache
    per-head K and V, rather than producing a number that does not mean anything.
    """
    # Some multimodal configs nest the language model. Descend before deciding
    # anything, since the top level of such a config has no attention fields.
    for nest in ("text_config", "llm_config", "language_config"):
        inner = config.get(nest)
        if isinstance(inner, dict) and "num_hidden_layers" in inner:
            merged = {**config, **inner}
            config = merged
            break

    present_mla = [k for k in _MLA_KEYS if config.get(k)]
    if present_mla:
        raise UnsupportedArchitecture(
            f"{model_id}: multi-head latent attention (keys {present_mla}); the cache is a "
            "compressed latent per token, not per-head K and V, so 2*L*H_kv*D does not apply"
        )
    present_ssm = [k for k in _SSM_KEYS if config.get(k) is not None]
    if present_ssm:
        raise UnsupportedArchitecture(
            f"{model_id}: state-space / hybrid architecture (keys {present_ssm}); its recurrent "
            "state is not a per-token KV cache"
        )

    layers = _require_int(config, "num_hidden_layers", "n_layer", "num_layers")
    heads = _require_int(config, "num_attention_heads", "n_head", "num_heads")

    kv_raw = config.get("num_key_value_heads", config.get("num_kv_heads"))
    if isinstance(kv_raw, int) and kv_raw > 0:
        kv_heads, kv_src = kv_raw, "config"
    else:
        # HF's documented meaning of an absent num_key_value_heads is MHA. This is
        # the only path on which the query head count is a legitimate answer.
        kv_heads, kv_src = heads, "mha_fallback"

    hidden = config.get("hidden_size", config.get("n_embd"))
    hd_raw = config.get("head_dim")
    if isinstance(hd_raw, int) and hd_raw > 0:
        head_dim, hd_src = hd_raw, "config"
    elif isinstance(hidden, int) and hidden > 0:
        if hidden % heads:
            raise UnsupportedArchitecture(
                f"{model_id}: hidden_size {hidden} is not divisible by num_attention_heads "
                f"{heads} and the config states no explicit head_dim"
            )
        head_dim, hd_src = hidden // heads, "derived"
    else:
        raise UnsupportedArchitecture(
            f"{model_id}: config states neither head_dim nor hidden_size"
        )

    # A `sliding_window` key is not by itself an active window. Qwen2.5 ships
    # `sliding_window: 131072` with `use_sliding_window: false`; treating the key
    # as authoritative would cap the predicted cache at a window the model does
    # not use.
    window = config.get("sliding_window")
    use_window = config.get("use_sliding_window")
    if use_window is False or not isinstance(window, int) or window <= 0:
        window = None

    archs = config.get("architectures")
    return ModelKVShape(
        model_id=model_id,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        model_type=config.get("model_type"),
        architecture=archs[0] if isinstance(archs, list) and archs else None,
        hidden_size=hidden if isinstance(hidden, int) else None,
        max_position_embeddings=config.get("max_position_embeddings"),
        torch_dtype=config.get("torch_dtype") or config.get("dtype"),
        sliding_window=window,
        head_dim_source=hd_src,
        kv_heads_source=kv_src,
        revision=revision,
        config_path=config_path,
    )


def resolve_dtype_bytes(kv_cache_dtype: str, shape: ModelKVShape) -> int:
    """Bytes per cached element for a vLLM ``--kv-cache-dtype`` spelling.

    ``auto`` is resolved against the model's own ``torch_dtype``: it means "cache
    in the compute dtype", which is bf16 for every model measured here but is not
    a safe constant to assume.
    """
    key = kv_cache_dtype.strip().lower()
    if key == "auto":
        if not shape.torch_dtype:
            raise ValueError(
                f"{shape.model_id}: kv_cache_dtype='auto' but the config states no torch_dtype; "
                "pass an explicit dtype rather than guessing"
            )
        key = str(shape.torch_dtype).strip().lower().removeprefix("torch.")
    if key not in DTYPE_BYTES:
        raise ValueError(f"unknown KV cache dtype {kv_cache_dtype!r} (resolved to {key!r})")
    return DTYPE_BYTES[key]


def block_padded_tokens(seq_len: int, block_size: int | None) -> int:
    """Tokens actually occupied once rounded up to PagedAttention block granularity.

    A sequence of 1 token in a 64-token block occupies the whole block. This is
    one of the named contributors to the predicted-vs-measured residual, so it is
    a first-class term rather than an afterthought.
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_size is None:
        return seq_len
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return math.ceil(seq_len / block_size) * block_size


def _cached_tokens(shape: ModelKVShape, seq_len: int, block_size: int | None) -> int:
    span = seq_len if shape.sliding_window is None else min(seq_len, shape.sliding_window)
    return block_padded_tokens(span, block_size)


def kv_bytes_per_token(
    shape: ModelKVShape, *, kv_cache_dtype: str = "auto", dtype_bytes: int | None = None
) -> int:
    """Bytes of KV cache one token of one sequence occupies, across all layers.

    This is the quantity capacity planning actually runs on: divide the KV pool
    by it to get the total token budget the engine can hold at once.
    """
    nbytes = dtype_bytes if dtype_bytes is not None else resolve_dtype_bytes(kv_cache_dtype, shape)
    return (
        2  # K and V
        * shape.num_hidden_layers
        * shape.num_key_value_heads  # NOT num_attention_heads
        * shape.head_dim
        * nbytes
    )


def kv_bytes(
    shape: ModelKVShape,
    *,
    seq_len: int,
    batch: int = 1,
    kv_cache_dtype: str = "auto",
    dtype_bytes: int | None = None,
    block_size: int | None = None,
) -> int:
    """Predicted KV cache bytes for ``batch`` sequences of ``seq_len`` tokens.

    ``block_size`` applies PagedAttention block-granularity rounding; leave it
    None for the unrounded analytical figure.
    """
    if seq_len < 0 or batch < 0:
        raise ValueError("seq_len and batch must be non-negative")
    per_token = kv_bytes_per_token(shape, kv_cache_dtype=kv_cache_dtype, dtype_bytes=dtype_bytes)
    return per_token * _cached_tokens(shape, seq_len, block_size) * batch


def max_tokens_in_budget(
    shape: ModelKVShape, budget_bytes: int, *, kv_cache_dtype: str = "auto"
) -> int:
    """How many cached tokens fit in a KV pool of ``budget_bytes``."""
    return budget_bytes // kv_bytes_per_token(shape, kv_cache_dtype=kv_cache_dtype)


def max_concurrent_sequences(
    shape: ModelKVShape,
    budget_bytes: int,
    *,
    seq_len: int,
    kv_cache_dtype: str = "auto",
    block_size: int | None = None,
) -> int:
    """The capacity-planning answer: sequences of ``seq_len`` servable at once.

    Floor, not round: a partially resident sequence is not a servable sequence,
    it is a preemption waiting to happen.
    """
    per_seq = kv_bytes(
        shape,
        seq_len=seq_len,
        batch=1,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
    )
    if per_seq <= 0:
        raise ValueError("per-sequence KV bytes resolved to zero")
    return budget_bytes // per_seq
