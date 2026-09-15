"""Unit tests for the analytical KV cache calculator.

Every expected value below is computed by hand in the test itself from the four
structural terms, so a test failure points at the calculator rather than at a
remembered constant. Nothing here touches a GPU or reads a measurement.

The centrepiece is :class:`TestGQATrap`: it asserts not only that the calculator
is right, but that it is right *for the reason that matters*, by pinning the size
of the error the `num_attention_heads` mistake would have produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kv_cache_envelope.calculator import (
    UnsupportedArchitecture,
    block_padded_tokens,
    kv_bytes,
    kv_bytes_per_token,
    max_concurrent_sequences,
    max_tokens_in_budget,
    resolve_dtype_bytes,
    shape_from_config,
)
from kv_cache_envelope.hf_config import read_all_bundles

BUNDLES = Path(__file__).resolve().parents[1] / "configs" / "model_configs"

# Qwen2.5-7B-Instruct, verbatim from the provisioned snapshot
# a09a35458c702b33eeacc393d103063234e8bc28 (see provision.lock.json). Inlined so
# the core tests run even before the config bundles are fetched.
QWEN_7B = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "max_position_embeddings": 32768,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "sliding_window": 131072,
    "torch_dtype": "bfloat16",
    "use_sliding_window": False,
}


class TestShapeExtraction:
    def test_qwen7b_terms(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        assert s.num_hidden_layers == 28
        assert s.num_attention_heads == 28
        assert s.num_key_value_heads == 4
        assert s.head_dim == 3584 // 28 == 128
        assert s.head_dim_source == "derived"
        assert s.kv_heads_source == "config"
        assert s.attention_kind == "GQA"
        assert s.gqa_ratio == 7.0

    def test_explicit_head_dim_wins_over_derivation(self) -> None:
        """head_dim is not always hidden_size // num_attention_heads."""
        cfg = {**QWEN_7B, "head_dim": 96, "torch_dtype": "bfloat16"}
        s = shape_from_config(cfg, model_id="synthetic/explicit-head-dim")
        assert s.head_dim == 96
        assert s.head_dim_source == "config"

    def test_indivisible_hidden_size_without_head_dim_is_refused(self) -> None:
        cfg = {"num_hidden_layers": 4, "num_attention_heads": 7, "hidden_size": 100}
        with pytest.raises(UnsupportedArchitecture, match="not divisible"):
            shape_from_config(cfg, model_id="synthetic/indivisible")

    def test_absent_kv_heads_means_mha_and_is_recorded(self) -> None:
        cfg = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "torch_dtype": "float16",
        }
        s = shape_from_config(cfg, model_id="synthetic/mha")
        assert s.num_key_value_heads == 32
        assert s.kv_heads_source == "mha_fallback"
        assert s.attention_kind == "MHA"

    def test_falcon_7b_multi_query_is_one_kv_head_not_seventy_one(self) -> None:
        """The MHA fallback is a trap of its own, and Falcon is where it bites.

        falcon-7b omits `num_key_value_heads` entirely and states `multi_query:
        true` with 71 attention heads. Falling back to the query head count there
        overstates the cache by 71x. Fields verbatim from the fetched bundle at
        revision ec89142b67d748a1865ea4451372db8313ada0d8.
        """
        cfg = {
            "architectures": ["FalconForCausalLM"],
            "model_type": "falcon",
            "hidden_size": 4544,
            "num_attention_heads": 71,
            "num_hidden_layers": 32,
            "multi_query": True,
            "torch_dtype": "bfloat16",
        }
        s = shape_from_config(cfg, model_id="tiiuae/falcon-7b")
        assert s.num_key_value_heads == 1
        assert s.kv_heads_source == "falcon_multi_query"
        assert s.attention_kind == "MQA"
        assert s.head_dim == 4544 // 71 == 64
        assert kv_bytes_per_token(s) == 2 * 32 * 1 * 64 * 2
        naive_mha = 2 * 32 * 71 * 64 * 2
        assert naive_mha == 71 * kv_bytes_per_token(s)

    def test_falcon_new_decoder_architecture_uses_num_kv_heads(self) -> None:
        cfg = {
            "model_type": "falcon",
            "hidden_size": 8192,
            "num_attention_heads": 128,
            "num_hidden_layers": 60,
            "new_decoder_architecture": True,
            "num_kv_heads": 8,
            "multi_query": True,
            "torch_dtype": "bfloat16",
        }
        s = shape_from_config(cfg, model_id="synthetic/falcon-new-decoder")
        assert s.num_key_value_heads == 8
        assert s.kv_heads_source == "falcon_new_decoder_architecture"

    def test_generic_multi_query_flag_is_honoured(self) -> None:
        cfg = {
            "model_type": "somethingelse",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "multi_query": True,
            "torch_dtype": "bfloat16",
        }
        s = shape_from_config(cfg, model_id="synthetic/mq-flag")
        assert (s.num_key_value_heads, s.kv_heads_source) == (1, "multi_query_flag")

    def test_mqa_is_one_kv_head(self) -> None:
        cfg = {**QWEN_7B, "num_key_value_heads": 1}
        s = shape_from_config(cfg, model_id="synthetic/mqa")
        assert s.attention_kind == "MQA"
        assert s.gqa_ratio == 28.0

    def test_nested_text_config_is_descended(self) -> None:
        cfg = {
            "model_type": "multimodal",
            "text_config": {
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "hidden_size": 2048,
                "torch_dtype": "bfloat16",
            },
        }
        s = shape_from_config(cfg, model_id="synthetic/multimodal")
        assert (s.num_hidden_layers, s.num_key_value_heads, s.head_dim) == (16, 2, 128)


class TestUnsupportedArchitectures:
    """Refusing to answer beats answering wrongly."""

    def test_mla_is_refused(self) -> None:
        cfg = {
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "hidden_size": 2048,
            "kv_lora_rank": 512,
            "torch_dtype": "bfloat16",
        }
        with pytest.raises(UnsupportedArchitecture, match="latent"):
            shape_from_config(cfg, model_id="synthetic/mla")

    def test_ssm_hybrid_is_refused(self) -> None:
        cfg = {
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "hidden_size": 2048,
            "mamba_d_state": 128,
            "torch_dtype": "bfloat16",
        }
        with pytest.raises(UnsupportedArchitecture, match="state-space"):
            shape_from_config(cfg, model_id="synthetic/ssm")

    def test_missing_layer_count_is_refused(self) -> None:
        with pytest.raises(UnsupportedArchitecture):
            shape_from_config({"num_attention_heads": 8}, model_id="synthetic/empty")


class TestGQATrap:
    """The single mistake this project exists to get right.

    Only K and V are cached, and under GQA there are `num_key_value_heads` of
    them, not `num_attention_heads`. These tests pin the magnitude of the error so
    a regression cannot pass as a rounding difference.
    """

    def test_qwen7b_per_token_bytes_by_hand(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        # 2 (K,V) * 28 layers * 4 kv heads * 128 head_dim * 2 bytes (bf16)
        expected = 2 * 28 * 4 * 128 * 2
        assert expected == 57344  # 56 KiB per token, all layers
        assert kv_bytes_per_token(s, kv_cache_dtype="auto") == expected

    def test_using_query_heads_would_overstate_by_the_gqa_ratio(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        correct = kv_bytes_per_token(s, kv_cache_dtype="auto")
        wrong = 2 * s.num_hidden_layers * s.num_attention_heads * s.head_dim * 2
        assert wrong == correct * 7, "Qwen2.5-7B has 28 query heads to 4 KV heads"
        # At 8k context that is 3.06 GiB predicted against 21.4 GiB: an error large
        # enough to turn a fitting deployment into an impossible one.
        assert wrong - correct == 344064

    def test_mha_is_the_one_case_where_the_two_agree(self) -> None:
        cfg = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "hidden_size": 4096,
            "torch_dtype": "bfloat16",
        }
        s = shape_from_config(cfg, model_id="synthetic/mha")
        assert kv_bytes_per_token(s) == 2 * 32 * 32 * 128 * 2

    def test_mqa_cache_is_head_count_times_smaller(self) -> None:
        mha = shape_from_config(
            {**QWEN_7B, "num_key_value_heads": 28}, model_id="synthetic/mha28"
        )
        mqa = shape_from_config({**QWEN_7B, "num_key_value_heads": 1}, model_id="synthetic/mqa")
        assert kv_bytes_per_token(mha) == 28 * kv_bytes_per_token(mqa)


class TestScaling:
    def test_linear_in_seq_len_and_batch(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        per_token = kv_bytes_per_token(s)
        assert kv_bytes(s, seq_len=1024, batch=1) == per_token * 1024
        assert kv_bytes(s, seq_len=1024, batch=8) == per_token * 1024 * 8
        assert kv_bytes(s, seq_len=8192, batch=1) == 8 * kv_bytes(s, seq_len=1024, batch=1)

    def test_zero_length_and_zero_batch_are_zero(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        assert kv_bytes(s, seq_len=0, batch=4) == 0
        assert kv_bytes(s, seq_len=4096, batch=0) == 0

    def test_negative_inputs_are_rejected(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError):
            kv_bytes(s, seq_len=-1)


class TestDtype:
    def test_auto_resolves_against_the_model_dtype(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        assert resolve_dtype_bytes("auto", s) == 2

    def test_fp8_halves_the_cache(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        assert kv_bytes_per_token(s, kv_cache_dtype="fp8") * 2 == kv_bytes_per_token(s)

    def test_auto_without_a_stated_dtype_refuses_to_guess(self) -> None:
        cfg = {k: v for k, v in QWEN_7B.items() if k != "torch_dtype"}
        s = shape_from_config(cfg, model_id="synthetic/no-dtype")
        with pytest.raises(ValueError, match="no torch_dtype"):
            kv_bytes_per_token(s, kv_cache_dtype="auto")

    def test_unknown_dtype_is_rejected(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        with pytest.raises(ValueError, match="unknown KV cache dtype"):
            kv_bytes_per_token(s, kv_cache_dtype="int3")


class TestBlockGranularity:
    @pytest.mark.parametrize(
        ("seq_len", "block", "expected"),
        [(1, 16, 16), (16, 16, 16), (17, 16, 32), (1024, 16, 1024), (1025, 64, 1088), (0, 16, 0)],
    )
    def test_rounding(self, seq_len: int, block: int, expected: int) -> None:
        assert block_padded_tokens(seq_len, block) == expected

    def test_no_block_size_means_no_rounding(self) -> None:
        assert block_padded_tokens(1025, None) == 1025

    def test_waste_is_worst_at_large_blocks(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        one_token_b16 = kv_bytes(s, seq_len=1, block_size=16)
        one_token_b64 = kv_bytes(s, seq_len=1, block_size=64)
        assert one_token_b64 == 4 * one_token_b16


class TestSlidingWindow:
    def test_inactive_window_does_not_cap(self) -> None:
        """Qwen2.5 ships sliding_window with use_sliding_window: false."""
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        assert s.sliding_window is None
        assert kv_bytes(s, seq_len=32768) == kv_bytes_per_token(s) * 32768

    def test_active_window_caps_the_cached_span(self) -> None:
        cfg = {**QWEN_7B, "sliding_window": 4096, "use_sliding_window": True}
        s = shape_from_config(cfg, model_id="synthetic/window")
        assert s.sliding_window == 4096
        assert kv_bytes(s, seq_len=32768) == kv_bytes(s, seq_len=4096)


class TestCapacityPlanning:
    def test_token_budget_and_concurrency_are_consistent(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        budget = 20 * 1024**3  # 20 GiB of KV pool
        tokens = max_tokens_in_budget(s, budget)
        assert tokens == budget // 57344
        assert max_concurrent_sequences(s, budget, seq_len=8192) == tokens // 8192

    def test_concurrency_floors_rather_than_rounds(self) -> None:
        s = shape_from_config(QWEN_7B, model_id="Qwen/Qwen2.5-7B-Instruct")
        per_seq = kv_bytes(s, seq_len=1024)
        assert max_concurrent_sequences(s, per_seq * 3 - 1, seq_len=1024) == 2


class TestAgainstFetchedConfigs:
    """Validation against >=3 real model configs with differing GQA ratios.

    Skips when the bundles have not been fetched yet (they come from the Hub via
    a compute-node job; the login node has no network path). The bundles are
    committed, so this is a real check on a clean checkout.
    """

    @pytest.fixture(scope="class")
    def bundles(self) -> list:
        if not BUNDLES.exists() or not list(BUNDLES.glob("*.json")):
            pytest.skip("configs/model_configs not fetched yet")
        return [b for b in read_all_bundles(BUNDLES)]

    def test_at_least_three_distinct_gqa_ratios(self, bundles: list) -> None:
        ratios = {}
        for b in bundles:
            try:
                s = shape_from_config(b.config, model_id=b.model_id, revision=b.revision)
            except UnsupportedArchitecture:
                continue
            ratios[b.model_id] = s.gqa_ratio
        assert len(set(ratios.values())) >= 3, f"need >=3 distinct GQA ratios, got {ratios}"

    def test_every_bundle_computes_a_positive_per_token_size(self, bundles: list) -> None:
        for b in bundles:
            try:
                s = shape_from_config(b.config, model_id=b.model_id, revision=b.revision)
            except UnsupportedArchitecture:
                continue  # deliberately-included MLA/SSM bundles
            n = kv_bytes_per_token(s, kv_cache_dtype="auto")
            assert n > 0
            # The formula must reproduce from the recorded terms exactly.
            assert n == 2 * s.num_hidden_layers * s.num_key_value_heads * s.head_dim * 2

    def test_every_bundle_carries_a_revision(self, bundles: list) -> None:
        missing = [b.model_id for b in bundles if not b.revision]
        assert not missing, f"bundles without a recorded revision: {missing}"

    def test_mha_fallback_is_never_used_on_a_config_that_declares_mqa(
        self, bundles: list
    ) -> None:
        """Regression guard for the Falcon bug, checked against every real config."""
        for b in bundles:
            try:
                s = shape_from_config(b.config, model_id=b.model_id, revision=b.revision)
            except UnsupportedArchitecture:
                continue
            if s.kv_heads_source == "mha_fallback":
                assert not b.config.get("multi_query"), (
                    f"{b.model_id} declares multi_query but resolved via the MHA fallback"
                )
                assert "num_kv_heads" not in b.config

    def test_bundle_config_is_verbatim_json(self, bundles: list) -> None:
        for b in bundles:
            assert isinstance(b.config, dict)
            json.dumps(b.config)
