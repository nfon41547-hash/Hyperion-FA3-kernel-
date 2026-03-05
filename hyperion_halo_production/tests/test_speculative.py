"""
test_speculative.py – Unit tests for the speculative decoding pipeline.

All tests use stub (random-logit) models so they run on CPU without real
weights or a GPU.
"""

import pytest
import torch

from hyperion_halo_production.speculative.draft_model import DraftModel, DraftModelConfig
from hyperion_halo_production.speculative.target_model import TargetModel, TargetModelConfig
from hyperion_halo_production.speculative.verifier import Verifier, VerifierConfig
from hyperion_halo_production.speculative.speculative_decoder import (
    SpeculativeDecoder,
    SpeculativeDecoderConfig,
)


# ============================================================
# Shared fixtures
# ============================================================
@pytest.fixture
def tiny_draft():
    cfg = DraftModelConfig(vocab_size=256, draft_steps=3, device="cpu")
    return DraftModel(config=cfg)


@pytest.fixture
def tiny_target():
    cfg = TargetModelConfig(vocab_size=256, device="cpu")
    return TargetModel(config=cfg)


@pytest.fixture
def verifier():
    return Verifier()


# ============================================================
# DraftModel
# ============================================================
class TestDraftModel:
    def test_output_shape(self, tiny_draft):
        ids = torch.randint(0, 256, (1, 8))
        draft_ids, draft_probs = tiny_draft.draft_tokens(ids, num_draft_steps=3)
        assert draft_ids.shape == (1, 3)
        assert draft_probs.shape == (1, 3, 256)

    def test_probs_sum_to_one(self, tiny_draft):
        ids = torch.randint(0, 256, (2, 8))
        _, probs = tiny_draft.draft_tokens(ids, num_draft_steps=4)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_batch_size_preserved(self, tiny_draft):
        B = 4
        ids = torch.randint(0, 256, (B, 10))
        draft_ids, _ = tiny_draft.draft_tokens(ids, num_draft_steps=2)
        assert draft_ids.shape[0] == B

    def test_stats_increments(self, tiny_draft):
        ids = torch.randint(0, 256, (1, 5))
        tiny_draft.draft_tokens(ids)
        tiny_draft.draft_tokens(ids)
        assert tiny_draft.stats()["call_count"] == 2

    def test_top_k_filter(self):
        probs = torch.tensor([[0.1, 0.4, 0.3, 0.2]])
        filtered = DraftModel._top_k_filter(probs, k=2)
        # Bottom 2 should be zeroed out
        assert filtered[0, 0] == 0.0 or filtered[0, 3] == 0.0

    def test_top_p_filter(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        filtered = DraftModel._top_p_filter(probs, p=0.8)
        # Cumulative up to 0.8 covers top 2 values
        assert filtered[0, 2] == 0.0 or filtered[0, 3] == 0.0


# ============================================================
# TargetModel
# ============================================================
class TestTargetModel:
    def test_output_shape(self, tiny_target):
        cfg = TargetModelConfig(vocab_size=256, device="cpu")
        model = TargetModel(config=cfg)
        ids = torch.randint(0, 256, (1, 8))
        draft = torch.randint(0, 256, (1, 3))
        probs, logits = model.score_tokens(ids, draft)
        assert probs.shape == (1, 4, 256)   # 3 draft + 1 correction
        assert logits.shape == (1, 4, 256)

    def test_probs_sum_to_one(self, tiny_target):
        ids = torch.randint(0, 256, (2, 6))
        draft = torch.randint(0, 256, (2, 4))
        probs, _ = tiny_target.score_tokens(ids, draft)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_stats(self, tiny_target):
        ids = torch.randint(0, 256, (1, 8))
        draft = torch.randint(0, 256, (1, 2))
        tiny_target.score_tokens(ids, draft)
        assert tiny_target.stats()["call_count"] == 1


# ============================================================
# Verifier
# ============================================================
class TestVerifier:
    def _make_uniform_probs(self, B, S, V):
        return torch.ones(B, S, V) / V

    def test_shapes(self, verifier):
        B, S, V = 2, 4, 256
        draft_ids = torch.randint(0, V, (B, S))
        draft_probs = self._make_uniform_probs(B, S, V)
        target_probs = self._make_uniform_probs(B, S + 1, V)
        acc_ids, corr, num_acc = verifier.verify(draft_ids, draft_probs, target_probs)
        assert acc_ids.shape == (B, S)
        assert corr.shape == (B,)
        assert num_acc.shape == (B,)

    def test_num_accepted_in_range(self, verifier):
        B, S, V = 1, 4, 256
        draft_ids = torch.randint(0, V, (B, S))
        draft_probs = self._make_uniform_probs(B, S, V)
        target_probs = self._make_uniform_probs(B, S + 1, V)
        _, _, num_acc = verifier.verify(draft_ids, draft_probs, target_probs)
        assert 0 <= int(num_acc[0]) <= S

    def test_high_acceptance_when_probs_match(self):
        """When draft == target distribution, acceptance rate → 100%."""
        B, S, V = 1, 8, 64
        ver = Verifier()
        # Run many trials and check aggregate acceptance is high
        total_acc = 0
        total_draft = 0
        for _ in range(50):
            draft_ids = torch.randint(0, V, (B, S))
            probs = torch.ones(B, S, V) / V
            target_probs_ext = torch.ones(B, S + 1, V) / V
            _, _, num_acc = ver.verify(draft_ids, probs, target_probs_ext)
            total_acc += int(num_acc.sum())
            total_draft += B * S
        rate = total_acc / total_draft
        # With uniform distributions acceptance should be ~100%
        assert rate > 0.90, f"acceptance rate too low: {rate:.2%}"

    def test_acceptance_rate_metric(self, verifier):
        B, S, V = 1, 4, 32
        for _ in range(10):
            draft_ids = torch.randint(0, V, (B, S))
            probs = torch.ones(B, S, V) / V
            tp = torch.ones(B, S + 1, V) / V
            verifier.verify(draft_ids, probs, tp)
        assert 0.0 <= verifier.acceptance_rate() <= 1.0

    def test_stats_keys(self, verifier):
        s = verifier.stats()
        assert "total_drafted" in s
        assert "total_accepted" in s
        assert "acceptance_rate" in s


# ============================================================
# SpeculativeDecoder (single-batch, stub models)
# ============================================================
class TestSpeculativeDecoder:
    def test_generate_output_longer_than_input(self):
        V = 256
        cfg = SpeculativeDecoderConfig(draft_steps=2, max_new_tokens=6)
        dm = DraftModel(DraftModelConfig(vocab_size=V, device="cpu"))
        tm = TargetModel(TargetModelConfig(vocab_size=V, device="cpu"))
        decoder = SpeculativeDecoder(draft_model=dm, target_model=tm, config=cfg)
        ids = torch.randint(3, V, (1, 5))
        out = decoder.generate(ids, max_new_tokens=6)
        assert out.shape[1] >= ids.shape[1]

    def test_stats_keys(self):
        decoder = SpeculativeDecoder()
        s = decoder.stats()
        for key in ("decode_calls", "total_tokens_generated", "tokens_per_sec",
                    "draft_stats", "target_stats", "verifier_stats"):
            assert key in s

    def test_eos_stops_generation(self):
        """If the correction token is always EOS (id=2), generation stops early."""
        eos = 2

        def eos_forward(input_ids):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, 256)
            logits[:, :, eos] = 100.0   # push all probability to EOS
            return logits

        dm = DraftModel(
            config=DraftModelConfig(vocab_size=256, draft_steps=2, device="cpu"),
            forward_fn=eos_forward,
        )
        tm = TargetModel(
            config=TargetModelConfig(vocab_size=256, device="cpu"),
            forward_fn=eos_forward,
        )
        cfg = SpeculativeDecoderConfig(
            draft_steps=2, max_new_tokens=20, eos_token_id=eos
        )
        decoder = SpeculativeDecoder(draft_model=dm, target_model=tm, config=cfg)
        ids = torch.randint(3, 256, (1, 5))
        out = decoder.generate(ids)
        # Should stop well before 20 new tokens
        assert out.shape[1] < ids.shape[1] + 20
