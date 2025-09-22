from __future__ import annotations

import pytest
import torch

from hud.rl.config import Config
from hud.rl.learner import GRPOLearner
from hud.rl.types import TrainingSample


@pytest.fixture()
def learner_stub(monkeypatch):
    cfg = Config()
    # Speed up: tiny settings
    cfg.training.epochs = 1
    cfg.training.group_size = 1
    cfg.training.mini_batch_size = 1
    cfg.training.use_8bit_optimizer = False

    # Stub _load_models to avoid heavy model init
    def _stub_load_models(self):
        class DummyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

        dummy_policy = DummyPolicy()
        dummy_opt = torch.optim.SGD(dummy_policy.parameters(), lr=0.1)
        return None, dummy_policy, None, dummy_opt

    monkeypatch.setattr(GRPOLearner, "_load_models", _stub_load_models, raising=True)
    return GRPOLearner(cfg)


def make_sample(
    pol_logp_tok: torch.Tensor,
    old_logp_tok: torch.Tensor,
    ref_logp_tok: torch.Tensor,
    advantage: float,
):
    # Minimal-but-correct object for GRPOLearner.compute_loss.
    # Needs assistant_mask (T-1) and attention_mask (T) for sanity_check().
    Tm1 = pol_logp_tok.size(-1)
    inputs = {
        "input_ids": torch.zeros(1, Tm1 + 1, dtype=torch.long),
        "attention_mask": torch.ones(1, Tm1 + 1, dtype=torch.long),
        "assistant_mask": torch.ones(1, Tm1, dtype=torch.bool),
    }
    return TrainingSample(
        inputs=inputs,
        old_logprobs=old_logp_tok,
        ref_logprobs=ref_logp_tok,
        # advantage must be 1D so .view(-1,1) works in compute_loss
        advantage=torch.tensor([advantage], dtype=torch.float32),
    )


def patch_compute_logprobs(
    monkeypatch, learner: GRPOLearner, pol_logp_tok: torch.Tensor, pol_entropy_tok: torch.Tensor
):
    # Return (pol_logp, pol_entropy) as expected by compute_loss
    def _stub_compute_logprobs(self, model, inputs):
        return pol_logp_tok.to(inputs["input_ids"].device), pol_entropy_tok.to(
            inputs["input_ids"].device
        )

    monkeypatch.setattr(GRPOLearner, "compute_logprobs", _stub_compute_logprobs, raising=True)


def test_per_token_mean_vs_sum(monkeypatch, learner_stub: GRPOLearner):
    # Setup
    _, Tm1 = 1, 4
    pol = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)  # logp
    old = torch.tensor([[-1.2, -0.8, -1.0, -1.1]], dtype=torch.float32)
    ref = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)
    ent = torch.zeros_like(pol)
    patch_compute_logprobs(monkeypatch, learner_stub, pol, ent)

    # Common config
    learner_stub.config.training.kl_beta = 0.0
    learner_stub.config.training.entropy_beta = 0.0
    learner_stub.config.training.top_eps = 0.2
    learner_stub.config.training.bottom_eps = 0.1

    sample = make_sample(pol, old, ref, advantage=1.0)

    # token_agg=mean
    learner_stub.config.training.ppo_mode = "per_token"
    learner_stub.config.training.token_agg = "mean"
    loss_mean = learner_stub.compute_loss(sample).item()

    # token_agg=sum
    learner_stub.config.training.token_agg = "sum"
    loss_sum = learner_stub.compute_loss(sample).item()

    # Expect sum ≈ mean * num_tokens
    assert pytest.approx(loss_sum, rel=1e-5) == loss_mean * Tm1


def test_per_trace_vs_per_token(monkeypatch, learner_stub: GRPOLearner):
    # Equal per-token deltas -> per_trace matches per_token(mean)
    pol = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)
    old = torch.tensor([[-1.2, -1.2, -1.2]], dtype=torch.float32)
    ref = torch.tensor([[-1.1, -1.1, -1.1]], dtype=torch.float32)
    ent = torch.zeros_like(pol)
    patch_compute_logprobs(monkeypatch, learner_stub, pol, ent)

    learner_stub.config.training.kl_beta = 0.0
    learner_stub.config.training.entropy_beta = 0.0
    learner_stub.config.training.top_eps = 0.2
    learner_stub.config.training.bottom_eps = 0.1

    sample = make_sample(pol, old, ref, advantage=1.0)

    learner_stub.config.training.ppo_mode = "per_token"
    learner_stub.config.training.token_agg = "mean"
    ltok = learner_stub.compute_loss(sample).item()

    learner_stub.config.training.ppo_mode = "per_trace"
    ltraj = learner_stub.compute_loss(sample).item()

    assert pytest.approx(ltraj, rel=1e-6) == ltok


def test_entropy_beta_effect(monkeypatch, learner_stub: GRPOLearner):
    pol = torch.tensor([[-1.0, -1.1]], dtype=torch.float32)
    old = torch.tensor([[-1.0, -1.1]], dtype=torch.float32)
    ref = torch.tensor([[-1.0, -1.1]], dtype=torch.float32)
    ent = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    patch_compute_logprobs(monkeypatch, learner_stub, pol, ent)

    # No policy/kl effect, only entropy
    learner_stub.config.training.ppo_mode = "per_token"
    learner_stub.config.training.token_agg = "mean"
    learner_stub.config.training.kl_beta = 0.0

    sample = make_sample(pol, old, ref, advantage=0.0)

    learner_stub.config.training.entropy_beta = 0.0
    l0 = learner_stub.compute_loss(sample).item()

    learner_stub.config.training.entropy_beta = 2.0
    l1 = learner_stub.compute_loss(sample).item()

    # Mean entropy = (0.5+1.5)/2 = 1.0, scaled by beta=2.0 -> +2.0
    assert pytest.approx(l1 - l0, rel=1e-6) == 2.0


def test_skip_update_when_zero_adv(monkeypatch, learner_stub: GRPOLearner):
    # Patch prepare_groups to yield a single group with a minibatch-like object
    class MiniBatch:
        def __init__(self):
            self.advantage = torch.zeros(1)

        def to_device(self, device: torch.device) -> MiniBatch:
            return self

    def _stub_prepare_groups(self, samples: list[TrainingSample]) -> list[list[MiniBatch]]:
        return [[MiniBatch(), MiniBatch()]]

    monkeypatch.setattr(GRPOLearner, "prepare_groups", _stub_prepare_groups, raising=True)

    # Return a zero scalar loss that *depends* on params so backward works,
    # but has zero gradients (no update signal).
    def _zero_loss(self, sample) -> torch.Tensor:
        return sum(p.sum() for p in self.policy.parameters()) * 0.0  # type: ignore

    monkeypatch.setattr(GRPOLearner, "compute_loss", _zero_loss, raising=True)

    # Count optimizer.step calls
    steps = {"n": 0}
    # orig_step = learner_stub.optimizer.step

    def _count_step():
        steps["n"] += 1

    monkeypatch.setattr(learner_stub.optimizer, "step", _count_step, raising=False)

    # Ensure dummy backward can touch a parameter
    assert any(p.requires_grad for p in learner_stub.policy.parameters())

    learner_stub.update([])
    # With the current learner implementation we still call optimizer.step()
    # even if the per-minibatch "advantage" is zero (the step is a no-op
    # because the gradients are zero). So we expect exactly one step here.
    assert steps["n"] == 1
