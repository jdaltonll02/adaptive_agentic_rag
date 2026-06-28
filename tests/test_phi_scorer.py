"""Unit tests for condor/phi_scorer.py"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.phi_scorer import PhiScorer


@pytest.fixture
def scorer():
    return PhiScorer(t_warmup=100, tau_trans=50.0, lr=1e-3)


@pytest.fixture
def sample_inputs():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((32,)).astype(np.float32)
    return emb, 1, 0  # emb, mechanism_id, qtype_id


# ── Initialisation ────────────────────────────────────────────────────────────

def test_scorer_is_nn_module(scorer):
    assert isinstance(scorer, torch.nn.Module)

def test_step_starts_at_zero(scorer):
    assert scorer._t == 0


# ── get_blend_alpha ───────────────────────────────────────────────────────────

def test_blend_alpha_before_warmup(scorer):
    scorer._t = 0
    alpha = scorer.get_blend_alpha()
    # sigmoid((0 - 100) / 50) ≈ sigmoid(-2) ≈ 0.119
    assert 0.0 < alpha < 0.2

def test_blend_alpha_at_warmup(scorer):
    scorer._t = 100
    alpha = scorer.get_blend_alpha()
    # sigmoid(0) = 0.5
    assert abs(alpha - 0.5) < 1e-4

def test_blend_alpha_after_warmup(scorer):
    scorer._t = 300
    alpha = scorer.get_blend_alpha()
    assert alpha > 0.9

def test_blend_alpha_in_unit_interval(scorer):
    for t in [0, 50, 100, 200, 500]:
        scorer._t = t
        alpha = scorer.get_blend_alpha()
        assert 0.0 < alpha < 1.0


# ── score ─────────────────────────────────────────────────────────────────────

def test_score_returns_float(scorer, sample_inputs):
    emb, m, qt = sample_inputs
    val = scorer.score(emb, m, qt)
    assert isinstance(val, float)

def test_score_all_mechanisms(scorer):
    emb = np.random.randn(32).astype(np.float32)
    for m in range(5):
        for qt in range(4):
            val = scorer.score(emb, m, qt)
            assert isinstance(val, float)
            assert np.isfinite(val)


# ── blended_baseline ─────────────────────────────────────────────────────────

def test_blended_baseline_pure_phi(scorer):
    scorer.t = 0  # alpha ≈ 0 → heavily phi
    alpha = scorer.get_blend_alpha()
    result = scorer.blended_baseline(phi_val=1.0, critic_val=0.0)
    expected = (1 - alpha) * 1.0 + alpha * 0.0
    assert abs(result - expected) < 1e-5

def test_blended_baseline_pure_critic(scorer):
    scorer.t = 500  # alpha ≈ 1 → heavily critic
    alpha = scorer.get_blend_alpha()
    result = scorer.blended_baseline(phi_val=0.0, critic_val=1.0)
    expected = (1 - alpha) * 0.0 + alpha * 1.0
    assert abs(result - expected) < 0.01


# ── update ────────────────────────────────────────────────────────────────────

def test_update_returns_float(scorer):
    rng = np.random.default_rng(0)
    embs = [rng.standard_normal(32).astype(np.float32) for _ in range(4)]
    m_ids = [0, 1, 2, 3]
    qt_ids = [0, 1, 0, 1]
    targets = [0.5, 0.6, 0.3, 0.8]
    loss = scorer.update(embs, m_ids, qt_ids, targets)
    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert loss >= 0.0

def test_update_step_increments(scorer):
    rng = np.random.default_rng(1)
    embs = [rng.standard_normal(32).astype(np.float32) for _ in range(2)]
    scorer.update(embs, [0, 1], [0, 1], [0.5, 0.5])
    assert scorer._t == 1

def test_update_reduces_loss_over_time():
    s = PhiScorer(t_warmup=100, tau_trans=50.0, lr=1e-2)
    rng = np.random.default_rng(2)
    embs = [rng.standard_normal(32).astype(np.float32) for _ in range(8)]
    m_ids = [i % 5 for i in range(8)]
    qt_ids = [i % 4 for i in range(8)]
    targets = [0.7] * 8  # fixed targets → should converge

    first_loss = s.update(embs, m_ids, qt_ids, targets)
    for _ in range(200):
        s.update(embs, m_ids, qt_ids, targets)
    last_loss = s.update(embs, m_ids, qt_ids, targets)
    assert last_loss < first_loss, "Loss should decrease with many update steps"
