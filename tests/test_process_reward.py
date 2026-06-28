"""Unit tests for condor/process_reward.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.process_reward import ProcessRewardModel
from condor.mechanism_router import MECHANISM_SELECTION_COST


@pytest.fixture
def prm():
    return ProcessRewardModel()


# ── compute_l0_reward ─────────────────────────────────────────────────────────

def test_l0_reward_returns_float(prm):
    r = prm.compute_l0_reward(f1=0.7, m_star=1, lambda_hi=0.1, baseline_f1=0.5)
    assert isinstance(r, float)

def test_l0_reward_positive_for_good_f1(prm):
    # f1 > baseline_f1 and small mechanism cost → reward should be positive
    r = prm.compute_l0_reward(f1=0.8, m_star=0, lambda_hi=0.0, baseline_f1=0.5)
    assert r > 0.0

def test_l0_reward_negative_for_bad_f1(prm):
    # f1 < baseline and high lambda penalty → reward should be negative
    r = prm.compute_l0_reward(f1=0.1, m_star=4, lambda_hi=5.0, baseline_f1=0.6)
    assert r < 0.0

def test_l0_reward_mechanism_cost_subtracted(prm):
    # Same f1, different mechanism costs should change reward
    r_cheap = prm.compute_l0_reward(f1=0.6, m_star=0, lambda_hi=1.0, baseline_f1=0.5)
    r_expensive = prm.compute_l0_reward(f1=0.6, m_star=4, lambda_hi=1.0, baseline_f1=0.5)
    assert r_cheap > r_expensive, "Cheaper mechanism should yield higher reward"

def test_l0_reward_formula(prm):
    f1, m, lhi, b = 0.6, 1, 0.2, 0.5
    expected = (f1 - b) - lhi * MECHANISM_SELECTION_COST[m]
    actual = prm.compute_l0_reward(f1=f1, m_star=m, lambda_hi=lhi, baseline_f1=b)
    assert abs(actual - expected) < 1e-6


# ── compute_l1_cost ───────────────────────────────────────────────────────────

def test_l1_cost_returns_float(prm):
    cost = prm.compute_l1_cost(token_cost=0.01, retrieval_api_count=2, turn_latency=3)
    assert isinstance(cost, float)

def test_l1_cost_negative(prm):
    # All penalty terms → cost should be negative
    cost = prm.compute_l1_cost(token_cost=1.0, retrieval_api_count=3, turn_latency=5)
    assert cost < 0.0

def test_l1_cost_zero_inputs(prm):
    cost = prm.compute_l1_cost(token_cost=0.0, retrieval_api_count=0, turn_latency=0)
    assert cost == 0.0

def test_l1_cost_formula(prm):
    # Coefficients: -1.0 * tc + -0.25 * rac + -0.5 * tl
    expected = -1.0 * 0.5 + -0.25 * 2 + -0.5 * 3
    actual = prm.compute_l1_cost(token_cost=0.5, retrieval_api_count=2, turn_latency=3)
    assert abs(actual - expected) < 1e-6


# ── scale_l1_cost ─────────────────────────────────────────────────────────────

def test_scale_l1_cost_with_lambda_zero(prm):
    assert prm.scale_l1_cost(-1.0, lambda_lo=0.0) == 0.0

def test_scale_l1_cost_proportional(prm):
    c1 = prm.scale_l1_cost(-1.0, lambda_lo=1.0)
    c2 = prm.scale_l1_cost(-1.0, lambda_lo=2.0)
    assert abs(c2 - 2 * c1) < 1e-9


# ── l0_correct ────────────────────────────────────────────────────────────────

def test_l0_correct_above_threshold(prm):
    assert prm.l0_correct(f1=0.5, threshold=0.4) == 1.0

def test_l0_correct_below_threshold(prm):
    assert prm.l0_correct(f1=0.3, threshold=0.4) == 0.0

def test_l0_correct_at_threshold(prm):
    # Exact threshold — test that the implementation is consistent
    val = prm.l0_correct(f1=0.4, threshold=0.4)
    assert val in (0.0, 1.0)

def test_l0_correct_returns_float(prm):
    result = prm.l0_correct(f1=0.6, threshold=0.4)
    assert isinstance(result, float)
