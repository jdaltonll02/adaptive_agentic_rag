"""Unit tests for condor/dual_alpha_manager.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.dual_alpha_manager import DualAlphaManager


@pytest.fixture
def dam_hi():
    return DualAlphaManager(budget=0.0005, lr=5e-4, lam_init=0.10, lam_max=5.0)


@pytest.fixture
def dam_lo():
    return DualAlphaManager(budget=0.0004, lr=1e-3, lam_init=0.08, lam_max=5.0)


# ── Initialisation ────────────────────────────────────────────────────────────

def test_initial_lambda_hi(dam_hi):
    assert abs(dam_hi.value - 0.10) < 1e-9

def test_initial_lambda_lo(dam_lo):
    assert abs(dam_lo.value - 0.08) < 1e-9

def test_initial_values_in_bounds(dam_hi, dam_lo):
    assert 0.0 <= dam_hi.value <= dam_hi.lam_max
    assert 0.0 <= dam_lo.value <= dam_lo.lam_max


# ── update() ─────────────────────────────────────────────────────────────────

def test_update_returns_float(dam_hi):
    result = dam_hi.update(mean_cost=0.001)
    assert isinstance(result, float)

def test_update_increases_when_cost_exceeds_budget(dam_hi):
    old = dam_hi.value
    for _ in range(50):
        dam_hi.update(mean_cost=0.01)   # >> budget 0.0005
    assert dam_hi.value > old

def test_update_decreases_when_cost_below_budget():
    d = DualAlphaManager(budget=0.5, lr=1e-2, lam_init=5.0, lam_max=10.0)
    for _ in range(100):
        d.update(mean_cost=0.1)
    assert d.value < 5.0

def test_lambda_clamped_to_max():
    d = DualAlphaManager(budget=0.0, lr=10.0, lam_init=4.9, lam_max=5.0)
    for _ in range(100):
        d.update(mean_cost=1.0)
    assert d.value <= 5.0

def test_lambda_clamped_to_min():
    d = DualAlphaManager(budget=1.0, lr=10.0, lam_init=0.1, lam_max=5.0)
    for _ in range(100):
        d.update(mean_cost=0.0)
    assert d.value >= 0.0


# ── state_dict / load_state_dict ─────────────────────────────────────────────

def test_state_dict_roundtrip(dam_hi):
    dam_hi.update(mean_cost=0.001)
    sd = dam_hi.state_dict()
    d2 = DualAlphaManager(budget=0.0005, lr=5e-4, lam_init=0.0)
    d2.load_state_dict(sd)
    assert abs(d2.value - dam_hi.value) < 1e-9

def test_state_dict_has_required_keys(dam_hi):
    sd = dam_hi.state_dict()
    for key in ('lam', 't', 'm', 'v'):
        assert key in sd, f"state_dict missing key '{key}'"

def test_load_state_dict_restores_step_count(dam_hi):
    for _ in range(10):
        dam_hi.update(mean_cost=0.001)
    sd = dam_hi.state_dict()
    d2 = DualAlphaManager(budget=0.0005)
    d2.load_state_dict(sd)
    assert d2.state_dict()['t'] == sd['t']

def test_two_instances_independent():
    """L0 and L1 managers must not share state."""
    hi = DualAlphaManager(budget=0.0005, lr=5e-4, lam_init=0.10)
    lo = DualAlphaManager(budget=0.0004, lr=1e-3, lam_init=0.08)
    for _ in range(20):
        hi.update(mean_cost=0.01)   # over budget
        lo.update(mean_cost=0.0)    # under budget
    assert hi.value > 0.10
    assert lo.value < 0.08
