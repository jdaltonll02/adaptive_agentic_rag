"""Unit tests for condor/dual_alpha_manager.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.dual_alpha_manager import DualAlphaManager


@pytest.fixture
def dam():
    return DualAlphaManager(
        lambda_hi_init=0.1,
        lambda_lo_init=0.0,
        budget_hi=0.5,
        budget_lo=0.3,
        lr=1e-3,
        lambda_min=0.0,
        lambda_max=10.0,
    )


# ── Initialisation ────────────────────────────────────────────────────────────

def test_initial_lambda_hi(dam):
    assert abs(dam.lambda_hi - 0.1) < 1e-9

def test_initial_lambda_lo(dam):
    assert abs(dam.lambda_lo - 0.0) < 1e-9

def test_initial_values_in_bounds(dam):
    assert dam.lambda_min <= dam.lambda_hi <= dam.lambda_max
    assert dam.lambda_min <= dam.lambda_lo <= dam.lambda_max


# ── update() ─────────────────────────────────────────────────────────────────

def test_update_returns_two_floats(dam):
    lhi, llo = dam.update(avg_cost_hi=0.6, avg_cost_lo=0.4)
    assert isinstance(lhi, float)
    assert isinstance(llo, float)

def test_update_hi_increases_when_cost_exceeds_budget(dam):
    # cost > budget → constraint violated → lambda should increase
    old_hi = dam.lambda_hi
    for _ in range(50):
        dam.update(avg_cost_hi=0.9, avg_cost_lo=0.1)
    assert dam.lambda_hi > old_hi

def test_update_hi_decreases_when_cost_below_budget():
    d = DualAlphaManager(lambda_hi_init=5.0, lambda_lo_init=0.0,
                          budget_hi=0.5, budget_lo=0.3, lr=1e-2)
    for _ in range(100):
        d.update(avg_cost_hi=0.1, avg_cost_lo=0.1)
    assert d.lambda_hi < 5.0

def test_update_lo_increases_when_cost_exceeds_budget(dam):
    old_lo = dam.lambda_lo
    for _ in range(50):
        dam.update(avg_cost_hi=0.1, avg_cost_lo=0.9)
    assert dam.lambda_lo > old_lo

def test_lambda_clamped_to_max():
    d = DualAlphaManager(lambda_hi_init=9.9, lambda_lo_init=0.0,
                          budget_hi=0.0, budget_lo=0.0,
                          lr=10.0, lambda_max=10.0)
    for _ in range(100):
        d.update(avg_cost_hi=1.0, avg_cost_lo=1.0)
    assert d.lambda_hi <= 10.0

def test_lambda_clamped_to_min():
    d = DualAlphaManager(lambda_hi_init=0.1, lambda_lo_init=0.1,
                          budget_hi=1.0, budget_lo=1.0,
                          lr=10.0, lambda_min=0.0)
    for _ in range(100):
        d.update(avg_cost_hi=0.0, avg_cost_lo=0.0)
    assert d.lambda_hi >= 0.0
    assert d.lambda_lo >= 0.0


# ── state_dict / load_state_dict ─────────────────────────────────────────────

def test_state_dict_roundtrip(dam):
    dam.update(0.6, 0.4)
    sd = dam.state_dict()
    d2 = DualAlphaManager(lambda_hi_init=0.0, lambda_lo_init=0.0,
                           budget_hi=0.5, budget_lo=0.3, lr=1e-3)
    d2.load_state_dict(sd)
    assert abs(d2.lambda_hi - dam.lambda_hi) < 1e-9
    assert abs(d2.lambda_lo - dam.lambda_lo) < 1e-9

def test_state_dict_has_required_keys(dam):
    sd = dam.state_dict()
    for key in ('lambda_hi', 'lambda_lo', 'm_hi', 'v_hi', 'm_lo', 'v_lo', 'step'):
        assert key in sd, f"state_dict missing key '{key}'"

def test_load_state_dict_restores_step_count(dam):
    for _ in range(10):
        dam.update(0.5, 0.3)
    sd = dam.state_dict()
    d2 = DualAlphaManager()
    d2.load_state_dict(sd)
    assert d2.state_dict()['step'] == sd['step']
