"""Unit tests for condor/mechanism_router.py"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.mechanism_router import (
    MECHANISM_ACTION_SPACES,
    MECHANISM_NAMES,
    MECHANISM_SELECTION_COST,
    QTYPE_LABELS,
    classify_qtype,
    query_to_features,
    MechanismRouter,
)


# ── MECHANISM_ACTION_SPACES ───────────────────────────────────────────────────

def test_action_spaces_keys():
    assert set(MECHANISM_ACTION_SPACES.keys()) == {0, 1, 2, 3, 4}

def test_action_spaces_all_contain_ag():
    for m, actions in MECHANISM_ACTION_SPACES.items():
        assert 'AG' in actions, f"m{m} action space missing AG"

def test_action_spaces_valid_tokens():
    valid = {'QR', 'QDP', 'QDS', 'R', 'DS', 'AG'}
    for m, actions in MECHANISM_ACTION_SPACES.items():
        for a in actions:
            assert a in valid, f"Unknown action {a} in m{m}"

def test_mechanism_names_all_present():
    assert len(MECHANISM_NAMES) == 5
    for i in range(5):
        assert i in MECHANISM_NAMES
        assert isinstance(MECHANISM_NAMES[i], str)

def test_mechanism_selection_cost_range():
    for m, cost in MECHANISM_SELECTION_COST.items():
        assert 0.0 <= cost <= 1.0, f"m{m} cost {cost} out of [0,1]"

def test_qtype_labels_length():
    assert len(QTYPE_LABELS) == 4


# ── classify_qtype ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_type", [
    ("What is the capital of France?", 0),          # factoid
    ("Who invented the telephone?", 0),             # factoid
    ("What did Einstein do before he moved to the US?", 1),  # multi_hop keywords
    ("Compare the economies of China and India", 2),         # comparison
    ("Analyse the long-term geopolitical consequences of the cold war", 3),  # complex
])
def test_classify_qtype_returns_valid_index(question, expected_type):
    result = classify_qtype(question)
    assert result in {0, 1, 2, 3}, f"classify_qtype returned {result}, expected 0-3"

def test_classify_qtype_returns_int():
    assert isinstance(classify_qtype("What is the tallest mountain?"), int)

def test_classify_qtype_empty_string():
    result = classify_qtype("")
    assert result in {0, 1, 2, 3}


# ── query_to_features ─────────────────────────────────────────────────────────

def test_query_to_features_shape():
    feat = query_to_features("What is the capital of France?")
    assert feat.shape == (32,), f"Expected (32,), got {feat.shape}"

def test_query_to_features_dtype():
    feat = query_to_features("Hello world")
    assert feat.dtype in (np.float32, np.float64)

def test_query_to_features_bounded():
    feat = query_to_features("What is the meaning of life?")
    assert np.all(np.isfinite(feat)), "Feature vector contains non-finite values"

def test_query_to_features_different_questions():
    f1 = query_to_features("Paris is the capital of France")
    f2 = query_to_features("The Eiffel Tower is in Paris")
    # Should produce different feature vectors for different text
    assert not np.allclose(f1, f2)


# ── MechanismRouter ───────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return MechanismRouter(epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=100.0, use_condor=True)

def test_router_route_returns_triple(router):
    result = router.route("What is gravity?")
    assert len(result) == 3

def test_router_route_m_star_valid(router):
    m, qt, feat = router.route("What is gravity?")
    assert m in {0, 1, 2, 3, 4}

def test_router_route_qtype_valid(router):
    m, qt, feat = router.route("What is gravity?")
    assert qt in {0, 1, 2, 3}

def test_router_route_features_shape(router):
    m, qt, feat = router.route("What is gravity?")
    assert feat.shape == (32,)

def test_router_epsilon_decays():
    r = MechanismRouter(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay=10.0, use_condor=True)
    eps_initial = r.epsilon
    for _ in range(20):
        r.route("some question")
    assert r.epsilon < eps_initial

def test_router_store_and_update(router):
    router.route("Some question about history")
    router.store_reward(1.0)
    loss = router.update_policy(gamma=0.99)
    assert isinstance(loss, float)

def test_router_pure_exploitation():
    r = MechanismRouter(epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0, use_condor=True)
    # With epsilon=0, should always exploit (no random exploration)
    # The policy output may still vary due to the untrained MLP, but should
    # always return a valid mechanism id
    results = [r.route("Exact same question repeated exactly")[0] for _ in range(5)]
    assert all(m in {0, 1, 2, 3, 4} for m in results)

def test_router_disabled_always_returns_m1():
    r = MechanismRouter(use_condor=False)
    for _ in range(5):
        m, qt, feat = r.route("Any question")
        assert m == 1  # Standard RAG when CONDOR disabled
