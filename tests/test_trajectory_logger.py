"""Unit tests for condor/trajectory_logger.py"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.trajectory_logger import TrajectoryLogger


def _make_features():
    return np.random.randn(32).astype(np.float32)


@pytest.fixture
def logger():
    return TrajectoryLogger(max_size=10)


# ── log / len ─────────────────────────────────────────────────────────────────

def test_log_adds_entry(logger):
    assert len(logger) == 0
    logger.log(1.0, 0.7, 0.01, 3, _make_features(), 1, 0, "question?")
    assert len(logger) == 1

def test_log_multiple(logger):
    for i in range(5):
        logger.log(float(i % 2), 0.5, 0.02, 2, _make_features(), i % 5, i % 4, f"q{i}")
    assert len(logger) == 5


# ── FIFO behaviour ────────────────────────────────────────────────────────────

def test_fifo_respects_max_size():
    log = TrajectoryLogger(max_size=3)
    for i in range(6):
        log.log(1.0, 0.5, 0.01, 1, _make_features(), 0, 0, f"q{i}")
    assert len(log) == 3

def test_fifo_keeps_newest():
    log = TrajectoryLogger(max_size=3)
    for i in range(5):
        log.log(float(i), float(i) * 0.1, 0.01, 1, _make_features(), 0, 0, f"q{i}")
    # Only last 3 entries (i=2,3,4) should remain
    mat = log.get_feature_matrix()
    assert mat.shape[0] == 3


# ── get_feature_matrix ────────────────────────────────────────────────────────

def test_feature_matrix_shape(logger):
    for i in range(4):
        logger.log(1.0, 0.6, 0.01, 2, _make_features(), 0, 0, "q")
    mat = logger.get_feature_matrix()
    # [l0_correct(1), f1(1), cost(1), n_turns(1), emb32(32)] = 36
    assert mat.shape == (4, TrajectoryLogger.FEATURE_DIM)

def test_feature_matrix_dtype(logger):
    logger.log(1.0, 0.5, 0.02, 3, _make_features(), 2, 1, "q")
    mat = logger.get_feature_matrix()
    assert mat.dtype in (np.float32, np.float64)

def test_feature_matrix_values_correct():
    log = TrajectoryLogger(max_size=100)
    feat = np.ones(32, dtype=np.float32)
    log.log(l0_correct=1.0, f1=0.75, cost=0.005, n_turns=4,
            query_features=feat, m_star=2, qtype_id=1, question="test")
    mat = log.get_feature_matrix()
    assert abs(mat[0, 0] - 1.0) < 1e-6   # l0_correct
    assert abs(mat[0, 1] - 0.75) < 1e-6  # f1
    assert abs(mat[0, 2] - 0.005) < 1e-6 # cost
    assert abs(mat[0, 3] - 4.0) < 1e-6   # n_turns
    assert np.allclose(mat[0, 4:], 1.0)   # embedding


# ── get_summary ───────────────────────────────────────────────────────────────

def test_summary_keys(logger):
    logger.log(1.0, 0.6, 0.01, 2, _make_features(), 0, 0, "q")
    summary = logger.get_summary()
    assert 'avg_f1' in summary
    assert 'avg_l0_correct' in summary
    assert 'avg_cost' in summary

def test_summary_avg_f1(logger):
    f1s = [0.4, 0.6, 0.8]
    for f1 in f1s:
        logger.log(1.0, f1, 0.01, 2, _make_features(), 0, 0, "q")
    summary = logger.get_summary()
    assert abs(summary['avg_f1'] - sum(f1s) / len(f1s)) < 1e-5

def test_summary_empty_logger():
    log = TrajectoryLogger()
    summary = log.get_summary()
    assert isinstance(summary, dict)
