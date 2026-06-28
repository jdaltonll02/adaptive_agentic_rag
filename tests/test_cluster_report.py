"""Unit tests for condor/analysis/cluster_report.py"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.analysis.cluster_report import ClusterReport


@pytest.fixture
def report():
    return ClusterReport(n_clusters=4)


def _make_feature_matrix(n=40, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 35)).astype(np.float64)


# ── analyze ───────────────────────────────────────────────────────────────────

def test_analyze_returns_dict(report):
    result = report.analyze(_make_feature_matrix())
    assert isinstance(result, dict)

def test_analyze_has_cluster_keys(report):
    result = report.analyze(_make_feature_matrix())
    for k in range(4):
        assert f'cluster_{k}' in result

def test_analyze_each_cluster_has_taxonomy(report):
    result = report.analyze(_make_feature_matrix())
    valid_taxonomies = {'efficient_correct', 'efficient_incorrect',
                        'expensive_correct', 'expensive_incorrect', 'empty'}
    for k in range(4):
        assert result[f'cluster_{k}']['taxonomy'] in valid_taxonomies

def test_analyze_each_cluster_has_size(report):
    result = report.analyze(_make_feature_matrix())
    for k in range(4):
        assert 'size' in result[f'cluster_{k}']
        assert result[f'cluster_{k}']['size'] >= 0

def test_analyze_sizes_sum_to_n(report):
    n = 40
    result = report.analyze(_make_feature_matrix(n=n))
    total = sum(result[f'cluster_{k}']['size'] for k in range(4))
    assert total == n

def test_analyze_too_few_samples_returns_error_dict(report):
    result = report.analyze(np.zeros((2, 35)))
    assert 'error' in result

def test_analyze_cluster_avg_f1_is_float(report):
    result = report.analyze(_make_feature_matrix())
    for k in range(4):
        if result[f'cluster_{k}']['size'] > 0:
            assert isinstance(result[f'cluster_{k}']['avg_f1'], float)

def test_analyze_reproducible(report):
    mat = _make_feature_matrix()
    r1 = report.analyze(mat)
    r2 = report.analyze(mat)
    # Same data → same assignments (random_state is fixed)
    for k in range(4):
        assert r1[f'cluster_{k}']['size'] == r2[f'cluster_{k}']['size']


# ── wandb_summary ─────────────────────────────────────────────────────────────

def test_wandb_summary_returns_flat_dict(report):
    result = report.analyze(_make_feature_matrix())
    summary = report.wandb_summary(result)
    assert isinstance(summary, dict)
    for key, val in summary.items():
        assert not isinstance(val, dict), f"wandb_summary value for {key!r} should be scalar"

def test_wandb_summary_prefix(report):
    result = report.analyze(_make_feature_matrix())
    summary = report.wandb_summary(result)
    for key in summary.keys():
        assert key.startswith('condor/cluster/')

def test_wandb_summary_values_are_scalars(report):
    result = report.analyze(_make_feature_matrix())
    summary = report.wandb_summary(result)
    for val in summary.values():
        assert isinstance(val, (int, float, str))

def test_wandb_summary_nonempty(report):
    result = report.analyze(_make_feature_matrix())
    summary = report.wandb_summary(result)
    assert len(summary) > 0
