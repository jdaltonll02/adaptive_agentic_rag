"""Integration / smoke tests — verify the full CONDOR pipeline imports and
runs end-to-end with minimal synthetic data (no GPU, no baseline API needed).

These tests intentionally avoid any baseline imports so they can run without
the full verl/Ray stack.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Import smoke tests ────────────────────────────────────────────────────────

def test_import_condor_package():
    from condor import (
        MechanismRouter, DualAlphaManager, PhiScorer,
        TrajectoryLogger, ProcessRewardModel,
        MECHANISM_ACTION_SPACES, MECHANISM_NAMES,
        MECHANISM_SELECTION_COST, QTYPE_LABELS,
        classify_qtype, query_to_features,
    )

def test_import_condor_analysis():
    from condor.analysis import ClusterReport

def test_import_condor_metrics():
    from condor.metrics import MetricsCollector, TablesGenerator

def test_import_condor_plots():
    from condor.plots import (
        plot_pareto, plot_ablation, plot_training_curves,
        plot_mechanism_distribution, plot_cluster_heatmap,
        plot_mechanism_f1_bar, plot_phi_blend_curve,
    )

def test_import_retriever():
    from retriever import GraphRetriever, WebRetriever

def test_import_qa_manager_config():
    from qa_manager.config import MECHANISM_ACTION_SPACES, MECHANISM_GOAL_DESCRIPTIONS

def test_import_planning_agent():
    from qa_manager.PlanningAgent import PlanningAgent


# ── End-to-end mini pipeline (CPU only) ──────────────────────────────────────

def test_e2e_mechanism_to_planning():
    """Route a question → mechanism → goal-conditioned plan (no LLM call)."""
    from condor.mechanism_router import MechanismRouter, classify_qtype
    from qa_manager.PlanningAgent import PlanningAgent

    router = MechanismRouter(epsilon_start=0.0, epsilon_end=0.0,
                              epsilon_decay=1.0, use_condor=True)
    pa = PlanningAgent()

    question = "What was the GDP of Japan in 2019 compared to 2020?"
    m_star, qtype_id, features = router.route(question)

    msgs = pa.create_planning_messages(question, goal=(m_star, qtype_id))
    assert len(msgs) == 5
    assert 'Qustion' in msgs[-1]['content']
    assert question in msgs[-1]['content']


def test_e2e_reward_and_trajectory():
    """Compute L0 reward and log to trajectory buffer."""
    from condor.process_reward import ProcessRewardModel
    from condor.trajectory_logger import TrajectoryLogger
    from condor.mechanism_router import query_to_features

    prm = ProcessRewardModel()
    log = TrajectoryLogger(max_size=100)

    for i in range(5):
        feat = query_to_features(f"Question number {i}")
        reward = prm.compute_l0_reward(f1=0.5 + i * 0.05, m_star=1,
                                        lambda_hi=0.1, baseline_f1=0.4)
        l0_ok = prm.l0_correct(0.5 + i * 0.05, threshold=0.4)
        log.log(l0_ok, 0.5 + i * 0.05, 0.01, 2, feat, 1, 0, f"q{i}")

    assert len(log) == 5
    from condor.trajectory_logger import TrajectoryLogger as TL
    mat = log.get_feature_matrix()
    assert mat.shape == (5, TL.FEATURE_DIM)


def test_e2e_dual_alpha_converges():
    """DualAlphaManager should converge lambdas in response to cost signals."""
    from condor.dual_alpha_manager import DualAlphaManager

    hi = DualAlphaManager(budget=0.0005, lr=5e-3, lam_init=0.10)
    lo = DualAlphaManager(budget=0.0004, lr=5e-3, lam_init=0.08)

    # Simulate over-budget scenario
    for _ in range(200):
        hi.update(mean_cost=0.01)
        lo.update(mean_cost=0.01)

    assert hi.value > 0.10, "lambda_hi should increase when cost > budget"
    assert lo.value > 0.08, "lambda_lo should increase when cost > budget"


def test_e2e_phi_update_and_blend():
    """PhiScorer update and blended baseline should produce consistent output."""
    from condor.phi_scorer import PhiScorer
    from condor.mechanism_router import query_to_features

    phi = PhiScorer(t_warmup=10, tau_trans=5.0, lr=1e-2)
    embs = [query_to_features(f"question {i}") for i in range(8)]
    targets = [0.5 + 0.1 * (i % 3) for i in range(8)]

    for _ in range(20):
        phi.update(embs, [i % 5 for i in range(8)], [i % 4 for i in range(8)], targets)

    phi_val = phi.score(embs[0], 1, 0)
    alpha = phi.get_blend_alpha()
    blended = phi.blended_baseline(phi_val, 0.6)

    assert isinstance(blended, float)
    assert np.isfinite(blended)


def test_e2e_cluster_report():
    """ClusterReport should produce a non-empty report from trajectory features."""
    from condor.analysis import ClusterReport
    from condor.trajectory_logger import TrajectoryLogger
    from condor.mechanism_router import query_to_features

    log = TrajectoryLogger(max_size=100)
    rng = np.random.default_rng(0)
    for i in range(20):
        log.log(float(i % 2), rng.uniform(0.2, 0.9), rng.uniform(0.001, 0.01),
                rng.integers(1, 6), query_to_features(f"q{i}"), i % 5, i % 4, f"q{i}")

    cr = ClusterReport(n_clusters=4)
    mat = log.get_feature_matrix()
    report = cr.analyze(mat)

    cluster_keys = [k for k in report if k.startswith('cluster_')]
    assert len(cluster_keys) > 0
    summary = cr.wandb_summary(report)
    assert all(k.startswith('condor/cluster/') for k in summary.keys())


def test_e2e_metrics_collector(tmp_path):
    """MetricsCollector should write valid CSV and JSON."""
    from condor.metrics import MetricsCollector

    col = MetricsCollector(metrics_dir=str(tmp_path), experiment_name='smoke')

    # Training steps
    for step in range(5):
        col.record_training_step(step=step, metrics={
            'train/f1': 0.5 + step * 0.01,
            'condor/lambda_hi': 0.1,
            'condor/lambda_lo': step * 0.01,
        })

    # Eval batch
    records = [
        col.make_per_query_record(
            question=f'q{i}', predicted_answer='Paris', golden_answer=['Paris'],
            f1=0.6, em=1.0, accuracy=1.0, token_cost_usd=0.001,
            retrieval_calls=1, turns=2, m_star=1, qtype_id=0,
            workflow='QR,R,AG', l0_correct=1.0,
        )
        for i in range(4)
    ]
    col.record_eval_batch(step=5, dataset='nq', per_query_records=records)
    col.close()

    csv_path = tmp_path / 'smoke_training.csv'
    summary_path = tmp_path / 'smoke_eval_summary.json'

    assert csv_path.exists()
    assert summary_path.exists()

    with open(summary_path) as fh:
        summary = json.load(fh)
    assert 'nq' in summary
    assert abs(summary['nq']['f1'] - 0.6) < 1e-5


def test_e2e_generate_all_plots(tmp_path):
    """generate_all plots CLI should run without error on synthetic data."""
    import csv as _csv
    import json as _json

    metrics_dir = tmp_path / 'metrics'
    plots_dir = tmp_path / 'plots'
    metrics_dir.mkdir()
    plots_dir.mkdir()
    exp = 'smoke_plot_test'

    # Minimal training CSV
    from condor.metrics.collector import _TRAINING_COLS
    train_path = metrics_dir / f'{exp}_training.csv'
    with open(train_path, 'w', newline='') as fh:
        writer = _csv.DictWriter(fh, fieldnames=_TRAINING_COLS)
        writer.writeheader()
        for step in range(10):
            writer.writerow({'step': step, 'condor/lambda_hi': 0.1 + step * 0.01,
                             'condor/lambda_lo': step * 0.005, 'train/f1': 0.5,
                             'condor/phi_loss': 0.1, 'condor/router_epsilon': 0.9})

    # Minimal eval summary JSON
    summary = {
        'nq': {
            'f1': 0.6, 'em': 0.5, 'token_cost_milliusd_mean': 0.4,
            'retrieval_calls_mean': 1.5, 'turns_mean': 2.0,
            'mech_1_f1': 0.65, 'mech_1_count': 10,
            'mech_2_f1': 0.55, 'mech_2_count': 5,
        }
    }
    with open(metrics_dir / f'{exp}_eval_summary.json', 'w') as fh:
        _json.dump(summary, fh)

    # Run via module
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'condor.plots.generate_all',
         '--metrics-dir', str(metrics_dir),
         '--plots-dir', str(plots_dir),
         '--experiment', exp,
         '--dataset', 'NQ'],
        cwd=os.path.join(os.path.dirname(__file__), '..'),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"generate_all failed:\n{result.stderr}"
    # At least one output file should exist
    output_files = list(plots_dir.iterdir())
    assert len(output_files) > 0
