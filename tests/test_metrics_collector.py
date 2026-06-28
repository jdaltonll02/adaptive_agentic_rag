"""Unit tests for condor/metrics/collector.py"""

import csv
import json
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from condor.metrics.collector import MetricsCollector


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def collector(tmpdir):
    c = MetricsCollector(metrics_dir=tmpdir, experiment_name='test_run')
    yield c
    c.close()


def _make_record(f1=0.6, em=0.5, acc=0.5, tc=0.001, ret=1, turns=2, m=1, qt=0, wf='QR,R,AG'):
    return {
        'question': 'What is the capital of France?',
        'predicted_answer': 'Paris',
        'golden_answer': ['Paris'],
        'f1': f1, 'em': em, 'accuracy': acc,
        'token_cost_usd': tc,
        'retrieval_calls': ret,
        'turns': turns,
        'm_star': m, 'qtype_id': qt,
        'workflow': wf,
        'l0_correct': 1.0 if f1 >= 0.4 else 0.0,
    }


# ── record_training_step ──────────────────────────────────────────────────────

def test_training_csv_created(collector, tmpdir):
    assert os.path.exists(os.path.join(tmpdir, 'test_run_training.csv'))

def test_record_training_step_writes_row(collector, tmpdir):
    collector.record_training_step(step=1, metrics={'train/f1': 0.5, 'condor/lambda_hi': 0.1})
    csv_path = os.path.join(tmpdir, 'test_run_training.csv')
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]['step'] == '1'

def test_record_training_step_multiple_rows(collector, tmpdir):
    for i in range(5):
        collector.record_training_step(step=i, metrics={'train/f1': i * 0.1})
    csv_path = os.path.join(tmpdir, 'test_run_training.csv')
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5

def test_record_training_step_mechanism_key_normalised(collector, tmpdir):
    collector.record_training_step(step=1, metrics={
        'condor/mechanism/Pure_LLM': 0.2,
    })
    csv_path = os.path.join(tmpdir, 'test_run_training.csv')
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    # Should be stored as condor/mech_Pure_LLM
    assert 'condor/mech_Pure_LLM' in rows[0]


# ── record_eval_batch ─────────────────────────────────────────────────────────

def test_eval_jsonl_created_on_first_batch(collector, tmpdir):
    records = [_make_record()]
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=records)
    assert os.path.exists(os.path.join(tmpdir, 'test_run_eval.jsonl'))

def test_eval_summary_json_created(collector, tmpdir):
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=[_make_record()])
    assert os.path.exists(os.path.join(tmpdir, 'test_run_eval_summary.json'))

def test_eval_summary_f1_correct(collector, tmpdir):
    records = [_make_record(f1=0.4), _make_record(f1=0.8)]
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=records)
    with open(os.path.join(tmpdir, 'test_run_eval_summary.json')) as fh:
        summary = json.load(fh)
    assert abs(summary['nq']['f1'] - 0.6) < 1e-6

def test_eval_summary_token_cost_milliusd(collector, tmpdir):
    records = [_make_record(tc=0.002)]
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=records)
    with open(os.path.join(tmpdir, 'test_run_eval_summary.json')) as fh:
        summary = json.load(fh)
    assert abs(summary['nq']['token_cost_milliusd_mean'] - 2.0) < 1e-4

def test_eval_per_mechanism_breakdown(collector, tmpdir):
    records = [_make_record(m=0), _make_record(m=0), _make_record(m=2)]
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=records)
    with open(os.path.join(tmpdir, 'test_run_eval_summary.json')) as fh:
        summary = json.load(fh)
    assert 'mech_0_count' in summary['nq']
    assert summary['nq']['mech_0_count'] == 2

def test_eval_empty_records_skipped(collector, tmpdir):
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=[])
    assert not os.path.exists(os.path.join(tmpdir, 'test_run_eval.jsonl'))

def test_eval_multiple_batches_append(collector, tmpdir):
    collector.record_eval_batch(step=0, dataset='nq', per_query_records=[_make_record()])
    collector.record_eval_batch(step=10, dataset='nq', per_query_records=[_make_record()])
    jsonl_path = os.path.join(tmpdir, 'test_run_eval.jsonl')
    with open(jsonl_path) as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == 2


# ── make_per_query_record ─────────────────────────────────────────────────────

def test_make_per_query_record_has_required_keys(collector):
    r = collector.make_per_query_record(
        question='q', predicted_answer='a', golden_answer=['a'],
        f1=0.5, em=0.5, accuracy=0.5, token_cost_usd=0.001,
        retrieval_calls=1, turns=2,
    )
    for key in ('question', 'predicted_answer', 'golden_answer', 'f1', 'em',
                'accuracy', 'token_cost_usd', 'retrieval_calls', 'turns'):
        assert key in r

def test_make_per_query_record_optional_fields_default_none(collector):
    r = collector.make_per_query_record(
        question='q', predicted_answer='a', golden_answer='a',
        f1=0.5, em=0.0, accuracy=0.0, token_cost_usd=0.0,
        retrieval_calls=0, turns=1,
    )
    assert r['m_star'] is None
    assert r['qtype_id'] is None
    assert r['workflow'] is None
