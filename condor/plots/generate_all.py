"""CLI entry point — generate all figures from a completed CONDOR run.

Usage
-----
    python -m condor.plots.generate_all \\
        --metrics-dir outputs/metrics \\
        --plots-dir   outputs/plots \\
        --experiment  condor_nq_12345_20260601

All plots are written to plots-dir.  The script reads:
    {metrics_dir}/{experiment}_training.csv
    {metrics_dir}/{experiment}_eval_summary.json
    {metrics_dir}/{experiment}_eval.jsonl          (for per-query records)
"""

import argparse
import json
import os
import sys
import csv


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_eval_summary(metrics_dir: str, experiment: str) -> dict:
    path = os.path.join(metrics_dir, f'{experiment}_eval_summary.json')
    if not os.path.exists(path):
        print(f'[generate_all] No eval summary at {path}')
        return {}
    with open(path) as fh:
        return json.load(fh)


def _load_per_query(metrics_dir: str, experiment: str) -> list:
    path = os.path.join(metrics_dir, f'{experiment}_eval.jsonl')
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.extend(obj.get('per_query', []))
            except json.JSONDecodeError:
                pass
    return records


def _load_training_rows(metrics_dir: str, experiment: str) -> list:
    path = os.path.join(metrics_dir, f'{experiment}_training.csv')
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate all CONDOR plots from a completed run.'
    )
    parser.add_argument('--metrics-dir',  default='outputs/metrics',
                        help='Directory containing *_training.csv and *_eval*.json files')
    parser.add_argument('--plots-dir',    default='outputs/plots',
                        help='Output directory for figures')
    parser.add_argument('--experiment',   required=True,
                        help='Experiment name prefix (e.g. condor_nq_12345_20260601)')
    parser.add_argument('--dataset',      default='',
                        help='Dataset label for figure titles (e.g. NQ)')
    parser.add_argument('--t-warmup',     type=int,   default=100,
                        help='t_warmup for phi blend curve (default 100)')
    parser.add_argument('--tau-trans',    type=float, default=50.0,
                        help='tau_trans for phi blend curve (default 50.0)')
    parser.add_argument('--smooth',       type=int,   default=20,
                        help='Smoothing window for training curves (default 20)')
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    eval_summary   = _load_eval_summary(args.metrics_dir, args.experiment)
    per_query      = _load_per_query(args.metrics_dir, args.experiment)
    training_rows  = _load_training_rows(args.metrics_dir, args.experiment)

    training_csv = os.path.join(args.metrics_dir, f'{args.experiment}_training.csv')

    # ── Figure 3: Pareto scatter ──────────────────────────────────────────
    if per_query:
        from condor.plots.pareto import plot_pareto
        plot_pareto(
            per_query,
            output_path=os.path.join(args.plots_dir, 'fig3_pareto.pdf'),
            dataset_label=args.dataset,
        )
        plot_pareto(
            per_query,
            output_path=os.path.join(args.plots_dir, 'fig3_pareto.png'),
            dataset_label=args.dataset,
        )
    else:
        print('[generate_all] No per-query eval data — skipping Fig 3.')

    # ── Figure 4: Ablation ────────────────────────────────────────────────
    if training_rows:
        from condor.plots.ablation import plot_ablation, sweep_from_training_csv
        sweep_data = sweep_from_training_csv(training_csv, n_bins=10)
        if sweep_data:
            plot_ablation(
                sweep_data,
                output_path=os.path.join(args.plots_dir, 'fig4_ablation.pdf'),
            )
            plot_ablation(
                sweep_data,
                output_path=os.path.join(args.plots_dir, 'fig4_ablation.png'),
            )
    else:
        print('[generate_all] No training CSV data — skipping Fig 4.')

    # ── Training curves ───────────────────────────────────────────────────
    if os.path.exists(training_csv):
        from condor.plots.training_curves import plot_training_curves
        plot_training_curves(
            training_csv,
            output_path=os.path.join(args.plots_dir, 'training_curves.pdf'),
            smooth_window=args.smooth,
        )
        plot_training_curves(
            training_csv,
            output_path=os.path.join(args.plots_dir, 'training_curves.png'),
            smooth_window=args.smooth,
        )

    # ── CONDOR-specific plots ─────────────────────────────────────────────
    from condor.plots.condor_specific import (
        plot_mechanism_distribution,
        plot_mechanism_f1_bar,
        plot_phi_blend_curve,
    )

    if training_rows:
        # Build step-indexed mechanism distribution data
        mech_steps = [r for r in training_rows if 'condor/mech_Pure_LLM' in r]
        if mech_steps:
            plot_mechanism_distribution(
                mech_steps,
                output_path=os.path.join(args.plots_dir, 'mechanism_distribution.pdf'),
            )
            plot_mechanism_distribution(
                mech_steps,
                output_path=os.path.join(args.plots_dir, 'mechanism_distribution.png'),
            )

    if eval_summary:
        plot_mechanism_f1_bar(
            eval_summary,
            output_path=os.path.join(args.plots_dir, 'mechanism_f1_bar.pdf'),
        )
        plot_mechanism_f1_bar(
            eval_summary,
            output_path=os.path.join(args.plots_dir, 'mechanism_f1_bar.png'),
        )

    if os.path.exists(training_csv):
        plot_phi_blend_curve(
            training_csv,
            output_path=os.path.join(args.plots_dir, 'phi_blend_curve.pdf'),
            t_warmup=args.t_warmup,
            tau_trans=args.tau_trans,
        )
        plot_phi_blend_curve(
            training_csv,
            output_path=os.path.join(args.plots_dir, 'phi_blend_curve.png'),
            t_warmup=args.t_warmup,
            tau_trans=args.tau_trans,
        )

    # ── Tables ────────────────────────────────────────────────────────────
    from condor.metrics.tables import TablesGenerator
    tables = TablesGenerator(
        metrics_dir=args.metrics_dir,
        output_dir=args.plots_dir,
        experiment_name=args.experiment,
    )
    tables.generate_all()

    print(f'\n[generate_all] Done. Outputs in: {args.plots_dir}')


if __name__ == '__main__':
    main()
