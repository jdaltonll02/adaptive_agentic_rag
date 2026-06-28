"""Figure 4 equivalent — ablation curves.

Four subplots showing how key metrics change across discrete values of a
hyperparameter (default: lambda_lo or the eval sweep variable):

  (a) F1 (%)
  (b) Token Cost (milli-USD)
  (c) Retrieval Calls
  (d) Total Turns

Each subplot shows CONDOR and (where available) the MAO-ARAG baseline as a
horizontal dashed reference line.

Usage
-----
    from condor.plots.ablation import plot_ablation
    plot_ablation(sweep_results, output_path='outputs/plots/fig4_ablation.pdf')

sweep_results is a list of dicts, one per sweep point:
    [
        {'param_value': 0.0, 'f1': 0.52, 'token_cost_milliusd': 0.40, ...},
        {'param_value': 0.1, 'f1': 0.54, ...},
        ...
    ]
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })
    return plt


_BASELINE_REFS = {
    'MAO-ARAG': {'f1': 52.91, 'token_cost_milliusd': 1.129,
                 'retrieval_calls': 2.327, 'turns': 2.899},
}


def plot_ablation(
    sweep_results: List[Dict[str, Any]],
    output_path: str = 'outputs/plots/fig4_ablation.pdf',
    x_label: str = r'$\lambda_{lo}$',
    figsize: Tuple[float, float] = (14, 4),
    experiment_label: str = 'CONDOR',
) -> str:
    """Generate four-panel ablation figure (Figure 4 equivalent).

    Parameters
    ----------
    sweep_results : list[dict]
        Each element must have 'param_value' and at least one of:
        'f1', 'token_cost_milliusd', 'retrieval_calls', 'turns'.
    output_path : str
        Where to save the figure.
    x_label : str
        X-axis label (latex OK).
    """
    plt = _setup_mpl()
    import matplotlib.pyplot as _plt

    if not sweep_results:
        print('[plot_ablation] No sweep results — skipping.')
        return output_path

    x_vals = [r['param_value'] for r in sweep_results]

    panels = [
        ('f1',                  'F1 (%)',                   '(a)', False),
        ('token_cost_milliusd', 'Token Cost (milli-USD)',   '(b)', True),
        ('retrieval_calls',     'Retrieval Calls',           '(c)', True),
        ('turns',               'Total Turns',              '(d)', True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    for ax, (key, y_label, panel, lower_better) in zip(axes, panels):
        y_vals = [r.get(key) for r in sweep_results]
        valid = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker='o', color='#4E79A7',
                    linewidth=2, markersize=6, label=experiment_label)
            ax.fill_between(
                xs,
                [y - _std(sweep_results, key) for y in ys],
                [y + _std(sweep_results, key) for y in ys],
                alpha=0.15, color='#4E79A7',
            )

        # Baseline reference lines
        for b_name, b_vals in _BASELINE_REFS.items():
            if key in b_vals:
                ax.axhline(b_vals[key], linestyle='--', color='#E15759',
                           linewidth=1.4, label=b_name)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label if ax is axes[0] else '')
        ax.set_title(panel)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.04), fontsize=9)

    fig.suptitle('Figure 4: Ablation Study', fontsize=12, y=1.08)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_ablation] Saved → {output_path}')
    return output_path


def _std(sweep_results: List[Dict], key: str) -> float:
    vals = [r[key] for r in sweep_results if r.get(key) is not None]
    if len(vals) < 2:
        return 0.0
    arr = np.array(vals, dtype=float)
    return float(np.std(arr))


# ── Convenience: build sweep_results from the training CSV ──────────────────

def sweep_from_training_csv(csv_path: str, param_col: str = 'condor/lambda_lo',
                             metric_col: str = 'condor/avg_f1',
                             n_bins: int = 10) -> List[Dict[str, Any]]:
    """Bin training-step data to create sweep_results for plot_ablation.

    Reads the training CSV produced by MetricsCollector and bins by param_col
    to approximate an ablation curve.
    """
    import csv as _csv
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({k: float(v) for k, v in row.items() if v})
            except ValueError:
                pass

    if not rows:
        return []

    param_vals = [r[param_col] for r in rows if param_col in r]
    if not param_vals:
        return []

    lo, hi = min(param_vals), max(param_vals)
    bin_size = (hi - lo) / n_bins if hi > lo else 1.0
    bins: Dict[int, List] = {}
    for r in rows:
        if param_col not in r:
            continue
        bin_idx = int((r[param_col] - lo) / bin_size)
        bins.setdefault(bin_idx, []).append(r)

    results = []
    for idx in sorted(bins.keys()):
        bin_rows = bins[idx]
        param_value = lo + (idx + 0.5) * bin_size

        def _mean(col):
            vals = [r[col] for r in bin_rows if col in r]
            return float(sum(vals) / len(vals)) if vals else None

        results.append({
            'param_value': param_value,
            'f1': _mean('condor/avg_f1'),
            'token_cost_milliusd': _mean('train/token_cost_mean'),
            'retrieval_calls': _mean('train/retrieval_calls_mean'),
            'turns': _mean('train/turns_mean'),
        })

    return results
