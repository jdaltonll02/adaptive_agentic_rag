"""Training dynamics plots.

Produces a grid of time-series plots from the training CSV:

  Row 0: Actor PG loss | Critic VF loss | Actor KL
  Row 1: F1 (train)   | Token Cost      | Turns
  Row 2: lambda_hi     | lambda_lo       | Phi loss
  Row 3: Epsilon       | L0 Policy loss  | L0 Avg reward

Usage
-----
    from condor.plots.training_curves import plot_training_curves
    plot_training_curves('outputs/metrics/run_training.csv',
                         output_path='outputs/plots/training_curves.pdf')
"""

import os
from typing import List, Optional, Tuple

import numpy as np


def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'figure.dpi': 150,
    })
    return plt


_PANEL_SPECS = [
    # (col_name, panel_title, y_label, smoothing)
    ('actor/pg_loss',          'PG Loss',           'Loss',           True),
    ('critic/vf_loss',         'VF Loss',            'Loss',           True),
    ('actor/kl',               'Actor KL',           'KL',             True),
    ('train/f1',               'Train F1',           'F1',             True),
    ('train/token_cost_mean',  'Token Cost (mean)',  'USD',            True),
    ('train/turns_mean',       'Avg Turns',          'Turns',          True),
    ('condor/lambda_hi',       r'$\lambda_{hi}$',   r'$\lambda$',     False),
    ('condor/lambda_lo',       r'$\lambda_{lo}$',   r'$\lambda$',     False),
    ('condor/phi_loss',        'Phi Loss',           'MSE',            True),
    ('condor/router_epsilon',  r'Router $\epsilon$', r'$\epsilon$',   False),
    ('condor/l0_policy_loss',  'L0 Policy Loss',     'Loss',           True),
    ('condor/avg_l0_reward',   'L0 Avg Reward',      'Reward',         True),
]


def _smooth(y: np.ndarray, window: int = 20) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def plot_training_curves(
    csv_path: str,
    output_path: str = 'outputs/plots/training_curves.pdf',
    smooth_window: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
) -> str:
    """Plot all training-curve panels from the training CSV.

    Parameters
    ----------
    csv_path : str
        Path to *_training.csv produced by MetricsCollector.
    output_path : str
        Where to save the figure.
    smooth_window : int
        Rolling-average window in training steps (0 = no smoothing).
    """
    import csv as _csv

    plt = _setup_mpl()
    import matplotlib.pyplot as _plt

    rows = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        print('[plot_training_curves] Empty CSV — skipping.')
        return output_path

    steps = []
    data: dict = {col: [] for col, *_ in _PANEL_SPECS}
    for row in rows:
        try:
            steps.append(int(row.get('step', 0)))
        except (ValueError, TypeError):
            steps.append(len(steps))
        for col, *_ in _PANEL_SPECS:
            val = row.get(col, '')
            try:
                data[col].append(float(val))
            except (ValueError, TypeError):
                data[col].append(float('nan'))

    steps = np.array(steps)
    n_panels = len(_PANEL_SPECS)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 4.5, nrows * 3.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    for i, (col, title, y_label, do_smooth) in enumerate(_PANEL_SPECS):
        ax = axes_flat[i]
        y = np.array(data[col], dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            continue

        x, y = steps[valid], y[valid]
        ax.plot(x, y, color='#BAB0AC', linewidth=0.6, alpha=0.5)
        if do_smooth and smooth_window > 1:
            y_sm = _smooth(y, window=smooth_window)
            ax.plot(x, y_sm, color='#4E79A7', linewidth=1.8)
        else:
            ax.plot(x, y, color='#4E79A7', linewidth=1.6)

        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    # Hide unused panels
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Training Dynamics', fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_training_curves] Saved → {output_path}')
    return output_path
