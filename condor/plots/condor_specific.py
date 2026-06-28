"""CONDOR-specific visualisations (no baseline paper equivalent).

Functions
---------
plot_mechanism_distribution(records_by_step, output_path)
    Stacked-bar chart: fraction of queries routed to each mechanism per step.

plot_cluster_heatmap(cluster_report, output_path)
    Heatmap of mean feature values per trajectory cluster.

plot_mechanism_f1_bar(eval_summary, output_path)
    Grouped bar chart: F1 per mechanism, grouped by dataset.

plot_phi_blend_curve(training_csv, output_path)
    Dual-axis plot: phi blend alpha and phi loss vs training step.
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


_MECH_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
_MECH_LABELS = ['m0 Pure LLM', 'm1 Std RAG', 'm2 Adv Iter', 'm3 Graph', 'm4 Web']


# ─── Mechanism Distribution ────────────────────────────────────────────────

def plot_mechanism_distribution(
    records_by_step: List[Dict[str, Any]],
    output_path: str = 'outputs/plots/mechanism_distribution.pdf',
    figsize: Tuple[float, float] = (10, 4),
) -> str:
    """Stacked-bar chart of mechanism selection fractions over training steps.

    Parameters
    ----------
    records_by_step : list[dict]
        Each dict: {'step': int, 'mech_counts': {0: n, 1: n, ...}} or
                   {'step': int, 'condor/mech_Pure_LLM': frac, ...}
    """
    plt = _setup_mpl()
    import matplotlib.pyplot as _plt

    if not records_by_step:
        print('[plot_mechanism_distribution] No data — skipping.')
        return output_path

    steps, fracs = [], []
    mech_keys = [
        ('condor/mech_Pure_LLM', 0),
        ('condor/mech_Standard_RAG', 1),
        ('condor/mech_Advanced_Iterative_RAG', 2),
        ('condor/mech_Graph_RAG', 3),
        ('condor/mech_Web_Search_RAG', 4),
    ]

    for r in records_by_step:
        step = r.get('step', len(steps))
        frac_row = []
        for key, m_id in mech_keys:
            raw_val = r.get(key, r.get(f'mech_{m_id}_frac', 0.0))
            try:
                frac_row.append(float(raw_val) if raw_val != '' else 0.0)
            except (ValueError, TypeError):
                frac_row.append(0.0)
        total = sum(frac_row) or 1.0
        fracs.append([f / total for f in frac_row])
        steps.append(step)

    fracs = np.array(fracs)  # (T, 5)

    fig, ax = plt.subplots(figsize=figsize)
    bottoms = np.zeros(len(steps))
    for m_id in range(5):
        ax.bar(steps, fracs[:, m_id], bottom=bottoms,
               color=_MECH_COLORS[m_id], label=_MECH_LABELS[m_id], width=0.8)
        bottoms += fracs[:, m_id]

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fraction')
    ax.set_title('Mechanism Distribution over Training')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_mechanism_distribution] Saved → {output_path}')
    return output_path


# ─── Cluster Heatmap ──────────────────────────────────────────────────────

def plot_cluster_heatmap(
    cluster_report: Dict[str, Any],
    output_path: str = 'outputs/plots/cluster_heatmap.pdf',
    figsize: Tuple[float, float] = (8, 4),
) -> str:
    """Heatmap of cluster mean features from ClusterReport.

    Parameters
    ----------
    cluster_report : dict
        Output of ClusterReport.analyze(); contains 'cluster_stats' list with
        dicts including 'label', 'mean_l0_correct', 'mean_f1', 'mean_cost',
        'mean_turns', 'count'.
    """
    plt = _setup_mpl()
    import matplotlib.pyplot as _plt

    # cluster_report is {'cluster_0': {...}, 'cluster_1': {...}, ...}
    # (output of ClusterReport.analyze())
    cluster_keys = sorted(k for k in cluster_report.keys()
                          if k.startswith('cluster_') and cluster_report[k].get('size', 0) > 0)
    if not cluster_keys:
        print('[plot_cluster_heatmap] No cluster data — skipping.')
        return output_path

    stats = [cluster_report[k] for k in cluster_keys]
    feature_keys = ['avg_l0_correct', 'avg_f1', 'avg_cost', 'avg_turns']
    feature_labels = ['L0 Correct', 'F1', 'Cost', 'Turns']
    cluster_labels = [s.get('taxonomy', k) for k, s in zip(cluster_keys, stats)]

    matrix = np.array([
        [s.get(k, 0.0) for k in feature_keys] for s in stats
    ])

    # Normalise each column to [0,1]
    col_min = matrix.min(axis=0, keepdims=True)
    col_max = matrix.max(axis=0, keepdims=True)
    normed = (matrix - col_min) / np.where(col_max - col_min > 0, col_max - col_min, 1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(normed, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalised value')

    ax.set_xticks(range(len(feature_keys)))
    ax.set_xticklabels(feature_labels)
    ax.set_yticks(range(len(cluster_labels)))
    ax.set_yticklabels(cluster_labels)

    for i in range(len(stats)):
        for j in range(len(feature_keys)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                    color='black')

    counts = [s.get('size', '') for s in stats]
    ax.set_ylabel('Cluster (n queries)')
    yticks = [f'{lbl} (n={cnt})' for lbl, cnt in zip(cluster_labels, counts)]
    ax.set_yticklabels(yticks)

    ax.set_title('Trajectory Cluster Heatmap')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_cluster_heatmap] Saved → {output_path}')
    return output_path


# ─── Mechanism F1 Bar Chart ───────────────────────────────────────────────

def plot_mechanism_f1_bar(
    eval_summary: Dict[str, Any],
    output_path: str = 'outputs/plots/mechanism_f1_bar.pdf',
    figsize: Optional[Tuple[float, float]] = None,
) -> str:
    """Grouped bar chart: per-mechanism F1 across datasets.

    Parameters
    ----------
    eval_summary : dict
        Nested dict {dataset_key: {mech_0_f1: ..., mech_1_f1: ..., ...}}.
    """
    plt = _setup_mpl()
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches

    datasets = list(eval_summary.keys())
    if not datasets:
        print('[plot_mechanism_f1_bar] No datasets — skipping.')
        return output_path

    # Check which mechanisms have data
    mechs_with_data = [m for m in range(5)
                       if any(f'mech_{m}_f1' in eval_summary[d] for d in datasets)]
    if not mechs_with_data:
        print('[plot_mechanism_f1_bar] No mechanism F1 data — skipping.')
        return output_path

    n_ds = len(datasets)
    n_m  = len(mechs_with_data)
    bar_width = 0.8 / n_m
    if figsize is None:
        figsize = (max(8, n_ds * 2), 4)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_ds)

    for i, m_id in enumerate(mechs_with_data):
        f1_vals = [eval_summary[d].get(f'mech_{m_id}_f1', np.nan) * 100
                   for d in datasets]
        offset = (i - n_m / 2 + 0.5) * bar_width
        ax.bar(x + offset, f1_vals, width=bar_width * 0.9,
               color=_MECH_COLORS[m_id], label=_MECH_LABELS[m_id], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel('F1 (%)')
    ax.set_title('Per-Mechanism F1 by Dataset')
    ax.legend(ncol=n_m, loc='upper right', fontsize=8)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_mechanism_f1_bar] Saved → {output_path}')
    return output_path


# ─── Phi Blend Curve ──────────────────────────────────────────────────────

def plot_phi_blend_curve(
    training_csv: str,
    output_path: str = 'outputs/plots/phi_blend_curve.pdf',
    t_warmup: int = 100,
    tau_trans: float = 50.0,
    figsize: Tuple[float, float] = (8, 3.5),
) -> str:
    """Dual-axis: phi blend alpha (theoretical) and phi MSE loss (actual) vs step.

    The theoretical curve is drawn from the formula  sigmoid((t-t_warmup)/tau)
    using the stored config values.  The actual phi_loss is read from the CSV.
    """
    import csv as _csv

    plt = _setup_mpl()
    import matplotlib.pyplot as _plt

    steps, phi_losses = [], []
    with open(training_csv, newline='', encoding='utf-8') as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                steps.append(int(row['step']))
                phi_losses.append(float(row.get('condor/phi_loss', 'nan')))
            except (ValueError, KeyError):
                pass

    if not steps:
        print('[plot_phi_blend_curve] No training CSV data — skipping.')
        return output_path

    steps_arr = np.array(steps, dtype=float)
    phi_arr   = np.array(phi_losses, dtype=float)

    # Theoretical alpha curve
    alpha_arr = 1.0 / (1.0 + np.exp(-(steps_arr - t_warmup) / tau_trans))

    fig, ax1 = plt.subplots(figsize=figsize)
    color_alpha = '#4E79A7'
    color_phi   = '#E15759'

    ax1.plot(steps_arr, alpha_arr, color=color_alpha, linewidth=2,
             label=r'$\alpha_t$ (blend factor)')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel(r'Blend $\alpha_t$', color=color_alpha)
    ax1.tick_params(axis='y', labelcolor=color_alpha)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(0.5, linestyle=':', color=color_alpha, linewidth=0.8, alpha=0.5)

    ax2 = ax1.twinx()
    valid = ~np.isnan(phi_arr)
    if valid.sum() > 1:
        ax2.plot(steps_arr[valid], phi_arr[valid], color=color_phi, linewidth=1.5,
                 alpha=0.7, label='Phi MSE loss')
    ax2.set_ylabel('Phi MSE Loss', color=color_phi)
    ax2.tick_params(axis='y', labelcolor=color_phi)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    ax1.set_title('Phi Blend Factor and Phi Loss over Training')
    ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_phi_blend_curve] Saved → {output_path}')
    return output_path
