"""Figure 3 equivalent — Pareto scatter plots.

Three subplots:
  (a) F1 vs Token Cost  (milli-USD)
  (b) F1 vs Retrieval Calls
  (c) F1 vs Total Turns

Each subplot shows CONDOR per-query points (coloured by mechanism) plus the
three reference baselines as large starred markers.

Usage
-----
    from condor.plots.pareto import plot_pareto
    plot_pareto(per_query_records, output_path='outputs/plots/fig3_pareto.pdf')

Or call from generate_all.py which reads the JSONL eval file.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── matplotlib style ────────────────────────────────────────────────────────

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


# ── Baseline reference anchors (Avg across datasets) ────────────────────────

_BASELINES: Dict[str, Tuple[float, float, float, float]] = {
    # name: (F1%, token_cost_mUSD, retrieval_calls, turns)
    'LLM w/o RAG': (36.23, 0.086, 0.0, 1.0),
    'Vanilla RAG':  (43.94, 0.262, 1.0, 1.0),
    'MAO-ARAG':     (52.91, 1.129, 2.327, 2.899),
}

_MECH_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
_MECH_LABELS = [
    'm0 Pure LLM', 'm1 Std RAG',
    'm2 Adv Iter RAG', 'm3 Graph RAG', 'm4 Web RAG',
]
_BASELINE_MARKERS = ['*', 'D', 's']
_BASELINE_COLORS  = ['#9C755F', '#BAB0AC', '#B07AA1']


def plot_pareto(
    per_query_records: List[Dict],
    output_path: str = 'outputs/plots/fig3_pareto.pdf',
    dataset_label: str = '',
    figsize: Tuple[float, float] = (14, 4),
) -> str:
    """Generate a three-panel Pareto scatter figure (Figure 3 equivalent).

    Parameters
    ----------
    per_query_records : list[dict]
        Each dict must contain at minimum: 'f1', 'token_cost_usd',
        'retrieval_calls', 'turns', 'm_star'.
    output_path : str
        Where to save the figure (PDF / PNG; extension determines format).
    dataset_label : str
        Shown in the suptitle (e.g. 'NQ').
    """
    plt = _setup_mpl()
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches

    # Extract arrays
    records = [r for r in per_query_records
               if r.get('f1') is not None and r.get('token_cost_usd') is not None]
    if not records:
        print('[plot_pareto] No valid records — skipping.')
        return output_path

    f1_arr    = np.array([r['f1'] * 100 for r in records])
    cost_arr  = np.array([r.get('token_cost_usd', 0) * 1000 for r in records])
    ret_arr   = np.array([r.get('retrieval_calls', 0) for r in records])
    turn_arr  = np.array([r.get('turns', 1) for r in records])
    mech_arr  = np.array([r.get('m_star', 1) for r in records])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = list(axes)

    pairs = [
        (axes[0], cost_arr,  'Token Cost (milli-USD)', '(a)'),
        (axes[1], ret_arr,   'Retrieval Calls',         '(b)'),
        (axes[2], turn_arr,  'Total Turns',             '(c)'),
    ]

    for ax, x_arr, x_label, panel_label in pairs:
        # CONDOR scatter — per mechanism
        for m_id in range(5):
            mask = mech_arr == m_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                x_arr[mask], f1_arr[mask],
                c=_MECH_COLORS[m_id], label=_MECH_LABELS[m_id],
                alpha=0.55, s=20, edgecolors='none',
            )

        # Pareto frontier over CONDOR points
        _draw_pareto_frontier(ax, x_arr, f1_arr)

        # Baseline anchors
        for i, (b_name, (b_f1, b_cost, b_ret, b_turn)) in enumerate(_BASELINES.items()):
            x_ref = {'Token Cost (milli-USD)': b_cost,
                     'Retrieval Calls': b_ret,
                     'Total Turns': b_turn}[x_label]
            ax.scatter(
                x_ref, b_f1,
                marker=_BASELINE_MARKERS[i],
                color=_BASELINE_COLORS[i],
                s=140, zorder=5, linewidths=0.8, edgecolors='k',
                label=b_name,
            )
            ax.annotate(
                b_name, (x_ref, b_f1),
                textcoords='offset points', xytext=(4, 4),
                fontsize=7.5, color='#333333',
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel('F1 (%)' if ax is axes[0] else '')
        ax.set_title(panel_label)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Shared legend in last panel
    mech_patches = [mpatches.Patch(color=_MECH_COLORS[i], label=_MECH_LABELS[i])
                    for i in range(5)]
    axes[-1].legend(handles=mech_patches, loc='lower right', fontsize=8)

    suptitle = 'Figure 3: Pareto Efficiency'
    if dataset_label:
        suptitle += f' — {dataset_label}'
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_pareto] Saved → {output_path}')
    return output_path


def _draw_pareto_frontier(ax, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
    """Draw the empirical Pareto frontier on ax (lower-x + higher-y optimal)."""
    # Sort by x ascending; keep only non-dominated (max y seen so far)
    order = np.argsort(x_arr)
    xs, ys = x_arr[order], y_arr[order]
    pf_x, pf_y = [], []
    best_y = -np.inf
    for xi, yi in zip(xs, ys):
        if yi > best_y:
            pf_x.append(xi)
            pf_y.append(yi)
            best_y = yi
    ax.step(pf_x, pf_y, where='post', color='crimson',
            linewidth=1.6, linestyle='--', label='Pareto frontier', zorder=4)
