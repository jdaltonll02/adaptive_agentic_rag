"""2×2 failure taxonomy via K-Means on CONDOR trajectory features.

Cluster axes:
  x: cost efficiency   (low cost vs high cost)
  y: answer quality    (low F1  vs high F1)

Resulting taxonomy labels:
  efficient_correct    – ideal: cheap and accurate
  efficient_incorrect  – wrong mechanism but cheap; explore harder
  expensive_correct    – accurate but over-budget; optimise L1
  expensive_incorrect  – worst case; mechanism and workflow both failing
"""

from typing import Dict, Optional

import numpy as np

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class ClusterReport:
    """Run K-Means on trajectory feature matrices and emit a taxonomy report."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans: Optional[object] = None

    def analyze(self, feature_matrix: np.ndarray) -> Dict:
        """Cluster trajectories and return per-cluster taxonomy dict.

        Args:
            feature_matrix: (N, 35) array from TrajectoryLogger.get_feature_matrix()

        Returns:
            dict keyed 'cluster_0' … 'cluster_{n_clusters-1}', each with
            keys: taxonomy, size, avg_f1, avg_cost, avg_l0_correct.
        """
        if not _SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not installed; run pip install scikit-learn'}
        if feature_matrix.shape[0] < self.n_clusters:
            return {'error': f'need ≥ {self.n_clusters} trajectories, got {feature_matrix.shape[0]}'}

        # Normalise columns 0-3 (scalar features) before clustering
        mat = feature_matrix.copy()
        for col in range(4):
            col_std = mat[:, col].std()
            if col_std > 1e-8:
                mat[:, col] = (mat[:, col] - mat[:, col].mean()) / col_std

        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        labels = self._kmeans.fit_predict(mat)

        # Compute per-cluster stats using unnormalised values
        raw = feature_matrix
        report = {}
        for k in range(self.n_clusters):
            mask = labels == k
            cluster = raw[mask]
            if len(cluster) == 0:
                report[f'cluster_{k}'] = {'taxonomy': 'empty', 'size': 0}
                continue

            avg_f1 = float(cluster[:, 1].mean())
            avg_cost = float(cluster[:, 2].mean())
            avg_l0 = float(cluster[:, 0].mean())
            avg_turns = float(cluster[:, 3].mean())

            # Global medians as thresholds
            f1_thresh = float(raw[:, 1].median()) if hasattr(raw[:, 1], 'median') else float(np.median(raw[:, 1]))
            cost_thresh = float(raw[:, 2].median()) if hasattr(raw[:, 2], 'median') else float(np.median(raw[:, 2]))

            high_f1 = avg_f1 >= f1_thresh
            high_cost = avg_cost >= cost_thresh

            if high_f1 and not high_cost:
                taxonomy = 'efficient_correct'
            elif high_f1 and high_cost:
                taxonomy = 'expensive_correct'
            elif not high_f1 and not high_cost:
                taxonomy = 'efficient_incorrect'
            else:
                taxonomy = 'expensive_incorrect'

            report[f'cluster_{k}'] = {
                'taxonomy': taxonomy,
                'size': int(mask.sum()),
                'avg_f1': round(avg_f1, 4),
                'avg_cost': round(avg_cost, 4),
                'avg_l0_correct': round(avg_l0, 4),
                'avg_turns': round(avg_turns, 4),
            }

        return report

    def wandb_summary(self, report: Dict) -> Dict:
        """Flatten report into W&B-compatible flat dict."""
        out = {}
        for key, val in report.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (int, float)):
                        out[f'condor/cluster/{key}/{sub_key}'] = sub_val
                    else:
                        out[f'condor/cluster/{key}/taxonomy_label'] = str(sub_val)
        return out
