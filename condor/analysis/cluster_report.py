"""2×2 failure taxonomy via K-Means on CONDOR trajectory features.

Cluster axes (matching spec Section 4):
  x: L0 routing accuracy  (L0acc <= 0.7  vs  L0acc > 0.7)
  y: answer quality        (F1   <= 0.6   vs  F1   > 0.6)

Taxonomy labels (fixed thresholds F1>0.6, L0acc>0.7):
  success_l0_l1_correct   – L0 and L1 both correct
  l1_workflow_failure      – L0 correct but L1 workflow failed
  l0_wrong_l1_recovered    – L0 wrong, L1 recovered anyway
  systemic_failure         – both L0 and L1 wrong
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

            avg_l0 = float(cluster[:, 0].mean())   # L0 routing accuracy
            avg_f1 = float(cluster[:, 1].mean())
            avg_cost = float(cluster[:, 2].mean())
            avg_turns = float(cluster[:, 3].mean())

            # Fixed thresholds from spec (Section 4 / Algorithm 2)
            high_f1 = avg_f1 > 0.6
            high_l0 = avg_l0 > 0.7

            if high_f1 and high_l0:
                taxonomy = 'success_l0_l1_correct'
            elif high_l0 and not high_f1:
                taxonomy = 'l1_workflow_failure'
            elif high_f1 and not high_l0:
                taxonomy = 'l0_wrong_l1_recovered'
            else:
                taxonomy = 'systemic_failure'

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
