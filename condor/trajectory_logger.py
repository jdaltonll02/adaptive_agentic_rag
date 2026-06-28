"""Trajectory logger for CONDOR failure taxonomy and clustering."""

from typing import List, Optional
import numpy as np


class TrajectoryLogger:
    """Stores (L0_correct, F1, cost, n_turns, query_emb[:32]) tuples.

    Used by ClusterReport to build the 2×2 failure taxonomy via K-Means
    on the 35-dimensional feature space.

    Buffer is bounded at max_size entries (FIFO eviction).
    """

    FEATURE_DIM = 36  # l0_correct + f1 + cost + n_turns + emb32 = 4 + 32

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._trajectories: List[dict] = []

    def log(
        self,
        l0_correct: float,
        f1: float,
        cost: float,
        n_turns: int,
        query_features: np.ndarray,
        m_star: Optional[int] = None,
        qtype_id: Optional[int] = None,
        question: Optional[str] = None,
    ) -> None:
        """Append one trajectory entry."""
        emb32 = np.zeros(32, dtype=np.float32)
        q = query_features if query_features is not None else np.zeros(32, dtype=np.float32)
        n = min(len(q), 32)
        emb32[:n] = q[:n]

        entry = {
            'l0_correct': float(l0_correct),
            'f1': float(f1),
            'cost': float(cost),
            'n_turns': float(n_turns),
            'emb32': emb32,
            'm_star': m_star,
            'qtype_id': qtype_id,
            'question': question,
        }
        self._trajectories.append(entry)
        if len(self._trajectories) > self.max_size:
            self._trajectories.pop(0)

    def get_feature_matrix(self) -> np.ndarray:
        """Return (N, 35) matrix: [l0_correct, f1, cost, n_turns, emb32]."""
        if not self._trajectories:
            return np.zeros((0, self.FEATURE_DIM), dtype=np.float32)
        rows = []
        for t in self._trajectories:
            row = np.concatenate([
                [t['l0_correct'], t['f1'], t['cost'], t['n_turns']],
                t['emb32'],
            ])
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def get_summary(self) -> dict:
        """Return mean statistics over stored trajectories."""
        if not self._trajectories:
            return {}
        mat = self.get_feature_matrix()
        return {
            'count': len(self._trajectories),
            'avg_l0_correct': float(mat[:, 0].mean()),
            'avg_f1': float(mat[:, 1].mean()),
            'avg_cost': float(mat[:, 2].mean()),
            'avg_n_turns': float(mat[:, 3].mean()),
        }

    def __len__(self) -> int:
        return len(self._trajectories)
