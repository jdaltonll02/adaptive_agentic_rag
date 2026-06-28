import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PhiScorer(nn.Module):
    """MLP counterfactual baseline for L0 mechanism selection.

    Input: concat(query_emb[:32], mechanism_onehot[5], qtype_onehot[4]) = 41 dims
    Output: scalar value estimate (predicted F1)

    The actual advantage baseline blends this MLP with the critic V_hi:
        alpha_t = sigmoid((t - t_warmup) / tau_trans)
        baseline = (1 - alpha_t) * phi(s) + alpha_t * V_hi(s)

    Early in training (t << t_warmup) the MLP dominates;
    late in training the critic takes over as it becomes accurate.
    """

    _INPUT_DIM = 32 + 5 + 4  # 41

    def __init__(
        self,
        hidden_dim: int = 64,
        t_warmup: int = 500,
        tau_trans: float = 100.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.t_warmup = t_warmup
        self.tau_trans = tau_trans
        self._t = 0

        self.mlp = nn.Sequential(
            nn.Linear(self._INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._optimizer = optim.Adam(self.parameters(), lr=lr)

    def get_blend_alpha(self) -> float:
        """Return alpha_t ∈ (0,1); near 0 early, near 1 late."""
        return 1.0 / (1.0 + math.exp(-(self._t - self.t_warmup) / self.tau_trans))

    def _encode(self, query_emb: np.ndarray, mechanism_id: int, qtype_id: int) -> torch.Tensor:
        emb = np.zeros(32, dtype=np.float32)
        n = min(len(query_emb), 32)
        emb[:n] = query_emb[:n]

        m_hot = np.zeros(5, dtype=np.float32)
        m_hot[int(mechanism_id)] = 1.0

        qt_hot = np.zeros(4, dtype=np.float32)
        qt_hot[int(qtype_id)] = 1.0

        x = np.concatenate([emb, m_hot, qt_hot])
        return torch.FloatTensor(x).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

    def score(self, query_emb: np.ndarray, mechanism_id: int, qtype_id: int) -> float:
        """Return scalar phi estimate (no grad)."""
        x = self._encode(query_emb, mechanism_id, qtype_id)
        with torch.no_grad():
            return float(self.forward(x).item())

    def blended_baseline(
        self,
        phi_val: float,
        critic_val: float,
    ) -> float:
        """Return (1-alpha)*phi + alpha*critic blended baseline."""
        alpha = self.get_blend_alpha()
        return (1.0 - alpha) * phi_val + alpha * critic_val

    def update(
        self,
        query_embs: List[np.ndarray],
        mechanism_ids: List[int],
        qtype_ids: List[int],
        targets: List[float],
    ) -> float:
        """Train PhiScorer on (query_emb, m, qtype) → actual_F1 targets."""
        self._t += 1
        if not query_embs:
            return 0.0
        x = torch.cat(
            [self._encode(qe, m, qt) for qe, m, qt in zip(query_embs, mechanism_ids, qtype_ids)],
            dim=0,
        )
        y = torch.FloatTensor(targets)
        self._optimizer.zero_grad()
        pred = self.forward(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self._optimizer.step()
        return float(loss.item())

    def state_dict_phi(self) -> dict:
        return {
            't': self._t,
            'mlp': self.mlp.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def load_state_dict_phi(self, d: dict) -> None:
        self._t = d['t']
        self.mlp.load_state_dict(d['mlp'])
        self._optimizer.load_state_dict(d['optimizer'])
