"""Level-0 mechanism router for CONDOR.

Selects one of 5 retrieval mechanisms before the L1 PlanningAgent runs:
  m0 = Pure LLM          (no retrieval)
  m1 = Standard RAG      (FAISS HTTP API)
  m2 = Advanced Iterative RAG  (decomposition + iterative retrieval)
  m3 = Graph RAG         (knowledge graph retrieval)
  m4 = Web Search RAG    (live web search)

Exploration: epsilon-greedy with rule-based prior.
Exploitation: lightweight MLP policy updated via REINFORCE.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Action-space definitions
# ---------------------------------------------------------------------------

MECHANISM_ACTION_SPACES = {
    0: ['AG'],
    1: ['QR', 'R', 'DS', 'AG'],
    2: ['QR', 'QDP', 'QDS', 'R', 'DS', 'AG'],
    3: ['QR', 'R', 'DS', 'AG'],
    4: ['QR', 'R', 'DS', 'AG'],
}

MECHANISM_NAMES = {
    0: 'Pure_LLM',
    1: 'Standard_RAG',
    2: 'Advanced_Iterative_RAG',
    3: 'Graph_RAG',
    4: 'Web_Search_RAG',
}

# Fixed per-mechanism selection cost (used in L0 reward)
MECHANISM_SELECTION_COST = {
    0: 0.00,
    1: 0.01,
    2: 0.02,
    3: 0.03,
    4: 0.05,
}

QTYPE_LABELS = {0: 'factoid', 1: 'multi_hop', 2: 'comparison', 3: 'complex'}

# Rule-based prior P(m | query) — spec Eq. 20.
# Resolved from raw query-text surface features, not qtype labels.
# Prior(m | q) given as [m0_Pure_LLM, m1_Std_RAG, m2_Adv_RAG, m3_Graph_RAG, m4_Web_RAG]
_TEMPORAL_KWS  = {'current', 'today', '2024', '2025', 'latest', 'now'}
_RELATIONAL_KWS = {'connected', 'related', 'between', 'relationship'}


def _rule_based_prior_from_text(question: str) -> np.ndarray:
    """Return P(m|q) as a 5-element array using spec Eq. 20 surface rules."""
    q = question.lower()
    words = set(q.split())
    temporal  = bool(_TEMPORAL_KWS & words)
    relational = bool(_RELATIONAL_KWS & words)
    short     = len(question.split()) < 9
    if temporal:
        return np.array([0.05, 0.10, 0.15, 0.15, 0.55], dtype=np.float32)
    if relational:
        return np.array([0.05, 0.15, 0.20, 0.55, 0.05], dtype=np.float32)
    if short:
        return np.array([0.65, 0.20, 0.08, 0.04, 0.03], dtype=np.float32)
    return np.array([0.10, 0.35, 0.35, 0.10, 0.10], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def classify_qtype(question: str) -> int:
    """Map a question string to a qtype index in {0,1,2,3}."""
    q = question.lower()
    comparison_kw = ['versus', 'compare', 'difference between', 'better than',
                     'more than', 'less than', 'which is', 'both', 'similar to']
    complex_kw = ['analyze', 'explain', 'why', 'how does', 'what causes',
                  'impact of', 'effect of', 'relationship between', 'describe']
    multi_hop_kw = ['who was', 'what was', 'born in', 'died in', 'third',
                    'second', 'first', 'before', 'after', 'then', 'and then',
                    'directed by', 'written by', 'founded by']
    if any(kw in q for kw in comparison_kw):
        return 2
    if any(kw in q for kw in complex_kw):
        return 3
    if any(kw in q for kw in multi_hop_kw):
        return 1
    return 0


def query_to_features(question: str) -> np.ndarray:
    """Convert a question to a 32-dim float feature vector for the L0 policy."""
    features = np.zeros(32, dtype=np.float32)
    q = question.lower()
    words = q.split()

    # 0-9: interrogative keyword flags
    kws = ['who', 'what', 'when', 'where', 'which', 'how', 'why',
           'compare', 'versus', 'analyze']
    for i, kw in enumerate(kws):
        features[i] = 1.0 if kw in q else 0.0

    # 10: normalised question length
    features[10] = min(len(words) / 30.0, 1.0)
    # 11: contains digit
    features[11] = 1.0 if any(c.isdigit() for c in question) else 0.0

    # 12-19: multi-word phrase flags
    phrases = [
        'born in', 'died in', 'president of', 'city of',
        'capital of', 'known for', 'made by', 'written by',
    ]
    for i, ph in enumerate(phrases):
        features[12 + i] = 1.0 if ph in q else 0.0

    # 20-31: character-level hash micro-features
    for j, ch in enumerate(q[:12]):
        features[20 + j] = float(ord(ch) % 10) / 10.0

    return features


# ---------------------------------------------------------------------------
# Learned L0 policy (lightweight MLP)
# ---------------------------------------------------------------------------

class _MechanismPolicy(nn.Module):
    def __init__(self, input_dim: int = 32, n_mechanisms: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_mechanisms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MechanismRouter
# ---------------------------------------------------------------------------

class MechanismRouter:
    """Epsilon-greedy L0 mechanism selector.

    epsilon decays exponentially from epsilon_start → epsilon_end over
    epsilon_decay steps.  At each step:
      - With prob epsilon: sample m from rule-based prior (exploration).
      - Otherwise: sample m from learned policy softmax (exploitation).

    The log-prob of the selected m is stored for a later REINFORCE update.
    """

    def __init__(
        self,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 1000.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_condor: bool = True,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.use_condor = use_condor
        self._step = 0

        self._policy = _MechanismPolicy()
        self._optimizer = optim.Adam(self._policy.parameters(), lr=lr)

        # Buffers for REINFORCE update
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self._step / max(self.epsilon_decay, 1.0))

    def route(self, question: str) -> Tuple[int, int, np.ndarray]:
        """Select a mechanism for one question.

        Returns:
            m_star   : selected mechanism id ∈ {0,1,2,3,4}
            qtype_id : query-type label ∈ {0,1,2,3}
            features : 32-dim feature vector used by PhiScorer
        """
        if not self.use_condor:
            return 1, 0, np.zeros(32, dtype=np.float32)

        qtype_id = classify_qtype(question)
        features = query_to_features(question)

        x = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            logits = self._policy(x)

        if np.random.random() < self.epsilon:
            prior = _rule_based_prior_from_text(question)
            prior /= prior.sum()
            m_star = int(np.random.choice(5, p=prior))
        else:
            probs = torch.softmax(logits, dim=-1)
            m_star = int(torch.multinomial(probs, 1).item())

        # Recompute with grad for the chosen action
        logits_grad = self._policy(x)
        lp = torch.log_softmax(logits_grad, dim=-1)[0, m_star]
        self._log_probs.append(lp)

        self._step += 1
        return m_star, qtype_id, features

    def store_reward(self, reward: float) -> None:
        """Register an L0 reward signal for the most recent route() call."""
        self._rewards.append(reward)

    def update_policy(self, gamma: float = 0.99) -> float:
        """REINFORCE update over accumulated (log_prob, reward) pairs.

        Returns the scalar policy loss (0.0 if no data).
        """
        n = min(len(self._log_probs), len(self._rewards))
        if n == 0:
            return 0.0

        log_probs = self._log_probs[:n]
        rewards = self._rewards[:n]

        # Discounted returns
        returns: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        ret_t = torch.FloatTensor(returns)
        if ret_t.std() > 1e-8:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        loss = -torch.stack([lp * R for lp, R in zip(log_probs, ret_t)]).sum()
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
        self._optimizer.step()

        self._log_probs.clear()
        self._rewards.clear()
        return float(loss.item())

    def state_dict_router(self) -> dict:
        return {
            'step': self._step,
            'policy': self._policy.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def load_state_dict_router(self, d: dict) -> None:
        self._step = d['step']
        self._policy.load_state_dict(d['policy'])
        self._optimizer.load_state_dict(d['optimizer'])
