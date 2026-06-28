import math
from typing import Tuple


class DualAlphaManager:
    """Adam-stabilised dual gradient descent for Lagrange multipliers lambda_hi and lambda_lo.

    lambda_hi controls the L0 mechanism-selection budget constraint.
    lambda_lo controls the L1 token/retrieval cost budget constraint
    (replaces the hard-coded coeff_cost=0.0 in the baseline RewardManager).

    Update rule (gradient ascent on the Lagrangian):
        g = avg_cost - budget          (positive → constraint violated)
        m ← beta1*m + (1-beta1)*g
        v ← beta2*v + (1-beta2)*g²
        λ ← clip(λ + lr * m̂ / (√v̂ + eps), λ_min, λ_max)
    """

    def __init__(
        self,
        lambda_hi_init: float = 0.1,
        lambda_lo_init: float = 0.0,
        budget_hi: float = 0.5,
        budget_lo: float = 0.3,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lambda_min: float = 0.0,
        lambda_max: float = 10.0,
    ):
        self.lambda_hi = lambda_hi_init
        self.lambda_lo = lambda_lo_init
        self.budget_hi = budget_hi
        self.budget_lo = budget_lo
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Adam first/second moment accumulators
        self._m_hi = 0.0
        self._v_hi = 0.0
        self._m_lo = 0.0
        self._v_lo = 0.0
        self._step = 0

    def update(self, avg_cost_hi: float, avg_cost_lo: float) -> Tuple[float, float]:
        """Compute one Adam-stabilised step and return (lambda_hi, lambda_lo)."""
        self._step += 1

        # Constraint violation signal: positive means over-budget
        g_hi = avg_cost_hi - self.budget_hi
        g_lo = avg_cost_lo - self.budget_lo

        self.lambda_hi = self._adam_step(
            self.lambda_hi, g_hi,
            '_m_hi', '_v_hi',
        )
        self.lambda_lo = self._adam_step(
            self.lambda_lo, g_lo,
            '_m_lo', '_v_lo',
        )

        return self.lambda_hi, self.lambda_lo

    def _adam_step(self, lam: float, g: float, m_attr: str, v_attr: str) -> float:
        m = getattr(self, m_attr)
        v = getattr(self, v_attr)
        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * g * g
        setattr(self, m_attr, m)
        setattr(self, v_attr, v)
        m_hat = m / (1.0 - self.beta1 ** self._step)
        v_hat = v / (1.0 - self.beta2 ** self._step)
        lam = lam + self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return max(self.lambda_min, min(self.lambda_max, lam))

    def state_dict(self) -> dict:
        return {
            'lambda_hi': self.lambda_hi,
            'lambda_lo': self.lambda_lo,
            'step': self._step,
            'm_hi': self._m_hi,
            'v_hi': self._v_hi,
            'm_lo': self._m_lo,
            'v_lo': self._v_lo,
        }

    def load_state_dict(self, d: dict) -> None:
        self.lambda_hi = d['lambda_hi']
        self.lambda_lo = d['lambda_lo']
        self._step = d['step']
        self._m_hi = d['m_hi']
        self._v_hi = d['v_hi']
        self._m_lo = d['m_lo']
        self._v_lo = d['v_lo']
