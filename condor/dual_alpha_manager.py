import math


class DualAlphaManager:
    """Adam-stabilised dual gradient descent for a single Lagrange multiplier.

    Instantiate once per level:
        dual_hi = DualAlphaManager(budget=0.0005, lr=5e-4, lam_init=0.10)
        dual_lo = DualAlphaManager(budget=0.0004, lr=1e-3, lam_init=0.08)

    Update rule (gradient ascent on the Lagrangian, spec Eq. 14–15):
        g = avg_cost - budget          (positive → constraint violated)
        m ← beta1*m + (1-beta1)*g
        v ← beta2*v + (1-beta2)*g²
        λ ← clip(λ + lr * m̂ / (√v̂ + eps), 0, lam_max)
    """

    def __init__(
        self,
        budget: float,
        lr: float = 1e-3,
        lam_init: float = 0.1,
        lam_max: float = 5.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.d = budget
        self.lr = lr
        self.lam = lam_init
        self.lam_max = lam_max
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self._m = 0.0
        self._v = 0.0
        self._t = 0

    def update(self, mean_cost: float) -> float:
        """Dual gradient ascent step. Returns updated lambda."""
        g = mean_cost - self.d
        self._t += 1
        self._m = self.b1 * self._m + (1 - self.b1) * g
        self._v = self.b2 * self._v + (1 - self.b2) * g * g
        m_hat = self._m / (1.0 - self.b1 ** self._t)
        v_hat = self._v / (1.0 - self.b2 ** self._t)
        self.lam += self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        self.lam = max(0.0, min(self.lam, self.lam_max))
        return self.lam

    @property
    def value(self) -> float:
        return self.lam

    def state_dict(self) -> dict:
        return {
            'lam': self.lam,
            't': self._t,
            'm': self._m,
            'v': self._v,
        }

    def load_state_dict(self, d: dict) -> None:
        self.lam = d['lam']
        self._t = d['t']
        self._m = d['m']
        self._v = d['v']
