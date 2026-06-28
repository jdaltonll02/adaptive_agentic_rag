"""Process-level reward computations for CONDOR's two-level hierarchy."""

from condor.mechanism_router import MECHANISM_SELECTION_COST


class ProcessRewardModel:
    """Computes per-level reward signals.

    L0 reward (mechanism selection):
        r_hi = ΔF1(m*, π_lo) - lambda_hi * C_hi(m*)

    L1 cost signal (used to scale token/retrieval penalty):
        final_cost = coeff_token * token_cost
                   + coeff_retrieval * retrieval_api_cost
                   + coeff_turn * turn_latency
        → scaled by lambda_lo (instead of baseline's coeff_cost = 0.0)
    """

    # L1 coefficient signs match baseline's RewardManager
    COEFF_TOKEN = -1.0
    COEFF_RETRIEVAL = -0.25
    COEFF_TURN = -0.5

    def compute_l0_reward(
        self,
        f1: float,
        m_star: int,
        lambda_hi: float,
        baseline_f1: float = 0.0,
    ) -> float:
        """L0 reward = ΔF1 - lambda_hi * mechanism_selection_cost."""
        delta_f1 = f1 - baseline_f1
        c_hi = MECHANISM_SELECTION_COST.get(m_star, 0.0)
        return delta_f1 - lambda_hi * c_hi

    def compute_l1_cost(
        self,
        token_cost: float,
        retrieval_api_count: float,
        turn_latency: float,
    ) -> float:
        """Raw (unsigned) L1 cost before lambda_lo scaling."""
        return (
            self.COEFF_TOKEN * token_cost
            + self.COEFF_RETRIEVAL * retrieval_api_count
            + self.COEFF_TURN * turn_latency
        )

    def scale_l1_cost(self, raw_cost: float, lambda_lo: float) -> float:
        """Apply adaptive lambda_lo to the raw L1 cost signal."""
        return raw_cost * lambda_lo

    def l0_correct(self, f1: float, threshold: float = 0.4) -> float:
        """Binary indicator: was the mechanism selection effective?"""
        return 1.0 if f1 >= threshold else 0.0
