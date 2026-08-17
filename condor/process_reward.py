"""Process-level reward computations for CONDOR's two-level hierarchy."""

from condor.mechanism_router import MECHANISM_SELECTION_COST

# Mechanism-specific reward weight vectors (spec Section 3.3 table)
MECHANISM_WEIGHTS = {
    0: dict(w_doc=0.0, w_ans=0.6, w_reas=0.3, w_ver=0.1),  # Pure LLM
    1: dict(w_doc=0.4, w_ans=0.5, w_reas=0.0, w_ver=0.1),  # Std RAG
    2: dict(w_doc=0.3, w_ans=0.4, w_reas=0.2, w_ver=0.1),  # Adv RAG
    3: dict(w_doc=0.3, w_ans=0.4, w_reas=0.1, w_ver=0.2),  # Graph RAG
    4: dict(w_doc=0.2, w_ans=0.5, w_reas=0.0, w_ver=0.3),  # Web RAG
}


class ProcessRewardModel:
    """Computes per-level reward signals.

    L1 process reward (spec Eq. 3.3, goal-conditioned):
        r_proc = w_doc * doc_rel + w_ans * partial_f1
               + w_reas * reason_coh  (if CoT/NRA active)
               + w_ver  * ver_score   (if AV/CC active)
               - w_fmt  * R_FP

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
    W_FMT = 0.1

    def compute_process_reward(
        self,
        m_star: int,
        partial_f1: float,
        doc_relevance: float = 0.0,
        reasoning_coherence: float = 0.0,
        verification_score: float = 0.0,
        cot_active: bool = False,
        ver_active: bool = False,
        format_penalty: float = 0.0,
    ) -> float:
        """Goal-conditioned process reward (spec Eq. 3.3).

        Args:
            m_star: selected mechanism id ∈ {0,1,2,3,4}
            partial_f1: PartialF1(sub_answer, gold)
            doc_relevance: DocRel(D_t, q, m*); 0.0 if no retrieval
            reasoning_coherence: ReasonCoh(chain_t); only used when cot_active
            verification_score: VerScore(a_t, D_t); only used when ver_active
            cot_active: True when CoT or NRA executor is active
            ver_active: True when AV or CC executor is active
            format_penalty: R_FP(t) format penalty signal
        """
        w = MECHANISM_WEIGHTS[m_star]
        r = w['w_doc'] * doc_relevance
        r += w['w_ans'] * partial_f1
        r += w['w_reas'] * (reasoning_coherence if cot_active else 0.0)
        r += w['w_ver'] * (verification_score if ver_active else 0.0)
        r -= self.W_FMT * format_penalty
        return r

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
