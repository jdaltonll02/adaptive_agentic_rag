"""CONDOR qa_manager/BaseAgent.py

Imports all baseline agents from baseline/qa_manager/BaseAgent4.py, then:
  - Overrides RetrievalAgent to dispatch to GraphRetriever (m3) or
    WebRetriever (m4) based on context['m_star'].
  - Overrides AgentPool to auto-register the extended RetrievalAgent.

When condor is not active (context['m_star'] not set or == 1/2), the
original FAISS HTTP-API path is taken unchanged.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Import all baseline agents by temporarily prepending baseline/qa_manager/
# ---------------------------------------------------------------------------
_BASELINE_QA = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline', 'qa_manager')
)
sys.path.insert(0, _BASELINE_QA)
try:
    from BaseAgent4 import (  # noqa: F401
        BaseAgent,
        ApiEnabledAgent,
        QueryRewriteAgent,
        QueryDecompositionAgentParallel,
        QueryDecompositionAgentSerial,
        DocumentSelectionAgent,
        AnswerGenerationAgent,
        AnswerSummarizationAgent,
        AgentPool as _BaselineAgentPool,
        RetrievalAgent as _BaselineRetrievalAgent,
    )
    # Also import config / tools used by baseline code
    from config import AGENT_CONFIG as _BASELINE_AGENT_CONFIG  # noqa: F401
    from config import EXAMPLE_PROMPT as _BASELINE_EXAMPLE_PROMPT  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"Could not import baseline BaseAgent4. "
        f"Checked path: {_BASELINE_QA}. Original error: {e}"
    )
finally:
    sys.path.pop(0)

# ---------------------------------------------------------------------------
# Our CONDOR config (superset of baseline)
# ---------------------------------------------------------------------------
from qa_manager.config import AGENT_CONFIG, EXAMPLE_PROMPT  # noqa: F401

# ---------------------------------------------------------------------------
# CONDOR-extended RetrievalAgent
# ---------------------------------------------------------------------------

class RetrievalAgent(_BaselineRetrievalAgent):
    """Extends baseline RetrievalAgent with mechanism-aware dispatch.

    - m3 → GraphRetriever  (knowledge-graph retrieval)
    - m4 → WebRetriever    (live web search)
    - all other m_star values → original FAISS HTTP API path
    """

    def run(self, context: dict) -> dict:
        m_star = context.get('m_star', 1)

        if m_star == 3:
            return self._run_graph(context)
        elif m_star == 4:
            return self._run_web(context)
        else:
            return super().run(context)

    # ------------------------------------------------------------------
    # m3: Graph RAG
    # ------------------------------------------------------------------
    def _run_graph(self, context: dict) -> dict:
        try:
            from retriever.graph_retriever import GraphRetriever
        except ImportError:
            # Fall back to standard retrieval if retriever not available
            return super().run(context)

        api_url = getattr(self, 'api_url', 'http://localhost:8000/search')
        num_results = getattr(self, 'num_results', 5)
        retriever = GraphRetriever(api_url=api_url, num_results=num_results)

        mode = context.get('mode', 'normal')
        if mode == 'normal':
            query = context.get('query', context.get('original_query', ''))
        else:
            step = context.get('current_step', 0)
            sub_queries = context.get('sub_query', [])
            query = sub_queries[step] if step < len(sub_queries) else context.get('query', '')

        docs = retriever.retrieve(query)
        if docs:
            context['results'] = docs
        context['token_cost']['RetrievalAgent'] += 1
        return context

    # ------------------------------------------------------------------
    # m4: Web Search RAG
    # ------------------------------------------------------------------
    def _run_web(self, context: dict) -> dict:
        try:
            from retriever.web_retriever import WebRetriever
        except ImportError:
            return super().run(context)

        num_results = getattr(self, 'num_results', 5)
        retriever = WebRetriever(num_results=num_results)

        mode = context.get('mode', 'normal')
        if mode == 'normal':
            query = context.get('query', context.get('original_query', ''))
        else:
            step = context.get('current_step', 0)
            sub_queries = context.get('sub_query', [])
            query = sub_queries[step] if step < len(sub_queries) else context.get('query', '')

        docs = retriever.retrieve(query)
        if docs:
            context['results'] = docs
        context['token_cost']['RetrievalAgent'] += 1
        return context


# ---------------------------------------------------------------------------
# CONDOR-extended AgentPool
# ---------------------------------------------------------------------------

class AgentPool(_BaselineAgentPool):
    """Singleton pool that registers the CONDOR-extended RetrievalAgent."""

    _instance = None
    _initialized = False

    def _initialize(self) -> None:
        """Auto-register all agents using CONDOR config."""
        from threading import Lock as _Lock
        agents = [
            QueryRewriteAgent(AGENT_CONFIG['QueryRewriteAgent']),
            QueryDecompositionAgentParallel(AGENT_CONFIG['QueryDecompositionAgentParallel']),
            QueryDecompositionAgentSerial(AGENT_CONFIG['QueryDecompositionAgentSerial']),
            RetrievalAgent(AGENT_CONFIG['RetrievalAgent']),           # CONDOR version
            DocumentSelectionAgent(AGENT_CONFIG['DocumentSelectionAgent']),
            AnswerGenerationAgent(AGENT_CONFIG['AnswerGenerationAgent']),
            AnswerSummarizationAgent(AGENT_CONFIG['AnswerSummarizationAgent']),
        ]
        for agent in agents:
            self._agents[agent.name] = agent
        self._initialized = True
