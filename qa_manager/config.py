"""CONDOR qa_manager config.

Extends the baseline config with:
  - MECHANISM_ACTION_SPACES: per-mechanism valid executor abbreviations
  - CONDOR_AGENT_CONFIG: GraphRAGAgent, WebSearchRAGAgent entries
  - All baseline AGENT_CONFIG entries are preserved unchanged.
"""

import os
import sys

# ------------------------------------------------------------------
# Mirror the baseline API settings (can be overridden via env vars)
# ------------------------------------------------------------------
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai-proxy.org/v1")

COMMON_CONFIG = {
    'api_key': API_KEY,
    'api_base': API_BASE,
}

# ------------------------------------------------------------------
# Baseline agent configs (identical to baseline/qa_manager/config.py)
# ------------------------------------------------------------------
AGENT_CONFIG = {
    "QueryRewriteAgent": {
        **COMMON_CONFIG,
        'name': 'QueryRewriteAgent',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
    },
    "QueryDecompositionAgentParallel": {
        **COMMON_CONFIG,
        'name': 'QueryDecompositionAgentParallel',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
    },
    "QueryDecompositionAgentSerial": {
        **COMMON_CONFIG,
        'name': 'QueryDecompositionAgentSerial',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
    },
    "DocumentSelectionAgent": {
        **COMMON_CONFIG,
        'name': 'DocumentSelectionAgent',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
    },
    "AnswerGenerationAgent": {
        **COMMON_CONFIG,
        'name': 'AnswerGenerationAgent',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
    },
    "RetrievalAgent": {
        **COMMON_CONFIG,
        'name': 'RetrievalAgent',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
        'api_url': os.environ.get("RETRIEVAL_API_URL", "http://localhost:8000/search"),
        'num_results': 5,
    },
    "AnswerSummarizationAgent": {
        **COMMON_CONFIG,
        'name': 'AnswerSummarizationAgent',
        'model': 'llama-3.1-8b-instant',
        'temperature': 0,
        'max_tokens': 256,
        'timeout': 300,
        'api_url': os.environ.get("RETRIEVAL_API_URL", "http://localhost:8000/search"),
        'num_results': 5,
    },
}

# ------------------------------------------------------------------
# EXAMPLE_PROMPT (identical to baseline)
# ------------------------------------------------------------------
EXAMPLE_PROMPT = '''
- Example:
Question: When did the simpsons first air on television?
Answer: December 17, 1989

Question: When did the lightning thief book come out?
Answer: 2005

Question: Who said i'm late i'm late for a very important date?
Answer: The White Rabbit

Question: Where does the short happy life of francis macomber take place?
Answer: Africa

Question: What was the fourth expansion pack for sims 2?
Answer: Pets

Question: Voice of the snake in the jungle book?
Answer: The Jungle Book (2016 film)

Question: How many seasons are there of star wars the clone wars?
Answer: 6

Question: Which us president appears as a character in the play annie?
Answer: Franklin D. Roosevelt

Question: Are Calochone and Adlumia both plants?
Answer: yes

Question: Yukio Mishima and Roberto Bolaño, are Chilean?
Answer: no
'''

# ------------------------------------------------------------------
# CONDOR: mechanism action-space mapping
# ------------------------------------------------------------------
MECHANISM_ACTION_SPACES = {
    0: ['AG'],                                   # m0: Pure LLM
    1: ['QR', 'R', 'DS', 'AG'],                 # m1: Standard RAG
    2: ['QR', 'QDP', 'QDS', 'R', 'DS', 'AG'],  # m2: Advanced Iterative RAG
    3: ['QR', 'R', 'DS', 'AG'],                 # m3: Graph RAG
    4: ['QR', 'R', 'DS', 'AG'],                 # m4: Web Search RAG
}

MECHANISM_GOAL_DESCRIPTIONS = {
    0: "Pure LLM (no retrieval): answer directly from parametric knowledge.",
    1: "Standard RAG: retrieve from local FAISS index then generate.",
    2: "Advanced Iterative RAG: decompose into sub-queries, retrieve iteratively.",
    3: "Graph RAG: retrieve via knowledge-graph entity traversal.",
    4: "Web Search RAG: retrieve via live web search.",
}
