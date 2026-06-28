"""CONDOR PlanningAgent.py

Goal-conditioned extension of the baseline PlanningAgent.

When goal=(m_star, qtype_id) is provided to create_planning_messages():
  1. Restricts the listed tool/agent descriptions to those valid for m_star
     (as defined by MECHANISM_ACTION_SPACES).
  2. Prepends a goal-context prefix to the system message so the LLM planner
     knows which mechanism it is working within.
  3. Falls back to full agent list when goal=None (baseline behaviour).

The "Qustion" typo in the last user message is intentional and preserved
from the baseline to keep token-level continuity with the SFT checkpoint.
"""

from typing import Dict, List, Optional, Tuple

from qa_manager.config import MECHANISM_ACTION_SPACES, MECHANISM_GOAL_DESCRIPTIONS


# Full agent descriptions (superset; subset selected per mechanism)
_AGENT_DESCRIPTIONS = {
    'QR': "Query Rewriter (QR): The input of QR is a question, and the output of QR is a "
          "rewritten question which is more concise, clearer, and more accurate.",
    'QDP': "Query Decomposition Parallel (QDP): The input of QDP is a question, and the output "
           "of QDP contains multiple independent sub-questions. And QDP is suitable for "
           "decomposing those multi-hop questions that can be decomposed into multiple searchable "
           "and independent sub-questions in parallel.",
    'QDS': "Query Decomposition Serial (QDS): The input of QDS is a question. The output of QDS "
           "is multiple related sub-questions, and later sub-questions may need answers to previous "
           "sub-questions before they become searchable. And QDS is suitable for decomposing those "
           "multi-hop questions with dependencies into multiple sub-questions.",
    'R': "Retrieval (R): The input of R is a question, and the output of R is multiple relevant "
         "candidate documents to the given question.",
    'DS': "Document Selector (DS): The input of DS is a given question and multiple relevant "
          "candidate documents from search engine, and the output of DS is the documents which "
          "are helpful to answer the given question.",
    'AG': "Answer Generator (AG): The input of G may contains the given question, the some related "
          "candidate documents from search engine. The output of AG is the answer to the given question.",
}

_SUB_AGENT_DESCRIPTIONS = {
    'R': _AGENT_DESCRIPTIONS['R'],
    'DS': _AGENT_DESCRIPTIONS['DS'],
    'AG': _AGENT_DESCRIPTIONS['AG'],
}


class PlanningAgent:
    def __init__(self):
        pass

    def create_planning_messages(
        self,
        question: str,
        is_sub: bool = False,
        goal: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, str]]:
        """Build the 5-message planning prompt.

        Args:
            question: the question to plan a workflow for.
            is_sub:   True when planning a sub-query workflow.
            goal:     (m_star, qtype_id) tuple from the L0 MechanismRouter.
                      When provided, restricts the agent list and adds a
                      goal-context prefix. None → baseline behaviour.
        """
        if is_sub:
            return self._sub_messages(question)

        if goal is not None:
            return self._goal_conditioned_messages(question, goal)

        return self._baseline_messages(question)

    # ------------------------------------------------------------------
    # Baseline (goal=None) – exact replica of baseline PlanningAgent
    # ------------------------------------------------------------------

    def _baseline_messages(self, question: str) -> List[Dict[str, str]]:
        agent_desc = "\n\n    ".join(
            _AGENT_DESCRIPTIONS[k] for k in ['QR', 'QDP', 'QDS', 'R', 'DS', 'AG']
        )
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant to plan a Workflow for the given information using the give tool/agent.",
            },
            {
                "role": "assistant",
                "content": "Ok, please show me you requirement information and the available tool/agent",
            },
            {
                "role": "user",
                "content": f"""The descriptions of the available tool/agent are as follows:

    {agent_desc}

    With above tools/agents you can build a Workflow to answer a given question.


    The followings are some examples:

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: QR,R,AG
    We can just retrieve the external information about the given question through Retrieval Agent (RA), then the Answer Generator (AG) can output the answer to the given question.

    For multi-hop questions like this:
    Question: "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?"
    Workflow: QDP
    This kind of problem requires first breaking the initial question into multiple independent searchable sub-questions. For such a question, only the QDP needs to be given, and no other tool/agent needs to be given.

    For multi-hop questions like this:
    Question: "What is the most played song on Michael Jackson's third best-selling album?"
    Workflow: QDS
    This kind of problem requires breaking down the original problem into multiple subproblems that depend on each other. Convenient for subsequent processing. For such a question, only the QDS needs to be given, and no other tool/agent needs to be given.

    For multi-hop questions like this:
    Question: When was the director of Terminator 2 born?
    Workflow: QDS
    This kind of problem requires breaking down the original problem into multiple subproblems that depend on each other. Convenient for subsequent processing. For such a question, only the QDS needs to be given, and no other tool/agent needs to be given.



    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """,
            },
            {
                "role": "assistant",
                "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question.",
            },
            {
                "role": "user",
                "content": f"""Please give the appropriate Workflow in the format requested above.
    Qustion: "{question}"
    Workflow: """,
            },
        ]

    # ------------------------------------------------------------------
    # Goal-conditioned (CONDOR) – restricted agent list + goal prefix
    # ------------------------------------------------------------------

    def _goal_conditioned_messages(
        self, question: str, goal: Tuple[int, int]
    ) -> List[Dict[str, str]]:
        m_star, qtype_id = goal
        valid_abbrevs = MECHANISM_ACTION_SPACES.get(m_star, list(_AGENT_DESCRIPTIONS.keys()))
        goal_desc = MECHANISM_GOAL_DESCRIPTIONS.get(m_star, "")

        agent_desc = "\n\n    ".join(
            _AGENT_DESCRIPTIONS[k] for k in valid_abbrevs if k in _AGENT_DESCRIPTIONS
        )

        # Mechanism-specific examples
        examples = self._examples_for_mechanism(m_star)

        goal_prefix = (
            f"[Goal] Selected mechanism: {goal_desc}\n"
            f"Use ONLY the agents listed below — no others are available for this mechanism.\n\n"
        )

        system_content = (
            "You are a helpful assistant to plan a Workflow for the given information "
            "using the give tool/agent.\n" + goal_prefix
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": "Ok, please show me you requirement information and the available tool/agent"},
            {
                "role": "user",
                "content": f"""The descriptions of the available tool/agent are as follows:

    {agent_desc}

    With above tools/agents you can build a Workflow to answer a given question.

{examples}

    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """,
            },
            {
                "role": "assistant",
                "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question.",
            },
            {
                "role": "user",
                "content": f"""Please give the appropriate Workflow in the format requested above.
    Qustion: "{question}"
    Workflow: """,
            },
        ]

    def _examples_for_mechanism(self, m_star: int) -> str:
        if m_star == 0:
            return """    The followings are some examples:

    For factoid questions answerable from parametric knowledge:
    Question: "What is the capital of France?"
    Workflow: AG
    The Answer Generator can answer directly without any retrieval.
"""
        elif m_star in (1, 3, 4):
            return """    The followings are some examples:

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: QR,R,AG
    We retrieve the external information then the Answer Generator outputs the answer.

    For questions where document filtering is useful:
    Question: "Who invented the telephone?"
    Workflow: R,DS,AG
    We retrieve, then select the most relevant documents, then generate the answer.
"""
        elif m_star == 2:
            return """    The followings are some examples:

    For multi-hop questions with independent sub-questions:
    Question: "Which city will have the highest GDP in 2024: Singapore, Houston, Beijing?"
    Workflow: QDP
    Only QDP is needed; subsequent turns handle each sub-question.

    For multi-hop questions with dependent sub-questions:
    Question: "When was the director of Terminator 2 born?"
    Workflow: QDS
    Only QDS is needed; subsequent turns handle each dependent sub-question.
"""
        return ""

    # ------------------------------------------------------------------
    # Sub-query messages (same for all mechanisms)
    # ------------------------------------------------------------------

    def _sub_messages(self, question: str) -> List[Dict[str, str]]:
        agent_desc = "\n\n    ".join(
            _SUB_AGENT_DESCRIPTIONS[k] for k in ['R', 'DS', 'AG']
        )
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant to plan a Workflow for the given information using the give tool/agent.",
            },
            {
                "role": "assistant",
                "content": "Ok, please show me you requirement information and the available tool/agent",
            },
            {
                "role": "user",
                "content": f"""The descriptions of the available tool/agent are as follows:

    {agent_desc}

    With above tools/agents you can build a Workflow to answer a given question.


    The followings are some examples:

    For the easiest question, the Answer Generator (AG) can answer it directly without any query reformulation and external information:
    Question: "Which city is the capital of Australia."
    Workflow: AG
    Then, we can use Answer Generator (AG) to get the answer directly.

    For simple questions that may require external knowledge to answer:
    Question: "Which team is the 2025 NBA champion?"
    Workflow: R,AG
    We can just retrieve the external information about the given question through Retrieval Agent (RA), then the Answer Generator (AG) can output the answer to the given question.


    Next I'll give you the Question that need to be answered, just output the appropriate Workflow in the format above, don't output anything else.
    """,
            },
            {
                "role": "assistant",
                "content": "Ok, you can provide the Question and I will give the appropriate Workflow to answer the Question.",
            },
            {
                "role": "user",
                "content": f"""Please give the appropriate Workflow in the format requested above.
    Qustion: "{question}"
    Workflow: """,
            },
        ]
