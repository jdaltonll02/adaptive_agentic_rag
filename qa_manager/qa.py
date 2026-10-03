"""CONDOR qa_manager/qa.py

Goal-conditioned extension of the baseline Agentic_RAG_Manager.

Changes vs baseline:
  1. trans_rawprompt_to_ids reads m_star/qtype_id from extra_info and
     passes goal=(m_star, qtype_id) to PlanningAgent.create_planning_messages.
  2. is_valid_list accepts an optional m_star parameter and validates the
     parsed workflow against MECHANISM_ACTION_SPACES[m_star] instead of the
     full mapping_dict when m_star is provided.

When m_star is None (or extra_info does not carry m_star), the code
behaves identically to the baseline.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto

from qa_manager.BaseAgent import (  # CONDOR versions
    AgentPool,
    QueryRewriteAgent,
    QueryDecompositionAgentParallel,
    QueryDecompositionAgentSerial,
    RetrievalAgent,
    DocumentSelectionAgent,
    AnswerGenerationAgent,
    AnswerSummarizationAgent,
    AGENT_CONFIG,
    EXAMPLE_PROMPT,
)
from qa_manager.PlanningAgent import PlanningAgent
from qa_manager.config import MECHANISM_ACTION_SPACES


def remove_trailing_marker(text: str) -> str:
    marker = "<|im_end|>"
    if text.endswith(marker):
        return text[:-len(marker)]
    return text


class Agentic_RAG_Manager:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.planning_agent = PlanningAgent()
        self.agent_pool = AgentPool()

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", True)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "right")

        self.api_url = os.environ.get("RETRIEVAL_API_URL", "http://localhost:8000/search")

    def trans_rawprompt_to_ids(self, batch_dict, is_sub_list):
        """Tokenise planning messages.

        CONDOR extension: reads m_star and qtype_id from extra_info[i] and
        passes them as goal=(m_star, qtype_id) to create_planning_messages
        so the prompt restricts the agent list to MECHANISM_ACTION_SPACES[m_star].
        Falls back to baseline behaviour when m_star is absent.
        """
        extra_info_list = batch_dict['extra_info']
        questions = [item['question'] for item in extra_info_list]

        messages_list = []
        for i, question in enumerate(questions):
            is_sub = is_sub_list[i]
            m_star = extra_info_list[i].get('m_star', None)
            qtype_id = extra_info_list[i].get('qtype_id', 0)
            goal = (m_star, qtype_id) if m_star is not None else None
            messages = self.planning_agent.create_planning_messages(
                question, is_sub=is_sub, goal=goal
            )
            messages_list.append(messages)

        update_dict_list = [self.get_single_ids(msgs) for msgs in messages_list]
        for i, update_dict in enumerate(update_dict_list):
            for key in ["input_ids", "attention_mask", "position_ids"]:
                batch_dict[key][i] = update_dict[key]

        batch_dict["raw_prompt"] = np.array(
            [ud["raw_prompt"] for ud in update_dict_list], dtype=object
        )
        batch_dict["raw_prompt_ids"] = np.array(
            [ud["raw_prompt_ids"] for ud in update_dict_list], dtype=object
        )
        return batch_dict

    def get_single_ids(self, messages):
        update_dict = {}

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        update_dict["input_ids"] = input_ids[0]
        update_dict["attention_mask"] = attention_mask[0]
        update_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                half = self.max_prompt_length // 2
                raw_prompt_ids = raw_prompt_ids[:half] + raw_prompt_ids[-(self.max_prompt_length - half):]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}."
                )
        update_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            update_dict["raw_prompt"] = messages
        return update_dict

    def parse_workflow(self, workflow_string):
        """Map abbreviations to full agent names (identical to baseline)."""
        mapping_dict = {
            'QR': 'QueryRewriteAgent',
            'QDP': 'QueryDecompositionAgentParallel',
            'QDS': 'QueryDecompositionAgentSerial',
            'R': 'RetrievalAgent',
            'DS': 'DocumentSelectionAgent',
            'AG': 'AnswerGenerationAgent',
            'AS': 'AnswerSummarizationAgent',
        }
        return [mapping_dict[m] for m in workflow_string if m in mapping_dict]

    def is_valid_list(
        self,
        input_list: List[str],
        context: dict,
        m_star: Optional[int] = None,
    ) -> bool:
        """Validate a parsed workflow list.

        CONDOR extension: when m_star is provided, validates that all
        abbreviations belong to MECHANISM_ACTION_SPACES[m_star].

        Args:
            input_list: list of abbreviation strings (e.g. ['QR', 'R', 'AG'])
            context:    execution context (used for begin_step check)
            m_star:     selected mechanism id; None → baseline behaviour

        Returns True iff the workflow is valid.
        """
        # Full mapping always used for name normalisation
        full_mapping = {
            'QR': 'QueryRewriteAgent',
            'QDP': 'QueryDecompositionAgentParallel',
            'QDS': 'QueryDecompositionAgentSerial',
            'R': 'RetrievalAgent',
            'DS': 'DocumentSelectionAgent',
            'AG': 'AnswerGenerationAgent',
            'AS': 'AnswerSummarizationAgent',
        }

        if not input_list:
            return False

        if len(input_list) != len(set(input_list)):
            return False

        # Determine the allowed abbreviation set
        if m_star is not None and m_star in MECHANISM_ACTION_SPACES:
            allowed = set(MECHANISM_ACTION_SPACES[m_star])
        else:
            allowed = set(full_mapping.keys())

        for item in input_list:
            if item not in allowed:
                return False

        if ('QDP' in input_list) and (len(input_list) != 1):
            return False
        if ('QDS' in input_list) and (len(input_list) != 1):
            return False

        if ('QDP' not in input_list) and ('QDS' not in input_list):
            if 'AG' not in input_list:
                return False
            if input_list[-1] != 'AG':
                return False

        if context.get('begin_step', -1) >= 0:
            if 'QDP' in input_list or 'QDS' in input_list:
                return False

        return True


# Re-export baseline QA_Manager so ray_trainer_agentic_rag_2 imports resolve.
import os as _os, sys as _sys
_baseline_qa = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'baseline', 'qa_manager')
)
_sys.path.insert(0, _baseline_qa)
from qa import QA_Manager  # noqa: F401  (baseline class, imported but unused by CONDOR trainer)
_sys.path.remove(_baseline_qa)
