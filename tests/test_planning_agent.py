"""Unit tests for qa_manager/PlanningAgent.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qa_manager.PlanningAgent import PlanningAgent
from condor.mechanism_router import MECHANISM_ACTION_SPACES


@pytest.fixture
def agent():
    return PlanningAgent()


QUESTION = "What is the capital of France?"


# ── _baseline_messages ────────────────────────────────────────────────────────

def test_baseline_messages_count(agent):
    msgs = agent.create_planning_messages(QUESTION)
    assert len(msgs) == 5

def test_baseline_messages_roles(agent):
    msgs = agent.create_planning_messages(QUESTION)
    expected_roles = ['system', 'assistant', 'user', 'assistant', 'user']
    assert [m['role'] for m in msgs] == expected_roles

def test_baseline_typo_preserved(agent):
    msgs = agent.create_planning_messages(QUESTION)
    last_user = msgs[-1]['content']
    assert 'Qustion' in last_user, "Intentional 'Qustion' typo must be preserved"

def test_baseline_question_in_last_message(agent):
    msgs = agent.create_planning_messages(QUESTION)
    assert QUESTION in msgs[-1]['content']

def test_baseline_all_agents_mentioned(agent):
    msgs = agent.create_planning_messages(QUESTION)
    user_msg = msgs[2]['content']
    for abbrev in ['QR', 'QDP', 'QDS', 'R', 'DS', 'AG']:
        assert abbrev in user_msg, f"Abbreviation {abbrev} not found in baseline prompt"


# ── _goal_conditioned_messages ────────────────────────────────────────────────

@pytest.mark.parametrize("m_star", [0, 1, 2, 3, 4])
def test_goal_conditioned_message_count(agent, m_star):
    msgs = agent.create_planning_messages(QUESTION, goal=(m_star, 0))
    assert len(msgs) == 5

def test_goal_conditioned_typo_preserved(agent):
    msgs = agent.create_planning_messages(QUESTION, goal=(1, 0))
    assert 'Qustion' in msgs[-1]['content']

def test_goal_conditioned_restricts_m0(agent):
    msgs = agent.create_planning_messages(QUESTION, goal=(0, 0))
    user_msg = msgs[2]['content']
    # m0 allows only AG — should not mention QR, R, DS, QDP, QDS
    assert 'AG' in user_msg
    for token in ['QR', 'R ', 'DS', 'QDP', 'QDS']:
        assert token not in user_msg, f"Token {token!r} should not appear for m0"

def test_goal_conditioned_restricts_m1(agent):
    msgs = agent.create_planning_messages(QUESTION, goal=(1, 0))
    user_msg = msgs[2]['content']
    # m1: ['QR','R','DS','AG'] — should NOT mention QDP, QDS
    allowed = set(MECHANISM_ACTION_SPACES[1])
    for token in ['QDP', 'QDS']:
        if token not in allowed:
            assert token not in user_msg

def test_goal_conditioned_has_goal_prefix(agent):
    msgs = agent.create_planning_messages(QUESTION, goal=(2, 1))
    system_msg = msgs[0]['content']
    assert '[Goal]' in system_msg

def test_goal_none_gives_baseline(agent):
    msgs_no_goal = agent.create_planning_messages(QUESTION, goal=None)
    msgs_baseline = agent._baseline_messages(QUESTION)
    assert msgs_no_goal == msgs_baseline


# ── _sub_messages ─────────────────────────────────────────────────────────────

def test_sub_messages_count(agent):
    msgs = agent.create_planning_messages(QUESTION, is_sub=True)
    assert len(msgs) == 5

def test_sub_messages_typo_preserved(agent):
    msgs = agent.create_planning_messages(QUESTION, is_sub=True)
    assert 'Qustion' in msgs[-1]['content']

def test_sub_messages_restricted_agents(agent):
    msgs = agent.create_planning_messages(QUESTION, is_sub=True)
    user_msg = msgs[2]['content']
    # Sub messages allow only R, DS, AG
    for abbrev in ['R', 'DS', 'AG']:
        assert abbrev in user_msg
    # Should NOT have decomposition agents
    for abbrev in ['QDP', 'QDS', 'QR']:
        assert abbrev not in user_msg


# ── is_sub takes priority over goal ──────────────────────────────────────────

def test_is_sub_overrides_goal(agent):
    msgs_sub = agent.create_planning_messages(QUESTION, is_sub=True, goal=(2, 0))
    msgs_sub_only = agent.create_planning_messages(QUESTION, is_sub=True)
    assert msgs_sub == msgs_sub_only
