"""Shim: re-export everything from baseline/qa_manager/BaseAgent4.py."""
import os
import sys

_baseline_qa = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline', 'qa_manager')
)
sys.path.insert(0, _baseline_qa)
from BaseAgent4 import *  # noqa: F401, F403
sys.path.remove(_baseline_qa)
