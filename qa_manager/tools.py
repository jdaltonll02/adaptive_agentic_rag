"""Re-export baseline qa_manager/tools.py so BaseAgent4 imports resolve correctly."""
import os
import sys

_baseline_qa = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline', 'qa_manager')
)
if _baseline_qa not in sys.path:
    sys.path.insert(0, _baseline_qa)

from tools import TokenUsageTracker, setup_logger, setup_logger_no_print, MODEL_PRICING  # noqa: F401
