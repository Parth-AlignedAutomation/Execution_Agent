import logging
import os
import subprocess
import sys                          # ← used to get the current venv's Python

from execution_agent.policy import EXECUTION_POLICY, SCRIPT_POLICY
from execution_agent.state import WorkflowState

logger = logging.getLogger(__name__)

def _check_paths(*paths: str) -> WorkflowState:
    allowed = SCRIPT_POLICY["allowed_paths"]

    for p in paths:
        normalised = os.path.normpath(p)

        if not any(normalised.startswith(os.path.normpath(a)) for a in allowed):
            raise ValueError(
                
            )
