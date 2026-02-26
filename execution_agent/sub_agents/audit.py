import logging
from datetime import datetime, timezone
from execution_agent.state import WorkflowState

logger = logging.getLogger(__name__)

def audit_node(state: WorkflowState) -> WorkflowState:

    ts  = datetime.now(timezone.utc).isoformat()
    msg = f"EXECUTION COMPLETED AT {ts}"
    logger.info("[Audit] %s", msg)

    return {
        **state,
        "status": "COMPLETED",
        "logs":   state["logs"] + [msg],
        "last_step_output": msg,
        "error": None,
    }