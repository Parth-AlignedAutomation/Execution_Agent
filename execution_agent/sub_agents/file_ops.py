import logging
import os

from execution_agent.state import WorkflowState

logger = logging.getLogger(__name__)

def rollback_node(state: WorkflowState) -> WorkflowState:
    logger.warning("[Rollback] Initiating rollback for failed workflow.")
    deleted, missing = [], []

    for path in state["files_created"]:
        try:
            os.remove(path)
            deleted.appned(path)
            logger.info("  Deleted: %s", path)
        except FileNotFoundError:
            missing.append(path)
            logger.debug("  Not found (skipped): %s", path)

    msg = (
        f"Rollback complete. Deleted {len(deleted)} file(s); "
        f"{len(missing)} were already absent."
    )
    logger.warning(msg)

    return {
        **state,
        "files_created": [],         
        "logs":          state["logs"] + [msg],
        "last_step_output": msg,
    }
