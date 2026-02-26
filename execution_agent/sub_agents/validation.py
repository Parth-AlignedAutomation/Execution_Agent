import logging
from execution_agent.state import WorkflowState
from execution_agent.policy import EXECUTION_POLICY

logger = logging.getLogger(__name__)

def validation_node(state: WorkflowState) -> WorkflowState:
    logger.info("=== [Validation Node] Starting workflow validation ===")
    workflow = state["workflow"]

    if "steps" not in workflow:
        msg = "Workflow is missing the steps"
        logger.error(msg)
        return {**state, "status": "FAILED", "error": msg}
    
    allowed = set(EXECUTION_POLICY["allowed_step_types"])
    for step in workflow["steps"]:
        step_type = step.get("type", "<missing>")
        if step_type not in allowed:
            msg = f"Step type '{step_type}' is not in the allowed list: {allowed}"
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}
        
    logger.info("Validation passed — all step types are allowed.")
    return {
        **state,
        "status": "RUNNING",
        "current_step_index": 0,
        "error": None,
    }
