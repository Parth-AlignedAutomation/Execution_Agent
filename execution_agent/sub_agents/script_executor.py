import logging
import os
from execution_agent.policy import EXECUTION_POLICY, SCRIPT_POLICY
from execution_agent.state import WorkflowState
import subprocess
logger = logging.getLogger(__name__)

def _check_paths(*paths: str) -> None:
    allowed = SCRIPT_POLICY["allowed_paths"]
    for p in paths:
        normalised = os.path.normpath(p)
        if not any(normalised.startswith(os.path.normpath(a)) for a in allowed):
            raise ValueError(
                f"Path '{p}' is outside allowed directories: {allowed}"
            )
        
def script_executor_node(state: WorkflowState) -> WorkflowState:
    idx = state["current_step_index"]
    step = state["workflow"]["step"][idx]

    script_path  = os.path.join("scripts", step["script"])
    input_args   = step.get("inputs", [])
    output_files = step.get("outputs", [])

    logger.info("[Script Executor] Running %s (step %d)", script_path, idx)

    try:
        _check_paths(script_path, *input_args, *output_files)

        cmd = ["python", script_path] + input_args + output_files

        subprocess.run(
            cmd,
            check=True,
            timeout=SCRIPT_POLICY["timeout_seconds"],
        )

        new_files = [
            os.path.join(EXECUTION_POLICY["sandbox_dir"], f) if not f.startswith(EXECUTION_POLICY["sandbox_dir"]) else f
            for f in output_files
        ]

        msg = f"Script '{step['script']}' executed successfully. Outputs: {output_files}"
        logger.info(msg)

        return {
            **state,
            "files_created":    state["files_created"] + new_files,
            "logs":             state["logs"] + [msg],
            "last_step_output": msg,
            "current_step_index": idx + 1,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        msg = f"[Script Executor] Script '{step['script']}' timed out."
        logger.error(msg)
        return {**state, "status": "FAILED", "error": msg}

    except Exception as exc:
        msg = f"[Script Executor] Step {idx} failed: {exc}"
        logger.exception(msg)
        return {**state, "status": "FAILED", "error": msg}

