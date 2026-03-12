import logging
from pathlib import Path
from typing import List, Optional, Any, Dict,TypedDict
from langgraph.graph import StateGraph, END

logger  = logging.getLogger(__name__)

USECASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = USECASE_DIR / "scripts"
SANDBOX_DIR = USECASE_DIR / "sandbox"

REPORT_FILE = "expenses.csv"
ONEDRIVE_DESTINATION = f"reports/{REPORT_FILE}"


class PlannerState(TypedDict):
    instructions: str
    workflow: Dict[str, Any]
    result : Any

def plan_node(state:PlannerState) -> str:
    instruction = state.get("instructions", "")
    logger.info(f"[Planner] Received instruction: '%s'", instruction)
    logger.info("[Planner] Building workflow plan...")

    local_csv = str(SANDBOX_DIR / REPORT_FILE)

    workflow = {
        "client_id":   "usecase3_expense_report",
        "description": instruction,
        "steps": [
            {
                "id":         "generate_expenses",
                "type":       "script_execution",
                "runner":     "python",
                "script":     str(SCRIPTS_DIR / "generate_expenses.py"),
                "inputs":     [],
                "outputs":    [local_csv],  # passed as CLI arg to script
                "timeout":    60,
                "sandbox_dir": str(SANDBOX_DIR),
            },

            {
                "id":          "upload_to_onedrive",
                "type":        "file_upload",
                "storage":     "onedrive",           # → _onedrive_upload()
                "source":      local_csv,            # local file to upload
                "destination": ONEDRIVE_DESTINATION, # path inside OneDrive
                # credentials resolved from .env automatically
                # ONEDRIVE_CLIENT_ID, ONEDRIVE_CLIENT_SECRET, ONEDRIVE_TENANT_ID
            },

        ]
    }
        

    logger.info(
        "[Planner] Plan ready — %d steps: %s",
        len(workflow["steps"]),
        [s["type"] for s in workflow["steps"]]
    )
    return {**state, "workflow": workflow}



def execute_node(state: PlannerState) -> PlannerState:
    from execution_agent.core.engine import build_graph_from_dict

    workflow = state["workflow"]
    logger.info(
        "[Planner] Handing off %d steps to Execution Engine...",
        len(workflow["steps"])
    )

    compiled_graph = build_graph_from_dict(workflow)

    initial_state = {
        "workflow":           workflow,
        "current_step_index": 0,
        "files_created":      [],
        "logs":               [],
        "status":             "INIT",
        "last_step_output":   None,
        "error":              None,
    }

    final_state = compiled_graph.invoke(initial_state)

    logger.info("[Planner] Execution finished — status: %s", final_state.get("status"))
    if final_state.get("error"):
        logger.error("[Planner] Error: %s", final_state["error"])

    return {**state, "result": final_state}


def _build_planner_graph():
    graph = StateGraph(PlannerState)
    graph.add_node("plan_node",    plan_node)
    graph.add_node("execute_node", execute_node)
    graph.set_entry_point("plan_node")
    graph.add_edge("plan_node",    "execute_node")
    graph.add_edge("execute_node", END)
    return graph.compile()


planner_graph = _build_planner_graph()


def run(instruction: str = "Upload monthly expense report to OneDrive") -> dict:
    result = planner_graph.invoke({
        "instruction": instruction,
        "workflow":    {},
        "result":      None,
    })
    return result["result"]
    