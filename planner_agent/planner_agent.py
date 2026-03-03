"""
planner_agent/planner_agent.py
───────────────────────────────
Planner Agent — decides WHAT needs to be executed.
Execution Agent — runs WHAT planner decided.
"""

import logging
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PlannerState(TypedDict):
    instruction : str
    workflow    : Dict[str, Any]
    result      : Any


def plan_node(state: PlannerState) -> PlannerState:
    instruction = state.get("instruction", "")
    logger.info("[Planner] Received instruction: '%s'", instruction)
    logger.info("[Planner] Building workflow plan...")

    workflow = {
        "client_id":   "sales_report",
        "description": instruction,
        "steps": [

            # ✅ ONLY script execution — database and notification removed
            {
                "id":      "generate_report",
                "type":    "script_execution",
                "runner":  "python",
                "script":  "generate_report.py",
                "inputs":  [],
                "outputs": [
                    "sandbox/runtime/sales_raw.csv",
                    "sandbox/runtime/sales_report.pdf",
                ],
                "timeout": 300,
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


def run(instruction: str = "Generate daily sales report") -> dict:
    result = planner_graph.invoke({
        "instruction": instruction,
        "workflow":    {},
        "result":      None,
    })
    return result["result"]