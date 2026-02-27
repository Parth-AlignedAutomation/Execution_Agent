import logging
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph, END

from execution_agent.execution import execution_graph
from execution_agent.state import WorkflowState

logger = logging.getLogger(__name__)




class PlannerState(TypedDict):
    objective: str                    # Human-readable goal (unused in hardcoded mode)
    workflow:  Dict[str, Any]         # Produced by plan_node
    result:    Any   # Final state returned by execution_graph


HARDCODED_WORKFLOW = {
    "workflow_id": "report-001",
    "steps": [
        {
            # generate_report.py now handles:
            #   - DB connection & data fetch
            #   - LLM report generation via Langfuse prompt
            #   - Saving CSV  (arg 1)
            #   - Saving PDF  (arg 2)
            "type":    "script_execution",
            "script":  "generate_report.py",
            "inputs":  [],
            "outputs": [
                "sandbox/runtime/sales_raw.csv",
                "sandbox/runtime/sales_report.pdf",
            ],
        },
        {
            "type":       "notification",
            "to":         ["team@example.com"],
            "attachment": "sandbox/runtime/sales_report.pdf",
        },
    ],
}


def plan_node(state: PlannerState) -> PlannerState:
    """
    Builds the workflow.
    Hardcoded for now; swap the body for an LLM call later.
    """
    logger.info("[Planner] Building workflow for objective: '%s'", state.get("objective", ""))
    return {**state, "workflow": HARDCODED_WORKFLOW}


def execute_node(state: PlannerState) -> PlannerState:
    """
    Passes the workflow to the Execution Agent graph and collects the result.
    """
    logger.info("[Planner] Handing off workflow '%s' to Execution Agent.",
                state["workflow"].get("workflow_id"))

    initial_exec_state: WorkflowState = {
        "workflow":           state["workflow"],
        "current_step_index": 0,
        "files_created":      [],
        "logs":               [],
        "status":             "INIT",
        "last_step_output":   None,
        "error":              None,
    }

    final_exec_state = execution_graph.invoke(initial_exec_state)

    logger.info(
        "[Planner] Execution finished with status: %s",
        final_exec_state.get("status"),
    )
    if final_exec_state.get("error"):
        logger.error("[Planner] Execution error: %s", final_exec_state["error"])

    return {**state, "result": final_exec_state}



def build_planner_graph() -> StateGraph:
    graph = StateGraph(PlannerState)

    graph.add_node("plan_node",    plan_node)
    graph.add_node("execute_node", execute_node)

    graph.set_entry_point("plan_node")
    graph.add_edge("plan_node",    "execute_node")
    graph.add_edge("execute_node", END)

    return graph.compile()


planner_graph = build_planner_graph()




def run_pipeline(objective: str = "Generate daily sales report") -> WorkflowState:
    """
    Entry point for external callers / CLI.

    Returns the final WorkflowState produced by the Execution Agent.
    """
    result = planner_graph.invoke({"objective": objective, "workflow": {}, "result": None})
    return result["result"]