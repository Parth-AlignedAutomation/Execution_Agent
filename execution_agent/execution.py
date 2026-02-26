from langgraph.graph import StateGraph, END, START
from execution_agent.state import WorkflowState
# from execution_agent.sub_agents import (
#     validation, 
#     db_executor, 
#     script_executor,
#     file_ops,
#     audit
# )
from execution_agent.sub_agents.validation import validation_node
from execution_agent.sub_agents.db_executor import db_executor_node
from execution_agent.sub_agents.file_ops import rollback_node
from execution_agent.sub_agents.script_executor import script_executor_node
from execution_agent.sub_agents.audit import audit_node

def router_after_validation(state: WorkflowState) -> str:
    if state["status"] == "FAILED":
        return "rollback"
    return "route_step"


def route_step(state: WorkflowState) -> str:
    steps = state["workflow"]["steps"]
    idx   = state["current_step_index"]

    if idx >= len(steps):
        return "audit"                  

    step_type = steps[idx]["type"]
    dispatch  = {
        "database_read":    "db_executor",
        "script_execution": "script_executor",
        "notification":     "audit",    
    }
    return dispatch.get(step_type, "rollback")


def route_after_step(state: WorkflowState) -> str:
    if state["status"] == "FAILED":
        return "rollback"
    return "route_step"


def build_execution_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("validation", validation_node)
    graph.add_node("db_executor", db_executor_node)
    graph.add_node("rollback", rollback_node)
    graph.add_node("audit", audit_node)
    graph.add_node("script_executor", script_executor_node)

    graph.set_entry_point("validation")

    graph.add_conditional_edges(
        "validation", router_after_validation, {"rollback": "rollback", "route_step": "route_step_node"},
    )

    graph.add_node("route_step_node", lambda s: s)

    graph.add_conditional_edges(
        "route_step_node",
        route_step,
        {
            "db_executor":     "db_executor",
            "script_executor": "script_executor",
            "audit":           "audit",
            "rollback":        "rollback",
        },
    )

    graph.add_conditional_edges(
        "db_executor",
        route_after_step,
        {"rollback": "rollback", "route_step": "route_step_node"},
    )
    graph.add_conditional_edges(
        "script_executor",
        route_after_step,
        {"rollback": "rollback", "route_step": "route_step_node"},
    )

    graph.add_edge("rollback", END)
    graph.add_edge("audit",    END)

    return graph.compile()


execution_graph = build_execution_graph()






