import logging
import os

import yaml
from langgraph.graph import StateGraph, END

from execution_agent.state import WorkflowState
from handlers.registry import registry, load_all_handlers

logger = logging.getLogger(__name__)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(client_id: str) -> dict:
    """
    Load client config from clients/<client_id>/config.yaml
    """
    config_path = os.path.join("clients", client_id, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found for client '{client_id}' at '{config_path}'.\n"
            f"Create the file or check the client_id spelling."
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(
        "[Engine] Loaded config for client '%s' — %d steps: %s",
        client_id,
        len(config.get("steps", [])),
        [s["type"] for s in config.get("steps", [])],
    )
    return config


# ── Built-in nodes (always present) ──────────────────────────────────────────

def _validation_node(state: WorkflowState) -> WorkflowState:
    """Validate that all step types in the workflow have registered handlers."""
    logger.info("[Validation] Checking workflow steps...")
    steps = state["workflow"].get("steps", [])

    if not steps:
        return {**state, "status": "FAILED", "error": "Workflow has no steps."}

    for step in steps:
        step_type = step.get("type", "")
        try:
            registry.get(step_type)
        except ValueError:
            msg = (
                f"[Validation] No handler for step type '{step_type}'. "
                f"Available: {registry.available()}"
            )
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}

    logger.info("[Validation] All %d steps validated.", len(steps))
    return {**state, "status": "RUNNING", "current_step_index": 0, "error": None}


def _audit_node(state: WorkflowState) -> WorkflowState:
    """Mark workflow as completed."""
    from datetime import datetime, timezone
    ts  = datetime.now(timezone.utc).isoformat()
    msg = f"EXECUTION COMPLETED AT {ts}"
    logger.info("[Audit] %s", msg)
    return {
        **state,
        "status": "COMPLETED",
        "logs":   state.get("logs", []) + [msg],
        "error":  None,
    }


def _rollback_node(state: WorkflowState) -> WorkflowState:
    """Delete all files created during a failed run."""
    logger.warning("[Rollback] Starting rollback...")
    deleted = []
    for path in state.get("files_created", []):
        try:
            os.remove(path)
            deleted.append(path)
        except FileNotFoundError:
            pass
    msg = f"Rollback complete. Deleted {len(deleted)} file(s)."
    logger.warning("[Rollback] %s", msg)
    return {
        **state,
        "files_created": [],
        "logs": state.get("logs", []) + [msg],
    }


# ── Dynamic step node factory ─────────────────────────────────────────────────

def _make_step_node(step_index: int):
    """
    Returns a LangGraph node function for a specific step index.
    The node reads the step config from state and calls the correct handler.
    """
    def step_node(state: WorkflowState) -> WorkflowState:
        steps = state["workflow"]["steps"]

        if step_index >= len(steps):
            return state

        step    = steps[step_index]
        handler = registry.get(step["type"])

        logger.info(
            "[Engine] Executing step %d/%d — type: '%s' id: '%s'",
            step_index + 1, len(steps),
            step.get("type"), step.get("id", ""),
        )
        return handler.execute(step, state)

    # Give the node a unique name for LangGraph
    step_node.__name__ = f"step_{step_index}"
    return step_node


# ── Routing helpers ───────────────────────────────────────────────────────────

def _route_after_validation(state: WorkflowState) -> str:
    if state.get("status") == "FAILED":
        return "rollback"
    return "step_0"


def _make_step_router(next_node: str):
    """Returns a routing function that goes to next_node or rollback."""
    def route(state: WorkflowState) -> str:
        if state.get("status") == "FAILED":
            return "rollback"
        return next_node
    return route


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(client_id: str) -> tuple:
    """
    Build and compile a LangGraph graph from the client config.

    Returns (compiled_graph, config)

    The graph contains ONLY the nodes the client needs.
    Steps not in config.yaml = not in graph = never executed.
    """
    # Load all handlers so they register themselves
    load_all_handlers()

    # Load client config
    config = load_config(client_id)
    steps  = config.get("steps", [])

    if not steps:
        raise ValueError(f"Client '{client_id}' config has no steps defined.")

    graph = StateGraph(WorkflowState)

    # ── Always-present nodes ──────────────────────────────────────────────────
    graph.add_node("validation", _validation_node)
    graph.add_node("audit",      _audit_node)
    graph.add_node("rollback",   _rollback_node)

    # ── Dynamic step nodes ────────────────────────────────────────────────────
    # One node per step in the config, named step_0, step_1, step_N
    step_node_names = []
    for i, step in enumerate(steps):
        node_name = f"step_{i}"
        graph.add_node(node_name, _make_step_node(i))
        step_node_names.append(node_name)
        logger.debug("[Engine] Added node '%s' for step type '%s'", node_name, step["type"])

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("validation")

    # ── Edges ─────────────────────────────────────────────────────────────────

    # After validation → first step or rollback
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"rollback": "rollback", "step_0": step_node_names[0]},
    )

    # Between steps → next step or rollback
    for i, node_name in enumerate(step_node_names):
        if i < len(step_node_names) - 1:
            # Not last step — route to next step or rollback
            next_node = step_node_names[i + 1]
            graph.add_conditional_edges(
                node_name,
                _make_step_router(next_node),
                {"rollback": "rollback", next_node: next_node},
            )
        else:
            # Last step — route to audit or rollback
            graph.add_conditional_edges(
                node_name,
                _make_step_router("audit"),
                {"rollback": "rollback", "audit": "audit"},
            )

    # Terminal edges
    graph.add_edge("audit",    END)
    graph.add_edge("rollback", END)

    compiled = graph.compile()
    logger.info(
        "[Engine] Graph built for client '%s' — nodes: %s",
        client_id,
        ["validation"] + step_node_names + ["audit", "rollback"],
    )
    return compiled, config


# ══════════════════════════════════════════════════════════════════════════════
# build_graph_from_dict
# Called by planner_agent — takes workflow dict directly, no config file needed
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_from_dict(workflow: dict):
    """
    Build and compile a LangGraph graph from a workflow dict.
    Called by planner_agent.execute_node() directly.

    This is the CONNECTION POINT between planner and execution agent.

    Args:
        workflow: dict with 'steps' list — produced by planner_agent.plan_node()

    Returns:
        compiled LangGraph graph ready to invoke()
    """
    load_all_handlers()

    steps = workflow.get("steps", [])
    if not steps:
        raise ValueError("Workflow has no steps.")

    graph = StateGraph(WorkflowState)

    # Always-present nodes
    graph.add_node("validation", _validation_node)
    graph.add_node("audit",      _audit_node)
    graph.add_node("rollback",   _rollback_node)

    # One dynamic node per step planner specified
    step_node_names = []
    for i, step in enumerate(steps):
        node_name = f"step_{i}"
        graph.add_node(node_name, _make_step_node(i))
        step_node_names.append(node_name)
        logger.info(
            "[Engine] Node '%s' → handler '%s'",
            node_name, step["type"]
        )

    # Entry
    graph.set_entry_point("validation")

    # Validation → first step or rollback
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"rollback": "rollback", "step_0": step_node_names[0]},
    )

    # Step → next step or rollback
    for i, node_name in enumerate(step_node_names):
        if i < len(step_node_names) - 1:
            next_node = step_node_names[i + 1]
            graph.add_conditional_edges(
                node_name,
                _make_step_router(next_node),
                {"rollback": "rollback", next_node: next_node},
            )
        else:
            graph.add_conditional_edges(
                node_name,
                _make_step_router("audit"),
                {"rollback": "rollback", "audit": "audit"},
            )

    graph.add_edge("audit",    END)
    graph.add_edge("rollback", END)

    compiled = graph.compile()
    logger.info(
        "[Engine] Graph built from dict — steps: %s",
        [s["type"] for s in steps]
    )
    return compiled