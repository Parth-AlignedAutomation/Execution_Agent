import logging
import os
import yaml
from langgraph.graph import StateGraph, END
from execution_agent.state import WorkflowState
from execution_agent.handlers.registry import registry, load_all_handler

logger = logging.getLogger(__name__)


def load_config(client_id: str) -> dict:
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


def _validation_node(state: WorkflowState) -> WorkflowState:
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
    from datetime import datetime, timezone
    from execution_agent.core.llm import run_llm_node

    ts = datetime.now(timezone.utc).isoformat()
    steps = state["workflow"].get("steps", [])

    variables = {
        "steps": "\n".join(
            f"  - Step {i+1}: {s.get('type')} (id: {s.get('id', 'unnamed')})"
            for i, s in enumerate(steps)
        ),
        "files": "\n".join(
            f"  - {f}" for f in state.get("files_created", [])
        ) or "  - No files created",
        "logs": "\n".join(
            f"  - {l}" for l in state.get("logs", [])
        ) or "  - No logs",
        "timestamp": ts,
    }

    summary = run_llm_node(
        prompt_name="execution-audit-prompt",
        variables=variables,
    )
    logger.info("[Audit] LLM summary generated (%d chars).", len(summary))

    return {
        **state,
        "status": "COMPLETED",
        "logs": state.get("logs", []) + [summary],
        "last_step_output": summary,
        "error": None,
    }


def _rollback_node(state: WorkflowState) -> WorkflowState:
    from execution_agent.core.llm import run_llm_node

    logger.warning("[Rollback] Starting rollback...")
    deleted = []
    for path in state.get("files_created", []):
        try:
            os.remove(path)
            deleted.append(path)
            logger.warning("[Rollback] Deleted: %s", path)
        except FileNotFoundError:
            pass

    steps = state["workflow"].get("steps", [])
    failed_index = state.get("current_step_index", 0)
    failed_step = steps[failed_index] if failed_index < len(steps) else {}

    variables = {
        "error": state.get("error", "Unknown error"),
        "failed_step": (
            f"Step {failed_index + 1}: "
            f"type='{failed_step.get('type', 'unknown')}' "
            f"id='{failed_step.get('id', 'unnamed')}'"
        ),
        "logs": "\n".join(
            f"  - {l}" for l in state.get("logs", [])
        ) or "  - No logs captured",
        "files_deleted": "\n".join(
            f"  - {f}" for f in deleted
        ) or "  - No files to delete",
    }

    diagnosis = run_llm_node(
        prompt_name="execution-rollback-prompt",
        variables=variables,
    )
    logger.warning("[Rollback] LLM diagnosis generated (%d chars).", len(diagnosis))

    plain_msg = f"Rollback complete. Deleted {len(deleted)} file(s)."
    logger.warning("[Rollback] %s", plain_msg)
    logger.warning("[Rollback] Diagnosis:\n%s", diagnosis)

    return {
        **state,
        "files_created": [],
        "logs": state.get("logs", []) + [plain_msg, diagnosis],
        "last_step_output": diagnosis,
    }


def _make_step_node(step_index: int):
    def step_node(state: WorkflowState) -> WorkflowState:
        steps = state["workflow"]["steps"]

        if step_index >= len(steps):
            return state

        step = steps[step_index]
        handler = registry.get(step["type"])

        logger.info(
            "[Engine] Executing step %d/%d — type: '%s' id: '%s'",
            step_index + 1, len(steps),
            step.get("type"), step.get("id", ""),
        )
        return handler.execute(step, state)

    step_node.__name__ = f"step_{step_index}"
    return step_node


def _route_after_validation(state: WorkflowState) -> str:
    if state.get("status") == "FAILED":
        return "rollback"
    return "step_0"


def _make_step_router(next_node: str):
    def route(state: WorkflowState) -> str:
        if state.get("status") == "FAILED":
            return "rollback"
        return next_node
    return route


def build_graph(client_id: str) -> tuple:
    load_all_handler()
    config = load_config(client_id)
    steps = config.get("steps", [])

    if not steps:
        raise ValueError(f"Client '{client_id}' config has no steps defined.")

    graph = StateGraph(WorkflowState)

    graph.add_node("validation", _validation_node)
    graph.add_node("audit", _audit_node)
    graph.add_node("rollback", _rollback_node)

    step_node_names = []
    for i, step in enumerate(steps):
        node_name = f"step_{i}"
        graph.add_node(node_name, _make_step_node(i))
        step_node_names.append(node_name)
        logger.debug("[Engine] Added node '%s' for step type '%s'", node_name, step["type"])

    graph.set_entry_point("validation")

    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"rollback": "rollback", "step_0": step_node_names[0]},
    )

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

    graph.add_edge("audit", END)
    graph.add_edge("rollback", END)

    compiled = graph.compile()
    logger.info(
        "[Engine] Graph built for client '%s' — nodes: %s",
        client_id,
        ["validation"] + step_node_names + ["audit", "rollback"],
    )
    return compiled, config


def build_graph_from_dict(workflow: dict):
    load_all_handler()

    steps = workflow.get("steps", [])
    if not steps:
        raise ValueError("Workflow has no steps.")

    graph = StateGraph(WorkflowState)

    graph.add_node("validation", _validation_node)
    graph.add_node("audit", _audit_node)
    graph.add_node("rollback", _rollback_node)

    step_node_names = []
    for i, step in enumerate(steps):
        node_name = f"step_{i}"
        graph.add_node(node_name, _make_step_node(i))
        step_node_names.append(node_name)
        logger.info(
            "[Engine] Node '%s' → handler '%s'",
            node_name, step["type"]
        )

    graph.set_entry_point("validation")

    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"rollback": "rollback", "step_0": step_node_names[0]},
    )

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

    graph.add_edge("audit", END)
    graph.add_edge("rollback", END)

    compiled = graph.compile()
    logger.info(
        "[Engine] Graph built from dict — steps: %s",
        [s["type"] for s in steps]
    )
    return compiled