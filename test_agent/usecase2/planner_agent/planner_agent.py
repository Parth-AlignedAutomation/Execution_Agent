"""
test_agent/usecase2/planner_agent/planner_agent.py
────────────────────────────────────────────────────
Usecase2 Planner — Weather data via public API + Slack notification.

Demonstrates:
    - http_request  → APIHandler  (no database, no script)
    - notification  → NotificationHandler

Steps:
    1. http_request
       GET https://api.openweathermap.org/data/2.5/weather?q=Mumbai
       API handler fetches + saves → weather_raw.json automatically

    2. notification
       Slack message with weather summary built from last_step_output

No scripts needed — API handler does the fetch and save entirely.

.env required:
    OPENWEATHER_API_KEY   → free key from openweathermap.org
    SLACK_WEBHOOK_URL     → Slack incoming webhook URL
"""

import logging
import os
from pathlib import Path
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ── Usecase2 base directory ───────────────────────────────────────────────────
USECASE_DIR = Path(__file__).parent.parent           # test_agent/usecase2/
SANDBOX_DIR = USECASE_DIR / "sandbox" / "runtime"    # test_agent/usecase2/sandbox/runtime/

# City to fetch weather for — change as needed
CITY = "Mumbai"


class PlannerState(TypedDict):
    instruction : str
    workflow    : Dict[str, Any]
    result      : Any


def plan_node(state: PlannerState) -> PlannerState:
    instruction = state.get("instruction", "")
    logger.info("[Planner] Received instruction: '%s'", instruction)
    logger.info("[Planner] Building workflow plan...")

    workflow = {
        "client_id":   "usecase2_weather_report",
        "description": instruction,
        "steps": [

            # Step 1 — Fetch weather from OpenWeatherMap REST API
            # APIHandler calls the URL, saves JSON response automatically
            # No script needed — api_handler does fetch + save
            {
                "id":          "fetch_weather",
                "type":        "http_request",        # → APIHandler
                "adapter":     "rest",                # → REST adapter
                "method":      "GET",
                "url":         f"https://api.openweathermap.org/data/2.5/weather",
                "params": {
                    "q":     CITY,
                    "appid": "${OPENWEATHER_API_KEY}", # resolved from .env
                    "units": "metric",                 # celsius
                },
                "output":      "weather_raw.json",    # saved to sandbox/runtime/
                "sandbox_dir": str(SANDBOX_DIR),
            },

            # Step 2 — Notify Slack with weather summary
            # Message is hardcoded here (LLM would build this dynamically later)
            # {
            #     "id":          "notify_slack",
            #     "type":        "notification",         # → NotificationHandler
            #     "channel":     "slack",                # → Slack adapter
            #     "webhook_url": "${SLACK_WEBHOOK_URL}", # resolved from .env
            #     "message":     (
            #         f"🌤 Weather Report — {CITY}\n"
            #         f"Fetched via OpenWeatherMap API.\n"
            #         f"Raw data saved to: sandbox/runtime/weather_raw.json"
            #     ),
            # },

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


def run(instruction: str = "Fetch weather data for Mumbai and notify team") -> dict:
    result = planner_graph.invoke({
        "instruction": instruction,
        "workflow":    {},
        "result":      None,
    })
    return result["result"]