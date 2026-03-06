import json
import logging
import os
import re

import requests

from execution_agent.handlers.base_handler import BaseHandler
from execution_agent.handlers.registry import registry

logger      = logging.getLogger(__name__)
SANDBOX_DIR = os.getenv("SANDBOX_DIR", "sandbox/runtime")

#helpers

def _resolve_env(value) -> str:
    """Replace ${VAR} or $VAR with actual env values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{?([A-Z_][A-Z0-9_]*)\}?",
            lambda m: os.getenv(m.group(1), m.group(0)),
            value,
        )
    return value

def _resolve_headers(headers: dict) -> dict:
    return {k: _resolve_env(v) for k, v in headers.items()}



def _rest_call(config: dict):
    url     = _resolve_env(config.get("url", ""))
    method  = config.get("method", "GET").upper()
    headers = _resolve_headers(config.get("headers", {}))
    params  = config.get("params", {})
    body    = config.get("body", None)
    timeout = int(config.get("timeout", 30))

    if not url:
        raise ValueError("REST adapter: 'url' is required.")

    response = requests.request(
        method=method, url=url,
        headers=headers, params=params,
        json=body, timeout=timeout,
    )
    response.raise_for_status()
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, response.text


def _graphql_call(config: dict):
    url       = _resolve_env(config.get("url", ""))
    query     = config.get("query", "")
    variables = config.get("variables", {})
    headers   = _resolve_headers(config.get("headers", {}))
    timeout   = int(config.get("timeout", 30))

    if not url:
        raise ValueError("GraphQL adapter: 'url' is required.")
    if not query:
        raise ValueError("GraphQL adapter: 'query' is required.")

    headers.setdefault("Content-Type", "application/json")
    response = requests.post(
        url,
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return response.status_code, data.get("data", data)


API_ADAPTERS = {
    "rest": {
        "call": _rest_call,
    },
    "graphql": {
        "call": _graphql_call,
    },

}

class APIHandler(BaseHandler):

    @property
    def name(self) -> str:
        return "http_request"

    def execute(self, step: dict, state: dict) -> dict:
        adapter_key = step.get("adapter", "rest").lower()

        if adapter_key not in API_ADAPTERS:
            return {
                **state, "status": "FAILED",
                "error": (
                    f"[APIHandler] Unknown adapter '{adapter_key}'. "
                    f"Available: {list(API_ADAPTERS.keys())}"
                ),
            }

        adapter_cfg = API_ADAPTERS[adapter_key]

        try:
            logger.info("[APIHandler] Adapter: %s | URL: %s", adapter_key, step.get("url", ""))
            status_code, response_data = adapter_cfg["call"](step)

            output_file = step.get("output", "api_response.json")
            output_path = os.path.join(SANDBOX_DIR, output_file)
            os.makedirs(SANDBOX_DIR, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                if isinstance(response_data, (dict, list)):
                    json.dump(response_data, f, indent=2, default=str)
                else:
                    f.write(str(response_data))

            msg = f"[APIHandler] HTTP {status_code} → {output_path}"
            logger.info(msg)

            return {
                **state,
                "files_created":      state.get("files_created", []) + [output_path],
                "logs":               state.get("logs", []) + [msg],
                "last_step_output":   msg,
                "current_step_index": state.get("current_step_index", 0) + 1,
                "error": None,
            }

        except Exception as exc:
            msg = f"[APIHandler] Failed: {exc}"
            logger.exception(msg)
            return {**state, "status": "FAILED", "error": msg}


# Auto-register
registry.register(APIHandler())