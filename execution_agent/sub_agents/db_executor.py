import csv
import logging
import os

import psycopg2
from dotenv import load_dotenv
load_dotenv()

from execution_agent.policy import EXECUTION_POLICY, SQL_POLICY
from execution_agent.state import WorkflowState
from execution_agent.guardrails import SQLSafetyGuardrail


logger = logging.getLogger(__name__)
_sql_guard = SQLSafetyGuardrail()

def db_executor_node(state: WorkflowState) -> WorkflowState:
    idx = state["current_step_index"]
    step = state["workflow"]["steps"][idx]

    query = step["query"]

    logger.info("[DB Executor] Step %d — query: %s", idx, query[:80])

    result = _sql_guard.check(query)

    if not result.passed:
        msg = (
            f"[SQL Guardrail] BLOCKED — Rule: '{result.rule}' | "
            f"Reason: {result.reason}"
        )
        logger.error(msg)
        return {
            **state,
            "status": "FAILED",
            "error":  msg,
            "files_created": state.get("files_created", []),
            "logs": state.get("logs", []) + [msg],
        }
    safe_query = result.sanitised_query or query
    logger.info("[SQL Guardrail] Query passed all checks. Executing...")

    try:
        conn_str = os.getenv("Neon_URL")
        if not conn_str:
            raise RuntimeError("Environment variable 'Neon_URL' is not set.")

        sandbox = EXECUTION_POLICY["sandbox_dir"]
        os.makedirs(sandbox, exist_ok=True)
        output_path = os.path.join(sandbox, step["output"])

        conn   = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(safe_query)

        rows    = cursor.fetchmany(SQL_POLICY["max_rows"])
        headers = [desc[0] for desc in cursor.description]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        cursor.close()
        conn.close()

        msg = f"DB data written to {output_path} ({len(rows)} rows)."
        logger.info(msg)

        return {
            **state,
            "files_created":      state.get("files_created", []) + [output_path],
            "logs":               state.get("logs", []) + [msg],
            "last_step_output":   msg,
            "current_step_index": idx + 1,
            "error": None,
        }

    except Exception as exc:
        msg = f"[DB Executor] Step {idx} failed: {exc}"
        logger.exception(msg)
        return {
            **state,
            "status": "FAILED",
            "error":  msg,
            "files_created": state.get("files_created", []),
        }
    