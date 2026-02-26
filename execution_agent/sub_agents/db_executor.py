import csv
import logging
import os

import psycopg2
from dotenv import load_dotenv

from execution_agent.policy import EXECUTION_POLICY, SQL_POLICY
from execution_agent.state import WorkflowState

load_dotenv()
logger = logging.getLogger(__name__)



def _enforce_sql_policy(query: str) -> None:
    """Raise ValueError if the query violates SQL_POLICY."""
    upper = query.upper().strip()
    allowed_cmds = SQL_POLICY["allowed_commands"]

    if not any(upper.startswith(cmd) for cmd in allowed_cmds):
        raise ValueError(
            f"Query must start with one of {allowed_cmds}. Got: {upper[:60]}"
        )
    if SQL_POLICY.get("require_limit") and "LIMIT" not in upper:
        raise ValueError("Query must include a LIMIT clause.")



def db_executor_node(state: WorkflowState) -> WorkflowState:
    """
    Handles a single 'database_read' step.
    Reads the step at state['current_step_index'], runs the query,
    writes a CSV to sandbox/runtime, and updates state.
    """
    idx   = state["current_step_index"]
    step  = state["workflow"]["steps"][idx]
    query = step["query"]

    logger.info("[DB Executor] Running query for step %d: %s", idx, query[:80])

    try:
        _enforce_sql_policy(query)

        conn_str = os.getenv("Neon_URL")
        if not conn_str:
            raise RuntimeError("Environment variable 'Neon_URL' is not set.")

        sandbox = EXECUTION_POLICY["sandbox_dir"]
        os.makedirs(sandbox, exist_ok=True)
        output_path = os.path.join(sandbox, step["output"])

        conn   = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(query)

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
            "files_created": state.get("files_created", []) + [output_path],
            "logs":          state.get("logs", []) + [msg],
            "last_step_output": msg,
            "current_step_index": idx + 1,
            "error": None,
        }

    except Exception as exc:
        msg = f"[DB Executor] Step {idx} failed: {exc}"
        logger.exception(msg)
        return {**state, "status": "FAILED", "error": msg, "files_created": state.get("files_created", [])}