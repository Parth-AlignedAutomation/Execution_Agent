import csv
import json
import logging
import os
from urllib.parse import urlparse
from handlers.base_handler import BaseHandler
# from dotenv import load_dotenv
# load_dotenv()
from handlers.registry import registry

logger = logging.getLogger(__name__)


SANDBOX_DIR = os.getenv("SANDBOX_DIR", "sandbox/runtime")


def _resolve_env(value: str) -> str:
    """Resolve $ENV_VAR_NAME to actual env value."""
    if isinstance(value, str) and value.startswith("$"):
        return os.getenv(value[1:], "")
    return value


#postgres sql

def _pg_connect(config: dict):
    import psycopg2
    conn_str = _resolve_env(config.get("connection_url", ""))
    if not conn_str:
        raise RuntimeError("PostgreSQL: 'connection_url' or env var is not set.")
    return psycopg2.connect(conn_str)

def _pg_execute(conn, operation: str, config: dict):
    cursor = conn.cursor()
    cursor.execute(operation)
    return cursor

def _pg_fetch(cursor) -> tuple:
    max_rows = 1000
    rows     = cursor.fetchmany(max_rows)
    headers  = [d[0] for d in cursor.description]
    return headers, [dict(zip(headers, r)) for r in rows]

def _pg_close(conn):
    conn.close()


def _mysql_connect(config: dict):
    import mysql.connector
    conn_str = _resolve_env(config.get("connection_url", ""))
    if not conn_str:
        raise RuntimeError("MySQL: 'connection_url' is not set.")
    p = urlparse(conn_str)
    return mysql.connector.connect(
        host=p.hostname, port=p.port or 3306,
        user=p.username, password=p.password,
        database=p.path.lstrip("/"),
    )

def _mysql_execute(conn, operation: str, config: dict):
    cursor = conn.cursor(dictionary=True)
    cursor.execute(operation)
    return cursor

def _mysql_fetch(cursor) -> tuple:
    rows    = cursor.fetchmany(1000)
    headers = list(cursor.column_names) if rows else []
    return headers, rows

def _mysql_close(conn):
    conn.close()


# ── SQLite ────────────────────────────────────────────────────────────────────

def _sqlite_connect(config: dict):
    import sqlite3
    db_path = _resolve_env(config.get("db_path", ""))
    if not db_path:
        raise RuntimeError("SQLite: 'db_path' is not set.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def _sqlite_execute(conn, operation: str, config: dict):
    cursor = conn.cursor()
    cursor.execute(operation)
    return cursor

def _sqlite_fetch(cursor) -> tuple:
    rows    = cursor.fetchmany(1000)
    headers = [d[0] for d in cursor.description] if cursor.description else []
    return headers, [dict(zip(headers, r)) for r in rows]

def _sqlite_close(conn):
    conn.close()


# ── MongoDB ───────────────────────────────────────────────────────────────────

def _mongo_connect(config: dict):
    from pymongo import MongoClient
    conn_str = _resolve_env(config.get("connection_url", ""))
    if not conn_str:
        raise RuntimeError("MongoDB: 'connection_url' is not set.")
    client = MongoClient(conn_str)
    # Store collection reference on client for later use
    client._exec_db         = config["database"]
    client._exec_collection = config["collection"]
    return client

def _mongo_execute(conn, operation: str, config: dict):
    import json as _json
    flt        = _json.loads(operation) if operation.strip() else {}
    max_rows   = int(config.get("max_rows", 1000))
    collection = conn[conn._exec_db][conn._exec_collection]
    return list(collection.find(flt).limit(max_rows))

def _mongo_fetch(docs) -> tuple:
    if not docs:
        return [], []
    cleaned = [{k: v for k, v in d.items() if k != "_id"} for d in docs]
    headers = list(cleaned[0].keys())
    return headers, cleaned

def _mongo_close(conn):
    conn.close()


# ── BigQuery ──────────────────────────────────────────────────────────────────

def _bq_connect(config: dict):
    from google.cloud import bigquery
    creds_path = os.getenv(config.get("credentials_env", "GOOGLE_APPLICATION_CREDENTIALS"), "")
    if creds_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    return bigquery.Client(project=config.get("project_id"))

def _bq_execute(conn, operation: str, config: dict):
    job = conn.query(operation)
    return list(job.result())

def _bq_fetch(results) -> tuple:
    if not results:
        return [], []
    headers = list(results[0].keys())
    rows    = [dict(r) for r in results]
    return headers, rows

def _bq_close(conn):
    conn.close()
  

DB_ENGINES = {
    # ── PostgreSQL (and its aliases) ──────────────────────────────────────────
    "postgres":   {
        "connect":       _pg_connect,
        "execute":       _pg_execute,
        "fetch":         _pg_fetch,
        "close":         _pg_close,
        "requires":      "psycopg2-binary",
        "sql_check":     True,
        "output_format": "csv",
    },
    "postgresql": {"alias": "postgres"},
    "neon":       {"alias": "postgres"},

    # ── MySQL / MariaDB ───────────────────────────────────────────────────────
    "mysql":   {
        "connect":       _mysql_connect,
        "execute":       _mysql_execute,
        "fetch":         _mysql_fetch,
        "close":         _mysql_close,
        "requires":      "mysql-connector-python",
        "sql_check":     True,
        "output_format": "csv",
    },
    "mariadb": {"alias": "mysql"},

    # ── SQLite ────────────────────────────────────────────────────────────────
    "sqlite": {
        "connect":       _sqlite_connect,
        "execute":       _sqlite_execute,
        "fetch":         _sqlite_fetch,
        "close":         _sqlite_close,
        "requires":      "built-in (no install needed)",
        "sql_check":     True,
        "output_format": "csv",
    },

    # ── MongoDB ───────────────────────────────────────────────────────────────
    "mongodb": {
        "connect":       _mongo_connect,
        "execute":       _mongo_execute,
        "fetch":         _mongo_fetch,
        "close":         _mongo_close,
        "requires":      "pymongo",
        "sql_check":     False,
        "output_format": "json",
    },
    "mongo": {"alias": "mongodb"},

    # ── BigQuery ──────────────────────────────────────────────────────────────
    "bigquery": {
        "connect":       _bq_connect,
        "execute":       _bq_execute,
        "fetch":         _bq_fetch,
        "close":         _bq_close,
        "requires":      "google-cloud-bigquery",
        "sql_check":     True,
        "output_format": "csv",
    },

}

class DatabaseHandler(BaseHandler):

    @property
    def name(self) -> str:
        return "database_read"

    def execute(self, step: dict, state: dict) -> dict:
        engine_key = step.get("engine", "postgres").lower()

        # Resolve alias
        engine_cfg = DB_ENGINES.get(engine_key)
        if engine_cfg and "alias" in engine_cfg:
            engine_key = engine_cfg["alias"]
            engine_cfg = DB_ENGINES.get(engine_key)

        if not engine_cfg:
            return {
                **state, "status": "FAILED",
                "error": (
                    f"[DatabaseHandler] Unknown engine '{engine_key}'. "
                    f"Available: {[k for k in DB_ENGINES if 'alias' not in DB_ENGINES[k]]}"
                ),
            }

        # SQL guardrail for SQL-based engines
        if engine_cfg.get("sql_check") and "query" in step:
            from execution_agent.guardrails.sql_safety import SQLSafetyGuardrail
            result = SQLSafetyGuardrail().check(step["query"])
            if not result.passed:
                msg = f"[SQL Guardrail] BLOCKED — Rule: '{result.rule}' | {result.reason}"
                logger.error(msg)
                return {**state, "status": "FAILED", "error": msg,
                        "files_created": state.get("files_created", [])}

        conn = None
        try:
            logger.info("[DatabaseHandler] Engine: %s | Step: %s", engine_key, step.get("id", ""))

            conn      = engine_cfg["connect"](step)
            operation = step.get("query") or step.get("filter", "{}")
            cursor    = engine_cfg["execute"](conn, operation, step)
            headers, rows = engine_cfg["fetch"](cursor)
            engine_cfg["close"](conn)
            conn = None

            # Save output
            output_file   = step.get("output", "output.csv")
            output_path   = os.path.join(SANDBOX_DIR, output_file)
            output_format = engine_cfg.get("output_format", "csv")
            os.makedirs(SANDBOX_DIR, exist_ok=True)

            _save(output_path, headers, rows, output_format)

            msg = f"[DatabaseHandler] {len(rows)} rows saved → {output_path}"
            logger.info(msg)

            return {
                **state,
                "files_created":      state.get("files_created", []) + [output_path],
                "logs":               state.get("logs", []) + [msg],
                "last_step_output":   msg,
                "current_step_index": state.get("current_step_index", 0) + 1,
                "error": None,
            }

        except ImportError as exc:
            pkg = engine_cfg.get("requires", "unknown")
            msg = f"[DatabaseHandler] Missing package for '{engine_key}'. Run: pip install {pkg}. Error: {exc}"
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}

        except Exception as exc:
            if conn:
                try:
                    engine_cfg["close"](conn)
                except Exception:
                    pass
            msg = f"[DatabaseHandler] Failed: {exc}"
            logger.exception(msg)
            return {**state, "status": "FAILED", "error": msg,
                    "files_created": state.get("files_created", [])}


def _save(path: str, headers: list, rows: list, fmt: str) -> None:
    if fmt == "json":
        actual_path = path.replace(".csv", ".json")
        with open(actual_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


# Auto-register
registry.register(DatabaseHandler())