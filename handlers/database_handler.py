import csv
import logging
import os
import json

from handlers.base_handler import BaseHandler
from handlers.registry import registry
from handlers.database import DB_ENGINES

logger = logging.getLogger(__name__)
SANDBOX_DIR = os.getenv("SANDBOX_DIR", "sandbox/runtime")

class DatabaseHandler(BaseHandler):

    @property
    def name(self) -> str:
        return "database_read"
    
    def execute(self, step: dict, state: dict) -> dict:
        engine_key = step.get("engine", "postgres").lower()

        engine_cfg = DB_ENGINES.get(engine_key)

        if engine_cfg and "alias" in engine_cfg:
            engine_key = engine_cfg["alias"]
            engine_cfg = DB_ENGINES.get(engine_key)

        if not engine_cfg or "connect" not in engine_cfg:
            available = [k for k in DB_ENGINES if "alias" not in DB_ENGINES[k]]
            return {
                **state, "status": "FAILED",
                "error": (
                    f"[DatabaseHandler] Unknown engine '{engine_key}'. "
                    f"Available: {available}"
                ),
            }   
        
        if engine_cfg.get("sql_check") and "query" in step:
            from execution_agent.guardrails.sql_safety import SQLSafetyGuardrail
            result = SQLSafetyGuardrail().check(step["query"])
            if not result.passed:
                msg = f"[SQL Guardrail] BLOCKED — Rule: '{result.rule}' | {result.reason}"
                logger.error(msg)
                return {
                    **state, "status": "FAILED", "error": msg,
                    "files_created": state.get("files_created", []),
                }

        conn = None


        try:
            logger.info("[DatabaseHandler] Engine: %s | Step: %s", engine_key, step.get("id", ""))

            conn = engine_cfg["connect"](step)
            operation = step.get("query") or step.get("filter", "{}")
            cursor = engine_cfg["execute"](conn, operation, step)
            headers, rows = engine_cfg["fetch"](cursor)
            engine_cfg["close"](conn)
            conn = None

            # Save output to sandbox
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
                "error":              None,
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

            return{
                **state, "status": "FAILED", "error": msg,
                "files_created": state.get("files_created", []),
            }
def _save(path: str, headers: str, rows: list, fmt: str) -> None:
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

