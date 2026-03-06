import os
import sqlite3


def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        return os.getenv(value[1:], "")
    return value


def connect(config: dict):
    db_path = _resolve_env(config.get("db_path", ""))
    if not db_path:
        raise RuntimeError("SQLite: 'db_path' is not set.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def execute(conn, operation: str, config: dict):
    cursor = conn.cursor()
    cursor.execute(operation)
    return cursor


def fetch(cursor) -> tuple:
    rows    = cursor.fetchmany(1000)
    headers = [d[0] for d in cursor.description] if cursor.description else []
    return headers, [dict(zip(headers, r)) for r in rows]


def close(conn):
    conn.close()



ENGINE = {
    "connect":       connect,
    "execute":       execute,
    "fetch":         fetch,
    "close":         close,
    "requires":      "built-in (no install needed)",
    "sql_check":     True,
    "output_format": "csv",
}


ALIASES = []