import os
from urllib.parse import urlparse


def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        return os.getenv(value[1:], "")
    return value


def connect(config: dict):
    import mysql.connector
    conn_str = _resolve_env(config.get("connection_url", ""))
    if not conn_str:
        raise RuntimeError("MySQL: 'connection_url' is not set.")
    p = urlparse(conn_str)
    return mysql.connector.connect(
        host     = p.hostname,
        port     = p.port or 3306,
        user     = p.username,
        password = p.password,
        database = p.path.lstrip("/"),
    )


def execute(conn, operation: str, config: dict):
    cursor = conn.cursor(dictionary=True)
    cursor.execute(operation)
    return cursor


def fetch(cursor) -> tuple:
    rows    = cursor.fetchmany(1000)
    headers = list(cursor.column_names) if rows else []
    return headers, rows


def close(conn):
    conn.close()


# Engine config — imported by database_handler.py
ENGINE = {
    "connect":       connect,
    "execute":       execute,
    "fetch":         fetch,
    "close":         close,
    "requires":      "mysql-connector-python",
    "sql_check":     True,
    "output_format": "csv",
}

# Aliases that point to this engine
ALIASES = ["mariadb"]