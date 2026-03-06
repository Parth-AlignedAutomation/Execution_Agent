import os


def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        return os.getenv(value[1:], "")
    return value

def connect(config: dict):
    import psycopg2
    conn_str = _resolve_env(config.get("connection_url", ""))
    if not conn_str:
        raise RuntimeError("PostgreSQL: 'connection_url' or env var is not set.")
    return psycopg2.connect(conn_str)

def execute(conn, operation: str, config: dict):
    cursor = conn.cursor()
    cursor.execute(operation)
    return cursor

def fetch(cursor) -> tuple:
    max_rows = 1000
    rows     = cursor.fetchmany(max_rows)
    headers  = [d[0] for d in cursor.description]
    return headers, [dict(zip(headers, r)) for r in rows]

def close(conn):
    conn.close()

ENGINE = {
   "connect": connect,
    "execute": execute,
    "fetch": fetch,
    "close": close,
    "requires": "psycopg2-binary",
    "sql_check": True,
    "output_format": "csv",
}

ALIASES = ["postgres", "neon"]