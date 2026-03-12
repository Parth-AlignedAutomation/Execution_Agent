import json
import os


def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        return os.getenv(value[1:], "")
    return value


def connect(config: dict):
    from pymongo import MongoClient
    conn_str = _resolve_env(config.get("connection_url", ""))
    if not conn_str:
        raise RuntimeError("MongoDB: 'connection_url' is not set.")
    client = MongoClient(conn_str)
    # Store db/collection on client object for execute() to use
    client._exec_db         = config["database"]
    client._exec_collection = config["collection"]
    return client


def execute(conn, operation: str, config: dict):
    flt        = json.loads(operation) if operation.strip() else {}
    max_rows   = int(config.get("max_rows", 1000))
    collection = conn[conn._exec_db][conn._exec_collection]
    return list(collection.find(flt).limit(max_rows))


def fetch(docs) -> tuple:
    if not docs:
        return [], []
    cleaned = [{k: v for k, v in d.items() if k != "_id"} for d in docs]
    headers = list(cleaned[0].keys())
    return headers, cleaned


def close(conn):
    conn.close()


# Engine config — imported by database_handler.py
ENGINE = {
    "connect":       connect,
    "execute":       execute,
    "fetch":         fetch,
    "close":         close,
    "requires":      "pymongo",
    "sql_check":     False,       # MongoDB uses filter, not SQL
    "output_format": "json",
}

# Aliases that point to this engine
ALIASES = ["mongo"]