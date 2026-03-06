import os


def connect(config: dict):
    from google.cloud import bigquery
    creds_path = os.getenv(
        config.get("credentials_env", "GOOGLE_APPLICATION_CREDENTIALS"), ""
    )
    if creds_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    return bigquery.Client(project=config.get("project_id"))


def execute(conn, operation: str, config: dict):
    job = conn.query(operation)
    return list(job.result())


def fetch(results) -> tuple:
    if not results:
        return [], []
    headers = list(results[0].keys())
    rows    = [dict(r) for r in results]
    return headers, rows


def close(conn):
    conn.close()


# Engine config — imported by database_handler.py
ENGINE = {
    "connect":       connect,
    "execute":       execute,
    "fetch":         fetch,
    "close":         close,
    "requires":      "google-cloud-bigquery",
    "sql_check":     True,
    "output_format": "csv",
}

# No aliases
ALIASES = []