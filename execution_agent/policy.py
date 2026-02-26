import yaml
import os

_BASE = os.path.join(os.path.dirname(__file__), "policies")

def _load(filename: str) -> dict:
    with open(os.path.join(_BASE, filename)) as f:
        return yaml.safe_load(f)

EXECUTION_POLICY = _load("execution_policy.yaml")
SQL_POLICY       = _load("sql_policy.yaml")
SCRIPT_POLICY    = _load("script_policy.yaml")

ALLOWED_EMAILS = ["team@example.com"]