import logging
import os
from dataclasses import dataclass
from typing import Optional
import yaml
import re
logger = logging.getLogger(__name__)

_POLICY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "policies",
    "sql_guardrail_policy.yaml",
)


@dataclass
class GuardrailResult:
    passed : bool
    rule : Optional[str] = None
    reason : Optional[str] = None
    sanitised_query : Optional[str] = None


class  SQLSafetyGuardrail:
    def __init__(self, policy_path: str = _POLICY_PATH):
        with open(os.path.normpath(policy_path)) as f:
            self.policy = yaml.safe_load(f)
        logging.info("[SQL Guardrail] Policy loaded from %s", policy_path)


    def check(self, query: str) -> GuardrailResult:
        logger.info("[SQL Guardrail] Checking query: %s", query[:120])
        clean = self._strip_comments(query) if self.policy.get("strip_comments", True) else query
        clean = clean.strip()

        if self.policy.get("blocked_multiple_statements", True):
            result = self._check_multiple_statements(clean)
            if not result.passed:
                return result
            
        result = self._check_allowed_commands(clean)
        if not result.passed:
            return result

        result  = self._check_dangerous_keywords(clean)
        if not result.passed:
            return result
        
        if self.policy.get("require_limit", True):
            result = self._check_limit_present(clean)
            if not result.passed:
                return result

        # Step 6 — LIMIT value
        result = self._check_limit_value(clean)
        if not result.passed:
            return result

        logger.info("[SQL Guardrail] Query passed all checks.")
        return GuardrailResult(passed=True, sanitised_query=clean)

    def _strip_comments(self, query: str) -> str:
        query = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
        query = re.sub(r"--[^\n]*", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        logger.debug("[SQL Guardrail] After comment strip: %s", query[:120])
        return query
    
    def _check_multiple_statements(self, query: str) -> GuardrailResult:
        stripped = query.rstrip().rstrip(";")
        if ";" in stripped:
            return GuardrailResult(
                passed=False,
                rule="block_multiple_statements",
                reason=(
                    "Query contains multiple statements separated by ';'. "
                    "Only a single SELECT statement is allowed."
                ),
            )
        return GuardrailResult(passed=True)
    


    def _check_allowed_commands(self, query: str) -> GuardrailResult:
        allowed = [cmd.upper() for cmd in self.policy.get("allowed_commands", ["SELECT"])]
        first_token = query.upper().split()[0] if query.split() else ""

        if first_token not in allowed:
            return GuardrailResult(
                passed=False,
                rule="allowed_commands",
                reason=(
                    f"Query starts with '{first_token}' which is not in the "
                    f"allowed command list: {allowed}."
                ),
            )
        return GuardrailResult(passed=True)
    
    def _check_dangerous_keywords(self, query: str) -> GuardrailResult:
        dangerous = [kw.upper() for kw in self.policy.get("dangerous_keywords", [])]
        query_upper = query.upper()

        for keyword in dangerous:
            # \b word boundary — won't match 'updated_at' for keyword 'UPDATE'
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, query_upper):
                return GuardrailResult(
                    passed=False,
                    rule="dangerous_keywords",
                    reason=(
                        f"Query contains forbidden keyword '{keyword}'. "
                        "Only read-only SELECT queries are permitted."
                    ),
                )
        return GuardrailResult(passed=True)
    

    def _check_limit_present(self, query: str) -> GuardrailResult:
        if not re.search(r"\bLIMIT\b", query.upper()):
            return GuardrailResult(
                passed=False,
                rule="require_limit",
                reason=(
                    "Query is missing a LIMIT clause. "
                    "All queries must include LIMIT to prevent unbounded result sets."
                ),
            )
        return GuardrailResult(passed=True)
    
    def _check_limit_value(self, query: str) -> GuardrailResult:
        """LIMIT N value must not exceed max_limit_value."""
        max_val = self.policy.get("max_limit_value", 1000)
        match   = re.search(r"\bLIMIT\s+(\d+)", query.upper())

        if match:
            limit_val = int(match.group(1))
            if limit_val > max_val:
                return GuardrailResult(
                    passed=False,
                    rule="max_limit_value",
                    reason=(
                        f"LIMIT {limit_val} exceeds the maximum allowed value of {max_val}. "
                        f"Please reduce your LIMIT to {max_val} or less."
                    ),
                )
        return GuardrailResult(passed=True)