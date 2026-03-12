import logging
import os
import subprocess
import sys
import platform

from execution_agent.handlers.base_handler import BaseHandler
from execution_agent.handlers.registry import registry


logger = logging.getLogger(__name__)

SCRIPTS_DIR      = os.getenv("SCRIPTS_DIR", "scripts")
DEFAULT_TIMEOUT  = 3000


def _python_cmd(script_path: str) -> list:
    # sys.executable = current venv Python — always has installed packages
    return [sys.executable, script_path]

def _r_cmd(script_path: str) -> list:
    return ["Rscript", script_path]

def _shell_cmd(script_path: str) -> list:
    if platform.system() == "Windows":
        # Try Git Bash on Windows
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        return [git_bash if os.path.exists(git_bash) else "bash", script_path]
    return ["/bin/bash", script_path]

def _node_cmd(script_path: str) -> list:
    return ["node", script_path]


SCRIPT_RUNNERS = {
    "python": {
        "cmd":      _python_cmd,
        "requires": "Python (current venv)",
    },
    "r": {
        "cmd":      _r_cmd,
        "requires": "R — install from https://cran.r-project.org/",
    },
    "shell": {
        "cmd":      _shell_cmd,
        "requires": "bash (Linux/Mac built-in) or Git Bash on Windows",
    },
    "bash": {"alias": "shell"},   # alias

    "node": {
        "cmd":      _node_cmd,
        "requires": "Node.js — install from https://nodejs.org/",
    },
    "javascript": {"alias": "node"},   # alias

}


class ScriptHandler(BaseHandler):
    @property
    def name(self) -> str:
        return "script_execution"
    
    def execute(self, step: dict, state:dict) -> dict:
        runner_key = step.get("runner", "python").lower()

        runner_cfg = SCRIPT_RUNNERS.get(runner_key)

        if runner_cfg and "alias" in runner_cfg:
            runner_key = runner_cfg["alias"]
            runner_cfg = SCRIPT_RUNNERS.get(runner_key)

        if not runner_cfg:
            return {
                **state, "status": "FAILED",
                "error": (
                    f"[ScriptHandler] Unknown runner '{runner_key}'. "
                    f"Available: {[k for k in SCRIPT_RUNNERS if 'alias' not in SCRIPT_RUNNERS[k]]}"
                ),
            }

        script_name = step.get("script", "")
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        inputs      = step.get("inputs",  [])
        outputs     = step.get("outputs", [])
        timeout     = int(step.get("timeout", DEFAULT_TIMEOUT))

        if not os.path.exists(script_path):
            msg = f"[ScriptHandler] Script not found: {script_path}"
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}

        cmd = runner_cfg["cmd"](script_path) + inputs + outputs
        logger.info("[ScriptHandler] Runner: %s | CMD: %s", runner_key, " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.info("[ScriptHandler] stdout:\n%s", result.stdout.strip())

            msg = f"Script '{script_name}' completed. Outputs: {outputs}"
            logger.info(msg)

            return {
                **state,
                "files_created":      state.get("files_created", []) + outputs,
                "logs":               state.get("logs", []) + [msg],
                "last_step_output":   msg,
                "current_step_index": state.get("current_step_index", 0) + 1,
                "error": None,
            }

        except FileNotFoundError:
            req = runner_cfg.get("requires", runner_key)
            msg = f"[ScriptHandler] Runner '{runner_key}' not found. Setup: {req}"
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}

        except subprocess.TimeoutExpired:
            msg = f"[ScriptHandler] Script '{script_name}' timed out after {timeout}s."
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}

        except subprocess.CalledProcessError as e:
            msg = f"[ScriptHandler] Script '{script_name}' failed.\nstderr: {e.stderr}"
            logger.error(msg)
            return {**state, "status": "FAILED", "error": msg}


# Auto-register
registry.register(ScriptHandler())