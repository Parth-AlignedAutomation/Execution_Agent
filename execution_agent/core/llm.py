import logging
import os
import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")


class LLMPromptError(Exception):
    pass

class LLMCallError(Exception):
    pass


def _get_langfuse_client():
    from langfuse import Langfuse
    return Langfuse(
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", ""),
        host       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def fetch_chat_prompt(prompt_name: str, label: str = "latest") -> list[dict]:
    try:
        lf     = _get_langfuse_client()
        prompt = lf.get_prompt(prompt_name, label=label)
        lf.flush()
    except Exception as exc:
        raise LLMPromptError(
            f"Failed to fetch prompt '{prompt_name}' (label='{label}') "
            f"from Langfuse.\n"
            f"Cause: {exc}\n"
            f"Check: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env\n"
            f"Check: prompt '{prompt_name}' exists in Langfuse with label '{label}'"
        )

    if isinstance(prompt.prompt, list):
        messages = prompt.prompt
    else:
        messages = [{"role": "user", "content": prompt.prompt}]

    logger.info(
        "[LLM] Fetched Langfuse prompt '%s' (label=%s, v%s) — %d message(s)",
        prompt_name, label, prompt.version, len(messages),
    )
    return messages


def compile_messages(messages: list[dict], variables: dict) -> list[dict]:
    compiled = []
    for msg in messages:
        content = msg["content"]
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
        compiled.append({"role": msg["role"], "content": content})
    return compiled


def call_ollama(messages: list[dict]) -> str:
    url = f"{OLLAMA_HOST}/v1/chat/completions"
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "model":      OLLAMA_MODEL,
                "messages":   messages,
                "max_tokens": 1024,
                "stream":     False,
            },
            timeout=120,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        logger.info("[LLM] Ollama response received (%d chars).", len(text))
        return text.strip()

    except requests.exceptions.ConnectionError:
        raise LLMCallError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}.\n"
            f"Fix: run 'ollama serve' in a separate terminal."
        )
    except Exception as exc:
        raise LLMCallError(
            f"Ollama call failed: {exc}\n"
            f"Model: {OLLAMA_MODEL} | Host: {OLLAMA_HOST}"
        )


def run_llm_node(prompt_name: str, variables: dict) -> str:
    messages = fetch_chat_prompt(prompt_name)
    compiled = compile_messages(messages, variables)
    return call_ollama(compiled)