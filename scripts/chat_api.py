import os
import sys
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.api_security import (
    MAX_MESSAGES,
    redact_sensitive_text,
    safe_error_message,
    validate_and_sanitize_messages,
    validate_api_base_url,
    validate_generation_params,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "minimind-local:latest"


def build_client_config(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Build validated OpenAI-compatible client configuration from environment variables."""
    source = env if env is not None else os.environ
    raw_base_url = source.get("MINIMIND_API_BASE_URL", DEFAULT_BASE_URL)
    base_url = validate_api_base_url(raw_base_url, allow_local=True)

    api_key = (source.get("MINIMIND_API_KEY") or source.get("OPENAI_API_KEY") or "").strip()
    host = (urlparse(base_url).hostname or "").lower()
    if not api_key:
        if host in ("localhost", "127.0.0.1", "::1"):
            api_key = "local-no-key"
        else:
            raise ValueError("MINIMIND_API_KEY or OPENAI_API_KEY is required for remote endpoints")

    model = (source.get("MINIMIND_MODEL") or DEFAULT_MODEL).strip()[:128] or DEFAULT_MODEL

    raw_temp = float(source.get("MINIMIND_TEMPERATURE", "0.8"))
    raw_top_p = float(source.get("MINIMIND_TOP_P", "0.8"))
    raw_max_tokens = int(source.get("MINIMIND_MAX_TOKENS", "2048"))
    temperature, top_p, max_tokens = validate_generation_params(
        temperature=raw_temp,
        top_p=raw_top_p,
        max_tokens=raw_max_tokens,
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def prepare_chat_messages(
    history: List[Dict[str, Any]],
    user_query: str,
    history_messages_num: int = 0,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sanitize user input, enforce role boundaries, and slice bounded conversation window."""
    if history_messages_num < 0 or history_messages_num % 2 != 0:
        raise ValueError("history_messages_num must be a non-negative even integer")

    window = history[-(min(history_messages_num, MAX_MESSAGES - 2)):] if history_messages_num > 0 else []
    assembled: List[Dict[str, Any]] = []
    if system_prompt:
        assembled.append({"role": "system", "content": system_prompt})
    for msg in window:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            assembled.append({"role": msg["role"], "content": str(msg.get("content", ""))})
    assembled.append({"role": "user", "content": str(user_query)})

    return validate_and_sanitize_messages(assembled, allow_mid_conversation_system=False)


def run_chat_cli(stream: bool = True, history_messages_num: int = 0) -> None:
    from openai import OpenAI

    cfg = build_client_config()
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )
    conversation_history: List[Dict[str, Any]] = []

    while True:
        try:
            query = input('[Q]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        try:
            payload_messages = prepare_chat_messages(
                conversation_history,
                query,
                history_messages_num=history_messages_num,
            )
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=payload_messages,
                stream=stream,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                top_p=cfg["top_p"],
                extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"},
            )
            if not stream:
                if not response.choices or response.choices[0].message is None:
                    raise ValueError("LLM returned empty or filtered response")
                assistant_res = response.choices[0].message.content or ""
                print('[A]: ', assistant_res)
            else:
                print('[A]: ', end='', flush=True)
                assistant_res = ''
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    r = getattr(delta, 'reasoning_content', None) or ""
                    c = delta.content or ""
                    if r:
                        print(f'\033[90m{r}\033[0m', end="", flush=True)
                    if c:
                        print(c, end="", flush=True)
                    assistant_res += c

            conversation_history.append({"role": "user", "content": payload_messages[-1]["content"]})
            conversation_history.append({"role": "assistant", "content": assistant_res})
            if len(conversation_history) > MAX_MESSAGES:
                conversation_history = conversation_history[-MAX_MESSAGES:]
            print('\n\n')
        except Exception as exc:
            print(f"[Error]: {safe_error_message(exc, 'Chat request failed')} ({redact_sensitive_text(type(exc).__name__)})\n")


if __name__ == "__main__":
    run_chat_cli(stream=True, history_messages_num=0)
