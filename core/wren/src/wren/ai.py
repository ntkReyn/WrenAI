"""LLM backends used by ``wren ask``.

The OpenAI path deliberately uses the standard library HTTP client. This keeps
the base Wren installation small while still supporting any OpenAI-compatible
Chat Completions endpoint. API keys are read from ``OPENAI_API_KEY`` after the
same dotenv discovery used by connection profiles.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from wren.config import DEFAULT_OPENAI_BASE_URL, AIConfig

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
MAX_AGENT_TURNS = 8


class AIError(RuntimeError):
    """A user-facing error from an LLM backend or agent loop."""


def effective_model(settings: AIConfig) -> str | None:
    """Return the configured model, applying the API default when needed."""
    if settings.model:
        return settings.model
    if settings.provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return None


def get_openai_api_key() -> str | None:
    """Return the API key from the shell or discovered dotenv files."""
    # Reuse profile dotenv discovery so ~/.wren/.env and project .env behave
    # identically for database and LLM credentials.
    from wren.profile import _ensure_env_loaded  # noqa: PLC0415

    _ensure_env_loaded()
    value = os.environ.get("OPENAI_API_KEY")
    return value.strip() if value and value.strip() else None


def _endpoint(base_url: str) -> str:
    base = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise AIError("OpenAI base URL must start with http:// or https://.")
    return f"{base}/chat/completions"


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    """Call the Chat Completions endpoint and return its decoded response."""
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    request = urllib.request.Request(
        _endpoint(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except OSError:
            detail = str(exc)
        raise AIError(
            f"OpenAI API returned HTTP {exc.code}: {detail[:2000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AIError(f"Could not reach the OpenAI API: {exc.reason}") from exc
    except TimeoutError as exc:
        raise AIError("The OpenAI API request timed out.") from exc
    except OSError as exc:
        raise AIError(f"Could not call the OpenAI API: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIError("The OpenAI API returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise AIError("The OpenAI API returned an unexpected response.")
    if result.get("error"):
        raise AIError(f"OpenAI API error: {result['error']}")
    return result


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(parts)
    return ""


def run_openai_agent(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    tools: list[dict[str, Any]] | None = None,
    execute_tool: Callable[[str, dict[str, Any]], Any] | None = None,
    system_prompt: str | None = None,
    max_turns: int = MAX_AGENT_TURNS,
) -> str:
    """Run a small tool-calling loop and return the final assistant text."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for _ in range(max_turns):
        result = chat_completion(
            messages,
            api_key=api_key,
            model=model,
            base_url=base_url,
            tools=tools,
        )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIError("The OpenAI API returned no assistant choices.")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise AIError("The OpenAI API returned an invalid assistant message.")

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            text = _message_text(message).strip()
            if not text:
                raise AIError("The OpenAI API returned an empty assistant response.")
            return text

        messages.append(message)
        if execute_tool is None:
            raise AIError("The model requested a tool, but no tool runtime is available.")

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str):
                continue
            raw_args = function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            try:
                tool_result = execute_tool(name, args)
            except Exception as exc:  # tool errors are recoverable by the model
                tool_result = {"ok": False, "error": str(exc)}
            try:
                content = json.dumps(tool_result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                content = json.dumps({"ok": False, "error": str(tool_result)})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": content,
                }
            )

    raise AIError(
        f"The model used all {max_turns} tool turns without returning an answer."
    )


def run_codex(
    prompt: str,
    *,
    model: str | None = None,
    command: str | None = None,
    timeout: float = 600,
) -> str:
    """Run the locally authenticated Codex CLI in non-interactive mode."""
    executable = command or os.environ.get("WREN_CODEX_COMMAND", "codex")
    args = [executable, "exec", "--skip-git-repo-check"]
    if model:
        args.extend(["--model", model])
    args.append(prompt)
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AIError(
            f"Codex CLI not found ({executable!r}). Install it or use "
            "`wren ai use openai`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AIError("The Codex CLI request timed out.") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise AIError(
            f"Codex CLI failed with exit code {completed.returncode}"
            + (f": {detail[:2000]}" if detail else ".")
        )
    output = completed.stdout.strip()
    if not output:
        raise AIError("Codex CLI returned an empty response.")
    return output


__all__ = [
    "AIError",
    "DEFAULT_OPENAI_MODEL",
    "MAX_AGENT_TURNS",
    "chat_completion",
    "effective_model",
    "get_openai_api_key",
    "run_codex",
    "run_openai_agent",
]
