"""Unit tests for the OpenAI and Codex LLM backends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wren.ai import chat_completion, run_codex, run_openai_agent

pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, payload: str):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload.encode()


def test_chat_completion_posts_selected_model_and_auth(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        captured["body"] = request.data
        return _Response('{"choices": [{"message": {"content": "ok"}}]}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = chat_completion(
        [{"role": "user", "content": "hello"}],
        api_key="secret-key",
        model="gpt-4o-mini",
        base_url="https://example.test/v1",
    )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["timeout"] == 120
    assert '"model": "gpt-4o-mini"' in captured["body"].decode()


def test_run_openai_agent_executes_tool_then_returns_final_answer(monkeypatch):
    calls = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "wren_query",
                                        "arguments": '{"sql":"SELECT 1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"role": "assistant", "content": "42"}}]},
        ]
    )

    def fake_completion(messages, **kwargs):
        calls.append((messages, kwargs))
        return next(responses)

    monkeypatch.setattr("wren.ai.chat_completion", fake_completion)
    answer = run_openai_agent(
        "How many?",
        model="gpt-4o-mini",
        api_key="secret-key",
        tools=[{"type": "function"}],
        execute_tool=lambda name, args: {"name": name, "args": args},
    )
    assert answer == "42"
    assert calls[0][0][0]["role"] == "user"
    assert calls[1][0][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"name": "wren_query", "args": {"sql": "SELECT 1"}}',
    }
    assert calls[0][1]["model"] == "gpt-4o-mini"


def test_run_codex_uses_existing_cli_login(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="answer\n", stderr="")

    monkeypatch.setattr("wren.ai.subprocess.run", fake_run)
    assert run_codex("hello", command="codex") == "answer"
    assert captured["args"] == ["codex", "exec", "--skip-git-repo-check", "hello"]
