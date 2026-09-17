"""Unit tests for LLM provider selection and API-key management."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from wren.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_ai_use_openai_persists_model(monkeypatch, tmp_path):
    monkeypatch.setenv("WREN_HOME", str(tmp_path))
    result = runner.invoke(app, ["ai", "use", "openai", "--model", "gpt-4o-mini"])
    assert result.exit_code == 0, result.output
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["ai"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
    }


def test_ai_use_different_provider_does_not_reuse_model(monkeypatch, tmp_path):
    monkeypatch.setenv("WREN_HOME", str(tmp_path))
    runner.invoke(app, ["ai", "use", "openai", "--model", "gpt-4o-mini"])
    result = runner.invoke(app, ["ai", "use", "codex"])
    assert result.exit_code == 0, result.output
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["ai"]["provider"] == "codex"
    assert raw["ai"]["model"] is None


def test_ai_auth_login_writes_key_without_echoing_it(monkeypatch, tmp_path):
    monkeypatch.setenv("WREN_HOME", str(tmp_path))
    result = runner.invoke(app, ["ai", "auth", "login"], input="sk-test-secret\n")
    assert result.exit_code == 0, result.output
    assert "sk-test-secret" not in result.output
    assert "OPENAI_API_KEY=sk-test-secret" in (tmp_path / ".env").read_text()


def test_ai_auth_logout_removes_only_openai_key(monkeypatch, tmp_path):
    monkeypatch.setenv("WREN_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("OTHER=value\nOPENAI_API_KEY=secret\n")
    result = runner.invoke(app, ["ai", "auth", "logout"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".env").read_text() == "OTHER=value\n"
