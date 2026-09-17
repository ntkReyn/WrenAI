"""Tests for `wren ask` prompt shaping."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import wren.ai
import wren.ask_cli
from wren import ask as ask_mod
from wren.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_no_mode_flag_rejected():
    result = runner.invoke(app, ["ask", "show me revenue"])
    assert result.exit_code != 0
    out = result.output + (result.stderr if result.stderr_bytes else "")
    assert "--guided" in out and "--direct" in out


def test_both_mode_flags_rejected():
    result = runner.invoke(app, ["ask", "show me revenue", "--guided", "--direct"])
    assert result.exit_code != 0


def test_guided_includes_task_flow_and_substitutes_prompt():
    result = runner.invoke(app, ["ask", "top 5 customers by revenue", "--guided"])
    assert result.exit_code == 0
    assert "TASK TYPE A" in result.output
    assert "TASK TYPE B" in result.output
    assert "wren context show" in result.output
    assert "top 5 customers by revenue" in result.output
    assert "<USER_PROMPT>" not in result.output  # placeholder substituted


def test_direct_minimal_and_substitutes_prompt():
    result = runner.invoke(app, ["ask", "monthly orders trend", "--direct"])
    assert result.exit_code == 0
    assert "wren skills list" in result.output
    assert "wren --help" in result.output
    assert "monthly orders trend" in result.output
    assert "<USER_PROMPT>" not in result.output
    # direct mode should NOT include the guided TASK TYPE structure
    assert "TASK TYPE A" not in result.output


def test_render_api_known_modes():
    for mode in ask_mod.MODES:
        out = ask_mod.render(mode, "hello world")
        assert "hello world" in out
        assert "<USER_PROMPT>" not in out


def test_render_unknown_mode_raises():
    with pytest.raises(ask_mod.UnknownAskModeError):
        ask_mod.render("auto", "anything")


def test_guided_recall_step_uses_a_real_cli_option():
    # The guided template's step 2 must emit an option `wren memory recall`
    # actually accepts. Regression for #2503: the template said `--nl`, an
    # option that belongs to `wren memory store`, not `recall`.
    result = runner.invoke(app, ["ask", "x", "--guided"])
    assert "--nl" not in result.output

    recall_line = next(
        line for line in result.output.splitlines() if "wren memory recall" in line
    )
    args = recall_line.split("wren memory recall", 1)[1].split("#", 1)[0].split()
    recall_result = runner.invoke(app, ["memory", "recall", *args])
    # A bad option is a click usage error (exit code 2, "No such option").
    # Any other exit code means the option parsed and the command moved on
    # to its own logic (e.g. failing later for lacking a wren project).
    assert recall_result.exit_code != 2, recall_result.output


def test_user_prompt_with_template_placeholder_substring_is_safe():
    # Prompt containing the literal placeholder shouldn't break rendering;
    # we only do one replacement of the bundled-template placeholder.
    prompt = "Show literal <USER_PROMPT> usage examples"
    out = ask_mod.render("direct", prompt)
    # the bundled placeholder is gone and the prompt is present (verbatim)
    assert prompt in out


def test_openai_provider_calls_selected_model(monkeypatch, tmp_path):
    monkeypatch.setattr(wren.ask_cli, "_WREN_HOME", tmp_path)
    monkeypatch.setattr(wren.ai, "get_openai_api_key", lambda: "sk-test")
    captured = {}

    def fake_agent(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "API answer"

    monkeypatch.setattr(wren.ai, "run_openai_agent", fake_agent)
    result = runner.invoke(
        app,
        [
            "ask",
            "show me revenue",
            "--direct",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "API answer"
    assert captured["kwargs"]["model"] == "gpt-4o-mini"
    assert captured["kwargs"]["api_key"] == "sk-test"


def test_wren_runtime_exposes_context_and_memory_store(tmp_path):
    project = tmp_path
    (project / "wren_project.yml").write_text(
        "schema_version: 2\nname: test\ndata_source: duckdb\n"
    )
    model_dir = project / "models" / "orders"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.yml").write_text(
        "name: orders\n"
        "table_reference:\n  table: orders\n"
        "columns:\n  - name: id\n    type: INTEGER\n"
    )

    tools, execute, engine = wren.ask_cli._build_wren_runtime(project)
    names = {
        spec["function"]["name"]
        for spec in tools
    }
    assert names == {
        "wren_context_show",
        "wren_memory_recall",
        "wren_memory_fetch",
        "wren_dry_plan",
        "wren_query",
        "wren_memory_store",
    }
    assert engine == [None]
    context = execute("wren_context_show", {})
    assert context["manifest"]["models"][0]["name"] == "orders"
    stored = execute(
        "wren_memory_store",
        {"question": "How many orders?", "sql": "SELECT COUNT(*) FROM orders"},
    )
    assert stored["stored"] is True
    assert (project / stored["path"]).exists()
