"""Typer commands for selecting and authenticating LLM backends."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer

from wren.config import AI_PROVIDERS, AIConfig, load_config, save_ai_config
from wren.model.error import WrenError

ai_app = typer.Typer(name="ai", help="Select and authenticate an LLM backend.")
auth_app = typer.Typer(name="auth", help="Manage OpenAI API authentication.")
ai_app.add_typer(auth_app)


def _wren_home() -> Path:
    return Path(os.environ.get("WREN_HOME", Path.home() / ".wren")).expanduser()


def _load_ai() -> AIConfig:
    return load_config(_wren_home()).ai


def _save(ai: AIConfig) -> None:
    try:
        save_ai_config(_wren_home(), ai)
    except WrenError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@ai_app.command("use")
def use(
    provider: Annotated[
        str,
        typer.Argument(help="Backend: prompt, openai, or codex."),
    ],
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name (OpenAI defaults to gpt-4o-mini).",
        ),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="OpenAI-compatible API base URL."),
    ] = None,
) -> None:
    """Select the LLM backend and optionally set its model."""
    provider = provider.strip().lower()
    if provider not in AI_PROVIDERS:
        typer.echo(
            f"Error: unknown provider {provider!r}. "
            f"Choose: {', '.join(AI_PROVIDERS)}.",
            err=True,
        )
        raise typer.Exit(1)
    try:
        current = _load_ai()
    except WrenError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    selected_model = (
        model.strip()
        if model and model.strip()
        else (current.model if provider == current.provider else None)
    )
    selected_url = base_url.strip() if base_url and base_url.strip() else current.base_url
    new = AIConfig(
        provider=provider,
        model=selected_model,
        base_url=selected_url.rstrip("/"),
    )
    _save(new)
    effective = new.model or (
        "gpt-4o-mini" if provider == "openai" else "CLI default"
    )
    typer.echo(f"LLM backend set to {provider} (model: {effective}).")


@ai_app.command("status")
def status() -> None:
    """Show the selected backend without printing credentials."""
    try:
        ai = _load_ai()
    except WrenError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    model = ai.model or (
        "gpt-4o-mini" if ai.provider == "openai" else "CLI default"
    )
    typer.echo(f"Provider: {ai.provider}")
    typer.echo(f"Model: {model}")
    if ai.provider == "openai":
        from wren.ai import get_openai_api_key  # noqa: PLC0415

        configured = bool(get_openai_api_key())
        typer.echo(f"OPENAI_API_KEY: {'configured' if configured else 'missing'}")
        typer.echo(f"Base URL: {ai.base_url}")
    elif ai.provider == "codex":
        typer.echo("Authentication: Codex CLI login")


def _update_env_file(env_path: Path, key: str, value: str | None) -> None:
    """Set or remove one dotenv key, preserving unrelated lines."""
    lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else []
    )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    kept = [line for line in lines if not pattern.match(line)]
    if value is not None:
        kept.append(f"{key}={value}")
    payload = "\n".join(kept).rstrip() + ("\n" if kept else "")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=env_path.parent, suffix=".env.tmp")
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, env_path)
    except Exception:
        os.unlink(tmp_path)
        raise
    os.chmod(env_path, 0o600)


@auth_app.command("login")
def login() -> None:
    """Save an OpenAI API key in ~/.wren/.env without echoing it."""
    key = typer.prompt("OpenAI API key", hide_input=True).strip()
    if not key:
        typer.echo("Error: API key cannot be empty.", err=True)
        raise typer.Exit(1)
    env_path = _wren_home() / ".env"
    _update_env_file(env_path, "OPENAI_API_KEY", key)
    typer.echo(f"Saved OPENAI_API_KEY to {env_path}.")
    typer.echo(
        "This API key is separate from `codex login` and is used by the "
        "openai backend."
    )


@auth_app.command("logout")
def logout() -> None:
    """Remove the OpenAI API key stored in ~/.wren/.env."""
    env_path = _wren_home() / ".env"
    if not env_path.exists():
        typer.echo("No ~/.wren/.env file found.")
        return
    _update_env_file(env_path, "OPENAI_API_KEY", None)
    typer.echo(f"Removed OPENAI_API_KEY from {env_path}.")


@auth_app.command("status")
def auth_status() -> None:
    """Report whether an OpenAI API key is available."""
    from wren.ai import get_openai_api_key  # noqa: PLC0415

    configured = bool(get_openai_api_key())
    typer.echo(f"OPENAI_API_KEY: {'configured' if configured else 'missing'}")
