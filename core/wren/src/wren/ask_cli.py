"""``wren ask`` — prompt shaping plus optional LLM execution.

With no configured provider, this command keeps its original behaviour and
prints a prompt for an external agent. ``wren ai use openai`` enables the
direct API agent loop; ``wren ai use codex`` delegates to a locally logged-in
Codex CLI.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from wren import ask as _ask
from wren.config import AI_PROVIDERS, AIConfig, load_config
from wren.model.error import WrenError

_WREN_HOME = Path(os.environ.get("WREN_HOME", Path.home() / ".wren")).expanduser()


def _normalize_json(value: Any) -> Any:
    """Convert common database scalar values to JSON-native values."""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if hasattr(value, "item"):
        return _normalize_json(value.item())
    if isinstance(value, dict):
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value


def _table_result(table: Any) -> dict[str, Any]:
    """Serialize an Arrow table without importing the optional MCP package."""
    columns = [field.name for field in table.schema]
    return {
        "columns": columns,
        "rows": [
            {key: _normalize_json(value) for key, value in row.items()}
            for row in table.to_pylist()
        ],
        "row_count": table.num_rows,
    }


def _tool_spec(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


def _build_wren_runtime(project: Path | None):
    """Build OpenAI tool definitions and a local Wren tool dispatcher."""
    if project is None:
        return [], None, None

    from wren import context  # noqa: PLC0415

    if not project.exists():
        raise ValueError(f"Wren project not found: {project}")
    manifest = context.build_json(project)
    rules, _ = context.load_rules(project)
    target = project / "target" / "mdl.json"
    engine: list[Any | None] = [None]
    memory_index: list[Any | None] = [None]

    def get_engine():
        if engine[0] is None:
            if not target.exists():
                raise ValueError(
                    f"Compiled MDL not found: {target}. Run `wren context build` first."
                )
            from wren.cli import _build_engine  # noqa: PLC0415

            engine[0] = _build_engine(str(target), None, None, conn_required=True)
        return engine[0]

    def get_memory_index():
        if memory_index[0] is None:
            from wren.memory.index_backend import get_index  # noqa: PLC0415

            memory_index[0] = get_index(
                project, str(project / ".wren" / "memory")
            )
        return memory_index[0]

    def execute(name: str, args: dict[str, Any]) -> Any:
        if name == "wren_context_show":
            return {"manifest": manifest, "rules": rules}
        if name == "wren_memory_recall":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("wren_memory_recall requires a non-empty query")
            limit = int(args.get("limit", 3))
            return get_memory_index().search(query, limit=max(0, min(limit, 10)))
        if name == "wren_memory_fetch":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("wren_memory_fetch requires a non-empty query")
            index = get_memory_index()
            if hasattr(index, "store"):
                return index.store.get_context(
                    manifest,
                    query,
                    limit=max(1, min(int(args.get("limit", 5)), 20)),
                    model_name=args.get("model"),
                )
            from wren.memory.schema_indexer import describe_schema  # noqa: PLC0415

            return {"strategy": "full", "schema": describe_schema(manifest)}
        if name == "wren_dry_plan":
            sql = args.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("wren_dry_plan requires a non-empty sql")
            return {"sql": get_engine().dry_plan(sql)}
        if name == "wren_query":
            sql = args.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("wren_query requires a non-empty sql")
            limit = max(0, min(int(args.get("limit", 1000)), 10000))
            return _table_result(get_engine().query(sql, limit))
        if name == "wren_memory_store":
            nl = args.get("question")
            sql = args.get("sql")
            if not isinstance(nl, str) or not nl.strip():
                raise ValueError("wren_memory_store requires a non-empty question")
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("wren_memory_store requires a non-empty sql")
            from wren.memory.markdown import write_query_markdown  # noqa: PLC0415

            path = write_query_markdown(
                project,
                nl,
                sql,
                datasource=manifest.get("dataSource"),
            )
            # Markdown is the source of truth. Keep an already-installed
            # LanceDB index in sync too, without making the optional memory
            # extra a requirement for API mode.
            index = get_memory_index()
            if hasattr(index, "store"):
                index.store.store_query(
                    nl,
                    sql,
                    datasource=manifest.get("dataSource"),
                )
            return {"stored": True, "path": str(path.relative_to(project))}
        raise ValueError(f"Unknown Wren tool: {name}")

    properties = {
        "query": {"type": "string", "description": "Natural-language question."},
        "limit": {"type": "integer", "minimum": 0, "maximum": 20},
    }
    tools = [
        _tool_spec(
            "wren_context_show",
            "Read the current Wren MDL manifest and project business rules.",
        ),
        _tool_spec(
            "wren_memory_recall",
            "Find confirmed natural-language to SQL examples relevant to a question.",
            properties,
            ["query"],
        ),
        _tool_spec(
            "wren_memory_fetch",
            "Fetch schema context relevant to a natural-language question.",
            {
                **properties,
                "model": {
                    "type": "string",
                    "description": "Optional Wren model name filter.",
                },
            },
            ["query"],
        ),
        _tool_spec(
            "wren_dry_plan",
            "Validate SQL against Wren MDL and return the expanded target-dialect SQL.",
            {"sql": {"type": "string"}},
            ["sql"],
        ),
        _tool_spec(
            "wren_query",
            "Execute read-only SQL through Wren and return rows.",
            {
                "sql": {"type": "string"},
                "limit": {"type": "integer", "minimum": 0, "maximum": 10000},
            },
            ["sql"],
        ),
        _tool_spec(
            "wren_memory_store",
            "Store a successful, confirmed question and its Wren SQL for future recall.",
            {
                "question": {"type": "string"},
                "sql": {"type": "string"},
            },
            ["question", "sql"],
        ),
    ]
    return tools, execute, engine


def _resolve_project(path: str | None) -> Path | None:
    from wren.context import discover_project_path  # noqa: PLC0415

    try:
        return discover_project_path(explicit=path)
    except SystemExit:
        return None


def ask(
    prompt: str = typer.Argument(
        ..., help="The user's natural-language question to wrap."
    ),
    guided: bool = typer.Option(
        False,
        "--guided",
        help="Wrap in a strict-flow guided prompt (for weaker LLMs).",
    ),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Wrap in a minimal direct prompt (for stronger LLMs).",
    ),
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            help="Override backend: prompt, openai, or codex.",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Override the selected LLM model."),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Override the OpenAI-compatible API base URL."),
    ] = None,
    path: Annotated[
        Optional[str],
        typer.Option("--path", "-p", help="Wren project path for API tool execution."),
    ] = None,
) -> None:
    """Wrap PROMPT into a processed prompt for an agent.

    Choose exactly one of ``--guided`` or ``--direct``.
    """
    if guided == direct:
        # both False or both True
        typer.echo(
            "Error: choose exactly one of --guided or --direct (no default).",
            err=True,
        )
        raise typer.Exit(2)
    mode = "guided" if guided else "direct"
    rendered = _ask.render(mode, prompt)

    try:
        configured = load_config(_WREN_HOME).ai
    except WrenError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    selected_provider = (provider or configured.provider).strip().lower()
    if selected_provider not in AI_PROVIDERS:
        typer.echo(
            f"Error: unknown provider {selected_provider!r}. "
            f"Choose: {', '.join(AI_PROVIDERS)}.",
            err=True,
        )
        raise typer.Exit(1)
    selected = AIConfig(
        provider=selected_provider,
        model=(
            model.strip()
            if model and model.strip()
            else (
                configured.model
                if selected_provider == configured.provider
                else None
            )
        ),
        base_url=(base_url.strip() if base_url and base_url.strip() else configured.base_url),
    )
    if selected.provider == "prompt":
        typer.echo(rendered)
        return

    from wren.ai import (  # noqa: PLC0415
        AIError,
        effective_model,
        get_openai_api_key,
        run_codex,
        run_openai_agent,
    )

    if selected.provider == "codex":
        try:
            typer.echo(run_codex(rendered, model=effective_model(selected)))
        except AIError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        return

    api_key = get_openai_api_key()
    if not api_key:
        typer.echo(
            "Error: OPENAI_API_KEY is missing. Run `wren ai auth login` or "
            "set it in the environment.",
            err=True,
        )
        raise typer.Exit(1)
    project = _resolve_project(path)
    try:
        tools, execute_tool, engine = _build_wren_runtime(project)
        answer = run_openai_agent(
            rendered,
            model=effective_model(selected) or "gpt-4o-mini",
            api_key=api_key,
            base_url=selected.base_url,
            tools=tools,
            execute_tool=execute_tool,
            system_prompt=(
                "You are the Wren data assistant. Use Wren tools for any data "
                "claim. Start by reading context and memory, use MDL model names "
                "only, dry-plan non-trivial SQL before querying, and never invent "
                "results. Explain the final answer clearly."
            ),
        )
    except (AIError, ValueError, WrenError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        if "engine" in locals() and engine and engine[0] is not None:
            engine[0].close()
    typer.echo(answer)
