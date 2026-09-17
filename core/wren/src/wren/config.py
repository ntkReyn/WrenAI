"""Wren CLI configuration loaded from ~/.wren/config.json."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wren.model.error import ErrorCode, WrenError

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
AI_PROVIDERS = ("prompt", "openai", "codex")


@dataclass(frozen=True)
class AIConfig:
    """LLM backend preferences for ``wren ask``."""

    provider: str = "prompt"
    model: str | None = None
    base_url: str = DEFAULT_OPENAI_BASE_URL


@dataclass(frozen=True)
class WrenConfig:
    """Immutable configuration for the Wren CLI.

    Attributes
    ----------
    strict_mode:
        When ``True``, all table references in SQL must be defined in the MDL
        manifest.  Queries referencing non-MDL tables are rejected.
    denied_functions:
        Set of function names (lowercase) that are forbidden in SQL queries.
        Matching is case-insensitive.
    allowed_source_functions:
        Set of synthetic-generator function names (lowercase) the operator
        explicitly opts in as query sources under strict mode (e.g.
        ``generate_series``). Empty by default — generators are blocked unless
        listed here. Data/file readers (``read_csv``, ``dblink``, ...) can
        NEVER be allowed via this list; they are always blocked in strict mode.
    ai:
        LLM provider preferences used by ``wren ask``.
    """

    strict_mode: bool = False
    denied_functions: frozenset[str] = field(default_factory=frozenset)
    allowed_source_functions: frozenset[str] = field(default_factory=frozenset)
    ai: AIConfig = field(default_factory=AIConfig)


def _parse_ai_config(raw: Any, config_path: Path) -> AIConfig:
    """Validate the optional ``ai`` section of config.json."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'ai' must be a JSON object.",
        )

    provider = raw.get("provider", "prompt")
    if not isinstance(provider, str) or provider.lower() not in AI_PROVIDERS:
        choices = "|".join(AI_PROVIDERS)
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'ai.provider' must be one of {choices}.",
        )

    model = raw.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'ai.model' must be a non-empty string or null.",
        )

    base_url = raw.get("base_url", DEFAULT_OPENAI_BASE_URL)
    if not isinstance(base_url, str) or not base_url.strip():
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'ai.base_url' must be a non-empty string.",
        )

    return AIConfig(
        provider=provider.lower(),
        model=model.strip() if isinstance(model, str) else None,
        base_url=base_url.rstrip("/"),
    )


def load_config(wren_home: Path) -> WrenConfig:
    """Load configuration from ``wren_home/config.json``.

    Returns default ``WrenConfig`` when the file does not exist.
    Raises ``WrenError`` when the file exists but contains invalid JSON.
    """
    config_path = wren_home / "config.json"
    if not config_path.exists():
        return WrenConfig()

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as e:
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"Failed to read {config_path}: {e}",
        ) from e

    if not isinstance(raw, dict):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path} must contain a JSON object.",
        )

    strict_mode_raw = raw.get("strict_mode", False)
    if not isinstance(strict_mode_raw, bool):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'strict_mode' must be a JSON boolean.",
        )

    denied_raw = raw.get("denied_functions", [])
    if not isinstance(denied_raw, list):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'denied_functions' must be a JSON array.",
        )
    if any(not isinstance(f, str) for f in denied_raw):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'denied_functions' must contain only strings.",
        )
    denied_functions = frozenset(f.lower() for f in denied_raw)

    allowed_src_raw = raw.get("allowed_source_functions", [])
    if not isinstance(allowed_src_raw, list):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'allowed_source_functions' must be a JSON array.",
        )
    if any(not isinstance(f, str) for f in allowed_src_raw):
        raise WrenError(
            ErrorCode.GENERIC_USER_ERROR,
            f"{config_path}: 'allowed_source_functions' must contain only strings.",
        )
    allowed_source_functions = frozenset(f.lower() for f in allowed_src_raw)

    return WrenConfig(
        strict_mode=strict_mode_raw,
        denied_functions=denied_functions,
        allowed_source_functions=allowed_source_functions,
        ai=_parse_ai_config(raw.get("ai"), config_path),
    )


def save_ai_config(wren_home: Path, ai: AIConfig) -> None:
    """Persist LLM preferences while preserving other config.json keys."""
    config_path = wren_home / "config.json"
    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as e:
            raise WrenError(
                ErrorCode.GENERIC_USER_ERROR,
                f"Failed to read {config_path}: {e}",
            ) from e
        if not isinstance(loaded, dict):
            raise WrenError(
                ErrorCode.GENERIC_USER_ERROR,
                f"{config_path} must contain a JSON object.",
            )
        raw = loaded

    raw["ai"] = {
        "provider": ai.provider,
        "model": ai.model,
        "base_url": ai.base_url,
    }
    wren_home.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=wren_home, suffix=".json.tmp")
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, config_path)
    except Exception:
        os.unlink(tmp_path)
        raise
    os.chmod(config_path, 0o600)
