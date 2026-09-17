"""Unit tests for wren.config — config loading from ~/.wren/config.json."""

from __future__ import annotations

import json

import pytest

from wren.config import AIConfig, WrenConfig, load_config, save_ai_config
from wren.model.error import WrenError

pytestmark = pytest.mark.unit


def test_load_config_no_file(tmp_path):
    config = load_config(tmp_path)
    assert config == WrenConfig()
    assert config.strict_mode is False
    assert config.denied_functions == frozenset()


def test_load_config_strict_enabled(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"strict_mode": True}))
    config = load_config(tmp_path)
    assert config.strict_mode is True
    assert config.denied_functions == frozenset()


def test_load_config_with_denied_functions(tmp_path):
    data = {
        "strict_mode": True,
        "denied_functions": ["pg_read_file", "DBLINK", "Lo_Import"],
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = load_config(tmp_path)
    assert config.strict_mode is True
    assert config.denied_functions == frozenset(["pg_read_file", "dblink", "lo_import"])


def test_load_config_function_names_lowercased(tmp_path):
    data = {"denied_functions": ["PG_READ_FILE"]}
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = load_config(tmp_path)
    assert "pg_read_file" in config.denied_functions


def test_load_config_malformed_json(tmp_path):
    (tmp_path / "config.json").write_text("not valid json{{{")
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_not_a_dict(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps([1, 2, 3]))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_denied_functions_not_array(tmp_path):
    data = {"denied_functions": "pg_read_file"}
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_unknown_keys_ignored(tmp_path):
    data = {"strict_mode": True, "unknown_key": "value", "another": 42}
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = load_config(tmp_path)
    assert config.strict_mode is True


def test_load_config_partial_only_denied_functions(tmp_path):
    data = {"denied_functions": ["dblink"]}
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = load_config(tmp_path)
    assert config.strict_mode is False
    assert config.denied_functions == frozenset(["dblink"])


def test_load_config_empty_object(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({}))
    config = load_config(tmp_path)
    assert config == WrenConfig()


def test_load_config_ai_settings(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "strict_mode": True,
                "ai": {
                    "provider": "OPENAI",
                    "model": "gpt-4o-mini",
                    "base_url": "https://proxy.example/v1/",
                },
            }
        )
    )
    config = load_config(tmp_path)
    assert config.ai == AIConfig(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://proxy.example/v1",
    )


def test_save_ai_config_preserves_existing_settings(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"strict_mode": True, "denied_functions": ["secret"]})
    )
    save_ai_config(
        tmp_path,
        AIConfig(provider="openai", model="gpt-4o-mini"),
    )
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["strict_mode"] is True
    assert raw["denied_functions"] == ["secret"]
    assert raw["ai"]["provider"] == "openai"


@pytest.mark.parametrize("value", ["bad", 1, [], {}])
def test_load_config_invalid_ai_provider_rejected(tmp_path, value):
    (tmp_path / "config.json").write_text(json.dumps({"ai": {"provider": value}}))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_strict_mode_string_rejected(tmp_path):
    """'strict_mode': 'false' must not silently coerce to True."""
    (tmp_path / "config.json").write_text(json.dumps({"strict_mode": "false"}))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_strict_mode_int_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"strict_mode": 1}))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_denied_functions_mixed_types_rejected(tmp_path):
    data = {"denied_functions": ["safe", 1, {"obj": True}]}
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_allowed_source_functions(tmp_path):
    data = {
        "strict_mode": True,
        "allowed_source_functions": ["Generate_Series", "RANGE"],
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = load_config(tmp_path)
    assert config.allowed_source_functions == frozenset(["generate_series", "range"])


def test_load_config_allowed_source_functions_default_empty(tmp_path):
    config = load_config(tmp_path)
    assert config.allowed_source_functions == frozenset()


def test_load_config_allowed_source_functions_not_array(tmp_path):
    data = {"allowed_source_functions": "generate_series"}
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(WrenError):
        load_config(tmp_path)


def test_load_config_allowed_source_functions_mixed_types_rejected(tmp_path):
    data = {"allowed_source_functions": ["generate_series", 3]}
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(WrenError):
        load_config(tmp_path)
