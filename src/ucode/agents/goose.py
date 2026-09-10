"""Goose coding agent support for the Databricks AI Gateway.

Goose has a built-in Databricks provider. Unity Gateway merges only the
provider settings it owns into Goose's shared YAML config so user preferences
and extensions survive. Databricks MCP servers are registered as local stdio
extensions that run ``ug mcp-proxy``; the proxy owns token refresh, so bearer
credentials are never persisted in Goose's config.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_yaml_safe,
    write_yaml_file,
)
from ucode.databricks import get_databricks_token
from ucode.state import mark_tool_managed, save_state

from .args import LaunchOptions

GOOSE_CONFIG_DIR = Path.home() / ".config" / "goose"
GOOSE_CONFIG_PATH = GOOSE_CONFIG_DIR / "config.yaml"
GOOSE_BACKUP_PATH = APP_DIR / "goose-config.backup.yaml"

SPEC: ToolSpec = {
    "binary": "goose",
    "package": "",  # native binary: https://github.com/aaif-goose/goose
    "display": "Goose",
    "config_path": GOOSE_CONFIG_PATH,
    "backup_path": GOOSE_BACKUP_PATH,
}

MANAGED_KEYS: list[list[str]] = [
    ["DATABRICKS_HOST"],
    ["GOOSE_PROVIDER"],
    ["GOOSE_MODEL"],
    ["extensions", "skills"],
]


def is_update_available() -> tuple[str, str] | None:
    return None


def default_model(state: dict) -> str | None:
    """Prefer Claude Sonnet, then Opus/Haiku, and finally Gemini."""
    claude_models = state.get("claude_models") or {}
    for family in ("sonnet", "opus", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    gemini_models = state.get("gemini_models") or []
    return gemini_models[0] if gemini_models else None


def render_overlay(workspace: str, model: str) -> dict:
    return {
        "DATABRICKS_HOST": workspace,
        "GOOSE_PROVIDER": "databricks",
        "GOOSE_MODEL": model,
        "extensions": {
            "skills": {
                "enabled": True,
                "type": "platform",
                "name": "skills",
                "description": "Load skills from standard agent skill directories",
                "bundled": True,
                "available_tools": [],
            }
        },
    }


def build_runtime_env(workspace: str, token: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = workspace
    env.pop("OAUTH_TOKEN", None)
    if token:
        env["DATABRICKS_TOKEN"] = token
    else:
        # Do not let an ambient credential override Goose's refreshable OAuth.
        env.pop("DATABRICKS_TOKEN", None)
    return env


def _mcp_slug(name: str) -> str:
    return name.lower().replace("-", "_")


def build_mcp_server_entry(name: str, argv: list[str]) -> dict:
    return {
        "enabled": True,
        "type": "stdio",
        "name": name,
        "description": f"Databricks MCP server: {name}",
        "cmd": argv[0],
        "args": list(argv[1:]),
        "envs": {},
        "env_keys": [],
        "timeout": 300,
        "bundled": None,
        "available_tools": [],
    }


def write_mcp_server_config(name: str, argv: list[str]) -> bool:
    backup_existing_file(GOOSE_CONFIG_PATH, GOOSE_BACKUP_PATH)
    existing = read_yaml_safe(GOOSE_CONFIG_PATH)
    extensions = existing.get("extensions")
    if not isinstance(extensions, dict):
        extensions = {}
    slug = _mcp_slug(name)
    replaced = slug in extensions
    extensions[slug] = build_mcp_server_entry(name, argv)
    existing["extensions"] = extensions
    write_yaml_file(GOOSE_CONFIG_PATH, existing)
    return replaced


def remove_mcp_server_config(name: str) -> bool:
    existing = read_yaml_safe(GOOSE_CONFIG_PATH)
    extensions = existing.get("extensions")
    if not isinstance(extensions, dict):
        return False
    slug = _mcp_slug(name)
    if slug not in extensions:
        return False
    extensions.pop(slug)
    existing["extensions"] = extensions
    write_yaml_file(GOOSE_CONFIG_PATH, existing)
    return True


def write_tool_config(state: dict, model: str) -> dict:
    backup_existing_file(GOOSE_CONFIG_PATH, GOOSE_BACKUP_PATH)
    existing = read_yaml_safe(GOOSE_CONFIG_PATH)
    deep_merge_dict(existing, render_overlay(state["workspace"], model))
    write_yaml_file(GOOSE_CONFIG_PATH, existing)
    state = mark_tool_managed(state, "goose", MANAGED_KEYS)
    save_state(state)
    return state


def _runtime_token(state: dict) -> str | None:
    """PAT configurations are static; OAuth is delegated to Goose for refresh."""
    if not state.get("use_pat"):
        return None
    return get_databricks_token(state["workspace"], state.get("profile"))


def launch(state: dict, tool_args: list[str], *, options: LaunchOptions) -> None:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Goose model is available on this workspace.")
    write_tool_config(state, model)
    env = build_runtime_env(state["workspace"], _runtime_token(state))
    proc = subprocess.Popen([SPEC["binary"], "session", *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "run",
        "--text",
        "say hi in 5 words or less",
        "--no-session",
        "--max-turns",
        "1",
    ]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    if not default_model(state):
        raise RuntimeError("No Goose model is available on this workspace.")
    return build_runtime_env(workspace, _runtime_token(state))
