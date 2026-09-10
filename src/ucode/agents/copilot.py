"""GitHub Copilot CLI agent: writes ~/.copilot/.env and injects BYOK env vars at launch.

Copilot CLI's BYOK config (COPILOT_PROVIDER_*) is documented as env-var only —
the CLI does not auto-load ~/.copilot/.env. We still write the file so users can
inspect what's configured (`cat ~/.copilot/.env`) and to give `revert` something
to clean up; the values are also injected directly into the child process's
environment at launch.

Copilot CLI supports two BYOK provider dialects. Claude models get its native
`anthropic` provider type, pointed at the same Messages-API gateway path
claude.py uses — Copilot's own runtime inserts `cache_control` breakpoints on
that path, so the (typically huge, shared) system/tool prefix actually caches.
Codex (gpt-5) has no native-dialect provider on Copilot's side, so it stays on
the `openai` provider against the Databricks MLflow chat-completions gateway.
Gemini is intentionally excluded from both — Databricks' Gemini translation
layer rejects the `stream_options` field that Copilot CLI sends, so Gemini
models 400 on every request.

The `anthropic` path needs two things not obvious from `copilot help
providers`, both confirmed live: Bearer auth, not the `x-api-key` that
provider type sends by default (Databricks' gateway 401s on it); and a
well-known model id kept separate from the actual wire id, or Copilot sends
`temperature`, which current-gen Claude models reject. See render_env_overlay.

That last part only works from Copilot 1.0.81-6 onward — verified live across
1.0.79 through 1.0.83. Below it, Copilot always sends `temperature` on the
`anthropic` path regardless of the model id, so Sonnet 5/Opus 5 (which reject
it outright) 400 on every request; Haiku 4.5 tolerates `temperature` and
would work either way, but the version gate below applies to every Claude
model for simplicity. Below 1.0.81-6, Claude models keep the old `openai`
path — uncached, but that's the status quo, not a regression.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from pathlib import Path

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    parse_dotenv,
    read_json_safe,
    write_dotenv,
    write_json_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_copilot_base_urls,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version

from .args import LaunchOptions

COPILOT_CONFIG_DIR = Path.home() / ".copilot"
COPILOT_ENV_PATH = COPILOT_CONFIG_DIR / "ucode.env"
COPILOT_MCP_CONFIG_PATH = COPILOT_CONFIG_DIR / "ucode-mcp-config.json"
COPILOT_BACKUP_PATH = APP_DIR / "copilot-ucode-env.backup"
COPILOT_MCP_BACKUP_PATH = APP_DIR / "copilot-ucode-mcp-config.backup.json"

SPEC: ToolSpec = {
    "binary": "copilot",
    "package": "@github/copilot",
    "display": "GitHub Copilot CLI",
    "config_path": COPILOT_ENV_PATH,
    "backup_path": COPILOT_BACKUP_PATH,
}

MANAGED_KEYS: list[str] = [
    "COPILOT_PROVIDER_TYPE",
    "COPILOT_PROVIDER_BASE_URL",
    "COPILOT_MODEL",
    "COPILOT_PROVIDER_MODEL_ID",
    "COPILOT_PROVIDER_WIRE_MODEL",
    "COPILOT_PROVIDER_BEARER_TOKEN",
    "COPILOT_OFFLINE",
    "OAUTH_TOKEN",
]
LEGACY_ENV_KEYS = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "COPILOT_PROVIDER_API_KEY",
]
# COPILOT_MODEL (openai) vs COPILOT_PROVIDER_MODEL_ID+COPILOT_PROVIDER_WIRE_MODEL
# (anthropic) are mutually exclusive — cleared before every write so switching
# families doesn't leave the other set stale in ~/.copilot/ucode.env.
_MODEL_SELECTION_KEYS = (
    "COPILOT_MODEL",
    "COPILOT_PROVIDER_MODEL_ID",
    "COPILOT_PROVIDER_WIRE_MODEL",
)

_CANONICAL_CLAUDE_MODEL_ID_RE = re.compile(r"claude-[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
# Same Bedrock version-marker pattern usage.py's normalize_price_key strips,
# e.g. "claude-opus-4-8-v1:0" -> "claude-opus-4-8" — Copilot's catalog doesn't
# carry the AWS version suffix, so leaving it in re-triggers the "unrecognized
# model" fallback (including the `temperature` send) this split is for.
_BEDROCK_VERSION_SUFFIX_RE = re.compile(r"-v\d+(:\d+)?$")

# (major, minor, patch, prerelease) — see the module docstring. A version with
# no prerelease suffix (a final release) is a 4th component of _UNRELEASED so
# it always sorts after every prerelease of the same (major, minor, patch).
MINIMUM_COPILOT_ANTHROPIC_VERSION = (1, 0, 81, 6)
_UNRELEASED = 999_999
_COPILOT_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-(\d+))?")


def default_model(state: dict) -> str | None:
    """Prefer Claude sonnet, then opus/haiku, then codex.

    A managed config's ``copilot_default_model`` and ``copilot_models`` both win outright: the former is
    the admin's chosen session start, the latter their allowlist. Workspace-wide discovery falls back.
    """
    if isinstance(state.get("copilot_default_model"), str):
        return state.get("copilot_default_model")
    copilot_models = state.get("copilot_models") or []
    if isinstance(copilot_models, list) and copilot_models:
        return copilot_models[0]
    claude_models = state.get("claude_models") or {}
    for family in ("sonnet", "opus", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_models = state.get("codex_models") or []
    if codex_models:
        return codex_models[0]
    return None


def _is_claude_model(model: str) -> bool:
    # Every Claude family/model id ucode discovers or pins contains "claude"
    # (canonical Anthropic names like "claude-sonnet-5", or Bedrock-style
    # slugs like "us.anthropic.claude-opus-4-8") — same substring check
    # `databricks.py` already uses elsewhere to special-case the family.
    return "claude" in model.lower()


def _canonical_claude_model_id(model: str) -> str:
    # e.g. "system.ai.claude-sonnet-5" -> "claude-sonnet-5" — the well-known
    # name Copilot needs to recognize the model (see render_env_overlay).
    # Lowercased and stripped of any Bedrock version suffix so it matches
    # Copilot's catalog regardless of the input's casing or source.
    match = _CANONICAL_CLAUDE_MODEL_ID_RE.search(model)
    canonical = match.group(0).lower() if match else model.lower()
    return _BEDROCK_VERSION_SUFFIX_RE.sub("", canonical)


def _parse_copilot_version(value: str) -> tuple[int, int, int, int] | None:
    match = _COPILOT_VERSION_RE.search(value)
    if not match:
        return None
    major, minor, patch, pre = match.groups()
    return int(major), int(minor), int(patch), int(pre) if pre is not None else _UNRELEASED


def _supports_anthropic_provider() -> bool:
    version = _parse_copilot_version(agent_version(SPEC["binary"]))
    return version is not None and version >= MINIMUM_COPILOT_ANTHROPIC_VERSION


def render_env_overlay(workspace: str, model: str, token: str) -> dict[str, str]:
    base_urls = build_copilot_base_urls(workspace)
    if _is_claude_model(model) and _supports_anthropic_provider():
        return {
            "COPILOT_PROVIDER_TYPE": "anthropic",
            "COPILOT_PROVIDER_BASE_URL": base_urls["anthropic"],
            # Not COPILOT_MODEL: that would default both ids below to the
            # unrecognized catalog id and Copilot would send `temperature`.
            "COPILOT_PROVIDER_MODEL_ID": _canonical_claude_model_id(model),
            "COPILOT_PROVIDER_WIRE_MODEL": model,
            "COPILOT_PROVIDER_BEARER_TOKEN": token,  # not API_KEY — see module docstring
            "COPILOT_OFFLINE": "true",
            "OAUTH_TOKEN": token,
        }
    return {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": base_urls["openai"],
        "COPILOT_MODEL": model,
        "COPILOT_PROVIDER_BEARER_TOKEN": token,
        "COPILOT_OFFLINE": "true",
        "OAUTH_TOKEN": token,
    }


def build_runtime_env(workspace: str, model: str, token: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(render_env_overlay(workspace, model, token))
    return env


def build_mcp_server_entry(argv: list[str]) -> dict:
    # A `local` MCP server runs a stdio command; `command`/`args` split the
    # argv. ucode registers the `ucode mcp-proxy ...` bridge here so Copilot
    # never speaks HTTP+bearer directly — the proxy handles token refresh. The
    # OAUTH_TOKEN env Copilot still injects at launch is for MODEL auth, not MCP.
    return {
        "type": "local",
        "command": argv[0],
        "args": list(argv[1:]),
        "tools": ["*"],
    }


def write_mcp_server_config(name: str, argv: list[str]) -> bool:
    backup_existing_file(COPILOT_MCP_CONFIG_PATH, COPILOT_MCP_BACKUP_PATH)
    existing = read_json_safe(COPILOT_MCP_CONFIG_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_mcp_server_entry(argv)
    existing["mcpServers"] = mcp_servers
    write_json_file(COPILOT_MCP_CONFIG_PATH, existing)
    return removed


def remove_mcp_server_config(name: str) -> bool:
    existing = read_json_safe(COPILOT_MCP_CONFIG_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict) or name not in mcp_servers:
        return False
    mcp_servers.pop(name)
    existing["mcpServers"] = mcp_servers
    write_json_file(COPILOT_MCP_CONFIG_PATH, existing)
    return True


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(COPILOT_ENV_PATH, COPILOT_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    overlay = render_env_overlay(state["workspace"], model, token)
    existing = parse_dotenv(COPILOT_ENV_PATH)
    for key in LEGACY_ENV_KEYS:
        existing.pop(key, None)
    for key in _MODEL_SELECTION_KEYS:
        existing.pop(key, None)
    existing.update(overlay)
    write_dotenv(COPILOT_ENV_PATH, existing)
    state = mark_tool_managed(state, "copilot", MANAGED_KEYS)
    save_state(state)
    return state, token


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> tuple[str, str]:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Copilot model is available on this workspace.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return model, token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def launch(state: dict, tool_args: list[str], *, options: LaunchOptions) -> None:
    model, token = _refresh_token_once(state)
    env = build_runtime_env(state["workspace"], model, token)

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen([SPEC["binary"], *mcp_config_args(), *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        *mcp_config_args(),
        "--prompt",
        "say hi in 5 words or less",
        "--allow-all-tools",
    ]


def mcp_config_args() -> list[str]:
    if not COPILOT_MCP_CONFIG_PATH.exists():
        return []
    return ["--additional-mcp-config", f"@{COPILOT_MCP_CONFIG_PATH}"]


def validate_env(state: dict) -> dict[str, str]:
    """Inject BYOK env vars for the validation subprocess (Copilot doesn't auto-load .env)."""
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    model = default_model(state)
    if not model:
        raise RuntimeError("No Copilot model is available on this workspace.")
    token = get_databricks_token(workspace, state.get("profile"))
    return build_runtime_env(workspace, model, token)
