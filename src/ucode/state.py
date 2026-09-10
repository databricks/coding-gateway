"""Persistent state for ucode (per-workspace, versioned)."""

from __future__ import annotations

import json
from typing import cast

from ucode.config_io import APP_DIR, is_dry_run
from ucode.custom_oauth import (
    CustomOAuthConfig,
    build_custom_auth_shell_command,
    build_custom_auth_token_argv,
)
from ucode.databricks import (
    build_auth_shell_command,
    build_auth_token_argv,
    build_shared_base_urls,
)

STATE_PATH = APP_DIR / "state.json"
STATE_VERSION = 3
# Transient key holding the developer's own values for whatever a managed config layered over them.
# Present only in memory: the layered values render the agent settings files, while `save_state`
# restores what's under it so `state.json` keeps recording the developer's own configuration.
MANAGED_OVERLAY_KEY = "_managed_overlay"
AUTH_COMMAND_TIMEOUT_MS = 5000
AUTH_REFRESH_INTERVAL_MS = 900_000


def load_full_state() -> dict:
    """Load the entire state file. Returns empty structure if missing or wrong version."""
    if not STATE_PATH.exists():
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    if not isinstance(data, dict) or data.get("state_version") != STATE_VERSION:
        return {"state_version": STATE_VERSION, "current_workspace": None, "workspaces": {}}
    return data


def load_state() -> dict:
    """Load the current workspace's state as a flat dict."""
    full = load_full_state()
    workspace = full.get("current_workspace")
    if not workspace:
        return {}
    ws_state = full.get("workspaces", {}).get(workspace, {})
    ws_state["workspace"] = workspace
    return hydrate_state(ws_state)


def save_state(state: dict) -> None:
    """Save workspace state back into the per-workspace structure.

    Values a managed config layered over the developer's own are stripped first (see
    ``MANAGED_OVERLAY_KEY``), so an admin-published config takes effect through the generated agent
    settings files without overwriting what the developer configured for themselves. Read
    non-destructively: a launch can save more than once from the same dict (e.g. the relayed proxy
    rewriting its port), and every one of those writes must restore the developer's values.
    """
    if is_dry_run():
        return
    full = load_full_state()
    workspace = state.get("workspace") or full.get("current_workspace")
    if workspace:
        full["current_workspace"] = workspace
        full["workspaces"][workspace] = hydrate_state(_without_managed_overlay(state))
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc


def _without_managed_overlay(state: dict) -> dict:
    """Return ``state`` with managed-config values swapped back for the developer's own.

    Returns a new dict and leaves ``state`` untouched, so the caller keeps the layered values it
    needs for rendering and repeated saves stay idempotent.
    """
    overlay = state.get(MANAGED_OVERLAY_KEY)
    if not isinstance(overlay, dict):
        return state
    persisted = {key: value for key, value in state.items() if key != MANAGED_OVERLAY_KEY}
    for key, value in overlay.items():
        # A key the developer never set is dropped rather than persisted as None.
        if value is None:
            persisted.pop(key, None)
        else:
            persisted[key] = value
    return persisted


def set_current_workspace(workspace: str | None) -> None:
    """Set ``current_workspace`` without touching the per-workspace blocks.

    Used by flows like ``configure tracing`` that operate on a non-current
    workspace and must not silently change which workspace ``ucode launch``
    targets afterwards."""
    if is_dry_run():
        return
    full = load_full_state()
    if full.get("current_workspace") == workspace:
        return
    full["current_workspace"] = workspace
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write state file: {STATE_PATH}") from exc


def hydrate_state(state: dict) -> dict:
    """Normalize a workspace state entry and add derived harness config.

    :param state: Raw workspace state entry from ``state.json``.
    :returns: Hydrated workspace state with stable ``managed_configs``,
        ``base_urls``, and per-agent ``agents`` entries.
    """
    if not isinstance(state, dict):
        return {}

    hydrated = dict(state)
    managed_configs = hydrated.get("managed_configs")
    if not isinstance(managed_configs, dict):
        managed_configs = {}
    normalized: dict[str, dict] = {}
    for tool, entry in managed_configs.items():
        if isinstance(entry, dict):
            keys = entry.get("keys") if isinstance(entry.get("keys"), list) else []
            normalized[tool] = {"keys": keys}
        elif entry:
            normalized[tool] = {"keys": []}
    hydrated["managed_configs"] = normalized

    workspace = hydrated.get("workspace")
    if workspace:
        hydrated["base_urls"] = build_shared_base_urls(workspace)
        hydrated["agents"] = build_agent_state(hydrated)
    else:
        hydrated["base_urls"] = {}
        hydrated["agents"] = {}

    return hydrated


def build_agent_state(state: dict) -> dict[str, dict]:
    """Build per-agent harness configuration for a workspace.

    The returned shape is intended for downstream tools that want to reuse
    ucode's configured gateway URLs and auth command without duplicating
    endpoint construction logic.

    :param state: Hydrated workspace state containing ``workspace``,
        ``base_urls``, and discovered model lists.
    :returns: Mapping from agent name to its reusable configuration.
    """
    workspace = state.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return {}

    from ucode.agents import default_model_for_tool

    profile = state.get("profile") if isinstance(state.get("profile"), str) else None
    base_urls_value = state.get("base_urls")
    base_urls = base_urls_value if isinstance(base_urls_value, dict) else {}
    use_pat = bool(state.get("use_pat"))
    auth_command = build_auth_shell_command(workspace, profile, use_pat=use_pat)
    auth_argv = build_auth_token_argv(workspace, profile, use_pat=use_pat)
    claude_auth_command = auth_command
    codex_auth_command = auth_command
    codex_auth_argv = auth_argv
    custom_oauth = state.get("custom_oauth")
    if isinstance(custom_oauth, dict):
        typed_custom_oauth = cast(CustomOAuthConfig, custom_oauth)
        claude_auth_command = build_custom_auth_shell_command(workspace, typed_custom_oauth)
        codex_auth_command = claude_auth_command
        codex_auth_argv = build_custom_auth_token_argv(workspace, typed_custom_oauth)
    claude_models_value = state.get("claude_models")
    claude_models: dict = claude_models_value if isinstance(claude_models_value, dict) else {}
    codex_models_value = state.get("codex_models")
    codex_models = codex_models_value if isinstance(codex_models_value, list) else []
    gemini_models_value = state.get("gemini_models")
    gemini_models = gemini_models_value if isinstance(gemini_models_value, list) else []

    selection_state = {
        **state,
        "claude_models": claude_models,
        "codex_models": codex_models,
        "gemini_models": gemini_models,
    }

    claude_model = (
        claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")
    )
    codex_model = default_model_for_tool("codex", selection_state)
    pi_model = default_model_for_tool("pi", selection_state)

    agents: dict[str, dict] = {
        "claude": {
            "model": claude_model,
            "base_url": base_urls.get("claude"),
            "auth_command": claude_auth_command,
            "auth_refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
            "env": {
                "ANTHROPIC_BASE_URL": base_urls.get("claude"),
                "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": str(AUTH_REFRESH_INTERVAL_MS),
                # Kept in sync with agents/claude.py render_overlay.
                "ENABLE_PROMPT_CACHING_1H": "1",
                "ENABLE_TOOL_SEARCH": "true",
                "CLAUDE_CODE_USE_GATEWAY": "1",
            },
        },
        "codex": {
            "model": codex_model,
            "base_url": base_urls.get("codex"),
            "auth_command": codex_auth_command,
            "auth": {
                "command": codex_auth_argv[0],
                "args": codex_auth_argv[1:],
                "timeout_ms": AUTH_COMMAND_TIMEOUT_MS,
                "refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
            },
        },
        "pi": {
            "model": pi_model,
            "base_urls": base_urls.get("pi") if isinstance(base_urls.get("pi"), dict) else {},
            "auth_command": auth_command,
            "auth_refresh_interval_ms": AUTH_REFRESH_INTERVAL_MS,
        },
    }
    return {
        name: {key: value for key, value in config.items() if value is not None}
        for name, config in agents.items()
    }


def clear_state() -> None:
    """Remove the current workspace entry from state."""
    full = load_full_state()
    workspace = full.get("current_workspace")
    if workspace:
        full.get("workspaces", {}).pop(workspace, None)
        full["current_workspace"] = None
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(full, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to clear state file: {STATE_PATH}") from exc


def mark_tool_managed(state: dict, tool: str, managed_keys: list) -> dict:
    """Record which config keys ucode manages for ``tool``."""
    managed_configs = dict(state.get("managed_configs") or {})
    managed_configs[tool] = {"keys": list(managed_keys)}
    state["managed_configs"] = managed_configs
    state["last_tool"] = tool
    return state


def get_provider_service(state: dict, tool: str) -> str | None:
    """Return the persisted Model Provider Service for ``tool``, if any.

    Launches route through this provider (skipping Databricks model pinning)
    unless overridden by an explicit ``--provider`` flag.
    """
    providers = state.get("provider_services")
    if not isinstance(providers, dict):
        return None
    name = providers.get(tool)
    return name if isinstance(name, str) and name else None


def set_provider_service(state: dict, tool: str, full_name: str | None) -> dict:
    """Persist (or clear, when ``full_name`` is None) ``tool``'s provider service."""
    providers = dict(state.get("provider_services") or {})
    if full_name:
        providers[tool] = full_name
    else:
        providers.pop(tool, None)
    if providers:
        state["provider_services"] = providers
    else:
        state.pop("provider_services", None)
    return state
