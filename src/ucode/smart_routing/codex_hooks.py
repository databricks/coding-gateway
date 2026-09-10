"""Codex hook configuration for smart subagent routing."""

from __future__ import annotations

import copy
import shlex
import subprocess

from ucode.databricks import build_auth_token_argv
from ucode.smart_routing import hooks

ROUTING_HOOK_COMMAND_MARKER = "codex-router-hook"


def routing_models(state: dict) -> list[str]:
    """Return the configured model services compatible with Codex routing."""
    models: list[str] = []
    for key in ("codex_models", "oss_models"):
        values = state.get(key)
        if isinstance(values, list):
            models.extend(value for value in values if isinstance(value, str) and value)
    return list(dict.fromkeys(models))


def sync_smart_routing_hooks(doc: dict, state: dict, *, enabled: bool) -> None:
    """Synchronize ucode-managed routing hooks in a Codex config document."""
    groups = _routing_hook_groups(state) if enabled else {}
    hooks.sync_managed_hooks(doc, ROUTING_HOOK_COMMAND_MARKER, groups)


def remove_smart_routing_hooks(doc: dict) -> bool:
    """Remove only ucode-managed smart-routing hooks."""
    return hooks.remove_managed_hooks(doc, ROUTING_HOOK_COMMAND_MARKER)


def _routing_hook_groups(state: dict) -> dict[str, list[dict]]:
    session_argv = _routing_hook_argv(state, "session-start")
    subagent_argv = _routing_hook_argv(state, "record-subagent")
    return {
        "PreToolUse": [_pre_tool_use_hook_group(state)],
        "SessionStart": [
            {
                "matcher": "startup|resume|clear",
                "hooks": [_routing_command_hook(session_argv)],
            }
        ],
        "SubagentStart": [
            {
                "hooks": [_routing_command_hook(subagent_argv)],
            }
        ],
    }


def merge_pre_tool_use_hooks(
    existing: list[dict], state: dict, *, available_models: list[str]
) -> list[dict]:
    """Add the ucode spawn hook to an existing Codex PreToolUse hook list."""
    doc = {"hooks": {"PreToolUse": copy.deepcopy(existing)}}
    hooks.sync_managed_hooks(
        doc,
        ROUTING_HOOK_COMMAND_MARKER,
        {"PreToolUse": [_pre_tool_use_hook_group(state, available_models=available_models)]},
    )
    return doc["hooks"]["PreToolUse"]


def _pre_tool_use_hook_group(state: dict, *, available_models: list[str] | None = None) -> dict:
    route_argv = _routing_hook_argv(
        state,
        "route-subagent",
        available_models=available_models,
    )
    return {
        "matcher": "Agent|.*spawn_agent$",
        "hooks": [_routing_command_hook(route_argv, status="Routing subagent model")],
    }


def _routing_hook_argv(
    state: dict, event: str, *, available_models: list[str] | None = None
) -> list[str]:
    workspace = str(state.get("workspace") or "")
    argv = [
        build_auth_token_argv(workspace, state.get("profile"), use_pat=bool(state.get("use_pat")))[
            0
        ],
        ROUTING_HOOK_COMMAND_MARKER,
        event,
    ]
    if event != "route-subagent":
        return argv
    argv += ["--host", workspace]
    profile = state.get("profile")
    if isinstance(profile, str) and profile:
        argv += ["--profile", profile]
    if state.get("use_pat"):
        argv.append("--use-pat")
    models = available_models if available_models is not None else routing_models(state)
    for model in models:
        if isinstance(model, str) and model:
            argv += ["--model", model]
    return argv


def _routing_command_hook(argv: list[str], *, status: str | None = None) -> dict:
    hook = {
        "type": "command",
        "command": shlex.join(argv),
        "command_windows": subprocess.list2cmdline(argv),
        "timeout": 35,
    }
    if status:
        hook["statusMessage"] = status
    return hook
