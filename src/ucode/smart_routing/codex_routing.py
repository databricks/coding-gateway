"""Databricks AI Gateway routing helpers for Codex sessions and subagents.

Codex-specific configuration on top of the shared :mod:`ucode.smart_routing.routing`
core: the workspace-backed route options, the ``spawn_agent`` tool
detector, the Codex model-id translation, and the artifact paths.
"""

from __future__ import annotations

import json
import re

# Re-exported so tests can patch the shared ``urlopen`` seam via
# ``codex_routing.urllib.request`` — the actual call lives in ``routing``, but
# Python modules are singletons so patching this name patches the one call site.
import urllib.request  # noqa: F401
from collections.abc import Callable
from typing import Any

from ucode.config_io import APP_DIR
from ucode.smart_routing import routing
from ucode.smart_routing.routing import RoutingDecision

ROUTER_NAME = routing.ROUTER_NAME
ROUTING_PATH = routing.ROUTING_PATH
REQUEST_TIMEOUT_S = routing.REQUEST_TIMEOUT_S
SPAWN_AGENT_TOOL_SUFFIX = "spawn_agent"
CANARY_PATH = APP_DIR / "codex-smart-routing-canary.json"
AUDIT_PATH = APP_DIR / "codex-smart-routing-audit.jsonl"
DECISIONS_PATH = APP_DIR / "codex-smart-routing-decisions.jsonl"

_GPT_RE = re.compile(r"gpt-(\d+)(?:[.-](\d+))?(?:[.-](\d+))?(-.+|[a-z].*)?")

_normalize_model = routing.normalize_model


def request_routing_decision(
    workspace: str,
    token: str,
    task: str,
    available_models: list[str],
    *,
    timeout: float = REQUEST_TIMEOUT_S,
    log: Callable[[str], None] | None = None,
) -> tuple[RoutingDecision | None, str | None]:
    """Ask the router for a servable Codex model."""
    available = {_normalize_model(model): model for model in available_models}
    route_options = [(model, "codex") for model in available]
    if not route_options:
        return None, "no cached model services are available"
    router_name = routing.configured_router_name()
    if log is not None:
        payload = {
            "route_options": [
                {"model": model, "harness": harness} for model, harness in route_options
            ],
            "task": {"prompt": task},
            "route_selector": {"router_name": router_name},
        }
        url = workspace.rstrip("/") + ROUTING_PATH
        log(f"[ROUTE] request POST {url}: {json.dumps(payload, separators=(',', ':'))}")
    return routing.select_route(
        workspace,
        token,
        task,
        route_options,
        lambda raw_model: available.get(_normalize_model(raw_model)),
        router_name=router_name,
        timeout=timeout,
    )


def resolve_routed_model(raw_model: str, available_models: list[str]) -> str | None:
    """Map a router arm to a model the configured workspace can serve."""
    normalized = {_normalize_model(model): model for model in available_models}
    return normalized.get(_normalize_model(raw_model))


def route_pre_tool_use(
    payload: dict[str, Any],
    *,
    workspace: str,
    token: str,
    available_models: list[str],
    timeout: float = REQUEST_TIMEOUT_S,
    audit_decision: bool = False,
) -> dict[str, Any] | None:
    """Route one Codex ``spawn_agent`` call and rewrite its model."""
    record = None
    if audit_decision:

        def record(payload, task, decision, requested):
            routing.write_decision_record(DECISIONS_PATH, payload, task, decision, requested)

    return routing.route_spawn_tool(
        payload,
        is_spawn_agent=is_spawn_agent_tool,
        decision_fn=lambda task: request_routing_decision(
            workspace, token, task, available_models, timeout=timeout
        ),
        default_task_label="Codex subagent task",
        model_id_mapper=codex_model_id,
        record_decision=record,
    )


def is_spawn_agent_tool(tool_name: Any) -> bool:
    """Return whether a hook payload names Codex's subagent spawn tool."""
    if not isinstance(tool_name, str):
        return False
    normalized = tool_name.strip().lower()
    return normalized == "agent" or normalized.endswith(SPAWN_AGENT_TOOL_SUFFIX)


def record_session_start(payload: dict[str, Any]) -> None:
    """Write a canary proving Codex trusted and ran the routing hooks."""
    routing.record_session_start(CANARY_PATH, payload)


def record_subagent_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Append the model Codex actually selected for a routed subagent."""
    return routing.record_subagent_start(DECISIONS_PATH, AUDIT_PATH, payload)


def clear_routing_artifacts() -> None:
    """Remove ucode-owned routing canary and audit files."""
    routing.clear_artifacts((CANARY_PATH, AUDIT_PATH, DECISIONS_PATH))


def _parse_gpt(model: str) -> tuple[int, int, int, str] | None:
    match = _GPT_RE.fullmatch(_normalize_model(model))
    if not match:
        return None
    major, minor, patch, suffix = match.groups()
    return int(major), int(minor or 0), int(patch or 0), suffix or ""


def _model_strength(model: str) -> tuple[int, int, int, int]:
    parsed = _parse_gpt(model)
    if parsed is None:
        return (0, 0, 0, 0)
    major, minor, patch, suffix = parsed
    return major, minor, patch, 1 if not suffix else 0


def codex_model_id(model: str) -> str:
    """Map a UC GPT service ID to Codex's bundled catalog slug.

    Codex's bundled GPT catalog owns the model metadata for these aliases,
    while the AI Gateway resolves them back to the matching ``system.ai`` service.
    Leave non-GPT models unchanged because their metadata comes from the gateway catalog.
    """
    tail = model.rsplit("/", 1)[-1]
    if tail in {"databricks-gpt-5-2-codex", "databricks-gpt-5-4-nano"}:
        return tail
    if model.startswith("system.ai."):
        bare = model.removeprefix("system.ai.")
    elif tail.startswith("databricks-"):
        bare = tail.removeprefix("databricks-")
    else:
        return model
    match = _GPT_RE.fullmatch(bare)
    if not match:
        return model
    major, minor, patch, suffix = match.groups()
    version = major
    if minor is not None:
        version += f".{minor}"
    if patch is not None:
        version += f".{patch}"
    return f"gpt-{version}{suffix or ''}"
