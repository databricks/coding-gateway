"""Admin-authored managed coding-agent config: fetch, normalize, and local persistence.

An org admin authors a ``CodingAgentConfig`` through the Databricks AI Gateway; developers read it
(non-admin) and ``ucode`` applies it locally. This module owns the fetch/normalize side and the one
local file, ``~/.ucode/managed-state.json`` (0600), that both roles share:

- fetching the raw manifest (via :func:`ucode.databricks.fetch_managed_coding_agent_configs`),
- normalizing the proto-JSON into a stable internal dict keyed by ucode's own tool names,
- persisting it via :func:`save_managed_state` / :func:`load_managed_state` — the admin-write side
  (``managed_setup`` / ``managed_wizard``) authors the manifest here, and the launch path pulls the
  published copy back into the same file, and
- re-reading it on each launch, falling back to the persisted copy when the read fails.

There is deliberately one file, not a separate authored ``managed-settings.json``: the workspace is
the source of truth, so an authored draft and the pulled copy are the same shape and coexist in
``managed-state.json``. ``ucode setup`` authors the draft; ``ucode publish`` publishes it; a launch
then pulls the published copy back into the same file.

:func:`refresh_managed_config` is the launch path's entry point. It is called before model discovery,
because the manifest decides whether that discovery is needed at all; the launch path then hands the
manifest to :func:`ucode.managed_resolve.resolve_state` once the state it layers over is final.
Deciding *which* value wins for a given key is :mod:`ucode.managed_resolve`'s job, kept separate so
that logic stays pure and I/O-free.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import NamedTuple, cast

import ucode.config_io as config_io
from ucode.databricks import (
    fetch_managed_coding_agent_configs,
    fetch_model_recommendation,
    get_databricks_token,
)
from ucode.ui import console, print_warning

MANAGED_STATE_PATH = config_io.APP_DIR / "managed-state.json"

MANAGED_CONFIG_TTL_SECONDS = 30 * 60

# Shown to a developer when their workspace has no admin-defined managed config yet — the normal
# case, not an error. Kept here so the CLI (which surfaces it) uses one consistent message.
NO_MANAGED_CONFIG_MESSAGE = "No coding-agent config has been set up by your workspace admin yet."

# CodingAgent proto enum -> ucode tool name. Anything unrecognized (e.g. a newer agent this ucode
# build doesn't know) is dropped during normalization rather than guessed at. Public because the
# admin-write side (``managed_setup``) inverts these maps to serialize, so a new agent or MCP type
# only has to be declared once.
AGENT_ENUM_TO_TOOL: dict[str, str] = {
    "CODING_AGENT_CLAUDE_CODE": "claude",
    "CODING_AGENT_CODEX": "codex",
    "CODING_AGENT_GEMINI": "gemini",
    "CODING_AGENT_COPILOT": "copilot",
    "CODING_AGENT_PI": "pi",
    "CODING_AGENT_OPENCODE": "opencode",
}

_AGENT_ENUM_PREFIX = "CODING_AGENT_"
AGENT_NAME_TO_TOOL: dict[str, str] = {
    enum[len(_AGENT_ENUM_PREFIX) :].lower(): tool for enum, tool in AGENT_ENUM_TO_TOOL.items()
}

MAX_SPEC_VERSION = 1


# McpServerType proto enum -> ucode's short type tag. Mirrors the selection prefixes in ``mcp.py``;
# the actual name->URL resolution happens there when the manifest is applied (a later change).
MCP_TYPE_ENUM_TO_TAG: dict[str, str] = {
    "MCP_SERVER_TYPE_UC_SERVICE": "mcp-service",
    "MCP_SERVER_TYPE_EXTERNAL": "external",
    "MCP_SERVER_TYPE_GENIE": "genie-space",
    "MCP_SERVER_TYPE_VECTOR_SEARCH": "vector-search",
    "MCP_SERVER_TYPE_UC_FUNCTIONS": "uc-functions",
    "MCP_SERVER_TYPE_DATABRICKS_APP": "app",
    "MCP_SERVER_TYPE_DATABRICKS_SQL": "sql",
}


class FetchedManagedConfig(NamedTuple):
    """A managed-config read: the normalized ``manifest`` (None when the workspace has none) and,
    when the read did not settle the question, the ``reason`` it failed (None on a clean answer)."""

    manifest: dict | None
    reason: str | None


class ManagedConfigResult(NamedTuple):
    """The launch-path refresh outcome: the ``manifest`` to apply (None when absent or dropped) and
    ``feature_disabled``, True when the coding-agent-configs feature is off server-side."""

    manifest: dict | None
    feature_disabled: bool


def _as_dict(value: object) -> dict[str, object]:
    """Return ``value`` as a ``dict[str, object]`` when it is a dict, else an empty dict.

    Centralizes the isinstance-narrowing so downstream ``.get`` calls type-check (a bare
    ``isinstance(x, dict)`` narrows to ``dict[Never, Never]``, which rejects string keys)."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _str(value: object) -> str | None:
    """Return a non-empty stripped string, or None."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = _str(item)
        if s:
            out.append(s)
    return out


def _normalize_model_config(model_config: object) -> dict | None:
    """Normalize an ``AgentModelConfig`` oneof into ``{model_provider_service?, default_model?,
    models}``.

    The proto is a oneof over per-agent variants (claude/codex/opencode/pi/gemini/copilot). We
    don't care which variant tag it is here — the enclosing agent already tells us — so we read the
    common fields. Claude's ``models`` is a dict of family slots; the rest are a flat list. Returns
    None when there's no usable model config.
    """
    mc = _as_dict(model_config)
    if not mc:
        return None
    # Unwrap the oneof: take whichever single variant sub-dict is present.
    variant = next((_as_dict(v) for v in mc.values() if isinstance(v, dict)), None)
    if not variant:
        return None
    result: dict = {}
    mps = _str(variant.get("model_provider_service"))
    if mps:
        result["model_provider_service"] = mps
    default_model = _str(variant.get("default_model"))
    if default_model:
        result["default_model"] = default_model
    models = variant.get("models")
    if isinstance(models, dict):
        # Claude family slots (default_opus_model, default_sonnet_model, ...).
        slots = {k: _str(v) for k, v in _as_dict(models).items() if _str(v)}
        if slots:
            result["models"] = slots
    else:
        model_list = _str_list(models)
        if model_list:
            result["models"] = model_list
    return result or None


def _normalize_agent_config(config: object) -> dict:
    """Normalize an ``AgentConfig`` (the inner per-agent config) into the internal shape.

    The v2 fields win over the deprecated predecessors the proto keeps: ``http_headers`` supersedes
    ``custom_headers``, and the ``models`` / ``default_model`` / ``default_models`` triple
    supersedes the ``model_config`` oneof. A pre-v2 config carrying only the old fields still
    normalizes, so the transition until the server stops emitting deprecated fields is covered.
    """
    config_in = _as_dict(config)
    agent_config: dict = {}
    headers = _clean_str_dict(config_in.get("http_headers")) or _clean_str_dict(
        config_in.get("custom_headers")
    )
    if headers:
        agent_config["custom_headers"] = headers
    tracing_table = _tracing_table(config_in.get("tracing")) or _tracing_table(
        config_in.get("tracing_config")
    )
    if tracing_table:
        agent_config["tracing_table"] = tracing_table
    model_config = _normalize_model_config_v2(config_in) or _normalize_model_config(
        config_in.get("model_config")
    )
    if model_config is not None:
        agent_config["model_config"] = model_config
    return agent_config


def _normalize_enabled_agent(entry: object) -> tuple[str, dict] | None:
    """Normalize one repeated ``EnabledAgent`` (``{agent, config}``) into ``(tool, agent_config)``.

    This is the wire shape both the legacy and the v2 server use: ``enabled_agents`` stays a repeated
    list keyed by the ``agent`` enum (proto map keys can't be enums), and only the inner ``config``
    gained v2 fields. Drops entries whose agent is unset/unknown to this ucode build.
    """
    entry_dict = _as_dict(entry)
    if not entry_dict:
        return None
    tool = _resolve_agent_tool(entry_dict.get("agent"))
    if tool is None:
        return None
    return tool, _normalize_agent_config(entry_dict.get("config"))


def _resolve_agent_tool(key: object) -> str | None:
    """Map an agent reference to a ucode tool name, accepting either spelling.

    v1 responses carry the proto enum (``CODING_AGENT_CLAUDE_CODE``); v2 keys agents by name
    (``claude_code``). Both resolve to the same tool, or None when this build doesn't know the agent.
    """
    name = _str(key)
    if name is None:
        return None
    return AGENT_ENUM_TO_TOOL.get(name) or AGENT_NAME_TO_TOOL.get(name)


def _normalize_model_config_v2(agent: dict[str, object]) -> dict | None:
    """Normalize a v2 ``AgentConfig``'s model fields into the internal ``model_config`` shape.

    Reads the v2 shape: ``default_models`` map (with keys ``default_model``, ``default_opus_model``,
    etc.), and the ``models`` object with its alternatives (``model_provider_service``,
    ``unity_catalog_location``, or ``model_services``). The server does not enforce exactly-one under
    ``models``, so all present forms are carried and each consumer picks its precedence. Returns None
    when the config carries none of them, so the caller can fall back to the deprecated ``model_config``
    oneof.
    """
    result: dict = {}
    default_models = _as_dict(agent.get("default_models"))
    overall_default = _str(default_models.get("default_model"))
    if overall_default:
        result["default_model"] = overall_default
    models_dict = _as_dict(agent.get("models"))
    provider = _str(models_dict.get("model_provider_service"))
    if provider:
        result["model_provider_service"] = provider
    location = _str(models_dict.get("unity_catalog_location"))
    if location:
        result["model_service_location"] = location
    model_services = _str_list(models_dict.get("model_services"))
    if model_services:
        result["names"] = model_services
    slots = {
        slot: model
        for slot in (
            "default_opus_model",
            "default_sonnet_model",
            "default_haiku_model",
            "default_fable_model",
        )
        if (model := _str(default_models.get(slot)))
    }
    if slots:
        result["models"] = slots
    return result or None


def _clean_str_dict(value: object) -> dict[str, str]:
    """Keep only the string->string entries of ``value`` (a headers map), or an empty dict."""
    return {k: v for k, v in _as_dict(value).items() if isinstance(k, str) and isinstance(v, str)}


def _tracing_table(tracing: object) -> str | None:
    """Extract ``TracingConfig.table`` (a UC table FQN), or None."""
    return _str(_as_dict(tracing).get("table"))


def _normalize_mcp_servers(value: object) -> list[dict]:
    """Normalize ``mcp_servers`` wire format into internal ``list[dict]`` with ``name`` and ``type``.

    Accepts both the new wire format (``{names: [...], tags: [...]}`` parallel arrays) and the old
    repeated-list format (for backward compat).
    """
    value_dict = _as_dict(value)
    if value_dict:
        names = value_dict.get("names")
        tags = value_dict.get("tags")
        if isinstance(names, list) and isinstance(tags, list) and len(names) == len(tags):
            out: list[dict] = []
            for name, tag in zip(names, tags, strict=True):
                name_str = _str(name)
                tag_str = MCP_TYPE_ENUM_TO_TAG.get(_str(tag) or "")
                if name_str and tag_str:
                    out.append({"name": name_str, "type": tag_str})
            return out
    if isinstance(value, list):
        out = []
        for entry in value:
            entry_dict = _as_dict(entry)
            name = _str(entry_dict.get("name"))
            tag = MCP_TYPE_ENUM_TO_TAG.get(_str(entry_dict.get("type")) or "")
            if name and tag:
                out.append({"name": name, "type": tag})
        return out
    return []


def _normalize_budget_policy(value: object) -> dict | None:
    bp = _as_dict(value)
    if not bp:
        return None
    policy: dict = {}
    display_name = _str(bp.get("display_name"))
    if display_name:
        policy["display_name"] = display_name
    budget_id = _str(bp.get("budget_id"))
    if budget_id:
        policy["budget_id"] = budget_id
    tiers: list[dict] = []
    raw_tiers = bp.get("tiers")
    for tier in raw_tiers if isinstance(raw_tiers, list) else []:
        tier_dict = _as_dict(tier)
        pct = tier_dict.get("spending_percentage")
        if not isinstance(pct, (int, float)) or isinstance(pct, bool):
            continue
        tier_out: dict = {"spending_percentage": float(pct)}
        agent = _resolve_agent_tool(
            tier_dict.get("recommended_agent") or tier_dict.get("default_agent")
        )
        if agent:
            tier_out["default_agent"] = agent
        model = _str(tier_dict.get("recommended_model")) or _str(tier_dict.get("default_model"))
        if model:
            tier_out["default_model"] = model
        tiers.append(tier_out)
    if tiers:
        policy["tiers"] = tiers
    return policy or None


def normalize_managed_config(raw: dict) -> dict:
    """Normalize a raw ``CodingAgentConfig`` proto-JSON dict into ucode's internal shape.

    The internal shape uses ucode's own tool names and short MCP type tags so downstream reconcile
    and apply code never touches proto enum spellings. Unknown agents / MCP types are dropped.
    """
    raw = _as_dict(raw)
    result: dict = {}
    name = _str(raw.get("name"))
    if name:
        result["name"] = name
    display_name = _str(raw.get("display_name"))
    if display_name:
        result["display_name"] = display_name
    default_agent = _resolve_agent_tool(raw.get("default_agent"))
    if default_agent:
        result["default_agent"] = default_agent
    enabled_agents = _normalize_enabled_agents(raw.get("enabled_agents"))
    if enabled_agents:
        result["enabled_agents"] = enabled_agents
    mcp_servers = _normalize_mcp_servers(raw.get("mcp_servers"))
    if mcp_servers:
        result["mcp_servers"] = mcp_servers
    skills_obj = _as_dict(raw.get("skills"))
    skill_names = _str_list(skills_obj.get("names"))
    if skill_names:
        result["skills"] = {"names": skill_names}
    tracing_table = _tracing_table(raw.get("tracing"))
    if tracing_table:
        result["tracing_table"] = tracing_table
    spend_tiers = _normalize_budget_policy(raw.get("spend_tiers") or raw.get("budget_policy"))
    if spend_tiers is not None:
        result["budget_policy"] = spend_tiers
    return result


def _normalize_enabled_agents(raw_agents: object) -> dict[str, dict]:
    """Normalize ``enabled_agents`` into ``{tool: agent_config}``.

    The server (v1 and v2 alike) sends a repeated ``EnabledAgent`` list, each entry carrying its own
    ``agent`` enum plus a per-agent ``config``. A map keyed by agent name is also accepted defensively.
    Either way the result keys by ucode tool name, dropping agents this build doesn't recognize.
    """
    enabled_agents: dict[str, dict] = {}
    if isinstance(raw_agents, list):
        for entry in raw_agents:
            normalized = _normalize_enabled_agent(entry)
            if normalized is not None:
                tool, agent_config = normalized
                enabled_agents[tool] = agent_config
    elif isinstance(raw_agents, dict):
        for key, agent in raw_agents.items():
            tool = _resolve_agent_tool(key)
            if tool is not None:
                enabled_agents[tool] = _normalize_agent_config(agent)
    return enabled_agents


def _decimal(value: object) -> float | None:
    """Parse one of the API's decimal-string money fields, or None when absent/unparseable."""
    text = _str(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def get_model_recommendation(workspace: str, token: str) -> tuple[dict | None, str | None]:
    """Fetch the agent and model the caller's budget tier allows, normalized for the launch path.

    Returns ``(recommendation, reason)`` where the recommendation is ``{"agent", "model",
    "current_spend", "effective_threshold"}``. Every field is optional server-side, so each is
    normalized independently: an agent this build doesn't recognize is dropped rather than failing
    the read, and a model can arrive without an agent.
    """
    payload, reason = fetch_model_recommendation(workspace, token)
    if reason is not None:
        return None, reason
    agent = AGENT_ENUM_TO_TOOL.get(_str(payload.get("recommended_agent")) or "")
    model = _str(payload.get("recommended_model"))
    spend = _decimal(payload.get("current_spend"))
    threshold = _decimal(payload.get("effective_threshold"))
    if agent is None and model is None and spend is None and threshold is None:
        return None, None
    return {
        "agent": agent,
        "model": model,
        "current_spend": spend,
        "effective_threshold": threshold,
    }, None


def get_managed_config(workspace: str, token: str) -> FetchedManagedConfig:
    """Fetch and normalize the workspace's managed config.

    Returns a :class:`FetchedManagedConfig`:
    - ``manifest`` set, ``reason`` None — the normalized manifest for the workspace's single config;
    - both None — the workspace definitively has no managed config (not an error);
    - ``manifest`` None, ``reason`` set — the read didn't settle the question; ``reason`` says why.

    The distinction matters to callers that cache: only "both None" is authoritative enough to clear
    a previously stored config. "No config defined" arrives two ways depending on the backend — an
    empty listing (HTTP 200 with no configs) or a NOT_FOUND — and both collapse to "both None".
    Anything else, including a PERMISSION_DENIED, leaves the question unanswered and is surfaced as a
    failure: an admin may have published a config the developer can't read, which they need to know
    about rather than silently launch without.

    v0 stores at most one config per workspace, so the first entry is the workspace's config.

    ``UCODE_MANAGED_CONFIG_STUB`` short-circuits the HTTP read: when it names a readable JSON file,
    that file's single CodingAgentConfig is used verbatim. It exists so this client can be exercised
    against the v2 shape before the server emits it (AIGTWY-4572); unset in normal use. See
    ``examples/managed-config-v2.stub.json`` for a sample.
    """
    stub = _stub_config()
    if stub is not None:
        return _gate_and_normalize(stub)
    configs, reason = fetch_managed_coding_agent_configs(workspace, token)
    if reason is not None:
        if _is_feature_disabled(reason):
            return FetchedManagedConfig(None, reason)
        # A NOT_FOUND means the admin hasn't defined a config for this workspace — not a failure.
        if _is_not_found(reason):
            return FetchedManagedConfig(None, None)
        return FetchedManagedConfig(None, reason)
    if not configs:
        return FetchedManagedConfig(None, None)
    return _gate_and_normalize(configs[0])


def _stub_config() -> dict | None:
    """The stub CodingAgentConfig named by ``UCODE_MANAGED_CONFIG_STUB``, or None when unset/bad."""
    path = os.environ.get("UCODE_MANAGED_CONFIG_STUB")
    if not path:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print_warning(f"UCODE_MANAGED_CONFIG_STUB could not be read ({exc}); ignoring it.")
        return None
    return raw if isinstance(raw, dict) else None


def _gate_and_normalize(raw: dict) -> FetchedManagedConfig:
    """Apply the ``spec_version`` forward-compat gate, then normalize.

    A config declaring a ``spec_version`` newer than this build understands is refused as an
    unresolved read (``reason`` set), so the launch path falls back to the last-known-good cache and
    never blocks — the same treatment as any read this build can't act on.
    """
    spec = raw.get("spec_version")
    if spec is not None:
        if isinstance(spec, bool) or not isinstance(spec, int):
            return FetchedManagedConfig(
                None,
                f"This workspace's managed config has an unrecognized spec_version ({spec!r}); "
                "update Unity Gateway with `ug upgrade`.",
            )
        if spec > MAX_SPEC_VERSION:
            return FetchedManagedConfig(
                None,
                f"This workspace's managed config needs a newer Unity Gateway (spec_version {spec}; "
                f"this build supports up to {MAX_SPEC_VERSION}). Run `ug upgrade`.",
            )
    return FetchedManagedConfig(normalize_managed_config(raw), None)


def _is_not_found(reason: str) -> bool:
    """True when a read failure reason means the workspace definitively has no managed config.

    ``_http_get_json`` formats failures as ``HTTP <code> <text>[: <body>]``; a NOT_FOUND surfaces
    as an ``HTTP 404`` there (and the API's error body carries ``NOT_FOUND``)."""
    lowered = reason.lower()
    return "http 404" in lowered or "not_found" in lowered


def _is_permission_denied(reason: str) -> bool:
    """True when the read was refused rather than answering whether a config exists.

    The read is meant to be available to any workspace user, so a refusal means the workspace's
    managed config isn't readable by this developer — worth telling them about, since an admin may
    have published a config that silently isn't reaching them. It settles nothing about whether one
    exists, so a cached config is left in place rather than cleared."""
    lowered = reason.lower()
    return "http 403" in lowered or "permission_denied" in lowered


def save_managed_state(workspace: str, config: dict, *, retrieved_at: float | None = None) -> None:
    """Persist the normalized managed config to ``~/.ucode/managed-state.json`` at mode 0600.

    The file is org-authored, not developer-editable — 0600 keeps it readable/writable only by the
    user (a light guard; hard enforcement / sudo ownership is a separate concern). No-op in dry-run.

    An empty ``config`` records "this workspace has no managed config", which matters because the
    file doubles as the fallback when a later read fails: without it, removing a config server-side
    would leave the old one on disk to be reapplied after a transient outage.

    ``retrieved_at`` (epoch seconds) records when this config was fetched from the workspace and is
    set only by :func:`refresh_managed_config`; it drives the launch-time TTL skip. It is left unset
    by default so a locally-authored draft (``ucode setup``) is never mistaken for fetched state that
    a launch could apply without reading the workspace.
    """
    payload: dict = {"workspace": workspace, "config": config}
    if retrieved_at is not None:
        payload["retrieved_at"] = retrieved_at
    if config_io.is_dry_run():
        # Print rather than write, matching how the agent config writers behave under --dry-run.
        console.print(
            f"\n[bold]\\[dry run] {MANAGED_STATE_PATH}[/bold]\n{json.dumps(payload, indent=2)}\n"
        )
        return
    config_io.ensure_parent_dir(MANAGED_STATE_PATH)
    try:
        MANAGED_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write managed state file: {MANAGED_STATE_PATH}") from exc
    _restrict_permissions(MANAGED_STATE_PATH)


def _restrict_permissions(path: Path) -> None:
    """Best-effort chmod 0600. No-op where unsupported (e.g. Windows), where the effective
    read-only guarantee is left to a later change."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def load_managed_state(workspace: str | None) -> dict | None:
    """Load the persisted managed config for ``workspace``, or None if absent/mismatched.

    Returns the normalized config dict (the ``config`` field), only when the stored file is for the
    same workspace — so a stale file from another workspace is ignored rather than misapplied.

    This is the single local managed config: ``ucode setup`` authors it here, ``ucode publish``
    publishes it, and a launch refreshes it from the workspace. The admin-authored draft and the
    pulled copy share one file because the workspace is the source of truth — to keep a draft,
    publish it with ``ucode publish``.
    """
    if not workspace:
        return None
    data = config_io.read_json_safe(MANAGED_STATE_PATH)
    if data.get("workspace") != workspace:
        return None
    config = data.get("config")
    return config if isinstance(config, dict) else None


def _load_retrieved_at(workspace: str) -> float | None:
    """When the persisted config for ``workspace`` was last fetched, or None if absent/mismatched.

    A file from before this field existed simply has no timestamp, so it reads as stale and the next
    launch re-fetches — no migration needed.
    """
    data = config_io.read_json_safe(MANAGED_STATE_PATH)
    if data.get("workspace") != workspace:
        return None
    retrieved_at = data.get("retrieved_at")
    if isinstance(retrieved_at, bool) or not isinstance(retrieved_at, (int, float)):
        return None
    try:
        value = float(retrieved_at)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def managed_state_workspace() -> str | None:
    """The workspace the on-disk managed config was authored/pulled for, or None when there is none.

    Lets a caller that has no workspace in local ucode state (e.g. ``ucode setup --show`` before
    ``ucode configure``) still find the manifest on disk and report which workspace it belongs to.
    """
    workspace = config_io.read_json_safe(MANAGED_STATE_PATH).get("workspace")
    return workspace if isinstance(workspace, str) and workspace else None


def refresh_managed_config(state: dict, *, force: bool = False) -> ManagedConfigResult:
    """Fetch the workspace's managed config and persist it as a :class:`ManagedConfigResult`.

    Runs on every launch so a developer picks up an admin's edits without re-running
    ``ucode configure``. The manifest is None when the workspace has no managed config — the normal
    case for a workspace whose admin hasn't published one.

    A recently-fetched config is reused without a network round trip: when a non-empty config was
    persisted within :data:`MANAGED_CONFIG_TTL_SECONDS`, that cached config is returned as-is and no
    fetch or re-sync happens, so back-to-back launches don't re-hit the control plane. ``force``
    bypasses the TTL to always fetch. Only a positive cached config short-circuits — an empty cache
    (no config, or a feature-disabled marker) always re-fetches so those states stay accurate.

    A failed fetch never blocks the launch: an unreachable control plane shouldn't stop someone from
    coding. Instead it falls back to the last config persisted for this workspace, so the admin's
    most recent known policy still applies; only when there is no persisted config either does the
    launch fall through to the developer's own settings. ``FEATURE_DISABLED`` is the exception — it
    is an authoritative "off", not a transient failure, so it drops the cache rather than falling
    back (see below).

    ``coding_agent_config_feature_disabled`` is True whenever the gateway returned ``FEATURE_DISABLED`` —
    the coding-agent-configs feature isn't enabled server-side, so callers suppress the ``ucode
    setup`` recommendation. A config cached from when the feature was enabled is discarded in that
    case (returned manifest is None), so a launch doesn't re-apply a policy the workspace has turned
    off and ``ug configure`` doesn't route into a managed-setup flow that would dead-end.
    """
    workspace = state.get("workspace")
    if not workspace:
        return ManagedConfigResult(None, False)
    if not force:
        cached = _fresh_cached_config(workspace)
        if cached is not None:
            return ManagedConfigResult(cached, False)
    try:
        token = get_databricks_token(workspace, state.get("profile"))
    except RuntimeError as exc:
        return ManagedConfigResult(_persisted_fallback(workspace, str(exc)), False)
    managed, reason = get_managed_config(workspace, token)
    if reason is not None:
        if _is_feature_disabled(reason):
            save_managed_state(workspace, {})
            return ManagedConfigResult(None, True)
        fallback = _persisted_fallback(workspace, reason, refused=_is_permission_denied(reason))
        return ManagedConfigResult(fallback, False)
    if managed is None:
        # Record that this workspace has no config, rather than leaving an earlier one on disk:
        # the file doubles as the fallback above, so a removed policy would otherwise come back
        # into force after the next transient outage.
        save_managed_state(workspace, {})
        return ManagedConfigResult(None, False)
    save_managed_state(workspace, managed, retrieved_at=time.time())
    return ManagedConfigResult(managed, False)


def _fresh_cached_config(workspace: str) -> dict | None:
    """The persisted config for ``workspace`` when it is still within the TTL, else None.

    Returns only a non-empty config: an empty cache (no config, or a feature-disabled marker) reads
    as "not fresh" so the caller re-fetches and keeps those states current.
    """
    retrieved_at = _load_retrieved_at(workspace)
    if retrieved_at is None:
        return None
    age = time.time() - retrieved_at
    if not 0 <= age < MANAGED_CONFIG_TTL_SECONDS:
        return None
    return load_managed_state(workspace) or None


def _is_feature_disabled(reason: str) -> bool:
    return "feature_disabled" in reason.lower()


def _persisted_fallback(workspace: str, reason: str, *, refused: bool = False) -> dict | None:
    """Return the last persisted config for ``workspace`` after a failed fetch.

    Warns only when there is a config to fall back on, because then the launch proceeds on an admin
    policy that may be out of date. With nothing persisted there is no managed config in play at
    all, so staying quiet keeps someone with (say) an expired session from being told about a
    feature they don't use — including when the read was ``refused``, since a refusal is no evidence
    that a config exists.
    """
    # An empty persisted config means the last successful read found none, so there is no admin
    # policy to fall back to — treat it the same as having no file at all.
    persisted = load_managed_state(workspace)
    if not persisted:
        return None
    summary = _summarize_read_failure(reason)
    if refused:
        print_warning(
            f"Your workspace's managed config is not readable by you ({summary}); using the last "
            "one saved for this workspace. Ask an admin to grant access."
        )
    else:
        print_warning(
            f"Could not read your workspace's managed config ({summary}); "
            "using the last one saved for this workspace."
        )
    return persisted


def _summarize_read_failure(reason: str) -> str:
    """Condense a read failure into one short line fit for a terminal warning.

    ``_http_get_json`` appends the raw response body, which for a gateway error is a multi-line JSON
    blob (error_code, message, request_id, trace ids). Surface just the status and the API's own
    message; the full text is still available under ``UCODE_DEBUG=1``.
    """
    status, _, body = reason.partition(": ")
    body = body.strip()
    if body.startswith("{"):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            message = _str(parsed.get("message")) or _str(parsed.get("error_code"))
            if message:
                return f"{status.strip()}: {message}"
        return status.strip()
    condensed = " ".join(reason.split())
    return condensed if len(condensed) <= 160 else condensed[:157] + "..."
