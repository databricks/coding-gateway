"""Claude Code agent: writes ~/.claude/settings.json env block."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ucode import gateway_proxy
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    write_json_file,
)
from ucode.constants import LOOPBACK_HOST
from ucode.custom_oauth import CustomOAuthConfig, build_custom_auth_shell_command
from ucode.databricks import (
    build_auth_shell_command,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.launcher import exec_or_spawn
from ucode.managed_files import (
    OS,
    current_os,
    managed_file_conflicts,
    managed_file_is_verified,
    managed_file_status,
    managed_writes_allowed,
    mark_managed_file_verified,
    read_managed_file,
    reconcile_managed_file,
    revert_managed_file,
)
from ucode.smart_routing import v2 as smart_routing_v2
from ucode.smart_routing.claude_hooks import (
    remove_smart_routing_hooks,
    sync_smart_routing_hooks,
)
from ucode.state import MANAGED_OVERLAY_KEY, get_provider_service, mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version
from ucode.tracing import tracing_env
from ucode.ui import print_note, print_success, print_warning

from .args import LaunchOptions, has_explicit_model_arg

GATEWAY_MODEL_DISCOVERY_ENV_VAR = "ENABLE_CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY"
CLAUDE_CONFIG_DIR = Path.home() / ".claude"
CLAUDE_SETTINGS_PATH = CLAUDE_CONFIG_DIR / "ucode-settings.json"
CLAUDE_MCP_CONFIG_PATH = Path.home() / ".claude.json"
# The default model is stored in Claude's default user settings, not the ucode settings.
CLAUDE_USER_SETTINGS_PATH = CLAUDE_CONFIG_DIR / "settings.json"
CLAUDE_BACKUP_PATH = APP_DIR / "claude-ucode-settings.backup.json"
WEB_SEARCH_MCP_STATE_KEY = "claude_web_search_mcp"
MINIMUM_CLAUDE_VERSION = (2, 1, 248)
MINIMUM_CLAUDE_VERSION_TEXT = "2.1.248"

SPEC: ToolSpec = {
    "binary": "claude",
    "package": "@anthropic-ai/claude-code",
    "display": "Claude Code",
    "config_path": CLAUDE_SETTINGS_PATH,
    "backup_path": CLAUDE_BACKUP_PATH,
}

# Retained only to identify and remove state written by the legacy persisted opt-in.
SMART_ROUTING_STATE_KEY = smart_routing_v2.LEGACY_STATE_KEY


def _parse_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _minimum_version_requirement_message(version: str) -> str:
    feature = "Smart routing" if smart_routing_v2.enabled() else "Model discovery"
    return (
        f"{feature} requires Claude Code {MINIMUM_CLAUDE_VERSION_TEXT} or newer. "
        f"Your current version is Claude Code {version}."
    )


def minimum_version_error() -> str | None:
    if os.environ.get(GATEWAY_MODEL_DISCOVERY_ENV_VAR) != "1" and not smart_routing_v2.enabled():
        return None
    version = agent_version(SPEC["binary"])
    parsed = _parse_version(version)
    if parsed is None or parsed >= MINIMUM_CLAUDE_VERSION:
        return None
    return _minimum_version_requirement_message(version)


def _resolve_web_search_model(state: dict) -> str | None:
    """Pick the model the web_search MCP server should call. Prefers an
    explicit override in state, otherwise the first endpoint discovered as
    Responses-API-capable. Returns None if no GPT endpoint is available —
    callers should skip the MCP wiring in that case."""
    override = state.get("web_search_model")
    if isinstance(override, str) and override.strip():
        return override.strip()
    codex_models = state.get("codex_models") or []
    if isinstance(codex_models, list) and codex_models:
        first = codex_models[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


WEB_SEARCH_MCP_NAME = "web_search"
# Matches both the AI Gateway form (`databricks-claude-opus-4-8`) and the UC
# model-services form (`system.ai.claude-opus-4-8`).
_CLAUDE_MODEL_RE = re.compile(
    r"^(?:system\.ai\.)?(?:databricks-)?claude-(opus|sonnet)-(\d+)(?:-(\d+))?(.*)$"
)

# Env keys the MLflow Stop hook reads to route traces. Written into the
# settings `env` block alongside the hook itself.
CLAUDE_TRACING_ENV_KEYS = (
    "MLFLOW_CLAUDE_TRACING_ENABLED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_ID",
    "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
)
# Model-selection env keys ucode manages. Existing family defaults in the enterprise-managed file
# are preserved unless Coding Agent Config explicitly supplies that family.
CLAUDE_MANAGED_MODEL_ENV_KEYS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL_NAME",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
)
CLAUDE_DEFAULT_MODEL_ENV_KEYS = {
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
}
# Launch-scoped feature flags that ucode may write into Claude settings. These
# must be removed again when the corresponding launch flag is absent.
CLAUDE_CONDITIONAL_ENV_KEYS = ("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",)
# Env keys ucode used to write but no longer does; stripped from the managed
# settings file on every launch so stale values never linger.
CLAUDE_REMOVED_ENV_KEYS = ("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",)
ANTHROPIC_CUSTOM_HEADERS_ENV_KEY = "ANTHROPIC_CUSTOM_HEADERS"
CLAUDE_MANAGED_CUSTOM_HEADER_NAMES = frozenset(
    {
        "x-databricks-use-coding-agent-mode",
        "user-agent",
        "databricks-model-provider-service",
    }
)
CLAUDE_TRACING_STOP_HOOK_SUFFIX = " autolog claude stop-hook"
# Tracing is driven by an `mlflow autolog claude stop-hook` Stop hook, run by
# the `mlflow` CLI on each session end. Pin to 3.11.x: 3.12 dropped the Unity
# Catalog trace-write path, so traces silently land in the classic store
# instead of the experiment's UC table. ucode installs this via `uv tool` at
# `configure tracing` time (where UV_INDEX_URL is set), then writes the hook
# with the resolved absolute path — so the hook needs no uv or index at run
# time, and can't be shadowed by a project venv's mlflow.
MLFLOW_CLI_SPEC = "mlflow[databricks]>=3.11,<3.12"
MINIMUM_MLFLOW_VERSION = (3, 11)
# Upper bound (exclusive) — an installed mlflow at or above this is too new and
# must be replaced, not just left alone.
MAXIMUM_MLFLOW_VERSION = (3, 12)

# Relayed drops the user scope to deliberately omit the stale apiKeyHelper. Only applied to relayed
# launches — normal launches keep loading user settings (hooks/permissions) as before.
_RELAYED_SETTING_SOURCES = "project,local"


def _managed_settings_path() -> Path | None:
    """OS-specific location of Claude Code's enterprise managed-settings.json.
    Returns None on unsupported platforms."""
    if current_os() is OS.LINUX:
        return Path("/etc/claude-code/managed-settings.json")
    if current_os() is OS.MACOS:
        return Path("/Library/Application Support/ClaudeCode/managed-settings.json")
    return None


def _parse_managed_settings(text: str) -> dict:
    try:
        settings = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(settings, dict):
        raise RuntimeError("the top-level JSON value must be an object")
    return settings


def _dump_managed_settings(settings: dict) -> str:
    return json.dumps(settings, indent=2) + "\n"


def managed_settings_are_current(state: dict) -> bool:
    path = _managed_settings_path()
    if path is None:
        return True
    if state.get("claude_relayed"):
        required_scope = "relay-compatible"
    elif managed_writes_allowed():
        required_scope = "managed"
    else:
        required_scope = None
    return managed_file_is_verified(state, "claude", path, required_scope=required_scope)


def gateway_model_discovery_setting_is_absent() -> bool:
    """Return whether model discovery is absent from persistent Claude settings."""
    env = read_json_safe(CLAUDE_SETTINGS_PATH).get("env")
    actual = (
        env.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") if isinstance(env, dict) else None
    )
    return actual is None


def managed_settings_status(state: dict) -> tuple[Path | None, str, str]:
    path = _managed_settings_path()
    status, backup = managed_file_status(state, "claude", path, parser=_parse_managed_settings)
    return path, status, backup


def revert_managed_settings() -> str:
    return revert_managed_file(
        "claude",
        display="Claude Code",
        parser=_parse_managed_settings,
        dumper=_dump_managed_settings,
    )


def _managed_relayed_conflicts(path: Path) -> list[str]:
    """Return managed settings that would override Claude subscription relay auth."""
    text = read_managed_file(path)
    if text is None:
        return []
    try:
        settings = _parse_managed_settings(text)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot safely inspect Claude Code managed settings at {path}: {exc}. Repair the "
            "file or contact your administrator."
        ) from exc
    conflicts: list[str] = []
    if settings.get("apiKeyHelper"):
        conflicts.append("apiKeyHelper")
    env = settings.get("env")
    if isinstance(env, dict):
        if env.get("ANTHROPIC_BASE_URL"):
            conflicts.append("env.ANTHROPIC_BASE_URL")
        if env.get("ANTHROPIC_CUSTOM_HEADERS"):
            conflicts.append("env.ANTHROPIC_CUSTOM_HEADERS")
    return conflicts


def relayed_proxy_base_url(state: dict) -> str:
    """Loopback base URL for the relayed refresh proxy, allocating a free port
    on first call and caching it in state so config and launch agree."""
    port = state.get("relayed_proxy_port")
    if not isinstance(port, int):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((LOOPBACK_HOST, 0))
            port = sock.getsockname()[1]
        state["relayed_proxy_port"] = port
    return f"http://{LOOPBACK_HOST}:{port}"


def _web_search_mcp_entry(workspace: str, search_model: str, profile: str | None = None) -> dict:
    """Stdio MCP server entry pointing at `ucode mcp web-search`. Resolves
    the absolute path to the `ucode` binary so launchers without the right
    PATH (e.g. desktop GUI launchers) still find it."""
    ucode_binary = shutil.which("ucode") or "ucode"
    env: dict[str, str] = {
        "DATABRICKS_HOST": workspace,
        "UCODE_WEB_SEARCH_MODEL": search_model,
    }
    if profile:
        env["DATABRICKS_CONFIG_PROFILE"] = profile
    return {
        "type": "stdio",
        "command": ucode_binary,
        "args": ["mcp", "web-search"],
        "env": env,
    }


def render_overlay(
    workspace: str,
    model: str | None,
    claude_models: dict[str, str] | None = None,
    disable_web_search: bool = False,
    profile: str | None = None,
    use_pat: bool = False,
    custom_oauth: CustomOAuthConfig | None = None,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
    fable_enabled: bool = False,
    relayed: bool = False,
    relayed_base_url: str | None = None,
    route_root_model: str | None = None,
    custom_model: str | None = None,
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for Claude settings.json.

    NOTE: MCP servers are NOT written here. Claude Code reads `mcpServers`
    from `~/.claude.json`, not `~/.claude/settings.json` — registration goes
    through `claude mcp add-json` (see `_register_web_search_mcp`).

    When `provider` is set (a `<catalog>.<schema>.<name>` Model Provider
    Service), the request is routed to that external provider via the
    `Databricks-Model-Provider-Service` header. An Anthropic-backed provider
    understands Claude Code's own canonical model names, so no model id is
    pinned. A Bedrock-backed provider exposes different model ids (e.g.
    `us.anthropic.claude-sonnet-4-6`), passed in `provider_models` by family —
    those get pinned via the `ANTHROPIC_DEFAULT_*_MODEL` env vars.

    When `relayed` is set (a credential-less Anthropic subscription-relay MPS,
    Claude Max/Team/Enterprise), Claude Code's own keychain OAuth must remain the
    `Authorization` credential, so no `apiKeyHelper` is written (it would outrank
    the subscription OAuth). The Databricks credential rides in the
    `X-Databricks-AI-Gateway-Token` swap header, injected per request by a local
    refresh proxy at `relayed_base_url` — not written here."""
    if relayed:
        if not relayed_base_url:
            raise RuntimeError("Relayed launch requires a proxy base URL.")
        base_url = relayed_base_url
    else:
        base_url = build_tool_base_url("claude", workspace)
    # ANTHROPIC_CUSTOM_HEADERS is parsed as `key: value` pairs separated by
    # newlines (Anthropic SDK convention). Setting User-Agent here overrides
    # the SDK's default UA on outbound requests so the gateway can attribute
    # traffic to ucode.
    header_lines = [
        "x-databricks-use-coding-agent-mode: true",
        f"User-Agent: ucode/{ucode_version()} claude/{agent_version('claude')}",
    ]
    if provider:
        header_lines.append(f"Databricks-Model-Provider-Service: {provider}")
    # Relayed: the X-Databricks-AI-Gateway-Token swap header is added per request
    # by the refresh proxy, not here — a static value would go stale mid-session.
    custom_headers = "\n".join(header_lines)
    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": custom_headers,
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "900000",
        # 1h prompt caching needs the extended-cache-ttl beta header, which
        # Claude Code only sends when experimental betas are enabled — so we must
        # not set CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS (see CLAUDE_REMOVED_ENV_KEYS).
        "ENABLE_PROMPT_CACHING_1H": "1",
        "ENABLE_TOOL_SEARCH": "true",
        "CLAUDE_CODE_USE_GATEWAY": "1",
    }
    # Intentionally NOT setting ANTHROPIC_MODEL by default. Setting it produces a
    # duplicate catalog row in Claude Code's /model picker (e.g. "Opus 4.8 (1M
    # context) ✓") on top of the family-alias row from ANTHROPIC_DEFAULT_OPUS_MODEL.
    # Without it, Default resolves through the pinned family alias and the picker
    # shows only one row per model. `ucode claude -- --model X` still overrides for
    # a single session via Claude Code's own --model flag.
    #
    # The one exception is smart routing: `route_root_model` pins the
    # router's per-launch pick for the root session as ANTHROPIC_MODEL. The
    # duplicate-picker-row cost is acceptable because the whole point is to launch
    # on the routed model rather than the family default.
    _ = model  # API stability; no longer pinned via env.
    if route_root_model:
        env["ANTHROPIC_MODEL"] = route_root_model
    # A Bedrock-backed provider needs its provider-side ids pinned verbatim
    # (Claude Code's canonical names aren't routable there). These come from the
    # service's targets, already de-duped to one id per family upstream.
    elif provider and provider_models:
        if provider_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = provider_models["opus"]
        if provider_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = provider_models["sonnet"]
        if provider_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = provider_models["haiku"]
    # With an Anthropic Model Provider Service, the header routes to the external
    # provider and Claude Code's own canonical model names are sent verbatim —
    # pinning a Databricks model id here would mislabel the picker and isn't
    # routable.
    elif claude_models and not provider:
        # Picker rows show the raw routable id (e.g. "system.ai.claude-opus-4-8[1m]")
        # so users can see which gateway-routable model is behind each shortcut.
        # We deliberately don't set the `_NAME` companion env vars — the raw id
        # is more useful than a friendly label for debugging gateway routing.
        #
        # Fable is opt-in only (`ucode configure --enable-fable`): it's a premium
        # model, so we don't pin the family alias unless the user asked for it.
        # When off, ANTHROPIC_DEFAULT_FABLE_MODEL is simply never written — and
        # since it's in CLAUDE_MANAGED_MODEL_ENV_KEYS, any stale value from a
        # prior `--enable-fable` run is pruned from settings.json on next launch.
        if fable_enabled and claude_models.get("fable"):
            env["ANTHROPIC_DEFAULT_FABLE_MODEL"] = claude_models["fable"]
        if claude_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = _maybe_add_1m_suffix(claude_models["opus"])
        if claude_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = _maybe_add_1m_suffix(claude_models["sonnet"])
        if claude_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = claude_models["haiku"]
    # Relayed omits apiKeyHelper so Claude Code's subscription OAuth stays the
    # Authorization credential; every other path uses it as the gateway auth.
    overlay: dict = {"env": env}
    if relayed:
        keys = [["env", k] for k in env]
    else:
        if custom_oauth:
            overlay["apiKeyHelper"] = build_custom_auth_shell_command(workspace, custom_oauth)
        else:
            overlay["apiKeyHelper"] = build_auth_shell_command(workspace, profile, use_pat=use_pat)
        keys = [["apiKeyHelper"]] + [["env", k] for k in env]

    # Disable Claude Code's built-in WebSearch: it declares Anthropic's hosted
    # `web_search_20250305` server tool, which the Databricks gateway rejects
    # (HTTP 400: "Input tag 'web_search_20250305' ... does not match"), so the
    # model wastes a turn on it before falling back. A *bare* `permissions.deny`
    # entry removes the tool from Claude's context entirely, so it is never
    # advertised to the model nor sent to the gateway. (Claude Code has no
    # `disabledTools` setting — the `permissions` block is the only settings.json
    # mechanism for built-in tools; a bare tool name in `deny` drops it, whereas
    # a scoped rule like `WebSearch(*)` would leave it advertised.) The
    # replacement `web_search` MCP server is registered separately via the
    # claude CLI.
    if disable_web_search:
        overlay["permissions"] = {"deny": ["WebSearch"]}
        keys.append(["permissions", "deny"])

    return overlay, keys


def _maybe_add_1m_suffix(model: str) -> str:
    if model.endswith("[1m]"):
        return model
    match = _CLAUDE_MODEL_RE.match(model)
    if not match:
        return model

    family, major_raw, minor_raw, _ = match.groups()
    major = int(major_raw)
    minor = int(minor_raw or 0)
    should_suffix = (family == "opus" and (major, minor) >= (4, 6)) or (
        family == "sonnet" and (major, minor) >= (4, 6)
    )
    return f"{model}[1m]" if should_suffix else model


def _enforce_model_default_hierarchy(
    family: str,
    *,
    coding_agent_config_defaults: dict[str, str],
    settings_file_existing_defaults: dict[str, str],
    ucode_defaults: dict[str, str],
) -> str | None:
    """Apply managed-file model precedence for one Claude family."""
    coding_agent_config_default_model = coding_agent_config_defaults.get(family)
    settings_file_existing_default_model = settings_file_existing_defaults.get(family)
    ucode_default_model = ucode_defaults.get(family)

    if coding_agent_config_default_model is not None:
        selected_default_model = coding_agent_config_default_model
    elif settings_file_existing_default_model is not None:
        return settings_file_existing_default_model
    else:
        selected_default_model = ucode_default_model

    if selected_default_model is None:
        return None
    if family in ("opus", "sonnet"):
        return _maybe_add_1m_suffix(selected_default_model)
    return selected_default_model


def _register_web_search_mcp(workspace: str, search_model: str, profile: str | None = None) -> bool:
    """Register (or replace) the web_search MCP server in Claude Code's user
    scope via `claude mcp add-json`. Removes any prior entry first so re-runs
    pick up changes to the workspace, model, or ucode binary path.

    Returns True if registration succeeded. Failures are non-blocking: we warn
    and return False so the rest of `ucode claude` setup can complete.
    """
    # Imported lazily to avoid a circular import via ucode.mcp -> ucode.agents.
    from ucode.mcp import (
        MCP_CLEANUP_SCOPES,
        add_claude_mcp_server,
        remove_claude_mcp_server,
    )

    for scope in MCP_CLEANUP_SCOPES:
        try:
            remove_claude_mcp_server(WEB_SEARCH_MCP_NAME, scope)
        except RuntimeError:
            # Best-effort cleanup of stale entries — keep going.
            pass
    entry = _web_search_mcp_entry(workspace, search_model, profile)
    try:
        add_claude_mcp_server(WEB_SEARCH_MCP_NAME, entry)
    except RuntimeError as exc:
        print_warning(f"{exc} Web search will be unavailable; re-run `ucode claude` to retry.")
        return False
    return True


def _web_search_mcp_is_current(state: dict, entry: dict) -> bool:
    """Return whether the desired web-search entry is already registered.

    The persisted entry acts as a cheap fingerprint, while reading Claude's config repairs a
    registration removed or edited outside ucode. Avoiding the Claude CLI here matters: each
    ``claude mcp`` subprocess takes roughly 0.8 seconds during a launch.
    """
    if state.get(WEB_SEARCH_MCP_STATE_KEY) != entry:
        return False
    config = read_json_safe(CLAUDE_MCP_CONFIG_PATH)
    servers = config.get("mcpServers")
    return isinstance(servers, dict) and servers.get(WEB_SEARCH_MCP_NAME) == entry


def _unregister_web_search_mcp() -> None:
    """Remove the web_search MCP server from all scopes. Used by revert."""
    from ucode.mcp import MCP_CLEANUP_SCOPES, remove_claude_mcp_server

    for scope in MCP_CLEANUP_SCOPES:
        try:
            remove_claude_mcp_server(WEB_SEARCH_MCP_NAME, scope)
        except RuntimeError:
            pass


def disable_smart_routing(state: dict) -> bool:
    """Disable routing and remove only ucode's Claude Code routing hooks."""
    state.pop(SMART_ROUTING_STATE_KEY, None)
    if state.get("workspace"):
        save_state(state)
    changed = False
    if CLAUDE_SETTINGS_PATH.exists():
        doc = read_json_safe(CLAUDE_SETTINGS_PATH)
        if remove_smart_routing_hooks(doc):
            write_json_file(CLAUDE_SETTINGS_PATH, doc)
            changed = True
    from ucode.smart_routing.claude_routing import clear_routing_artifacts

    clear_routing_artifacts()
    return changed


def write_tool_config(
    state: dict,
    model: str | None,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
    relayed: bool = False,
    route_root_model: str | None = None,
    custom_model: str | None = None,
    coding_agent_config_defaults: dict[str, str] | None = None,
) -> dict:
    backup_existing_file(CLAUDE_SETTINGS_PATH, CLAUDE_BACKUP_PATH)
    web_search_model = _resolve_web_search_model(state)
    # Relayed inference points at a local refresh proxy; its loopback base URL is
    # recorded in state so launch starts the proxy on the matching port.
    relayed_base_url = relayed_proxy_base_url(state) if relayed else None
    overlay, managed_keys = render_overlay(
        state["workspace"],
        model,
        state.get("claude_models") or {},
        disable_web_search=web_search_model is not None,
        profile=state.get("profile"),
        use_pat=bool(state.get("use_pat")),
        custom_oauth=state.get("custom_oauth"),
        provider=provider,
        provider_models=provider_models,
        fable_enabled=bool(state.get("fable_enabled")),
        relayed=relayed,
        relayed_base_url=relayed_base_url,
        route_root_model=route_root_model,
        custom_model=custom_model,
    )
    tracing_env_vars = tracing_env(state, "claude")
    stop_hook_command = claude_tracing_stop_hook_command() if tracing_env_vars else None
    if tracing_env_vars:
        overlay["env"]["MLFLOW_CLAUDE_TRACING_ENABLED"] = "true"
        overlay["env"].update(tracing_env_vars)
        managed_keys = managed_keys + [["env", key] for key in CLAUDE_TRACING_ENV_KEYS]
        if stop_hook_command:
            managed_keys = managed_keys + [["hooks", "Stop"]]
        else:
            print_warning(
                "MLflow tracing env was written, but the `mlflow` CLI could not be located "
                "to install the Claude Stop hook — traces won't be emitted. Re-run "
                "`ucode configure tracing`."
            )
    managed_file_keys = list(managed_keys)
    for path in (
        [["env", key] for key in CLAUDE_MANAGED_MODEL_ENV_KEYS]
        + [["env", key] for key in CLAUDE_CONDITIONAL_ENV_KEYS]
        + [["env", key] for key in CLAUDE_REMOVED_ENV_KEYS]
        + [["env", key] for key in CLAUDE_TRACING_ENV_KEYS]
        + [["hooks", "Stop"]]
        + [["hooks", event] for event in ("PreToolUse", "SessionStart", "SubagentStart")]
    ):
        if path not in managed_file_keys:
            managed_file_keys.append(path)

    # V2 installs routing hooks in a transient per-launch settings file. Persistent settings must
    # contain no ucode routing hooks; surgically strip legacy ones while preserving user hooks.
    def _compose(base: dict, *, enforce_model_default_hierarchy: bool) -> dict:
        base_env = base.get("env")
        existing_custom_headers = (
            base_env.get(ANTHROPIC_CUSTOM_HEADERS_ENV_KEY) if isinstance(base_env, dict) else None
        )
        # Copy the overlay per file so merging into one base cannot affect the other.
        overlay_for_merge = copy.deepcopy(overlay)
        if enforce_model_default_hierarchy:
            settings_file_env = base_env if isinstance(base_env, dict) else {}
            target_env = overlay_for_merge["env"]
            configured_defaults = coding_agent_config_defaults or {}
            settings_file_existing_defaults = {
                family: model
                for family, key in CLAUDE_DEFAULT_MODEL_ENV_KEYS.items()
                if isinstance((model := settings_file_env.get(key)), str)
            }
            managed_overlay = state.get(MANAGED_OVERLAY_KEY, {})
            ucode_defaults = (
                managed_overlay.get("claude_models") or state.get("claude_models") or {}
            )

            for family, key in CLAUDE_DEFAULT_MODEL_ENV_KEYS.items():
                if family == "fable" and not state.get("fable_enabled"):
                    target_env.pop(key, None)
                    continue

                selected_default_model = _enforce_model_default_hierarchy(
                    family,
                    coding_agent_config_defaults=configured_defaults,
                    settings_file_existing_defaults=settings_file_existing_defaults,
                    ucode_defaults=ucode_defaults,
                )
                if selected_default_model is None:
                    target_env.pop(key, None)
                else:
                    target_env[key] = selected_default_model
        merged = deep_merge_dict(base, overlay_for_merge)
        overlay_custom_headers = overlay_for_merge["env"][ANTHROPIC_CUSTOM_HEADERS_ENV_KEY]
        merged["env"][ANTHROPIC_CUSTOM_HEADERS_ENV_KEY] = _merge_anthropic_custom_headers(
            existing_custom_headers, overlay_custom_headers
        )
        # Drop any apiKeyHelper a prior non-relayed launch left in the file; relayed
        # must not carry one (it would outrank the subscription OAuth).
        if relayed:
            merged.pop("apiKeyHelper", None)
        if tracing_env_vars and stop_hook_command:
            _upsert_tracing_stop_hook(merged, stop_hook_command)
        if not tracing_env_vars:
            env_block = merged.get("env")
            if isinstance(env_block, dict):
                for key in CLAUDE_TRACING_ENV_KEYS:
                    env_block.pop(key, None)
            # Strip only ucode's tracing Stop hook so user hooks stay intact.
            _remove_tracing_stop_hook(merged)
        # Prune ucode-managed model env keys we deliberately don't write this run
        # (e.g. ANTHROPIC_MODEL — see render_overlay).
        overlay_env = overlay_for_merge.get("env", {})
        merged_env = merged.get("env")
        if isinstance(merged_env, dict):
            for key in CLAUDE_MANAGED_MODEL_ENV_KEYS:
                if key not in overlay_env:
                    merged_env.pop(key, None)
            for key in CLAUDE_CONDITIONAL_ENV_KEYS:
                if key not in overlay_env:
                    merged_env.pop(key, None)
            # deep_merge_dict keeps keys already in the file, so drop the ones ucode no
            # longer writes.
            for key in CLAUDE_REMOVED_ENV_KEYS:
                merged_env.pop(key, None)
        sync_smart_routing_hooks(merged, state, enabled=False)
        return merged

    write_json_file(
        CLAUDE_SETTINGS_PATH,
        _compose(read_json_safe(CLAUDE_SETTINGS_PATH), enforce_model_default_hierarchy=False),
    )

    _reconcile_managed_settings(
        state,
        lambda base: _compose(base, enforce_model_default_hierarchy=True),
        managed_file_keys,
        relayed,
    )

    if web_search_model:
        web_search_entry = _web_search_mcp_entry(
            state["workspace"], web_search_model, state.get("profile")
        )
        if not _web_search_mcp_is_current(state, web_search_entry):
            # Registration runs multiple `claude mcp` subprocesses and can take several seconds.
            registration_success = _register_web_search_mcp(
                state["workspace"], web_search_model, state.get("profile")
            )
            if registration_success:
                state[WEB_SEARCH_MCP_STATE_KEY] = web_search_entry
    else:
        state.pop(WEB_SEARCH_MCP_STATE_KEY, None)

    # Persist relayed mode + proxy port so launch() wires the refresh proxy and
    # subscription login; cleared on a non-relayed launch.
    if relayed:
        state["claude_relayed"] = True
    else:
        state.pop("claude_relayed", None)
        state.pop("relayed_proxy_port", None)
    state = mark_tool_managed(state, "claude", managed_keys)
    save_state(state)
    return state


def _merge_anthropic_custom_headers(existing: object, ucode_headers: str) -> str:
    """Preserve user headers while replacing the header names managed by ucode.

    Claude's ``ANTHROPIC_CUSTOM_HEADERS`` value is a newline-delimited string. To merge it, we:

    1. Split the existing custom headers by newline into individual header items.
    2. Split each item on ``:`` to identify its header name.
    3. Replace headers in ``CLAUDE_MANAGED_CUSTOM_HEADER_NAMES`` with ucode's values in their
       existing positions, while preserving all other existing headers.
    4. Append any ucode-managed headers that were not already present.

    Header names are compared case-insensitively. Non-header lines are also preserved to avoid
    silently discarding user configuration we do not understand.
    """

    if not isinstance(existing, str) or not existing:
        return ucode_headers

    ucode_lines_by_name: dict[str, str] = {}
    ucode_header_names: list[str] = []
    for line in ucode_headers.splitlines():
        name, separator, _value = line.partition(":")
        normalized_name = name.strip().casefold()
        if separator and normalized_name not in ucode_lines_by_name:
            ucode_header_names.append(normalized_name)
        if separator:
            ucode_lines_by_name[normalized_name] = line

    merged: list[str] = []
    replaced_names: set[str] = set()
    for line in existing.splitlines():
        name, separator, _value = line.partition(":")
        normalized_name = name.strip().casefold()
        if separator and normalized_name in CLAUDE_MANAGED_CUSTOM_HEADER_NAMES:
            replacement = ucode_lines_by_name.get(normalized_name)
            if replacement is not None and normalized_name not in replaced_names:
                merged.append(replacement)
                replaced_names.add(normalized_name)
            continue
        if line:
            merged.append(line)

    for name in ucode_header_names:
        if name not in replaced_names:
            merged.append(ucode_lines_by_name[name])
    return "\n".join(merged)


def _reconcile_managed_settings(
    state: dict,
    compose: Callable[[dict], dict],
    owned_paths: list[list[str]],
    relayed: bool,
) -> None:
    """Reconcile Claude Code's OS-managed settings so a bare ``claude`` uses the gateway.

    The managed file is root-owned and the highest-precedence scope, so every normal Claude
    configuration mirrors ucode's settings there. The same compose operation that produced the
    private file is applied to the existing managed file, preserving unrelated IT-authored keys.

    `ug configure` updates gateway-owned fields in this file, but does not generate or modify
    the `modelPicker` object; an existing picker is retained by the merge.

    Relayed launches are skipped: they depend on a per-session loopback refresh proxy that only runs
    during `ucode claude`, so a bare `claude` could not reach the gateway anyway.
    """
    path = _managed_settings_path()
    if path is None:
        print_warning(
            "Machine-wide Claude settings aren't supported on this platform; skipped the managed "
            "settings."
        )
        return
    if path.is_symlink():
        raise RuntimeError(
            f"Refusing to use Claude Code managed settings through symlink {path}. Replace it "
            "with a regular file or contact your administrator."
        )
    if relayed:
        conflicts = _managed_relayed_conflicts(path)
        if conflicts:
            raise RuntimeError(
                "Claude subscription relay cannot start because enterprise managed settings "
                f"define {', '.join(conflicts)} at {path}. Ask your administrator to remove "
                "those entries or use standard Databricks authentication. If ucode previously "
                "created them, run `ucode revert` from an interactive terminal first."
            )
        mark_managed_file_verified(state, "claude", path, scope="relay-compatible")
        return

    current_text = read_managed_file(path)
    try:
        existing = _parse_managed_settings(current_text) if current_text is not None else {}
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot safely update Claude Code managed settings at {path}: {exc}. "
            "ucode did not modify the file. Repair it or contact your administrator."
        ) from exc
    managed_before = copy.deepcopy(existing)
    desired_settings = compose(existing)
    _preserve_permission_denies(managed_before, desired_settings)
    if not managed_writes_allowed():
        conflicts = managed_file_conflicts(managed_before, desired_settings, owned_paths)
        if conflicts:
            raise RuntimeError(
                "Claude Code configuration cannot be applied non-interactively because "
                f"OS-managed settings at {path} override ucode values: {', '.join(conflicts)}. "
                "Run `ucode configure --agent claude` from an interactive terminal or contact "
                "your administrator."
            )
        mark_managed_file_verified(state, "claude", path, scope="local-compatible")
        return
    reconcile_managed_file(
        path,
        _dump_managed_settings(desired_settings),
        tool="claude",
        display="Claude Code",
        owned_paths=owned_paths,
    )
    mark_managed_file_verified(state, "claude", path)


def _preserve_permission_denies(existing: dict, desired: dict) -> None:
    existing_permissions = existing.get("permissions")
    desired_permissions = desired.get("permissions")
    if not isinstance(existing_permissions, dict) or not isinstance(desired_permissions, dict):
        return
    existing_denies = existing_permissions.get("deny")
    desired_denies = desired_permissions.get("deny")
    if not isinstance(existing_denies, list) or not isinstance(desired_denies, list):
        return
    desired_permissions["deny"] = [
        *existing_denies,
        *(rule for rule in desired_denies if rule not in existing_denies),
    ]


def _is_tracing_stop_hook(hook: object) -> bool:
    if not isinstance(hook, dict):
        return False
    hook = cast(dict, hook)
    if hook.get("type") != "command":
        return False
    command = hook.get("command")
    return isinstance(command, str) and command.endswith(CLAUDE_TRACING_STOP_HOOK_SUFFIX)


def _remove_tracing_stop_hook(settings: dict) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        return

    cleaned_entries = []
    for entry in stop_entries:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue
        hook_list = entry.get("hooks")
        if not isinstance(hook_list, list):
            cleaned_entries.append(entry)
            continue
        cleaned_hooks = [hook for hook in hook_list if not _is_tracing_stop_hook(hook)]
        if cleaned_hooks:
            cleaned_entry = dict(entry)
            cleaned_entry["hooks"] = cleaned_hooks
            cleaned_entries.append(cleaned_entry)

    if cleaned_entries:
        hooks["Stop"] = cleaned_entries
    else:
        hooks.pop("Stop", None)
    if not hooks:
        settings.pop("hooks", None)


def _upsert_tracing_stop_hook(settings: dict, command: str) -> None:
    _remove_tracing_stop_hook(settings)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        stop_entries = []
        hooks["Stop"] = stop_entries
    stop_entries.append({"hooks": [{"type": "command", "command": command}]})


def ensure_tracing_runtime() -> bool:
    """Ensure the MLflow tracing runtime is ready: a pinned `mlflow` CLI (3.11.x)
    installed via `uv tool`, whose absolute path the Stop hook will call.

    Best-effort — warns and returns False if it can't be set up, so
    `ucode configure tracing` can still finish for other agents."""
    return _ensure_mlflow_cli()


def _parse_mlflow_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _uv_tool_mlflow_path() -> str | None:
    """Absolute path to the `mlflow` installed by `uv tool`, or None.

    Resolved from `uv tool dir --bin` rather than ``shutil.which`` so a project
    venv's (possibly wrong-versioned) mlflow can't shadow the one ucode pins —
    the Stop hook must always run the uv-tool copy."""
    if not shutil.which("uv"):
        return None
    try:
        result = subprocess.run(
            ["uv", "tool", "dir", "--bin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    bin_dir = (result.stdout or "").strip()
    if result.returncode != 0 or not bin_dir:
        return None
    candidate = Path(bin_dir) / "mlflow"
    return str(candidate) if candidate.exists() else None


def _installed_mlflow_version() -> tuple[int, int] | None:
    """The (major, minor) of the uv-tool `mlflow`, or None if absent."""
    path = _uv_tool_mlflow_path()
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_mlflow_version(result.stdout or result.stderr or "")


def claude_tracing_stop_hook_command() -> str | None:
    """The Stop hook command string: the absolute uv-tool `mlflow` invoking its
    `autolog claude stop-hook` handler. None when mlflow isn't installed.

    Using the absolute path means the hook needs neither `uv` nor a package
    index at run time (the minimal env Claude runs hooks in lacks UV_INDEX_URL),
    and can't be shadowed by another mlflow on PATH."""
    path = _uv_tool_mlflow_path()
    if not path:
        return None
    return f"{path} autolog claude stop-hook"


def _ensure_mlflow_cli() -> bool:
    """Ensure the pinned `mlflow` CLI (3.11.x) is installed via `uv tool`,
    installing or replacing an out-of-range version when needed."""
    current = _installed_mlflow_version()
    if current and MINIMUM_MLFLOW_VERSION <= current < MAXIMUM_MLFLOW_VERSION:
        return True

    if not shutil.which("uv"):
        verb = "replace" if current else "install"
        print_warning(
            f"Claude tracing needs the `mlflow` CLI ({MLFLOW_CLI_SPEC}), but `uv` is not "
            f'available to {verb} it. Run `uv tool install "{MLFLOW_CLI_SPEC}"`, then '
            "re-run `ucode configure tracing`."
        )
        return False

    print_note(f"{'Replacing' if current else 'Installing'} the mlflow CLI ({MLFLOW_CLI_SPEC})...")
    # Always --force: it installs fresh when absent and replaces in place when
    # present. Keying it on `current` broke when an mlflow existed but its
    # version couldn't be parsed — uv still errors "Executable already exists".
    cmd = ["uv", "tool", "install", "--force", MLFLOW_CLI_SPEC]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print_warning(f"Could not install the mlflow CLI automatically: {exc}")
        return False

    if not _uv_tool_mlflow_path():
        print_warning(
            "Installed mlflow via `uv tool`, but its binary could not be located. "
            "Re-run `ucode configure tracing`."
        )
        return False
    print_success("mlflow CLI ready")
    return True


def default_model(state: dict) -> str | None:
    claude_models = state.get("claude_models") or {}
    return claude_models.get("opus") or claude_models.get("sonnet") or claude_models.get("haiku")


def _extract_caller_settings(tool_args: list[str]) -> tuple[list[str], list[str]]:
    """Split caller-supplied ``--settings`` values out of *tool_args*.

    Returns ``(values, remaining_args)``, handling both ``--settings <value>``
    and ``--settings=<value>`` spellings. Each value is either a JSON string or
    a path to a settings file — Claude Code accepts either.
    """
    values: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(tool_args):
        arg = tool_args[i]
        if arg == "--settings" and i + 1 < len(tool_args):
            values.append(tool_args[i + 1])
            i += 2
            continue
        if arg.startswith("--settings="):
            values.append(arg[len("--settings=") :])
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return values, remaining


def _load_caller_settings(value: str) -> dict:
    """Resolve a ``--settings`` value (inline JSON or file path) to a dict.

    Claude Code accepts either inline JSON or a path to a JSON file. Raises
    ``RuntimeError`` (surfaced by the CLI as an actionable error) when the value
    is neither, rather than silently dropping it: a dropped value would also be
    passed through as a second ``--settings`` flag, and Claude Code honors only
    one — so either the caller's settings or ucode's gateway config would be
    silently ignored. Failing loudly lets the caller fix their input.
    """
    text = value.strip()
    if text.startswith("{"):
        source, malformed = text, "value is not valid JSON"
    else:
        path = Path(text)
        if not path.exists():
            raise RuntimeError(
                f"--settings file not found: {value!r}. "
                "Pass inline JSON or a path to an existing JSON file."
            )
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"--settings file could not be read: {value!r} ({exc}).") from exc
        malformed = "file is not valid JSON"
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"--settings {malformed} ({exc}): {value!r}. Pass inline JSON or a path to a JSON file."
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"--settings must be a JSON object, got {type(parsed).__name__}: {value!r}."
        )
    return parsed


def _union_claude_hooks(base: dict, overlay: dict) -> dict:
    """Union two Claude Code ``hooks`` maps.

    ``hooks`` is ``{event: [entry, ...]}``. Per event we concatenate the entry
    lists so hooks from BOTH settings sources fire, rather than one replacing
    the other (which is what a plain deep-merge does to lists). This is what
    lets ucode's tracing Stop hook and a caller's own hooks coexist.
    """
    result: dict = {}
    for event in [*base, *(e for e in overlay if e not in base)]:
        entries: list = []
        for src in (base, overlay):
            val = src.get(event)
            if isinstance(val, list):
                entries.extend(val)
        result[event] = entries
    return result


def _merge_claude_settings(base: dict, overlay: dict) -> dict:
    """Deep-merge *overlay* onto *base* (overlay wins on conflicting leaves),
    but UNION the ``hooks`` so neither side's hooks are dropped. Inputs are not
    mutated.
    """
    merged = deep_merge_dict(copy.deepcopy(base), overlay)
    base_hooks = base.get("hooks")
    overlay_hooks = overlay.get("hooks")
    if isinstance(base_hooks, dict) or isinstance(overlay_hooks, dict):
        merged["hooks"] = _union_claude_hooks(
            base_hooks if isinstance(base_hooks, dict) else {},
            overlay_hooks if isinstance(overlay_hooks, dict) else {},
        )
    return merged


def _compose_v2_settings(tool_args: list[str]) -> tuple[dict, list[str]]:
    """Compose caller settings with ucode's Claude settings for a v2 launch."""
    caller_values, remaining = _extract_caller_settings(tool_args)
    settings: dict = {}
    for value in caller_values:
        settings = _merge_claude_settings(settings, _load_caller_settings(value))
    return _merge_claude_settings(settings, read_json_safe(CLAUDE_SETTINGS_PATH)), remaining


def _original_launch_model(state: dict) -> str | None:
    override = state.get("_claude_launch_model")
    if isinstance(override, str) and override.strip():
        return override.strip()
    value = read_json_safe(CLAUDE_USER_SETTINGS_PATH).get("model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_model(state)


def _has_provider_launch(state: dict) -> bool:
    transient = state.get("_claude_launch_provider")
    return (isinstance(transient, str) and bool(transient.strip())) or bool(
        get_provider_service(state, "claude")
    )


def _launch_model_args(tool_args: list[str], launch_model: str | None) -> list[str]:
    if not launch_model or has_explicit_model_arg(tool_args):
        return []
    return ["--model", launch_model]


def _build_claude_argv(
    binary: str,
    tool_args: list[str],
    relayed: bool = False,
    settings_override: dict | None = None,
) -> list[str]:
    """Build the ``claude`` argv, composing any caller ``--settings`` with
    ucode's managed settings.

    ucode needs its own settings (gateway ``apiKeyHelper`` + env) to reach
    Claude, and normally passes ``--settings <ucode-file>``. But Claude Code
    honors only ONE ``--settings`` flag, so a caller that ALSO passes
    ``--settings`` (e.g. an integration injecting hooks) would have exactly one
    of the two silently dropped. To let ucode compose with any prior command,
    we merge a caller-supplied ``--settings`` with ucode's — ucode's gateway
    keys win, hooks from both are unioned — and hand Claude a single merged
    ``--settings`` (inline JSON). The merge is per-launch and is never written
    back to the shared ucode settings file, so concurrent launches cannot
    accumulate one another's hooks. A caller ``--settings`` value ucode cannot
    resolve raises (see :func:`_load_caller_settings`) rather than being passed
    through as a second, colliding flag.

    ``relayed`` adds ``--setting-sources`` to exclude the user scope (see
    :data:`_RELAYED_SETTING_SOURCES`), so a stale user-scope apiKeyHelper cannot
    filter through and shadow the subscription OAuth.
    """
    source_args = ["--setting-sources", _RELAYED_SETTING_SOURCES] if relayed else []
    caller_values, remaining = _extract_caller_settings(tool_args)
    if not caller_values and settings_override is None:
        # No caller --settings: hand Claude ucode's settings file directly (the
        # common path; behavior unchanged).
        return [binary, *source_args, "--settings", str(CLAUDE_SETTINGS_PATH), *tool_args]
    caller_settings: dict = {}
    for value in caller_values:
        caller_settings = _merge_claude_settings(caller_settings, _load_caller_settings(value))
    # ucode wins over the caller for conflicting keys (protects gateway auth);
    # hooks from both sides survive.
    merged = _merge_claude_settings(caller_settings, read_json_safe(CLAUDE_SETTINGS_PATH))
    if settings_override is not None:
        merged = _merge_claude_settings(merged, settings_override)
    merged_env = merged.get("env")
    if isinstance(merged_env, dict):
        merged_env.pop("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", None)
    return [
        binary,
        *source_args,
        "--settings",
        json.dumps(merged, separators=(",", ":")),
        *remaining,
    ]


def _has_subscription_login() -> bool:
    """True when Claude Code already holds a subscription login (`claude auth
    status` exits 0). Never inspects or captures the credential itself."""
    try:
        result = subprocess.run(
            [SPEC["binary"], "auth", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _ensure_subscription_login() -> None:
    """Ensure Claude Code has a persisted subscription login, running the browser
    flow via `claude auth login` if not. ucode never sees or stores the token —
    Claude Code persists it to its own secure store and refreshes it natively."""
    if _has_subscription_login():
        return
    print_note("Opening browser to sign in with your Claude subscription...")
    try:
        subprocess.run([SPEC["binary"], "auth", "login"], check=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`claude auth login` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`claude auth login` timed out.") from exc
    print_success("Claude subscription authenticated")


def _rewrite_relayed_port(state: dict, port: int) -> None:
    """Point the persisted config + state at ``port`` after the proxy had to bind
    a different port than the cached one. Keeps ANTHROPIC_BASE_URL (which Claude
    Code reads) in sync with the live proxy so requests reach it."""
    state["relayed_proxy_port"] = port
    save_state(state)
    settings = read_json_safe(CLAUDE_SETTINGS_PATH)
    env = settings.get("env")
    if isinstance(env, dict):
        env["ANTHROPIC_BASE_URL"] = f"http://{LOOPBACK_HOST}:{port}"
        write_json_file(CLAUDE_SETTINGS_PATH, settings)


def _launch_relayed(state: dict, binary: str, tool_args: list[str]) -> None:
    """Relayed launch: sign into the Claude subscription, start the loopback
    refresh proxy, then run Claude Code alongside it (the proxy must outlive the
    exec, so we spawn-and-wait rather than replacing the process)."""
    _ensure_subscription_login()
    workspace = state["workspace"]
    port = state.get("relayed_proxy_port")
    if not isinstance(port, int):
        raise RuntimeError("Relayed proxy port was not configured; re-run `ucode claude`.")

    server, cache, client = gateway_proxy.start_proxy(
        workspace,
        state.get("profile"),
        port,
        token_header=gateway_proxy.AI_GATEWAY_TOKEN_HEADER,
        force_refresh_near_expiry=False,
    )
    # start_proxy falls back to an OS-assigned port when the cached one is taken
    # (stale proxy from a killed session). Reconcile settings + state to whatever
    # it actually bound, so Claude Code connects to the live port.
    bound_port = server.server_address[1]
    if bound_port != port:
        _rewrite_relayed_port(state, bound_port)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    proc = subprocess.Popen(_build_claude_argv(binary, tool_args, relayed=True))
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        cache.stop()
        server.shutdown()
        client.close()
    raise SystemExit(returncode)


def launch(
    state: dict,
    tool_args: list[str],
    *,
    options: LaunchOptions,
) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if state.get("claude_relayed"):
        _launch_relayed(state, binary, tool_args)
        return
    # Smart routing v2 needs Unix PTY support, which Windows does not provide.
    if options.launch_smart_routing and os.name == "nt":
        raise RuntimeError(
            "Smart routing in Claude Code is currently not supported on Windows. "
            "Please use Codex or disable smart routing."
        )
    if options.launch_smart_routing:
        smart_routing_v2.launch_claude(
            state,
            tool_args,
            binary=binary,
            user_settings_path=CLAUDE_USER_SETTINGS_PATH,
            launch_model=_original_launch_model(state),
            compose_settings=_compose_v2_settings,
            launch_model_args=_launch_model_args,
            model_name=_maybe_add_1m_suffix,
        )
        return
    if (
        workspace
        and os.environ.get(GATEWAY_MODEL_DISCOVERY_ENV_VAR) == "1"
        and not _has_provider_launch(state)
    ):
        # Discovery is launch-scoped. Pass it in the process environment rather
        # than persisting it in Claude's private or OS-managed settings.
        os.environ["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    if workspace:
        os.environ["OAUTH_TOKEN"] = get_databricks_token(workspace, state.get("profile"))
    if options.claude_launch_model:
        os.environ["ANTHROPIC_MODEL"] = options.claude_launch_model
    exec_or_spawn(_build_claude_argv(binary, tool_args))


def validate_cmd(binary: str) -> list[str]:
    return [
        binary,
        "--settings",
        str(CLAUDE_SETTINGS_PATH),
        "-p",
        "say hi in 5 words or less",
        "--max-turns",
        "1",
    ]


def skip_validation(state: dict) -> bool:
    """Relayed configs can't be probed with a live message: the loopback proxy
    and subscription login are only established at launch, so a validation-time
    request has nothing listening and would hang (and burn subscription quota)."""
    return bool(state.get("claude_relayed"))
