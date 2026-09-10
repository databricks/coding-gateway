"""MCP (Model Context Protocol) server registration for coding tools."""

from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, IsDone
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.shortcuts import PromptSession
from questionary.prompts.common import InquirerControl
from questionary.question import Question
from questionary.styles import merge_styles_default

from ucode.agents import copilot, cursor, gemini, opencode
from ucode.config_io import restore_file
from ucode.databricks import (
    PermissionDeniedError,
    apply_pat_environment,
    build_mcp_proxy_argv,
    build_mcp_service_url,
    build_skills_mcp_url,
    ensure_databricks_auth,
    get_databricks_token,
    list_all_mcp_services,
    list_databricks_apps,
    list_mcp_services,
    workspace_hostname,
)
from ucode.mcp_oauth import (
    CLAUDE_CODE_OAUTH_CLIENT_ID,
    MCP_OAUTH_CALLBACK_PORT,
    oauth_client_available,
)
from ucode.state import load_full_state, load_state, save_state
from ucode.ui import (
    console,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    spinner,
)

MCP_USER_SCOPE = "user"
MCP_CLEANUP_SCOPES = ("local", "project", MCP_USER_SCOPE)
MCP_PICKER_VISIBLE_ROWS = 10

# AI Gateway MCP-services endpoints carry this path segment. These are the
# connection-backed services that need a per-user connection login.
AIGW_MCP_SERVICES_PATH = "/ai-gateway/mcp-services/"

# Per-agent published OAuth app used for the direct-HTTP MCP connection login.
# ONLY Claude Code's `mcp add` supports pinning an OAuth client (`--client-id`),
# so only it can drive the `/oidc` login for a connection-backed service and show
# `/mcp` "needs authentication". The other agents' `mcp add` accept only a static
# bearer, not an OAuth client — codex (`--bearer-token-env-var`), gemini
# (`--header`), cursor (config) — so despite their published apps (codex-cli,
# cursor-desktop, …) they can't drive this flow and stay on the stdio proxy. Add
# an agent here — with its own `add_<agent>_http_mcp_server` — once its CLI gains
# `--client-id` support.
AGENT_OAUTH_CLIENT = {"claude": CLAUDE_CODE_OAUTH_CLIENT_ID}


class _Back:
    """Sentinel type: a wizard step returns the `_BACK` instance when the user
    presses Left (←) to go back. Distinct from None (cancel) and [] (empty)."""


# Singleton instance used everywhere; compare with `is _BACK`.
_BACK = _Back()
MCP_CLIENTS = {
    "claude": {
        "binary": "claude",
        "display": "Claude Code",
        "list_command": "claude mcp list",
    },
    "codex": {
        "binary": "codex",
        "display": "Codex",
        "list_command": "codex mcp list",
    },
    "gemini": {
        "binary": "gemini",
        "display": "Gemini CLI",
        "list_command": "gemini mcp list",
    },
    "opencode": {
        "binary": "opencode",
        "display": "OpenCode",
        "list_command": "opencode mcp list",
    },
    "copilot": {
        "binary": "copilot",
        "display": "GitHub Copilot CLI",
        "list_command": "copilot mcp list",
    },
    "cursor": {
        "binary": "cursor-agent",
        "display": "Cursor",
        "list_command": "cursor-agent mcp list",
    },
}
SKILLS_MCP_KIND = "skills"
SKILLS_MCP_SERVER_NAME = "databricks-skill-registry"
# MCP-only clients ucode never launches for model routing, so they never land in
# `available_tools`; they're eligible for MCP config purely on being installed.
MCP_ONLY_CLIENTS = ("cursor",)
EXTERNAL_MCP_SELECTION_PREFIX = "external:"
SQL_MCP_VALUE = "managed:sql"
GENIE_SPACE_SELECTION_PREFIX = "genie-space:"
APP_MCP_SELECTION_PREFIX = "app:"
MCP_SERVICE_SELECTION_PREFIX = "mcp-service:"
VECTOR_SEARCH_SELECTION_PREFIX = "vector-search:"
UC_FUNCTIONS_SELECTION_PREFIX = "uc-functions:"
MCP_ADD_PREFIX = "add:"


def add_claude_mcp_server(
    name: str,
    server: list[str] | dict,
    scope: str = MCP_USER_SCOPE,
    *,
    always_load: bool = False,
) -> None:
    # Three registration shapes share this helper. The plain proxy path passes an
    # argv list (`ucode mcp-proxy ...`), registered via `claude mcp add ... -- <argv>`
    # where `--` fences the proxy's own flags off from claude's parser. The
    # web_search server (agents/claude.py) passes a full stdio entry dict with its
    # own env, which only `add-json` can express — so a dict routes there. Finally,
    # `always_load` (the skills registry) needs `alwaysLoad: true`, which plain
    # `mcp add` can't set, so build a stdio entry dict and route it to add-json too.
    if isinstance(server, dict):
        cmd = ["claude", "mcp", "add-json", name, json.dumps(server), "-s", scope]
    elif always_load:
        entry = {
            "type": "stdio",
            "command": server[0],
            "args": list(server[1:]),
            "alwaysLoad": True,
        }
        cmd = ["claude", "mcp", "add-json", name, json.dumps(entry), "-s", scope]
    else:
        cmd = ["claude", "mcp", "add", name, "-s", scope, "--", *server]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via claude CLI.") from exc


def add_claude_http_mcp_server(
    name: str,
    url: str,
    scope: str = MCP_USER_SCOPE,
    *,
    client_id: str = CLAUDE_CODE_OAUTH_CLIENT_ID,
    callback_port: int = MCP_OAUTH_CALLBACK_PORT,
) -> None:
    """Register a Databricks MCP endpoint as a **direct HTTP** server so Claude
    Code is the OAuth client and drives the RFC 8707 connection login itself.

    Unlike the stdio proxy (which injects a plain workspace token and hides the
    per-user connection state), a direct HTTP server lets Claude Code do MCP OAuth
    against ``/oidc`` with the ``resource`` indicator: on a missing/expired
    connection credential, ``/mcp`` shows "needs authentication" and Authenticate
    runs the login (``/oidc`` -> ``/mcp-service-login``). ``client_id`` is the
    published ``claude-code`` app (it has the loopback ``/callback`` redirect
    registered); the callback port is arbitrary because ``/oidc`` ignores the port
    for loopback redirects."""
    cmd = [
        "claude",
        "mcp",
        "add",
        "--transport",
        "http",
        "-s",
        scope,
        "--client-id",
        client_id,
        "--callback-port",
        str(callback_port),
        name,
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add HTTP MCP server '{name}' via claude CLI.") from exc


def _is_missing_mcp_server_output(output: str) -> bool:
    normalized = output.lower()
    return (
        "not found" in normalized
        or "no mcp server" in normalized
        or "no server named" in normalized
        or ("mcp server found with name" in normalized and "no " in normalized)
    )


def remove_claude_mcp_server(name: str, scope: str) -> bool:
    try:
        subprocess.run(
            ["claude", "mcp", "remove", name, "-s", scope],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True
    except subprocess.CalledProcessError as exc:
        output = f"{exc.stderr or ''}\n{exc.stdout or ''}"
        if _is_missing_mcp_server_output(output):
            return False
        raise RuntimeError(f"Failed to remove MCP server '{name}' via claude CLI.") from exc


def add_codex_mcp_server(name: str, argv: list[str]) -> None:
    # `--` fences the proxy argv off from codex's own flag parser, registering
    # it as a stdio server (codex spawns the command and speaks MCP over it).
    try:
        subprocess.run(
            ["codex", "mcp", "add", name, "--", *argv],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via codex CLI.") from exc


def remove_codex_mcp_server(name: str) -> bool:
    try:
        result = subprocess.run(
            ["codex", "mcp", "remove", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out removing MCP server '{name}' via codex CLI.") from exc

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _is_missing_mcp_server_output(output):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Failed to remove MCP server '{name}' via codex CLI.")
    return True


def _gemini_cli_env() -> dict[str, str]:
    # Pin GEMINI_CLI_HOME to the same directory the launcher.
    env = os.environ.copy()
    env["GEMINI_CLI_HOME"] = str(gemini.GEMINI_HOME_DIR)
    return env


def add_gemini_mcp_server(name: str, argv: list[str]) -> None:
    # Register the proxy as a stdio server: `gemini mcp add <name> <cmd> <args…>
    # --type stdio`. The scope/type flags trail the captured command + args.
    try:
        subprocess.run(
            [
                "gemini",
                "mcp",
                "add",
                name,
                *argv,
                "--type",
                "stdio",
                "--scope",
                MCP_USER_SCOPE,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_gemini_cli_env(),
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via gemini CLI.") from exc


def remove_gemini_mcp_server(name: str) -> bool:
    try:
        result = subprocess.run(
            ["gemini", "mcp", "remove", name, "--scope", MCP_USER_SCOPE],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=_gemini_cli_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out removing MCP server '{name}' via gemini CLI.") from exc

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _is_missing_mcp_server_output(output):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Failed to remove MCP server '{name}' via gemini CLI.")
    return True


def available_mcp_clients() -> list[str]:
    return [client for client, spec in MCP_CLIENTS.items() if shutil.which(str(spec["binary"]))]


def configured_mcp_clients(state: dict, installed_clients: list[str]) -> list[str]:
    configured_tools = state.get("available_tools") or []
    if not isinstance(configured_tools, list):
        configured_tools = []
    configured = set(configured_tools)
    return [
        client
        for client in MCP_CLIENTS
        if client in installed_clients and (client in configured or client in MCP_ONLY_CLIENTS)
    ]


def configure_client_mcp_server(
    client: str,
    name: str,
    url: str,
    workspace: str,
    profile: str | None = None,
    *,
    use_pat: bool = False,
    always_load: bool = False,
) -> list[str]:
    # Connection-backed AI Gateway MCP services register as a direct HTTP server so
    # the agent drives the connection login natively — but only for an agent whose
    # CLI can pin an OAuth client (AGENT_OAUTH_CLIENT: today just Claude) and only
    # when that client is registered on the workspace. Everything else keeps the
    # stdio proxy: non-connection MCPs, the skills registry (`always_load`), PAT
    # auth (no interactive OAuth), agents without a mapped OAuth client, and
    # workspaces where the mapped client isn't published.
    oauth_client = AGENT_OAUTH_CLIENT.get(client)
    if (
        oauth_client is not None
        and AIGW_MCP_SERVICES_PATH in url
        and not always_load
        and not use_pat
        and oauth_client_available(workspace, oauth_client)
    ):
        removed_scopes = [
            scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)
        ]
        add_claude_http_mcp_server(name, url, client_id=oauth_client)
        return removed_scopes

    # Every other case registers the `ucode mcp-proxy ...` stdio command; the proxy
    # forwards to `url` and refreshes the Databricks token itself. Only the
    # per-client registration syntax differs. `always_load` (skills registry) is
    # a Claude-only hint to load the server's tools at session start; other
    # clients don't support it and ignore it.
    argv = build_mcp_proxy_argv(url, workspace, profile, use_pat=use_pat)
    if client == "claude":
        removed_scopes = [
            scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)
        ]
        add_claude_mcp_server(name, argv, MCP_USER_SCOPE, always_load=always_load)
        return removed_scopes
    if client == "codex":
        removed = remove_codex_mcp_server(name)
        add_codex_mcp_server(name, argv)
        return [MCP_USER_SCOPE] if removed else []
    if client == "gemini":
        removed = remove_gemini_mcp_server(name)
        add_gemini_mcp_server(name, argv)
        return [MCP_USER_SCOPE] if removed else []
    if client == "opencode":
        removed = opencode.write_mcp_server_config(name, argv)
        return [MCP_USER_SCOPE] if removed else []
    if client == "copilot":
        removed = copilot.write_mcp_server_config(name, argv)
        return [MCP_USER_SCOPE] if removed else []
    if client == "cursor":
        removed = cursor.write_mcp_server_config(name, argv)
        return [MCP_USER_SCOPE] if removed else []
    raise RuntimeError(f"Unsupported MCP client '{client}'.")


def remove_client_mcp_server(client: str, name: str) -> list[str]:
    if client == "claude":
        return [scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)]
    if client == "codex":
        return [MCP_USER_SCOPE] if remove_codex_mcp_server(name) else []
    if client == "gemini":
        return [MCP_USER_SCOPE] if remove_gemini_mcp_server(name) else []
    if client == "opencode":
        return [MCP_USER_SCOPE] if opencode.remove_mcp_server_config(name) else []
    if client == "copilot":
        return [MCP_USER_SCOPE] if copilot.remove_mcp_server_config(name) else []
    if client == "cursor":
        return [MCP_USER_SCOPE] if cursor.remove_mcp_server_config(name) else []
    raise RuntimeError(f"Unsupported MCP client '{client}'.")


def revert_mcp_configs(state: dict) -> dict[str, bool]:
    results: dict[str, bool] = {}
    # Both the developer's own servers and any registered from the workspace's managed config, so a
    # revert leaves no ucode-added MCP server behind in an agent's config.
    all_servers = list(state.get("mcp_servers") or []) + list(
        state.get("managed_mcp_servers") or []
    )
    for server in all_servers:
        name = server.get("name")
        if not isinstance(name, str) or not name:
            continue
        for client in server.get("clients") or []:
            if client not in MCP_CLIENTS:
                continue
            removed_scopes = remove_client_mcp_server(client, name)
            results[client] = bool(removed_scopes) or results.get(client, False)

    # OpenCode MCP entries live in the normal OpenCode config and are restored
    # by the main agent config revert. Copilot stores MCP servers separately,
    # so restore its original MCP file after removing per-server entries above.
    results["copilot"] = restore_file(
        copilot.COPILOT_MCP_CONFIG_PATH,
        copilot.COPILOT_MCP_BACKUP_PATH,
        any(
            "copilot" in (server.get("clients") or []) for server in state.get("mcp_servers") or []
        ),
    ) or results.get("copilot", False)
    return results


def discover_mcp_service_names(workspace: str, profile: str | None = None) -> list[str]:
    """Curated `system.ai.*` MCP services. Empty list if discovery fails so
    callers can fall back to legacy connection discovery without surfacing
    every error to the picker."""
    token = get_databricks_token(workspace, profile)
    names, _reason = list_mcp_services(workspace, token)
    return names


def discover_all_mcp_service_names(
    workspace: str,
    profile: str | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_services: Callable[[list[str]], None] | None = None,
) -> list[str]:
    """All MCP services across every `<catalog>.<schema>` in the workspace. This
    walks the workspace (see `list_all_mcp_services`) and is the workspace-wide
    counterpart to `discover_mcp_service_names`. `on_progress` is forwarded to
    the walk for live count reporting, and `on_services` to stream newly-found
    service names into the picker as the walk progresses."""
    token = get_databricks_token(workspace, profile)
    names, _reason = list_all_mcp_services(
        workspace, token, on_progress=on_progress, on_services=on_services
    )
    return names


def _normalize_workspace_title(text: str) -> str:
    """Collapse a Databricks workspace title to lowercase alphanumerics joined
    by single hyphens, trimmed at the edges. Output is safe to use as an MCP
    server-name token across every supported agent CLI."""
    chars: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")


def app_mcp_servers(apps: list[dict]) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for app in apps:
        app_name = app.get("name")
        app_url = app.get("url")
        if not isinstance(app_name, str) or not app_name.strip():
            continue
        if not app_name.strip().startswith("mcp-"):
            continue
        if not isinstance(app_url, str) or not app_url.strip():
            continue
        name = app_name.strip()
        server_name = f"databricks-app-{name}"
        if server_name in seen_names:
            continue
        seen_names.add(server_name)
        servers.append(
            {
                "name": server_name,
                "title": name,
                "url": f"{app_url.strip().rstrip('/')}/mcp",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_app_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    return app_mcp_servers(list_databricks_apps(workspace, profile))


def _catalog_schema_server_name(prefix: str, catalog: str, schema: str, taken: set[str]) -> str:
    """Stable server name for a per-(catalog, schema) managed MCP entry.

    Prefers the lowercase alphanumeric slug; falls back to a numeric suffix on
    collision so two schemas that slug to the same value still both render."""
    slug = f"{_normalize_workspace_title(catalog)}-{_normalize_workspace_title(schema)}".strip("-")
    candidate = f"{prefix}-{slug}" if slug else prefix
    if candidate not in taken:
        return candidate
    counter = 2
    while f"{candidate}-{counter}" in taken:
        counter += 1
    return f"{candidate}-{counter}"


def _picker_style() -> questionary.Style:
    return questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )


def _server_name(server: dict) -> str | None:
    name = server.get("name")
    return name if isinstance(name, str) and name else None


def _servers_by_name(mcp_servers: list[dict]) -> dict[str, dict]:
    servers: dict[str, dict] = {}
    for server in mcp_servers:
        name = _server_name(server)
        if name:
            servers[name] = server
    return servers


def _mcp_entry_url_host(entry: dict) -> str | None:
    """Return the host of an MCP entry's URL, or ``None`` if missing/malformed."""
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        return None
    try:
        return urlparse(url).hostname
    except ValueError:
        return None


def _partition_mcp_entries_by_workspace(
    entries: list[dict], workspace: str
) -> tuple[list[dict], list[dict]]:
    """Split MCP entries into ones that belong to ``workspace`` and ones that don't."""
    workspace_host = workspace_hostname(workspace)
    current: list[dict] = []
    foreign: list[dict] = []
    for entry in entries:
        if _mcp_entry_url_host(entry) == workspace_host:
            current.append(entry)
        else:
            foreign.append(entry)
    return current, foreign


def _mcp_entries_only_in_other_workspaces(current_workspace: str) -> dict[str, set[str]]:
    """Return ``{name: {client, ...}}`` for MCPs ucode tracks only in workspaces other than ``current_workspace``."""
    full_state = load_full_state()
    workspaces = full_state.get("workspaces")
    if not isinstance(workspaces, dict):
        return {}

    current_names: set[str] = set()
    current_bucket = workspaces.get(current_workspace)
    if isinstance(current_bucket, dict):
        for entry in current_bucket.get("mcp_servers") or []:
            name = _server_name(entry)
            if name:
                current_names.add(name)

    external_entries: dict[str, set[str]] = {}
    for ws, bucket in workspaces.items():
        if ws == current_workspace or not isinstance(bucket, dict):
            continue
        for entry in bucket.get("mcp_servers") or []:
            name = _server_name(entry)
            if not name or name in current_names:
                continue
            client_set = external_entries.setdefault(name, set())
            for client in entry.get("clients") or []:
                client_set.add(client)
    return external_entries


def _server_choice(name: str, checked: bool, title: str | None = None) -> questionary.Choice:
    return questionary.Choice(
        title=title or name,
        value=name,
        checked=checked,
    )


def _add_choice(selection: str, title: str) -> questionary.Choice:
    return questionary.Choice(title=title, value=f"{MCP_ADD_PREFIX}{selection}")


def _mcp_service_choice(name: str, known_names: set[str], additive: bool) -> questionary.Choice:
    """Picker choice for one MCP-service full name (`<catalog>.<schema>.<id>`).

    Shared by the initial `build_mcp_picker_choices` render and the background walk that
    streams more services in, so a streamed row is built identically to an up-front one
    (and dedupes by value against what's already shown). An already-registered service is
    a removable toggle under `configure mcp` and a non-toggleable note under `mcp add`
    (additive); an unregistered one is an add-choice."""
    registered_as = name.replace(".", "-")
    display_title = f"MCP: {name}"
    if registered_as in known_names:
        if additive:
            return questionary.Choice(
                title=display_title, value=registered_as, disabled="already configured"
            )
        return _server_choice(registered_as, True, display_title)
    return _add_choice(f"{MCP_SERVICE_SELECTION_PREFIX}{name}", display_title)


class _StreamingInquirerControl(InquirerControl):
    """`InquirerControl` that tolerates an empty or all-disabled choice list.

    Stock `InquirerControl.__init__` ends with ``if not self.is_selection_valid(): raise`` and
    `is_selection_valid` dereferences `pointed_at`, which `_init_choices` leaves unset when no row
    is selectable — so constructing it with an empty (or every-row-disabled) list raises, and
    navigation later hits the same unset cursor. Our picker intentionally opens on an empty list
    and fills it in via the background loader, and `ug mcp add` can legitimately show only
    already-configured (disabled) rows. Default the cursor and treat "nothing selectable" as valid
    so construction, rendering, and navigation don't crash."""

    def is_selection_valid(self) -> bool:
        if getattr(self, "pointed_at", None) is None:
            self.pointed_at = 0
        selectable = any(
            not isinstance(c, questionary.Separator) and not c.disabled for c in self.choices
        )
        if not selectable:
            # Empty, or every row a separator/disabled: nothing to validate, and nothing for the
            # navigation skip-loop to land on — report valid so we neither raise nor spin.
            return True
        if self.pointed_at >= len(self.choices):
            return False
        return super().is_selection_valid()

    def _get_choice_tokens(self):
        # Stock rendering unconditionally reads `filtered_choices[pointed_at]`, which raises on an
        # empty list. Render nothing when there are no rows (the picker is still streaming them in).
        if not self.filtered_choices:
            return []
        return super()._get_choice_tokens()


def _merge_new_choices(
    existing: list[questionary.Choice | questionary.Separator],
    new_choices: list[questionary.Choice],
) -> list[questionary.Choice]:
    """Return the choices from ``new_choices`` not already present in ``existing`` (compared by
    Choice value). Used to dedupe background-streamed picker rows against what's already shown."""
    shown = {c.value for c in existing if isinstance(c, questionary.Choice)}
    return [c for c in new_choices if isinstance(c, questionary.Choice) and c.value not in shown]


def _scrolling_checkbox(
    message: str,
    choices: list[questionary.Choice | questionary.Separator],
    instruction: str,
    style: questionary.Style,
    allow_back: bool = False,
    background_loader: Callable[[Callable[[list[questionary.Choice]], None]], None] | None = None,
) -> Question:
    """Multi-select checkbox picker.

    ``background_loader``, if given, streams more choices in after the picker is already
    on screen: it's run on a daemon thread and handed an ``append(choices)`` callback that
    adds rows (deduped by value) and repaints, so the picker opens instantly on whatever
    ``choices`` are ready and fills in the rest without blocking. A footer shows a live
    "loading more…" count while it runs."""
    merged_style = merge_styles_default(
        [
            questionary.Style([("bottom-toolbar", "noreverse")]),
            style,
        ]
    )
    # Empty-tolerant control: the picker can open with zero selectable rows (streaming in via the
    # background loader, or an `mcp add` where everything is already configured) — see the subclass.
    control = _StreamingInquirerControl(
        choices,
        pointer="›",
        show_description=False,
    )
    # Live loading state for the background-loader footer (see below).
    loading = {"active": background_loader is not None, "found": 0}

    def get_prompt_tokens() -> list[tuple[str, str]]:
        tokens = [("class:qmark", ""), ("class:question", f" {message} ")]
        if control.is_answered:
            selected_count = len(control.selected_options)
            answer = "done" if selected_count == 0 else f"done ({selected_count} selections)"
            tokens.append(("class:answer", answer))
        else:
            tokens.append(("class:instruction", instruction))
        return tokens

    def get_selected_values() -> list[Any]:
        return [choice.value for choice in control.get_selected_values()]

    def perform_validation() -> bool:
        control.error_message = None
        return True

    @Condition
    def has_more_choices() -> bool:
        # Live so the scroll hint appears as background-loaded rows stream in.
        return len(control.choices) > MCP_PICKER_VISIBLE_ROWS

    @Condition
    def is_loading() -> bool:
        return bool(loading["active"])

    def loading_tokens() -> list[tuple[str, str]]:
        return [
            ("class:instruction", f"  ⏳ loading more MCP services… ({loading['found']} found)")
        ]

    @Condition
    def has_search_string() -> bool:
        return control.get_search_string_tokens() is not None

    validation_prompt: PromptSession = PromptSession(bottom_toolbar=lambda: control.error_message)
    # Render the prompt as a fixed 1-row window rather than a PromptSession
    # container: the latter expands to fill the terminal height, which in a tall
    # window pushes the choices list to the very bottom (a large blank gap).
    layout = Layout(
        HSplit(
            [
                Window(
                    height=Dimension.exact(1),
                    content=FormattedTextControl(get_prompt_tokens),
                ),
                ConditionalContainer(
                    # Height tracks the live choice count (capped at the visible max) so the
                    # window grows as background-loaded rows stream in, with no blank gap when
                    # only a few choices are present.
                    Window(
                        control,
                        height=lambda: Dimension.exact(
                            min(MCP_PICKER_VISIBLE_ROWS, max(1, len(control.choices)))
                        ),
                    ),
                    filter=~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(1),
                        content=FormattedTextControl(
                            lambda: [("class:instruction", "  ↑/↓ scroll for more")]
                        ),
                    ),
                    filter=has_more_choices & ~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(1),
                        content=FormattedTextControl(loading_tokens),
                    ),
                    filter=is_loading & ~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(2),
                        content=FormattedTextControl(control.get_search_string_tokens),
                    ),
                    filter=has_search_string & ~IsDone(),
                ),
                ConditionalContainer(
                    validation_prompt.layout.container,
                    filter=Condition(lambda: control.error_message is not None),
                ),
            ]
        )
    )

    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(" ", eager=True)
    def _(_event: Any) -> None:
        if control.choice_count == 0:
            return  # nothing to toggle (e.g. picker still streaming, or all rows filtered out)
        pointed = control.get_pointed_at()
        if isinstance(pointed, questionary.Separator) or pointed.disabled:
            return  # separators and already-configured (disabled) rows aren't toggleable
        pointed_choice = pointed.value
        if pointed_choice in control.selected_options:
            control.selected_options.remove(pointed_choice)
        else:
            control.selected_options.append(pointed_choice)
        perform_validation()

    @bindings.add(Keys.ControlA, eager=True)
    def _(_event: Any) -> None:
        # Toggle-all: select every selectable choice, or clear the selection if
        # everything is already selected. `a` alone is reserved for type-to-filter.
        selectable = [
            choice.value
            for choice in control.choices
            if not isinstance(choice, questionary.Separator) and not choice.disabled
        ]
        if all(value in control.selected_options for value in selectable):
            control.selected_options = []
        else:
            control.selected_options = list(selectable)
        perform_validation()

    def move_cursor_down(event: Any) -> None:
        if control.choice_count == 0:
            return
        control.select_next()
        # Bound the skip-past-disabled scan so an all-disabled list can't spin forever.
        tries = 0
        while not control.is_selection_valid() and tries < control.choice_count:
            control.select_next()
            tries += 1

    def move_cursor_up(event: Any) -> None:
        if control.choice_count == 0:
            return
        control.select_previous()
        tries = 0
        while not control.is_selection_valid() and tries < control.choice_count:
            control.select_previous()
            tries += 1

    def search_filter(event: Any) -> None:
        control.add_search_character(event.key_sequence[0].key)

    for character in string.printable:
        if character in string.whitespace:
            continue
        bindings.add(character, eager=True)(search_filter)
    bindings.add(Keys.Backspace, eager=True)(search_filter)

    bindings.add(Keys.Down, eager=True)(move_cursor_down)
    bindings.add(Keys.Up, eager=True)(move_cursor_up)
    bindings.add(Keys.ControlN, eager=True)(move_cursor_down)
    bindings.add(Keys.ControlP, eager=True)(move_cursor_up)

    @bindings.add(Keys.ControlM, eager=True)
    def _(event: Any) -> None:
        control.submission_attempted = True
        if perform_validation():
            control.is_answered = True
            event.app.exit(result=get_selected_values())

    if allow_back:

        @bindings.add(Keys.Left, eager=True)
        def _(event: Any) -> None:
            # Wizard back-navigation: exit this step with the _BACK sentinel so
            # the caller re-shows the previous step. Left arrow is otherwise
            # unused in this multi-select (cursor moves with up/down).
            event.app.exit(result=_BACK)

    @bindings.add(Keys.Any)
    def _(_event: Any) -> None:
        """Ignore other text input."""

    app: Application = Application(
        layout=layout,
        key_bindings=bindings,
        style=merged_style,
    )

    if background_loader is not None:
        started = False

        def run_on_loop(fn: Callable[[], None]) -> None:
            # Background updates MUST run on the picker's event-loop thread: mutating the control
            # off-thread drops rows (prompt_toolkit's invalidate() no-ops until the app is running)
            # and disturbs live input (toggling/removal). Wait briefly for the app to start, then
            # hand `fn` to the loop; bail once the picker has closed or if it never starts in time.
            nonlocal started
            deadline = time.monotonic() + 15.0
            while not (app.is_running and app.loop is not None):
                if started or time.monotonic() > deadline:
                    return
                time.sleep(0.02)
            started = True
            with suppress(Exception):
                app.loop.call_soon_threadsafe(fn)

        def append(new_choices: list[questionary.Choice]) -> None:
            def apply() -> None:
                # On the UI thread: rebind choices (atomic; render reads the list live) and repaint.
                # Selections track by value, so appended rows never disturb checkboxes/scroll/filter.
                additions = _merge_new_choices(control.choices, new_choices)
                if additions:
                    control.choices = [*control.choices, *additions]
                    loading["found"] += len(additions)
                    app.invalidate()

            run_on_loop(apply)

        def worker() -> None:
            try:
                background_loader(append)
            except Exception:
                # Discovery is best-effort; a failed background walk just stops streaming.
                pass
            finally:

                def finish() -> None:
                    loading["active"] = False
                    app.invalidate()

                run_on_loop(finish)

        threading.Thread(target=worker, name="mcp-picker-loader", daemon=True).start()

    return Question(app)


def build_mcp_picker_choices(
    available_external_names: list[str],
    available_genie_servers: list[dict],
    available_app_servers: list[dict],
    original_servers: list[dict],
    available_mcp_service_names: list[str] | None = None,
    available_vector_search_servers: list[dict] | None = None,
    available_uc_functions_servers: list[dict] | None = None,
    additive: bool = False,
) -> list[questionary.Choice | questionary.Separator]:
    original_by_name = _servers_by_name(original_servers)
    known_names = set(original_by_name)

    def known_choice(name: str, title: str | None = None) -> questionary.Choice:
        # `ucode mcp add` (additive) never removes an already-configured server, so
        # show it as a non-toggleable note rather than a pre-checked box whose
        # unchecking would be silently ignored. `configure mcp` (replace) keeps it a
        # pre-checked toggle so unchecking removes it.
        if additive:
            return questionary.Choice(
                title=title or name, value=name, disabled="already configured"
            )
        return _server_choice(name, True, title)

    choices: list[questionary.Choice | questionary.Separator] = []
    displayed_names: set[str] = set()

    # Databricks SQL is intentionally NOT offered as an up-front add-choice — we don't promote
    # it. If it's exposed as a `system.ai` MCP service it shows like any other service, and an
    # already-configured `databricks-sql` still appears (removable) via the known-server fallback
    # at the end. The `managed:sql` selection value is still resolvable for managed configs.

    for name in available_mcp_service_names or []:
        # Picker shows the dotted UC name; state/agents store the dashed form
        # (see resolver). The shared helper is also used by the background walk that
        # streams more services in, so up-front and streamed rows match exactly.
        choices.append(_mcp_service_choice(name, known_names, additive))
        displayed_names.add(name.replace(".", "-"))

    for name in available_external_names:
        display_title = f"Connection: {name}"
        if name in known_names:
            choices.append(known_choice(name, display_title))
        else:
            choices.append(_add_choice(f"{EXTERNAL_MCP_SELECTION_PREFIX}{name}", display_title))
        displayed_names.add(name)

    for server in available_genie_servers:
        name = _server_name(server)
        title = server.get("title")
        if not name:
            continue
        display_title = f"Genie: {title}" if isinstance(title, str) and title else name
        if name in known_names:
            choices.append(known_choice(name, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{GENIE_SPACE_SELECTION_PREFIX}{name.removeprefix('databricks-genie-')}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_app_servers:
        name = _server_name(server)
        title = server.get("title")
        if not name:
            continue
        display_title = f"App: {title}" if isinstance(title, str) and title else name
        if name in known_names:
            choices.append(known_choice(name, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{APP_MCP_SELECTION_PREFIX}{name.removeprefix('databricks-app-')}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_vector_search_servers or []:
        name = _server_name(server)
        catalog = server.get("catalog")
        schema = server.get("schema")
        if not name or not isinstance(catalog, str) or not isinstance(schema, str):
            continue
        display_title = f"Vector Search: {catalog}.{schema}"
        if name in known_names:
            choices.append(known_choice(name, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{VECTOR_SEARCH_SELECTION_PREFIX}{catalog}.{schema}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_uc_functions_servers or []:
        name = _server_name(server)
        catalog = server.get("catalog")
        schema = server.get("schema")
        if not name or not isinstance(catalog, str) or not isinstance(schema, str):
            continue
        display_title = f"UC Functions: {catalog}.{schema}"
        if name in known_names:
            choices.append(known_choice(name, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{UC_FUNCTIONS_SELECTION_PREFIX}{catalog}.{schema}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for name in sorted(known_names - displayed_names):
        choices.append(known_choice(name))
    return choices


def prompt_for_mcp_server_choices(
    available_external_names: list[str],
    available_genie_servers: list[dict],
    available_app_servers: list[dict],
    original_servers: list[dict],
    available_mcp_service_names: list[str] | None = None,
    available_vector_search_servers: list[dict] | None = None,
    available_uc_functions_servers: list[dict] | None = None,
    allow_back: bool = False,
    additive: bool = False,
    background_loader: Callable[[Callable[[list[questionary.Choice]], None]], None] | None = None,
) -> list[str] | None | _Back:
    """Show the MCP server picker. Returns the list of selected values, `None`
    if cancelled (Ctrl-C), or `_BACK` if `allow_back` and the user pressed Left
    to return to the previous wizard step.

    ``additive`` (``ucode mcp add``) shows already-configured servers as
    non-toggleable notes instead of pre-checked, removable boxes.

    ``background_loader`` streams more choices in after the picker opens (see
    `_scrolling_checkbox`) — used to load the workspace-wide MCP-services walk without
    blocking on it up front."""
    instruction = "(space to toggle, ctrl-a all, enter to save, type to filter)"
    if allow_back:
        instruction = "(space to toggle, ctrl-a all, ← back, enter to save, type to filter)"
    selection = _scrolling_checkbox(
        "MCP:",
        choices=build_mcp_picker_choices(
            available_external_names,
            available_genie_servers,
            available_app_servers,
            original_servers,
            available_mcp_service_names,
            available_vector_search_servers,
            available_uc_functions_servers,
            additive=additive,
        ),
        style=_picker_style(),
        instruction=instruction,
        allow_back=allow_back,
        background_loader=background_loader,
    ).ask()
    if selection is None:
        return None
    if selection is _BACK:
        return _BACK
    return [str(value) for value in selection]


def _mcp_server_clients(server: dict) -> list[str]:
    return [client for client in (server.get("clients") or []) if client in MCP_CLIENTS]


def _is_app_mcp_server(server: dict) -> bool:
    """Whether a registered server points at a Databricks app (an off-workspace ``*/mcp`` host).

    Apps are the residual ``/mcp`` URL shape — everything else ucode registers is a known
    workspace-relative path. Used to hide already-registered apps from the picker where they can't be
    published (``ucode setup``)."""
    url = server.get("url")
    if not isinstance(url, str):
        return False
    stripped = url.rstrip("/")
    known = (
        "/ai-gateway/mcp-services/",
        "/api/2.0/mcp/external/",
        "/api/2.0/mcp/genie/",
        "/api/2.0/mcp/vector-search/",
        "/api/2.0/mcp/functions/",
    )
    if any(fragment in url for fragment in known):
        return False
    if stripped.endswith("/api/2.0/mcp/sql"):
        return False
    return stripped.endswith("/mcp")


def managed_mcp_server_entry(name: str, mcp_type: str, workspace: str) -> tuple[str, str] | None:
    """Rebuild an ``(entry_name, url)`` pair from a managed config's ``{name, type}`` entry.

    ``entry_name`` is the identifier the server is registered under with the agent (dots stripped,
    since the agent CLIs reject them); ``url`` is what the proxy forwards to. Returns None for a
    type/name this can't reconstruct, so the caller skips it rather than registering a broken server.
    Mirrors the shapes :func:`_resolve_mcp_selection` builds for the interactive picker, so a managed
    and a locally-configured copy of the same server land on the same name.

    The ai-gateway ``McpServer.name`` field is interpreted per ``type`` (see the proto): a UC name for
    a UC service, a Genie space id for a genie space, a connection name for external, and — as ucode
    serializes them — a `<catalog>.<schema>` for vector-search / uc-functions.
    """
    if mcp_type == "sql":
        return "databricks-sql", f"{workspace}/api/2.0/mcp/sql"
    if mcp_type == "external":
        return name, f"{workspace}/api/2.0/mcp/external/{name}"
    if mcp_type == "mcp-service":
        # Stored in dash form (`system-ai-dbsql`), which is already the registered name; the URL wants
        # the UC dotted form. Only the catalog and schema separators (first two dashes) become dots —
        # the service name keeps its own dashes/underscores.
        parts = name.split("-", 2)
        if len(parts) != 3:
            return None
        return name, build_mcp_service_url(workspace, ".".join(parts))
    if mcp_type == "genie-space":
        # `name` is the Genie space id (per the proto); register under the id-based name the
        # interactive path falls back to, and point the URL at the space.
        return f"databricks-genie-{name}", f"{workspace}/api/2.0/mcp/genie/{name}"
    if mcp_type in ("vector-search", "uc-functions"):
        # `name` is a `<catalog>.<schema>`; the URL is workspace-relative on that pair, and the
        # registered name is the same dot-free slug the interactive path uses.
        catalog, _, schema = name.partition(".")
        if not catalog or not schema or "." in schema:
            return None
        url_path = "vector-search" if mcp_type == "vector-search" else "functions"
        name_prefix = (
            "databricks-vector-search" if mcp_type == "vector-search" else "databricks-functions"
        )
        entry_name = _catalog_schema_server_name(name_prefix, catalog, schema, set())
        return entry_name, f"{workspace}/api/2.0/mcp/{url_path}/{catalog}/{schema}"
    return None


def apply_managed_mcp_servers(
    managed: dict, tool: str, workspace: str, profile: str | None = None, *, use_pat: bool = False
) -> list[dict]:
    """Register the managed config's MCP servers with ``tool`` so they reach its `/mcp` list.

    The managed config only lists ``{name, type}`` entries; nothing else on the launch path turns
    them into agent MCP registrations, so without this a workspace-published server never shows up.
    Reconstructs each entry's ``(name, url)`` (see :func:`managed_mcp_server_entry`), diffs against
    what ucode previously registered, and applies the change for the launching tool only. Entries
    whose URL can't be rebuilt (e.g. ``app``, which needs an off-workspace host) are skipped.

    Returns the server dicts registered (for state persistence); an empty list when the config names
    none, or names only types that can't yet be reconstructed.
    """
    if tool not in MCP_CLIENTS:
        return []
    entries = managed.get("mcp_servers")
    if not isinstance(entries, list):
        return []
    working: list[dict] = []
    seen: set[str] = set()
    skipped: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        mcp_type = entry.get("type")
        if not isinstance(name, str) or not name or not isinstance(mcp_type, str):
            continue
        resolved = managed_mcp_server_entry(name, mcp_type, workspace)
        if resolved is None:
            skipped.append(f"{name} ({mcp_type})")
            continue
        entry_name, url = resolved
        if entry_name in seen:
            continue
        seen.add(entry_name)
        working.append({"name": entry_name, "url": url, "auth": "proxy", "clients": [tool]})
    if skipped:
        print_warning(
            "Skipping managed MCP server(s) ucode can't yet auto-register from the workspace "
            f"config: {', '.join(skipped)}. Add them with `ucode configure mcp`."
        )
    if not working:
        return []
    # Diff against the managed servers ucode registered on a prior launch so a removed entry is
    # unregistered and an unchanged one is a no-op. Only this tool's managed servers are considered.
    state = load_state()
    previous = [
        server
        for server in (state.get("managed_mcp_servers") or [])
        if isinstance(server, dict) and tool in (server.get("clients") or [])
    ]
    apply_mcp_server_changes(previous, working, [tool], workspace, profile, use_pat=use_pat)
    return working


def _resolve_mcp_selection(
    selection: str,
    workspace: str,
    available_app_servers: list[dict] | None = None,
    available_genie_servers: list[dict] | None = None,
    available_vector_search_servers: list[dict] | None = None,
    available_uc_functions_servers: list[dict] | None = None,
) -> tuple[str, str]:
    if selection.startswith(APP_MCP_SELECTION_PREFIX):
        app_name = selection.removeprefix(APP_MCP_SELECTION_PREFIX)
        if not app_name:
            raise RuntimeError("missing Databricks app name")
        server = _servers_by_name(available_app_servers or []).get(f"databricks-app-{app_name}")
        if not server:
            raise RuntimeError(f"Databricks app `{app_name}` was not in the discovered app list")
        url = server.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError(f"Databricks app `{app_name}` has no MCP URL")
        return f"databricks-app-{app_name}", url

    if selection.startswith(GENIE_SPACE_SELECTION_PREFIX):
        suffix = selection.removeprefix(GENIE_SPACE_SELECTION_PREFIX)
        if not suffix:
            raise RuntimeError("missing Genie space id")
        server_name = f"databricks-genie-{suffix}"
        server = _servers_by_name(available_genie_servers or []).get(server_name)
        if server:
            url = server.get("url")
            if isinstance(url, str) and url:
                return server_name, url
        # Fallback for legacy picker values that carried the raw space_id.
        return server_name, f"{workspace}/api/2.0/mcp/genie/{suffix}"

    if selection.startswith(EXTERNAL_MCP_SELECTION_PREFIX):
        server_name = selection.removeprefix(EXTERNAL_MCP_SELECTION_PREFIX)
        if not server_name:
            raise RuntimeError("missing external connection name")
        return server_name, f"{workspace}/api/2.0/mcp/external/{server_name}"

    if selection.startswith(MCP_SERVICE_SELECTION_PREFIX):
        full_name = selection.removeprefix(MCP_SERVICE_SELECTION_PREFIX)
        if not full_name:
            raise RuntimeError("missing MCP service name")
        # Agent CLIs (claude/codex/gemini) reject dots in registered names.
        # URL keeps the UC `<cat>.<schema>.<id>` form; entry name uses dashes.
        return full_name.replace(".", "-"), build_mcp_service_url(workspace, full_name)

    if selection.startswith(VECTOR_SEARCH_SELECTION_PREFIX):
        return _resolve_catalog_schema_selection(
            selection.removeprefix(VECTOR_SEARCH_SELECTION_PREFIX),
            kind="vector search",
            url_path="vector-search",
            name_prefix="databricks-vector-search",
            workspace=workspace,
            available_servers=available_vector_search_servers,
        )

    if selection.startswith(UC_FUNCTIONS_SELECTION_PREFIX):
        return _resolve_catalog_schema_selection(
            selection.removeprefix(UC_FUNCTIONS_SELECTION_PREFIX),
            kind="UC functions",
            url_path="functions",
            name_prefix="databricks-functions",
            workspace=workspace,
            available_servers=available_uc_functions_servers,
        )

    if selection == SQL_MCP_VALUE:
        return "databricks-sql", f"{workspace}/api/2.0/mcp/sql"

    raise RuntimeError(f"unrecognized selection prefix in `{selection}`")


def _resolve_catalog_schema_selection(
    payload: str,
    *,
    kind: str,
    url_path: str,
    name_prefix: str,
    workspace: str,
    available_servers: list[dict] | None,
) -> tuple[str, str]:
    """Map a `catalog.schema` picker value back to the discovered server's name
    and URL, falling back to a deterministic slug when discovery has been lost
    (e.g. picker reopened on a stale workspace)."""
    if not payload or "." not in payload:
        raise RuntimeError(f"missing catalog.schema for {kind}")
    catalog, _, schema = payload.partition(".")
    if not catalog or not schema:
        raise RuntimeError(f"missing catalog.schema for {kind}")
    for server in available_servers or []:
        if server.get("catalog") == catalog and server.get("schema") == schema:
            name = _server_name(server)
            url = server.get("url")
            if name and isinstance(url, str) and url:
                return name, url
    name = _catalog_schema_server_name(name_prefix, catalog, schema, set())
    return name, f"{workspace}/api/2.0/mcp/{url_path}/{catalog}/{schema}"


def _discover_mcp_source(label: str, discover: Callable[[], list[Any]]) -> list[Any]:
    try:
        with spinner(f"Discovering {label}..."):
            return discover()
    except PermissionDeniedError:
        # Consumer-only identities lack workspace access, so this source 403s for them.
        # Skip it quietly (not as a scary warning) so setup completes (AIGTWY-4471).
        print_note(f"Skipped {label} (no workspace access).")
        return []
    except (RuntimeError, OSError) as exc:
        # Discovery is best-effort: a failure here (network timeout, transient error)
        # skips just this source so the rest of the picker still works.
        print_warning(f"Skipped {label} ({exc}).")
        return []


def _mcp_services_background_loader(
    workspace: str,
    profile: str | None,
    known_names: set[str],
    additive: bool,
) -> Callable[[Callable[[list[questionary.Choice]], None]], None]:
    """Return a picker `background_loader` that runs the workspace-wide MCP-services walk and
    streams each schema's newly-found services into the open picker as choices, so the walk
    never blocks the picker from opening. Deduping against already-shown rows (e.g. the fast
    `system.ai` list) is handled by the picker's append."""

    def loader(append: Callable[[list[questionary.Choice]], None]) -> None:
        def on_services(new_names: list[str]) -> None:
            append([_mcp_service_choice(name, known_names, additive) for name in new_names])

        discover_all_mcp_service_names(workspace, profile, on_services=on_services)

    return loader


def _discover_selected_mcp_sources(
    workspace: str, profile: str | None, sources: set[str]
) -> dict[str, list]:
    """Discover the picker's sources. The picker searches a single source — MCP services — so
    this fetches the fast curated `system.ai` list synchronously (the slow workspace-wide walk
    streams in afterward via the picker's background loader). The other keys stay in the returned
    dict as empty lists so the picker call is unchanged and can still render/remove
    already-registered servers of any type."""
    services: list[str] = []
    if MCP_SERVICES_SOURCE in sources:
        services = _discover_mcp_source(
            "MCP services",
            lambda: discover_mcp_service_names(workspace, profile),
        )
    return {
        "external": [],
        "apps": [],
        "services": services,
        "genie": [],
        "vector_search": [],
        "uc_functions": [],
    }


def apply_mcp_server_changes(
    original_servers: list[dict],
    working_servers: list[dict],
    clients: list[str],
    workspace: str,
    profile: str | None = None,
    *,
    use_pat: bool = False,
) -> bool:
    original_by_name = _servers_by_name(original_servers)
    working_by_name = _servers_by_name(working_servers)

    # Build the per-client work lists. Each add/remove shells out to a CLI or
    # rewrites a config file, so a large diff means hundreds of operations; we
    # run them concurrently ACROSS clients but SERIALLY within a client, since
    # every operation for one client mutates that client's single shared config
    # (`claude mcp add-json` edits ~/.claude.json, etc.) and concurrent
    # read-modify-writes would clobber each other.
    work: dict[str, list[Callable[[], object]]] = {client: [] for client in clients}
    changed = False

    for name, server in original_by_name.items():
        if name not in working_by_name:
            for client in _mcp_server_clients(server):
                work.setdefault(client, []).append(
                    lambda c=client, n=name: remove_client_mcp_server(c, n)
                )
            changed = True

    for name, server in working_by_name.items():
        original = original_by_name.get(name)
        if original == server:
            continue
        url = server.get("url")
        if not isinstance(url, str) or not url:
            continue
        # alwaysLoad (Claude-only) keeps the skills registry's utility tools
        # discoverable without an explicit mention; other clients ignore it.
        always_load = server.get("kind") == SKILLS_MCP_KIND
        for client in clients:
            work[client].append(
                lambda c=client, n=name, u=url, al=always_load: configure_client_mcp_server(
                    c, n, u, workspace, profile, use_pat=use_pat, always_load=al
                )
            )
        changed = True

    total_ops = sum(len(ops) for ops in work.values())
    if total_ops == 0:
        return changed

    completed = _Counter()

    def run_client_ops(ops: list[Callable[[], object]]) -> None:
        for op in ops:
            op()
            completed.increment()

    def message() -> str:
        return f"Configuring MCP servers... {completed.value()}/{total_ops}"

    with spinner(message):
        with ThreadPoolExecutor(max_workers=max(1, len(work))) as pool:
            futures = [pool.submit(run_client_ops, ops) for ops in work.values() if ops]
            # Surface the first failure (if any) once all client threads finish.
            for future in as_completed(futures):
                future.result()

    return changed


class _Counter:
    """Thread-safe monotonic counter for cross-thread progress reporting."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def value(self) -> int:
        with self._lock:
            return self._value


def purge_cross_workspace_mcp_residue(state: dict, workspace: str) -> None:
    installed = set(available_mcp_clients())

    raw_mcp_servers = list(state.get("mcp_servers") or [])
    current_mcp_servers, foreign_mcp_servers = _partition_mcp_entries_by_workspace(
        raw_mcp_servers, workspace
    )
    if foreign_mcp_servers:
        foreign_names = ", ".join(
            (_server_name(server) or "(unnamed)") for server in foreign_mcp_servers
        )
        noun = "entry" if len(foreign_mcp_servers) == 1 else "entries"
        print_warning(
            f"Dropping {len(foreign_mcp_servers)} stale MCP {noun} "
            f"not bound to this workspace: {foreign_names}."
        )
        for server in foreign_mcp_servers:
            name = _server_name(server)
            if not name:
                continue
            for client in server.get("clients") or []:
                if client not in installed or client not in MCP_CLIENTS:
                    continue
                try:
                    remove_client_mcp_server(client, name)
                except RuntimeError as exc:
                    print_warning(
                        f"Failed to remove `{name}` from {MCP_CLIENTS[client]['display']}: {exc}"
                    )
        state["mcp_servers"] = current_mcp_servers
        save_state(state)

    other_ws_mcps = _mcp_entries_only_in_other_workspaces(workspace)
    actually_removed: list[str] = []
    for name in sorted(other_ws_mcps):
        any_removed = False
        for client in other_ws_mcps[name]:
            if client not in installed or client not in MCP_CLIENTS:
                continue
            try:
                removed_scopes = remove_client_mcp_server(client, name)
            except RuntimeError as exc:
                print_warning(
                    f"Failed to remove `{name}` from {MCP_CLIENTS[client]['display']}: {exc}"
                )
                continue
            if removed_scopes:
                any_removed = True
        if any_removed:
            actually_removed.append(name)
    if actually_removed:
        noun = "entry" if len(actually_removed) == 1 else "entries"
        print_warning(
            f"Removed {len(actually_removed)} MCP {noun} left over from "
            f"previously-configured workspaces: {', '.join(actually_removed)}."
        )


def _skills_entries(servers: list[dict]) -> list[dict]:
    return [s for s in servers if s.get("kind") == SKILLS_MCP_KIND]


def _resolve_location_mcp_servers(
    workspace: str,
    profile: str | None,
    clients: list[str],
    location: str,
    original_servers: list[dict],
    services: set[str] | None = None,
) -> list[dict]:
    """Build the desired MCP server list for ``--location <cat>.<schema>``.

    Strict replacement for mcp-services: the returned list is exactly the ones
    discovered at ``location`` (any previously-registered mcp-service outside it
    is removed by ``apply_mcp_server_changes``), plus any existing skills
    connection, preserved untouched. Raises ``RuntimeError`` for an invalid
    location (HTTP 404 from the listing API) or any other listing failure.

    When ``services`` is given, the discovered set is narrowed to exactly that
    subset (matched by full name like ``system.ai.github`` or bare short name
    like ``github``); names not found at ``location`` are skipped with a
    warning rather than failing, so a saved selection that references a
    since-removed service still configures the rest. An empty set selects
    nothing (every previously-registered service in the location is removed).
    ``None`` keeps the whole schema."""
    if location.count(".") != 1 or not all(part.strip() for part in location.split(".")):
        raise RuntimeError(f"--location must be `<catalog>.<schema>`, got `{location}`.")

    token = get_databricks_token(workspace, profile)
    with spinner(f"Discovering MCP services in {location}..."):
        names, reason = list_mcp_services(workspace, token, parent=location)

    if reason and reason.startswith("HTTP 404"):
        raise RuntimeError(
            f"Invalid location: `{location}` is not a valid Unity Catalog schema "
            "in this workspace (or you lack USE permission on it)."
        )
    if reason:
        raise RuntimeError(f"Failed to list MCP services at `{location}`: {reason}")
    if not names:
        print_note(f"No MCP services exist at `{location}`.")

    if services is not None:
        discovered_full = set(names)
        discovered_short = {full_name.split(".")[-1] for full_name in names}
        unknown = services - discovered_full - discovered_short
        if unknown:
            print_warning(
                f"Ignoring requested MCP services not found in `{location}`: "
                f"{', '.join(sorted(unknown))}."
            )
        names = [
            full_name
            for full_name in names
            if full_name in services or full_name.split(".")[-1] in services
        ]

    original_by_name = _servers_by_name(original_servers)
    working_servers: list[dict] = []
    for full_name in names:
        entry_name = full_name.replace(".", "-")
        original = original_by_name.get(entry_name)
        original_clients = list((original or {}).get("clients") or [])
        merged_clients = original_clients + [c for c in clients if c not in original_clients]
        candidate = {
            "name": entry_name,
            "url": build_mcp_service_url(workspace, full_name),
            "auth": "proxy",
            "clients": merged_clients,
        }
        if original is not None and original == candidate:
            working_servers.append(original.copy())
        else:
            working_servers.append(candidate)
    return [*working_servers, *_skills_entries(original_servers)]


# The interactive picker searches a single source: MCP services (the `/ai-gateway/mcp-services/`
# path), the one source a consumer-only identity can reach. The V2 AI Gateway sources — external
# connections, Databricks apps, Genie spaces, Vector Search, and UC functions, all served under
# `/api/2.0/mcp/*` — aren't offered in the picker because consumer entitlements don't grant access
# to them; workspace users add one non-interactively with a typed `--services` selector (see
# `V2_MCP_SELECTOR_PREFIXES` and `_configure_v2_mcp_selectors`). Since there's a single source,
# there is no "choose sources" wizard step.
MCP_SERVICES_SOURCE = "mcp-services"

# Typed `--services` selectors that name a V2 AI Gateway MCP server directly, e.g.
# `vector-search:main.docs` or `uc-functions:main.tools`. These bypass the interactive
# picker (which no longer offers V2 sources) so workspace users can still add them on
# request; a consumer-only identity is blocked with a clear error before registering.
V2_MCP_SELECTOR_PREFIXES = (
    VECTOR_SEARCH_SELECTION_PREFIX,
    UC_FUNCTIONS_SELECTION_PREFIX,
    EXTERNAL_MCP_SELECTION_PREFIX,
    GENIE_SPACE_SELECTION_PREFIX,
    APP_MCP_SELECTION_PREFIX,
)


def _is_v2_mcp_selector(service: str) -> bool:
    """Whether a `--services` entry is a typed V2 MCP selector (see `V2_MCP_SELECTOR_PREFIXES`)."""
    return service.startswith(V2_MCP_SELECTOR_PREFIXES)


def setup_mcp_clients(
    state: dict,
    section: str,
    *,
    require_auth: bool = True,
    action_note: str = "Configuring for",
    agents: set[str] | None = None,
) -> tuple[str, str | None, list[str]]:
    """Validate the workspace, resolve configured MCP clients, and prepare auth.

    Returns ``(workspace, profile, clients)`` and prints the section header, the
    ``action_note`` line, and a warning per configured-but-uninstalled client.

    ``require_auth`` forces a Databricks login (needed to register a server); the
    removal path passes ``False`` since unregistering a server is purely local and
    should work even when the workspace token has expired.

    ``agents`` (from ``--agents``) scopes the returned clients to that subset of
    the configured MCP clients, so the operation touches only those agents instead
    of every configured one. Requested agents that aren't configured/installed
    raise a clear error.
    """
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    purge_cross_workspace_mcp_residue(state, workspace)

    installed_clients = available_mcp_clients()
    if not installed_clients:
        raise RuntimeError(
            "No supported MCP clients are installed. Install Claude, Codex, Gemini, OpenCode, "
            "or GitHub Copilot CLI."
        )
    clients = configured_mcp_clients(state, installed_clients)
    if agents is not None:
        missing = sorted(a for a in agents if a not in clients)
        if missing:
            raise RuntimeError(
                f"Requested agent(s) not configured for MCP: {', '.join(missing)}. "
                f"Configure them first with `ucode configure --agents {','.join(missing)}`."
            )
        clients = [client for client in clients if client in agents]
    if not clients:
        raise RuntimeError(
            "No configured MCP-capable coding agents are installed. Run `ucode configure` "
            "for Codex, Claude, Gemini, OpenCode, or GitHub Copilot CLI first."
        )
    configured_tools = set(state.get("available_tools") or [])
    missing_clients = [
        client for client in MCP_CLIENTS if client in configured_tools and client not in clients
    ]

    profile = state.get("profile")
    if require_auth:
        apply_pat_environment(state)
        ensure_databricks_auth(workspace, profile)

    print_section(section)
    client_names = ", ".join(str(MCP_CLIENTS[client]["display"]) for client in clients)
    print_note(f"{action_note}: {client_names}")
    for client in missing_clients:
        print_warning(
            f"{MCP_CLIENTS[client]['display']} is configured in ucode but not installed; "
            "skipping MCP config."
        )
    return workspace, profile, clients


def _union_missing(base: list[dict], selected: list[dict]) -> list[dict]:
    """Return ``selected`` followed by every ``base`` server whose name isn't
    already in it. Used by ``ucode mcp add`` so registering new servers never
    removes ones that are already configured (append semantics)."""
    have = _servers_by_name(selected)
    extra = [s for s in base if (_server_name(s) or "") not in have]
    return [*selected, *extra]


def add_mcp_command(
    location: str | None = None,
    services: set[str] | None = None,
    agents: set[str] | None = None,
) -> int:
    """`ucode mcp add`: register Databricks MCP servers WITHOUT removing any that
    are already configured.

    Uses the same discovery and options as `configure mcp` — the interactive
    picker, or the non-interactive `--location`/`--services` paths — but is purely
    additive: unlike `configure mcp`, it never removes servers outside the
    selection.

    ``agents`` scopes the registration to that subset of configured MCP clients
    (the agents must already be configured — the `--agents` CLI option sets up any
    that aren't before calling this)."""
    if services is not None and not services:
        # An empty `--services` selects nothing. For `configure mcp` that means
        # "remove all"; for the additive `add` there is simply nothing to register,
        # so it's a no-op (and doesn't need --location the way a real subset does).
        print_note("No MCP services given to add (empty --services); nothing to do.")
        return 0
    return configure_mcp_command(location=location, services=services, append=True, agents=agents)


def _configure_v2_mcp_selectors(
    selectors: list[str],
    *,
    append: bool,
    agents: set[str] | None,
) -> int:
    """Non-interactive add for V2 AI Gateway MCP servers named by typed `--services`
    selectors (`vector-search:`/`uc-functions:`/`external:`/`genie-space:`/`app:`).

    The interactive picker no longer offers these sources; this is how a workspace user adds one
    on request. These require workspace access, which consumer-only identities lack — but that's
    enforced upstream at the AI Gateway (which ucode already hits during model setup), not here:
    the listing calls this uses don't reliably signal consumer access (see `PermissionDeniedError`).
    Registration mirrors the interactive add path: additive under ``append`` (`ucode mcp add`), an
    exact replacement otherwise (`ucode configure mcp`), always preserving the skills connection."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(
        state, "Add MCP Servers" if append else "MCP Servers", agents=agents
    )

    # `app:` selectors need the app's off-workspace URL, which only discovery knows. A 403 here
    # means the caller can't list apps (no workspace access, or no apps permission).
    available_app_servers: list[dict] = []
    if any(s.startswith(APP_MCP_SELECTION_PREFIX) for s in selectors):
        try:
            available_app_servers = discover_app_mcp_servers(workspace, profile)
        except PermissionDeniedError as exc:
            raise RuntimeError(
                f"{exc} This needs workspace access to the Databricks apps listing; ask a "
                "workspace admin if you're missing it."
            ) from exc

    original_mcp_servers: list[dict] = list(state.get("mcp_servers") or [])
    skills_servers = _skills_entries(original_mcp_servers)
    picker_servers = [s for s in original_mcp_servers if s.get("kind") != SKILLS_MCP_KIND]
    original_by_name = _servers_by_name(picker_servers)

    working_mcp_servers: list[dict] = list(skills_servers)
    working_names: set[str] = set()
    for selection in selectors:
        entry_name, url = _resolve_mcp_selection(selection, workspace, available_app_servers)
        if entry_name in working_names:
            continue
        working_mcp_servers.append(
            {"name": entry_name, "url": url, "auth": "proxy", "clients": clients}
        )
        working_names.add(entry_name)

    if append:
        working_mcp_servers = _union_missing(original_mcp_servers, working_mcp_servers)

    changed = apply_mcp_server_changes(
        original_mcp_servers,
        working_mcp_servers,
        clients,
        workspace,
        profile,
        use_pat=bool(state.get("use_pat")),
    )
    if changed or original_mcp_servers != working_mcp_servers:
        state["mcp_servers"] = working_mcp_servers
        save_state(state)
        added = sorted(working_names - set(original_by_name))
        removed = [] if append else sorted(set(original_by_name) - working_names)
        print_success(_mcp_change_summary(added, removed, clients))
    return 0


def configure_mcp_command(
    location: str | None = None,
    services: set[str] | None = None,
    *,
    exclude_sources: set[str] | None = None,
    append: bool = False,
    agents: set[str] | None = None,
) -> int:
    """Interactive MCP picker. ``exclude_sources`` hides search sources the caller can't use —
    `ucode setup` passes ``{"apps"}`` because a managed config can't carry an app's off-workspace
    host, so an app picked here would be silently dropped from the published config.

    ``append`` (used by `ucode mcp add`) makes the command purely additive: the
    final server list is unioned with the already-configured servers, so nothing
    outside the current selection is removed. ``agents`` scopes the operation to
    that subset of configured MCP clients."""
    if services is not None:
        # A typed V2 MCP selector (`vector-search:main.docs`, `uc-functions:main.tools`,
        # `external:conn`, `genie-space:<id>`, `app:<name>`) names a server the picker no
        # longer offers. Route it through the dedicated non-interactive path so workspace
        # users can still add it on request; consumer-only identities are blocked there.
        v2_selectors = sorted(s for s in services if _is_v2_mcp_selector(s))
        if v2_selectors:
            other = sorted(s for s in services if not _is_v2_mcp_selector(s))
            if other or location is not None:
                raise RuntimeError(
                    "V2 MCP selectors (vector-search:/uc-functions:/external:/genie-space:/app:) "
                    "can't be combined with --location or plain MCP-service names in one call; add "
                    "them in a separate command."
                )
            return _configure_v2_mcp_selectors(v2_selectors, append=append, agents=agents)
    if services is not None and location is None:
        # `--services` works standalone with full names (`system.ai.github`): the
        # `<catalog>.<schema>` to configure is derived from them. Bare short names
        # (`github`) can't be located without `--location`.
        schemas = {".".join(s.split(".")[:2]) for s in services if s.count(".") >= 2}
        bare = sorted(s for s in services if s.count(".") < 2)
        if bare:
            raise RuntimeError(
                "--services short names need --location (or pass full names like "
                f"`system.ai.<name>`): {', '.join(bare)}"
            )
        if len(schemas) != 1:
            raise RuntimeError(
                "--services without --location must all share one `<catalog>.<schema>` "
                f"(got: {', '.join(sorted(schemas)) or 'none'}); pass --location instead."
            )
        location = next(iter(schemas))
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(
        state, "Add MCP Servers" if append else "MCP Servers", agents=agents
    )

    original_mcp_servers_for_location: list[dict] = list(state.get("mcp_servers") or [])
    if location is not None:
        working_mcp_servers = _resolve_location_mcp_servers(
            workspace, profile, clients, location, original_mcp_servers_for_location, services
        )
        if append:
            working_mcp_servers = _union_missing(
                original_mcp_servers_for_location, working_mcp_servers
            )
        changed = apply_mcp_server_changes(
            original_mcp_servers_for_location,
            working_mcp_servers,
            clients,
            workspace,
            profile,
            use_pat=bool(state.get("use_pat")),
        )
        if changed or original_mcp_servers_for_location != working_mcp_servers:
            state["mcp_servers"] = working_mcp_servers
            save_state(state)
            print_success("Saved")
        return 0

    excluded_sources = exclude_sources or set()
    original_mcp_servers: list[dict] = list(state.get("mcp_servers") or [])
    # Skills connections are managed by `configure skills`, so keep them out of
    # the picker and carry them through untouched.
    skills_servers = _skills_entries(original_mcp_servers)
    picker_servers = [s for s in original_mcp_servers if s.get("kind") != SKILLS_MCP_KIND]
    # Drop already-registered servers from an excluded source too (e.g. a previously-added app under
    # `ucode setup`), so the picker never shows a server the caller couldn't re-add.
    if "apps" in excluded_sources:
        picker_servers = [s for s in picker_servers if not _is_app_mcp_server(s)]
    original_by_name = _servers_by_name(picker_servers)

    # Single source (MCP services), so there's no "choose sources" step — discover the fast
    # `system.ai` list, show the picker immediately, and let the workspace-wide walk stream in
    # behind it via the background loader so the picker never blocks on it.
    discovered = _discover_selected_mcp_sources(workspace, profile, {MCP_SERVICES_SOURCE})
    services_loader = _mcp_services_background_loader(
        workspace, profile, set(original_by_name), additive=append
    )
    selections = prompt_for_mcp_server_choices(
        discovered["external"],
        discovered["genie"],
        discovered["apps"],
        picker_servers,
        discovered["services"],
        discovered["vector_search"],
        discovered["uc_functions"],
        additive=append,
        background_loader=services_loader,
    )
    if selections is None or isinstance(selections, _Back):
        # No back-navigation without the source step; `_Back` can't occur, but keep the guard
        # so the type narrows to a selection list below.
        return 0

    available_app_mcp_servers = discovered["apps"]
    available_genie_mcp_servers = discovered["genie"]
    available_vector_search_servers = discovered["vector_search"]
    available_uc_functions_servers = discovered["uc_functions"]

    working_mcp_servers: list[dict] = list(skills_servers)
    working_names: set[str] = set()
    add_selections: list[str] = []
    for selection in selections:
        if selection.startswith(MCP_ADD_PREFIX):
            add_selections.append(selection.removeprefix(MCP_ADD_PREFIX))
            continue
        original = original_by_name.get(selection)
        if original and selection not in working_names:
            working_mcp_servers.append(original.copy())
            working_names.add(selection)

    for selection in add_selections:
        try:
            entry_name, url = _resolve_mcp_selection(
                selection,
                workspace,
                available_app_mcp_servers,
                available_genie_mcp_servers,
                available_vector_search_servers,
                available_uc_functions_servers,
            )
        except RuntimeError as exc:
            print_warning(f"Skipped MCP selection `{selection}`: {exc}.")
            continue
        if entry_name in working_names:
            continue
        working_mcp_servers.append(
            {
                "name": entry_name,
                "url": url,
                "auth": "proxy",
                "clients": clients,
            }
        )
        working_names.add(entry_name)

    if append:
        working_mcp_servers = _union_missing(original_mcp_servers, working_mcp_servers)

    changed = apply_mcp_server_changes(
        original_mcp_servers,
        working_mcp_servers,
        clients,
        workspace,
        profile,
        use_pat=bool(state.get("use_pat")),
    )
    if changed or original_mcp_servers != working_mcp_servers:
        state["mcp_servers"] = working_mcp_servers
        save_state(state)
        added = sorted(working_names - set(original_by_name))
        # `add` never removes; the union above re-keeps unselected servers.
        removed = [] if append else sorted(set(original_by_name) - working_names)
        print_success(_mcp_change_summary(added, removed, clients))
    elif not selections and not original_mcp_servers:
        # User submitted the picker without toggling anything --> make it clear nothing was selected
        print_note("No MCP servers selected. Press space to toggle an item, then enter to save.")
    return 0


def _mcp_change_summary(added: list[str], removed: list[str], clients: list[str]) -> str:
    """Human-readable one-liner describing what `configure mcp` just saved, e.g.
    `Added 2, removed 1 MCP server across Claude Code, Codex`. Falls back to a
    plain `Saved` when only client bindings changed (no add/remove)."""
    client_names = ", ".join(str(MCP_CLIENTS[c]["display"]) for c in clients if c in MCP_CLIENTS)
    parts: list[str] = []
    if added:
        parts.append(f"added {len(added)}")
    if removed:
        parts.append(f"removed {len(removed)}")
    if not parts:
        return "Saved"
    total = len(added) + len(removed)
    noun = "MCP server" if total == 1 else "MCP servers"
    summary = ", ".join(parts).capitalize()
    return f"{summary} {noun} across {client_names}" if client_names else f"{summary} {noun}"


def _prompt_for_mcp_removal(servers: list[dict]) -> list[str] | None:
    """Checklist of already-configured MCP servers to remove. Each item shows the
    registered name and the tools it's currently on. Returns the selected server
    names, ``None`` if cancelled (Ctrl-C), or ``[]`` if nothing was checked."""
    choices: list[questionary.Choice | questionary.Separator] = []
    for server in servers:
        name = _server_name(server)
        if not name:
            continue
        on_clients = [str(MCP_CLIENTS[c]["display"]) for c in _mcp_server_clients(server)]
        title = f"{name} ({', '.join(on_clients)})" if on_clients else name
        choices.append(questionary.Choice(title=title, value=name, checked=False))
    if not choices:
        return []
    selection = _scrolling_checkbox(
        "Remove MCP:",
        choices=choices,
        style=_picker_style(),
        instruction="(space to toggle, ctrl-a all, enter to remove, type to filter)",
    ).ask()
    if selection is None:
        return None
    return [str(value) for value in selection]


def remove_mcp_command(agents: set[str] | None = None) -> int:
    """`ucode mcp remove`: interactively unregister configured MCP servers.

    Shows the servers currently configured (skills connections excluded — they're
    owned by `configure skills`) and removes the ones you select. It never adds or
    reconfigures anything, and needs no Databricks auth.

    Without ``agents``, a selected server is removed from every coding tool it's
    registered on. With ``agents`` (from ``--agents``), removal is scoped to those
    agents: a server registered on several agents is unregistered only from the
    named ones and kept on the rest; only servers registered on a named agent are
    offered."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(
        state, "Remove MCP Servers", require_auth=False, action_note="Removing from", agents=agents
    )

    original_mcp_servers = list(state.get("mcp_servers") or [])
    removable = [
        s
        for s in original_mcp_servers
        if s.get("kind") != SKILLS_MCP_KIND
        and (agents is None or bool(set(_mcp_server_clients(s)) & agents))
    ]
    if not removable:
        scope = "" if agents is None else f" for {', '.join(sorted(agents))}"
        print_note(f"No MCP servers are configured to remove{scope}.")
        return 0

    selection = _prompt_for_mcp_removal(removable)
    if selection is None:
        return 0
    if not selection:
        print_note("No MCP servers selected.")
        return 0
    remove_names = set(selection)

    # Present each removed server to `apply_mcp_server_changes` with its client list
    # narrowed to just the agents we're removing from (all recorded clients when
    # `agents` is None), and drop it from the working list — so the machinery
    # unregisters it from exactly those agents and no others.
    removal_view: list[dict] = []
    for server in original_mcp_servers:
        name = _server_name(server)
        if name not in remove_names:
            continue
        recorded = _mcp_server_clients(server)
        targets = recorded if agents is None else [c for c in recorded if c in agents]
        if targets:
            removal_view.append({**server, "clients": targets})
    changed = apply_mcp_server_changes(
        removal_view, [], clients, workspace, profile, use_pat=bool(state.get("use_pat"))
    )

    # Update saved state: drop a fully-removed server, or keep it with the named
    # agents stripped from its client list when the removal was agent-scoped.
    new_servers: list[dict] = []
    for server in original_mcp_servers:
        name = _server_name(server)
        if name not in remove_names:
            new_servers.append(server)
            continue
        remaining = (
            [] if agents is None else [c for c in (server.get("clients") or []) if c not in agents]
        )
        if remaining:
            new_servers.append({**server, "clients": remaining})

    if changed or new_servers != original_mcp_servers:
        state["mcp_servers"] = new_servers
        save_state(state)
        print_success(_mcp_change_summary([], sorted(remove_names), clients))
    return 0


def _merge_clients(prior: list[str] | None, new: list[str]) -> list[str]:
    """Order-preserving union of a prior client list with newly-configured ones."""
    prior = list(prior or [])
    return prior + [c for c in new if c not in prior]


def _build_skills_entry(workspace: str, locations: list[str], clients: list[str]) -> dict:
    """Canonical single skills-registry entry. ``skill_locations`` is the source
    of truth; the URL is always derived from it, never parsed back."""
    return {
        "name": SKILLS_MCP_SERVER_NAME,
        "kind": SKILLS_MCP_KIND,
        "skill_locations": list(locations),
        "url": build_skills_mcp_url(workspace, locations),
        "auth": "proxy",
        "clients": clients,
    }


def _resolve_skills_mcp_servers(
    workspace: str,
    clients: list[str],
    locations: list[str],
    original_servers: list[dict],
) -> list[dict]:
    """Rebuild the MCP server list around exactly one skills entry.

    Drops every prior ``kind=="skills"`` entry and any entry named
    ``SKILLS_MCP_SERVER_NAME`` (single-connection invariant; also sweeps up a
    stray old-named entry via ``apply_mcp_server_changes``), keeps everything
    else, and appends one rebuilt entry whose clients merge the prior skills
    entry's clients with ``clients``.
    """
    prior = next((s for s in original_servers if s.get("kind") == SKILLS_MCP_KIND), None)
    merged = _merge_clients((prior or {}).get("clients"), clients)
    kept = [
        s
        for s in original_servers
        if s.get("kind") != SKILLS_MCP_KIND and _server_name(s) != SKILLS_MCP_SERVER_NAME
    ]
    return [*kept, _build_skills_entry(workspace, locations, merged)]


def _join_with_and(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def _skills_tools_description(locations: list[str]) -> str:
    if not locations:
        return "UC skill utility tools"
    return f"UC skill utility tools + skills tools in schema {_join_with_and(locations)}"


def _print_skills_summary(entry: dict) -> None:
    """Report the registered skills connection and how to start using it."""
    clients = [
        str(MCP_CLIENTS[client]["display"])
        for client in (entry.get("clients") or [])
        if client in MCP_CLIENTS
    ]
    console.print()
    print_success("Skills MCP registered")
    print_kv("Server", str(entry.get("name") or SKILLS_MCP_SERVER_NAME))
    print_kv("URL", str(entry.get("url") or ""))
    print_kv("Configured", ", ".join(clients) if clients else "none")
    print_kv("Tools", _skills_tools_description(entry.get("skill_locations") or []))
    print_note(
        "Run `ucode <agent>` to use the skills MCP. For existing sessions, "
        "restart the agent for the skills to take effect."
    )


def _update_skills_mcp(
    state: dict, workspace: str, profile: str | None, clients: list[str], locations: list[str]
) -> None:
    """Rebuild the single skills connection for ``locations`` and persist it."""
    original = list(state.get("mcp_servers") or [])
    working = _resolve_skills_mcp_servers(workspace, clients, locations, original)
    changed = apply_mcp_server_changes(original, working, clients, workspace, profile)
    if changed or original != working:
        state["mcp_servers"] = working
        save_state(state)
    entry = next(s for s in working if s.get("kind") == SKILLS_MCP_KIND)
    _print_skills_summary(entry)


def configure_skills_mcp_command(locations: list[str]) -> int:
    """Set the skills MCP connection's ``skill_locations`` to exactly ``locations``,
    replacing any previous set."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "Skills MCP")
    _update_skills_mcp(state, workspace, profile, clients, locations)
    return 0


def _skill_mcp_locations(state: dict) -> list[str]:
    """The skills MCP connection's ``skill_locations``, or ``[]`` if none exists."""
    entry = next(iter(_skills_entries(list(state.get("mcp_servers") or []))), None)
    return list((entry or {}).get("skill_locations") or [])


def register_schemaless_skills_connection(
    state: dict, workspace: str, profile: str | None, clients: list[str]
) -> None:
    """Register/keep the skills MCP connection without changing its schema set.

    Download mode calls this after writing files: it preserves any prior
    ``--mcp`` ``skill_locations`` and otherwise registers the bare schema-less
    route (utility tools only)."""
    _update_skills_mcp(state, workspace, profile, clients, _skill_mcp_locations(state))


def _union_locations(base: list[str], new: list[str]) -> list[str]:
    have = set(base)
    merged = list(base)
    for location in new:
        if location not in have:
            merged.append(location)
            have.add(location)
    return merged


def add_skills_command(locations: list[str]) -> int:
    """Add ``locations`` to the skills MCP connection's scope, keeping any already configured."""
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "Add Skills MCP")
    merged = _union_locations(_skill_mcp_locations(state), locations)
    _update_skills_mcp(state, workspace, profile, clients, merged)
    return 0
