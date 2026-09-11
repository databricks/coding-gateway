"""Cursor agent: registers Databricks MCP servers in ~/.cursor/mcp.json.

Cursor is an MCP-only integration. `cursor-agent` runs models on the user's own
Cursor account and exposes no gateway base URL, so ucode configures no models
for it (it stays out of `agents.__init__._MODULES`). What ucode does is register
Databricks MCP servers in Cursor's config, using the same uniform mechanism as
every other client: a local **stdio** server that runs `ucode mcp-proxy`, which
bridges to the Databricks MCP endpoint and mints a fresh OAuth token per request
(see `ucode.mcp_proxy`). So Cursor needs no token in its config and no launch-
time token export — `cursor-agent` just spawns the proxy like any stdio server.

`cursor-agent` reads `~/.cursor/mcp.json` directly, so entries are merged into
that shared file (preserving anything already there) and removed surgically,
mirroring how Claude/Codex edit their shared config rather than restoring a
whole-file backup.
"""

from __future__ import annotations

from pathlib import Path

from ucode.config_io import read_json_safe, write_json_file
from ucode.launcher import exec_or_spawn

CURSOR_BINARY = "cursor-agent"
CURSOR_CONFIG_DIR = Path.home() / ".cursor"
CURSOR_MCP_CONFIG_PATH = CURSOR_CONFIG_DIR / "mcp.json"


def build_mcp_server_entry(argv: list[str]) -> dict:
    # Cursor's stdio MCP schema: `command` + `args`. ucode registers the
    # `ucode mcp-proxy ...` bridge here so the proxy handles auth/refresh.
    return {
        "command": argv[0],
        "args": list(argv[1:]),
    }


def write_mcp_server_config(name: str, argv: list[str]) -> bool:
    """Add (or replace) one MCP server entry in ~/.cursor/mcp.json.

    Merges into the existing `mcpServers` map so unrelated entries the user
    already configured survive. Returns True when an entry with this name was
    already present (i.e. this was a replacement)."""
    existing = read_json_safe(CURSOR_MCP_CONFIG_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_mcp_server_entry(argv)
    existing["mcpServers"] = mcp_servers
    write_json_file(CURSOR_MCP_CONFIG_PATH, existing)
    return removed


def build_http_mcp_server_entry(url: str, client_id: str) -> dict:
    # Cursor's remote-MCP-with-OAuth schema: a `url` server plus an `auth` object
    # naming a pre-registered OAuth client. Cursor drives the OAuth itself (to its
    # fixed `http://localhost:8787/callback` redirect) instead of the stdio proxy,
    # so the user gets Cursor's native connection login. Scopes are omitted — the
    # MCP protected-resource metadata advertises them, and Cursor discovers them
    # (the same way Claude Code's `--transport http --client-id` flow does).
    return {
        "url": url,
        "auth": {"CLIENT_ID": client_id},
    }


def write_http_mcp_server_config(name: str, url: str, client_id: str) -> bool:
    """Add (or replace) one **OAuth HTTP** MCP server entry in ~/.cursor/mcp.json.

    Used for connection-backed AI Gateway services when the workspace has Cursor's
    OAuth client published: Cursor authenticates directly rather than going through
    the token-injecting stdio proxy. Merges into `mcpServers` like the stdio path;
    returns True when an entry with this name was already present."""
    existing = read_json_safe(CURSOR_MCP_CONFIG_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_http_mcp_server_entry(url, client_id)
    existing["mcpServers"] = mcp_servers
    write_json_file(CURSOR_MCP_CONFIG_PATH, existing)
    return removed


def remove_mcp_server_config(name: str) -> bool:
    """Surgically remove one MCP server entry from ~/.cursor/mcp.json.

    Returns True when an entry was removed, False when it wasn't present."""
    existing = read_json_safe(CURSOR_MCP_CONFIG_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict) or name not in mcp_servers:
        return False
    mcp_servers.pop(name)
    existing["mcpServers"] = mcp_servers
    write_json_file(CURSOR_MCP_CONFIG_PATH, existing)
    return True


def launch(state: dict, tool_args: list[str]) -> None:
    """Hand the terminal to `cursor-agent`.

    No token wiring here: the Databricks MCP servers in ~/.cursor/mcp.json run
    `ucode mcp-proxy`, which authenticates itself, so `ucode cursor` is a thin
    convenience wrapper over `cursor-agent` (kept for symmetry with the other
    `ucode <agent>` launchers)."""
    exec_or_spawn([CURSOR_BINARY, *tool_args])
