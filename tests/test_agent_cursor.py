"""Tests for the Cursor MCP-only agent integration."""

from __future__ import annotations

import json

from ucode.agents import cursor

WS = "https://example.databricks.com"
# The proxy argv ucode registers for every MCP server (see build_mcp_proxy_argv).
PROXY_ARGV = ["ucode", "mcp-proxy", "--url", f"{WS}/api/2.0/mcp/functions/system/ai"]


class TestMcpServerEntry:
    def test_builds_stdio_command_entry_from_proxy_argv(self):
        entry = cursor.build_mcp_server_entry(PROXY_ARGV)
        # Cursor's stdio schema splits argv into command + args; no token here —
        # the proxy authenticates itself.
        assert entry == {"command": "ucode", "args": PROXY_ARGV[1:]}
        assert "headers" not in entry


class TestWriteMcpServerConfig:
    def test_writes_without_clobbering_existing_entries(self, tmp_path, monkeypatch):
        config_file = tmp_path / "mcp.json"
        monkeypatch.setattr(cursor, "CURSOR_MCP_CONFIG_PATH", config_file)
        # A pre-existing Cursor config (e.g. the user's own proxy entry).
        config_file.write_text(
            json.dumps({"mcpServers": {"proxy": {"command": "mcp", "args": ["start"]}}}),
            encoding="utf-8",
        )

        removed = cursor.write_mcp_server_config("databricks-system-ai", PROXY_ARGV)

        written = json.loads(config_file.read_text())
        assert removed is False
        assert written["mcpServers"]["proxy"] == {"command": "mcp", "args": ["start"]}
        assert written["mcpServers"]["databricks-system-ai"] == {
            "command": "ucode",
            "args": PROXY_ARGV[1:],
        }

    def test_creates_config_when_absent(self, tmp_path, monkeypatch):
        config_file = tmp_path / "mcp.json"
        monkeypatch.setattr(cursor, "CURSOR_MCP_CONFIG_PATH", config_file)

        removed = cursor.write_mcp_server_config("databricks-sql", PROXY_ARGV)

        assert removed is False
        assert json.loads(config_file.read_text())["mcpServers"]["databricks-sql"]["command"] == (
            "ucode"
        )

    def test_reports_replaced_entry(self, tmp_path, monkeypatch):
        config_file = tmp_path / "mcp.json"
        monkeypatch.setattr(cursor, "CURSOR_MCP_CONFIG_PATH", config_file)
        config_file.write_text(
            json.dumps({"mcpServers": {"databricks-sql": {"command": "old"}}}),
            encoding="utf-8",
        )

        removed = cursor.write_mcp_server_config("databricks-sql", PROXY_ARGV)

        assert removed is True
        assert (
            json.loads(config_file.read_text())["mcpServers"]["databricks-sql"]["args"]
            == (PROXY_ARGV[1:])
        )


AIGW_URL = f"{WS}/ai-gateway/mcp-services/system.ai.github"


class TestHttpMcpServerEntry:
    def test_builds_url_plus_auth_entry_with_pinned_client(self):
        entry = cursor.build_http_mcp_server_entry(AIGW_URL, "cursor-desktop")
        # A direct OAuth HTTP server: url + auth.CLIENT_ID, no stdio command and no
        # static bearer — Cursor drives the OAuth itself. Scopes come from the MCP
        # protected-resource metadata, so none are hard-coded here.
        assert entry == {"url": AIGW_URL, "auth": {"CLIENT_ID": "cursor-desktop"}}
        assert "command" not in entry
        assert "headers" not in entry

    def test_write_merges_and_reports_replacement(self, tmp_path, monkeypatch):
        config_file = tmp_path / "mcp.json"
        monkeypatch.setattr(cursor, "CURSOR_MCP_CONFIG_PATH", config_file)
        config_file.write_text(
            json.dumps({"mcpServers": {"proxy": {"command": "keep"}}}),
            encoding="utf-8",
        )

        removed = cursor.write_http_mcp_server_config("github", AIGW_URL, "cursor-desktop")

        written = json.loads(config_file.read_text())
        assert removed is False
        assert written["mcpServers"]["proxy"] == {"command": "keep"}
        assert written["mcpServers"]["github"] == {
            "url": AIGW_URL,
            "auth": {"CLIENT_ID": "cursor-desktop"},
        }


class TestRemoveMcpServerConfig:
    def test_removes_without_clobbering_others(self, tmp_path, monkeypatch):
        config_file = tmp_path / "mcp.json"
        monkeypatch.setattr(cursor, "CURSOR_MCP_CONFIG_PATH", config_file)
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "databricks-sql": {"command": "ucode"},
                        "proxy": {"command": "keep"},
                    }
                }
            ),
            encoding="utf-8",
        )

        removed = cursor.remove_mcp_server_config("databricks-sql")

        written = json.loads(config_file.read_text())
        assert removed is True
        assert "databricks-sql" not in written["mcpServers"]
        assert written["mcpServers"]["proxy"] == {"command": "keep"}

    def test_returns_false_when_absent(self, tmp_path, monkeypatch):
        config_file = tmp_path / "mcp.json"
        monkeypatch.setattr(cursor, "CURSOR_MCP_CONFIG_PATH", config_file)
        config_file.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

        assert cursor.remove_mcp_server_config("databricks-sql") is False


class TestLaunch:
    def test_execs_cursor_agent_without_token_wiring(self, monkeypatch):
        # No OAUTH_TOKEN is set: the proxy handles auth, so launch is a thin exec.
        execs: list[list[str]] = []
        monkeypatch.setattr(cursor, "exec_or_spawn", lambda argv: execs.append(argv))

        cursor.launch({"workspace": WS}, ["--resume"])

        assert execs == [["cursor-agent", "--resume"]]
