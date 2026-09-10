"""Tests for MCP server registration."""

from __future__ import annotations

import json
import subprocess
import threading
from unittest.mock import MagicMock

import pytest

from ucode import mcp

WS = "https://example.databricks.com"
CLAUDE_STATE = {"workspace": WS, "available_tools": ["claude"]}
ALL_MCP_CLIENTS = ["claude", "codex", "gemini", "opencode", "copilot"]


class TestMcpChangeSummary:
    def test_added_and_removed_with_clients(self):
        summary = mcp._mcp_change_summary(["a", "b"], ["c"], ["claude", "codex"])
        assert summary == "Added 2, removed 1 MCP servers across Claude Code, Codex"

    def test_single_add_uses_singular_noun(self):
        summary = mcp._mcp_change_summary(["a"], [], ["claude"])
        assert summary == "Added 1 MCP server across Claude Code"

    def test_only_removed(self):
        summary = mcp._mcp_change_summary([], ["a"], ["claude"])
        assert summary == "Removed 1 MCP server across Claude Code"

    def test_no_changes_falls_back_to_saved(self):
        assert mcp._mcp_change_summary([], [], ["claude"]) == "Saved"


# The proxy argv every client registers as a stdio command. The leading element
# is the resolved `ucode` binary path, so tests assert the tail (the stable part).
GH_URL = f"{WS}/api/2.0/mcp/external/github"
PROXY_TAIL = ["mcp-proxy", "--url", GH_URL, "--host", WS, "--profile", "p"]


def _unwrap(text: str) -> str:
    """Collapse rich's line-wrapping so assertions match regardless of terminal width."""
    return " ".join(text.split())


def _proxy_argv() -> list[str]:
    from ucode.databricks import build_mcp_proxy_argv

    return build_mcp_proxy_argv(GH_URL, WS, "p")


class TestBuildMcpProxyArgv:
    def test_argv_is_ucode_mcp_proxy_command(self):
        argv = _proxy_argv()
        # First element is the resolved ucode binary; the rest is stable.
        assert argv[1:] == PROXY_TAIL
        assert argv[0].endswith("ucode") or argv[0] == "ucode"

    def test_use_pat_appends_flag_and_profile_optional(self):
        from ucode.databricks import build_mcp_proxy_argv

        with_pat = build_mcp_proxy_argv(GH_URL, WS, "p", use_pat=True)
        assert with_pat[-1] == "--use-pat"
        no_profile = build_mcp_proxy_argv(GH_URL, WS, None)
        assert "--profile" not in no_profile


class TestAddClaudeMcpServer:
    def test_registers_stdio_proxy_command(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_claude_mcp_server("github", _proxy_argv())

        args = calls[0]["args"]
        assert args[:4] == ["claude", "mcp", "add", "github"]
        assert args[4:6] == ["-s", "user"]
        # `--` fences the proxy argv; everything after it is the stdio command.
        assert args[6] == "--"
        assert args[7:] == _proxy_argv()

    def test_always_load_routes_through_add_json_stdio_entry(self, monkeypatch):
        # The skills registry needs `alwaysLoad: true`, which plain `mcp add`
        # can't set — so the proxy argv is wrapped in a stdio entry dict and
        # registered via add-json instead.
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_claude_mcp_server("skills", _proxy_argv(), always_load=True)

        args = calls[0]["args"]
        assert args[:4] == ["claude", "mcp", "add-json", "skills"]
        entry = json.loads(args[4])
        assert entry == {
            "type": "stdio",
            "command": _proxy_argv()[0],
            "args": _proxy_argv()[1:],
            "alwaysLoad": True,
        }
        assert args[5:] == ["-s", "user"]

    def test_dict_entry_routes_through_add_json(self, monkeypatch):
        # The web_search server (agents/claude.py) registers a full stdio entry
        # dict with its own env, which only `add-json` can express — a dict must
        # route there rather than through the proxy `mcp add -- <argv>` path.
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        entry = {"type": "stdio", "command": "ucode", "args": ["mcp", "web-search"]}
        mcp.add_claude_mcp_server("web_search", entry)

        args = calls[0]["args"]
        assert args[:4] == ["claude", "mcp", "add-json", "web_search"]
        assert json.loads(args[4]) == entry
        assert args[5:] == ["-s", "user"]


class TestAddCodexMcpServer:
    def test_registers_stdio_proxy_command(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_codex_mcp_server("github", _proxy_argv())

        args = calls[0]["args"]
        assert args[:4] == ["codex", "mcp", "add", "github"]
        assert args[4] == "--"
        assert args[5:] == _proxy_argv()


class TestAddGeminiMcpServer:
    def test_registers_stdio_proxy_command(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        mcp.add_gemini_mcp_server("github", _proxy_argv())

        call = calls[0]
        args = call["args"]
        assert args[:4] == ["gemini", "mcp", "add", "github"]
        # command + args, then the transport/scope flags.
        assert args[4 : 4 + len(_proxy_argv())] == _proxy_argv()
        assert args[-4:] == ["--type", "stdio", "--scope", "user"]
        # GEMINI_CLI_HOME must point at the launcher's home so `gemini mcp add`
        # writes the same settings.json the ucode session reads from.
        assert call["kwargs"]["env"]["GEMINI_CLI_HOME"] == str(mcp.gemini.GEMINI_HOME_DIR)


class TestRemoveClaudeMcpServer:
    def test_returns_true_when_server_removed(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return MagicMock(returncode=0)

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is True
        assert calls == [["claude", "mcp", "remove", "github", "-s", "user"]]

    def test_returns_false_when_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr="No MCP server named github found")

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is False

    def test_returns_false_when_project_local_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                args,
                stderr="No project-local MCP server found with name: github",
            )

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "project") is False

    def test_returns_false_when_user_scoped_server_missing(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(
                1,
                args,
                stderr="No user-scoped MCP server found with name: github",
            )

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        assert mcp.remove_claude_mcp_server("github", "user") is False

    def test_unexpected_failure_raises(self, monkeypatch):
        def fake_run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args, stderr="permission denied")

        monkeypatch.setattr(mcp.subprocess, "run", fake_run)

        try:
            mcp.remove_claude_mcp_server("github", "user")
        except RuntimeError as exc:
            assert "Failed to remove MCP server 'github'" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


class TestExternalMcpConnectionNames:
    def test_returns_sorted_http_connection_names(self):
        assert mcp.external_mcp_connection_names(
            [
                {"name": "jira-mcp", "connection_type": "HTTP"},
                {"name": "not-http", "connection_type": "POSTGRESQL"},
                {"name": "confluence-mcp", "connection_type": "http"},
                {"name": "jira-mcp", "connection_type": "HTTP"},
            ]
        ) == ["confluence-mcp", "jira-mcp"]

    def test_excludes_explicit_non_mcp_http_connections(self):
        assert mcp.external_mcp_connection_names(
            [
                {
                    "name": "analytics-api",
                    "connection_type": "HTTP",
                    "options": {"is_mcp": "false"},
                },
                {"name": "github-mcp", "connection_type": "HTTP", "options": {"is_mcp": "true"}},
            ]
        ) == ["github-mcp"]


class TestCursorMcpClient:
    def test_cursor_registered_as_mcp_only_client(self):
        assert "cursor" in mcp.MCP_CLIENTS
        assert mcp.MCP_CLIENTS["cursor"]["binary"] == "cursor-agent"
        assert "cursor" in mcp.MCP_ONLY_CLIENTS

    def test_configure_dispatches_proxy_argv_to_cursor_writer(self, monkeypatch):
        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(
            mcp.cursor,
            "write_mcp_server_config",
            lambda name, argv: calls.append((name, argv)) or False,
        )

        removed_scopes = mcp.configure_client_mcp_server("cursor", "github", GH_URL, WS, "p")

        assert removed_scopes == []
        assert calls == [("github", _proxy_argv())]

    def test_configure_reports_user_scope_on_replace(self, monkeypatch):
        monkeypatch.setattr(mcp.cursor, "write_mcp_server_config", lambda name, argv: True)
        assert mcp.configure_client_mcp_server("cursor", "github", GH_URL, WS, "p") == [
            mcp.MCP_USER_SCOPE
        ]

    def test_remove_dispatches_to_cursor_remover(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            mcp.cursor, "remove_mcp_server_config", lambda name: calls.append(name) or True
        )
        assert mcp.remove_client_mcp_server("cursor", "github-mcp") == [mcp.MCP_USER_SCOPE]
        assert calls == ["github-mcp"]

    def test_eligible_when_installed_even_without_configured_tools(self):
        # Cursor is MCP-only: it never appears in available_tools, so eligibility
        # rests on the binary being installed (MCP_ONLY_CLIENTS).
        clients = mcp.configured_mcp_clients({"available_tools": ["claude"]}, ["claude", "cursor"])
        assert "cursor" in clients

    def test_skipped_when_binary_not_installed(self):
        clients = mcp.configured_mcp_clients({"available_tools": ["claude"]}, ["claude"])
        assert "cursor" not in clients


class TestConfigureClientMcpServer:
    def test_configures_copilot_with_proxy_argv(self, monkeypatch):
        calls: list[tuple[str, list[str]]] = []

        monkeypatch.setattr(
            mcp.copilot,
            "write_mcp_server_config",
            lambda name, argv: calls.append((name, argv)) or False,
        )

        removed_scopes = mcp.configure_client_mcp_server("copilot", "github", GH_URL, WS, "p")

        assert removed_scopes == []
        # Copilot receives the proxy argv, not a URL/bearer entry.
        assert calls == [("github", _proxy_argv())]


class TestMcpPicker:
    def test_prompt_uses_scrolling_checkbox_selector(self, monkeypatch):
        checkbox_calls: list[dict] = []

        class FakePrompt:
            def ask(self):
                return [f"{mcp.MCP_ADD_PREFIX}external:github-mcp"]

        def fake_checkbox(*args, **kwargs):
            checkbox_calls.append({"args": args, "kwargs": kwargs})
            return FakePrompt()

        monkeypatch.setattr(mcp, "_scrolling_checkbox", fake_checkbox)

        assert mcp.prompt_for_mcp_server_choices(["github-mcp"], [], [], []) == [
            f"{mcp.MCP_ADD_PREFIX}external:github-mcp"
        ]

        assert checkbox_calls
        choices = checkbox_calls[0]["kwargs"]["choices"]
        choice_text = [choice.title for choice in choices]
        assert "External connections" not in choice_text
        assert "Databricks managed services" not in choice_text
        assert "Custom servers" not in choice_text
        assert choice_text == [
            "Databricks SQL",
            "Connection: github-mcp",
        ]
        assert "Built-in AI tools" not in choice_text
        assert checkbox_calls[0]["kwargs"]["instruction"] == (
            "(space to toggle, ctrl-a all, enter to save, type to filter)"
        )

    def test_prompt_returns_none_when_cancelled(self, monkeypatch):
        class FakePrompt:
            def ask(self):
                return None

        monkeypatch.setattr(mcp, "_scrolling_checkbox", lambda *args, **kwargs: FakePrompt())

        assert mcp.prompt_for_mcp_server_choices(["github-mcp"], [], [], []) is None

    def test_picker_marks_configured_servers(self):
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [],
            [],
            [{"name": "github-mcp", "url": f"{WS}/api/2.0/mcp/external/github-mcp"}],
        )
        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["Connection: github-mcp"].checked is True
        assert choices_by_title["Databricks SQL"].checked is False

    def test_additive_picker_shows_configured_servers_as_disabled(self):
        """In `ucode mcp add` mode an already-configured server can't be removed, so
        it's shown as a non-toggleable note rather than a pre-checked box."""
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [],
            [],
            [{"name": "github-mcp", "url": f"{WS}/api/2.0/mcp/external/github-mcp"}],
            additive=True,
        )
        choices_by_title = {choice.title: choice for choice in choices}
        configured = choices_by_title["Connection: github-mcp"]
        assert configured.disabled == "already configured"
        assert configured.checked is False
        # A not-yet-configured server stays an addable, toggleable choice.
        assert choices_by_title["Databricks SQL"].disabled is None

    def test_removal_picker_lists_configured_servers_with_their_clients(self, monkeypatch):
        checkbox_calls: list[dict] = []

        class FakePrompt:
            def ask(self):
                return ["system-ai-github"]

        def fake_checkbox(*args, **kwargs):
            checkbox_calls.append(kwargs)
            return FakePrompt()

        monkeypatch.setattr(mcp, "_scrolling_checkbox", fake_checkbox)

        result = mcp._prompt_for_mcp_removal(
            [
                {"name": "system-ai-github", "url": f"{WS}/x", "clients": ["claude", "codex"]},
                {"name": "", "url": f"{WS}/y", "clients": ["claude"]},  # unnamed: skipped
            ]
        )

        assert result == ["system-ai-github"]
        choices = checkbox_calls[0]["choices"]
        # The unnamed server is skipped; the named one shows the tools it's on.
        assert [c.title for c in choices] == ["system-ai-github (Claude Code, Codex)"]
        assert [c.value for c in choices] == ["system-ai-github"]
        assert all(c.checked is False for c in choices)

    def test_picker_keeps_databricks_sql_when_nothing_discovered(self):
        choices = mcp.build_mcp_picker_choices([], [], [], [])
        assert [choice.title for choice in choices] == ["Databricks SQL"]
        assert choices[0].value == f"{mcp.MCP_ADD_PREFIX}managed:sql"

    def test_discovers_genie_spaces_as_mcp_servers(self):
        assert mcp.genie_mcp_servers(
            [
                {"space_id": "space-2", "title": "Second Space"},
                {"space_id": "space-1", "title": "First Space"},
                {"title": "Missing ID"},
            ],
            WS,
        ) == [
            {
                "name": "databricks-genie-first-space",
                "title": "First Space",
                "url": f"{WS}/api/2.0/mcp/genie/space-1",
            },
            {
                "name": "databricks-genie-second-space",
                "title": "Second Space",
                "url": f"{WS}/api/2.0/mcp/genie/space-2",
            },
        ]

    def test_genie_server_name_falls_back_to_space_id_on_slug_collision(self):
        assert mcp.genie_mcp_servers(
            [
                {"space_id": "space-1", "title": "New Space"},
                {"space_id": "space-2", "title": "new space"},
                {"space_id": "space-3", "title": ""},
            ],
            WS,
        ) == [
            {
                "name": "databricks-genie-new-space",
                "title": "New Space",
                "url": f"{WS}/api/2.0/mcp/genie/space-1",
            },
            {
                "name": "databricks-genie-space-2",
                "title": "new space",
                "url": f"{WS}/api/2.0/mcp/genie/space-2",
            },
            {
                "name": "databricks-genie-space-3",
                "title": "space-3",
                "url": f"{WS}/api/2.0/mcp/genie/space-3",
            },
        ]

    def test_picker_lists_discovered_genie_spaces(self):
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [
                {
                    "name": "databricks-genie-space-1",
                    "title": "First Space",
                    "url": f"{WS}/api/2.0/mcp/genie/space-1",
                }
            ],
            [],
            [],
        )

        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["Genie: First Space"].value == (
            f"{mcp.MCP_ADD_PREFIX}genie-space:space-1"
        )

    def test_discovers_apps_as_mcp_servers(self):
        assert mcp.app_mcp_servers(
            [
                {
                    "name": "mcp-my-app",
                    "url": "https://mcp-my-app.example.databricksapps.com",
                },
                {
                    "name": "regular-app",
                    "url": "https://regular-app.example.databricksapps.com",
                },
                {"name": "missing-url"},
            ]
        ) == [
            {
                "name": "databricks-app-mcp-my-app",
                "title": "mcp-my-app",
                "url": "https://mcp-my-app.example.databricksapps.com/mcp",
            }
        ]

    def test_picker_lists_discovered_app_mcps(self):
        choices = mcp.build_mcp_picker_choices(
            ["github-mcp"],
            [],
            [
                {
                    "name": "databricks-app-mcp-my-app",
                    "title": "mcp-my-app",
                    "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                }
            ],
            [],
        )

        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["App: mcp-my-app"].value == f"{mcp.MCP_ADD_PREFIX}app:mcp-my-app"

    def test_vector_search_mcp_servers_emit_managed_url_per_pair(self):
        servers = mcp.vector_search_mcp_servers(
            [("main", "search"), ("Marketing", "Docs")],
            WS,
        )
        assert servers == [
            {
                "name": "databricks-vector-search-main-search",
                "title": "main.search",
                "catalog": "main",
                "schema": "search",
                "url": f"{WS}/api/2.0/mcp/vector-search/main/search",
            },
            {
                "name": "databricks-vector-search-marketing-docs",
                "title": "Marketing.Docs",
                "catalog": "Marketing",
                "schema": "Docs",
                "url": f"{WS}/api/2.0/mcp/vector-search/Marketing/Docs",
            },
        ]

    def test_uc_functions_mcp_servers_emit_managed_url_per_pair(self):
        servers = mcp.uc_functions_mcp_servers(
            [("analytics", "tools"), ("ml", "udfs")],
            WS,
        )
        assert servers == [
            {
                "name": "databricks-functions-analytics-tools",
                "title": "analytics.tools",
                "catalog": "analytics",
                "schema": "tools",
                "url": f"{WS}/api/2.0/mcp/functions/analytics/tools",
            },
            {
                "name": "databricks-functions-ml-udfs",
                "title": "ml.udfs",
                "catalog": "ml",
                "schema": "udfs",
                "url": f"{WS}/api/2.0/mcp/functions/ml/udfs",
            },
        ]

    def test_picker_lists_discovered_vector_search_and_uc_functions(self):
        choices = mcp.build_mcp_picker_choices(
            [],
            [],
            [],
            [],
            available_vector_search_servers=mcp.vector_search_mcp_servers([("main", "search")], WS),
            available_uc_functions_servers=mcp.uc_functions_mcp_servers(
                [("analytics", "tools")], WS
            ),
        )
        choices_by_title = {choice.title: choice for choice in choices}
        assert (
            choices_by_title["Vector Search: main.search"].value
            == f"{mcp.MCP_ADD_PREFIX}vector-search:main.search"
        )
        assert (
            choices_by_title["UC Functions: analytics.tools"].value
            == f"{mcp.MCP_ADD_PREFIX}uc-functions:analytics.tools"
        )

    def test_picker_keeps_saved_legacy_servers_for_removal(self):
        choices = mcp.build_mcp_picker_choices(
            [],
            [],
            [],
            [
                {
                    "name": "databricks-vector-search-main-search-docs",
                    "url": f"{WS}/api/2.0/mcp/vector-search/main/search/docs",
                }
            ],
        )

        choices_by_title = {choice.title: choice for choice in choices}
        assert choices_by_title["databricks-vector-search-main-search-docs"].checked is True


def _patch_mcp_choices(monkeypatch, *values: str, categories: set[str] | None = None) -> None:
    monkeypatch.setattr(
        mcp,
        "prompt_for_mcp_server_choices",
        lambda *args, **kwargs: list(values),
    )
    # The first wizard step chooses which sources to search. Default to the
    # fast pre-checked ones (external, apps, MCP services, genie); tests that
    # exercise the slow walks (vector-search / uc-functions) pass those keys via
    # `categories`, which are unioned in.
    default_sources = {"external", "apps", "mcp-services", "genie"}
    selected_sources = default_sources | (categories or set())
    monkeypatch.setattr(
        mcp, "prompt_for_mcp_search_sources", lambda exclude_sources=None: selected_sources
    )
    # Stub the always-on discoveries so configure_mcp_command tests don't hit
    # real APIs. Individual tests override these after calling the helper.
    monkeypatch.setattr(mcp, "discover_mcp_service_names", lambda workspace, profile=None: [])
    monkeypatch.setattr(
        mcp,
        "discover_all_mcp_service_names",
        lambda workspace, profile=None, on_progress=None: [],
    )
    monkeypatch.setattr(
        mcp,
        "discover_vector_search_mcp_servers",
        lambda workspace, profile=None, on_progress=None: [],
    )
    monkeypatch.setattr(
        mcp,
        "discover_uc_functions_mcp_servers",
        lambda workspace, profile=None, on_progress=None: [],
    )


class TestApplyMcpServerChanges:
    def _server(self, name, clients):
        return {"name": name, "url": f"{WS}/api/2.0/mcp/external/{name}", "clients": clients}

    def test_adds_across_clients_serial_within_client(self, monkeypatch):
        # Record (client, name) for every add. Each client's ops must stay in
        # order even though clients run concurrently.
        recorded: list[tuple[str, str]] = []
        lock = threading.Lock()

        def fake_configure(client, name, url, *a, **kw):
            with lock:
                recorded.append((client, name))
            return []

        monkeypatch.setattr(mcp, "configure_client_mcp_server", fake_configure)

        working = [
            self._server("a", ["claude", "codex"]),
            self._server("b", ["claude", "codex"]),
            self._server("c", ["claude", "codex"]),
        ]

        changed = mcp.apply_mcp_server_changes([], working, ["claude", "codex"], WS)

        assert changed is True
        # 3 servers x 2 clients = 6 operations.
        assert len(recorded) == 6
        # Within each client, the server order is preserved.
        assert [n for c, n in recorded if c == "claude"] == ["a", "b", "c"]
        assert [n for c, n in recorded if c == "codex"] == ["a", "b", "c"]

    def test_removes_servers_dropped_from_working_set(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **k: [])

        original = [self._server("gone", ["claude"])]

        changed = mcp.apply_mcp_server_changes(original, [], ["claude"], WS)

        assert changed is True
        assert removed == [("claude", "gone")]

    def test_no_ops_returns_false_without_spinner(self, monkeypatch):
        # Identical original/working, so nothing to do.
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda *a, **k: pytest.fail("should not configure"),
        )
        servers = [self._server("a", ["claude"])]

        assert mcp.apply_mcp_server_changes(servers, servers, ["claude"], WS) is False


class TestConfigureMcpWizardNavigation:
    def test_back_reshows_source_screen(self, monkeypatch):
        """Pressing ← in the picker (returns _BACK) re-runs the source screen,
        then the picker again; a real selection on the second pass proceeds."""
        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_mcp_service_names", lambda workspace, profile=None: [])
        monkeypatch.setattr(
            mcp,
            "discover_all_mcp_service_names",
            lambda workspace, profile=None, on_progress=None: [],
        )

        source_calls: list[int] = []

        def fake_sources(exclude_sources=None):
            source_calls.append(1)
            return {"external", "apps", "mcp-services", "genie"}

        monkeypatch.setattr(mcp, "prompt_for_mcp_search_sources", fake_sources)

        # First picker press = back, second = submit nothing.
        picker_results = [mcp._BACK, []]
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_choices",
            lambda *a, **k: picker_results.pop(0),
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        assert mcp.configure_mcp_command() == 0
        # Source screen shown twice (initial + after back).
        assert len(source_calls) == 2

    def test_cancel_on_source_screen_exits(self, monkeypatch):
        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        # Cancelling the first screen (None) returns without discovering anything.
        monkeypatch.setattr(mcp, "prompt_for_mcp_search_sources", lambda exclude_sources=None: None)
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_choices",
            lambda *a, **k: pytest.fail("picker should not run after cancel"),
        )

        assert mcp.configure_mcp_command() == 0


class TestConfigureMcpCommand:
    def test_skips_existing_server_state_by_name(self, monkeypatch):
        saved_states: list[dict] = []
        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "available_tools": ["claude"],
                "mcp_servers": [
                    {
                        "name": "github",
                        "url": f"{WS}/old",
                        "client_id": "old-client-id",
                        "client_secret": "old-client-secret",
                    }
                ],
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, "github")
        monkeypatch.setattr(mcp, "remove_claude_mcp_server", lambda name, scope: False)
        monkeypatch.setattr(mcp, "add_claude_mcp_server", lambda name, entry, scope: None)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert saved_states == []

    def test_registers_discovered_external_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {"workspace": WS, "available_tools": ALL_MCP_CLIENTS},
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(
            mcp,
            "available_mcp_clients",
            lambda: ALL_MCP_CLIENTS,
        )
        monkeypatch.setattr(
            mcp,
            "discover_external_mcp_connection_names",
            lambda workspace, profile=None: ["confluence-mcp", "github-mcp"],
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}external:github-mcp")

        def fake_configure_client_mcp_server(client, name, url, *a, **kw):
            configured.append((client, name, url))
            return []

        monkeypatch.setattr(mcp, "configure_client_mcp_server", fake_configure_client_mcp_server)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert sorted(configured) == sorted(
            [
                ("claude", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp"),
                ("codex", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp"),
                ("gemini", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp"),
                ("opencode", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp"),
                ("copilot", "github-mcp", f"{WS}/api/2.0/mcp/external/github-mcp"),
            ]
        )
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "github-mcp",
                "url": f"{WS}/api/2.0/mcp/external/github-mcp",
                "auth": "proxy",
                "clients": ["claude", "codex", "gemini", "opencode", "copilot"],
            }
        ]

    def test_registers_discovered_genie_space_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(
            mcp,
            "discover_genie_mcp_servers",
            lambda workspace, profile=None: [
                {
                    "name": "databricks-genie-space-123",
                    "title": "Sales Genie",
                    "url": f"{WS}/api/2.0/mcp/genie/space-123",
                }
            ],
        )
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(
            monkeypatch, f"{mcp.MCP_ADD_PREFIX}genie-space:space-123", categories={"genie"}
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-genie-space-123",
                f"{WS}/api/2.0/mcp/genie/space-123",
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-genie-space-123",
                "url": f"{WS}/api/2.0/mcp/genie/space-123",
                "auth": "proxy",
                "clients": ["claude"],
            }
        ]

    def test_registers_discovered_vector_search_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(
            monkeypatch,
            f"{mcp.MCP_ADD_PREFIX}{mcp.VECTOR_SEARCH_SELECTION_PREFIX}main.search",
            categories={"vector-search"},
        )
        monkeypatch.setattr(
            mcp,
            "discover_vector_search_mcp_servers",
            lambda workspace, profile=None, on_progress=None: mcp.vector_search_mcp_servers(
                [("main", "search")], workspace
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-vector-search-main-search",
                f"{WS}/api/2.0/mcp/vector-search/main/search",
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-vector-search-main-search",
                "url": f"{WS}/api/2.0/mcp/vector-search/main/search",
                "auth": "proxy",
                "clients": ["claude"],
            }
        ]

    def test_registers_discovered_uc_functions_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(
            monkeypatch,
            f"{mcp.MCP_ADD_PREFIX}{mcp.UC_FUNCTIONS_SELECTION_PREFIX}analytics.tools",
            categories={"uc-functions"},
        )
        monkeypatch.setattr(
            mcp,
            "discover_uc_functions_mcp_servers",
            lambda workspace, profile=None, on_progress=None: mcp.uc_functions_mcp_servers(
                [("analytics", "tools")], workspace
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-functions-analytics-tools",
                f"{WS}/api/2.0/mcp/functions/analytics/tools",
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-functions-analytics-tools",
                "url": f"{WS}/api/2.0/mcp/functions/analytics/tools",
                "auth": "proxy",
                "clients": ["claude"],
            }
        ]

    def test_registers_mcp_service_from_workspace_wide_walk(self, monkeypatch):
        """The workspace-wide walk runs by default and folds its services into
        the picker via the same mcp-service path as the curated system.ai list."""
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str]] = []
        walk_calls: list[str] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(
            monkeypatch,
            f"{mcp.MCP_ADD_PREFIX}{mcp.MCP_SERVICE_SELECTION_PREFIX}mycat.myschema.weather",
        )

        def fake_walk(workspace, profile=None, on_progress=None):
            walk_calls.append(workspace)
            return ["mycat.myschema.weather"]

        monkeypatch.setattr(mcp, "discover_all_mcp_service_names", fake_walk)
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert walk_calls == [WS]
        assert configured == [
            (
                "claude",
                "mycat-myschema-weather",
                f"{WS}/ai-gateway/mcp-services/mycat.myschema.weather",
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "mycat-myschema-weather",
                "url": f"{WS}/ai-gateway/mcp-services/mycat.myschema.weather",
                "auth": "proxy",
                "clients": ["claude"],
            }
        ]

    def test_skips_slow_walks_unless_source_selected(self, monkeypatch):
        """Vector Search and UC functions walk the workspace and are OFF by
        default on the search-sources screen, so their discovery must not run
        unless the user selects them. Genie is a fast default and may run."""
        called: list[str] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        # Default sources only (no vector-search / uc-functions). Set the
        # trackers AFTER `_patch_mcp_choices` since it stubs the same discoveries.
        _patch_mcp_choices(monkeypatch)

        def track(name):
            def _discover(workspace, profile=None, on_progress=None):
                called.append(name)
                return []

            return _discover

        monkeypatch.setattr(mcp, "discover_vector_search_mcp_servers", track("vector-search"))
        monkeypatch.setattr(mcp, "discover_uc_functions_mcp_servers", track("uc-functions"))
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        assert mcp.configure_mcp_command() == 0
        assert called == []

    def test_registers_discovered_app_mcp_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(
            mcp,
            "discover_app_mcp_servers",
            lambda workspace, profile=None: [
                {
                    "name": "databricks-app-mcp-my-app",
                    "title": "mcp-my-app",
                    "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                }
            ],
        )
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}app:mcp-my-app")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-app-mcp-my-app",
                "https://mcp-my-app.example.databricksapps.com/mcp",
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-app-mcp-my-app",
                "url": "https://mcp-my-app.example.databricksapps.com/mcp",
                "auth": "proxy",
                "clients": ["claude"],
            }
        ]

    def test_hints_when_no_selections_and_no_existing_servers(self, monkeypatch, capsys):
        saved_states: list[dict] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "No MCP servers selected" in output
        assert "space to toggle" in output
        assert saved_states == []

    def test_preserves_skills_connection_and_hides_it_from_picker(self, monkeypatch):
        """The picker manages mcp-services only; a skills connection is kept out
        of the choices and never removed. Selecting an mcp-service saves both."""
        saved_states: list[dict] = []
        picker_servers: list[list[dict]] = []
        removed: list[tuple[str, str]] = []
        skills_entry = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["main.default"],
            "url": f"{WS}/ai-gateway/skills/?schema=main.default",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }

        monkeypatch.setattr(
            mcp, "load_state", lambda: {**CLAUDE_STATE, "mcp_servers": [skills_entry]}
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_mcp_service_names", lambda workspace, profile=None: [])
        monkeypatch.setattr(
            mcp, "discover_vector_search_mcp_servers", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(
            mcp, "discover_uc_functions_mcp_servers", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_search_sources",
            lambda exclude_sources=None: {"external", "apps", "mcp-services", "genie"},
        )
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_choices",
            lambda ext, genie, app, servers, *a, **kw: (
                picker_servers.append(servers) or [f"{mcp.MCP_ADD_PREFIX}{mcp.SQL_MCP_VALUE}"]
            ),
        )
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert picker_servers == [[]]
        assert removed == []
        assert skills_entry in saved_states[-1]["mcp_servers"]

    def test_drops_stale_foreign_workspace_mcp_entries(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        cleanup_calls: list[tuple[str, str]] = []
        other_ws = "https://other-workspace.cloud.databricks.com"
        stale_entry = {
            "name": "databricks-genie-foreign",
            "url": f"{other_ws}/api/2.0/mcp/genie/foreign",
            "auth": "proxy",
            "clients": ["claude", "codex"],
        }
        kept_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "proxy",
            "clients": ["claude"],
        }

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "available_tools": ["claude"],
                "mcp_servers": [stale_entry, kept_entry],
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, "databricks-sql")
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: cleanup_calls.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Dropping 1 stale MCP entry" in output
        assert "databricks-genie-foreign" in output
        # codex is listed on the stale entry but not installed -> skipped.
        assert cleanup_calls == [("claude", "databricks-genie-foreign")]
        assert saved_states, "expected sanitized state to be persisted"
        assert saved_states[0]["mcp_servers"] == [kept_entry]

    def test_removes_orphan_mcp_entries_from_other_workspace_buckets(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        cleanup_calls: list[tuple[str, str]] = []
        other_ws = "https://other-workspace.cloud.databricks.com"
        current_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "proxy",
            "clients": ["claude"],
        }
        orphan_entry = {
            "name": "orphan-mcp",
            "url": f"{other_ws}/api/2.0/mcp/external/orphan-mcp",
            "auth": "proxy",
            "clients": ["claude", "codex"],
        }

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "workspace": WS,
                "available_tools": ["claude"],
                "mcp_servers": [current_entry],
            },
        )
        monkeypatch.setattr(
            mcp,
            "load_full_state",
            lambda: {
                "current_workspace": WS,
                "workspaces": {
                    WS: {"mcp_servers": [current_entry]},
                    other_ws: {"mcp_servers": [orphan_entry]},
                },
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, "databricks-sql")
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: cleanup_calls.append((client, name)) or [mcp.MCP_USER_SCOPE],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "left over from previously-configured workspaces" in output
        assert "orphan-mcp" in output
        # codex was in orphan-mcp's clients but isn't installed -> skipped.
        assert cleanup_calls == [("claude", "orphan-mcp")]

    def test_skips_orphan_warning_when_nothing_was_actually_removed(self, monkeypatch, capsys):
        """Re-running configure mcp on the same workspace shouldn't repeat the warning
        if the leftover entries were already removed by a previous run."""
        cleanup_calls: list[tuple[str, str]] = []
        other_ws = "https://other-workspace.cloud.databricks.com"
        orphan_entry = {
            "name": "orphan-mcp",
            "url": f"{other_ws}/api/2.0/mcp/external/orphan-mcp",
            "auth": "proxy",
            "clients": ["claude"],
        }

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {"workspace": WS, "available_tools": ["claude"]},
        )
        monkeypatch.setattr(
            mcp,
            "load_full_state",
            lambda: {
                "current_workspace": WS,
                "workspaces": {
                    WS: {},
                    other_ws: {"mcp_servers": [orphan_entry]},
                },
            },
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch)
        # Stub returns empty list -> "entry wasn't in this agent's config".
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: cleanup_calls.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "left over from previously-configured workspaces" not in output
        # The removal attempt was still made (cheap and safe); we just don't announce it.
        assert cleanup_calls == [("claude", "orphan-mcp")]

    def test_warns_when_app_selection_is_no_longer_discoverable(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}app:mcp-vanished")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Skipped MCP selection `app:mcp-vanished`" in output
        assert "mcp-vanished" in output
        assert configured == []

    def test_warns_for_unrecognized_selection_prefix(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}bogus:value")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Skipped MCP selection `bogus:value`" in output
        assert "unrecognized" in output
        assert configured == []

    def test_continues_when_optional_discovery_fails(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp,
            "discover_external_mcp_connection_names",
            lambda workspace, profile=None: (_ for _ in ()).throw(
                RuntimeError("permission denied")
            ),
        )
        monkeypatch.setattr(
            mcp,
            "discover_genie_mcp_servers",
            lambda workspace, profile=None: (_ for _ in ()).throw(
                RuntimeError("permission denied")
            ),
        )
        monkeypatch.setattr(
            mcp,
            "discover_app_mcp_servers",
            lambda workspace, profile=None: (_ for _ in ()).throw(
                RuntimeError("permission denied")
            ),
        )
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}managed:sql", categories={"genie"})
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Skipped external connections" in output
        assert "Skipped Genie spaces" in output
        assert "Skipped Databricks apps" in output
        assert configured[0][1] == "databricks-sql"
        assert saved_states[-1]["mcp_servers"][0]["name"] == "databricks-sql"

    def test_forwards_profile_to_discovery(self, monkeypatch):
        saved_states: list[dict] = []
        seen_profiles: dict[str, str | None] = {}

        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {**CLAUDE_STATE, "profile": "my-profile"},
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])

        def fake_external(workspace, profile=None):
            seen_profiles["external"] = profile
            return []

        def fake_genie(workspace, profile=None):
            seen_profiles["genie"] = profile
            return []

        def fake_apps(workspace, profile=None):
            seen_profiles["apps"] = profile
            return []

        monkeypatch.setattr(mcp, "discover_external_mcp_connection_names", fake_external)
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", fake_genie)
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", fake_apps)
        _patch_mcp_choices(monkeypatch, categories={"genie"})
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0
        assert seen_profiles == {
            "external": "my-profile",
            "genie": "my-profile",
            "apps": "my-profile",
        }

    def test_configures_only_ucode_configured_clients(self, monkeypatch, capsys):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {"workspace": WS, "available_tools": ["claude", "codex"]},
        )
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ALL_MCP_CLIENTS)
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}managed:sql")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        output = capsys.readouterr().out
        assert "Configuring for: Claude Code, Codex" in output
        assert [call[0] for call in configured] == ["claude", "codex"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-sql",
                "url": f"{WS}/api/2.0/mcp/sql",
                "auth": "proxy",
                "clients": ["claude", "codex"],
            }
        ]

    def test_registers_databricks_sql_server(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        monkeypatch.setattr(mcp, "load_state", lambda: {**CLAUDE_STATE})
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch, f"{mcp.MCP_ADD_PREFIX}managed:sql")
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert configured == [
            (
                "claude",
                "databricks-sql",
                f"{WS}/api/2.0/mcp/sql",
            )
        ]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "databricks-sql",
                "url": f"{WS}/api/2.0/mcp/sql",
                "auth": "proxy",
                "clients": ["claude"],
            }
        ]

    def test_removes_saved_server(self, monkeypatch):
        state = {
            "workspace": WS,
            "available_tools": ["claude"],
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": f"{WS}/api/2.0/mcp/external/github-mcp",
                    "auth": "proxy",
                    "clients": ["claude"],
                }
            ],
        }
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []

        monkeypatch.setattr(mcp, "load_state", lambda: state)
        monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(
            mcp, "discover_external_mcp_connection_names", lambda workspace, profile=None: []
        )
        monkeypatch.setattr(mcp, "discover_genie_mcp_servers", lambda workspace, profile=None: [])
        monkeypatch.setattr(mcp, "discover_app_mcp_servers", lambda workspace, profile=None: [])
        _patch_mcp_choices(monkeypatch)
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command() == 0

        assert removed == [("claude", "github-mcp")]
        assert saved_states[-1]["mcp_servers"] == []


def _stub_location_base(monkeypatch, state):
    """Shared scaffolding for --location tests: no installed agents missing,
    auth no-ops, no cross-workspace residue."""
    monkeypatch.setattr(mcp, "load_state", lambda: state)
    monkeypatch.setattr(mcp.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr(mcp, "ensure_databricks_auth", lambda workspace, profile=None: None)
    monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
    monkeypatch.setattr(mcp, "purge_cross_workspace_mcp_residue", lambda state, workspace: None)
    monkeypatch.setattr(mcp, "get_databricks_token", lambda workspace, profile=None: "token")


class TestConfigureMcpFromLocation:
    def test_rejects_malformed_location(self, monkeypatch):
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(mcp, "list_mcp_services", lambda *a, **kw: ([], None))

        for bad in ("system", "system.ai.extra", ".ai", "system.", ""):
            try:
                mcp.configure_mcp_command(location=bad)
            except RuntimeError as exc:
                assert "--location" in str(exc)
            else:
                raise AssertionError(f"expected RuntimeError for `{bad}`")

    def test_invalid_location_raises_with_clear_message(self, monkeypatch):
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: ([], "HTTP 404 Not Found: NOT_FOUND"),
        )
        try:
            mcp.configure_mcp_command(location="nope.nope")
        except RuntimeError as exc:
            assert "Invalid location" in str(exc)
            assert "nope.nope" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    def test_other_listing_failure_surfaces_reason(self, monkeypatch):
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: ([], "HTTP 500 Server Error"),
        )
        try:
            mcp.configure_mcp_command(location="system.ai")
        except RuntimeError as exc:
            assert "HTTP 500" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    def test_registers_every_discovered_service(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        seen: dict[str, str] = {}
        picker_called: list[bool] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})

        def fake_list(workspace, token, parent):
            seen["parent"] = parent
            return ["system.ai.github", "system.ai.slack"], None

        monkeypatch.setattr(mcp, "list_mcp_services", fake_list)
        monkeypatch.setattr(
            mcp,
            "prompt_for_mcp_server_choices",
            lambda *a, **kw: picker_called.append(True) or [],
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert seen == {"parent": "system.ai"}
        assert picker_called == []
        assert [c[1] for c in configured] == ["system-ai-github", "system-ai-slack"]
        assert configured[0][2] == f"{WS}/ai-gateway/mcp-services/system.ai.github"
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "proxy",
                "clients": ["claude"],
            },
            {
                "name": "system-ai-slack",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.slack",
                "auth": "proxy",
                "clients": ["claude"],
            },
        ]

    def test_replaces_servers_outside_location(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        removed: list[tuple[str, str]] = []
        outside_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "proxy",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [outside_entry]},
        )
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert removed == [("claude", "databricks-sql")]
        assert [c[1] for c in configured] == ["system-ai-github"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "proxy",
                "clients": ["claude"],
            },
        ]

    def test_preserves_skills_connection(self, monkeypatch):
        """A skills connection is owned by `configure skills`, so `configure mcp
        --location` must leave it registered rather than treating it as a removal."""
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []
        skills_entry = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["main.default"],
            "url": f"{WS}/ai-gateway/skills/?schema=main.default",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [skills_entry]},
        )
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert removed == []
        assert _find_skills(saved_states[-1]["mcp_servers"]) == [skills_entry]

    def test_existing_entry_gets_reconfigured_for_newly_added_clients(self, monkeypatch):
        """An entry registered before a second agent was configured should
        get registered for that agent on the next --location run."""
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str, dict]] = []
        existing = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "proxy",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {
                "workspace": WS,
                "available_tools": ["claude", "codex"],
                "mcp_servers": [existing],
            },
        )
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai") == 0

        assert [c[0] for c in configured] == ["claude", "codex"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "proxy",
                "clients": ["claude", "codex"],
            }
        ]


class TestAddMcpCommand:
    """`ucode mcp add` (append) registers new servers without removing existing ones."""

    def test_keeps_servers_outside_location(self, monkeypatch):
        """Unlike `configure mcp --location`, `mcp add --location` preserves any
        server outside the location instead of removing it."""
        saved_states: list[dict] = []
        configured: list[tuple[str, str, str]] = []
        removed: list[tuple[str, str]] = []
        outside_entry = {
            "name": "databricks-sql",
            "url": f"{WS}/api/2.0/mcp/sql",
            "auth": "proxy",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [outside_entry]},
        )
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.add_mcp_command(location="system.ai") == 0

        # Nothing is removed; the new service is added and the outside one kept.
        assert removed == []
        assert [c[1] for c in configured] == ["system-ai-github"]
        assert saved_states[-1]["mcp_servers"] == [
            {
                "name": "system-ai-github",
                "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
                "auth": "proxy",
                "clients": ["claude"],
            },
            outside_entry,
        ]

    def test_services_subset_keeps_others_in_location(self, monkeypatch):
        """`mcp add --services` registers the named subset while leaving other
        already-registered services in the same schema untouched."""
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []
        existing = {
            "name": "system-ai-slack",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.slack",
            "auth": "proxy",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [existing]},
        )
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github", "system.ai.slack"], None),
        )
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.add_mcp_command(location="system.ai", services={"github"}) == 0

        assert removed == []
        names = [s["name"] for s in saved_states[-1]["mcp_servers"]]
        assert names == ["system-ai-github", "system-ai-slack"]

    def test_empty_services_is_a_noop(self, monkeypatch):
        """`mcp add --services ""` has nothing to add, so it's a no-op that never
        reaches configuration (and doesn't need --location the way a subset does)."""
        called: list[bool] = []
        monkeypatch.setattr(mcp, "load_state", lambda: called.append(True) or {})

        assert mcp.add_mcp_command(services=set()) == 0
        assert called == []

    def test_agents_scopes_registration_to_named_agent(self, monkeypatch):
        """With two agents configured, `agents={"claude"}` registers the server for
        claude only and records only claude on the saved entry."""
        saved_states: list[dict] = []
        configured: list[tuple[str, str]] = []
        _stub_location_base(
            monkeypatch,
            {"workspace": WS, "available_tools": ["claude", "codex"], "mcp_servers": []},
        )
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp, "list_mcp_services", lambda workspace, token, parent: (["system.ai.github"], None)
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.add_mcp_command(location="system.ai", agents={"claude"}) == 0

        assert configured == [("claude", "system-ai-github")]
        assert saved_states[-1]["mcp_servers"][0]["clients"] == ["claude"]

    def test_agents_not_configured_raises(self, monkeypatch):
        """`--agents` naming an agent that isn't configured for MCP is a clear error
        (the CLI sets agents up first, so this guards the library entry point)."""
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": []})
        monkeypatch.setattr(
            mcp, "list_mcp_services", lambda workspace, token, parent: (["system.ai.github"], None)
        )
        try:
            mcp.add_mcp_command(location="system.ai", agents={"gemini"})
        except RuntimeError as exc:
            assert "gemini" in str(exc)
            assert "not configured" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for an unconfigured agent")


class TestRemoveMcpCommand:
    """`ucode mcp remove` interactively unregisters configured MCP servers."""

    GITHUB = {
        "name": "system-ai-github",
        "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
        "auth": "proxy",
        "clients": ["claude"],
    }
    SQL = {
        "name": "databricks-sql",
        "url": f"{WS}/api/2.0/mcp/sql",
        "auth": "proxy",
        "clients": ["claude"],
    }

    def test_removes_only_selected_servers(self, monkeypatch):
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [self.GITHUB, self.SQL]},
        )
        monkeypatch.setattr(mcp, "_prompt_for_mcp_removal", lambda servers: ["system-ai-github"])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.remove_mcp_command() == 0

        assert removed == [("claude", "system-ai-github")]
        assert [s["name"] for s in saved_states[-1]["mcp_servers"]] == ["databricks-sql"]

    def test_cancel_makes_no_changes(self, monkeypatch):
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": [self.SQL]})
        monkeypatch.setattr(mcp, "_prompt_for_mcp_removal", lambda servers: None)
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state))

        assert mcp.remove_mcp_command() == 0
        assert saved_states == []

    def test_skills_connection_is_not_offered(self, monkeypatch):
        offered: dict[str, list[str]] = {}
        skills = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "url": f"{WS}/ai-gateway/skills/?schema=main.default",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        _stub_location_base(
            monkeypatch,
            {**CLAUDE_STATE, "mcp_servers": [skills, self.GITHUB]},
        )

        def fake_prompt(servers):
            offered["names"] = [s["name"] for s in servers]
            return []

        monkeypatch.setattr(mcp, "_prompt_for_mcp_removal", fake_prompt)

        assert mcp.remove_mcp_command() == 0
        # The skills connection is owned by `configure skills`, so it's never a
        # removal candidate; only the real MCP server is offered.
        assert offered["names"] == ["system-ai-github"]

    def test_no_configured_servers_skips_the_picker(self, monkeypatch):
        prompted: list[bool] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": []})
        monkeypatch.setattr(
            mcp, "_prompt_for_mcp_removal", lambda servers: prompted.append(True) or []
        )

        assert mcp.remove_mcp_command() == 0
        assert prompted == []

    def test_agents_scopes_removal_to_named_agent(self, monkeypatch):
        """`--agents claude` unregisters the server from claude only; a server that
        was also on codex is kept there with claude stripped from its clients."""
        saved_states: list[dict] = []
        removed: list[tuple[str, str]] = []
        both = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "proxy",
            "clients": ["claude", "codex"],
        }
        _stub_location_base(
            monkeypatch,
            {"workspace": WS, "available_tools": ["claude", "codex"], "mcp_servers": [both]},
        )
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(mcp, "_prompt_for_mcp_removal", lambda servers: ["system-ai-github"])
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.remove_mcp_command(agents={"claude"}) == 0

        # Only claude is unregistered; the state entry survives on codex.
        assert removed == [("claude", "system-ai-github")]
        assert saved_states[-1]["mcp_servers"] == [{**both, "clients": ["codex"]}]

    def test_agents_only_offers_servers_on_that_agent(self, monkeypatch):
        prompted: list[bool] = []
        codex_only = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "proxy",
            "clients": ["codex"],
        }
        _stub_location_base(
            monkeypatch,
            {"workspace": WS, "available_tools": ["claude", "codex"], "mcp_servers": [codex_only]},
        )
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp, "_prompt_for_mcp_removal", lambda servers: prompted.append(True) or []
        )

        # Nothing is registered on claude, so `--agents claude` has nothing to offer.
        assert mcp.remove_mcp_command(agents={"claude"}) == 0
        assert prompted == []


class TestConfigureMcpServicesSubset:
    """`--location <schema> --services a,b,...` configures exactly the named subset."""

    def test_configures_only_the_requested_subset(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (
                ["system.ai.github", "system.ai.slack", "system.ai.gmail"],
                None,
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert (
            mcp.configure_mcp_command(
                location="system.ai", services={"system.ai.github", "system.ai.gmail"}
            )
            == 0
        )

        # slack is dropped; only the two requested services are configured.
        assert sorted(c[1] for c in configured) == ["system-ai-github", "system-ai-gmail"]
        assert sorted(s["name"] for s in saved_states[-1]["mcp_servers"]) == [
            "system-ai-github",
            "system-ai-gmail",
        ]

    def test_matches_bare_short_names(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github", "system.ai.slack"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        assert mcp.configure_mcp_command(location="system.ai", services={"github"}) == 0

        assert [c[1] for c in configured] == ["system-ai-github"]

    def test_unknown_requested_service_warns_and_skips(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        warnings: list[str] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)
        monkeypatch.setattr(mcp, "print_warning", lambda msg: warnings.append(msg))

        assert (
            mcp.configure_mcp_command(
                location="system.ai", services={"system.ai.github", "system.ai.ghost"}
            )
            == 0
        )

        # The known service is still configured; the unknown one is reported, not fatal.
        assert [c[1] for c in configured] == ["system-ai-github"]
        assert any("system.ai.ghost" in w for w in warnings)

    def test_empty_services_removes_everything(self, monkeypatch):
        existing = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "proxy",
            "clients": ["claude"],
        }
        configured: list[tuple[str, str, str, dict]] = []
        removed: list[tuple[str, str]] = []
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": [existing]})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (["system.ai.github"], None),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_mcp_command(location="system.ai", services=set()) == 0

        assert configured == []
        assert removed == [("claude", "system-ai-github")]
        assert saved_states[-1]["mcp_servers"] == []

    def test_adds_and_removes_to_match_new_selection(self, monkeypatch):
        # The live case teammates want mid-session: started with github+slack,
        # then the user deselects slack and selects gmail.
        github = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "proxy",
            "clients": ["claude"],
        }
        slack = {
            "name": "system-ai-slack",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.slack",
            "auth": "proxy",
            "clients": ["claude"],
        }
        configured: list[tuple[str, str, str, dict]] = []
        removed: list[tuple[str, str]] = []
        saved_states: list[dict] = []
        _stub_location_base(monkeypatch, {**CLAUDE_STATE, "mcp_servers": [github, slack]})
        monkeypatch.setattr(
            mcp,
            "list_mcp_services",
            lambda workspace, token, parent: (
                ["system.ai.github", "system.ai.slack", "system.ai.gmail"],
                None,
            ),
        )
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert (
            mcp.configure_mcp_command(
                location="system.ai", services={"system.ai.github", "system.ai.gmail"}
            )
            == 0
        )

        # slack removed, gmail added, github untouched (entry unchanged).
        assert removed == [("claude", "system-ai-slack")]
        assert [c[1] for c in configured] == ["system-ai-gmail"]
        assert sorted(s["name"] for s in saved_states[-1]["mcp_servers"]) == [
            "system-ai-github",
            "system-ai-gmail",
        ]

    def test_full_names_without_location_derive_the_schema(self, monkeypatch):
        configured: list[tuple[str, str, str, dict]] = []
        seen: dict[str, str] = {}
        _stub_location_base(monkeypatch, {**CLAUDE_STATE})

        def fake_list(workspace, token, parent):
            seen["parent"] = parent
            return ["system.ai.github", "system.ai.slack"], None

        monkeypatch.setattr(mcp, "list_mcp_services", fake_list)
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, name, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: None)

        # No --location: the `<catalog>.<schema>` is derived from the full names.
        assert mcp.configure_mcp_command(services={"system.ai.github", "system.ai.slack"}) == 0

        assert seen == {"parent": "system.ai"}
        assert sorted(c[1] for c in configured) == ["system-ai-github", "system-ai-slack"]

    def test_short_name_without_location_raises(self):
        try:
            mcp.configure_mcp_command(services={"github"})
        except RuntimeError as exc:
            assert "--location" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for a bare short name without --location")

    def test_full_names_spanning_multiple_schemas_without_location_raises(self):
        try:
            mcp.configure_mcp_command(services={"system.ai.github", "other.cat.thing"})
        except RuntimeError as exc:
            assert "--location" in str(exc)
        else:
            raise AssertionError(
                "expected RuntimeError for multi-schema services without --location"
            )


def _find_skills(servers):
    return [s for s in servers if s.get("kind") == mcp.SKILLS_MCP_KIND]


def _by_client(clients, locations):
    return {client: list(locations) for client in clients}


class TestResolveSkillsMcpServers:
    def test_builds_single_canonical_entry(self):
        servers = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["main.default"]), []
        )
        assert _find_skills(servers) == servers
        entry = servers[0]
        assert entry["name"] == mcp.SKILLS_MCP_SERVER_NAME
        assert entry["kind"] == mcp.SKILLS_MCP_KIND
        assert entry["skill_locations"] == ["main.default"]
        assert entry["url"] == f"{WS}/ai-gateway/skills/?schema=main.default"
        assert entry["auth"] == "proxy"
        assert entry["clients"] == ["claude"]

    def test_keeps_other_entries_and_rebuilds_to_one_skills_entry(self):
        service_entry = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        stale_skills = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["old.one"],
            "url": f"{WS}/ai-gateway/skills/?schema=old.one",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["codex"],
        }
        servers = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["a.b"]), [service_entry, stale_skills]
        )
        assert service_entry in servers
        skills = _find_skills(servers)
        assert len(skills) == 1
        # Prior skills clients merge with the newly-configured ones (order-stable).
        assert skills[0]["clients"] == ["codex", "claude"]

    def test_drops_entry_matching_skills_server_name_even_without_kind(self):
        old_named = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "url": f"{WS}/ai-gateway/skills/",
            "clients": ["claude"],
        }
        servers = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["a.b"]), [old_named]
        )
        assert len(servers) == 1
        assert servers[0]["kind"] == mcp.SKILLS_MCP_KIND

    def test_url_derives_from_locations_not_stale_url(self):
        stale = {
            "name": mcp.SKILLS_MCP_SERVER_NAME,
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["old.one"],
            "url": f"{WS}/ai-gateway/skills/?schema=stale.value",
            "clients": ["claude"],
        }
        servers = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["new.two"]), [stale]
        )
        assert servers[0]["url"] == f"{WS}/ai-gateway/skills/?schema=new.two"

    def test_empty_locations_yields_bare_route(self):
        servers = mcp._resolve_skills_mcp_servers(WS, ["claude"], [], [])
        assert servers[0]["skill_locations"] == []
        assert servers[0]["url"] == f"{WS}/ai-gateway/skills/"


def _skills_state(mcp_servers=None):
    state = {"workspace": WS, "available_tools": ["claude"]}
    if mcp_servers is not None:
        state["mcp_servers"] = mcp_servers
    return state


class TestConfigureSkillsMcpCommand:
    def test_set_on_empty_registers_connection(self, monkeypatch):
        saved_states: list[dict] = []
        configured: list[dict] = []
        _stub_location_base(monkeypatch, _skills_state())
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: (
                configured.append(
                    {
                        "client": client,
                        "name": name,
                        "url": url,
                        "always_load": kw.get("always_load"),
                    }
                )
                or []
            ),
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["a.b"]) == 0

        skills = _find_skills(saved_states[-1]["mcp_servers"])
        assert len(skills) == 1
        assert skills[0]["skill_locations"] == ["a.b"]
        assert skills[0]["url"] == f"{WS}/ai-gateway/skills/?schema=a.b"
        # alwaysLoad is passed through the proxy registration (Claude-only hint).
        assert configured[0]["always_load"] is True

    def test_location_replaces_prior_set(self, monkeypatch):
        saved_states: list[dict] = []
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["A.a", "B.b"]), []
        )
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["X.x"]) == 0

        assert _find_skills(saved_states[-1]["mcp_servers"])[0]["skill_locations"] == ["X.x"]

    def test_multiple_locations_set_in_order(self, monkeypatch):
        saved_states: list[dict] = []
        prior = mcp._resolve_skills_mcp_servers(WS, ["claude"], _by_client(["claude"], ["A.a"]), [])
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["X.x", "Y.y"]) == 0

        assert _find_skills(saved_states[-1]["mcp_servers"])[0]["skill_locations"] == ["X.x", "Y.y"]

    def test_replaces_scope_for_configured_clients_only(self, monkeypatch):
        saved_states: list[dict] = []
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude", "codex"], {"claude": ["claude.old"], "codex": ["codex.kept"]}, []
        )
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["new.default"]) == 0

        entry = _find_skills(saved_states[-1]["mcp_servers"])[0]
        assert mcp.skill_locations_for_client(entry, "claude") == ["new.default"]
        assert mcp.skill_locations_for_client(entry, "codex") == ["codex.kept"]

    def test_preserves_mcp_service_entries_across_set(self, monkeypatch):
        saved_states: list[dict] = []
        service_entry = {
            "name": "system-ai-github",
            "url": f"{WS}/ai-gateway/mcp-services/system.ai.github",
            "auth": "env:OAUTH_TOKEN",
            "clients": ["claude"],
        }
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["A.a"]), [service_entry]
        )
        _stub_location_base(monkeypatch, _skills_state(prior))
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        assert mcp.configure_skills_mcp_command(["B.b"]) == 0

        names = [s["name"] for s in saved_states[-1]["mcp_servers"]]
        assert "system-ai-github" in names
        assert names.count(mcp.SKILLS_MCP_SERVER_NAME) == 1


class TestSkillMcpLocations:
    def test_reads_locations_off_skills_entry(self):
        state = _skills_state(
            mcp._resolve_skills_mcp_servers(
                WS, ["claude"], _by_client(["claude"], ["A.a", "B.b"]), []
            )
        )
        assert mcp._skill_mcp_locations(state) == ["A.a", "B.b"]

    def test_empty_when_no_skills_entry(self):
        assert mcp._skill_mcp_locations(_skills_state([])) == []
        assert mcp._skill_mcp_locations(_skills_state()) == []

    def test_ignores_malformed_default_locations(self):
        entry = {"kind": mcp.SKILLS_MCP_KIND, "skill_locations": "not-a-list"}
        state = _skills_state([entry])

        assert mcp._skill_mcp_locations(state) == []
        assert mcp.skill_locations_for_client(entry, "claude") == []

    def test_per_client_scopes_are_independent(self):
        entry = mcp._build_skills_entry(
            WS,
            {"claude": ["common.schema", "claude.only"], "codex": ["common.schema"]},
            ["claude", "codex"],
        )

        assert mcp.skill_locations_for_client(entry, "claude") == [
            "common.schema",
            "claude.only",
        ]
        assert mcp.skill_locations_for_client(entry, "codex") == ["common.schema"]

    def test_legacy_flat_scope_mirrors_to_every_client(self):
        entry = {
            "kind": mcp.SKILLS_MCP_KIND,
            "skill_locations": ["a.b", "c.d"],
            "clients": ["claude", "codex"],
        }

        assert mcp.skill_locations_for_client(entry, "claude") == ["a.b", "c.d"]
        assert mcp.skill_locations_for_client(entry, "codex") == ["a.b", "c.d"]


class TestUnionLocations:
    def test_appends_new_after_existing(self):
        assert mcp._union_locations(["a.b"], ["c.d"]) == ["a.b", "c.d"]

    def test_drops_locations_already_present(self):
        assert mcp._union_locations(["a.b", "c.d"], ["c.d", "e.f"]) == ["a.b", "c.d", "e.f"]

    def test_empty_base_returns_new(self):
        assert mcp._union_locations([], ["a.b", "c.d"]) == ["a.b", "c.d"]

    def test_drops_duplicate_new_locations(self):
        assert mcp._union_locations(["a.b"], ["c.d", "c.d"]) == ["a.b", "c.d"]


class TestAddSkillsCommand:
    """`ucode skill add --mcp` unions schemas into the connection scope rather
    than replacing it (unlike `configure_skills_mcp_command`)."""

    def test_unions_into_existing_scope(self, monkeypatch):
        state = _skills_state(
            mcp._resolve_skills_mcp_servers(WS, ["claude"], _by_client(["claude"], ["A.a"]), [])
        )
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda s: None)

        assert mcp.add_skills_command(["B.b"]) == 0

        assert _find_skills(state["mcp_servers"])[0]["skill_locations"] == ["A.a", "B.b"]

    def test_existing_schema_leaves_scope_unchanged(self, monkeypatch):
        state = _skills_state(
            mcp._resolve_skills_mcp_servers(
                WS, ["claude"], _by_client(["claude"], ["A.a", "B.b"]), []
            )
        )
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda s: None)

        assert mcp.add_skills_command(["A.a"]) == 0

        assert _find_skills(state["mcp_servers"])[0]["skill_locations"] == ["A.a", "B.b"]

    def test_registers_scope_from_empty_state(self, monkeypatch):
        state = _skills_state()
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda s: None)

        assert mcp.add_skills_command(["A.a"]) == 0

        assert _find_skills(state["mcp_servers"])[0]["skill_locations"] == ["A.a"]

    def test_agents_updates_only_selected_client_scope(self, monkeypatch):
        configured: list[tuple[str, str]] = []
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude", "codex"], _by_client(["claude", "codex"], ["A.a"]), []
        )
        state = {"workspace": WS, "available_tools": ["claude", "codex"], "mcp_servers": prior}
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda s: None)

        assert mcp.add_skills_command(["B.b"], agents={"claude"}) == 0

        entry = _find_skills(state["mcp_servers"])[0]
        assert mcp.skill_locations_for_client(entry, "claude") == ["A.a", "B.b"]
        assert mcp.skill_locations_for_client(entry, "codex") == ["A.a"]
        assert configured == [("claude", f"{WS}/ai-gateway/skills/?schema=A.a&schema=B.b")]

    def test_global_addition_reaches_every_client_and_keeps_divergence(self, monkeypatch):
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude", "codex"], {"claude": ["A.a", "B.b"], "codex": ["A.a"]}, []
        )
        state = {"workspace": WS, "available_tools": ["claude", "codex"], "mcp_servers": prior}
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "save_state", lambda s: None)

        assert mcp.add_skills_command(["C.c"]) == 0

        entry = _find_skills(state["mcp_servers"])[0]
        assert mcp.skill_locations_for_client(entry, "claude") == ["A.a", "B.b", "C.c"]
        assert mcp.skill_locations_for_client(entry, "codex") == ["A.a", "C.c"]

    def test_agents_add_matching_existing_scope_is_a_noop(self, monkeypatch):
        configured: list[tuple[str, str]] = []
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude", "codex"], _by_client(["claude", "codex"], ["A.a"]), []
        )
        state = {"workspace": WS, "available_tools": ["claude", "codex"], "mcp_servers": prior}
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda s: None)

        assert mcp.add_skills_command(["A.a"], agents={"claude"}) == 0

        entry = _find_skills(state["mcp_servers"])[0]
        assert mcp.skill_locations_for_client(entry, "claude") == ["A.a"]
        assert configured == []


class TestRemoveSkillsCommand:
    def _state(self, by_client=None):
        by_client = by_client or _by_client(["claude", "codex"], ["A.a", "B.b"])
        return {
            "workspace": WS,
            "available_tools": ["claude", "codex"],
            "mcp_servers": mcp._resolve_skills_mcp_servers(WS, list(by_client), by_client, []),
        }

    def _stub(self, monkeypatch, state, selection):
        configured: list[tuple[str, str]] = []
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(mcp, "_prompt_for_skill_removal", lambda scopes: selection)
        monkeypatch.setattr(
            mcp,
            "configure_client_mcp_server",
            lambda client, name, url, *a, **kw: configured.append((client, url)) or [],
        )
        monkeypatch.setattr(mcp, "save_state", lambda s: None)
        return configured

    def test_removes_selected_schema_from_every_client(self, monkeypatch):
        state = self._state()
        configured = self._stub(monkeypatch, state, ["A.a"])

        assert mcp.remove_skills_command() == 0

        entry = _find_skills(state["mcp_servers"])[0]
        assert mcp.skill_locations_for_client(entry, "claude") == ["B.b"]
        assert mcp.skill_locations_for_client(entry, "codex") == ["B.b"]
        assert sorted(configured) == [
            ("claude", f"{WS}/ai-gateway/skills/?schema=B.b"),
            ("codex", f"{WS}/ai-gateway/skills/?schema=B.b"),
        ]

    def test_removing_all_schemas_keeps_schemaless_connection(self, monkeypatch):
        state = self._state()
        configured = self._stub(monkeypatch, state, ["A.a", "B.b"])

        assert mcp.remove_skills_command() == 0

        entry = _find_skills(state["mcp_servers"])[0]
        assert entry["skill_locations"] == []
        assert sorted(configured) == [
            ("claude", f"{WS}/ai-gateway/skills/"),
            ("codex", f"{WS}/ai-gateway/skills/"),
        ]

    def test_nothing_configured_is_a_noop(self, monkeypatch):
        state = {
            "workspace": WS,
            "available_tools": ["claude", "codex"],
            "mcp_servers": mcp._resolve_skills_mcp_servers(WS, ["claude", "codex"], {}, []),
        }
        captured: dict[str, bool] = {}
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp, "_prompt_for_skill_removal", lambda scopes: captured.setdefault("called", True)
        )

        assert mcp.remove_skills_command() == 0
        assert "called" not in captured

    def test_offers_each_client_scope_to_the_picker(self, monkeypatch):
        state = self._state({"claude": ["A.a", "B.b"], "codex": ["A.a"]})
        captured: dict[str, dict[str, list[str]]] = {}
        _stub_location_base(monkeypatch, state)
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude", "codex"])
        monkeypatch.setattr(
            mcp,
            "_prompt_for_skill_removal",
            lambda scopes: captured.setdefault("scopes", scopes) and None,
        )

        assert mcp.remove_skills_command() == 0
        assert captured["scopes"] == {"claude": ["A.a", "B.b"], "codex": ["A.a"]}


class TestRegisterSchemalessSkillsConnection:
    def _stub(self, monkeypatch):
        saved_states: list[dict] = []
        monkeypatch.setattr(mcp, "configure_client_mcp_server", lambda *a, **kw: [])
        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))
        return saved_states

    def test_registers_bare_route_when_none_exists(self, monkeypatch):
        saved_states = self._stub(monkeypatch)
        state = _skills_state([])

        mcp.register_schemaless_skills_connection(state, WS, None, ["claude"])

        skills = _find_skills(saved_states[-1]["mcp_servers"])
        assert len(skills) == 1
        assert skills[0]["skill_locations"] == []
        assert skills[0]["url"] == f"{WS}/ai-gateway/skills/"

    def test_preserves_prior_mcp_location_set(self, monkeypatch):
        self._stub(monkeypatch)
        prior = mcp._resolve_skills_mcp_servers(
            WS, ["claude"], _by_client(["claude"], ["X.x", "Y.y"]), []
        )
        state = _skills_state(prior)

        mcp.register_schemaless_skills_connection(state, WS, None, ["claude"])

        assert _find_skills(state["mcp_servers"])[0]["skill_locations"] == ["X.x", "Y.y"]


class TestSkillsToolsDescription:
    def test_bare_route_names_utility_tools_only(self):
        assert mcp._skills_tools_description([]) == "UC skill utility tools"

    def test_scoped_names_utility_plus_skills_tools(self):
        assert mcp._skills_tools_description(["main.default"]) == (
            "UC skill utility tools + skills tools in schema main.default"
        )

    def test_multiple_schemas_joined_with_and(self):
        assert mcp._skills_tools_description(["a.b", "c.d", "e.f"]) == (
            "UC skill utility tools + skills tools in schema a.b, c.d and e.f"
        )


class TestPrintSkillsSummary:
    def _entry(self, locations):
        clients = ["claude", "codex"]
        return mcp._resolve_skills_mcp_servers(WS, clients, _by_client(clients, locations), [])[0]

    def test_reports_scoped_connection(self, capsys):
        mcp._print_skills_summary(self._entry(["main.default"]))
        assert _unwrap(capsys.readouterr().out) == (
            "✔ Skills MCP registered "
            "Server: databricks-skill-registry "
            f"URL: {WS}/ai-gateway/skills/?schema=main.default "
            "Configured: Claude Code, Codex "
            "Tools: UC skill utility tools + skills tools in schema main.default "
            "• Run `ucode <agent>` to use the skills MCP. For existing sessions, "
            "restart the agent for the skills to take effect."
        )

    def test_reports_schemaless_connection(self, capsys):
        mcp._print_skills_summary(self._entry([]))
        assert _unwrap(capsys.readouterr().out) == (
            "✔ Skills MCP registered "
            "Server: databricks-skill-registry "
            f"URL: {WS}/ai-gateway/skills/ "
            "Configured: Claude Code, Codex "
            "Tools: UC skill utility tools "
            "• Run `ucode <agent>` to use the skills MCP. For existing sessions, "
            "restart the agent for the skills to take effect."
        )


class TestRevertMcpConfigs:
    def test_removes_cli_registered_servers_and_restores_copilot_config(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        restored: list[tuple[object, object, bool]] = []

        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(
            mcp,
            "restore_file",
            lambda config_path, backup_path, managed: (
                restored.append((config_path, backup_path, managed)) or True
            ),
        )

        result = mcp.revert_mcp_configs(
            {
                "mcp_servers": [
                    {
                        "name": "github-mcp",
                        "clients": ["claude", "codex", "gemini", "opencode", "copilot"],
                    }
                ]
            }
        )

        assert removed == [
            ("claude", "github-mcp"),
            ("codex", "github-mcp"),
            ("gemini", "github-mcp"),
            ("opencode", "github-mcp"),
            ("copilot", "github-mcp"),
        ]
        assert restored == [
            (mcp.copilot.COPILOT_MCP_CONFIG_PATH, mcp.copilot.COPILOT_MCP_BACKUP_PATH, True)
        ]
        assert result == {
            "claude": True,
            "codex": True,
            "gemini": True,
            "opencode": True,
            "copilot": True,
        }

    def test_removes_skills_registry_across_its_clients(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(mcp, "restore_file", lambda *a, **kw: False)

        skills_entry = mcp._resolve_skills_mcp_servers(
            WS, ["claude", "codex"], _by_client(["claude", "codex"], ["a.b"]), []
        )[0]
        mcp.revert_mcp_configs({"mcp_servers": [skills_entry]})

        assert removed == [
            ("claude", mcp.SKILLS_MCP_SERVER_NAME),
            ("codex", mcp.SKILLS_MCP_SERVER_NAME),
        ]


class TestPurgeCrossWorkspaceSkillsEntry:
    def test_drops_foreign_workspace_skills_entry(self, monkeypatch):
        removed: list[tuple[str, str]] = []
        saved_states: list[dict] = []
        foreign = "https://other.databricks.com"
        skills_entry = mcp._resolve_skills_mcp_servers(
            foreign, ["claude"], _by_client(["claude"], ["a.b"]), []
        )[0]
        # The skills URL carries a `?schema=` query; its host must still parse.
        assert mcp._mcp_entry_url_host(skills_entry) == "other.databricks.com"
        state = {"mcp_servers": [skills_entry]}

        monkeypatch.setattr(mcp, "available_mcp_clients", lambda: ["claude"])
        monkeypatch.setattr(mcp, "load_full_state", lambda: {})
        monkeypatch.setattr(
            mcp,
            "remove_client_mcp_server",
            lambda client, name: removed.append((client, name)) or ["user"],
        )
        monkeypatch.setattr(mcp, "save_state", lambda state: saved_states.append(state.copy()))

        mcp.purge_cross_workspace_mcp_residue(state, WS)

        assert removed == [("claude", mcp.SKILLS_MCP_SERVER_NAME)]
        assert state["mcp_servers"] == []


class TestManagedMcpServerEntry:
    def test_sql(self):
        assert mcp.managed_mcp_server_entry("databricks-sql", "sql", WS) == (
            "databricks-sql",
            f"{WS}/api/2.0/mcp/sql",
        )

    def test_external_uses_the_connection_name(self):
        assert mcp.managed_mcp_server_entry("jira-prod", "external", WS) == (
            "jira-prod",
            f"{WS}/api/2.0/mcp/external/jira-prod",
        )

    def test_mcp_service_undashes_catalog_and_schema_only(self):
        # The manifest stores the dash form; only the first two dashes (catalog.schema) become dots,
        # so a service name keeps its own dashes/underscores. The entry name stays the dash form.
        assert mcp.managed_mcp_server_entry("system-ai-dbsql", "mcp-service", WS) == (
            "system-ai-dbsql",
            f"{WS}/ai-gateway/mcp-services/system.ai.dbsql",
        )
        assert mcp.managed_mcp_server_entry("system-ai-google_calendar", "mcp-service", WS) == (
            "system-ai-google_calendar",
            f"{WS}/ai-gateway/mcp-services/system.ai.google_calendar",
        )

    def test_mcp_service_needs_three_parts(self):
        assert mcp.managed_mcp_server_entry("justtwo-parts", "mcp-service", WS) is None

    def test_genie_space_uses_the_space_id(self):
        assert mcp.managed_mcp_server_entry("01ef9a", "genie-space", WS) == (
            "databricks-genie-01ef9a",
            f"{WS}/api/2.0/mcp/genie/01ef9a",
        )

    def test_uc_functions_splits_catalog_schema(self):
        # The dot-free entry name is a slug; the URL uses the raw catalog/schema path segments.
        entry_name, url = mcp.managed_mcp_server_entry("dev_cat.dev_fixture", "uc-functions", WS)
        assert "." not in entry_name
        assert url == f"{WS}/api/2.0/mcp/functions/dev_cat/dev_fixture"

    def test_vector_search_splits_catalog_schema(self):
        entry_name, url = mcp.managed_mcp_server_entry("my_cat.my_schema", "vector-search", WS)
        assert "." not in entry_name
        assert url == f"{WS}/api/2.0/mcp/vector-search/my_cat/my_schema"

    def test_catalog_schema_needs_exactly_two_parts(self):
        assert mcp.managed_mcp_server_entry("onlycatalog", "uc-functions", WS) is None
        assert mcp.managed_mcp_server_entry("a.b.c", "uc-functions", WS) is None

    def test_app_and_unknown_types_return_none(self):
        for mcp_type in ("app", "bogus"):
            assert mcp.managed_mcp_server_entry("x", mcp_type, WS) is None


class TestApplyManagedMcpServers:
    def _managed(self, *servers):
        return {"mcp_servers": list(servers)}

    def test_registers_supported_servers_for_the_launching_tool(self, monkeypatch):
        applied = {}
        monkeypatch.setattr(mcp, "load_state", lambda: {})
        monkeypatch.setattr(
            mcp,
            "apply_mcp_server_changes",
            lambda prev, working, clients, ws, profile=None, **kw: applied.update(
                {"working": working, "clients": clients, "prev": prev}
            ),
        )
        managed = self._managed(
            {"name": "system-ai-dbsql", "type": "mcp-service"},
            {"name": "databricks-sql", "type": "sql"},
        )
        registered = mcp.apply_managed_mcp_servers(managed, "claude", WS)
        assert applied["clients"] == ["claude"]
        assert {s["name"] for s in registered} == {"system-ai-dbsql", "databricks-sql"}
        assert all(s["clients"] == ["claude"] for s in registered)

    def test_registers_genie_and_catalog_schema_types(self, monkeypatch):
        applied = {}
        monkeypatch.setattr(mcp, "load_state", lambda: {})
        monkeypatch.setattr(
            mcp,
            "apply_mcp_server_changes",
            lambda prev, working, *a, **k: applied.update({"working": working}),
        )
        managed = self._managed(
            {"name": "01ef9a", "type": "genie-space"},
            {"name": "cat.sch", "type": "uc-functions"},
        )
        registered = mcp.apply_managed_mcp_servers(managed, "claude", WS)
        names = {s["name"] for s in registered}
        assert "databricks-genie-01ef9a" in names
        assert any(n.startswith("databricks-functions-") for n in names)

    def test_skips_and_warns_on_unsupported_types(self, monkeypatch):
        # `app` is the remaining type ucode can't rebuild from the config (needs an off-workspace
        # host); it is skipped with a warning while the supported entry still registers.
        warned: list[str] = []
        monkeypatch.setattr(mcp, "load_state", lambda: {})
        monkeypatch.setattr(mcp, "apply_mcp_server_changes", lambda *a, **k: None)
        monkeypatch.setattr(mcp, "print_warning", lambda msg: warned.append(msg))
        managed = self._managed(
            {"name": "system-ai-dbsql", "type": "mcp-service"},
            {"name": "my-app", "type": "app"},
        )
        registered = mcp.apply_managed_mcp_servers(managed, "claude", WS)
        assert {s["name"] for s in registered} == {"system-ai-dbsql"}
        assert warned and "my-app" in warned[0]

    def test_diffs_against_this_tools_previously_registered_servers(self, monkeypatch):
        seen_prev = {}
        monkeypatch.setattr(
            mcp,
            "load_state",
            lambda: {
                "managed_mcp_servers": [
                    {"name": "old", "url": "u", "clients": ["claude"]},
                    {"name": "other-tool", "url": "u", "clients": ["codex"]},
                ]
            },
        )
        monkeypatch.setattr(
            mcp,
            "apply_mcp_server_changes",
            lambda prev, working, *a, **k: seen_prev.update({"prev": prev}),
        )
        mcp.apply_managed_mcp_servers(
            self._managed({"name": "databricks-sql", "type": "sql"}), "claude", WS
        )
        # Only this tool's prior servers form the diff baseline; codex's are left alone.
        assert [s["name"] for s in seen_prev["prev"]] == ["old"]

    def test_no_supported_servers_does_nothing(self, monkeypatch):
        monkeypatch.setattr(mcp, "load_state", lambda: {})
        monkeypatch.setattr(
            mcp,
            "apply_mcp_server_changes",
            lambda *a, **k: pytest.fail("should not apply when nothing is registerable"),
        )
        monkeypatch.setattr(mcp, "print_warning", lambda msg: None)
        registered = mcp.apply_managed_mcp_servers(
            self._managed({"name": "s", "type": "app"}), "claude", WS
        )
        assert registered == []

    def test_mcp_only_client_returns_empty(self, monkeypatch):
        # A tool that isn't an MCP client can't have servers registered against it.
        monkeypatch.setattr(
            mcp,
            "apply_mcp_server_changes",
            lambda *a, **k: pytest.fail("should not apply for a non-client tool"),
        )
        registered = mcp.apply_managed_mcp_servers(
            self._managed({"name": "databricks-sql", "type": "sql"}), "not-a-client", WS
        )
        assert registered == []


class TestPromptForMcpSearchSourcesExclusion:
    def _values(self, monkeypatch, exclude):
        captured: dict = {}

        class FakePrompt:
            def ask(self):
                return []

        def fake_checkbox(*args, **kwargs):
            captured["choices"] = kwargs["choices"]
            return FakePrompt()

        monkeypatch.setattr(mcp, "_scrolling_checkbox", fake_checkbox)
        mcp.prompt_for_mcp_search_sources(exclude_sources=exclude)
        return [c.value for c in captured["choices"]]

    def test_apps_excluded(self, monkeypatch):
        values = self._values(monkeypatch, {"apps"})
        assert "apps" not in values
        assert "external" in values and "genie" in values

    def test_nothing_excluded_by_default(self, monkeypatch):
        assert "apps" in self._values(monkeypatch, None)


class TestIsAppMcpServer:
    def test_app_host_is_an_app(self):
        assert mcp._is_app_mcp_server({"url": "https://myapp-1.databricksapps.com/mcp"}) is True

    def test_workspace_relative_shapes_are_not_apps(self):
        for url in (
            f"{WS}/api/2.0/mcp/sql",
            f"{WS}/api/2.0/mcp/external/jira",
            f"{WS}/api/2.0/mcp/genie/01ef",
            f"{WS}/api/2.0/mcp/vector-search/main/default",
            f"{WS}/api/2.0/mcp/functions/main/default",
            f"{WS}/ai-gateway/mcp-services/system.ai.dbsql",
        ):
            assert mcp._is_app_mcp_server({"url": url}) is False

    def test_non_string_url_is_not_an_app(self):
        assert mcp._is_app_mcp_server({}) is False
