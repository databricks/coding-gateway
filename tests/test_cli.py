"""Tests for CLI subcommand routing and passthrough args."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import subprocess
import time
import tomllib
from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

import ucode.cli as cli_mod
import ucode.databricks as db_mod
from ucode.cli import app
from ucode.databricks import GatewayProbe

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Drop SGR escape sequences so substring assertions match regardless of
    whether the runner forces color rendering (e.g. CI sets FORCE_COLOR=1,
    which makes rich split styled tokens like ``--agents`` with ANSI codes)."""
    return _ANSI_RE.sub("", text)


runner = CliRunner()

TOOLS = ["codex", "claude", "gemini", "opencode"]

MODEL_SERVICE_PROBE = GatewayProbe(True, "reachable, accessible model service returned", True)


def _jwt(expires_at: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expires_at}).encode()).decode()
    return f"header.{payload.rstrip('=')}.signature"


@pytest.fixture(autouse=True)
def no_state_writes():
    """Prevent any test from writing to the real state file on disk."""
    with (
        patch("ucode.state.save_state"),
        patch("ucode.cli.save_state"),
        patch("ucode.agents.__init__.save_state"),
        patch("ucode.agents.codex.save_state"),
        patch("ucode.agents.claude.save_state"),
        patch("ucode.agents.claude._managed_settings_path", return_value=None),
        patch("ucode.agents.gemini.save_state"),
        patch("ucode.agents.opencode.save_state"),
    ):
        yield


MINIMAL_STATE = {
    "workspace": "https://example.databricks.com",
    "base_urls": {
        "codex": "https://example.databricks.com/ai-gateway/codex",
        "claude": "https://example.databricks.com/ai-gateway/anthropic",
        "gemini": "https://example.databricks.com/ai-gateway/gemini",
        "opencode": "https://example.databricks.com/ai-gateway/opencode",
    },
    "claude_models": {"sonnet": "databricks-claude-sonnet-4"},
    "gemini_models": ["gemini-2.0-flash"],
    "codex_models": ["codex-mini"],
    "opencode_models": {"anthropic": ["databricks-claude-sonnet-4"]},
    "managed_configs": {},
    "available_tools": TOOLS,
}


class TestHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # no_args_is_help=True exits with code 0 or 2 depending on typer version
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output

    def test_help_lists_all_agent_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for tool in TOOLS:
            assert tool in result.output

    @pytest.mark.parametrize("prog_name", ["ug", "ucode"])
    def test_help_uses_invoked_name_and_names_ucode_as_an_alias(self, prog_name):
        result = runner.invoke(app, ["--help"], prog_name=prog_name)
        output = _strip_ansi(result.output)

        assert result.exit_code == 0
        assert f"Usage: {prog_name}" in output
        assert "primary command is `ug`" in output
        assert "`ucode` remains supported as an alias" in output

    @pytest.mark.parametrize("tool", TOOLS)
    def test_subcommand_help(self, tool):
        result = runner.invoke(app, [tool, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_configure_help_lists_agents_flag(self):
        result = runner.invoke(app, ["configure", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        # Typer wraps long help text across lines and pads with box-drawing
        # characters; collapse whitespace + box chars before substring-matching.
        flat = re.sub(r"[│╭╮╯╰─\s]+", " ", output)
        assert "--agents" in output
        assert "comma-separated list of agents" in flat
        assert "--workspaces" in output


class TestProjectScripts:
    def test_ug_and_ucode_are_equivalent_entry_points(self):
        scripts = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())[
            "project"
        ]["scripts"]

        assert scripts["ug"] == "ucode.cli:main"
        assert scripts["ucode"] == "ucode.cli:main"


class TestUpgrade:
    @staticmethod
    def _ok() -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    @staticmethod
    def _which(command: str) -> str:
        return f"/tools/{command}"

    @staticmethod
    def _requirement(distribution: str) -> str:
        return f"{distribution} @ git+https://github.com/databricks/unity-gateway"

    def test_before_cutover_upgrades_ucode_normally_without_verification(self):
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="ucode"),
            patch("subprocess.run", return_value=self._ok()) as run,
        ):
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 0, result.output
        assert run.call_args_list == [
            call(
                ["uv", "tool", "install", "--reinstall", self._requirement("ucode")],
                check=False,
                capture_output=True,
                text=True,
            )
        ]
        assert "ucode upgraded" in result.output

    def test_cutover_migrates_legacy_distribution_and_verifies_commands(self):
        git_url = "git+https://github.com/databricks/unity-gateway"
        rename_failure = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr=("Package metadata name `unity-gateway` does not match given name `ucode`"),
        )
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="ucode"),
            patch("ucode.cli.shutil.which", side_effect=self._which),
            patch(
                "subprocess.run",
                side_effect=[
                    rename_failure,
                    self._ok(),
                    self._ok(),
                    self._ok(),
                    self._ok(),
                ],
            ) as run,
        ):
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 0, result.output
        assert run.call_args_list == [
            call(
                ["uv", "tool", "install", "--reinstall", self._requirement("ucode")],
                check=False,
                capture_output=True,
                text=True,
            ),
            call(["uv", "tool", "uninstall", "ucode"], check=True),
            call(["uv", "tool", "install", "--force", git_url], check=True),
            call(
                ["/tools/ug", "--version"],
                check=False,
                capture_output=True,
                text=True,
            ),
            call(
                ["/tools/ucode", "--version"],
                check=False,
                capture_output=True,
                text=True,
            ),
        ]
        assert "Migrated to `unity-gateway`" in result.output

    def test_after_cutover_upgrades_unity_gateway_normally_without_verification(self):
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="unity-gateway"),
            patch("subprocess.run", return_value=self._ok()) as run,
        ):
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 0, result.output
        run.assert_called_once_with(
            [
                "uv",
                "tool",
                "install",
                "--reinstall",
                self._requirement("unity-gateway"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert "unity-gateway upgraded" in result.output

    def test_unrelated_legacy_upgrade_failure_does_not_uninstall_ucode(self):
        failure = subprocess.CompletedProcess(
            [], 7, stdout="", stderr="Could not resolve host: github.com"
        )
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="ucode"),
            patch("subprocess.run", return_value=failure) as run,
        ):
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 1
        run.assert_called_once_with(
            ["uv", "tool", "install", "--reinstall", self._requirement("ucode")],
            check=False,
            capture_output=True,
            text=True,
        )
        assert "left unchanged" in result.output
        assert "ERROR 1" not in result.output

    def test_cutover_install_failure_has_recovery_command(self):
        rename_failure = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr=("Package metadata name `unity-gateway` does not match given name `ucode`"),
        )
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="ucode"),
            patch("subprocess.run") as run,
        ):
            run.side_effect = [
                rename_failure,
                self._ok(),
                subprocess.CalledProcessError(7, ["uv", "tool", "install"]),
            ]
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 1
        assert "legacy `ucode` tool was removed" in result.output
        assert "uv tool install --force git+https://github.com/databricks/unity-gateway" in re.sub(
            r"\s+", " ", result.output
        )

    def test_post_migration_verification_failure_is_actionable(self):
        rename_failure = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr=("Package metadata name `unity-gateway` does not match given name `ucode`"),
        )
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="ucode"),
            patch("ucode.cli.shutil.which", return_value=None),
            patch(
                "subprocess.run",
                side_effect=[rename_failure, self._ok(), self._ok()],
            ),
        ):
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 1
        assert "`ug` is not available on PATH" in result.output

    def test_missing_uv_is_actionable(self):
        with (
            patch("ucode.cli._installed_cli_distribution", return_value="ucode"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            result = runner.invoke(app, ["upgrade"])

        assert result.exit_code == 1
        assert "uv" in result.output.lower()

    def test_installed_distribution_prefers_unity_gateway(self):
        with patch("ucode.cli.metadata.version", return_value="1.0.0") as package_version:
            from ucode.cli import _installed_cli_distribution

            assert _installed_cli_distribution() == "unity-gateway"

        package_version.assert_called_once_with("unity-gateway")

    def test_installed_distribution_falls_back_to_ucode(self):
        def package_version(distribution_name: str) -> str:
            if distribution_name == "unity-gateway":
                raise metadata.PackageNotFoundError
            return "1.0.0"

        with patch("ucode.cli.metadata.version", side_effect=package_version):
            from ucode.cli import _installed_cli_distribution

            assert _installed_cli_distribution() == "ucode"


class TestVersion:
    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_prints_version_and_exits(self, flag):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0
        # Matches the derived version reported by importlib.metadata — either a
        # real string like "0.1.0" / "0.1.0+2.g93986a8" or the "unknown" fallback.
        assert _strip_ansi(result.output).strip() != ""

    def test_matches_telemetry_version(self):
        from ucode.telemetry import ucode_version

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert ucode_version() in _strip_ansi(result.output)


def _patch_launch(tool: str):
    """Return a context-manager stack that makes _launch_tool a no-op.

    load_state returns MINIMAL_STATE (workspace + tool already configured) so
    the auto-configure path is skipped entirely. configure_shared_state is
    also stubbed to avoid the launch-time refetch hitting the network.
    """
    return [
        patch("ucode.cli.ensure_bootstrap_dependencies"),
        patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
        patch(
            "ucode.cli.ensure_provider_state",
            return_value=MINIMAL_STATE,
        ),
        patch(
            "ucode.cli.configure_shared_state",
            return_value=MINIMAL_STATE,
        ),
        patch(
            "ucode.cli.resolve_launch_model",
            return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
        ),
        patch(
            "ucode.cli.configure_tool",
            return_value=MINIMAL_STATE,
        ),
        patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
        patch("ucode.cli.launch_agent"),
    ]


class TestSubcommandRouting:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_subcommand_calls_correct_tool(self, tool):
        patches = _patch_launch(tool)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7] as mock_launch,
        ):
            result = runner.invoke(app, [tool])
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        called_tool = mock_launch.call_args[0][0]
        assert called_tool == tool

    def test_no_agent_flag(self):
        """--agent flag must no longer exist."""
        result = runner.invoke(app, ["--agent", "claude"])
        assert result.exit_code != 0

    def test_workspace_flag_sets_current_workspace(self):
        """--workspace targets that workspace (normalized) before launch."""
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patch("ucode.cli.set_current_workspace") as mock_set,
        ):
            result = runner.invoke(
                app,
                ["claude", "--workspace", "https://eng-ml-inference.staging.cloud.databricks.com/"],
            )
        assert result.exit_code == 0, result.output
        mock_set.assert_called_once_with("https://eng-ml-inference.staging.cloud.databricks.com")

    def test_no_workspace_flag_leaves_current_workspace(self):
        """Without --workspace, launch never reassigns the current workspace."""
        patches = _patch_launch("claude")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patch("ucode.cli.set_current_workspace") as mock_set,
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_set.assert_not_called()

    def test_codex_enable_smart_routing_is_consumed_by_ucode(self):
        enabled_during_launch = []
        with patch(
            "ucode.cli._launch_tool",
            side_effect=lambda *_args, **_kwargs: enabled_during_launch.append(
                os.environ.get("ENABLE_SMART_ROUTING_V2")
            ),
        ) as mock_launch:
            result = runner.invoke(app, ["codex", "--enable-smart-routing"])

        assert result.exit_code == 0, result.output
        assert enabled_during_launch == ["1"]
        assert "ENABLE_SMART_ROUTING_V2" not in os.environ
        assert mock_launch.call_args.args[1].args == []

    def test_claude_enable_smart_routing_forwards_positional_prompt_to_v2(self):
        captured = []

        def capture(_tool, ctx, **_kwargs):
            captured.append((os.environ.get("ENABLE_SMART_ROUTING_V2"), ctx.args))

        with patch("ucode.cli._launch_tool", side_effect=capture):
            result = runner.invoke(
                app, ["claude", "--enable-smart-routing", "--", "fix the parser"]
            )

        assert result.exit_code == 0, result.output
        assert captured == [("1", ["fix the parser"])]

    @pytest.mark.parametrize(
        ("args", "forwarded", "has_separator"),
        [
            (["codex", "--", "fix the parser"], ["fix the parser"], True),
            (["codex", "--"], [], True),
            (["claude", "--", "doctor"], ["doctor"], True),
            (
                ["codex", "--model", "gpt-5.6-sol", "--", "fix the parser"],
                ["--model", "gpt-5.6-sol", "fix the parser"],
                False,
            ),
        ],
    )
    def test_agent_records_prompt_separator(self, args, forwarded, has_separator):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, args)

        assert result.exit_code == 0, result.output
        ctx = mock_launch.call_args.args[1]
        assert ctx.args == forwarded
        assert cli_mod._has_explicit_prompt(ctx) is has_separator

    @pytest.mark.parametrize("tool", ["codex", "claude"])
    @pytest.mark.parametrize(
        ("tool_args", "explicit_prompt", "model", "provider", "expected"),
        [
            ([], False, None, None, True),
            (["fix this"], True, None, None, True),
            (["fix this"], False, None, None, False),
            (["update"], False, None, None, False),
            (["--model", "fixed"], False, None, None, False),
            ([], False, "fixed", None, False),
            ([], False, None, "catalog.schema.service", False),
        ],
    )
    def test_codex_and_claude_share_smart_routing_policy(
        self, tool, tool_args, explicit_prompt, model, provider, expected
    ):
        options = cli_mod._launch_options(
            tool,
            tool_args,
            smart_routing_enabled=True,
            explicit_prompt=explicit_prompt,
            model=model,
            provider=provider,
        )

        assert options.launch_smart_routing is expected

    @pytest.mark.parametrize(
        ("tool_args", "expected"),
        [
            (["--session-id"], True),
            (["--session-id", "--verbose"], True),
            (["--session-id", "session-123"], True),
            (["update"], False),
            (["update", "--session-id"], False),
            (["--model", "fixed"], False),
            (["--session-id", "session-123", "--model", "fixed"], False),
        ],
    )
    def test_claude_options_allow_smart_routing_except_model(self, tool_args, expected):
        options = cli_mod._launch_options(
            "claude",
            tool_args,
            smart_routing_enabled=True,
            explicit_prompt=False,
            model=None,
            provider=None,
        )

        assert options.launch_smart_routing is expected

    def test_codex_refresh_is_consumed_by_ucode(self):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, ["codex", "--refresh"])

        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.kwargs["refresh"] is True

    def test_codex_forwarded_model_is_reported_in_launch_summary(self):
        state = {
            **MINIMAL_STATE,
            "codex_models": ["system.ai.gpt-5-6-luna"],
        }
        forwarded_args = ["--model", "system.ai.gpt-5-6-sol"]
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(state, "system.ai.gpt-5-6-luna"),
            ),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            result = runner.invoke(
                app,
                ["codex", "--workspace", "https://example.databricks.com", "--", *forwarded_args],
            )

        output = _strip_ansi(result.output)
        assert result.exit_code == 0, result.output
        assert "Model: system.ai.gpt-5-6-sol" in output
        assert "Model: system.ai.gpt-5-6-luna" not in output
        assert mock_launch.call_args.args[2] == forwarded_args

    def test_claude_enable_model_discovery_sets_ucode_env(self):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, ["claude", "--enable-model-discovery"])

        assert result.exit_code == 0, result.output
        assert os.environ["ENABLE_CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY"] == "1"
        assert mock_launch.call_args.args[1].args == []

    def test_claude_enable_model_discovery_is_hidden_from_help(self):
        result = runner.invoke(app, ["claude", "--help"])

        assert result.exit_code == 0, result.output
        assert "--enable-model-discovery" not in result.output

    def test_codex_disable_removes_hooks_without_launching(self):
        with (
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.codex_agent.disable_smart_routing") as mock_disable,
            patch("ucode.cli._launch_tool") as mock_launch,
        ):
            result = runner.invoke(app, ["codex", "--disable-smart-routing"])

        assert result.exit_code == 0, result.output
        mock_disable.assert_called_once_with(MINIMAL_STATE)
        mock_launch.assert_not_called()
        assert "routing hooks removed" in result.output

    def test_codex_routing_flags_are_mutually_exclusive(self):
        result = runner.invoke(
            app,
            ["codex", "--enable-smart-routing", "--disable-smart-routing"],
        )

        assert result.exit_code == 1
        assert "Use only one" in result.output

    @pytest.mark.parametrize("tool", ["codex", "claude"])
    def test_disable_smart_routing_is_hidden(self, tool):
        result = runner.invoke(app, [tool, "--help"])

        assert result.exit_code == 0, result.output
        assert "--disable-smart-routing" not in result.output

    def test_legacy_opt_in_migrates_both_agents(self):
        from ucode.cli import _migrate_legacy_smart_routing

        state = {**MINIMAL_STATE, "smart_routing_enabled": True}
        with (
            patch("ucode.cli.codex_agent.disable_smart_routing") as disable_codex,
            patch("ucode.cli.claude_agent.disable_smart_routing") as disable_claude,
        ):
            migrated = _migrate_legacy_smart_routing(state)

        assert migrated is state
        disable_codex.assert_called_once_with(state)
        disable_claude.assert_called_once_with(state)

    def test_claude_v2_skips_legacy_prelaunch_routing(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SMART_ROUTING_V2", "1")
        state = {
            **MINIMAL_STATE,
            "claude_models": {"opus": "system.ai.claude-opus-4-8"},
        }
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(state, "system.ai.claude-opus-4-8"),
            ),
            patch("ucode.cli.configure_tool", return_value=state) as mock_configure,
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            result = runner.invoke(app, ["claude"])

        assert result.exit_code == 0, result.output
        assert mock_configure.call_args.kwargs["route_root_model"] is None
        assert "_claude_launch_model" not in mock_launch.call_args.args[1]
        assert mock_launch.call_args.kwargs["options"].launch_smart_routing is True

    def test_claude_v2_first_prompt_hook_is_disabled_without_flag(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SMART_ROUTING_V2", raising=False)
        with patch("ucode.smart_routing.claude_pty.request_first_prompt_route") as mock_request:
            result = runner.invoke(
                app,
                ["claude-router-hook", "route-first-prompt", "--socket", "/tmp/v2.sock"],
                input='{"prompt":"fix the parser"}',
            )

        assert result.exit_code == 0, result.output
        assert result.output == ""
        mock_request.assert_not_called()

    @staticmethod
    def _invoke_codex_subagent_hook(token_env):
        routed = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {"message": "fix it", "model": "gpt-5.6-sol"},
            }
        }
        with (
            patch("ucode.cli.get_databricks_token", return_value="fresh-token") as mock_token,
            patch(
                "ucode.smart_routing.codex_routing.route_pre_tool_use",
                return_value=routed,
            ) as mock_route,
        ):
            result = runner.invoke(
                app,
                [
                    "codex-router-hook",
                    "route-subagent",
                    "--host",
                    "https://example.com",
                    "--profile",
                    "my-profile",
                    "--model",
                    "system.ai.gpt-5-6-sol",
                ],
                input='{"tool_name":"collaboration.spawn_agent","tool_input":{"message":"fix it"}}',
                env={"ENABLE_SMART_ROUTING_V2": "1", **token_env},
            )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == routed
        return mock_token, mock_route

    def test_codex_subagent_hook_reuses_fresh_oauth_token(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)
        token = _jwt(time.time() + 300)

        mock_token, mock_route = self._invoke_codex_subagent_hook({"OAUTH_TOKEN": token})

        mock_token.assert_not_called()
        assert mock_route.call_args.kwargs["token"] == token

    def test_codex_subagent_hook_refreshes_near_expiry_oauth_token(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)

        mock_token, mock_route = self._invoke_codex_subagent_hook(
            {"OAUTH_TOKEN": _jwt(time.time() + 90)}
        )

        mock_token.assert_called_once_with("https://example.com", "my-profile", force_refresh=True)
        assert mock_route.call_args.kwargs["token"] == "fresh-token"

    def test_codex_subagent_hook_refreshes_opaque_oauth_token(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)

        mock_token, mock_route = self._invoke_codex_subagent_hook({"OAUTH_TOKEN": "opaque-token"})

        mock_token.assert_called_once_with("https://example.com", "my-profile", force_refresh=True)
        assert mock_route.call_args.kwargs["token"] == "fresh-token"

    def test_codex_subagent_hook_reuses_bearer(self):
        mock_token, mock_route = self._invoke_codex_subagent_hook(
            {"DATABRICKS_BEARER": "pat-token", "OAUTH_TOKEN": "opaque-token"}
        )

        mock_token.assert_not_called()
        assert mock_route.call_args.kwargs["token"] == "pat-token"

    def test_claude_v2_subagent_hook_uses_v2_router(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SMART_ROUTING_V2", "1")
        routed = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {"prompt": "fix it", "model": "opus"},
            }
        }
        with (
            patch(
                "ucode.cli.smart_routing_v2.route_claude_pre_tool_use",
                return_value=routed,
            ) as mock_v2_route,
        ):
            result = runner.invoke(
                app,
                [
                    "claude-router-hook",
                    "route-subagent",
                    "--host",
                    "https://example.com",
                    "--model",
                    "system.ai.claude-opus-4-8",
                ],
                input='{"tool_name":"Agent","tool_input":{"prompt":"fix it"}}',
                env={"OAUTH_TOKEN": "token"},
            )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == routed
        mock_v2_route.assert_called_once()


class TestClaudeModelFlag:
    """`ucode claude --model <id>` pins the id into the family aliases so the gateway resolves any
    Databricks model id, instead of Claude Code's own --model flag rejecting non-catalog ids."""

    def test_model_threads_through_to_launch(self):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, ["claude", "--model", "cat.schema.claude-opus-5"])
        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.kwargs["model"] == "cat.schema.claude-opus-5"

    def test_refresh_threads_through_to_launch(self):
        with patch("ucode.cli._launch_tool") as mock_launch:
            result = runner.invoke(app, ["claude", "--refresh"])
        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.kwargs["refresh"] is True

    @pytest.mark.parametrize(
        "forwarded_args",
        [
            ["--model", "system.ai.claude-sonnet-5"],
            ["--model=system.ai.claude-sonnet-5"],
            ["-m", "system.ai.claude-sonnet-5"],
        ],
    )
    def test_forwarded_model_is_reported_in_launch_summary(self, forwarded_args):
        state = {
            **MINIMAL_STATE,
            "claude_models": {"opus": "system.ai.claude-opus-4-8"},
        }
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(state, "system.ai.claude-opus-4-8"),
            ),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            result = runner.invoke(
                app,
                ["claude", "--workspace", "https://example.databricks.com", "--", *forwarded_args],
            )

        output = _strip_ansi(result.output)
        assert result.exit_code == 0, result.output
        assert "Model: system.ai.claude-sonnet-5" in output
        assert "Model: system.ai.claude-opus-4-8" not in output
        assert mock_launch.call_args.args[2] == forwarded_args

    def test_model_is_launch_scoped_for_claude(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SMART_ROUTING_V2", raising=False)
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.resolve_launch_model", return_value=(MINIMAL_STATE, "system.ai.opus")),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE) as mock_configure,
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            result = runner.invoke(app, ["claude", "--model", "cat.schema.claude-opus-5"])
        assert result.exit_code == 0, result.output
        # The model is passed through invocation-scoped LaunchOptions, not persisted in settings.
        assert mock_configure.call_args.kwargs["custom_model"] is None
        assert mock_configure.call_args.kwargs["route_root_model"] is None
        assert (
            mock_launch.call_args.kwargs["options"].claude_launch_model
            == "cat.schema.claude-opus-5"
        )

    def test_v2_model_sets_transient_launch_override(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SMART_ROUTING_V2", "1")
        state = {**MINIMAL_STATE, "claude_models": {"opus": "system.ai.claude-opus-4-8"}}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.resolve_launch_model", return_value=(state, "system.ai.opus")),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            result = runner.invoke(app, ["claude", "--model", "system.ai.glm-5-2"])

        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.args[1]["_claude_launch_model"] == "system.ai.glm-5-2"
        assert mock_launch.call_args.kwargs["options"].launch_smart_routing is False

    @staticmethod
    def _provider_launch(monkeypatch, argv, provider_models, relayed=False):
        """Invoke a provider launch with model discovery/config stubbed, returning the
        configure_tool and launch_agent mocks so tests can assert what was threaded to each."""
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "ensure_bootstrap_dependencies", lambda *a, **k: None)
        monkeypatch.setattr(cli_mod, "load_state", lambda: MINIMAL_STATE)
        monkeypatch.setattr(cli_mod, "ensure_provider_state", lambda t: MINIMAL_STATE)
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: MINIMAL_STATE)
        monkeypatch.setattr(cli_mod, "_fetch_managed_config", lambda s: (None, False))
        monkeypatch.setattr(cli_mod, "_fetch_budget_recommendation", lambda s, m: None)
        mock_launch = MagicMock()
        monkeypatch.setattr(cli_mod, "launch_agent", mock_launch)
        monkeypatch.setattr(
            cli_mod, "resolve_provider_models", lambda t, s, p: (provider_models, None, relayed)
        )
        mock_configure = MagicMock(return_value=MINIMAL_STATE)
        monkeypatch.setattr(cli_mod, "configure_tool", mock_configure)
        result = runner.invoke(app, argv)
        return result, mock_configure, mock_launch

    def test_model_and_provider_now_pin_the_launch_tier(self, monkeypatch):
        # --model under a provider is no longer rejected: a family alias resolves to that tier's
        # declared target and is threaded as route_root_model (ANTHROPIC_MODEL), not custom_model.
        result, mock_configure, _ = self._provider_launch(
            monkeypatch,
            ["claude", "--model", "haiku", "--provider", "cat.schema.svc"],
            {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"},
        )
        assert result.exit_code == 0, result.output
        assert mock_configure.call_args.kwargs["route_root_model"] == "claude-haiku-4-5"
        assert mock_configure.call_args.kwargs["custom_model"] is None

    def test_provider_without_opus_auto_picks_best_servable_tier(self, monkeypatch):
        # No --model, and the service declares no opus target: launch on the most capable tier it
        # does offer (sonnet) instead of dead-ending on Claude Code's opus default.
        result, mock_configure, _ = self._provider_launch(
            monkeypatch,
            ["claude", "--provider", "cat.schema.svc"],
            {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"},
        )
        assert result.exit_code == 0, result.output
        assert mock_configure.call_args.kwargs["route_root_model"] == "claude-sonnet-5"

    def test_provider_with_opus_keeps_claude_default(self, monkeypatch):
        # Opus is offered, so Claude Code's own default already works — pin nothing (no ANTHROPIC_MODEL
        # and no duplicate /model picker row).
        result, mock_configure, _ = self._provider_launch(
            monkeypatch,
            ["claude", "--provider", "cat.schema.svc"],
            {"opus": "claude-opus-4-8", "sonnet": "claude-sonnet-5"},
        )
        assert result.exit_code == 0, result.output
        assert mock_configure.call_args.kwargs["route_root_model"] is None

    def test_model_family_not_offered_by_provider_errors(self, monkeypatch):
        result, _, _ = self._provider_launch(
            monkeypatch,
            ["claude", "--model", "opus", "--provider", "cat.schema.svc"],
            {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"},
        )
        assert result.exit_code == 1
        assert "does not offer a 'opus' model" in result.output

    def test_model_forwarded_to_claude_for_relayed_provider(self, monkeypatch):
        # Relayed = a subscription: --model rides Claude Code's own flag, not gateway env.
        result, mock_configure, mock_launch = self._provider_launch(
            monkeypatch,
            ["claude", "--model", "opus", "--provider", "cat.schema.svc"],
            None,
            relayed=True,
        )
        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.args[2] == ["--model", "opus"]
        assert mock_configure.call_args.kwargs["route_root_model"] is None
        assert mock_configure.call_args.kwargs["custom_model"] is None
        assert "ignored" not in _strip_ansi(result.output)

    def test_relayed_provider_without_model_forwards_nothing(self, monkeypatch):
        # No --model on a relayed launch: nothing to forward.
        result, _, mock_launch = self._provider_launch(
            monkeypatch,
            ["claude", "--provider", "cat.schema.svc"],
            None,
            relayed=True,
        )
        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.args[2] == []

    def test_provider_sets_transient_claude_launch_marker(self):
        state = dict(MINIMAL_STATE)
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.resolve_provider_models", return_value=(None, None, False)),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent") as mock_launch,
        ):
            result = runner.invoke(app, ["claude", "--provider", "main.default.anthropic"])

        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.args[1]["_claude_launch_provider"] == "main.default.anthropic"


class TestGeminiProviderLaunch:
    @staticmethod
    def _launch(monkeypatch, resolve_provider_models):
        state = dict(MINIMAL_STATE)
        monkeypatch.setattr("ucode.cli.ensure_bootstrap_dependencies", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: state)
        monkeypatch.setattr("ucode.cli.ensure_provider_state", lambda t: state)
        monkeypatch.setattr("ucode.cli.configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr("ucode.cli._fetch_managed_config", lambda s: (None, False))
        monkeypatch.setattr("ucode.cli.resolve_provider_models", resolve_provider_models)
        monkeypatch.setattr("ucode.cli.configure_tool", lambda *a, **k: state)
        monkeypatch.setattr(
            "ucode.cli.resolve_gemini_provider_model",
            lambda s, p, m: ("gemini-3.5-flash", None),
        )
        mock_launch = MagicMock()
        monkeypatch.setattr("ucode.cli.launch_agent", mock_launch)
        return runner.invoke(app, ["gemini", "--provider", "cat.schema.svc"])

    def test_prints_resolved_target_model(self, monkeypatch):
        # The concrete target gemini pins must show in the launch summary (not just the provider).
        result = self._launch(monkeypatch, lambda t, s, p: (None, None, False))
        assert result.exit_code == 0, result.output
        assert "Model: gemini-3.5-flash" in _strip_ansi(result.output)

    def test_skips_resolve_provider_models(self, monkeypatch):
        # Gemini resolves its own target, so the claude-family lookup must not run (no double fetch).
        mock_rpm = MagicMock(return_value=(None, None, False))
        result = self._launch(monkeypatch, mock_rpm)
        assert result.exit_code == 0, result.output
        mock_rpm.assert_not_called()


class TestMcpSubcommands:
    def test_web_search_subcommand_help(self):
        result = runner.invoke(app, ["mcp", "web-search", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_mcp_group_lists_web_search(self):
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "web-search" in result.output


class TestAuthTokenCommand:
    """`ucode auth-token` is the cross-platform apiKeyHelper (#116)."""

    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # The --use-pat path writes DATABRICKS_BEARER directly; restore it so
        # writes by code under test don't leak into other tests.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_prints_only_the_token_to_stdout(self):
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch("ucode.cli.get_databricks_token", return_value="tok-123") as fetch,
        ):
            result = runner.invoke(app, ["auth-token"])
        assert result.exit_code == 0
        # Nothing but the bare token (plus trailing newline) may reach stdout,
        # or the consuming agent will treat the noise as part of the token.
        assert result.stdout == "tok-123\n"
        fetch.assert_called_once_with("https://ws", None, force_refresh=False)

    def test_host_and_profile_override_state(self):
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://saved"}),
            patch("ucode.cli.get_databricks_token", return_value="tok") as fetch,
        ):
            result = runner.invoke(
                app, ["auth-token", "--host", "https://override", "--profile", "prod"]
            )
        assert result.exit_code == 0
        fetch.assert_called_once_with("https://override", "prod", force_refresh=False)

    def test_force_refresh_is_forwarded(self):
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch("ucode.cli.get_databricks_token", return_value="tok") as fetch,
        ):
            result = runner.invoke(app, ["auth-token", "--force-refresh"])
        assert result.exit_code == 0
        fetch.assert_called_once_with("https://ws", None, force_refresh=True)

    def test_errors_without_workspace(self):
        with patch("ucode.cli.load_state", return_value={}):
            result = runner.invoke(app, ["auth-token"])
        assert result.exit_code == 1
        # The error goes to stderr, never stdout.
        assert result.stdout == ""

    def test_hidden_from_top_level_help(self):
        result = runner.invoke(app, ["--help"])
        assert "auth-token" not in _strip_ansi(result.output)

    def test_use_pat_emits_resolved_pat(self, monkeypatch):
        # --use-pat reads the profile's static PAT, exports it as
        # DATABRICKS_BEARER, and get_databricks_token returns it directly.
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: "dapi-pat")
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch(
                "ucode.cli.get_databricks_token",
                side_effect=lambda w, p, **_kwargs: os.environ.get("DATABRICKS_BEARER", ""),
            ),
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 0
        assert result.stdout == "dapi-pat\n"

    def test_use_pat_ignores_empty_bearer_env(self, monkeypatch):
        # A stray empty DATABRICKS_BEARER must not shadow the PAT and force the
        # OAuth path (the regression that motivated ensure_pat_bearer).
        monkeypatch.setenv("DATABRICKS_BEARER", "")
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: "dapi-pat")
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch(
                "ucode.cli.get_databricks_token",
                side_effect=lambda w, p, **_kwargs: os.environ.get("DATABRICKS_BEARER", ""),
            ),
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 0
        assert result.stdout == "dapi-pat\n"

    def test_use_pat_fails_closed_without_pat(self, monkeypatch):
        # --use-pat with no resolvable PAT must error, NOT fall through to OAuth
        # (which can't serve a PAT-only profile and yields a misleading message).
        monkeypatch.delenv("DATABRICKS_BEARER", raising=False)
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: None)
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch("ucode.cli.get_databricks_token", return_value="oauth-tok") as fetch,
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 1
        # Never attempted OAuth, and nothing leaked to stdout.
        fetch.assert_not_called()
        assert result.stdout == ""

    def test_use_pat_honors_non_empty_bearer_env(self, monkeypatch):
        # A real pre-set bearer (CI escape hatch) wins over the profile PAT.
        monkeypatch.setenv("DATABRICKS_BEARER", "ci-bearer")
        monkeypatch.setattr("ucode.databricks.resolve_pat_token", lambda p: "dapi-pat")
        with (
            patch("ucode.cli.load_state", return_value={"workspace": "https://ws"}),
            patch(
                "ucode.cli.get_databricks_token",
                side_effect=lambda w, p, **_kwargs: os.environ.get("DATABRICKS_BEARER", ""),
            ),
        ):
            result = runner.invoke(app, ["auth-token", "--use-pat", "--profile", "p"])
        assert result.exit_code == 0
        assert result.stdout == "ci-bearer\n"


class TestStatus:
    def test_shows_mcp_list_commands(self):
        with patch("ucode.cli.load_state", return_value=MINIMAL_STATE):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Managed by Databricks" not in result.output
        assert "MCP list command:" in result.output
        assert "claude mcp list" in result.output
        assert "codex mcp list" in result.output
        assert "gemini mcp list" in result.output
        assert "opencode mcp list" in result.output
        assert "copilot mcp list" not in result.output

    def test_shows_mcp_servers_configured_by_ucode(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": "https://example.databricks.com/api/2.0/mcp/external/github-mcp",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude", "codex"],
                },
                {
                    "name": "databricks-sql",
                    "url": "https://example.databricks.com/api/2.0/mcp/sql",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["gemini"],
                },
            ],
        }
        with patch("ucode.cli.load_state", return_value=state):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "github-mcp" in result.output
        assert "MCP servers: github-mcp" in result.output
        assert "databricks-sql" in result.output
        assert "MCP servers: databricks-sql" in result.output
        assert "MCP Servers" not in result.output
        assert "MCP Server:" not in result.output
        assert "Configured tools:" not in result.output

    def test_status_treats_available_tools_as_configured_agents(self):
        state = {
            **MINIMAL_STATE,
            "available_tools": ["copilot"],
            "base_urls": {
                **MINIMAL_STATE["base_urls"],
                "copilot": "https://example.databricks.com/ai-gateway/copilot",
            },
            "mcp_servers": [
                {
                    "name": "databricks-sql",
                    "url": "https://example.databricks.com/api/2.0/mcp/sql",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["copilot"],
                }
            ],
        }
        with patch("ucode.cli.load_state", return_value=state):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "copilot mcp list" in result.output
        assert "MCP servers: databricks-sql" in result.output
        assert "codex mcp list" not in result.output
        assert "claude mcp list" not in result.output
        assert "gemini mcp list" not in result.output
        assert "https://example.databricks.com/ai-gateway/anthropic" not in result.output
        assert "https://example.databricks.com/ai-gateway/gemini" not in result.output

    def test_status_shows_managed_config_box_when_present_and_enabled(self, monkeypatch):
        managed = {
            "enabled_agents": {"claude": {}, "codex": {}},
            "mcp_servers": [{"name": "github-mcp", "type": "external"}],
            "skills": {"names": ["debug-ci"]},
        }
        with (
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.load_managed_state", return_value=managed),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Workspace-managed config" in result.output
        assert "Enabled agents:" in result.output
        assert "github-mcp" in result.output
        assert "debug-ci" in result.output

    def test_status_hides_managed_config_box_when_none_present(self, monkeypatch):
        with (
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.load_managed_state", return_value=None),
        ):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Workspace-managed config" not in result.output


class TestConfigureSkillsCommand:
    def test_mcp_flag_dispatches_location_set(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with(["a.b"])

    def test_comma_location_yields_multiple_schemas(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b, c.d", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with(["a.b", "c.d"])

    def test_default_mode_dispatches_download_with_path(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--path", "/tmp/skills"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path="/tmp/skills", skills=None)

    def test_default_mode_without_path_dispatches_download(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b"])
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills=None)

    def test_skill_filter_dispatches_download_with_subset(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--skill", "my_skill"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"my_skill"})

    def test_skill_filter_parses_comma_list(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--skill", "s1, s2"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"s1", "s2"})

    def test_skill_with_mcp_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--mcp", "--skill", "my_skill"]
            )
        assert result.exit_code == 1
        assert "--skill" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()

    def test_skill_without_location_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(app, ["configure", "skills", "--skill", "my_skill"])
        assert result.exit_code == 1
        assert "--skill" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()

    def test_skill_with_multiple_locations_exit_1(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b, c.d", "--skill", "my_skill"]
            )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "--skill requires a single --location" in output
        mock_download.assert_not_called()

    def test_path_with_mcp_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(
                app, ["configure", "skills", "--location", "a.b", "--mcp", "--path", "/tmp/skills"]
            )
        assert result.exit_code == 1
        assert "--path" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()

    def test_three_part_location_exit_1(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "a.b.c", "--mcp"])
        assert result.exit_code == 1
        mock_mcp.assert_not_called()

    def test_malformed_location_exit_1_names_location(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--location", "justone", "--mcp"])
        assert result.exit_code == 1
        assert "--location" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()

    def test_bare_command_registers_schemaless_connection(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with([])

    def test_mcp_without_location_registers_schemaless_connection(self):
        with patch("ucode.cli.configure_skills_mcp_command") as mock_mcp:
            result = runner.invoke(app, ["configure", "skills", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_mcp.assert_called_once_with([])

    def test_path_without_location_exit_1(self):
        with (
            patch("ucode.cli.configure_skills_mcp_command") as mock_mcp,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(app, ["configure", "skills", "--path", "/tmp/skills"])
        assert result.exit_code == 1
        assert "--path" in _strip_ansi(result.output)
        mock_mcp.assert_not_called()
        mock_download.assert_not_called()


class TestSkillsAddCommand:
    """`ucode skill add` is the additive sibling of `configure skills`: `--mcp`
    unions schemas into the connection scope, the default mode downloads."""

    def test_mcp_flag_unions_locations(self):
        with patch("ucode.cli.add_skills_command") as mock_add:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_add.assert_called_once_with(["a.b"])

    def test_comma_location_yields_multiple_schemas(self):
        with patch("ucode.cli.add_skills_command") as mock_add:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b, c.d", "--mcp"])
        assert result.exit_code == 0, result.output
        mock_add.assert_called_once_with(["a.b", "c.d"])

    def test_default_mode_dispatches_download(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b", "--path", "/tmp/s"])
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path="/tmp/s", skills=None)

    def test_skill_filter_dispatches_download_with_subset(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b", "--skills", "s1, s2"])
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"s1", "s2"})

    def test_fully_qualified_skills_derive_location(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--skills", "a.b.s1, a.b.s2"])
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"s1", "s2"})

    def test_fully_qualified_skills_match_explicit_location(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["skill", "add", "--location", "a.b", "--skills", "a.b.s1, a.b.s2"]
            )
        assert result.exit_code == 0, result.output
        mock_download.assert_called_once_with(["a.b"], path=None, skills={"s1", "s2"})

    def test_fully_qualified_skills_must_match_explicit_location(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b", "--skills", "c.d.s1"])
        assert result.exit_code == 1
        assert "must match --location `a.b`" in _strip_ansi(result.output)
        mock_download.assert_not_called()

    @pytest.mark.parametrize("skill", ["a.b", "a..s1", "a.b.c.d"])
    def test_malformed_fully_qualified_skill_exit_1(self, skill):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b", "--skills", skill])
        assert result.exit_code == 1
        assert "must be bare names or fully qualified" in _strip_ansi(result.output)
        mock_download.assert_not_called()

    def test_bare_skills_without_location_exit_1(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--skills", "s1"])
        assert result.exit_code == 1
        assert "--skills short names need --location" in _strip_ansi(result.output)
        mock_download.assert_not_called()

    def test_fully_qualified_skills_across_schemas_exit_1(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(app, ["skill", "add", "--skills", "a.b.s1, c.d.s2"])
        assert result.exit_code == 1
        assert "must all share one" in _strip_ansi(result.output)
        mock_download.assert_not_called()

    def test_without_location_exit_1(self):
        with (
            patch("ucode.cli.add_skills_command") as mock_add,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(app, ["skill", "add"])
        assert result.exit_code == 1
        assert "--location is required" in _strip_ansi(result.output)
        mock_add.assert_not_called()
        mock_download.assert_not_called()

    def test_skill_with_mcp_exit_1(self):
        with (
            patch("ucode.cli.add_skills_command") as mock_add,
            patch("ucode.cli.configure_skills_download_command") as mock_download,
        ):
            result = runner.invoke(
                app, ["skill", "add", "--location", "a.b", "--mcp", "--skills", "s1"]
            )
        assert result.exit_code == 1
        assert "--skills" in _strip_ansi(result.output)
        mock_add.assert_not_called()
        mock_download.assert_not_called()

    def test_path_with_mcp_exit_1(self):
        with patch("ucode.cli.add_skills_command") as mock_add:
            result = runner.invoke(
                app, ["skill", "add", "--location", "a.b", "--mcp", "--path", "/tmp/s"]
            )
        assert result.exit_code == 1
        assert "--path" in _strip_ansi(result.output)
        mock_add.assert_not_called()

    def test_skill_with_multiple_locations_exit_1(self):
        with patch("ucode.cli.configure_skills_download_command") as mock_download:
            result = runner.invoke(
                app, ["skill", "add", "--location", "a.b, c.d", "--skills", "s1"]
            )
        assert result.exit_code == 1
        assert "--skills requires a single --location" in _strip_ansi(result.output)
        mock_download.assert_not_called()

    def test_malformed_location_exit_1(self):
        with patch("ucode.cli.add_skills_command") as mock_add:
            result = runner.invoke(app, ["skill", "add", "--location", "a.b.c", "--mcp"])
        assert result.exit_code == 1
        assert "--location" in _strip_ansi(result.output)
        mock_add.assert_not_called()


class TestManagedSkillsOnLaunch:
    """Managed skills are delivered by download only: the launch path downloads them and never
    registers them on the skills MCP connection (only a developer's own `skill add --mcp` schemas
    live there)."""

    def _state(self):
        return {"workspace": "https://example.databricks.com", "profile": "prod"}

    def _launch(self, monkeypatch, *, managed):
        state = dict(MINIMAL_STATE)
        monkeypatch.setattr("ucode.cli.get_model_recommendation", lambda ws, tok: (None, None))
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli.get_databricks_token", return_value="tok"),
            patch("ucode.cli._fetch_managed_config", return_value=(managed, False)),
            patch("ucode.cli.apply_managed_mcp_servers", return_value=[]),
            patch("ucode.cli.launch_agent"),
            patch(
                "ucode.cli.download_managed_skills_on_launch", return_value=["main.default"]
            ) as mock_dl,
        ):
            result = runner.invoke(app, ["claude"])
        return result, state, mock_dl

    def test_launch_downloads_managed_skills_and_skips_mcp_registration(self, monkeypatch):
        from ucode import cli

        managed = {
            "enabled_agents": {"claude": {}},
            "skills": {"names": ["main.default", "ml.prod"]},
        }
        result, state, mock_dl = self._launch(monkeypatch, managed=managed)

        assert result.exit_code == 0, result.output
        mock_dl.assert_called_once_with(
            "https://example.databricks.com", "tok", ["main.default", "ml.prod"]
        )
        skills = [
            s for s in (state.get("mcp_servers") or []) if s.get("kind") == cli.SKILLS_MCP_KIND
        ]
        assert skills == []

    def test_no_managed_skills_skips_the_download(self):
        with (
            patch("ucode.cli.get_databricks_token") as mock_token,
            patch("ucode.cli.download_managed_skills_on_launch") as mock_dl,
        ):
            from ucode import cli

            cli._download_managed_skills({}, self._state())

        mock_token.assert_not_called()
        mock_dl.assert_not_called()

    def test_download_failure_never_blocks_launch(self):
        with (
            patch("ucode.cli.get_databricks_token", side_effect=RuntimeError("no auth")),
            patch("ucode.cli.download_managed_skills_on_launch") as mock_dl,
        ):
            from ucode import cli

            # Must not raise.
            cli._download_managed_skills({"skills": {"names": ["main.default"]}}, self._state())

        mock_dl.assert_not_called()


class TestStatusSkillsSection:
    def _run(self, state):
        with patch("ucode.cli.load_state", return_value=state):
            return runner.invoke(app, ["status"])

    def test_not_configured_when_no_skills_entry(self):
        result = self._run(MINIMAL_STATE)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "Skills" in out
        assert "not configured" in out

    def test_renders_locations_and_configured_agents(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "databricks-skill-registry",
                    "kind": "skills",
                    "skill_locations": ["main.default", "ml.prod"],
                    "url": "https://example.databricks.com/ai-gateway/skills/?schema=main.default&schema=ml.prod",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude", "codex"],
                }
            ],
        }
        result = self._run(state)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "Skill MCP Locations: main.default, ml.prod" in out
        assert "Configured: Claude Code, Codex" in out

    def test_renders_placeholder_when_no_locations(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "databricks-skill-registry",
                    "kind": "skills",
                    "skill_locations": [],
                    "url": "https://example.databricks.com/ai-gateway/skills/",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                }
            ],
        }
        result = self._run(state)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        assert "Skill MCP Locations: none — utility tools only" in out

    def test_skills_entry_absent_from_per_client_mcp_lines(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [
                {
                    "name": "github-mcp",
                    "url": "https://example.databricks.com/api/2.0/mcp/external/github-mcp",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                },
                {
                    "name": "databricks-skill-registry",
                    "kind": "skills",
                    "skill_locations": ["main.default"],
                    "url": "https://example.databricks.com/ai-gateway/skills/?schema=main.default",
                    "auth": "env:OAUTH_TOKEN",
                    "clients": ["claude"],
                },
            ],
        }
        result = self._run(state)
        assert result.exit_code == 0, result.output
        out = _strip_ansi(result.output)
        # The skills registry is managed in the Skills section, never listed on
        # a per-client "MCP servers:" line.
        for line in out.splitlines():
            if "MCP servers:" in line:
                assert "databricks-skill-registry" not in line
        assert "Skill MCP Locations: main.default" in out


class TestRevert:
    def test_reverts_mcp_configs_before_clearing_state(self):
        state = {
            **MINIMAL_STATE,
            "mcp_servers": [{"name": "github-mcp", "clients": ["claude"]}],
        }
        reverted_mcp: list[dict] = []
        cleared: list[bool] = []

        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.restore_file", return_value=False),
            patch(
                "ucode.cli.revert_mcp_configs",
                side_effect=lambda loaded_state: (
                    reverted_mcp.append(loaded_state) or {"claude": True}
                ),
            ),
            patch("ucode.cli.clear_state", side_effect=lambda: cleared.append(True)),
        ):
            result = runner.invoke(app, ["revert"])

        assert result.exit_code == 0, result.output
        assert reverted_mcp == [state]
        assert cleared == [True]
        assert "Claude Code MCP config: restored" in result.output


class TestDoctorCommand:
    def test_invokes_doctor(self):
        with patch("ucode.doctor.doctor", return_value=0) as mock_doctor:
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, result.output
        mock_doctor.assert_called_once_with()

    def test_reports_runtime_error(self):
        with patch("ucode.doctor.doctor", side_effect=RuntimeError("boom")):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "boom" in _strip_ansi(result.output)


class TestAutoConfigureOnFirstRun:
    def test_triggers_when_no_workspace(self):
        """Auto-configure runs when state has no workspace."""
        empty_state = {}
        configured_state = {**MINIMAL_STATE}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=empty_state),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=configured_state,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(configured_state, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=configured_state),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_bootstrap.assert_called_once_with("claude", update_existing=True)
        mock_auto.assert_called_once_with("claude")

    def test_triggers_when_tool_not_in_available_tools(self):
        """Auto-configure runs when workspace exists but the tool wasn't configured."""
        state_without_tool = {**MINIMAL_STATE, "available_tools": ["codex"]}
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=state_without_tool),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=MINIMAL_STATE,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        mock_bootstrap.assert_called_once_with("claude", update_existing=True)
        mock_auto.assert_called_once_with("claude")

    def test_skipped_when_already_configured(self):
        """Auto-configure is skipped when workspace and tool are already set up."""
        with (
            patch("ucode.cli.ensure_bootstrap_dependencies") as mock_bootstrap,
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli._auto_configure_tool") as mock_auto,
            patch("ucode.cli.configure_shared_state", return_value=MINIMAL_STATE),
            patch(
                "ucode.cli.ensure_provider_state",
                return_value=MINIMAL_STATE,
            ),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent"),
        ):
            runner.invoke(app, ["claude"])
        mock_bootstrap.assert_called_once_with("claude", update_existing=False)
        mock_auto.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("claude", "Launching Claude Code with Unity Gateway"),
        ("codex", "Launching Codex with Unity Gateway"),
        ("gemini", "Launching Gemini CLI with Unity Gateway"),
        ("opencode", "Launching OpenCode with Unity Gateway"),
        ("copilot", "Launching GitHub Copilot CLI with Unity Gateway"),
        ("pi", "Launching Pi with Unity Gateway"),
    ],
)
def test_launch_title(tool, expected):
    from ucode.cli import _launch_title

    assert _launch_title(tool) == expected


def test_cursor_launch_uses_unity_gateway_branding():
    with (
        patch("ucode.cli.shutil.which", return_value="/usr/local/bin/cursor-agent"),
        patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
        patch("ucode.agents.cursor.launch"),
    ):
        result = runner.invoke(app, ["cursor"])

    assert result.exit_code == 0, result.output
    assert "Unity Gateway with Cursor" in result.output


class TestConfigureAgentFlag:
    def test_no_flag_calls_configure_all(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=True, offer_optional_setup=True)

    def test_optional_setup_installs_ai_tools_and_configures_mcp(self):
        import ucode.cli as cli_mod

        state = {"available_tools": ["claude", "codex"]}
        with (
            patch("ucode.cli.prompt_yes_no", return_value=True) as mock_prompt,
            patch("ucode.cli.save_state") as mock_save,
            patch("ucode.cli.install_databricks_ai_tools_for_agents") as mock_install,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            cli_mod._configure_optional_setup(state, ["claude", "codex"])

        mock_prompt.assert_called_once_with("Configure MCP servers, skills, and plugins?")
        assert state["databricks_ai_tools_enabled"] is True
        mock_save.assert_called_once_with(state)
        mock_install.assert_called_once_with(["claude", "codex"], state)
        mock_mcp.assert_called_once_with()

    def test_optional_setup_decline_does_nothing(self):
        import ucode.cli as cli_mod

        with (
            patch("ucode.cli.prompt_yes_no", return_value=False),
            patch("ucode.cli.save_state") as mock_save,
            patch("ucode.cli.install_databricks_ai_tools_for_agents") as mock_install,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            state = {}
            cli_mod._configure_optional_setup(state, ["claude"])

        assert state["databricks_ai_tools_enabled"] is False
        mock_save.assert_called_once_with(state)
        mock_install.assert_not_called()
        mock_mcp.assert_not_called()

    def test_agents_flag_skips_mcp_prompt(self):
        # Flag-driven (non-interactive) runs must stay scriptable: no MCP prompt.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command"),
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output

    def test_agents_flag_calls_configure_with_tools(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_install.assert_not_called()
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
        )

    def test_agents_flag_normalizes_aliases_and_dedupes(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", " claude-code, codex,claude "])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
        )

    def test_workspaces_flag_calls_configure_with_workspaces(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--workspaces",
                    "first.databricks.com,https://second.databricks.com/",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            workspaces=[
                ("https://first.databricks.com", None),
                ("https://second.databricks.com", None),
            ],
            prompt_optional_updates=True,
        )

    def test_agents_and_workspaces_flags_call_configure_with_both(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agents", "claude,codex", "--workspaces", "https://first.com"],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.com", None)],
            prompt_optional_updates=True,
        )

    def test_agent_and_workspaces_flags_call_configure_with_both(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agent", "claude", "--workspaces", "https://first.com"],
            )
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", workspaces=[("https://first.com", None)])

    def test_agent_flag_calls_configure_with_tool(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude")

    def test_disable_fable_alone_implicitly_targets_claude(self):
        # Fable is Claude-only, so `--disable-fable` on its own should configure
        # claude directly instead of dropping into the interactive agent picker.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--disable-fable"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", fable_enabled=False)

    def test_enable_fable_alone_implicitly_targets_claude(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--enable-fable"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=True
        )
        mock_cfg.assert_called_once_with("claude", fable_enabled=True)

    def test_enable_fable_with_explicit_agents_does_not_override(self):
        # An explicit --agents selection wins; the fable flag rides along without
        # forcing the claude-only single-agent path.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--enable-fable", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
            fable_enabled=True,
        )

    def test_skip_upgrade_flag_disables_optional_update_prompt(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            # Fully-interactive configure ends by offering the MCP step; decline it.
            patch("ucode.cli.prompt_yes_no", return_value=False),
            patch("ucode.cli.configure_mcp_command"),
        ):
            result = runner.invoke(app, ["configure", "--skip-upgrade"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(prompt_optional_updates=False, offer_optional_setup=True)

    def test_disable_databricks_ai_tools_forwards_false_and_skips_prompt(self):
        # An explicit flag suppresses the interactive prompt and forwards the choice.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            # Fully-interactive configure ends by offering the MCP step; decline it.
            patch("ucode.cli.prompt_yes_no", return_value=False),
            patch("ucode.cli.configure_mcp_command"),
        ):
            result = runner.invoke(app, ["configure", "--disable-databricks-ai-tools"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            prompt_optional_updates=True, databricks_ai_tools_enabled=False
        )

    def test_enable_databricks_ai_tools_with_agents_forwards_true(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app, ["configure", "--enable-databricks-ai-tools", "--agents", "claude,codex"]
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=True,
            databricks_ai_tools_enabled=True,
        )

    def test_skip_upgrade_flag_with_agent_skips_optional_update(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary") as mock_install,
            patch("ucode.cli.configure_workspace_command"),
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--skip-upgrade"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once_with(
            "claude", strict=True, update_existing=True, prompt_optional_updates=False
        )

    def test_skip_upgrade_flag_with_agents_forwards_to_configure(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex", "--skip-upgrade"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            prompt_optional_updates=False,
        )

    def test_agent_flag_normalizes_alias(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude-code"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with("claude")

    def test_agent_flag_rejects_unknown(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "bogus"])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_agents_flag_rejects_unknown(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,bogus"])
        assert result.exit_code != 0
        assert "Unsupported tool 'bogus'" in result.output
        assert "codex, claude, gemini, opencode, copilot, pi" in " ".join(result.output.split())
        mock_cfg.assert_not_called()

    def test_agents_flag_rejects_empty_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", ","])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_agent_and_agents_flags_are_mutually_exclusive(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--agents", "codex"])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()

    def test_workspaces_flag_rejects_empty_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--workspaces", ","])
        assert result.exit_code != 0
        mock_cfg.assert_not_called()


class TestConfigureMcpFlag:
    def test_mcp_with_agents_configures_then_registers_services(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(
                app,
                ["configure", "--agents", "claude", "--mcp", "system.ai.slack,system.ai.github"],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude"],
            prompt_optional_updates=True,
        )
        mock_mcp.assert_called_once_with(services={"system.ai.slack", "system.ai.github"})

    def test_mcp_only_configures_workspace_without_agent_picker(self):
        # `--mcp` with no --agents (e.g. Cursor): configure the workspace directly,
        # never the interactive agent picker, then register the MCP service.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
            patch("ucode.cli._configure_shared_workspace_states") as mock_shared,
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--workspaces",
                    "https://ws.databricks.com",
                    "--mcp",
                    "system.ai.slack",
                ],
            )
        assert result.exit_code == 0, result.output
        # Never the model-agent picker path.
        mock_cfg.assert_not_called()
        mock_shared.assert_called_once()
        # Workspace-only: no model tools fetched.
        assert (
            mock_shared.call_args.kwargs.get("tools") == [] or mock_shared.call_args.args[1] == []
        )
        mock_mcp.assert_called_once_with(services={"system.ai.slack"})

    def test_mcp_rejects_bare_short_name(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command"),
            patch("ucode.cli._configure_shared_workspace_states"),
            patch("ucode.cli.configure_mcp_command") as mock_mcp,
        ):
            result = runner.invoke(
                app, ["configure", "--workspaces", "https://ws.databricks.com", "--mcp", "slack"]
            )
        assert result.exit_code != 0
        mock_mcp.assert_not_called()


class TestConfigureAgentsSelection:
    def test_selected_tools_skip_picker(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(
            cli_mod, "check_gateway_endpoint", lambda state, tool: tool in {"claude", "codex"}
        )
        monkeypatch.setattr(
            cli_mod,
            "prompt_for_tools",
            lambda available: pytest.fail("prompt_for_tools should not be called"),
        )
        install_calls: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "install_tool_binary",
            lambda tool, strict=False, update_existing=False, prompt_optional_updates=True: (
                install_calls.append(tool) or True
            ),
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command(selected_tools=["claude", "codex"]) == 0
        assert install_calls == ["claude", "codex"]
        assert configured == [["claude", "codex"]]

    def test_provider_picker_gated_by_interactive_path(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda s, t: t == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *a, **k: True)
        monkeypatch.setattr(
            cli_mod, "configure_selected_tools", lambda s, tools: {**s, "available_tools": tools}
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda s: None)
        picked_for: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "_maybe_select_provider_service",
            lambda tool, s: picked_for.append(tool) or s,
        )

        # Non-interactive (--agents passed): no provider picker.
        cli_mod.configure_workspace_command(
            selected_tools=["claude"], workspaces=[("https://w.com", None)]
        )
        assert picked_for == []

        # Interactive (`ucode configure`): picker offered for each picked tool.
        monkeypatch.setattr(
            cli_mod, "_prompt_for_configuration", lambda tool=None: ("https://w.com", None)
        )
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda options: ["claude"])
        cli_mod.configure_workspace_command()
        assert picked_for == ["claude"]

    def test_unavailable_selected_tool_errors_before_configure(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://example.com", None),
        )
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *args, **kwargs: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: tool == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: pytest.fail("configure_selected_tools should not be called"),
        )

        with pytest.raises(RuntimeError, match="Codex"):
            cli_mod.configure_workspace_command(selected_tools=["claude", "codex"])

    def test_strict_error_mentions_skip_unavailable(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: tool == "claude")
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *a, **k: None)

        with pytest.raises(RuntimeError, match="--skip-unavailable"):
            cli_mod.configure_workspace_command(
                selected_tools=["claude", "codex"],
                workspaces=[("https://example.com", None)],
            )

    def test_skip_unavailable_configures_available_subset(self, monkeypatch):
        """A workspace with no OpenAI models still configures claude and pi."""
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(
            cli_mod, "check_gateway_endpoint", lambda state, tool: tool in {"claude", "pi"}
        )
        installed: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "install_tool_binary",
            lambda tool, **kwargs: installed.append(tool) or True,
        )
        configured: list[list[str]] = []
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: configured.append(tools) or {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)
        warnings: list[str] = []
        monkeypatch.setattr(cli_mod, "print_warning", lambda msg: warnings.append(msg))

        assert (
            cli_mod.configure_workspace_command(
                selected_tools=["claude", "codex", "pi"],
                workspaces=[("https://example.com", None)],
                skip_unavailable=True,
            )
            == 0
        )
        # Order of the original --agents list is preserved, minus codex.
        assert configured == [["claude", "pi"]]
        assert installed == ["claude", "pi"]
        assert any("Codex" in msg for msg in warnings)

    def test_skip_unavailable_still_fails_when_none_available(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "available_tools": []}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: False)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: pytest.fail("configure_selected_tools should not be called"),
        )

        assert (
            cli_mod.configure_workspace_command(
                selected_tools=["codex"],
                workspaces=[("https://example.com", None)],
                skip_unavailable=True,
            )
            == 1
        )

    def test_picker_selected_profile_flows_to_configure_shared_state(self, monkeypatch):
        """Picker's (host, profile) tuple must reach configure_shared_state's
        `profile` kwarg, otherwise downstream --profile calls fall back to
        host-based resolution and silently pick the wrong profile."""
        import ucode.cli as cli_mod

        monkeypatch.setattr(
            cli_mod,
            "_prompt_for_configuration",
            lambda tool=None: ("https://shared.cloud.databricks.com", "picked-profile"),
        )
        captured: dict = {}

        def fake_configure_shared_state(
            workspace,
            profile=None,
            tools=None,
            force_login=False,
            use_pat=False,
            fable_enabled=None,
            databricks_ai_tools_enabled=None,
            clear_custom_oauth=False,
        ):
            captured["workspace"] = workspace
            captured["profile"] = profile
            return {**MINIMAL_STATE, "workspace": workspace, "profile": profile}

        monkeypatch.setattr(cli_mod, "configure_shared_state", fake_configure_shared_state)
        monkeypatch.setattr(cli_mod, "save_state", lambda state: (None, False))
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: True)
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["claude"])
        monkeypatch.setattr(cli_mod, "_maybe_select_provider_service", lambda tool, state: state)
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: {**state, "available_tools": tools},
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert cli_mod.configure_workspace_command() == 0
        assert captured["profile"] == "picked-profile"

    def test_multiple_workspaces_configure_all_and_use_first(self, monkeypatch):
        import ucode.cli as cli_mod

        states = {
            "https://first.com": {**MINIMAL_STATE, "workspace": "https://first.com"},
            "https://second.com": {**MINIMAL_STATE, "workspace": "https://second.com"},
        }
        configured_shared: list[tuple[str, str | None, tuple[str, ...] | None, bool]] = []

        def fake_configure_shared_state(
            workspace,
            profile=None,
            tools=None,
            force_login=False,
            use_pat=False,
            fable_enabled=None,
            databricks_ai_tools_enabled=None,
            clear_custom_oauth=False,
        ):
            configured_shared.append(
                (workspace, profile, tuple(tools) if tools is not None else None, force_login)
            )
            return states[workspace]

        saved: list[str] = []
        configured_tools: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(cli_mod, "configure_shared_state", fake_configure_shared_state)
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(state["workspace"]))
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda state, tool: True)
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["codex"])
        monkeypatch.setattr(cli_mod, "_maybe_select_provider_service", lambda tool, state: state)
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda state, tools: (
                configured_tools.append((state["workspace"], tools))
                or {**state, "available_tools": tools}
            ),
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        assert (
            cli_mod.configure_workspace_command(
                workspaces=[("https://first.com", None), ("https://second.com", None)]
            )
            == 0
        )
        assert configured_shared == [
            ("https://first.com", None, None, True),
            ("https://second.com", None, None, True),
        ]
        assert saved == ["https://first.com"]
        assert configured_tools == [("https://first.com", ["codex"])]


class TestParseProfilesOption:
    @staticmethod
    def _patch_profiles(monkeypatch, entries):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "list_profile_entries", lambda: entries)
        return cli_mod

    def test_resolves_profiles_to_workspace_entries(self, monkeypatch):
        cli_mod = self._patch_profiles(
            monkeypatch,
            [
                {"name": "DEFAULT", "host": "https://first.databricks.com/", "auth_type": "pat"},
                {
                    "name": "second",
                    "host": "https://second.databricks.com",
                    "auth_type": "databricks-cli",
                },
            ],
        )
        assert cli_mod._parse_profiles_option("DEFAULT, second") == [
            ("https://first.databricks.com", "DEFAULT"),
            ("https://second.databricks.com", "second"),
        ]

    def test_unknown_profile_raises_with_available_names(self, monkeypatch):
        cli_mod = self._patch_profiles(
            monkeypatch,
            [{"name": "DEFAULT", "host": "https://first.databricks.com", "auth_type": "pat"}],
        )
        with pytest.raises(RuntimeError, match=r"'missing' was not found.*DEFAULT"):
            cli_mod._parse_profiles_option("missing")

    def test_profile_without_host_raises(self, monkeypatch):
        cli_mod = self._patch_profiles(monkeypatch, [{"name": "DEFAULT", "auth_type": "pat"}])
        with pytest.raises(RuntimeError, match="no host configured"):
            cli_mod._parse_profiles_option("DEFAULT")

    def test_empty_value_raises(self, monkeypatch):
        cli_mod = self._patch_profiles(monkeypatch, [])
        with pytest.raises(RuntimeError, match="No profiles provided"):
            cli_mod._parse_profiles_option(" , ")


class TestConfigureProfilesFlag:
    PROFILE_ENTRIES = [
        {"name": "DEFAULT", "host": "https://first.databricks.com", "auth_type": "pat"}
    ]

    def test_profiles_flag_resolves_workspaces(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--profiles", "DEFAULT"])
        assert result.exit_code == 0, result.output
        # Auth behaves like --workspaces: no skip flags are forwarded, so the
        # default forced OAuth login applies.
        mock_cfg.assert_called_once_with(
            workspaces=[("https://first.databricks.com", "DEFAULT")],
            prompt_optional_updates=True,
        )

    def test_profiles_flag_with_agents(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app, ["configure", "--agents", "claude,codex", "--profiles", "DEFAULT"]
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.databricks.com", "DEFAULT")],
            prompt_optional_updates=True,
        )

    def test_profiles_flag_with_agent(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agent", "claude", "--profiles", "DEFAULT"])
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            "claude",
            workspaces=[("https://first.databricks.com", "DEFAULT")],
        )

    def test_use_pat_and_skip_validate_are_forwarded(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.list_profile_entries", return_value=self.PROFILE_ENTRIES),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--agents",
                    "claude,codex",
                    "--profiles",
                    "DEFAULT",
                    "--use-pat",
                    "--skip-validate",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_cfg.assert_called_once_with(
            selected_tools=["claude", "codex"],
            workspaces=[("https://first.databricks.com", "DEFAULT")],
            prompt_optional_updates=True,
            use_pat=True,
            skip_validate=True,
        )

    def test_use_pat_requires_profiles(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                ["configure", "--workspaces", "https://first.databricks.com", "--use-pat"],
            )
        assert result.exit_code == 1
        assert "--use-pat requires --profiles" in _strip_ansi(result.output)
        mock_cfg.assert_not_called()

    def test_skip_unavailable_requires_agents(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--skip-unavailable"])
        assert result.exit_code == 1
        assert "--skip-unavailable requires --agents" in _strip_ansi(result.output)
        mock_cfg.assert_not_called()

    def test_skip_unavailable_forwarded_with_agents(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--workspaces",
                    "https://example.azuredatabricks.net",
                    "--agents",
                    "claude,codex,pi",
                    "--skip-unavailable",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_cfg.call_args.kwargs["skip_unavailable"] is True
        assert mock_cfg.call_args.kwargs["selected_tools"] == ["claude", "codex", "pi"]

    def test_skip_unavailable_absent_by_default(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(app, ["configure", "--agents", "claude,codex"])
        assert result.exit_code == 0, result.output
        assert "skip_unavailable" not in mock_cfg.call_args.kwargs

    def test_profiles_and_workspaces_are_mutually_exclusive(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.configure_workspace_command") as mock_cfg,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--profiles",
                    "DEFAULT",
                    "--workspaces",
                    "https://first.databricks.com",
                ],
            )
        assert result.exit_code == 1
        assert "not both" in _strip_ansi(result.output)
        mock_cfg.assert_not_called()


class TestConfigureSharedStateUsePat:
    """--use-pat reads the profile's PAT from ~/.databrickscfg, exports it as
    DATABRICKS_BEARER, persists the mode, and never opens a browser."""

    WS = "https://example.databricks.com"

    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # configure_shared_state writes DATABRICKS_BEARER directly; restore it
        # since monkeypatch can't track writes made by code under test.
        import os as os_mod

        original = os_mod.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os_mod.environ.pop("DATABRICKS_BEARER", None)
        else:
            os_mod.environ["DATABRICKS_BEARER"] = original

    @staticmethod
    def _stub_deps(monkeypatch, *, pat_token, existing_state=None):
        import ucode.cli as cli_mod

        logins: list[tuple] = []
        ensures: list[tuple] = []
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "load_state", lambda: dict(existing_state or {}))
        monkeypatch.setattr(cli_mod, "save_state", lambda s: saved.append(dict(s)))
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: logins.append((w, p)))
        monkeypatch.setattr(
            cli_mod, "ensure_databricks_auth", lambda w, p=None: ensures.append((w, p))
        )
        monkeypatch.setattr(cli_mod, "resolve_pat_token", lambda p: pat_token)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(
            cli_mod, "probe_unity_gateway_capabilities", lambda w, t: MODEL_SERVICE_PROBE
        )
        monkeypatch.setattr(cli_mod, "discover_model_services", lambda w, t: ({}, [], [], [], None))
        monkeypatch.setattr(cli_mod, "discover_claude_models", lambda w, t: ({}, None))
        monkeypatch.setattr(cli_mod, "discover_gemini_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "discover_codex_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})
        return cli_mod, logins, ensures, saved

    def test_use_pat_exports_bearer_and_skips_login(self, monkeypatch):
        import os as os_mod

        cli_mod, logins, ensures, saved = self._stub_deps(monkeypatch, pat_token="dapi-pat")

        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", force_login=True, use_pat=True
        )

        assert logins == []
        assert ensures == [(self.WS, "DEFAULT")]
        assert os_mod.environ["DATABRICKS_BEARER"] == "dapi-pat"
        assert state["use_pat"] is True
        assert saved and saved[-1]["use_pat"] is True

    def test_happy_path_prints_success_without_model_service_detail(self, monkeypatch, capsys):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")

        cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        output = _strip_ansi(capsys.readouterr().out)
        assert "Unity AI Gateway connected" in output
        assert "Model service:" not in output

    @pytest.mark.parametrize(
        ("responses", "expected_model_service"),
        [
            (
                [({}, None)],
                "reachable, no accessible model services returned; check USE CATALOG on system, "
                "and USE SCHEMA and EXECUTE on system.ai",
            ),
            (
                [({"next_page_token": "more"}, None)] * db_mod._MODEL_SERVICE_PROBE_MAX_PAGES,
                "reachable",
            ),
        ],
        ids=[
            "model-service-empty",
            "model-service-inconclusive",
        ],
    )
    def test_prints_warning_when_model_service_not_detected(
        self,
        monkeypatch,
        capsys,
        responses,
        expected_model_service,
    ):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        response_iter = iter(responses)
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: next(response_iter))
        monkeypatch.setattr(
            cli_mod, "probe_unity_gateway_capabilities", db_mod.probe_unity_gateway_capabilities
        )

        cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        output = " ".join(_strip_ansi(capsys.readouterr().out).split())
        assert f"Model service: {expected_model_service}" in output
        assert "Unity AI Gateway connected" not in output
        assert "(Legacy) endpoints:" not in output
        assert "V2" not in output
        assert "V3" not in output

    @pytest.mark.parametrize(
        ("responses", "error_match"),
        [
            (
                [(None, "HTTP 404 Not Found: model service unavailable")],
                "not enabled",
            ),
            (
                [(None, "HTTP 403 Forbidden: Missing Unity Catalog grants")],
                "model service access could not be verified",
            ),
            ([(None, "HTTP 401 Unauthorized")], "rejected the access token"),
        ],
        ids=[
            "model-service-unavailable",
            "model-service-forbidden",
            "invalid-token",
        ],
    )
    def test_local_gateway_probe_failures_do_not_print_success(
        self, monkeypatch, capsys, responses, error_match
    ):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        response_iter = iter(responses)
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: next(response_iter))
        monkeypatch.setattr(
            cli_mod, "probe_unity_gateway_capabilities", db_mod.probe_unity_gateway_capabilities
        )

        with pytest.raises(RuntimeError, match=error_match) as excinfo:
            cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        output = _strip_ansi(capsys.readouterr().out)
        assert "Unity AI Gateway connected" not in output
        message = str(excinfo.value)
        assert "v2" not in message.lower()
        assert "v3" not in message.lower()

    def test_use_pat_without_pat_profile_raises(self, monkeypatch):
        cli_mod, logins, _, _ = self._stub_deps(monkeypatch, pat_token=None)

        with pytest.raises(RuntimeError, match="no personal access token"):
            cli_mod.configure_shared_state(
                self.WS, profile="oauth-profile", force_login=True, use_pat=True
            )
        assert logins == []

    def test_use_pat_without_profile_raises(self, monkeypatch):
        cli_mod, _, _, _ = self._stub_deps(monkeypatch, pat_token="dapi-pat")

        with pytest.raises(RuntimeError, match="requires a Databricks CLI profile"):
            cli_mod.configure_shared_state(self.WS, force_login=True, use_pat=True)

    def test_launch_inherits_persisted_use_pat(self, monkeypatch):
        # A launch re-run passes use_pat=None; the persisted mode for the same
        # workspace must apply so no OAuth login is forced.
        cli_mod, logins, ensures, _ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "use_pat": True},
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT", force_login=False)

        assert logins == []
        assert state["use_pat"] is True

    def test_reconfigure_without_flag_clears_use_pat(self, monkeypatch):
        cli_mod, logins, _, _ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "use_pat": True},
        )

        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", force_login=True, use_pat=False
        )

        assert logins == [(self.WS, "DEFAULT")]
        assert "use_pat" not in state

    def test_uc_models_used_without_legacy_fallback(self, monkeypatch):
        # When model-services returns models, they're used and the legacy
        # per-family discovery is never consulted.
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: (
                {"opus": "system.ai.claude-opus-4-8"},
                ["system.ai.gpt-5"],
                [],
                [],
                None,
            ),
        )
        legacy_called: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "discover_claude_models",
            lambda w, t: legacy_called.append("claude") or ({}, None),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"] == {"opus": "system.ai.claude-opus-4-8"}
        assert state["codex_models"] == ["system.ai.gpt-5"]
        assert legacy_called == []
        assert "uc_enabled" not in state

    def test_codex_only_configure_persists_discovered_oss_models(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: (
                {},
                ["system.ai.gpt-5-6-sol"],
                [],
                ["system.ai.glm-5-2"],
                None,
            ),
        )

        state = cli_mod.configure_shared_state(
            self.WS,
            profile="DEFAULT",
            tools=["codex"],
        )

        assert state["codex_models"] == ["system.ai.gpt-5-6-sol"]
        assert state["oss_models"] == ["system.ai.glm-5-2"]

    def _stub_with_fable(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: (
                {"fable": "system.ai.claude-fable-5", "opus": "system.ai.claude-opus-4-8"},
                [],
                [],
                [],
                None,
            ),
        )
        return cli_mod

    def test_fable_stripped_from_discovery_when_not_enabled(self, monkeypatch):
        # Discovery buckets fable, but without --enable-fable it's dropped from
        # the persisted bundle so it never reaches any agent's config.
        cli_mod = self._stub_with_fable(monkeypatch)

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"] == {"opus": "system.ai.claude-opus-4-8"}
        assert "fable_enabled" not in state

    def test_fable_retained_and_persisted_when_enabled(self, monkeypatch):
        cli_mod = self._stub_with_fable(monkeypatch)

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT", fable_enabled=True)

        assert state["claude_models"]["fable"] == "system.ai.claude-fable-5"
        assert state["fable_enabled"] is True

    def test_launch_inherits_persisted_fable_opt_in(self, monkeypatch):
        # A launch re-run passes fable_enabled=None; the persisted opt-in for the
        # same workspace applies, so fable stays in the discovered bundle.
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "fable_enabled": True},
        )
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: ({"fable": "system.ai.claude-fable-5"}, [], [], [], None),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"]["fable"] == "system.ai.claude-fable-5"
        assert state["fable_enabled"] is True

    def test_reconfigure_with_disable_fable_clears_opt_in(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={"workspace": self.WS, "profile": "DEFAULT", "fable_enabled": True},
        )
        monkeypatch.setattr(
            cli_mod,
            "discover_model_services",
            lambda w, t: ({"fable": "system.ai.claude-fable-5"}, [], [], [], None),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT", fable_enabled=False)

        assert "fable_enabled" not in state
        assert "fable" not in state["claude_models"]

    def test_ai_tools_disable_persists(self, monkeypatch):
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", databricks_ai_tools_enabled=False
        )
        assert state["databricks_ai_tools_enabled"] is False

    def test_ai_tools_enable_persists_explicit_true(self, monkeypatch):
        # We ask explicitly, so store the on choice too (not just absent-default).
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", databricks_ai_tools_enabled=True
        )
        assert state["databricks_ai_tools_enabled"] is True

    def test_ai_tools_disable_inherited_same_workspace(self, monkeypatch):
        # No flag on a re-configure of the same workspace keeps the prior opt-out.
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={
                "workspace": self.WS,
                "profile": "DEFAULT",
                "databricks_ai_tools_enabled": False,
            },
        )
        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")
        assert state["databricks_ai_tools_enabled"] is False

    def test_ai_tools_disable_does_not_leak_across_workspaces(self, monkeypatch):
        # A different workspace's opt-out must NOT carry into this one; no flag
        # here resolves to the default (install=True), matching use_pat/fable scoping.
        cli_mod, *_ = self._stub_deps(
            monkeypatch,
            pat_token="dapi-pat",
            existing_state={
                "workspace": "https://other.databricks.com",
                "profile": "DEFAULT",
                "databricks_ai_tools_enabled": False,
            },
        )
        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")
        assert state["databricks_ai_tools_enabled"] is True

    def test_falls_back_to_legacy_when_uc_empty(self, monkeypatch):
        # No UC model-services: each family falls back to the legacy listing.
        cli_mod, *_ = self._stub_deps(monkeypatch, pat_token="dapi-pat")
        monkeypatch.setattr(
            cli_mod, "discover_model_services", lambda w, t: ({}, [], [], [], "no model services")
        )
        monkeypatch.setattr(
            cli_mod,
            "discover_claude_models",
            lambda w, t: (
                {"opus": "databricks-claude-opus-4-8", "sonnet": "databricks-claude-sonnet-4-6"},
                None,
            ),
        )

        state = cli_mod.configure_shared_state(self.WS, profile="DEFAULT")

        assert state["claude_models"] == {
            "opus": "databricks-claude-opus-4-8",
            "sonnet": "databricks-claude-sonnet-4-6",
        }


class TestConfigureSkipValidate:
    def test_skip_validate_skips_agent_validation(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "workspace": "https://first.com"}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "save_state", lambda s: None)
        monkeypatch.setattr(cli_mod, "check_gateway_endpoint", lambda s, t: True)
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda *a, **k: True)
        monkeypatch.setattr(
            cli_mod,
            "configure_selected_tools",
            lambda s, tools: {**s, "available_tools": tools},
        )
        validated: list = []
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda s: validated.append(s))

        result = cli_mod.configure_workspace_command(
            selected_tools=["codex"],
            workspaces=[("https://first.com", None)],
            skip_validate=True,
        )

        assert result == 0
        assert validated == []

    def test_skip_validate_skips_single_tool_validation(self, monkeypatch):
        import ucode.cli as cli_mod

        state = {**MINIMAL_STATE, "workspace": "https://first.com"}
        monkeypatch.setattr(cli_mod, "configure_shared_state", lambda *a, **k: state)
        monkeypatch.setattr(cli_mod, "configure_single_tool", lambda t, s: s)
        installed: list = []
        monkeypatch.setattr(
            cli_mod,
            "install_databricks_ai_tools_for_agents",
            lambda tools, s: installed.append(tools),
        )
        validated: list = []
        monkeypatch.setattr(cli_mod, "validate_tool", lambda t: validated.append(t) or (True, ""))

        result = cli_mod.configure_workspace_command(
            "claude",
            workspaces=[("https://first.com", None)],
            skip_validate=True,
        )

        assert result == 0
        assert validated == []
        # `ucode configure` (single-agent) still installs AI Tools — it's the
        # configure path, unlike launch which auto-configures without installing.
        assert installed == [["claude"]]


class TestConfigureSharedStateMcpCleanup:
    """A workspace switch should scrub the previous workspace's MCP entries from
    installed client configs. Switching to the same workspace must not."""

    @staticmethod
    def _stub_external_deps(monkeypatch):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", lambda w, p=None: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(
            cli_mod, "probe_unity_gateway_capabilities", lambda w, t: MODEL_SERVICE_PROBE
        )
        monkeypatch.setattr(cli_mod, "discover_model_services", lambda w, t: ({}, [], [], [], None))
        monkeypatch.setattr(cli_mod, "discover_claude_models", lambda w, t: ({}, None))
        monkeypatch.setattr(cli_mod, "discover_gemini_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "discover_codex_models", lambda w, t: ([], None))
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})

    def test_purges_residue_when_workspace_changes(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod, "load_state", lambda: {"workspace": "https://old.databricks.com"}
        )
        purge_calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(
            cli_mod,
            "purge_cross_workspace_mcp_residue",
            lambda state, workspace: purge_calls.append((state, workspace)),
        )

        cli_mod.configure_shared_state("https://new.databricks.com")

        assert len(purge_calls) == 1
        _, called_workspace = purge_calls[0]
        assert called_workspace == "https://new.databricks.com"

    def test_skips_purge_when_workspace_unchanged(self, monkeypatch):
        import ucode.cli as cli_mod

        self._stub_external_deps(monkeypatch)
        monkeypatch.setattr(
            cli_mod, "load_state", lambda: {"workspace": "https://same.databricks.com"}
        )
        purge_calls: list = []
        monkeypatch.setattr(
            cli_mod,
            "purge_cross_workspace_mcp_residue",
            lambda state, workspace: purge_calls.append((state, workspace)),
        )

        cli_mod.configure_shared_state("https://same.databricks.com")

        assert purge_calls == []


class TestConfigureSharedStateSkipDiscovery:
    """With skip_model_discovery (provider mode), the heavy family discovery is
    skipped; only a single web-search model is fetched, and existing model lists
    are preserved rather than clobbered."""

    @staticmethod
    def _stub(monkeypatch):
        import ucode.cli as cli_mod

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", lambda w, p=None: None)
        monkeypatch.setattr(cli_mod, "run_databricks_login", lambda w, p: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: None)
        monkeypatch.setattr(cli_mod, "get_databricks_token", lambda w, p: "token")
        monkeypatch.setattr(
            cli_mod, "probe_unity_gateway_capabilities", lambda w, t: MODEL_SERVICE_PROBE
        )
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {})
        monkeypatch.setattr(cli_mod, "save_state", lambda s: None)

    def test_skips_family_discovery_and_fetches_web_search_model(self, monkeypatch):
        import ucode.cli as cli_mod

        ws = "https://prov.databricks.com"
        self._stub(monkeypatch)
        # Pretend a prior Databricks configure left models behind.
        monkeypatch.setattr(
            cli_mod,
            "load_state",
            lambda: {"workspace": ws, "claude_models": {"opus": "databricks-claude-opus-4-8"}},
        )

        def _boom(*a, **k):
            raise AssertionError("discover_model_services must not run in provider mode")

        monkeypatch.setattr(cli_mod, "discover_model_services", _boom)
        codex_calls: list = []
        monkeypatch.setattr(
            cli_mod,
            "discover_codex_models",
            lambda w, t: codex_calls.append((w, t)) or (["databricks-gpt-5"], None),
        )

        state = cli_mod.configure_shared_state(ws, tools=["claude"], skip_model_discovery=True)

        assert codex_calls == [(ws, "token")]
        assert state["web_search_model"] == "databricks-gpt-5"
        # Existing model list preserved, not overwritten to {}.
        assert state["claude_models"] == {"opus": "databricks-claude-opus-4-8"}


class TestConfigureSharedStateSkipPreflight:
    """With skip_preflight (--skip-preflight), a prior configure is trusted:
    no auth login, token fetch, gateway probe, or model discovery runs — but the
    profile and base URLs are still resolved and state is persisted."""

    WS = "https://cfg.databricks.com"

    @staticmethod
    def _stub(monkeypatch):
        import ucode.cli as cli_mod

        def _boom(name):
            def _f(*a, **k):
                raise AssertionError(f"{name} must not run under skip_preflight")

            return _f

        monkeypatch.setattr(cli_mod, "normalize_workspace_url", lambda w: w)
        # Any network round-trip is a hard failure in this mode.
        monkeypatch.setattr(cli_mod, "ensure_databricks_auth", _boom("ensure_databricks_auth"))
        monkeypatch.setattr(cli_mod, "run_databricks_login", _boom("run_databricks_login"))
        monkeypatch.setattr(cli_mod, "ensure_pat_bearer", _boom("ensure_pat_bearer"))
        monkeypatch.setattr(cli_mod, "get_databricks_token", _boom("get_databricks_token"))
        monkeypatch.setattr(
            cli_mod,
            "probe_unity_gateway_capabilities",
            _boom("probe_unity_gateway_capabilities"),
        )
        monkeypatch.setattr(cli_mod, "discover_model_services", _boom("discover_model_services"))
        monkeypatch.setattr(cli_mod, "discover_codex_models", _boom("discover_codex_models"))
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda w: "resolved")
        monkeypatch.setattr(cli_mod, "build_shared_base_urls", lambda w: {"codex": "u/codex"})
        saved: list[dict] = []
        monkeypatch.setattr(cli_mod, "save_state", lambda s: saved.append(dict(s)))
        return cli_mod, saved

    def test_skips_auth_gateway_and_discovery_but_persists(self, monkeypatch):
        cli_mod, saved = self._stub(monkeypatch)
        monkeypatch.setattr(
            cli_mod,
            "load_state",
            lambda: {"workspace": self.WS, "codex_models": ["databricks-gpt-5"]},
        )

        state = cli_mod.configure_shared_state(
            self.WS, profile="DEFAULT", tools=["codex"], skip_preflight=True
        )

        # base_urls rebuilt and state saved, but the prior model list is left intact.
        assert state["base_urls"] == {"codex": "u/codex"}
        assert state["codex_models"] == ["databricks-gpt-5"]
        assert saved and saved[-1]["base_urls"] == {"codex": "u/codex"}

    def test_resolves_profile_locally_when_missing(self, monkeypatch):
        cli_mod, _ = self._stub(monkeypatch)
        monkeypatch.setattr(cli_mod, "load_state", lambda: {"workspace": self.WS})

        state = cli_mod.configure_shared_state(self.WS, profile=None, skip_preflight=True)

        # find_profile_name_for_host is a local ~/.databrickscfg lookup (no network).
        assert state["profile"] == "resolved"


class TestSkipPreflightFlag:
    """`--skip-preflight` on a launch command threads through _launch_tool to
    configure_shared_state as skip_preflight."""

    LAUNCH_TOOLS = ["codex", "claude", "gemini", "opencode", "copilot", "pi"]

    @staticmethod
    def _patches(cfg):
        return [
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli._auto_configure_tool"),
            patch("ucode.cli.load_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.ensure_provider_state", return_value=MINIMAL_STATE),
            patch("ucode.cli.configure_shared_state", cfg),
            patch("ucode.cli.codex_agent.has_ucode_config", return_value=False),
            patch(
                "ucode.cli.resolve_launch_model",
                return_value=(MINIMAL_STATE, "databricks-claude-sonnet-4"),
            ),
            patch("ucode.cli.configure_tool", return_value=MINIMAL_STATE),
            patch("ucode.cli._fetch_managed_config", return_value=(None, False)),
            patch("ucode.cli.launch_agent"),
        ]

    @pytest.mark.parametrize("tool", LAUNCH_TOOLS)
    def test_flag_sets_skip_preflight_true(self, tool):
        cfg = MagicMock(return_value=MINIMAL_STATE)
        with contextlib.ExitStack() as stack:
            for p in self._patches(cfg):
                stack.enter_context(p)
            result = runner.invoke(app, [tool, "--skip-preflight"])
        assert result.exit_code == 0, result.output
        assert cfg.call_args.kwargs["skip_preflight"] is True

    @pytest.mark.parametrize("tool", ["codex", "gemini"])
    def test_absent_flag_defaults_false(self, tool):
        cfg = MagicMock(return_value=MINIMAL_STATE)
        with contextlib.ExitStack() as stack:
            for p in self._patches(cfg):
                stack.enter_context(p)
            result = runner.invoke(app, [tool])
        assert result.exit_code == 0, result.output
        assert cfg.call_args.kwargs["skip_preflight"] is False


class TestRejectDisabledAgent:
    """`enabled_agents` is an allowlist: an agent the admin didn't enable would launch unmanaged."""

    @staticmethod
    def _reject(managed, tool):
        import ucode.cli as cli_mod

        cli_mod._reject_disabled_agent(managed, tool)

    def test_raises_naming_the_enabled_agents(self):
        managed = {"enabled_agents": {"claude": {}, "opencode": {}}}
        with pytest.raises(RuntimeError, match="doesn't enable Gemini CLI") as exc:
            self._reject(managed, "gemini")
        assert "Claude Code, OpenCode" in str(exc.value)

    def test_allows_an_enabled_agent(self):
        self._reject({"enabled_agents": {"claude": {}}}, "claude")

    @pytest.mark.parametrize("managed", [None, {}, {"budget_policy": {}}])
    def test_a_config_naming_no_agents_blocks_nothing(self, managed):
        # No managed config, or one that only sets a budget policy, expresses no opinion on agents.
        self._reject(managed, "gemini")


class TestFetchManagedConfig:
    """The launch path's managed-config read, which gates both the allowlist and model discovery."""

    @staticmethod
    def _fetch(state):
        import ucode.cli as cli_mod

        return cli_mod._fetch_managed_config(state)

    def test_fetches_fresh_when_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config", lambda state: ({"enabled_agents": {}}, False)
        )
        assert self._fetch({"workspace": "https://w"}) == ({"enabled_agents": {}}, False)

    def test_feature_disabled_returns_none_and_the_flag(self, monkeypatch):
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: (None, True))
        assert self._fetch({"workspace": "https://w"}) == (None, True)


class TestManagedConfigDecidesDiscoveryFromFreshRead:
    def test_a_removed_model_list_no_longer_skips_discovery(self, monkeypatch):
        """The sweep decision must come from the fetched config, not the cached one.

        An admin who removes a previously published model list leaves a cache that still names
        models. Deciding from that cache would skip discovery for a config that no longer supplies
        models, so the launch would have neither.
        """
        stale_cache = {
            "enabled_agents": {"claude": {"model_config": {"models": {"default_opus_model": "m"}}}}
        }
        fresh = {"enabled_agents": {"claude": {"model_config": {}}}}
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: stale_cache)
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: (fresh, False))

        state = dict(MINIMAL_STATE)
        with (
            patch("ucode.cli.normalize_tool", return_value="claude"),
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state) as mock_shared,
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])

        assert result.exit_code == 0, result.output
        assert mock_shared.call_args.kwargs["skip_model_discovery"] is False


class TestBareUcode:
    """Bare `ucode` launches the managed default agent, or explains why it can't."""

    MANAGED = {"default_agent": "claude", "enabled_agents": {"claude": {}, "opencode": {}}}

    @staticmethod
    def _run(
        monkeypatch,
        *,
        managed,
        is_admin=False,
        args=None,
        cached=None,
        coding_agent_config_feature_disabled=False,
    ):
        launched: list[tuple] = []
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})

        if coding_agent_config_feature_disabled:
            monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: (None, True))
        else:
            monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: (managed, False))

        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: cached)
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda *a, **k: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda *a, **k: is_admin)
        monkeypatch.setattr(
            "ucode.cli._launch_tool",
            lambda tool, ctx, **kw: launched.append((tool, kw)),
        )
        result = runner.invoke(app, args or [])
        return result, launched

    def test_launches_the_managed_default_agent(self, monkeypatch):
        result, launched = self._run(monkeypatch, managed=self.MANAGED)
        assert result.exit_code == 0, result.output
        assert launched and launched[0][0] == "claude"
        assert "paved" not in result.output  # no policy set in this config
        assert "Claude Code" in result.output

    def test_falls_back_to_the_first_enabled_agent(self, monkeypatch):
        managed = {"enabled_agents": {"opencode": {}}}
        result, launched = self._run(monkeypatch, managed=managed)
        assert result.exit_code == 0, result.output
        assert launched[0][0] == "opencode"

    def test_launch_banner_is_abridged_not_the_full_box(self, monkeypatch):
        managed = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": {"default_model": "system.ai.opus"}}},
            "mcp_servers": [{"name": "system.ai.slack", "type": "mcp-service"}],
            "skills": {"names": ["main.default.my_skill"]},
        }
        result, _ = self._run(monkeypatch, managed=managed)
        assert result.exit_code == 0, result.output
        # One-line banner: the agent it launches, and the model.
        assert "launching Claude Code as the default agent" in result.output
        assert "system.ai.opus" in result.output
        # The full box's per-config enumeration is left to `ucode status`.
        assert "Enabled agents:" not in result.output
        assert "system.ai.slack" not in result.output
        assert "main.default.my_skill" not in result.output

    def test_launch_banner_omits_default_agent_when_a_tier_overrides(self, monkeypatch):
        # A budget tier can launch a different agent than the config's default; the banner must not
        # then call it "the default agent" (the tier note in _launch_tool explains the swap).
        managed = {"default_agent": "claude", "enabled_agents": {"claude": {}, "opencode": {}}}
        monkeypatch.setattr(
            "ucode.cli._fetch_budget_recommendation", lambda state, m: {"agent": "opencode"}
        )
        result, launched = self._run(monkeypatch, managed=managed)
        assert result.exit_code == 0, result.output
        assert launched[0][0] == "opencode"
        assert "launching OpenCode" in result.output
        assert "as the default agent" not in result.output

    def test_admin_without_a_config_is_pointed_at_setup(self, monkeypatch):
        result, launched = self._run(monkeypatch, managed=None, is_admin=True)
        assert result.exit_code == 0, result.output
        assert launched == []
        assert "ug setup" in result.output

    def test_non_admin_without_a_config_is_told_to_ask(self, monkeypatch):
        result, launched = self._run(monkeypatch, managed=None, is_admin=False)
        assert result.exit_code == 0, result.output
        assert launched == []
        assert "Ask a workspace admin" in result.output

    def test_feature_disabled_guides_without_managed_mention(self, monkeypatch):
        result, launched = self._run(
            monkeypatch, managed=None, is_admin=True, coding_agent_config_feature_disabled=True
        )
        assert result.exit_code == 0, result.output
        assert launched == []
        assert "ug configure" in result.output
        assert "managed" not in result.output.lower()

    def test_dry_run_uses_the_cache_and_does_not_fetch(self, monkeypatch):
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("--dry-run must not fetch"),
        )
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: self.MANAGED)
        launched: list[tuple] = []
        monkeypatch.setattr(
            "ucode.cli._launch_tool", lambda tool, ctx, **kw: launched.append((tool, kw))
        )
        result = runner.invoke(app, ["--dry-run"])
        assert result.exit_code == 0, result.output
        # The config bare `ucode` already read is handed down, so the launch path does not refetch.
        assert launched[0][1]["managed"] == self.MANAGED

    def test_dry_run_with_no_cached_config_does_not_crash(self, monkeypatch):
        # --dry-run doesn't fetch, so the feature-disabled flag is never assigned by the fetch path.
        # With no cached config it must still be well-defined (defaults False) rather than raising
        # UnboundLocalError when the guidance check reads it.
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("--dry-run must not fetch"),
        )
        monkeypatch.setattr("ucode.cli.load_managed_state", lambda ws: None)
        # The no-config guidance checks admin status; stub the token/admin calls so the test
        # doesn't shell out to the `databricks` binary (absent in CI).
        monkeypatch.setattr("ucode.cli.get_databricks_token", lambda *a, **k: "tok")
        monkeypatch.setattr("ucode.cli.is_workspace_admin", lambda *a, **k: False)
        monkeypatch.setattr(
            "ucode.cli._launch_tool",
            lambda *a, **k: pytest.fail("nothing to launch without a config"),
        )
        result = runner.invoke(app, ["--dry-run"])
        assert result.exit_code == 0, result.output

    def test_skip_preflight_still_resolves_an_agent_from_the_managed_config(self, monkeypatch):
        # --skip-preflight is now only about auth/gateway re-validation, decoupled from managed
        # config, so bare `ucode --skip-preflight` still fetches the config and picks its agent.
        monkeypatch.setattr("ucode.cli.install_databricks_cli", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.apply_pat_environment", lambda *a, **k: None)
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        managed = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": {"default_model": "m"}}},
        }
        monkeypatch.setattr("ucode.cli.refresh_managed_config", lambda state: (managed, False))
        monkeypatch.setattr("ucode.cli._fetch_budget_recommendation", lambda state, m: None)
        monkeypatch.setattr("ucode.cli._print_managed_summary", lambda *a, **k: None)
        seen: dict = {}
        monkeypatch.setattr(
            "ucode.cli._launch_tool",
            lambda tool, ctx, **kw: seen.update({"tool": tool, **kw}),
        )
        result = runner.invoke(app, ["--skip-preflight"])
        assert result.exit_code == 0, result.output
        assert seen["tool"] == "claude"
        assert seen["skip_preflight"] is True

    def test_subcommands_still_work(self, monkeypatch):
        # The callback runs for every invocation, so it must not intercept `ucode status`.
        monkeypatch.setattr(
            "ucode.cli.refresh_managed_config",
            lambda state: pytest.fail("the callback must not run for a subcommand"),
        )
        monkeypatch.setattr("ucode.cli.load_state", lambda: {"workspace": "https://w"})
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0, result.output


class TestBudgetRecommendationAtLaunch:
    """The budget read informs the launch; it never blocks it."""

    @staticmethod
    def _launch(monkeypatch, *, tool="claude", managed, recommendation=None, reason=None):
        state = dict(MINIMAL_STATE)
        calls: list[str] = []

        def fake_recommendation(workspace, token):
            calls.append(workspace)
            return recommendation, reason

        monkeypatch.setattr("ucode.cli.get_model_recommendation", fake_recommendation)
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.configure_tool", return_value=state) as cfg,
            patch("ucode.cli.get_databricks_token", return_value="tok"),
            patch("ucode.cli._fetch_managed_config", return_value=(managed, False)),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, [tool])
        return result, calls, cfg

    def test_not_checked_without_a_managed_config(self, monkeypatch):
        result, calls, _ = self._launch(monkeypatch, managed=None)
        assert result.exit_code == 0, result.output
        assert calls == []

    def test_the_recommended_agent_gets_the_recommended_model(self, monkeypatch):
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
            }
        }
        _result, _calls, cfg = self._launch(
            monkeypatch,
            managed=managed,
            recommendation={"agent": "claude", "model": "system.ai.claude-haiku-4-5"},
        )
        assert cfg.call_args.args[2] == "system.ai.claude-haiku-4-5"

    def test_passes_configured_claude_defaults_to_writer(self, monkeypatch):
        managed = {
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "models": {
                            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                        }
                    }
                }
            }
        }

        result, _calls, cfg = self._launch(monkeypatch, managed=managed)

        assert result.exit_code == 0, result.output
        assert cfg.call_args.kwargs["coding_agent_config_defaults"] == {
            "sonnet": "system.ai.claude-sonnet-4-6"
        }

    def test_another_agent_keeps_its_own_model_and_is_told_why(self, monkeypatch):
        # A tier's model belongs to the tier's agent; pinning it on claude would land a Kimi id in
        # ANTHROPIC_MODEL, which the Anthropic-dialect endpoint cannot serve.
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}},
                "opencode": {},
            }
        }
        result, _calls, cfg = self._launch(
            monkeypatch,
            managed=managed,
            recommendation={
                "agent": "opencode",
                "model": "system.ai.kimi-k2-7-code",
                "current_spend": 412.5,
                "effective_threshold": 500.0,
            },
        )
        assert result.exit_code == 0, result.output
        assert cfg.call_args.args[2] == "system.ai.claude-opus-4-8"
        assert "recommends OpenCode" in result.output

    def test_a_failed_read_does_not_block_the_launch(self, monkeypatch):
        result, _calls, _cfg = self._launch(
            monkeypatch,
            managed={"enabled_agents": {"claude": {}}},
            recommendation=None,
            reason="HTTP 500",
        )
        assert result.exit_code == 0, result.output
        assert "Could not check your budget" in result.output

    def test_a_token_failure_does_not_block_the_launch(self, monkeypatch):
        # Auth can lapse between the config refresh and the budget check.
        state = dict(MINIMAL_STATE)
        monkeypatch.setattr("ucode.cli.get_model_recommendation", lambda ws, tok: (None, None))
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.apply_pat_environment"),
            patch("ucode.cli.ensure_bootstrap_dependencies"),
            patch("ucode.cli.ensure_provider_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state),
            patch("ucode.cli.configure_tool", return_value=state),
            patch("ucode.cli.get_databricks_token", side_effect=RuntimeError("token expired")),
            patch(
                "ucode.cli._fetch_managed_config",
                return_value=({"enabled_agents": {"claude": {}}}, False),
            ),
            patch("ucode.cli.launch_agent"),
        ):
            result = runner.invoke(app, ["claude"])
        assert result.exit_code == 0, result.output
        assert "Could not check your budget" in result.output

    def test_shows_the_budget_bar(self, monkeypatch):
        result, _calls, _cfg = self._launch(
            monkeypatch,
            managed={"enabled_agents": {"claude": {}}},
            recommendation={
                "agent": "claude",
                "model": "m",
                "current_spend": 412.5,
                "effective_threshold": 500.0,
            },
        )
        assert result.exit_code == 0, result.output
        assert "83% used" in result.output
        assert "█" in result.output


class TestMcpProxyCmdForwardsUsePat:
    """`ucode mcp-proxy` forwards the PAT choice to `serve`, which owns the
    actual PAT resolution. Behavior of that resolution lives in test_mcp_proxy."""

    def _invoke(self, monkeypatch, *, flag, state):
        captured: dict = {}
        monkeypatch.setattr("ucode.cli.load_state", lambda: state)
        monkeypatch.setattr(
            "ucode.mcp_proxy.serve",
            lambda *a, **kw: captured.update(args=a, kwargs=kw),
        )
        args = ["mcp-proxy", "--url", "https://x/mcp", "--host", "https://x"]
        if flag:
            args.append("--use-pat")
        result = runner.invoke(app, args)
        return result, captured

    def test_flag_forwards_use_pat_true(self, monkeypatch):
        result, captured = self._invoke(
            monkeypatch, flag=True, state={"workspace": "https://x", "profile": "p"}
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["use_pat"] is True

    def test_saved_use_pat_state_forwards_true(self, monkeypatch):
        # A workspace configured with --use-pat persists use_pat=True; the proxy
        # honors it without the flag being repeated.
        result, captured = self._invoke(
            monkeypatch, flag=False, state={"workspace": "https://x", "use_pat": True}
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["use_pat"] is True

    def test_no_flag_and_no_state_forwards_false(self, monkeypatch):
        result, captured = self._invoke(monkeypatch, flag=False, state={"workspace": "https://x"})
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["use_pat"] is False
