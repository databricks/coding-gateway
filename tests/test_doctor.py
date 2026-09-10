"""Tests for `ucode doctor` — check classification and the fix-apply flow."""

from __future__ import annotations

from unittest.mock import patch

import ucode.doctor as doctor_mod
from ucode.databricks import MIN_DATABRICKS_CLI_VERSION
from ucode.doctor import (
    Check,
    Suggestion,
    _check_agent_clis,
    _check_anthropic_env_collision,
    _check_databricks_auth,
    _check_databricks_cli,
    _check_npm,
    _check_tracing_mlflow,
    _check_uv,
    _check_workspace,
    doctor,
)


class TestUvCheck:
    def test_ok_when_present(self):
        with patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/uv"):
            check = _check_uv()
        assert check.status == "ok"
        assert check.suggestion is None

    def test_error_when_missing(self):
        with patch.object(doctor_mod.shutil, "which", return_value=None):
            check = _check_uv()
        assert check.status == "error"


class TestNpmCheck:
    def test_ok_when_present(self):
        with patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/npm"):
            assert _check_npm().status == "ok"

    def test_warn_when_missing(self):
        with patch.object(doctor_mod.shutil, "which", return_value=None):
            assert _check_npm().status == "warn"


class TestDatabricksCliCheck:
    def test_error_and_install_suggestion_when_missing(self):
        with patch.object(doctor_mod.shutil, "which", return_value=None):
            check = _check_databricks_cli()
        assert check.status == "error"
        assert check.suggestion is not None
        assert "Install" in check.suggestion.prompt

    def test_warn_and_upgrade_suggestion_when_below_floor(self):
        # A public-preview build below the aitools floor: recommend an upgrade,
        # not a hard failure.
        old = (MIN_DATABRICKS_CLI_VERSION[0], MIN_DATABRICKS_CLI_VERSION[1] - 1, 0)
        with (
            patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/databricks"),
            patch.object(doctor_mod, "databricks_cli_version", return_value=old),
        ):
            check = _check_databricks_cli()
        assert check.status == "warn"
        assert check.suggestion is not None
        assert "Upgrade" in check.suggestion.prompt

    def test_ok_when_at_or_above_floor(self):
        with (
            patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/databricks"),
            patch.object(
                doctor_mod, "databricks_cli_version", return_value=MIN_DATABRICKS_CLI_VERSION
            ),
        ):
            check = _check_databricks_cli()
        assert check.status == "ok"
        assert check.suggestion is None

    def test_warn_when_version_unreadable(self):
        with (
            patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/databricks"),
            patch.object(doctor_mod, "databricks_cli_version", return_value=None),
        ):
            check = _check_databricks_cli()
        assert check.status == "warn"
        assert check.suggestion is None


class TestWorkspaceCheck:
    def test_ok_when_configured(self):
        with patch.object(doctor_mod, "load_state", return_value={"workspace": "https://ws"}):
            check = _check_workspace()
        assert check.status == "ok"
        assert "https://ws" in check.detail

    def test_warn_when_unconfigured(self):
        with patch.object(doctor_mod, "load_state", return_value={}):
            assert _check_workspace().status == "warn"


class TestAgentCliChecks:
    def test_missing_binary_offers_install(self):
        state = {"available_tools": ["claude"]}
        with (
            patch.object(doctor_mod, "load_state", return_value=state),
            patch.object(doctor_mod, "tool_binary_installed", return_value=False),
        ):
            checks = _check_agent_clis()
        assert len(checks) == 1
        assert checks[0].status == "warn"
        assert checks[0].suggestion is not None

    def test_outdated_offers_update(self):
        state = {"available_tools": ["opencode"]}
        with (
            patch.object(doctor_mod, "load_state", return_value=state),
            patch.object(doctor_mod, "tool_binary_installed", return_value=True),
            patch.object(doctor_mod, "tool_update_available", return_value=("1.0.0", "1.2.0")),
        ):
            checks = _check_agent_clis()
        assert checks[0].status == "warn"
        assert "1.2.0" in checks[0].detail
        assert checks[0].suggestion is not None

    def test_up_to_date_is_ok(self):
        state = {"available_tools": ["claude"]}
        with (
            patch.object(doctor_mod, "load_state", return_value=state),
            patch.object(doctor_mod, "tool_binary_installed", return_value=True),
            patch.object(doctor_mod, "tool_version_error", return_value=None),
            patch.object(
                doctor_mod,
                "tool_update_available",
                side_effect=AssertionError("native updater must not use npm detection"),
            ),
        ):
            checks = _check_agent_clis()
        assert checks[0].status == "ok"
        assert "agent CLI" in checks[0].detail
        assert checks[0].suggestion is None

    def test_blocked_native_agent_offers_native_upgrade(self):
        state = {"available_tools": ["claude"]}
        with (
            patch.object(doctor_mod, "load_state", return_value=state),
            patch.object(doctor_mod, "tool_binary_installed", return_value=True),
            patch.object(doctor_mod, "tool_version_error", return_value="version too old"),
        ):
            checks = _check_agent_clis()
        assert checks[0].status == "warn"
        assert checks[0].detail == "version too old"
        assert checks[0].suggestion is not None
        assert checks[0].suggestion.prompt == "Upgrade Claude Code if available?"

    def test_unknown_tool_is_skipped(self):
        state = {"available_tools": ["not-a-real-tool"]}
        with patch.object(doctor_mod, "load_state", return_value=state):
            assert _check_agent_clis() == []


class TestDatabricksAuthCheck:
    def test_none_when_no_workspace(self):
        with patch.object(doctor_mod, "load_state", return_value={}):
            assert _check_databricks_auth() is None

    def test_ok_when_valid(self):
        with (
            patch.object(doctor_mod, "load_state", return_value={"workspace": "https://ws"}),
            patch.object(doctor_mod, "has_valid_databricks_auth", return_value=True),
        ):
            check = _check_databricks_auth()
        assert check.status == "ok"
        assert check.suggestion is None

    def test_warn_and_login_suggestion_when_invalid(self):
        with (
            patch.object(doctor_mod, "load_state", return_value={"workspace": "https://ws"}),
            patch.object(doctor_mod, "has_valid_databricks_auth", return_value=False),
        ):
            check = _check_databricks_auth()
        assert check.status == "warn"
        assert check.suggestion is not None
        assert "Log in" in check.suggestion.prompt

    def test_login_fix_reports_success(self):
        with (
            patch.object(doctor_mod, "load_state", return_value={"workspace": "https://ws"}),
            # invalid at first, then valid after login
            patch.object(doctor_mod, "has_valid_databricks_auth", side_effect=[False, True]),
            patch.object(doctor_mod, "run_databricks_login") as login,
        ):
            check = _check_databricks_auth()
            assert check.suggestion.apply() is True
        login.assert_called_once()

    def test_login_fix_reports_failure_when_login_raises(self):
        with (
            patch.object(doctor_mod, "load_state", return_value={"workspace": "https://ws"}),
            patch.object(doctor_mod, "has_valid_databricks_auth", return_value=False),
            patch.object(doctor_mod, "run_databricks_login", side_effect=RuntimeError("nope")),
        ):
            check = _check_databricks_auth()
            assert check.suggestion.apply() is False


class TestAnthropicEnvCollision:
    def test_none_when_unset(self, monkeypatch):
        for var in doctor_mod._CLAUDE_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert _check_anthropic_env_collision() is None

    def test_warns_when_set(self, monkeypatch):
        for var in doctor_mod._CLAUDE_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
        check = _check_anthropic_env_collision()
        assert check.status == "warn"
        assert "ANTHROPIC_AUTH_TOKEN" in check.detail
        # Advisory only — no auto-fix for a parent shell's env.
        assert check.suggestion is None

    def test_blank_value_is_ignored(self, monkeypatch):
        for var in doctor_mod._CLAUDE_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        assert _check_anthropic_env_collision() is None


class TestTracingMlflowCheck:
    def test_none_when_tracing_disabled(self):
        with patch.object(doctor_mod, "tracing_config", return_value=None):
            assert _check_tracing_mlflow() is None

    def test_ok_when_mlflow_present(self):
        with (
            patch.object(doctor_mod, "tracing_config", return_value={"enabled": True}),
            patch.object(doctor_mod, "tracing_mlflow_ok", return_value=True),
        ):
            check = _check_tracing_mlflow()
        assert check.status == "ok"
        assert check.suggestion is None

    def test_warn_and_install_suggestion_when_missing(self):
        with (
            patch.object(doctor_mod, "tracing_config", return_value={"enabled": True}),
            patch.object(doctor_mod, "tracing_mlflow_ok", return_value=False),
        ):
            check = _check_tracing_mlflow()
        assert check.status == "warn"
        assert check.suggestion is not None


class TestUcodeCheck:
    def test_upgrade_uses_renamed_repository(self):
        with (
            patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/uv"),
            patch.object(doctor_mod.subprocess, "run") as run,
        ):
            assert doctor_mod._upgrade_ucode()

        run.assert_called_once_with(
            [
                "uv",
                "tool",
                "install",
                "--reinstall",
                "git+https://github.com/databricks/unity-gateway",
            ],
            check=True,
        )


class TestDoctorFlow:
    def _only(self, checks: list[Check]):
        """Run doctor() with a fixed set of checks and a stubbed prompter."""
        return patch.object(doctor_mod, "_gather_checks", return_value=checks)

    def test_applies_fix_when_user_accepts(self):
        applied = []
        suggestion = Suggestion("Fix it?", lambda: applied.append(True) or True)
        check = Check("thing", "warn", "broken", suggestion)
        with (
            self._only([check]),
            patch.object(doctor_mod, "prompt_yes_no_default", return_value=True),
        ):
            rc = doctor()
        assert rc == 0
        assert applied == [True]

    def test_skips_fix_when_user_declines(self):
        applied = []
        suggestion = Suggestion("Fix it?", lambda: applied.append(True) or True)
        check = Check("thing", "warn", "broken", suggestion)
        with (
            self._only([check]),
            patch.object(doctor_mod, "prompt_yes_no_default", return_value=False),
        ):
            doctor()
        assert applied == []

    def test_reports_fix_failure_without_raising(self):
        suggestion = Suggestion("Fix it?", lambda: False)
        check = Check("thing", "error", "broken", suggestion)
        with (
            self._only([check]),
            patch.object(doctor_mod, "prompt_yes_no_default", return_value=True),
            patch.object(doctor_mod, "print_warning") as warn,
        ):
            rc = doctor()
        assert rc == 0
        warn.assert_called()

    def test_ok_check_is_never_prompted(self):
        check = Check("thing", "ok", "healthy", None)
        with (
            self._only([check]),
            patch.object(doctor_mod, "prompt_yes_no_default") as prompt,
        ):
            doctor()
        prompt.assert_not_called()
