"""Tests for the hidden custom-client OAuth feature."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import pytest
from databricks.sdk import oauth
from typer.testing import CliRunner

import ucode.cli as cli_mod
import ucode.databricks as db_mod
from ucode.cli import app
from ucode.custom_oauth import get_custom_client_token

WS = "https://example.databricks.com"
TEST_SCOPES = ("offline_access", "catalog.catalogs:read")
runner = CliRunner()


class TestCustomClientToken:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(oauth.TokenCache, "BASE_PATH", str(tmp_path / "oauth"))
        monkeypatch.setenv("DATABRICKS_BEARER", "unrelated-bearer")
        monkeypatch.setattr(db_mod, "run", Mock(side_effect=AssertionError("CLI not expected")))
        monkeypatch.setattr(
            db_mod, "find_profile_name_for_host", Mock(side_effect=AssertionError("No profile"))
        )
        self.endpoints = oauth.OidcEndpoints(
            authorization_endpoint=f"{WS}/oidc/v1/authorize",
            token_endpoint=f"{WS}/oidc/v1/token",
        )
        self.discovery = Mock(return_value=self.endpoints)
        monkeypatch.setattr(oauth, "get_workspace_endpoints", self.discovery)
        self.browser = Mock(return_value=self._credentials("browser-token", "browser-refresh"))
        monkeypatch.setattr(oauth.Consent, "launch_external_browser", self.browser)
        self.refresh = Mock(return_value=self._credentials("refreshed", "rotated-refresh").token())
        monkeypatch.setattr(oauth, "retrieve_token", self.refresh)

    def _credentials(self, access_token, refresh_token):
        return oauth.SessionCredentials(
            token=oauth.Token(
                access_token=access_token,
                token_type="Bearer",
                refresh_token=refresh_token,
                expiry=datetime.now(UTC) + timedelta(hours=1),
            ),
            token_endpoint=self.endpoints.token_endpoint,
            client_id="custom-client",
            redirect_url="http://localhost:8020",
        )

    def _cache(self, workspace=WS, client_id="custom-client"):
        return oauth.TokenCache(
            host=workspace,
            oidc_endpoints=self.endpoints,
            client_id=client_id,
            redirect_url="http://localhost:8020",
            scopes=list(TEST_SCOPES),
        )

    def test_browser_login_uses_custom_client_and_redirect(self, capsys):
        redirect_url = "http://localhost:41735/ai-devtools-workspace-oauth"
        token = get_custom_client_token(
            WS, client_id="custom-client", redirect_url=redirect_url, scopes=TEST_SCOPES
        )
        assert token == "browser-token"
        cached = self._cache().load().token()
        assert cached.refresh_token == "browser-refresh"
        output = capsys.readouterr()
        assert output.out == ""
        query = parse_qs(output.err.split("?", 1)[1].strip())
        assert query["client_id"] == ["custom-client"]
        assert query["redirect_uri"] == [redirect_url]
        assert query["scope"][0].split() == list(TEST_SCOPES)

    def test_reuses_cached_token_without_refresh_or_login(self):
        self._cache().save(self._credentials("cached", "refresh"))
        assert (
            get_custom_client_token(WS + "/", client_id="custom-client", scopes=TEST_SCOPES)
            == "cached"
        )
        self.browser.assert_not_called()
        self.refresh.assert_not_called()

    def test_expired_token_refreshes_with_custom_client_and_saves_rotation(self):
        cached = self._credentials("expired", "old-refresh")
        self._cache().save(cached)
        cache_path = Path(self._cache().filename)
        payload = json.loads(cache_path.read_text())
        payload["token"]["expiry"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        cache_path.write_text(json.dumps(payload))
        assert (
            get_custom_client_token(WS, client_id="custom-client", scopes=TEST_SCOPES)
            == "refreshed"
        )
        self.refresh.assert_called_once()
        self.browser.assert_not_called()
        assert self._cache().load().token().refresh_token == "rotated-refresh"

    def test_force_refresh_bypasses_fresh_access_token(self):
        self._cache().save(self._credentials("cached", "refresh"))
        assert (
            get_custom_client_token(
                WS, client_id="custom-client", scopes=TEST_SCOPES, force_refresh=True
            )
            == "refreshed"
        )
        self.refresh.assert_called_once()
        self.browser.assert_not_called()

    def test_refresh_failure_falls_back_to_browser(self, capsys):
        self._cache().save(self._credentials("cached", "revoked-refresh"))
        self.refresh.side_effect = ValueError("sensitive server response")
        assert (
            get_custom_client_token(
                WS, client_id="custom-client", scopes=TEST_SCOPES, force_refresh=True
            )
            == "browser-token"
        )
        self.browser.assert_called_once()
        output = capsys.readouterr()
        assert "Sign in again" in output.err
        assert "sensitive server response" not in output.err

    def test_invalid_redirect_fails_before_network(self):
        with pytest.raises(RuntimeError, match="--redirect-url must be"):
            get_custom_client_token(
                WS,
                client_id="custom-client",
                redirect_url="https://example.com/callback",
                scopes=TEST_SCOPES,
            )
        self.discovery.assert_not_called()

    def test_login_failure_is_actionable_and_does_not_expose_response(self):
        self.browser.side_effect = ValueError("sensitive server response")
        with pytest.raises(RuntimeError, match="registered redirect URL") as error:
            get_custom_client_token(WS, client_id="custom-client", scopes=TEST_SCOPES)
        assert "sensitive server response" not in str(error.value)

    @pytest.mark.parametrize("scopes", [["offline_access"], ["catalog.catalogs:read"]])
    def test_api_scopes_are_required(self, scopes):
        with pytest.raises(RuntimeError, match="OAuth scopes|API OAuth scope"):
            get_custom_client_token(WS, client_id="custom-client", scopes=scopes)
        self.discovery.assert_not_called()

    def test_missing_sdk_explains_how_to_install_custom_oauth(self, monkeypatch):
        monkeypatch.setattr("ucode.custom_oauth.oauth", None)
        with pytest.raises(RuntimeError, match=r"ucode\[custom-oauth\]"):
            get_custom_client_token(WS, client_id="custom-client", scopes=TEST_SCOPES)


class TestCustomClientCommand:
    @pytest.fixture(autouse=True)
    def _no_production_auth(self, monkeypatch):
        monkeypatch.setattr(
            "ucode.cli.get_databricks_token",
            Mock(side_effect=AssertionError("Custom auth must not use production auth")),
        )

    def test_custom_client_dispatches_with_explicit_scopes(self):
        with (
            patch(
                "ucode.cli.load_state",
                return_value={"workspace": "https://ws", "profile": "saved", "use_pat": True},
            ),
            patch("ucode.cli.ensure_pat_bearer") as pat,
            patch(
                "ucode.custom_oauth.get_custom_client_token", return_value="custom-token"
            ) as fetch,
        ):
            result = runner.invoke(
                app,
                [
                    "auth-token",
                    "--client-id",
                    "my-client",
                    "--redirect-url",
                    "http://localhost:41735/callback",
                    "--scopes",
                    "offline_access,catalog.catalogs:read",
                    "--force-refresh",
                ],
            )
        assert result.exit_code == 0, result.output
        assert result.stdout == "custom-token\n"
        pat.assert_not_called()
        fetch.assert_called_once_with(
            "https://ws",
            client_id="my-client",
            redirect_url="http://localhost:41735/callback",
            scopes=["offline_access", "catalog.catalogs:read"],
            force_refresh=True,
        )

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (["--client-id", "my-client", "--use-pat"], "cannot be combined"),
            (["--redirect-url", "http://localhost:8020"], "requires --client-id"),
            (["--client-id", "my-client"], "--scopes is required"),
        ],
    )
    def test_invalid_custom_client_options(self, options, message):
        with patch("ucode.custom_oauth.get_custom_client_token") as fetch:
            result = runner.invoke(app, ["auth-token", *options])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert message in result.stderr
        fetch.assert_not_called()


class TestConfigureCustomOAuth:
    def test_forwards_config_to_claude_configuration(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.install_tool_binary"),
            patch("ucode.cli.configure_workspace_command") as configure_workspace,
        ):
            result = runner.invoke(
                app,
                [
                    "configure",
                    "--agent",
                    "claude",
                    "--workspaces",
                    WS,
                    "--client-id",
                    "custom-client",
                    "--redirect-url",
                    "http://localhost:8020/callback",
                    "--scopes",
                    "offline_access,model-serving",
                ],
            )
        assert result.exit_code == 0, result.output
        configure_workspace.assert_called_once_with(
            "claude",
            workspaces=[(WS, None)],
            custom_oauth={
                "client_id": "custom-client",
                "redirect_url": "http://localhost:8020/callback",
                "scopes": ["offline_access", "model-serving"],
            },
        )

    def test_configure_without_custom_options_resets_custom_oauth(self):
        state = {"workspace": WS, "available_tools": ["claude"]}
        with (
            patch("ucode.cli._configure_shared_workspace_states", return_value=[state]) as shared,
            patch("ucode.cli.configure_single_tool", return_value=state),
            patch("ucode.cli.install_databricks_ai_tools_for_agents"),
        ):
            result = cli_mod.configure_workspace_command(
                "claude",
                workspaces=[(WS, None)],
                skip_validate=True,
            )

        assert result == 0
        assert shared.call_args.kwargs["clear_custom_oauth"] is True

    def test_shared_state_persists_custom_oauth(self, monkeypatch):
        custom_oauth = {
            "client_id": "custom-client",
            "redirect_url": "http://localhost:8020/callback",
            "scopes": ["offline_access", "model-serving"],
        }
        saved = []
        monkeypatch.setattr(cli_mod, "load_state", lambda: {"workspace": WS})
        monkeypatch.setattr(cli_mod, "save_state", lambda state: saved.append(dict(state)))
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda _workspace: None)

        state = cli_mod.configure_shared_state(
            WS,
            tools=["claude"],
            skip_preflight=True,
            custom_oauth=custom_oauth,
        )

        assert state["custom_oauth"] == custom_oauth
        assert saved[-1]["custom_oauth"] == custom_oauth

    def test_shared_state_clears_custom_oauth(self, monkeypatch):
        monkeypatch.setattr(
            cli_mod,
            "load_state",
            lambda: {"workspace": WS, "custom_oauth": {"client_id": "old"}},
        )
        monkeypatch.setattr(cli_mod, "save_state", lambda _state: None)
        monkeypatch.setattr(cli_mod, "find_profile_name_for_host", lambda _workspace: None)

        state = cli_mod.configure_shared_state(
            WS,
            tools=["claude"],
            skip_preflight=True,
            clear_custom_oauth=True,
        )

        assert "custom_oauth" not in state


class TestLaunchCustomOAuth:
    @pytest.mark.parametrize("command", ["auth-token", "configure", "claude", "codex"])
    def test_options_are_hidden(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--client-id" not in result.output
        assert "--redirect-url" not in result.output
        assert "--scopes" not in result.output

    @pytest.mark.parametrize("tool", ["claude", "codex"])
    def test_launch_forwards_custom_oauth(self, tool):
        with patch("ucode.cli._launch_tool") as launch:
            result = runner.invoke(
                app,
                [
                    tool,
                    "--workspace",
                    WS,
                    "--client-id",
                    "custom-client",
                    "--redirect-url",
                    "http://localhost:8020/callback",
                    "--scopes",
                    "offline_access,model-serving",
                ],
            )

        assert result.exit_code == 0, result.output
        assert launch.call_args.kwargs["custom_oauth"] == {
            "client_id": "custom-client",
            "redirect_url": "http://localhost:8020/callback",
            "scopes": ["offline_access", "model-serving"],
        }

    def test_auto_configure_receives_custom_oauth(self):
        custom_oauth = {
            "client_id": "custom-client",
            "redirect_url": "http://localhost:8020/callback",
            "scopes": ["offline_access", "model-serving"],
        }
        state = {"workspace": WS, "profile": None, "available_tools": ["codex"]}
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state) as configure_shared,
            patch("ucode.cli.configure_single_tool", return_value=state),
            patch("ucode.cli.validate_tool", return_value=(True, None)),
        ):
            cli_mod._auto_configure_tool("codex", custom_oauth=custom_oauth)

        configure_shared.assert_called_once_with(
            WS,
            profile=None,
            tools=["codex"],
            custom_oauth=custom_oauth,
        )

    def test_auto_configure_does_not_clear_custom_oauth(self):
        state = {"workspace": WS, "profile": None, "available_tools": ["claude"]}
        with (
            patch("ucode.cli.load_state", return_value=state),
            patch("ucode.cli.configure_shared_state", return_value=state) as configure_shared,
            patch("ucode.cli.configure_single_tool", return_value=state),
            patch("ucode.cli.validate_tool", return_value=(True, None)),
        ):
            cli_mod._auto_configure_tool("claude")

        configure_shared.assert_called_once_with(WS, profile=None, tools=["claude"])
