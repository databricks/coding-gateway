"""Tests for agents/gemini.py."""

from __future__ import annotations

import json

import pytest

from ucode.agents import gemini

WS = "https://example.databricks.com"


@pytest.fixture(autouse=True)
def _redirect_gemini_home(tmp_path, monkeypatch):
    gemini_home = tmp_path / ".gemini-home"
    monkeypatch.setattr(gemini, "GEMINI_HOME_DIR", gemini_home)
    monkeypatch.setattr(gemini, "GEMINI_SETTINGS_PATH", gemini_home / ".gemini" / "settings.json")


class TestGeminiSpec:
    def test_binary(self):
        assert gemini.SPEC["binary"] == "gemini"

    def test_package(self):
        assert gemini.SPEC["package"] == "@google/gemini-cli"

    def test_display(self):
        assert gemini.SPEC["display"] == "Gemini CLI"

    def test_config_path_is_ucode_env_file(self):
        assert gemini.SPEC["config_path"].name == "ucode.env"


class TestRenderEnvOverlay:
    def test_sets_gemini_model(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GEMINI_MODEL"] == "gemini-2.0-flash"

    def test_sets_base_url(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GOOGLE_GEMINI_BASE_URL"] == f"{WS}/ai-gateway/gemini"

    def test_sets_api_key(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GEMINI_API_KEY"] == "tok123"

    def test_sets_oauth_token_for_mcp(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["OAUTH_TOKEN"] == "tok123"

    def test_sets_bearer_auth_mechanism(self):
        env = gemini.render_env_overlay(WS, "gemini-2.0-flash", "tok123")
        assert env["GEMINI_API_KEY_AUTH_MECHANISM"] == "bearer"

    def test_sets_user_agent_via_custom_headers(self, monkeypatch):
        monkeypatch.setattr(gemini, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.40.0")
        env = gemini.render_env_overlay(WS, "gemini-2", "tok")
        assert env["GEMINI_CLI_CUSTOM_HEADERS"] == "User-Agent:ucode/0.1.0 gemini/0.40.0"

    def test_provider_adds_routing_header_and_pins_target(self, monkeypatch):
        monkeypatch.setattr(gemini, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.40.0")
        env = gemini.render_env_overlay(
            WS, "gemini-3.5-flash", "tok", provider="cat.sch.gemini-enterprise"
        )
        assert env["GEMINI_MODEL"] == "gemini-3.5-flash"
        assert env["GEMINI_CLI_CUSTOM_HEADERS"] == (
            "User-Agent:ucode/0.1.0 gemini/0.40.0,"
            "Databricks-Model-Provider-Service:cat.sch.gemini-enterprise"
        )


class TestBuildRuntimeEnv:
    def test_merges_os_environment(self):
        env = gemini.build_runtime_env(WS, "gemini-2", "tok")
        assert "PATH" in env

    def test_overrides_gemini_vars(self):
        env = gemini.build_runtime_env(WS, "gemini-2.0-flash", "mytoken")
        assert env["GEMINI_MODEL"] == "gemini-2.0-flash"
        assert env["GEMINI_API_KEY"] == "mytoken"
        assert env["GEMINI_API_KEY_AUTH_MECHANISM"] == "bearer"

    def test_sets_base_url(self):
        env = gemini.build_runtime_env(WS, "gemini-2", "tok")
        assert env["GOOGLE_GEMINI_BASE_URL"] == f"{WS}/ai-gateway/gemini"

    def test_sets_oauth_token_for_mcp(self):
        env = gemini.build_runtime_env(WS, "gemini-2", "tok")
        assert env["OAUTH_TOKEN"] == "tok"

    def test_sets_private_gemini_home(self, tmp_path, monkeypatch):
        gemini_home = tmp_path / "private-gemini-home"
        monkeypatch.setattr(gemini, "GEMINI_HOME_DIR", gemini_home)
        monkeypatch.setattr(
            gemini, "GEMINI_SETTINGS_PATH", gemini_home / ".gemini" / "settings.json"
        )

        env = gemini.build_runtime_env(WS, "gemini-2", "tok")

        assert env["GEMINI_CLI_HOME"] == str(gemini_home)

    def test_writes_private_auth_settings(self):
        gemini.build_runtime_env(WS, "gemini-2", "tok")

        settings = json.loads(gemini.GEMINI_SETTINGS_PATH.read_text())
        assert settings == {"security": {"auth": {"selectedType": "gemini-api-key"}}}

    def test_preserves_existing_private_settings(self):
        gemini.GEMINI_SETTINGS_PATH.parent.mkdir(parents=True)
        gemini.GEMINI_SETTINGS_PATH.write_text(
            json.dumps({"theme": "dark", "security": {"auth": {"selectedType": "oauth"}}})
        )

        gemini.build_runtime_env(WS, "gemini-2", "tok")

        settings = json.loads(gemini.GEMINI_SETTINGS_PATH.read_text())
        assert settings["theme"] == "dark"
        assert settings["security"]["auth"]["selectedType"] == "gemini-api-key"


class TestGeminiDefaultModel:
    def test_returns_first_model(self):
        state = {"gemini_models": ["gemini-2", "gemini-1"]}
        assert gemini.default_model(state) == "gemini-2"

    def test_returns_none_when_empty_list(self):
        assert gemini.default_model({"gemini_models": []}) is None

    def test_returns_none_when_missing(self):
        assert gemini.default_model({}) is None

    def test_gemini_default_model_wins_over_allowlist(self):
        state = {
            "gemini_default_model": "admin-chosen-default",
            "gemini_models": ["gemini-2"],
        }
        assert gemini.default_model(state) == "admin-chosen-default"


class TestGeminiVersionGating:
    def test_too_new_version_flags_045(self, monkeypatch):
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.45.0-nightly.20260602")
        assert gemini.too_new_version() == "0.45.0-nightly.20260602"

    def test_too_new_version_allows_044(self, monkeypatch):
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.44.1")
        assert gemini.too_new_version() is None

    def test_too_new_version_none_when_unknown(self, monkeypatch):
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "unknown")
        assert gemini.too_new_version() is None

    def test_too_new_downgrade_returns_target(self, monkeypatch):
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.45.0-nightly.20260602")
        monkeypatch.setattr(gemini, "latest_version_below", lambda pkg, ceiling: "0.44.1")
        assert gemini.too_new_downgrade() == ("0.45.0-nightly.20260602", "0.44.1")

    def test_too_new_downgrade_none_when_safe(self, monkeypatch):
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.44.1")
        monkeypatch.setattr(gemini, "latest_version_below", lambda pkg, ceiling: "0.44.1")
        assert gemini.too_new_downgrade() is None

    def test_too_new_downgrade_none_when_no_target(self, monkeypatch):
        monkeypatch.setattr(gemini, "agent_version", lambda binary: "0.45.0")
        monkeypatch.setattr(gemini, "latest_version_below", lambda pkg, ceiling: None)
        assert gemini.too_new_downgrade() is None


class TestGeminiValidateCmd:
    def test_starts_with_binary(self):
        cmd = gemini.validate_cmd("gemini")
        assert cmd[0] == "gemini"

    def test_has_p_flag(self):
        cmd = gemini.validate_cmd("gemini")
        assert "-p" in cmd

    def test_has_prompt_text(self):
        cmd = gemini.validate_cmd("gemini")
        assert len(cmd) > 2


class TestGeminiManagedKeys:
    def test_managed_keys_not_empty(self):
        assert len(gemini.MANAGED_KEYS) > 0

    def test_managed_keys_includes_model(self):
        assert "GEMINI_MODEL" in gemini.MANAGED_KEYS

    def test_managed_keys_includes_api_key(self):
        assert "GEMINI_API_KEY" in gemini.MANAGED_KEYS

    def test_managed_keys_includes_oauth_token(self):
        assert "OAUTH_TOKEN" in gemini.MANAGED_KEYS


class TestWriteToolConfig:
    def test_writes_ucode_env_file(self, tmp_path, monkeypatch):
        import ucode.config_io as config_io_mod

        env_path = tmp_path / "ucode.env"
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", env_path)
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "backup")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr("ucode.agents.gemini.save_state", lambda s: None)
        monkeypatch.setattr(
            "ucode.agents.gemini.get_databricks_token",
            lambda ws, profile=None, **kwargs: "fake-token",
        )

        gemini.write_tool_config({"workspace": WS}, "some-model")

        env = env_path.read_text()
        assert 'GEMINI_MODEL="some-model"' in env
        assert f'GOOGLE_GEMINI_BASE_URL="{WS}/ai-gateway/gemini"' in env

    def test_does_not_write_settings_json(self, tmp_path, monkeypatch):
        import ucode.config_io as config_io_mod

        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({"theme": "dark", "otherKey": 123}))
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / "ucode.env")
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "backup")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr("ucode.agents.gemini.save_state", lambda s: None)
        monkeypatch.setattr(
            "ucode.agents.gemini.get_databricks_token",
            lambda ws, profile=None, **kwargs: "fake-token",
        )

        gemini.write_tool_config({"workspace": WS}, "some-model")

        settings = json.loads(settings_path.read_text())
        assert settings["theme"] == "dark"
        assert settings["otherKey"] == 123
        assert "security" not in settings
