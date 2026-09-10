"""Tests for agents/copilot.py."""

from __future__ import annotations

import json
import threading

import pytest

from ucode.agents import copilot

WS = "https://example.databricks.com"


class TestCopilotSpec:
    def test_binary(self):
        assert copilot.SPEC["binary"] == "copilot"

    def test_package(self):
        assert copilot.SPEC["package"] == "@github/copilot"

    def test_display(self):
        assert copilot.SPEC["display"] == "GitHub Copilot CLI"

    def test_config_path_is_ucode_env_file(self):
        assert copilot.SPEC["config_path"].name == "ucode.env"


class TestRenderEnvOverlay:
    def test_sets_provider_base_url(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")
        assert env["COPILOT_PROVIDER_BASE_URL"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_sets_provider_type(self):
        env = copilot.render_env_overlay(WS, "m", "t")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"

    def test_sets_model(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")
        assert env["COPILOT_MODEL"] == "claude-sonnet-4-6"

    def test_sets_bearer_token(self):
        env = copilot.render_env_overlay(WS, "m", "tok123")
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "tok123"

    def test_sets_offline_true(self):
        env = copilot.render_env_overlay(WS, "m", "t")
        assert env["COPILOT_OFFLINE"] == "true"


class TestBuildRuntimeEnv:
    def test_inherits_path(self):
        env = copilot.build_runtime_env(WS, "m", "t")
        assert "PATH" in env

    def test_overrides_copilot_vars(self):
        env = copilot.build_runtime_env(WS, "m", "tok")
        assert env["COPILOT_PROVIDER_BASE_URL"] == f"{WS}/ai-gateway/mlflow/v1"
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "tok"

    def test_sets_oauth_token_for_mcp(self):
        env = copilot.build_runtime_env(WS, "m", "tok")
        assert env["OAUTH_TOKEN"] == "tok"


class TestMcpServerConfig:
    # ucode registers the `ucode mcp-proxy ...` bridge as a `local` (stdio) MCP
    # server; the proxy refreshes the token, so no URL/bearer header here.
    PROXY_ARGV = ["ucode", "mcp-proxy", "--url", f"{WS}/api/2.0/mcp/functions/system/ai"]

    def test_builds_local_server_entry_from_proxy_argv(self):
        entry = copilot.build_mcp_server_entry(self.PROXY_ARGV)

        assert entry == {
            "type": "local",
            "command": self.PROXY_ARGV[0],
            "args": self.PROXY_ARGV[1:],
            "tools": ["*"],
        }

    def test_writes_mcp_server_without_clobbering_existing_config(self, tmp_path, monkeypatch):
        import ucode.agents.copilot as cp_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "mcp-config.json"
        backup_file = tmp_path / "copilot-mcp-backup.json"
        monkeypatch.setattr(cp_mod, "COPILOT_MCP_CONFIG_PATH", config_file)
        monkeypatch.setattr(cp_mod, "COPILOT_MCP_BACKUP_PATH", backup_file)

        config_file.write_text(
            json.dumps(
                {
                    "other": True,
                    "mcpServers": {"old-server": {"type": "stdio", "command": "old"}},
                }
            ),
            encoding="utf-8",
        )

        removed = cp_mod.write_mcp_server_config("github", self.PROXY_ARGV)

        written = json.loads(config_file.read_text())
        assert removed is False
        assert written["other"] is True
        assert written["mcpServers"]["old-server"] == {"type": "stdio", "command": "old"}
        assert written["mcpServers"]["github"] == {
            "type": "local",
            "command": self.PROXY_ARGV[0],
            "args": self.PROXY_ARGV[1:],
            "tools": ["*"],
        }

    def test_reports_replaced_mcp_server(self, tmp_path, monkeypatch):
        import ucode.agents.copilot as cp_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "mcp-config.json"
        backup_file = tmp_path / "copilot-mcp-backup.json"
        monkeypatch.setattr(cp_mod, "COPILOT_MCP_CONFIG_PATH", config_file)
        monkeypatch.setattr(cp_mod, "COPILOT_MCP_BACKUP_PATH", backup_file)

        config_file.write_text(
            json.dumps({"mcpServers": {"github": {"old": True}}}),
            encoding="utf-8",
        )

        removed = cp_mod.write_mcp_server_config("github", self.PROXY_ARGV)

        assert removed is True
        written = json.loads(config_file.read_text())
        assert written["mcpServers"]["github"]["command"] == self.PROXY_ARGV[0]

    def test_removes_mcp_server_without_clobbering_others(self, tmp_path, monkeypatch):
        import ucode.agents.copilot as cp_mod

        config_file = tmp_path / "mcp-config.json"
        monkeypatch.setattr(cp_mod, "COPILOT_MCP_CONFIG_PATH", config_file)
        config_file.write_text(
            json.dumps(
                {
                    "other": True,
                    "mcpServers": {
                        "github": {"url": "old"},
                        "jira": {"url": "keep"},
                    },
                }
            ),
            encoding="utf-8",
        )

        removed = cp_mod.remove_mcp_server_config("github")

        written = json.loads(config_file.read_text())
        assert removed is True
        assert "github" not in written["mcpServers"]
        assert written["mcpServers"]["jira"] == {"url": "keep"}
        assert written["other"] is True


class TestDefaultModel:
    def test_prefers_claude_sonnet(self):
        state = {
            "claude_models": {"sonnet": "s4", "opus": "o4", "haiku": "h4"},
            "codex_models": ["gpt-5"],
        }
        assert copilot.default_model(state) == "s4"

    def test_falls_back_to_opus(self):
        state = {"claude_models": {"opus": "o4", "haiku": "h4"}}
        assert copilot.default_model(state) == "o4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert copilot.default_model(state) == "h4"

    def test_falls_back_to_codex_when_no_claude(self):
        state = {"codex_models": ["gpt-5", "gpt-5-mini"]}
        assert copilot.default_model(state) == "gpt-5"

    def test_returns_none_when_no_models(self):
        assert copilot.default_model({}) is None

    def test_ignores_gemini_models(self):
        # Gemini is excluded — Databricks' Gemini translator rejects copilot's request shape.
        state = {"gemini_models": ["gemini-2-5-pro"]}
        assert copilot.default_model(state) is None


class TestManagedKeys:
    def test_includes_required_vars(self):
        for key in (
            "COPILOT_PROVIDER_TYPE",
            "COPILOT_PROVIDER_BASE_URL",
            "COPILOT_MODEL",
            "COPILOT_PROVIDER_BEARER_TOKEN",
            "COPILOT_OFFLINE",
            "OAUTH_TOKEN",
        ):
            assert key in copilot.MANAGED_KEYS


class TestRefreshTokenOnceModelOverride:
    def test_model_override_wins_over_default_model(self, monkeypatch):
        seen: dict = {}

        def fake_write_tool_config(state, model, force_refresh=False):
            seen["model"] = model
            return state, "tok"

        monkeypatch.setattr(copilot, "write_tool_config", fake_write_tool_config)
        state = {"claude_models": {"sonnet": "discovered"}}

        model, token = copilot._refresh_token_once(state, model_override="explicit-model")

        assert model == "explicit-model"
        assert token == "tok"
        assert seen["model"] == "explicit-model"

    def test_falls_back_to_default_model_without_override(self, monkeypatch):
        monkeypatch.setattr(
            copilot, "write_tool_config", lambda state, model, force_refresh=False: (state, "tok")
        )
        state = {"claude_models": {"sonnet": "discovered"}}

        model, _ = copilot._refresh_token_once(state)

        assert model == "discovered"

    def test_raises_when_no_override_and_no_default(self):
        with pytest.raises(RuntimeError, match="No Copilot model is available"):
            copilot._refresh_token_once({})


class TestRefreshForeverModelOverride:
    def test_forwards_model_override_to_each_refresh(self, monkeypatch):
        calls: list[str | None] = []

        def fake_refresh_token_once(state, *, force_refresh=False, model_override=None):
            calls.append(model_override)
            if len(calls) >= 2:
                stop_event.set()
            return model_override, "tok"

        monkeypatch.setattr(copilot, "_refresh_token_once", fake_refresh_token_once)
        monkeypatch.setattr(copilot, "TOKEN_REFRESH_INTERVAL_SECONDS", 0)

        stop_event = threading.Event()
        copilot._refresh_forever({}, stop_event, model_override="pinned-model")

        assert calls == ["pinned-model", "pinned-model"]


class TestValidateCmd:
    def test_starts_with_binary(self):
        cmd = copilot.validate_cmd("copilot")
        assert cmd[0] == "copilot"

    def test_has_prompt_flag(self):
        cmd = copilot.validate_cmd("copilot")
        assert "--prompt" in cmd

    def test_adds_ucode_mcp_config_when_present(self, tmp_path, monkeypatch):
        mcp_path = tmp_path / "ucode-mcp-config.json"
        mcp_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(copilot, "COPILOT_MCP_CONFIG_PATH", mcp_path)

        cmd = copilot.validate_cmd("copilot")

        assert cmd[:3] == ["copilot", "--additional-mcp-config", f"@{mcp_path}"]


class TestManagedModels:
    def test_managed_models_win_over_the_shared_discovery_lists(self):
        state = {
            "copilot_models": ["system.ai.gpt-5"],
            "claude_models": {"sonnet": "shared-should-not-win"},
        }
        assert copilot.default_model(state) == "system.ai.gpt-5"

    def test_falls_back_to_the_shared_lists_without_a_managed_config(self):
        assert copilot.default_model({"claude_models": {"sonnet": "discovered"}}) == "discovered"

    def test_copilot_default_model_wins_over_allowlist(self):
        state = {
            "copilot_default_model": "admin-chosen-default",
            "copilot_models": ["system.ai.gpt-5"],
        }
        assert copilot.default_model(state) == "admin-chosen-default"
