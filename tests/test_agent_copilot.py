"""Tests for agents/copilot.py."""

from __future__ import annotations

import json

import pytest

from ucode.agents import copilot

WS = "https://example.databricks.com"
# >= MINIMUM_COPILOT_ANTHROPIC_VERSION, for tests that need the anthropic path reachable.
NEW_ENOUGH_VERSION = "1.0.82"


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
    """Claude models get Copilot's native `anthropic` provider (so Copilot's own
    cache_control logic runs); everything else (codex/gpt-5) stays on `openai`
    against the MLflow gateway.

    Verified live against a real workspace: Databricks' AI Gateway 401s on the
    `x-api-key` auth Copilot's `anthropic` provider sends by default
    (COPILOT_PROVIDER_API_KEY) — it wants `Authorization: Bearer`
    (COPILOT_PROVIDER_BEARER_TOKEN, used for both provider types here). And an
    unrecognized COPILOT_MODEL (a Databricks catalog id) makes Copilot fall
    back to defaults that include sending `temperature`, which current-gen
    Claude models 400 on — hence the separate COPILOT_PROVIDER_MODEL_ID
    (canonical name) / COPILOT_PROVIDER_WIRE_MODEL (actual wire id) split.

    All of that only works on Copilot >= 1.0.81-6 (see TestSupportsAnthropicProvider),
    so every test here mocks a new-enough installed version."""

    @pytest.fixture(autouse=True)
    def _new_enough_copilot(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: NEW_ENOUGH_VERSION)

    def test_claude_model_uses_anthropic_provider_type(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"

    def test_claude_model_uses_native_anthropic_base_url(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")
        assert env["COPILOT_PROVIDER_BASE_URL"] == f"{WS}/ai-gateway/anthropic"

    def test_claude_model_sets_bearer_token_not_api_key(self):
        env = copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok123")
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "tok123"
        assert "COPILOT_PROVIDER_API_KEY" not in env

    def test_claude_model_splits_canonical_id_from_wire_model(self):
        env = copilot.render_env_overlay(WS, "system.ai.claude-sonnet-5", "tok")
        assert env["COPILOT_PROVIDER_MODEL_ID"] == "claude-sonnet-5"
        assert env["COPILOT_PROVIDER_WIRE_MODEL"] == "system.ai.claude-sonnet-5"
        assert "COPILOT_MODEL" not in env

    def test_claude_model_matches_case_insensitively(self):
        env = copilot.render_env_overlay(WS, "us.anthropic.Claude-Opus-4-8", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"
        assert env["COPILOT_PROVIDER_MODEL_ID"] == "claude-opus-4-8"

    def test_non_claude_model_uses_openai_provider_type(self):
        env = copilot.render_env_overlay(WS, "gpt-5", "t")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"

    def test_non_claude_model_uses_mlflow_base_url(self):
        env = copilot.render_env_overlay(WS, "gpt-5", "t")
        assert env["COPILOT_PROVIDER_BASE_URL"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_non_claude_model_sets_model_and_bearer_token(self):
        env = copilot.render_env_overlay(WS, "gpt-5", "tok123")
        assert env["COPILOT_MODEL"] == "gpt-5"
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "tok123"
        assert "COPILOT_PROVIDER_API_KEY" not in env
        assert "COPILOT_PROVIDER_MODEL_ID" not in env
        assert "COPILOT_PROVIDER_WIRE_MODEL" not in env

    def test_sets_offline_true(self):
        env = copilot.render_env_overlay(WS, "gpt-5", "t")
        assert env["COPILOT_OFFLINE"] == "true"

    def test_sets_oauth_token_for_both_families(self):
        assert copilot.render_env_overlay(WS, "claude-sonnet-4-6", "tok")["OAUTH_TOKEN"] == "tok"
        assert copilot.render_env_overlay(WS, "gpt-5", "tok")["OAUTH_TOKEN"] == "tok"


class TestOldCopilotFallsBackToOpenai:
    """Below 1.0.81-6, Copilot always sends `temperature` on the anthropic
    path regardless of model id, and current-gen Claude models 400 on it —
    verified live (1.0.79, 1.0.80, 1.0.81-0 all fail; 1.0.81-6 onward works).
    So an old Copilot must keep getting the openai path even for Claude
    models: uncached, but that's the pre-fix status quo, not a regression."""

    def test_old_version_keeps_claude_on_openai(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.80")
        env = copilot.render_env_overlay(WS, "system.ai.claude-sonnet-5", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"
        assert env["COPILOT_MODEL"] == "system.ai.claude-sonnet-5"

    def test_unknown_version_keeps_claude_on_openai(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "unknown")
        env = copilot.render_env_overlay(WS, "system.ai.claude-sonnet-5", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"

    def test_new_enough_version_uses_anthropic(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.81-6")
        env = copilot.render_env_overlay(WS, "system.ai.claude-sonnet-5", "tok")
        assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"


class TestSupportsAnthropicProvider:
    def test_false_below_the_minimum_patch(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.80")
        assert copilot._supports_anthropic_provider() is False

    def test_false_just_below_the_minimum_prerelease(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.81-0")
        assert copilot._supports_anthropic_provider() is False

    def test_true_at_the_exact_minimum_prerelease(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.81-6")
        assert copilot._supports_anthropic_provider() is True

    def test_true_above_the_minimum_prerelease(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.81-7")
        assert copilot._supports_anthropic_provider() is True

    def test_true_for_the_final_release_of_the_minimum_version(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.81")
        assert copilot._supports_anthropic_provider() is True

    def test_true_for_a_newer_minor_version(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "1.0.83")
        assert copilot._supports_anthropic_provider() is True

    def test_false_when_version_cannot_be_determined(self, monkeypatch):
        monkeypatch.setattr(copilot, "agent_version", lambda binary: "unknown")
        assert copilot._supports_anthropic_provider() is False


class TestParseCopilotVersion:
    def test_parses_a_final_release(self):
        assert copilot._parse_copilot_version("1.0.82") == (1, 0, 82, copilot._UNRELEASED)

    def test_parses_a_prerelease(self):
        assert copilot._parse_copilot_version("1.0.81-6") == (1, 0, 81, 6)

    def test_a_final_release_sorts_after_its_prereleases(self):
        final = copilot._parse_copilot_version("1.0.81")
        prerelease = copilot._parse_copilot_version("1.0.81-14")
        assert final > prerelease

    def test_returns_none_for_unparseable_input(self):
        assert copilot._parse_copilot_version("unknown") is None


class TestIsClaudeModel:
    def test_true_for_canonical_name(self):
        assert copilot._is_claude_model("claude-sonnet-5") is True

    def test_true_for_bedrock_style_slug(self):
        assert copilot._is_claude_model("us.anthropic.claude-opus-4-8") is True

    def test_false_for_codex(self):
        assert copilot._is_claude_model("gpt-5") is False

    def test_false_for_catalog_qualified_gpt(self):
        assert copilot._is_claude_model("system.ai.gpt-5") is False


class TestCanonicalClaudeModelId:
    def test_strips_databricks_catalog_prefix(self):
        assert copilot._canonical_claude_model_id("system.ai.claude-sonnet-5") == "claude-sonnet-5"

    def test_strips_bedrock_region_and_provider_prefix(self):
        result = copilot._canonical_claude_model_id("us.anthropic.claude-opus-4-8")
        assert result == "claude-opus-4-8"

    def test_leaves_bare_canonical_name_unchanged(self):
        assert copilot._canonical_claude_model_id("claude-haiku-4-5") == "claude-haiku-4-5"

    def test_falls_back_to_the_input_when_no_match(self):
        assert copilot._canonical_claude_model_id("weird-model") == "weird-model"

    def test_lowercases_a_mixed_case_id(self):
        assert (
            copilot._canonical_claude_model_id("us.anthropic.Claude-Opus-4-8") == "claude-opus-4-8"
        )

    def test_lowercases_the_fallback_when_no_match(self):
        assert copilot._canonical_claude_model_id("Weird-Model") == "weird-model"

    def test_strips_a_trailing_bedrock_version_suffix(self):
        assert copilot._canonical_claude_model_id("claude-opus-4-8-v1:0") == "claude-opus-4-8"

    def test_strips_a_bedrock_version_suffix_without_a_colon_part(self):
        assert copilot._canonical_claude_model_id("claude-opus-4-8-v1") == "claude-opus-4-8"


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


class TestWriteToolConfig:
    def test_switching_families_clears_the_stale_model_selection_keys(self, tmp_path, monkeypatch):
        import ucode.agents.copilot as cp_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        env_path = tmp_path / "ucode.env"
        monkeypatch.setattr(cp_mod, "COPILOT_ENV_PATH", env_path)
        monkeypatch.setattr(cp_mod, "COPILOT_BACKUP_PATH", tmp_path / "backup")
        monkeypatch.setattr(cp_mod, "save_state", lambda state: None)
        monkeypatch.setattr(cp_mod, "agent_version", lambda binary: NEW_ENOUGH_VERSION)

        state = {"workspace": WS}
        cp_mod.write_tool_config(state, "system.ai.claude-sonnet-5", token="tok-a")
        written = config_io_mod.parse_dotenv(env_path)
        assert written["COPILOT_PROVIDER_MODEL_ID"] == "claude-sonnet-5"
        assert written["COPILOT_PROVIDER_WIRE_MODEL"] == "system.ai.claude-sonnet-5"
        assert "COPILOT_MODEL" not in written

        cp_mod.write_tool_config(state, "gpt-5", token="tok-b")
        written = config_io_mod.parse_dotenv(env_path)
        assert written["COPILOT_MODEL"] == "gpt-5"
        assert "COPILOT_PROVIDER_MODEL_ID" not in written
        assert "COPILOT_PROVIDER_WIRE_MODEL" not in written


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
            "COPILOT_PROVIDER_MODEL_ID",
            "COPILOT_PROVIDER_WIRE_MODEL",
            "COPILOT_PROVIDER_BEARER_TOKEN",
            "COPILOT_OFFLINE",
            "OAUTH_TOKEN",
        ):
            assert key in copilot.MANAGED_KEYS


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
