"""Tests for agents/opencode.py."""

from __future__ import annotations

import json
from unittest.mock import patch

from ucode.agents import opencode

WS = "https://example.databricks.com"


def _base_urls() -> dict[str, str]:
    return {
        "anthropic": f"{WS}/ai-gateway/anthropic/v1",
        "gemini": f"{WS}/ai-gateway/gemini/v1beta",
        "oss": f"{WS}/ai-gateway/mlflow/v1",
    }


class TestOpencodeSpec:
    def test_binary(self):
        assert opencode.SPEC["binary"] == "opencode"

    def test_package(self):
        assert opencode.SPEC["package"] == "opencode-ai@1"

    def test_display(self):
        assert opencode.SPEC["display"] == "OpenCode"

    def test_config_path_is_under_ucode_xdg_home(self):
        assert opencode.SPEC["config_path"] == (
            opencode.OPENCODE_XDG_CONFIG_HOME / "opencode" / "opencode.json"
        )

    def test_update_check_uses_latest_stable_v1(self, monkeypatch):
        monkeypatch.setattr(
            opencode,
            "available_npm_package_update",
            lambda _package: ("1.18.15", "1.18.16"),
        )

        assert opencode.is_update_available() == ("1.18.15", "1.18.16")

    def test_update_check_ignores_npm_beta(self, monkeypatch):
        monkeypatch.setattr(
            opencode,
            "available_npm_package_update",
            lambda _package: ("1.18.16", "0.0.0-beta-202605152242"),
        )

        assert opencode.is_update_available() is None

    def test_requires_version_with_custom_provider_fetch(self, monkeypatch):
        monkeypatch.setattr(opencode, "agent_version", lambda _binary: "1.0.219")

        message = opencode.minimum_version_error()

        assert message is not None
        assert "requires OpenCode 1.0.220 or newer" in message
        assert "npm install -g opencode-ai@1" in message

    def test_supported_version_needs_no_required_update(self, monkeypatch):
        monkeypatch.setattr(opencode, "agent_version", lambda _binary: "1.0.220")

        assert opencode.minimum_version_error() is None


class TestAuthPlugin:
    def test_calls_cross_platform_auth_token_helper_only_when_refreshing(self, monkeypatch):
        monkeypatch.setattr(
            opencode,
            "build_auth_token_argv",
            lambda workspace, profile, use_pat=False: [
                "/opt/ucode",
                "auth-token",
                "--host",
                workspace,
                "--profile",
                profile,
            ],
        )

        plugin = opencode.render_auth_plugin({"workspace": WS, "profile": "my profile"})

        assert (
            'const AUTH_COMMAND = ["/opt/ucode", "auth-token", "--host", '
            f'"{WS}", "--profile", "my profile", "--force-refresh"]'
        ) in plugin
        assert "run(AUTH_COMMAND[0], AUTH_COMMAND.slice(1)" in plugin
        assert '"chat.headers"' not in plugin

    def test_installs_cached_refreshing_fetch_on_databricks_providers(self):
        plugin = opencode.render_auth_plugin({"workspace": WS})

        assert "config: async (config)" in plugin
        assert "options.fetch = databricksFetch" in plugin
        assert "expiresAt <= Date.now() + REFRESH_SKEW_MS" in plugin
        assert 'headers.set("Authorization", "Bearer " + token)' in plugin
        assert "if (response.status !== 401) return response" in plugin
        assert "return fetch(input, requestWithToken(input, init, accessToken))" in plugin

    def test_refresh_is_single_flighted(self):
        plugin = opencode.render_auth_plugin({"workspace": WS})

        assert "if (!refreshPromise)" in plugin
        assert "mintToken().finally(() => { refreshPromise = undefined })" in plugin


class TestRenderOverlay:
    def test_sets_model(self):
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), {})
        assert overlay["model"] == "claude-sonnet"

    def test_anthropic_provider_added_when_models_present(self):
        models = {"anthropic": ["claude-sonnet"], "gemini": []}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert "databricks-anthropic" in overlay["provider"]

    def test_gemini_provider_added_when_models_present(self):
        models = {"anthropic": [], "gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        assert "databricks-google" in overlay["provider"]

    def test_oss_provider_added_when_models_present(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        assert "databricks-oss" in overlay["provider"]

    def test_oss_provider_uses_ai_sdk_openai_package(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        assert overlay["provider"]["databricks-oss"]["npm"] == "@ai-sdk/openai"

    def test_deepseek_uses_oss_provider(self):
        model = "system.ai.deepseek-v4-pro"

        overlay, _ = opencode.render_overlay(model, "tok", _base_urls(), {"oss": [model]})

        assert overlay["model"] == f"databricks-oss/{model}"
        assert model in overlay["provider"]["databricks-oss"]["models"]

    def test_both_providers_when_both_present(self):
        models = {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert "databricks-anthropic" in overlay["provider"]
        assert "databricks-google" in overlay["provider"]

    def test_no_provider_key_when_no_models(self):
        overlay, _ = opencode.render_overlay("model", "tok", _base_urls(), {})
        assert "provider" not in overlay

    def test_anthropic_base_url(self):
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        options = overlay["provider"]["databricks-anthropic"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/anthropic/v1"

    def test_gemini_base_url(self):
        models = {"gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        options = overlay["provider"]["databricks-google"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/gemini/v1beta"

    def test_oss_base_url(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        options = overlay["provider"]["databricks-oss"]["options"]
        assert options["baseURL"] == f"{WS}/ai-gateway/mlflow/v1"

    def test_glm_gets_token_limits(self):
        models = {"oss": ["system.ai.glm-5-2"]}
        overlay, _ = opencode.render_overlay("system.ai.glm-5-2", "tok", _base_urls(), models)
        glm = overlay["provider"]["databricks-oss"]["models"]["system.ai.glm-5-2"]
        # OpenCode's schema requires both context and output on `limit`.
        assert glm["limit"] == {"context": 200000, "output": 25000}

    def test_non_glm_oss_model_has_no_output_cap(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        kimi = overlay["provider"]["databricks-oss"]["models"]["system.ai.kimi-k2-7-code"]
        assert "limit" not in kimi

    def test_qwen_gets_token_limits(self):
        model = "system.ai.qwen35-122b-a10b"
        overlay, _ = opencode.render_overlay(model, "tok", _base_urls(), {"oss": [model]})
        qwen = overlay["provider"]["databricks-oss"]["models"][model]
        assert qwen["limit"] == {"context": 262144, "output": 25000}

    def test_oss_provider_opts_out_of_the_prompt_cache_key(self):
        # OpenCode stamps `prompt_cache_key` on every `@ai-sdk/openai` request and
        # the gateway rejects the unknown field. It reads the opt-out from the
        # provider options only, so a per-model entry would not suppress it.
        model = "system.ai.qwen35-122b-a10b"
        overlay, _ = opencode.render_overlay(model, "tok", _base_urls(), {"oss": [model]})
        assert overlay["provider"]["databricks-oss"]["options"]["setCacheKey"] is False
        assert "setCacheKey" not in overlay["provider"]["databricks-oss"]["models"][model]

    def test_token_in_api_key(self):
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "mytoken", _base_urls(), models)
        assert overlay["provider"]["databricks-anthropic"]["options"]["apiKey"] == "mytoken"

    def test_authorization_header(self):
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        headers = overlay["provider"]["databricks-anthropic"]["options"]["headers"]
        assert headers["Authorization"] == "Bearer tok"

    def test_anthropic_tool_streaming_disabled(self):
        # @ai-sdk/anthropic injects `eager_input_streaming: true` on tool defs,
        # which the Databricks gateway rejects. opencode's auto-disable skips
        # Claude models, so we opt out per-model. The setting must live in
        # `models.<m>.options` — per-call providerOptions — not provider options.
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        model_entry = overlay["provider"]["databricks-anthropic"]["models"]["claude-sonnet"]
        assert model_entry["options"]["toolStreaming"] is False

    def test_user_agent_header_anthropic(self, monkeypatch):
        # UA must live at the per-model level — OpenCode clobbers
        # provider-level `headers["User-Agent"]` in session/llm.ts.
        monkeypatch.setattr(opencode, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(opencode, "agent_version", lambda binary: "0.74.0")
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        model_headers = overlay["provider"]["databricks-anthropic"]["models"]["claude-sonnet"][
            "headers"
        ]
        assert model_headers["User-Agent"] == "ucode/0.1.0 opencode/0.74.0"

    def test_user_agent_header_gemini(self, monkeypatch):
        monkeypatch.setattr(opencode, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(opencode, "agent_version", lambda binary: "0.74.0")
        models = {"gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        model_headers = overlay["provider"]["databricks-google"]["models"]["gemini-2"]["headers"]
        assert model_headers["User-Agent"] == "ucode/0.1.0 opencode/0.74.0"

    def test_provider_level_headers_only_authorization(self, monkeypatch):
        # Sanity: provider-level headers should NOT include User-Agent (since
        # it's clobbered there) — only Authorization.
        models = {"anthropic": ["claude-sonnet"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        provider_headers = overlay["provider"]["databricks-anthropic"]["options"]["headers"]
        assert "User-Agent" not in provider_headers
        assert provider_headers["Authorization"] == "Bearer tok"

    def test_managed_keys_include_model(self):
        _, keys = opencode.render_overlay("model", "tok", _base_urls(), {})
        assert ["model"] in keys

    def test_managed_keys_include_anthropic_provider(self):
        models = {"anthropic": ["claude-sonnet"]}
        _, keys = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert ["provider", "databricks-anthropic"] in keys

    def test_managed_keys_include_gemini_provider(self):
        models = {"gemini": ["gemini-2"]}
        _, keys = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        assert ["provider", "databricks-google"] in keys

    def test_managed_keys_include_oss_provider(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        _, keys = opencode.render_overlay("system.ai.kimi-k2-7-code", "tok", _base_urls(), models)
        assert ["provider", "databricks-oss"] in keys

    def test_anthropic_models_listed(self):
        models = {"anthropic": ["claude-sonnet", "claude-haiku"]}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        provider_models = overlay["provider"]["databricks-anthropic"]["models"]
        assert "claude-sonnet" in provider_models
        assert "claude-haiku" in provider_models

    def test_prefixes_anthropic_model_with_provider_id(self):
        models = {"anthropic": ["claude-sonnet"], "gemini": []}
        overlay, _ = opencode.render_overlay("claude-sonnet", "tok", _base_urls(), models)
        assert overlay["model"] == "databricks-anthropic/claude-sonnet"

    def test_prefixes_gemini_model_with_provider_id(self):
        models = {"anthropic": [], "gemini": ["gemini-2"]}
        overlay, _ = opencode.render_overlay("gemini-2", "tok", _base_urls(), models)
        assert overlay["model"] == "databricks-google/gemini-2"

    def test_prefixes_oss_model_with_provider_id(self):
        models = {"oss": ["system.ai.kimi-k2-7-code"]}
        overlay, _ = opencode.render_overlay(
            "system.ai.kimi-k2-7-code", "tok", _base_urls(), models
        )
        assert overlay["model"] == "databricks-oss/system.ai.kimi-k2-7-code"


class TestMcpServerConfig:
    # ucode registers the `ucode mcp-proxy ...` bridge as a `local` (stdio) MCP
    # server; the proxy handles token refresh, so no URL/bearer header here.
    PROXY_ARGV = ["ucode", "mcp-proxy", "--url", f"{WS}/api/2.0/mcp/functions/system/ai"]

    def test_builds_local_server_entry_from_proxy_argv(self):
        entry = opencode.build_mcp_server_entry(self.PROXY_ARGV)

        assert entry == {
            "type": "local",
            "command": self.PROXY_ARGV,
            "enabled": True,
        }

    def test_writes_mcp_server_without_clobbering_existing_config(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        config_file.write_text(
            json.dumps(
                {
                    "model": "existing-model",
                    "mcp": {"old-server": {"type": "local", "command": ["old"]}},
                }
            ),
            encoding="utf-8",
        )

        removed = oc_mod.write_mcp_server_config("github", self.PROXY_ARGV)

        written = json.loads(config_file.read_text())
        assert removed is False
        assert written["model"] == "existing-model"
        assert written["mcp"]["old-server"] == {"type": "local", "command": ["old"]}
        assert written["mcp"]["github"] == {
            "type": "local",
            "command": self.PROXY_ARGV,
            "enabled": True,
        }

    def test_reports_replaced_mcp_server(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        config_file.write_text(json.dumps({"mcp": {"github": {"old": True}}}), encoding="utf-8")

        removed = oc_mod.write_mcp_server_config("github", self.PROXY_ARGV)

        assert removed is True
        written = json.loads(config_file.read_text())
        assert written["mcp"]["github"]["command"] == self.PROXY_ARGV

    def test_removes_mcp_server_without_clobbering_others(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod

        config_file = tmp_path / "opencode.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        config_file.write_text(
            json.dumps(
                {
                    "model": "existing-model",
                    "mcp": {
                        "github": {"url": "old"},
                        "jira": {"url": "keep"},
                    },
                }
            ),
            encoding="utf-8",
        )

        removed = oc_mod.remove_mcp_server_config("github")

        written = json.loads(config_file.read_text())
        assert removed is True
        assert "github" not in written["mcp"]
        assert written["mcp"]["jira"] == {"url": "keep"}
        assert written["model"] == "existing-model"


class TestBuildRuntimeEnv:
    def test_sets_oauth_token_for_mcp(self):
        env = opencode.build_runtime_env("tok")

        assert env["OAUTH_TOKEN"] == "tok"

    def test_sets_ucode_xdg_config_home(self):
        env = opencode.build_runtime_env("tok")

        assert env["XDG_CONFIG_HOME"] == str(opencode.OPENCODE_XDG_CONFIG_HOME)


class TestOpencodeDefaultModel:
    def test_prefers_anthropic(self):
        state = {"opencode_models": {"anthropic": ["claude-sonnet"], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "claude-sonnet"

    def test_falls_back_to_gemini(self):
        state = {"opencode_models": {"anthropic": [], "gemini": ["gemini-2"]}}
        assert opencode.default_model(state) == "gemini-2"

    def test_falls_back_to_oss(self):
        state = {
            "opencode_models": {
                "anthropic": [],
                "gemini": [],
                "oss": ["system.ai.kimi-k2-7-code"],
            }
        }
        assert opencode.default_model(state) == "system.ai.kimi-k2-7-code"

    def test_returns_none_when_empty(self):
        assert opencode.default_model({}) is None
        assert opencode.default_model({"opencode_models": {}}) is None

    def test_opencode_default_model_wins_over_bucketed_models(self):
        state = {
            "opencode_default_model": "admin-chosen-default",
            "opencode_models": {"anthropic": ["claude-sonnet"]},
        }
        assert opencode.default_model(state) == "admin-chosen-default"


class TestOpencodeValidateCmd:
    def test_starts_with_binary(self):
        cmd = opencode.validate_cmd("opencode")
        assert cmd[0] == "opencode"

    def test_uses_run_subcommand(self):
        cmd = opencode.validate_cmd("opencode")
        assert "run" in cmd

    def test_has_prompt(self):
        cmd = opencode.validate_cmd("opencode")
        assert len(cmd) > 2


class TestWriteToolConfigStaleProviderCleanup:
    def test_stale_providers_removed_before_merge(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        stale = {
            "provider": {
                "databricks-anthropic": {"old": True},
                "databricks-google": {"old": True},
                "other-provider": {"keep": True},
            }
        }
        config_file.write_text(json.dumps(stale), encoding="utf-8")

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"anthropic": ["claude-sonnet"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        providers = written.get("provider", {})
        # stale entry is replaced with new data, not kept as-is
        assert providers.get("databricks-anthropic") != {"old": True}
        # unmanaged provider entry survives
        assert providers.get("other-provider") == {"keep": True}
        # OpenCode 1.0.0 discovers `plugin/`; plural `plugins/` came later.
        plugin = config_file.parent / "plugin" / opencode.OPENCODE_AUTH_PLUGIN_PATH.name
        assert plugin.exists()
        assert "options.fetch = databricksFetch" in plugin.read_text()

    def test_config_written_with_correct_model(self, tmp_path, monkeypatch):
        import ucode.agents.opencode as oc_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "opencode.json"
        backup_file = tmp_path / "opencode-backup.json"
        monkeypatch.setattr(oc_mod, "OPENCODE_CONFIG_PATH", config_file)
        monkeypatch.setattr(oc_mod, "OPENCODE_BACKUP_PATH", backup_file)

        state = {
            "workspace": WS,
            "base_urls": {"opencode": _base_urls()},
            "opencode_models": {"anthropic": ["claude-sonnet"]},
            "managed_configs": {},
        }

        with (
            patch("ucode.agents.opencode.get_databricks_token", return_value="tok"),
            patch("ucode.agents.opencode.save_state"),
        ):
            oc_mod.write_tool_config(state, "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-anthropic/claude-sonnet"
