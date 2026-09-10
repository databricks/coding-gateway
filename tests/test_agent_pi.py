"""Tests for agents/pi.py."""

from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import patch

from ucode.agents import pi

WS = "https://example.databricks.com"


def _base_urls() -> dict[str, str]:
    # Native API per family — see agents/pi.py docstring for path conventions.
    return {
        "claude": f"{WS}/ai-gateway/anthropic",
        "openai": f"{WS}/ai-gateway/codex/v1",
        "gemini": f"{WS}/ai-gateway/gemini/v1beta",
    }


def _empty() -> dict:
    """No-models input bundle for render_overlay."""
    return {
        "claude_models": {},
        "codex_models": [],
        "gemini_models": [],
    }


def _overlay(model: str, token: str = "tok", **kwargs):
    """Wrapper to call render_overlay with sensible defaults so tests stay terse."""
    bundle = {**_empty(), **kwargs}
    return pi.render_overlay(
        model,
        token,
        _base_urls(),
        bundle["claude_models"],
        bundle["codex_models"],
        bundle["gemini_models"],
    )


class TestPiSpec:
    def test_binary(self):
        assert pi.SPEC["binary"] == "pi"

    def test_package(self):
        assert pi.SPEC["package"] == "@earendil-works/pi-coding-agent"

    def test_display(self):
        assert pi.SPEC["display"] == "Pi"

    def test_config_path_under_pi_agent_dir(self):
        assert pi.SPEC["config_path"].name == "models.json"
        assert pi.SPEC["config_path"].parent.name == "agent"
        assert pi.PI_UCODE_HOME in pi.SPEC["config_path"].parents


class TestRenderOverlayProviders:
    def test_no_providers_when_no_models(self):
        overlay, _ = _overlay("foo")
        assert "providers" not in overlay

    def test_claude_provider_uses_anthropic_messages(self):
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        provider = overlay["providers"]["databricks-claude"]
        assert provider["api"] == "anthropic-messages"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/anthropic"

    def test_openai_provider_uses_openai_responses(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        provider = overlay["providers"]["databricks-openai"]
        assert provider["api"] == "openai-responses"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/codex/v1"

    def test_gemini_provider_uses_google_generative_ai(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2"])
        provider = overlay["providers"]["databricks-gemini"]
        assert provider["api"] == "google-generative-ai"
        assert provider["baseUrl"] == f"{WS}/ai-gateway/gemini/v1beta"

    def test_all_three_providers_when_all_present(self):
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        assert set(overlay["providers"].keys()) == {
            "databricks-claude",
            "databricks-openai",
            "databricks-gemini",
        }


class TestRenderOverlayUserAgent:
    def test_user_agent_set_on_all_three_providers(self, monkeypatch):
        monkeypatch.setattr(pi, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(pi, "agent_version", lambda binary: "0.74.0")
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        expected = "ucode/0.1.0 pi/0.74.0"
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert overlay["providers"][name]["headers"]["User-Agent"] == expected


class TestRenderOverlayCompatFlags:
    def test_claude_disables_eager_tool_input_streaming(self):
        # Gateway's Anthropic translator rejects per-tool
        # `eager_input_streaming`; this flag makes pi send the legacy beta
        # header instead.
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        compat = overlay["providers"]["databricks-claude"]["compat"]
        assert compat["supportsEagerToolInputStreaming"] is False

    def test_openai_and_gemini_have_no_compat_flags(self):
        # Their gateway routes accept pi's request shape as-is.
        overlay, _ = _overlay(
            "gpt-5",
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        assert "compat" not in overlay["providers"]["databricks-openai"]
        assert "compat" not in overlay["providers"]["databricks-gemini"]


class TestRenderOverlayAuthAndModels:
    def test_api_key_config_value_embedded_verbatim(self):
        # render_overlay takes a pi config value, so it must not reinterpret it.
        overlay, _ = _overlay(
            "claude-sonnet", token="!mint.sh", claude_models={"sonnet": "claude-sonnet"}
        )
        assert overlay["providers"]["databricks-claude"]["apiKey"] == "!mint.sh"

    def test_auth_header_flag_set_on_all_providers(self):
        overlay, _ = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert overlay["providers"][name]["authHeader"] is True

    def test_claude_models_listed(self):
        claude_models = {"opus": "claude-opus", "sonnet": "claude-sonnet"}
        overlay, _ = _overlay("claude-sonnet", claude_models=claude_models)
        ids = {m["id"] for m in overlay["providers"]["databricks-claude"]["models"]}
        assert ids == {"claude-opus", "claude-sonnet"}

    def test_openai_models_listed(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5", "gpt-5-mini"])
        ids = {m["id"] for m in overlay["providers"]["databricks-openai"]["models"]}
        assert ids == {"gpt-5", "gpt-5-mini"}

    def test_gemini_models_listed(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2", "gemini-2-pro"])
        ids = {m["id"] for m in overlay["providers"]["databricks-gemini"]["models"]}
        assert ids == {"gemini-2", "gemini-2-pro"}


class TestRenderOverlayManagedKeys:
    def test_managed_keys_include_model(self):
        _, keys = _overlay("foo")
        assert ["model"] in keys

    def test_managed_keys_include_each_provider_present(self):
        _, keys = _overlay(
            "claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
            codex_models=["gpt-5"],
            gemini_models=["gemini-2"],
        )
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert ["providers", name] in keys


class TestRenderOverlayModelSelector:
    def test_prefixes_claude_model(self):
        overlay, _ = _overlay("claude-sonnet", claude_models={"sonnet": "claude-sonnet"})
        assert overlay["model"] == "databricks-claude/claude-sonnet"

    def test_prefixes_openai_model(self):
        overlay, _ = _overlay("gpt-5", codex_models=["gpt-5"])
        assert overlay["model"] == "databricks-openai/gpt-5"

    def test_prefixes_gemini_model(self):
        overlay, _ = _overlay("gemini-2", gemini_models=["gemini-2"])
        assert overlay["model"] == "databricks-gemini/gemini-2"

    def test_preserves_already_prefixed_model(self):
        overlay, _ = _overlay(
            "databricks-claude/claude-sonnet",
            claude_models={"sonnet": "claude-sonnet"},
        )
        assert overlay["model"] == "databricks-claude/claude-sonnet"

    def test_unknown_model_passes_through_unprefixed(self):
        # Lets a user override `model` to whatever pi accepts even if we
        # didn't classify it.
        overlay, _ = _overlay("custom/whatever")
        assert overlay["model"] == "custom/whatever"


class TestPiDefaultModel:
    def test_prefers_claude_opus(self):
        state = {"claude_models": {"opus": "o4", "sonnet": "s4", "haiku": "h4"}}
        assert pi.default_model(state) == "o4"

    def test_falls_back_to_sonnet(self):
        state = {"claude_models": {"sonnet": "s4"}}
        assert pi.default_model(state) == "s4"

    def test_falls_back_to_haiku(self):
        state = {"claude_models": {"haiku": "h4"}}
        assert pi.default_model(state) == "h4"

    def test_falls_back_to_codex(self):
        state = {"claude_models": {}, "codex_models": ["gpt-5"]}
        assert pi.default_model(state) == "gpt-5"

    def test_falls_back_to_gemini(self):
        state = {"claude_models": {}, "codex_models": [], "gemini_models": ["gemini-2"]}
        assert pi.default_model(state) == "gemini-2"

    def test_returns_none_when_empty(self):
        assert pi.default_model({}) is None
        assert (
            pi.default_model({"claude_models": {}, "codex_models": [], "gemini_models": []}) is None
        )


class TestBuildRuntimeEnv:
    def test_sets_oauth_token(self):
        env = pi.build_runtime_env("tok")
        assert env["OAUTH_TOKEN"] == "tok"

    def test_sets_private_agent_dir_without_replacing_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/real-user-home")

        env = pi.build_runtime_env("tok")

        assert env["PI_CODING_AGENT_DIR"] == str(pi.PI_CONFIG_DIR)
        assert env["HOME"] == "/real-user-home"


class TestPiValidateCmd:
    def test_starts_with_binary(self):
        cmd = pi.validate_cmd("pi")
        assert cmd[0] == "pi"

    def test_uses_print_flag(self):
        # `--print` puts pi in non-interactive mode; without it the TUI hangs on stdin.
        cmd = pi.validate_cmd("pi")
        assert "--print" in cmd

    def test_has_prompt(self):
        cmd = pi.validate_cmd("pi")
        assert len(cmd) > 2


class TestWriteToolConfig:
    def _setup(self, tmp_path, monkeypatch):
        import ucode.agents.pi as pi_mod
        import ucode.config_io as config_io_mod

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_file = tmp_path / "models.json"
        backup_file = tmp_path / "pi-backup.json"
        settings_file = tmp_path / "settings.json"
        settings_backup_file = tmp_path / "pi-settings-backup.json"
        monkeypatch.setattr(pi_mod, "PI_CONFIG_PATH", config_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(pi_mod, "PI_BACKUP_PATH", backup_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", settings_backup_file)
        return pi_mod, config_file, settings_file, settings_backup_file

    def _state(self, **overrides) -> dict:
        state = {
            "workspace": WS,
            "base_urls": {"pi": _base_urls()},
            "claude_models": {"sonnet": "claude-sonnet"},
            "codex_models": [],
            "gemini_models": [],
            "managed_configs": {},
        }
        state.update(overrides)
        return state

    def test_stale_managed_providers_removed_before_merge(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        stale = {
            "providers": {
                "databricks-claude": {"old": True},
                "databricks-openai": {"old": True},
                "databricks-gemini": {"old": True},
                "user-provider": {"keep": True},
            }
        }
        config_file.write_text(json.dumps(stale), encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        providers = written.get("providers", {})
        assert providers.get("databricks-claude") != {"old": True}
        assert "old" not in providers.get("databricks-claude", {})
        assert providers.get("user-provider") == {"keep": True}

    def test_legacy_providers_removed_on_upgrade(self, tmp_path, monkeypatch):
        """Earlier ucode versions wrote `databricks-anthropic`, `databricks-codex`,
        and `databricks-oss` providers. They must be stripped on the next write
        so users don't end up with stale entries pointing at routes that 400."""
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        config_file.write_text(
            json.dumps(
                {
                    "providers": {
                        "databricks-anthropic": {"api": "anthropic-messages"},
                        "databricks-codex": {"api": "openai-responses"},
                        "databricks-oss": {"api": "openai-completions"},
                    }
                }
            ),
            encoding="utf-8",
        )

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written_providers = json.loads(config_file.read_text()).get("providers", {})
        for legacy in ("databricks-anthropic", "databricks-codex", "databricks-oss"):
            assert legacy not in written_providers
        assert "databricks-claude" in written_providers

    def test_config_written_with_correct_model_and_auth_command(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        written = json.loads(config_file.read_text())
        assert written["model"] == "databricks-claude/claude-sonnet"
        api_key = written["providers"]["databricks-claude"]["apiKey"]
        assert api_key.startswith("!")
        assert "auth-token" in api_key

    def test_bearer_never_written_to_the_config(self, tmp_path, monkeypatch):
        # The whole point: a token on disk is a token that can go stale, and pi
        # resolves the command per request instead.
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="dapi-secret"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet")

        assert "dapi-secret" not in config_file.read_text()

    def test_every_provider_gets_the_auth_command(self, tmp_path, monkeypatch):
        pi_mod, config_file, _, _ = self._setup(tmp_path, monkeypatch)
        state = self._state(codex_models=["gpt-5"], gemini_models=["gemini-2"])

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(state, "claude-sonnet", token="tok")

        providers = json.loads(config_file.read_text())["providers"]
        for name in ("databricks-claude", "databricks-openai", "databricks-gemini"):
            assert providers[name]["apiKey"].startswith("!"), name

    def test_settings_pins_default_provider_and_model(self, tmp_path, monkeypatch):
        # Without this, Pi's `findInitialModel` can fall through to a built-in
        # provider when an unrelated env var (e.g. HF_TOKEN) makes one look
        # auth-configured. Pinning the default keeps Pi on our provider.
        pi_mod, _, settings_file, _ = self._setup(tmp_path, monkeypatch)

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        settings = json.loads(settings_file.read_text())
        assert settings["defaultProvider"] == "databricks-claude"
        assert settings["defaultModel"] == "claude-sonnet"

    def test_pre_existing_settings_are_backed_up_before_first_write(self, tmp_path, monkeypatch):
        pi_mod, _, settings_file, settings_backup_file = self._setup(tmp_path, monkeypatch)

        original = '{"theme": "Default Dark", "defaultProvider": "openai"}'
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(original, encoding="utf-8")

        with (
            patch("ucode.agents.pi.get_databricks_token", return_value="tok"),
            patch("ucode.agents.pi.save_state"),
        ):
            pi_mod.write_tool_config(self._state(), "claude-sonnet", token="tok")

        assert settings_backup_file.read_text(encoding="utf-8") == original
        # The on-disk settings still get the ucode pin applied via deep_merge.
        merged = json.loads(settings_file.read_text())
        assert merged["defaultProvider"] == "databricks-claude"
        assert merged["theme"] == "Default Dark"


class TestValidateAllToolsPiRollback:
    def test_failed_pi_validation_rolls_back_settings(self, tmp_path, monkeypatch):
        import ucode.agents as agents_mod
        import ucode.agents.pi as pi_mod

        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(pi_mod, "PI_SETTINGS_BACKUP_PATH", tmp_path / "settings.backup.json")
        # Keep the generic models.json rollback off the user's real config dir.
        monkeypatch.setitem(agents_mod.TOOL_SPECS["pi"], "config_path", tmp_path / "models.json")
        monkeypatch.setitem(
            agents_mod.TOOL_SPECS["pi"], "backup_path", tmp_path / "models.backup.json"
        )
        monkeypatch.setattr(agents_mod, "validate_tool", lambda tool: (False, "boom"))
        monkeypatch.setattr(agents_mod, "save_state", lambda s: None)
        monkeypatch.setattr(agents_mod, "spinner", lambda *_a, **_kw: nullcontext())

        agents_mod.validate_all_tools({"available_tools": ["pi"], "managed_configs": {"pi": True}})

        assert not settings_file.exists()


class TestManagedModels:
    """A managed config's models arrive as `pi_models` and must not come from the shared keys."""

    def test_managed_models_win_over_the_shared_discovery_lists(self):
        state = {
            "pi_models": ["system.ai.claude-opus-4-8"],
            "claude_models": {"opus": "shared-should-not-win"},
        }
        assert pi.default_model(state) == "system.ai.claude-opus-4-8"

    def test_falls_back_to_the_shared_lists_without_a_managed_config(self):
        assert pi.default_model({"claude_models": {"opus": "discovered"}}) == "discovered"

    def test_managed_models_split_into_pis_per_provider_inputs(self):
        # Pi builds one provider block per family, so a flat list has to be classified back out.
        state = {
            "pi_models": [
                "system.ai.claude-opus-4-8",
                "system.ai.gpt-5",
                "system.ai.gemini-3-flash",
            ]
        }
        assert pi._managed_model_families(state) == (
            {"opus": "system.ai.claude-opus-4-8"},
            ["system.ai.gpt-5"],
            ["system.ai.gemini-3-flash"],
        )

    def test_no_split_without_managed_models(self):
        assert pi._managed_model_families({"claude_models": {"opus": "x"}}) is None

    def test_none_when_no_managed_model_is_servable(self):
        # Pi has no OSS provider, so an oss-only list yields no families. Returning an all-empty
        # tuple would be truthy and suppress the fallback, writing a config with zero providers.
        assert pi._managed_model_families({"pi_models": ["system.ai.kimi-k2-7-code"]}) is None

    def test_partially_servable_list_still_splits(self):
        families = pi._managed_model_families(
            {"pi_models": ["system.ai.kimi-k2-7-code", "system.ai.claude-opus-4-8"]}
        )
        assert families == ({"opus": "system.ai.claude-opus-4-8"}, [], [])


class TestManagedDefaultModel:
    """A managed config's `pi_default_model` takes priority over the allowlist."""

    def test_pi_default_model_wins_over_allowlist(self):
        state = {
            "pi_default_model": "admin-chosen-default",
            "pi_models": ["system.ai.claude-opus-4-8", "system.ai.gpt-5"],
        }
        assert pi.default_model(state) == "admin-chosen-default"

    def test_falls_back_to_pi_models_without_default(self):
        state = {"pi_models": ["system.ai.claude-opus-4-8"]}
        assert pi.default_model(state) == "system.ai.claude-opus-4-8"


class TestBuildPiApiKey:
    """Pi resolves a leading-`!` config value as a command before every provider
    request, so the apiKey is that command rather than a baked bearer."""

    def test_is_a_pi_command_value_running_auth_token(self):
        api_key = pi.build_pi_api_key({"workspace": WS})

        assert api_key.startswith("!")
        assert "auth-token" in api_key
        assert f"--host {WS}" in api_key

    def test_omits_force_refresh(self):
        # Pi has no token cache on this path, so --force-refresh would round-trip
        # to the workspace on every turn.
        assert "--force-refresh" not in pi.build_pi_api_key({"workspace": WS})

    def test_passes_the_profile_through(self):
        api_key = pi.build_pi_api_key({"workspace": WS, "profile": "stablebox"})

        assert "--profile stablebox" in api_key

    def test_forwards_use_pat(self):
        assert "--use-pat" in pi.build_pi_api_key({"workspace": WS, "use_pat": True})
        assert "--use-pat" not in pi.build_pi_api_key({"workspace": WS})
