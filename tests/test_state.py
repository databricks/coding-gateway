"""Tests for state.py — load/save/hydrate/clear/mark_tool_managed."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ucode.state as state_mod
from ucode.state import (
    STATE_VERSION,
    build_agent_state,
    clear_state,
    get_provider_service,
    hydrate_state,
    load_full_state,
    load_state,
    mark_tool_managed,
    save_state,
    set_provider_service,
)

FAKE_WS = "https://example.databricks.com"
FAKE_URLS = {
    "codex": f"{FAKE_WS}/ai-gateway/codex/v1",
    "claude": f"{FAKE_WS}/ai-gateway/anthropic",
    "gemini": f"{FAKE_WS}/ai-gateway/gemini",
    "opencode": {
        "anthropic": f"{FAKE_WS}/ai-gateway/anthropic/v1",
        "gemini": f"{FAKE_WS}/ai-gateway/gemini/v1beta",
    },
    "copilot": f"{FAKE_WS}/ai-gateway/mlflow/v1",
    "pi": {
        "claude": f"{FAKE_WS}/ai-gateway/anthropic",
        "openai": f"{FAKE_WS}/ai-gateway/codex/v1",
        "gemini": f"{FAKE_WS}/ai-gateway/gemini/v1beta",
    },
}


@pytest.fixture(autouse=True)
def patch_state_path(tmp_path, monkeypatch):
    """Redirect STATE_PATH and APP_DIR to a temp directory for every test."""
    fake_state_path = tmp_path / "state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", fake_state_path)

    import ucode.config_io as config_io_mod

    monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)


@pytest.fixture(autouse=True)
def patch_build_urls():
    """Avoid real network calls from hydrate_state."""
    with patch("ucode.state.build_shared_base_urls", return_value=FAKE_URLS):
        yield


# ---------------------------------------------------------------------------
# load_full_state
# ---------------------------------------------------------------------------


class TestLoadFullState:
    def test_returns_empty_structure_when_missing(self):
        result = load_full_state()
        assert result["state_version"] == STATE_VERSION
        assert result["current_workspace"] is None
        assert result["workspaces"] == {}

    def test_returns_empty_when_wrong_version(self, tmp_path):
        state_mod.STATE_PATH.write_text(
            json.dumps({"state_version": 0, "current_workspace": None, "workspaces": {}}),
            encoding="utf-8",
        )
        result = load_full_state()
        assert result["workspaces"] == {}

    def test_returns_empty_on_corrupt_json(self, tmp_path):
        state_mod.STATE_PATH.write_text("not json", encoding="utf-8")
        result = load_full_state()
        assert result["current_workspace"] is None

    def test_loads_valid_state(self, tmp_path):
        data = {
            "state_version": STATE_VERSION,
            "current_workspace": FAKE_WS,
            "workspaces": {FAKE_WS: {"claude_models": {"sonnet": "s4"}}},
        }
        state_mod.STATE_PATH.write_text(json.dumps(data), encoding="utf-8")
        result = load_full_state()
        assert result["current_workspace"] == FAKE_WS


# ---------------------------------------------------------------------------
# save_state / load_state round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trip(self):
        state = {
            "workspace": FAKE_WS,
            "claude_models": {"sonnet": "databricks-claude-sonnet-4"},
        }
        save_state(state)
        loaded = load_state()
        assert loaded["workspace"] == FAKE_WS
        assert loaded["claude_models"]["sonnet"] == "databricks-claude-sonnet-4"

    def test_persists_codex_launcher_default_in_agent_state(self):
        save_state(
            {
                "workspace": FAKE_WS,
                "codex_models": [
                    "system.ai.gpt-5",
                    "system.ai.gpt-5-1",
                    "system.ai.gpt-5-6-luna",
                ],
            }
        )

        persisted = load_full_state()["workspaces"][FAKE_WS]
        assert persisted["codex_models"][0] == "system.ai.gpt-5"
        assert "model" not in persisted["agents"]["codex"]
        assert persisted["agents"]["pi"]["model"] == "system.ai.gpt-5"

    def test_save_respects_dry_run(self):
        import ucode.config_io as config_io_mod

        config_io_mod.set_dry_run(True)
        try:
            save_state({"workspace": FAKE_WS})
            assert not state_mod.STATE_PATH.exists()
        finally:
            config_io_mod.set_dry_run(False)

    def test_load_state_returns_empty_when_no_workspace(self):
        result = load_state()
        assert result == {}


# ---------------------------------------------------------------------------
# clear_state
# ---------------------------------------------------------------------------


class TestClearState:
    def test_clears_current_workspace(self):
        save_state({"workspace": FAKE_WS, "claude_models": {}})
        clear_state()
        full = load_full_state()
        assert full["current_workspace"] is None
        assert FAKE_WS not in full.get("workspaces", {})

    def test_clear_when_no_state_is_noop(self):
        clear_state()  # should not raise


class TestProviderService:
    def test_get_returns_none_when_unset(self):
        assert get_provider_service({}, "claude") is None
        assert get_provider_service({"provider_services": {}}, "claude") is None

    def test_set_and_get_roundtrip(self):
        state = set_provider_service({}, "claude", "main.a.anthropic")
        assert state["provider_services"]["claude"] == "main.a.anthropic"
        assert get_provider_service(state, "claude") == "main.a.anthropic"
        assert get_provider_service(state, "codex") is None

    def test_set_none_clears_entry_and_key(self):
        state = set_provider_service({}, "claude", "main.a.anthropic")
        state = set_provider_service(state, "claude", None)
        assert get_provider_service(state, "claude") is None
        # Drop the empty container entirely rather than leaving {}.
        assert "provider_services" not in state

    def test_clearing_one_tool_keeps_the_other(self):
        state = set_provider_service({}, "claude", "main.a.anthropic")
        state = set_provider_service(state, "codex", "main.a.openai")
        state = set_provider_service(state, "claude", None)
        assert get_provider_service(state, "claude") is None
        assert get_provider_service(state, "codex") == "main.a.openai"


# ---------------------------------------------------------------------------
# hydrate_state
# ---------------------------------------------------------------------------


class TestHydrateState:
    def test_empty_input_returns_empty(self):
        result = hydrate_state({})
        assert result == {"managed_configs": {}, "base_urls": {}, "agents": {}}

    def test_non_dict_returns_empty(self):
        assert hydrate_state(None) == {}  # type: ignore[arg-type]
        assert hydrate_state("string") == {}  # type: ignore[arg-type]

    def test_populates_base_urls_when_workspace_present(self):
        result = hydrate_state({"workspace": FAKE_WS})
        assert result["base_urls"] == FAKE_URLS

    def test_no_base_urls_when_no_workspace(self):
        result = hydrate_state({"claude_models": {}})
        assert result["base_urls"] == {}
        assert result["agents"] == {}

    def test_populates_agent_state_when_workspace_present(self):
        result = hydrate_state(
            {
                "workspace": FAKE_WS,
                "claude_models": {"opus": "claude-opus"},
                "codex_models": ["gpt-5"],
            }
        )

        assert result["agents"]["claude"]["model"] == "claude-opus"
        assert result["agents"]["claude"]["base_url"] == FAKE_URLS["claude"]
        # Cross-platform helper, not the old POSIX `if [ -n ... ]` pipeline (#116).
        assert "auth-token" in result["agents"]["claude"]["auth_command"]
        assert "if [ -n" not in result["agents"]["claude"]["auth_command"]
        assert "model" not in result["agents"]["codex"]
        assert result["agents"]["codex"]["base_url"] == FAKE_URLS["codex"]
        # Codex runs the helper as argv (command + args), never via `sh -c`.
        codex_auth = result["agents"]["codex"]["auth"]
        assert codex_auth["command"] != "sh"
        assert codex_auth["args"][0] == "auth-token"
        assert result["agents"]["pi"]["model"] == "claude-opus"
        assert result["agents"]["pi"]["base_urls"] == FAKE_URLS["pi"]

    def test_normalizes_managed_configs_dict_entry(self):
        state = {"managed_configs": {"claude": {"keys": [["env", "X"]]}}}
        result = hydrate_state(state)
        assert result["managed_configs"]["claude"] == {"keys": [["env", "X"]]}

    def test_normalizes_managed_configs_truthy_entry(self):
        state = {"managed_configs": {"codex": True}}
        result = hydrate_state(state)
        assert result["managed_configs"]["codex"] == {"keys": []}

    def test_drops_falsy_managed_configs(self):
        state = {"managed_configs": {"codex": False, "claude": None}}
        result = hydrate_state(state)
        assert "codex" not in result["managed_configs"]
        assert "claude" not in result["managed_configs"]


class TestBuildAgentState:
    def test_returns_empty_without_workspace(self):
        result = build_agent_state({"base_urls": FAKE_URLS})
        assert result == {}

    def test_use_pat_state_builds_pat_auth_command(self):
        result = build_agent_state(
            {
                "workspace": "https://example.databricks.com",
                "profile": "DEFAULT",
                "use_pat": True,
                "base_urls": FAKE_URLS,
            }
        )
        # --use-pat threads through to the `ucode auth-token --use-pat` helper,
        # which resolves the static PAT internally on every platform.
        for agent in ("claude", "codex", "pi"):
            assert "--use-pat" in result[agent]["auth_command"]
            assert "--profile DEFAULT" in result[agent]["auth_command"]

    def test_custom_oauth_applies_to_claude_and_codex(self):
        result = build_agent_state(
            {
                "workspace": "https://example.databricks.com",
                "base_urls": FAKE_URLS,
                "custom_oauth": {
                    "client_id": "custom-client",
                    "redirect_url": "http://localhost:8020/callback",
                    "scopes": ["offline_access", "model-serving"],
                },
            }
        )

        assert "--client-id custom-client" in result["claude"]["auth_command"]
        assert result["codex"]["auth"]["args"][-1] == "offline_access,model-serving"


# ---------------------------------------------------------------------------
# mark_tool_managed
# ---------------------------------------------------------------------------


class TestMarkToolManaged:
    def test_sets_managed_keys(self):
        state: dict = {}
        result = mark_tool_managed(state, "claude", [["env", "X"], ["apiKeyHelper"]])
        assert result["managed_configs"]["claude"] == {"keys": [["env", "X"], ["apiKeyHelper"]]}

    def test_sets_last_tool(self):
        state: dict = {}
        result = mark_tool_managed(state, "codex", [])
        assert result["last_tool"] == "codex"

    def test_preserves_existing_managed_configs(self):
        state = {"managed_configs": {"gemini": {"keys": [["GEMINI_MODEL"]]}}}
        result = mark_tool_managed(state, "codex", [["profile"]])
        assert "gemini" in result["managed_configs"]
        assert "codex" in result["managed_configs"]

    def test_records_only_keys(self):
        result = mark_tool_managed({}, "codex", [["model"]])
        assert result["managed_configs"]["codex"] == {"keys": [["model"]]}
