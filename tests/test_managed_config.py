"""Tests for managed_config.py — fetch/normalize/persist of the admin-authored managed config."""

from __future__ import annotations

import json
import os
import stat
import time

import pytest

import ucode.config_io as config_io_mod
import ucode.databricks as db_mod
import ucode.managed_config as mc_mod
from ucode.managed_config import (
    get_managed_config,
    load_managed_state,
    managed_state_workspace,
    normalize_managed_config,
    refresh_managed_config,
    save_managed_state,
)
from ucode.managed_setup import serialize_managed_config

# A representative raw CodingAgentConfig proto-JSON manifest (mirrors what the API returns).
RAW_MANIFEST = {
    "name": "coding-agent-configs/abc-123",
    "workspace_id": 1653573648247579,
    "default_agent": "CODING_AGENT_CLAUDE_CODE",
    "enabled_agents": [
        {
            "agent": "CODING_AGENT_CLAUDE_CODE",
            "config": {
                "custom_headers": {"x-databricks-workspace": "eng-ml-inference"},
                "tracing_config": {"table": "main.default.ucode_traces"},
                "model_config": {
                    "claude": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {
                            "default_opus_model": "system.ai.claude-opus-4-8",
                            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                            "default_haiku_model": "system.ai.claude-haiku-4-5",
                        },
                    }
                },
            },
        },
        {
            "agent": "CODING_AGENT_OPENCODE",
            "config": {
                "model_config": {
                    "opencode": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": ["system.ai.claude-opus-4-8", "system.ai.kimi-k2-7-code"],
                    }
                }
            },
        },
    ],
    "mcp_servers": [
        {"name": "system.ai.github", "type": "MCP_SERVER_TYPE_UC_SERVICE"},
        {"name": "some-space-id", "type": "MCP_SERVER_TYPE_GENIE"},
    ],
    "skills": {"names": ["system.ai.pdf-extraction"]},
    "tracing": {"table": "main.default.ucode_traces"},
    "budget_policy": {
        "display_name": "paved-path",
        "budget_id": "c6563b45-df9a-4b19-afb2-d42dc2b52576",
        "tiers": [
            {
                "spending_percentage": 0.8,
                "default_agent": "CODING_AGENT_CLAUDE_CODE",
                "default_model": "system.ai.claude-sonnet-4-6",
            },
            {
                "spending_percentage": 1.0,
                "default_agent": "CODING_AGENT_OPENCODE",
                "default_model": "system.ai.kimi-k2-7-code",
            },
        ],
    },
}


class TestNormalize:
    def test_full_manifest_maps_enums_to_tool_names(self):
        cfg = normalize_managed_config(RAW_MANIFEST)
        assert cfg["name"] == "coding-agent-configs/abc-123"
        assert cfg["default_agent"] == "claude"
        assert set(cfg["enabled_agents"]) == {"claude", "opencode"}

    def test_claude_agent_config_fields(self):
        claude = normalize_managed_config(RAW_MANIFEST)["enabled_agents"]["claude"]
        assert claude["custom_headers"] == {"x-databricks-workspace": "eng-ml-inference"}
        assert claude["tracing_table"] == "main.default.ucode_traces"
        assert claude["model_config"]["default_model"] == "system.ai.claude-opus-4-8"
        assert claude["model_config"]["models"]["default_opus_model"] == "system.ai.claude-opus-4-8"

    def test_opencode_model_list_is_flat(self):
        opencode = normalize_managed_config(RAW_MANIFEST)["enabled_agents"]["opencode"]
        assert opencode["model_config"]["models"] == [
            "system.ai.claude-opus-4-8",
            "system.ai.kimi-k2-7-code",
        ]

    def test_mcp_servers_map_type_enums_to_tags(self):
        mcp = normalize_managed_config(RAW_MANIFEST)["mcp_servers"]
        assert mcp == [
            {"name": "system.ai.github", "type": "mcp-service"},
            {"name": "some-space-id", "type": "genie-space"},
        ]

    def test_skills_and_tracing_and_budget(self):
        cfg = normalize_managed_config(RAW_MANIFEST)
        assert cfg["skills"] == {"names": ["system.ai.pdf-extraction"]}
        assert cfg["tracing_table"] == "main.default.ucode_traces"
        assert cfg["budget_policy"]["budget_id"] == "c6563b45-df9a-4b19-afb2-d42dc2b52576"
        assert cfg["budget_policy"]["tiers"][1]["default_agent"] == "opencode"

    def test_reads_top_level_display_name(self):
        cfg = normalize_managed_config({**RAW_MANIFEST, "display_name": "paved-path"})
        assert cfg["display_name"] == "paved-path"

    def test_display_name_survives_the_serialize_round_trip(self):
        manifest = normalize_managed_config({**RAW_MANIFEST_V2, "display_name": "paved-path"})
        assert serialize_managed_config(manifest)["display_name"] == "paved-path"
        assert normalize_managed_config(serialize_managed_config(manifest)) == manifest

    @pytest.mark.parametrize("agent_enum", ["CODING_AGENT_FUTURE", "CODING_AGENT_UNSPECIFIED"])
    def test_unrecognized_agent_enum_dropped(self, agent_enum):
        raw = {"enabled_agents": [{"agent": agent_enum, "config": {}}]}
        assert "enabled_agents" not in normalize_managed_config(raw)

    def test_unknown_mcp_type_dropped(self):
        raw = {"mcp_servers": [{"name": "x", "type": "MCP_SERVER_TYPE_UNSPECIFIED"}]}
        assert "mcp_servers" not in normalize_managed_config(raw)

    def test_empty_manifest_yields_empty_dict(self):
        assert normalize_managed_config({}) == {}


# A v2 CodingAgentConfig, as the FINAL ai-gateway-api shape emits it.
# The wire format uses: default_models map (not separate default_model + default_alias_models),
# new model field names (model_services, unity_catalog_location), new tier field names
# (recommended_agent, recommended_model), and mcp_servers/skills as parallel arrays.
RAW_MANIFEST_V2 = {
    "spec_version": 1,
    "retrieved_time": "2026-09-09T22:00:00Z",
    "default_agent": "CODING_AGENT_CLAUDE_CODE",
    "enabled_agents": [
        {
            "agent": "CODING_AGENT_CLAUDE_CODE",
            "config": {
                "models": {
                    "model_services": [
                        "system.ai.claude-opus-4-8",
                        "system.ai.claude-sonnet-4-6",
                        "system.ai.claude-haiku-4-5",
                    ],
                },
                "default_models": {
                    "default_model": "system.ai.claude-opus-4-8",
                    "default_opus_model": "system.ai.claude-opus-4-8",
                    "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    "default_haiku_model": "system.ai.claude-haiku-4-5",
                },
                "smart_routing": {"enabled": True},
                "http_headers": {"x-databricks-workspace": "eng-ml-inference"},
            },
        },
        {
            "agent": "CODING_AGENT_CODEX",
            "config": {
                "models": {"model_provider_service": "main.default.openai-mps"},
                "default_models": {"default_model": "gpt-5.4"},
            },
        },
    ],
    "mcp_servers": {
        "names": ["system.ai.github", "some-space-id"],
        "tags": ["MCP_SERVER_TYPE_UC_SERVICE", "MCP_SERVER_TYPE_GENIE"],
    },
    "skills": {"names": ["system.ai.pdf-extraction"], "tags": []},
    "tracing": {"enabled": True},
    "spend_tiers": {
        "budget_id": "c6563b45-df9a-4b19-afb2-d42dc2b52576",
        "tiers": [
            {
                "spending_percentage": 0.9,
                "recommended_agent": "CODING_AGENT_CODEX",
                "recommended_model": "gpt-5.4",
            },
        ],
    },
}


class TestNormalizeV2:
    def test_maps_agents_to_tool_names(self):
        cfg = normalize_managed_config(RAW_MANIFEST_V2)
        assert cfg["default_agent"] == "claude"
        assert set(cfg["enabled_agents"]) == {"claude", "codex"}

    def test_claude_alias_models_map_to_family_slots(self):
        claude = normalize_managed_config(RAW_MANIFEST_V2)["enabled_agents"]["claude"]
        assert claude["model_config"]["models"] == {
            "default_opus_model": "system.ai.claude-opus-4-8",
            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
            "default_haiku_model": "system.ai.claude-haiku-4-5",
        }
        assert claude["model_config"]["default_model"] == "system.ai.claude-opus-4-8"

    def test_http_headers_map_to_custom_headers(self):
        claude = normalize_managed_config(RAW_MANIFEST_V2)["enabled_agents"]["claude"]
        assert claude["custom_headers"] == {"x-databricks-workspace": "eng-ml-inference"}

    def test_static_names_are_carried(self):
        claude = normalize_managed_config(RAW_MANIFEST_V2)["enabled_agents"]["claude"]
        assert claude["model_config"]["names"] == [
            "system.ai.claude-opus-4-8",
            "system.ai.claude-sonnet-4-6",
            "system.ai.claude-haiku-4-5",
        ]

    def test_unity_catalog_location_is_carried(self):
        raw = {
            "spec_version": 1,
            "enabled_agents": [
                {
                    "agent": "CODING_AGENT_CLAUDE_CODE",
                    "config": {"models": {"unity_catalog_location": "main.agents"}},
                }
            ],
        }
        claude = normalize_managed_config(raw)["enabled_agents"]["claude"]
        assert claude["model_config"]["model_service_location"] == "main.agents"

    def test_codex_provider_service(self):
        codex = normalize_managed_config(RAW_MANIFEST_V2)["enabled_agents"]["codex"]
        assert codex["model_config"]["model_provider_service"] == "main.default.openai-mps"
        assert codex["model_config"]["default_model"] == "gpt-5.4"

    def test_budget_policy_carries_budget_id_and_tiers(self):
        cfg = normalize_managed_config(RAW_MANIFEST_V2)
        assert cfg["budget_policy"]["budget_id"] == "c6563b45-df9a-4b19-afb2-d42dc2b52576"
        assert cfg["budget_policy"]["tiers"][0]["default_agent"] == "codex"

    def test_v2_tracing_enabled_carries_no_table(self):
        # v2 `tracing.enabled` has no table FQN, so nothing lands in the (table-shaped) internal key.
        assert "tracing_table" not in normalize_managed_config(RAW_MANIFEST_V2)

    def test_spec_version_not_carried_into_internal_manifest(self):
        # Kept out so the serialize/normalize round trip (which never sees spec_version) is unaffected.
        assert "spec_version" not in normalize_managed_config(RAW_MANIFEST_V2)

    def test_legacy_agent_config_fields_still_read(self):
        # A pre-v2 entry with only the deprecated custom_headers + model_config oneof still normalizes.
        raw = {
            "enabled_agents": [
                {
                    "agent": "CODING_AGENT_CLAUDE_CODE",
                    "config": {
                        "custom_headers": {"x-h": "v"},
                        "model_config": {"claude": {"default_model": "system.ai.claude-opus-4-8"}},
                    },
                }
            ]
        }
        claude = normalize_managed_config(raw)["enabled_agents"]["claude"]
        assert claude["custom_headers"] == {"x-h": "v"}
        assert claude["model_config"]["default_model"] == "system.ai.claude-opus-4-8"


class TestGetManagedConfig:
    def test_returns_normalized_first_config(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([RAW_MANIFEST], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert reason is None
        assert cfg["default_agent"] == "claude"

    def test_no_config_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason is None

    def test_fetch_failure_surfaces_reason(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod,
            "fetch_managed_coding_agent_configs",
            lambda ws, tok: ([], "HTTP 500 Server Error"),
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason == "HTTP 500 Server Error"

    @pytest.mark.parametrize(
        "not_found_reason",
        [
            "HTTP 404 Not Found",
            'HTTP 404 Not Found: {"error_code":"NOT_FOUND","message":"..."}',
            'HTTP 400 Bad Request: {"error_code":"NOT_FOUND"}',
        ],
    )
    def test_not_found_is_treated_as_no_config(self, monkeypatch, not_found_reason):
        # A NOT_FOUND from the read means the admin hasn't defined a config — the normal
        # no-config case, not an error, so it collapses to (None, None).
        monkeypatch.setattr(
            mc_mod,
            "fetch_managed_coding_agent_configs",
            lambda ws, tok: ([], not_found_reason),
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason is None

    def test_feature_disabled_is_not_swallowed_as_not_found(self, monkeypatch):
        reason = (
            'HTTP 404 Not Found: {"error_code":"FEATURE_DISABLED",'
            '"message":"Coding agent config APIs are not enabled for this workspace."}'
        )
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([], reason)
        )
        cfg, reason_out = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason_out == reason

    def test_normalizes_v2_wire_shape(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([RAW_MANIFEST_V2], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert reason is None
        assert cfg["default_agent"] == "claude"
        assert set(cfg["enabled_agents"]) == {"claude", "codex"}

    def test_spec_version_newer_than_supported_is_refused(self, monkeypatch):
        raw = {**RAW_MANIFEST_V2, "spec_version": mc_mod.MAX_SPEC_VERSION + 1}
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([raw], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        # Refused (reason set) rather than misread, so the launch path keeps the last-known-good
        # cache instead of applying a config it can't parse.
        assert cfg is None
        assert reason is not None and "spec_version" in reason

    @pytest.mark.parametrize("bad_spec", ["2", 2.0, True])
    def test_malformed_spec_version_is_refused(self, monkeypatch, bad_spec):
        raw = {**RAW_MANIFEST_V2, "spec_version": bad_spec}
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([raw], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason is not None and "spec_version" in reason


class TestManagedConfigStub:
    def test_stub_short_circuits_the_http_read(self, tmp_path, monkeypatch):
        stub = tmp_path / "v2.json"
        stub.write_text(json.dumps(RAW_MANIFEST_V2), encoding="utf-8")
        monkeypatch.setenv("UCODE_MANAGED_CONFIG_STUB", str(stub))

        def _fail(ws, tok):
            raise AssertionError("stub set: the HTTP read must not run")

        monkeypatch.setattr(mc_mod, "fetch_managed_coding_agent_configs", _fail)
        cfg, reason = get_managed_config("https://ws", "tok")
        assert reason is None
        assert cfg["default_agent"] == "claude"

    def test_stub_applies_the_spec_version_gate(self, tmp_path, monkeypatch):
        stub = tmp_path / "v2.json"
        stub.write_text(
            json.dumps({**RAW_MANIFEST_V2, "spec_version": mc_mod.MAX_SPEC_VERSION + 1}),
            encoding="utf-8",
        )
        monkeypatch.setenv("UCODE_MANAGED_CONFIG_STUB", str(stub))
        cfg, reason = get_managed_config("https://ws", "tok")
        assert cfg is None
        assert reason is not None and "spec_version" in reason

    def test_unreadable_stub_falls_through_to_the_http_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UCODE_MANAGED_CONFIG_STUB", str(tmp_path / "missing.json"))
        monkeypatch.setattr(
            mc_mod, "fetch_managed_coding_agent_configs", lambda ws, tok: ([RAW_MANIFEST_V2], None)
        )
        cfg, reason = get_managed_config("https://ws", "tok")
        assert reason is None
        assert cfg["default_agent"] == "claude"


class TestPersistence:
    @pytest.fixture(autouse=True)
    def _managed_path(self, tmp_path, monkeypatch):
        path = tmp_path / ".ucode" / "managed-state.json"
        monkeypatch.setattr(mc_mod, "MANAGED_STATE_PATH", path)
        return path

    def test_save_then_load_round_trips(self, _managed_path):
        cfg = normalize_managed_config(RAW_MANIFEST)
        save_managed_state("https://ws.example.com", cfg)
        loaded = load_managed_state("https://ws.example.com")
        assert loaded == cfg

    def test_saved_file_is_0600(self, _managed_path):
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        mode = stat.S_IMODE(os.stat(_managed_path).st_mode)
        # Owner-only read/write; no group/other bits.
        assert mode == 0o600

    def test_load_ignores_other_workspace(self, _managed_path):
        save_managed_state("https://ws-a.example.com", {"default_agent": "claude"})
        assert load_managed_state("https://ws-b.example.com") is None

    def test_load_missing_returns_none(self, _managed_path):
        assert load_managed_state("https://ws.example.com") is None

    def test_load_none_workspace_returns_none(self, _managed_path):
        assert load_managed_state(None) is None

    def test_empty_config_overwrites_a_previous_one(self, _managed_path):
        # Saving an empty config is how "the admin removed it" is recorded: the stored config must
        # be replaced, not left behind for the read-failure fallback to reapply.
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        save_managed_state("https://ws.example.com", {})
        assert load_managed_state("https://ws.example.com") == {}

    def test_workspace_is_stored_alongside_the_config(self, _managed_path):
        # `ucode setup --show` reads this when local state carries no workspace yet, so the authored
        # file can still be found and attributed on disk.
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        assert managed_state_workspace() == "https://ws.example.com"

    def test_workspace_is_none_when_absent(self, _managed_path):
        assert managed_state_workspace() is None

    def test_dry_run_writes_nothing(self, _managed_path, monkeypatch):
        # Under --dry-run the config writers print instead of touching disk, so a launch that
        # dry-runs an admin's authored draft never overwrites it.
        monkeypatch.setattr(config_io_mod, "is_dry_run", lambda: True)
        save_managed_state("https://ws.example.com", {"default_agent": "claude"})
        assert not _managed_path.exists()

    def test_corrupt_file_reads_as_absent(self, _managed_path):
        # A truncated/hand-mangled file must not crash a launch: it reads as "no config" so the
        # launch falls through rather than raising on JSON it can't parse.
        _managed_path.parent.mkdir(parents=True, exist_ok=True)
        _managed_path.write_text("{not json", encoding="utf-8")
        assert load_managed_state("https://ws.example.com") is None
        assert managed_state_workspace() is None

    def test_loaded_config_serializes_to_a_json_encodable_payload(self, _managed_path):
        # `ucode publish` POSTs the serialized config, so a manifest that survives a disk round-trip
        # must still serialize to something json.dumps accepts with no custom encoder.
        cfg = normalize_managed_config(RAW_MANIFEST)
        save_managed_state("https://ws.example.com", cfg)
        loaded = load_managed_state("https://ws.example.com")
        assert loaded is not None
        assert json.loads(json.dumps(serialize_managed_config(loaded)))


class TestFetchClient:
    """fetch_managed_coding_agent_configs lives in databricks.py; test its response parsing."""

    def test_extracts_configs_list(self, monkeypatch):
        payload = {"coding_agent_configs": [RAW_MANIFEST]}
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: (payload, None),
        )
        configs, reason = db_mod.fetch_managed_coding_agent_configs("https://ws", "tok")
        assert reason is None
        assert len(configs) == 1
        assert configs[0]["default_agent"] == "CODING_AGENT_CLAUDE_CODE"

    def test_empty_list_when_no_configs(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: ({}, None),
        )
        configs, reason = db_mod.fetch_managed_coding_agent_configs("https://ws", "tok")
        assert configs == []
        assert reason is None

    def test_http_failure_surfaces_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=10: (None, "HTTP 403 Forbidden"),
        )
        configs, reason = db_mod.fetch_managed_coding_agent_configs("https://ws", "tok")
        assert configs == []
        assert reason == "HTTP 403 Forbidden"


WORKSPACE = "https://ws.example.com"

# A normalized managed config, as `normalize_managed_config` produces it.
MANAGED = {
    "default_agent": "claude",
    "enabled_agents": {
        "claude": {
            "model_config": {
                "default_model": "system.ai.claude-opus-5",
                "models": {"default_opus_model": "system.ai.claude-opus-5"},
            }
        }
    },
}


def _state(**overrides) -> dict:
    state = {"workspace": WORKSPACE, "managed_configs": {"claude": {"keys": []}}}
    state.update(overrides)
    return state


class TestRefreshManagedConfig:
    """The per-launch re-read, so an admin's edits land without re-running `ucode configure`."""

    @pytest.fixture(autouse=True)
    def _stub_token(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_databricks_token", lambda ws, profile: "tok")

    def test_persists_and_returns_the_manifest(self, monkeypatch):
        saved: list[tuple] = []
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (MANAGED, None))
        monkeypatch.setattr(
            mc_mod, "save_managed_state", lambda ws, cfg, **kwargs: saved.append((ws, cfg))
        )
        assert refresh_managed_config(_state()) == (MANAGED, False)
        assert saved == [(WORKSPACE, MANAGED)]

    def test_no_managed_config_returns_none(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg, **kwargs: None)
        result, _ = refresh_managed_config(_state())
        assert result is None

    def test_read_failure_falls_back_to_the_persisted_config(self, monkeypatch):
        # The admin's last known policy beats no policy, so a failed fetch reuses what we saved.
        warnings: list[str] = []
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: warnings.append(msg))
        assert refresh_managed_config(_state()) == (MANAGED, False)
        assert "HTTP 500" in warnings[0]
        assert "last one saved" in warnings[0]

    def test_read_failure_without_persisted_config_is_silent(self, monkeypatch):
        # Nothing persisted means no managed config is in play, so an expired session shouldn't
        # produce a warning about a feature this developer doesn't use.
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        result, _ = refresh_managed_config(_state())
        assert result is None

    def test_auth_failure_falls_back_to_the_persisted_config(self, monkeypatch):
        warnings: list[str] = []

        def boom(ws, profile):
            raise RuntimeError("no token")

        monkeypatch.setattr(mc_mod, "get_databricks_token", boom)
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: warnings.append(msg))
        assert refresh_managed_config(_state()) == (MANAGED, False)
        assert "no token" in warnings[0]

    def test_auth_failure_without_persisted_config_is_silent(self, monkeypatch):
        def boom(ws, profile):
            raise RuntimeError("no token")

        monkeypatch.setattr(mc_mod, "get_databricks_token", boom)
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        result, _ = refresh_managed_config(_state())
        assert result is None

    def test_permission_denied_without_cache_is_silent(self, monkeypatch):
        # A refusal is no evidence a config exists, so with nothing cached there is no managed
        # config in play and warning would be a false positive.
        denied = 'HTTP 403 Forbidden: {"error_code":"PERMISSION_DENIED"}'
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, denied))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        result, _ = refresh_managed_config(_state())
        assert result is None

    def test_permission_denied_warns_and_keeps_the_cached_config(self, monkeypatch):
        # A refused read is worth surfacing: an admin may have published a config that isn't
        # reaching this developer. It says nothing about whether one exists, so the cache stands.
        warnings: list[str] = []
        denied = 'HTTP 403 Forbidden: {"error_code":"PERMISSION_DENIED"}'
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, denied))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: warnings.append(msg))
        monkeypatch.setattr(
            mc_mod,
            "save_managed_state",
            lambda ws, cfg, **kwargs: pytest.fail("must not clear the cache"),
        )
        assert refresh_managed_config(_state()) == (MANAGED, False)
        assert "not readable by you" in warnings[0]

    def test_no_config_on_the_server_does_not_use_a_stale_persisted_file(self, monkeypatch):
        # A successful read saying "no config" means the admin removed it — that's authoritative,
        # so a previously persisted file must not resurrect the old policy.
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg, **kwargs: None)
        monkeypatch.setattr(
            mc_mod, "load_managed_state", lambda ws: pytest.fail("must not fall back")
        )
        result, _ = refresh_managed_config(_state())
        assert result is None

    def test_no_config_on_the_server_clears_the_persisted_one(self, monkeypatch):
        # Without this, removing the config server-side would leave the old one on disk and the next
        # failed read would put a dead policy back into force.
        saved: list[tuple] = []
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(
            mc_mod, "save_managed_state", lambda ws, cfg, **kwargs: saved.append((ws, cfg))
        )
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        result, _ = refresh_managed_config(_state())
        assert result is None
        assert saved == [(WORKSPACE, {})]

    def test_empty_persisted_config_is_not_treated_as_a_fallback(self, monkeypatch):
        # The empty marker means "no admin policy", so a later failed read falls through to the
        # developer's own settings rather than reporting a managed config.
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: {})
        monkeypatch.setattr(
            mc_mod, "print_warning", lambda msg: pytest.fail(f"should not warn: {msg}")
        )
        result, _ = refresh_managed_config(_state())
        assert result is None

    def test_no_workspace_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            mc_mod, "get_managed_config", lambda ws, tok: pytest.fail("should not fetch")
        )
        result, _ = refresh_managed_config({})
        assert result is None

    def test_feature_disabled_sets_flag_when_there_is_no_fallback(self, monkeypatch):
        # The workspace hasn't enabled coding-agent-configs server-side, so `ucode setup` can't
        # publish anything yet. The flag lets callers suppress the setup recommendation.
        reason = 'HTTP 400 Bad Request: {"error_code":"FEATURE_DISABLED"}'
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, reason))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: None)
        state = _state()
        result, flag = refresh_managed_config(state)
        assert result is None
        assert flag is True

    def test_feature_disabled_ignores_a_cached_config_and_sets_the_flag(self, monkeypatch):
        # FEATURE_DISABLED is authoritative, so a config cached while the feature was enabled no
        # longer applies: report "no config, feature off" and clear the persisted copy so a later
        # transient failure can't resurrect the disabled policy via the fallback.
        saved: list[tuple] = []
        reason = 'HTTP 400 Bad Request: {"error_code":"FEATURE_DISABLED"}'
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, reason))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: MANAGED)
        monkeypatch.setattr(
            mc_mod, "save_managed_state", lambda ws, cfg, **kwargs: saved.append((ws, cfg))
        )
        monkeypatch.setattr(
            mc_mod,
            "print_warning",
            lambda msg: pytest.fail("feature-disabled must not warn about falling back to a cache"),
        )
        state = _state()
        result, flag = refresh_managed_config(state)
        assert result is None
        assert flag is True
        assert saved == [(WORKSPACE, {})]

    def test_transient_failure_does_not_set_the_flag(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, "HTTP 500"))
        monkeypatch.setattr(mc_mod, "load_managed_state", lambda ws: None)
        monkeypatch.setattr(mc_mod, "print_warning", lambda msg: None)
        state = _state()
        result, flag = refresh_managed_config(state)
        assert result is None
        assert flag is False

    def test_successful_no_config_clears_the_flag(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_managed_config", lambda ws, tok: (None, None))
        monkeypatch.setattr(mc_mod, "save_managed_state", lambda ws, cfg, **kwargs: None)
        state = _state()
        result, flag = refresh_managed_config(state)
        assert result is None
        assert flag is False


class TestManagedConfigTtl:
    """The launch-time TTL gate: a recently-fetched config is reused without a network round trip.

    These exercise the real persistence (the conftest isolates ``MANAGED_STATE_PATH`` to a tmp file)
    so the on-disk ``retrieved_at`` actually drives the decision; only the network read is stubbed.
    """

    @pytest.fixture(autouse=True)
    def _stub_token(self, monkeypatch):
        monkeypatch.setattr(mc_mod, "get_databricks_token", lambda ws, profile: "tok")

    @staticmethod
    def _counting_fetch(monkeypatch, result=(MANAGED, None)):
        calls = {"n": 0}

        def fetch(ws, tok):
            calls["n"] += 1
            return result

        monkeypatch.setattr(mc_mod, "get_managed_config", fetch)
        return calls

    def test_fresh_cache_skips_the_fetch(self, monkeypatch):
        save_managed_state(WORKSPACE, MANAGED, retrieved_at=time.time())
        calls = self._counting_fetch(monkeypatch)
        result, flag = refresh_managed_config(_state())
        assert result == MANAGED
        assert flag is False
        assert calls["n"] == 0

    def test_stale_cache_refetches(self, monkeypatch):
        save_managed_state(
            WORKSPACE, MANAGED, retrieved_at=time.time() - mc_mod.MANAGED_CONFIG_TTL_SECONDS - 60
        )
        calls = self._counting_fetch(monkeypatch)
        refresh_managed_config(_state())
        assert calls["n"] == 1

    def test_unstamped_local_draft_is_not_treated_as_fresh(self, monkeypatch):
        # A locally-authored draft (ucode setup) is saved without a retrieved_at, so a launch must
        # still read the workspace rather than apply the unpublished draft as if it were fetched.
        save_managed_state(WORKSPACE, MANAGED)  # no retrieved_at -> not a fetched cache entry
        calls = self._counting_fetch(monkeypatch)
        refresh_managed_config(_state())
        assert calls["n"] == 1

    def test_force_bypasses_the_ttl(self, monkeypatch):
        save_managed_state(WORKSPACE, MANAGED, retrieved_at=time.time())  # fresh
        calls = self._counting_fetch(monkeypatch)
        refresh_managed_config(_state(), force=True)
        assert calls["n"] == 1

    def test_empty_cache_always_fetches(self, monkeypatch):
        # A "no config" / feature-disabled marker must be re-checked so those states stay accurate.
        save_managed_state(WORKSPACE, {}, retrieved_at=time.time())  # fresh but empty
        calls = self._counting_fetch(monkeypatch)
        refresh_managed_config(_state())
        assert calls["n"] == 1

    def test_first_launch_with_no_cache_fetches(self, monkeypatch):
        calls = self._counting_fetch(monkeypatch)
        refresh_managed_config(_state())
        assert calls["n"] == 1


class TestGetModelRecommendation:
    """The budget recommendation read. Every response field is optional server-side."""

    @staticmethod
    def _stub(monkeypatch, payload, reason=None):
        monkeypatch.setattr(mc_mod, "fetch_model_recommendation", lambda ws, tok: (payload, reason))

    def test_normalizes_agent_model_and_spend(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "recommended_agent": "CODING_AGENT_OPENCODE",
                "recommended_model": "system.ai.claude-haiku-4-5",
                "current_spend": "412.50",
                "effective_threshold": "500.00",
            },
        )
        rec, reason = mc_mod.get_model_recommendation("https://w", "tok")
        assert reason is None
        assert rec == {
            "agent": "opencode",
            "model": "system.ai.claude-haiku-4-5",
            "current_spend": 412.5,
            "effective_threshold": 500.0,
        }

    def test_model_without_an_agent(self, monkeypatch):
        # A model-only tier with no default_agent recommends a model but no agent.
        self._stub(monkeypatch, {"recommended_model": "system.ai.gpt-5", "current_spend": "1.00"})
        rec, _ = mc_mod.get_model_recommendation("https://w", "tok")
        assert rec is not None and rec["agent"] is None and rec["model"] == "system.ai.gpt-5"

    def test_agent_without_a_model(self, monkeypatch):
        self._stub(monkeypatch, {"recommended_agent": "CODING_AGENT_PI", "current_spend": "1.00"})
        rec, _ = mc_mod.get_model_recommendation("https://w", "tok")
        assert rec is not None and rec["agent"] == "pi" and rec["model"] is None

    @pytest.mark.parametrize("agent_enum", ["CODING_AGENT_UNSPECIFIED", "CODING_AGENT_FUTURE", ""])
    def test_unknown_agent_is_dropped_not_fatal(self, monkeypatch, agent_enum):
        self._stub(
            monkeypatch,
            {"recommended_agent": agent_enum, "recommended_model": "m", "current_spend": "1.00"},
        )
        rec, reason = mc_mod.get_model_recommendation("https://w", "tok")
        assert reason is None
        assert rec is not None and rec["agent"] is None and rec["model"] == "m"

    def test_threshold_alone_still_reports(self, monkeypatch):
        # A budget with no spend yet still has a threshold worth showing.
        self._stub(monkeypatch, {"effective_threshold": "500.00"})
        rec, _ = mc_mod.get_model_recommendation("https://w", "tok")
        assert rec is not None and rec["effective_threshold"] == 500.0

    def test_empty_response_is_no_recommendation(self, monkeypatch):
        self._stub(monkeypatch, {})
        assert mc_mod.get_model_recommendation("https://w", "tok") == (None, None)

    def test_failed_read_surfaces_the_reason(self, monkeypatch):
        self._stub(monkeypatch, {}, reason="HTTP 500")
        assert mc_mod.get_model_recommendation("https://w", "tok") == (None, "HTTP 500")

    def test_unparseable_decimals_become_none(self, monkeypatch):
        self._stub(
            monkeypatch, {"recommended_agent": "CODING_AGENT_PI", "current_spend": "not-a-number"}
        )
        rec, _ = mc_mod.get_model_recommendation("https://w", "tok")
        assert rec is not None and rec["current_spend"] is None
