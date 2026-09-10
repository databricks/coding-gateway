"""Tests for the admin-write half of the managed coding-agent config.

The most valuable case here is the round-trip: ``serialize_managed_config`` followed by
``managed_config.normalize_managed_config`` must return the manifest it started from. That single
property pins the write side to the read side, so the two cannot drift as the proto grows.
"""

from __future__ import annotations

import pytest

from ucode.managed_config import (
    AGENT_ENUM_TO_TOOL,
    MCP_TYPE_ENUM_TO_TAG,
    normalize_managed_config,
)
from ucode.managed_setup import (
    AGENT_TOOL_TO_ENUM,
    MCP_TAG_TO_TYPE_ENUM,
    claude_family_for_model,
    claude_model_slots,
    model_families_for_agent,
    model_options_for_agent,
    serialize_managed_config,
    supports_provider_service,
    validate_manifest,
)

WORKSPACE = "https://ws.example.com"

# The server requires `budget_policy.budget_id` to parse as a UUID, so fixtures that aren't
# *testing* that rule need a real one.
BUDGET_ID = "11111111-1111-1111-1111-111111111111"

# A workspace state shaped like `configure_shared_state` produces.
STATE = {
    "workspace": WORKSPACE,
    "claude_models": {
        "opus": "system.ai.claude-opus-4-8",
        "sonnet": "system.ai.claude-sonnet-4-6",
        "haiku": "system.ai.claude-haiku-4-5",
    },
    "codex_models": ["system.ai.gpt-5-6"],
    "gemini_models": ["system.ai.gemini-3-flash"],
    "oss_models": ["system.ai.kimi-k2-6"],
}


def _minimal_manifest() -> dict:
    """The smallest manifest that passes validation: one agent, which is the default."""
    return {
        "default_agent": "claude",
        "enabled_agents": {
            "claude": {
                "model_config": {"default_model": "system.ai.claude-opus-4-8"},
            }
        },
    }


def _full_manifest() -> dict:
    """A manifest exercising every field the read side normalizes."""
    return {
        "default_agent": "claude",
        "enabled_agents": {
            "claude": {
                "custom_headers": {"x-databricks-workspace": "eng-ml-inference"},
                "tracing_table": "main.default.claude-traces",
                "model_config": {
                    "default_model": "system.ai.claude-opus-4-8",
                    "models": {
                        "default_opus_model": "system.ai.claude-opus-4-8",
                        "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    },
                },
            },
            "codex": {
                "model_config": {"default_model": "system.ai.gpt-5-6"},
            },
            "opencode": {
                "model_config": {
                    "default_model": "system.ai.claude-opus-4-8",
                    "models": ["system.ai.claude-opus-4-8", "system.ai.kimi-k2-6"],
                },
            },
        },
        "mcp_servers": [
            {"name": "system.ai.github", "type": "mcp-service"},
            {"name": "genie-space-id", "type": "genie-space"},
        ],
        "skills": {"names": ["system.ai.pdf-extraction"]},
        "tracing_table": "main.default.ucode-traces",
        "budget_policy": {
            "display_name": "eng-tiered-routing",
            "budget_id": "c6563b45-df9a-4b19-afb2-d42dc2b52576",
            "tiers": [
                {
                    "spending_percentage": 0.8,
                    "default_agent": "claude",
                    "default_model": "system.ai.claude-sonnet-4-6",
                },
                {
                    "spending_percentage": 1.0,
                    "default_agent": "opencode",
                    "default_model": "system.ai.kimi-k2-6",
                },
            ],
        },
    }


class TestEnumMaps:
    def test_agent_map_is_the_inverse_of_the_read_side(self):
        assert AGENT_TOOL_TO_ENUM == {tool: enum for enum, tool in AGENT_ENUM_TO_TOOL.items()}

    def test_mcp_map_is_the_inverse_of_the_read_side(self):
        assert MCP_TAG_TO_TYPE_ENUM == {tag: enum for enum, tag in MCP_TYPE_ENUM_TO_TAG.items()}

    def test_agent_map_round_trips(self):
        for tool, enum in AGENT_TOOL_TO_ENUM.items():
            assert AGENT_ENUM_TO_TOOL[enum] == tool

    def test_inversion_is_lossless(self):
        # A duplicated tool name on the read side would silently collapse an entry here.
        assert len(AGENT_TOOL_TO_ENUM) == len(AGENT_ENUM_TO_TOOL)
        assert len(MCP_TAG_TO_TYPE_ENUM) == len(MCP_TYPE_ENUM_TO_TAG)


class TestRoundTrip:
    """serialize -> normalize must be the identity on a ucode-native manifest."""

    def test_full_manifest_round_trips(self):
        manifest = _full_manifest()
        assert normalize_managed_config(serialize_managed_config(manifest)) == manifest

    def test_minimal_manifest_round_trips(self):
        manifest = _minimal_manifest()
        assert normalize_managed_config(serialize_managed_config(manifest)) == manifest

    def test_every_known_agent_round_trips(self):
        # Each agent's oneof variant must survive a round trip, including the flat-list agents and
        # codex (which has no model list at all).
        for tool in AGENT_TOOL_TO_ENUM:
            model_config: dict = {"default_model": "system.ai.some-model"}
            if tool == "claude":
                model_config["models"] = {"default_opus_model": "system.ai.claude-opus-4-8"}
            elif tool != "codex":
                model_config["models"] = ["system.ai.some-model"]
            manifest = {
                "default_agent": tool,
                "enabled_agents": {tool: {"model_config": model_config}},
            }
            assert normalize_managed_config(serialize_managed_config(manifest)) == manifest, tool

    def test_every_mcp_type_round_trips(self):
        for tag in MCP_TAG_TO_TYPE_ENUM:
            manifest = {"mcp_servers": [{"name": "some-server", "type": tag}]}
            assert normalize_managed_config(serialize_managed_config(manifest)) == manifest, tag


class TestSerialize:
    def test_maps_tool_names_to_proto_enums(self):
        payload = serialize_managed_config(_minimal_manifest())
        assert payload["default_agent"] == "CODING_AGENT_CLAUDE_CODE"
        assert payload["enabled_agents"][0]["agent"] == "CODING_AGENT_CLAUDE_CODE"

    def test_claude_model_config_uses_family_slots(self):
        payload = serialize_managed_config(_full_manifest())
        claude = next(
            entry
            for entry in payload["enabled_agents"]
            if entry["agent"] == "CODING_AGENT_CLAUDE_CODE"
        )
        variant = claude["config"]["model_config"]
        assert set(variant) == {"claude"}
        assert variant["claude"]["models"] == {
            "default_opus_model": "system.ai.claude-opus-4-8",
            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
        }

    def test_codex_model_config_has_no_model_list(self):
        # CodexModelConfig carries only model_provider_service + default_model.
        manifest = {
            "default_agent": "codex",
            "enabled_agents": {
                "codex": {
                    "model_config": {
                        "default_model": "system.ai.gpt-5-6",
                        # Even if a caller passes a list, it must not be serialized.
                        "models": ["system.ai.gpt-5-6"],
                    }
                }
            },
        }
        payload = serialize_managed_config(manifest)
        variant = payload["enabled_agents"][0]["config"]["model_config"]["codex"]
        assert "models" not in variant
        assert variant["default_model"] == "system.ai.gpt-5-6"

    def test_flat_list_agents_use_repeated_models(self):
        payload = serialize_managed_config(_full_manifest())
        opencode = next(
            entry
            for entry in payload["enabled_agents"]
            if entry["agent"] == "CODING_AGENT_OPENCODE"
        )
        variant = opencode["config"]["model_config"]["opencode"]
        assert variant["models"] == ["system.ai.claude-opus-4-8", "system.ai.kimi-k2-6"]

    def test_model_provider_service_is_carried_through(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "model_provider_service": "main.default.anthropic-mps",
                        "default_model": "claude-sonnet-4-6",
                    }
                }
            },
        }
        payload = serialize_managed_config(manifest)
        variant = payload["enabled_agents"][0]["config"]["model_config"]["claude"]
        assert variant["model_provider_service"] == "main.default.anthropic-mps"

    def test_mcp_types_map_to_proto_enums(self):
        payload = serialize_managed_config(_full_manifest())
        assert payload["mcp_servers"] == [
            {"name": "system.ai.github", "type": "MCP_SERVER_TYPE_UC_SERVICE"},
            {"name": "genie-space-id", "type": "MCP_SERVER_TYPE_GENIE"},
        ]

    def test_tracing_becomes_a_table_object(self):
        payload = serialize_managed_config(_full_manifest())
        assert payload["tracing"] == {"table": "main.default.ucode-traces"}

    def test_per_agent_tracing_override(self):
        payload = serialize_managed_config(_full_manifest())
        claude = next(
            entry
            for entry in payload["enabled_agents"]
            if entry["agent"] == "CODING_AGENT_CLAUDE_CODE"
        )
        assert claude["config"]["tracing_config"] == {"table": "main.default.claude-traces"}

    def test_budget_tiers_keep_fractions(self):
        # The server validates 0 <= spending_percentage <= 1, so these stay fractions.
        payload = serialize_managed_config(_full_manifest())
        tiers = payload["budget_policy"]["tiers"]
        assert [tier["spending_percentage"] for tier in tiers] == [0.8, 1.0]
        assert tiers[1]["default_agent"] == "CODING_AGENT_OPENCODE"

    def test_the_deprecated_top_level_budget_id_is_never_emitted(self):
        # `CodingAgentConfig.budget_id` (field 3) is deprecated in favour of
        # `budget_policy.budget_id`, and the CRUD handler rejects a write that sets it. The budget
        # id must appear only under the policy.
        payload = serialize_managed_config(_full_manifest())
        assert "budget_id" not in payload
        assert payload["budget_policy"]["budget_id"] == "c6563b45-df9a-4b19-afb2-d42dc2b52576"

    def test_a_manifest_carrying_a_top_level_budget_id_still_omits_it(self):
        # A hand-written `--from-file` manifest could set it; the serializer must not pass it on.
        payload = serialize_managed_config({**_full_manifest(), "budget_id": BUDGET_ID})
        assert "budget_id" not in payload

    def test_unknown_agent_is_dropped(self):
        payload = serialize_managed_config(
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "m"}},
                    "some-future-agent": {"model_config": {"default_model": "m"}},
                },
            }
        )
        assert [entry["agent"] for entry in payload["enabled_agents"]] == [
            "CODING_AGENT_CLAUDE_CODE"
        ]

    def test_unknown_mcp_type_is_dropped(self):
        payload = serialize_managed_config(
            {"mcp_servers": [{"name": "a", "type": "not-a-type"}, {"name": "b", "type": "sql"}]}
        )
        assert payload["mcp_servers"] == [{"name": "b", "type": "MCP_SERVER_TYPE_DATABRICKS_SQL"}]

    def test_empty_manifest_serializes_to_empty_payload(self):
        assert serialize_managed_config({}) == {}

    def test_output_only_fields_are_never_emitted(self):
        # A caller (or a round-tripped GET) may carry server-owned fields; they must not be sent.
        payload = serialize_managed_config(
            {
                **_minimal_manifest(),
                "workspace_id": 12345,
                "create_time": "2026-01-01T00:00:00Z",
                "created_user_id": 42,
            }
        )
        assert "workspace_id" not in payload
        assert "create_time" not in payload
        assert "created_user_id" not in payload

    def test_name_is_carried_through_when_present(self):
        payload = serialize_managed_config(
            {**_minimal_manifest(), "name": "coding-agent-configs/abc"}
        )
        assert payload["name"] == "coding-agent-configs/abc"


class TestModelOptions:
    def test_claude_only_sees_claude_models(self):
        options = model_options_for_agent("claude", STATE)
        assert options == [
            "system.ai.claude-opus-4-8",
            "system.ai.claude-sonnet-4-6",
            "system.ai.claude-haiku-4-5",
        ]

    def test_gemini_only_sees_gemini_models(self):
        assert model_options_for_agent("gemini", STATE) == ["system.ai.gemini-3-flash"]

    def test_codex_sees_gpt_and_oss(self):
        assert model_options_for_agent("codex", STATE) == [
            "system.ai.gpt-5-6",
            "system.ai.kimi-k2-6",
        ]

    def test_multi_provider_agents_see_everything(self):
        for tool in ("opencode", "pi", "copilot", "omp"):
            options = model_options_for_agent(tool, STATE)
            assert "system.ai.claude-opus-4-8" in options, tool
            assert "system.ai.gpt-5-6" in options, tool
            assert "system.ai.gemini-3-flash" in options, tool
            assert "system.ai.kimi-k2-6" in options, tool

    def test_omp_matches_pi_options(self):
        # omp is the same multi-provider harness as pi; the setup picker
        # excludes it, but any residual path must offer pi-identical options.
        assert model_options_for_agent("omp", STATE) == model_options_for_agent("pi", STATE)

    def test_empty_state_yields_no_options(self):
        assert model_options_for_agent("claude", {}) == []

    def test_options_are_deduplicated(self):
        # An id can land in two family buckets (e.g. an OSS model also listed under gpt).
        state = {"codex_models": ["system.ai.kimi-k2-6"], "oss_models": ["system.ai.kimi-k2-6"]}
        assert model_options_for_agent("codex", state) == ["system.ai.kimi-k2-6"]

    def test_unknown_agent_has_no_families(self):
        assert model_families_for_agent("not-an-agent") == ()
        assert model_options_for_agent("not-an-agent", STATE) == []

    def test_malformed_state_is_ignored(self):
        state = {"claude_models": "not-a-dict", "codex_models": {"not": "a list"}}
        assert model_options_for_agent("claude", state) == []
        assert model_options_for_agent("codex", state) == []


class TestProviderServiceSupport:
    def test_claude_supports_anthropic_and_bedrock(self):
        assert supports_provider_service("claude", "anthropic")
        assert supports_provider_service("claude", "amazon_bedrock")

    def test_codex_supports_openai(self):
        assert supports_provider_service("codex", "openai")

    def test_claude_does_not_support_openai(self):
        assert not supports_provider_service("claude", "openai")

    def test_other_agents_have_no_provider_support(self):
        for tool in ("gemini", "opencode", "pi", "copilot"):
            assert not supports_provider_service(tool, "anthropic"), tool


class TestClaudeSlots:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("system.ai.claude-opus-4-8", "opus"),
            ("databricks-claude-sonnet-4-6", "sonnet"),
            ("system.ai.claude-haiku-4-5", "haiku"),
            ("system.ai.claude-fable-5", "fable"),
            ("system.ai.gpt-5-6", None),
            ("claude-without-family", None),
        ],
    )
    def test_family_detection(self, model, expected):
        assert claude_family_for_model(model) == expected

    def test_groups_models_into_slots(self):
        slots = claude_model_slots(["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-4-6"])
        assert slots == {
            "default_opus_model": "system.ai.claude-opus-4-8",
            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
        }

    def test_first_model_wins_within_a_family(self):
        slots = claude_model_slots(["system.ai.claude-opus-4-8", "system.ai.claude-opus-4-7"])
        assert slots == {"default_opus_model": "system.ai.claude-opus-4-8"}

    def test_unidentifiable_models_are_skipped(self):
        assert claude_model_slots(["system.ai.gpt-5-6"]) == {}

    def test_slots_serialize_into_the_claude_variant(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": claude_model_slots(["system.ai.claude-opus-4-8"]),
                    }
                }
            },
        }
        payload = serialize_managed_config(manifest)
        variant = payload["enabled_agents"][0]["config"]["model_config"]["claude"]
        assert variant["models"] == {"default_opus_model": "system.ai.claude-opus-4-8"}


class TestClaudeFamilyCandidates:
    """Discovery keeps one id per family for the launch path; authoring needs the alternatives."""

    ALL = [
        "system.ai.claude-opus-4-1",
        "system.ai.claude-opus-4-8",
        "system.ai.claude-opus-5",
        "system.ai.claude-sonnet-4-6",
        "system.ai.claude-sonnet-5",
        "system.ai.claude-haiku-4-5",
        "system.ai.gpt-5-6",
    ]

    def test_groups_by_family(self):
        from ucode.managed_setup import claude_family_candidates

        got = claude_family_candidates(self.ALL)
        assert set(got) == {"opus", "sonnet", "haiku"}
        assert got["haiku"] == ["system.ai.claude-haiku-4-5"]

    def test_newest_first_within_a_family(self):
        from ucode.managed_setup import claude_family_candidates

        assert claude_family_candidates(self.ALL)["opus"] == [
            "system.ai.claude-opus-5",
            "system.ai.claude-opus-4-8",
            "system.ai.claude-opus-4-1",
        ]

    def test_non_claude_models_are_ignored(self):
        from ucode.managed_setup import claude_family_candidates

        assert not any(
            "gpt" in m for models in claude_family_candidates(self.ALL).values() for m in models
        )

    def test_deduplicates(self):
        from ucode.managed_setup import claude_family_candidates

        got = claude_family_candidates(["system.ai.claude-opus-5", "system.ai.claude-opus-5"])
        assert got["opus"] == ["system.ai.claude-opus-5"]

    def test_falls_back_to_the_per_family_picks(self):
        # Without the full listing, the bucketed state is all that's available — one per family, but
        # enough for the per-slot prompts to work.
        from ucode.managed_setup import claude_family_candidates

        got = claude_family_candidates([], {"claude_models": {"opus": "system.ai.claude-opus-5"}})
        assert got == {"opus": ["system.ai.claude-opus-5"]}

    def test_empty_everything_yields_nothing(self):
        from ucode.managed_setup import claude_family_candidates

        assert claude_family_candidates([], {}) == {}

    def test_slot_names_match_the_proto(self):
        # ClaudeDefaultModels fields, verified against ai-gateway-api service.proto.
        from ucode.managed_setup import CLAUDE_SLOT_FOR_FAMILY

        assert set(CLAUDE_SLOT_FOR_FAMILY.values()) == {
            "default_fable_model",
            "default_opus_model",
            "default_sonnet_model",
            "default_haiku_model",
        }


class TestValidate:
    def test_full_manifest_is_valid(self):
        assert validate_manifest(_full_manifest(), STATE) == []

    def test_minimal_manifest_is_valid(self):
        assert validate_manifest(_minimal_manifest(), STATE) == []

    def test_empty_manifest_is_valid(self):
        # Nothing configured is not an error here; the wizard decides whether to publish it.
        assert validate_manifest({}, STATE) == []

    def test_default_agent_required_when_agents_present(self):
        manifest = {"enabled_agents": {"claude": {"model_config": {"default_model": "m"}}}}
        errors = validate_manifest(manifest)
        assert any("default_agent is required" in e for e in errors)

    def test_default_agent_must_be_enabled(self):
        manifest = {
            "default_agent": "codex",
            "enabled_agents": {"claude": {"model_config": {"default_model": "m"}}},
        }
        errors = validate_manifest(manifest)
        assert any("must appear in enabled_agents" in e for e in errors)

    def test_default_agent_needs_a_default_model(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {}},
        }
        errors = validate_manifest(manifest)
        assert any("model_config.default_model" in e for e in errors)

    def test_unknown_agent_is_rejected(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}},
                "not-an-agent": {},
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("not a supported agent" in e for e in errors)

    def test_unknown_model_is_rejected(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": {"default_model": "system.ai.nope"}}},
        }
        errors = validate_manifest(manifest, STATE)
        assert any("not available on this workspace" in e for e in errors)

    def test_older_claude_version_is_recognized(self):
        # `claude_models` holds only the newest per family, so without the full listing an older
        # version the per-family prompts offered would be wrongly rejected.
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {"default_opus_model": "system.ai.claude-opus-4-8"},
                    }
                }
            },
        }
        state = {
            "claude_models": {"opus": "system.ai.claude-opus-5"},
            "all_claude_models": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"],
        }
        assert validate_manifest(manifest, state) == []

    def test_unknown_claude_version_is_still_rejected(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-9-9"}}
            },
        }
        state = {
            "claude_models": {"opus": "system.ai.claude-opus-5"},
            "all_claude_models": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"],
        }
        errors = validate_manifest(manifest, state)
        assert any("not available on this workspace" in e for e in errors)

    def test_custom_model_is_accepted_via_the_marker(self):
        # A hand-typed model service outside the discovered inventory is listed in `custom_models`
        # (it was verified to exist when entered), so the inventory check must not reject it.
        manifest = {
            "default_agent": "codex",
            "enabled_agents": {
                "codex": {
                    "model_config": {
                        "default_model": "main.aarushi.gpt-5-custom",
                        "custom_models": ["main.aarushi.gpt-5-custom"],
                    }
                }
            },
        }
        assert validate_manifest(manifest, STATE) == []

    def test_unmarked_custom_model_is_still_rejected(self):
        # Without the marker the same id is an unknown model — the marker is what vouches for it.
        manifest = {
            "default_agent": "codex",
            "enabled_agents": {
                "codex": {"model_config": {"default_model": "main.aarushi.gpt-5-custom"}}
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("not available on this workspace" in e for e in errors)

    def test_model_check_skipped_without_state(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": {"default_model": "anything"}}},
        }
        assert validate_manifest(manifest) == []

    def test_model_check_skipped_for_provider_service(self):
        # MPS model ids come from the provider's catalog, not the UC inventory.
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "model_provider_service": "main.default.anthropic-mps",
                        "default_model": "claude-sonnet-5",
                    }
                }
            },
        }
        assert validate_manifest(manifest, STATE) == []

    def test_mcp_server_needs_a_name(self):
        errors = validate_manifest({"mcp_servers": [{"type": "sql"}]})
        assert any("name is required" in e for e in errors)

    def test_mcp_server_needs_a_known_type(self):
        errors = validate_manifest({"mcp_servers": [{"name": "a", "type": "bogus"}]})
        assert any("is not recognized" in e for e in errors)

    def test_empty_skill_name_is_rejected(self):
        errors = validate_manifest({"skills": {"names": ["ok", ""]}})
        assert any("skills.names" in e for e in errors)

    def test_empty_tracing_table_is_rejected(self):
        errors = validate_manifest({"tracing_table": ""})
        assert any("tracing_table" in e for e in errors)

    def test_budget_policy_needs_a_budget_id(self):
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {
                "tiers": [
                    {
                        "spending_percentage": 0.5,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    }
                ]
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("budget_policy.budget_id is required" in e for e in errors)

    @pytest.mark.parametrize("pct", [1.5, -0.1, 80])
    def test_tier_percentage_must_be_a_fraction(self, pct):
        # 80 is the classic mistake: the spec doc writes percents, the API wants fractions.
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": pct,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    }
                ],
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("fraction" in e for e in errors), errors

    def test_tier_percentages_must_be_unique(self):
        tier = {
            "spending_percentage": 0.5,
            "default_agent": "claude",
            "default_model": "system.ai.claude-opus-4-8",
        }
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {"budget_id": BUDGET_ID, "tiers": [tier, dict(tier)]},
        }
        errors = validate_manifest(manifest, STATE)
        assert any("must be unique" in e for e in errors)

    def test_tier_agent_model_pairs_must_differ(self):
        # Distinct percentages, same agent+model: the higher tier is a no-op, because the server
        # picks the highest crossed tier and it selects the same pair the lower one already did.
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.5,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    },
                    {
                        "spending_percentage": 0.9,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    },
                ],
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("no-op" in e for e in errors), errors

    def test_same_agent_different_model_across_tiers_is_allowed(self):
        # A real step-down: same agent, cheaper model. Must not be flagged as a duplicate. Both
        # models are configured on the agent, so the tier-model check passes and only the
        # duplicate-combo rule is under test.
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {
                            "default_opus_model": "system.ai.claude-opus-4-8",
                            "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                        },
                    }
                }
            },
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.5,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    },
                    {
                        "spending_percentage": 0.9,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-sonnet-4-6",
                    },
                ],
            },
        }
        assert validate_manifest(manifest, STATE) == []

    def test_tier_agent_must_be_enabled(self):
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.5,
                        "default_agent": "opencode",
                        "default_model": "system.ai.kimi-k2-6",
                    }
                ],
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("must appear in enabled_agents" in e for e in errors)

    def test_tier_needs_a_default_model(self):
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [{"spending_percentage": 0.5, "default_agent": "claude"}],
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("default_model is required" in e for e in errors)

    def test_tier_model_must_be_one_the_agent_has(self):
        # The server only checks that the tier's agent is enabled, so without this a tier activates
        # and hands the developer a model their agent was never configured with.
        manifest = {
            "default_agent": "pi",
            "enabled_agents": {
                "pi": {
                    "model_config": {
                        "default_model": "system.ai.kimi-k2-6",
                        "models": ["system.ai.kimi-k2-6"],
                    }
                }
            },
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.8,
                        "default_agent": "pi",
                        "default_model": "system.ai.gpt-5-6",
                    }
                ],
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("is not one of the models configured for 'pi'" in e for e in errors), errors

    def test_tier_model_from_the_agents_list_is_accepted(self):
        manifest = {
            "default_agent": "pi",
            "enabled_agents": {
                "pi": {
                    "model_config": {
                        "default_model": "system.ai.kimi-k2-6",
                        "models": ["system.ai.kimi-k2-6", "system.ai.gpt-5-6"],
                    }
                }
            },
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.8,
                        "default_agent": "pi",
                        "default_model": "system.ai.gpt-5-6",
                    }
                ],
            },
        }
        assert validate_manifest(manifest, STATE) == []

    def test_tier_model_matching_a_claude_family_slot_is_accepted(self):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {"default_sonnet_model": "system.ai.claude-sonnet-4-6"},
                    }
                }
            },
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.8,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-sonnet-4-6",
                    }
                ],
            },
        }
        assert validate_manifest(manifest, STATE) == []

    def test_tier_model_check_skipped_when_the_agent_lists_nothing(self):
        # A provider-service agent has no enumerable catalog, so there is nothing to check against.
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "model_provider_service": "main.default.anthropic-mps",
                        "default_model": "claude-sonnet-5",
                    }
                }
            },
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.8,
                        "default_agent": "claude",
                        "default_model": "claude-sonnet-5",
                    }
                ],
            },
        }
        assert validate_manifest(manifest, STATE) == []

    def test_budget_policy_alone_still_requires_a_default_agent(self):
        errors = validate_manifest({"budget_policy": {"budget_id": BUDGET_ID}})
        assert any("default_agent is required" in e for e in errors)

    @pytest.mark.parametrize("bad_id", ["not-a-uuid", "b", "1111", "11111111-1111-1111-1111"])
    def test_budget_id_must_be_a_uuid(self, bad_id):
        # The server requires a parseable UUID. The wizard can only offer real ids, but
        # `--from-file` can carry anything, and rejecting it here beats a round-trip failure.
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {"budget_id": bad_id, "tiers": []},
        }
        errors = validate_manifest(manifest, STATE)
        assert any("must be a UUID" in e for e in errors), errors

    def test_a_real_uuid_is_accepted(self):
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {"budget_id": "c6563b45-df9a-4b19-afb2-d42dc2b52576", "tiers": []},
        }
        assert validate_manifest(manifest, STATE) == []

    def test_tier_positions_are_reported_zero_based(self):
        # The server indexes tiers with `zipWithIndex`, so an admin comparing ucode's message with
        # the API's must see the same number for the same tier.
        manifest = {
            **_minimal_manifest(),
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [{"spending_percentage": 0.5, "default_agent": "claude"}],
            },
        }
        errors = validate_manifest(manifest, STATE)
        assert any("tiers[0]" in e for e in errors), errors
        assert not any("tiers[1]" in e for e in errors), errors

    def test_errors_accumulate(self):
        manifest = {
            "default_agent": "codex",
            "enabled_agents": {"claude": {}},
            "mcp_servers": [{"type": "bogus"}],
        }
        assert len(validate_manifest(manifest, STATE)) >= 3
