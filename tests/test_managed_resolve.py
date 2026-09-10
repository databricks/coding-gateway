"""Tests for managed_resolve.py / managed_apply.py — resolving and writing managed agent settings."""

from __future__ import annotations

import json

import pytest

import ucode.agents.claude as claude
import ucode.agents.opencode as opencode
import ucode.config_io as config_io
import ucode.state as state_mod
from ucode.managed_resolve import (
    managed_default_model,
    managed_enabled_tools,
    managed_launch_model,
    managed_provider_service,
    managed_state_overrides,
    managed_supplies_models,
    managed_unservable_models,
    recommended_agent,
    resolve_state,
)
from ucode.state import MANAGED_OVERLAY_KEY

WORKSPACE = "https://ws.example.com"

# A normalized managed config, as `managed_config.normalize_managed_config` produces it.
MANAGED = {
    "name": "coding-agent-configs/abc-123",
    "default_agent": "claude",
    "enabled_agents": {
        "claude": {
            "model_config": {
                "default_model": "system.ai.claude-opus-5",
                "models": {
                    "default_opus_model": "system.ai.claude-opus-5",
                    "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    "default_haiku_model": "system.ai.claude-haiku-4-5",
                },
            },
        },
        "codex": {
            "model_config": {
                "default_model": "databricks-gpt-5-3-codex",
                "models": ["databricks-gpt-5-3-codex", "databricks-gpt-5-2-codex"],
            }
        },
    },
    "budget_policy": {"display_name": "paved-path", "tiers": []},
}


def _state(**overrides) -> dict:
    state = {
        "workspace": WORKSPACE,
        "managed_configs": {"claude": {"keys": []}, "codex": {"keys": []}},
    }
    state.update(overrides)
    return state


class TestClaudeModels:
    def test_proto_slots_map_to_families(self):
        # The manifest keeps proto spelling (`default_opus_model`); render_overlay reads `opus`.
        # `fable` has no slot here, so it stays unset rather than inheriting `default_model`.
        assert managed_state_overrides(MANAGED, "claude") == {
            "claude_models": {
                "opus": "system.ai.claude-opus-5",
                "sonnet": "system.ai.claude-sonnet-4-6",
                "haiku": "system.ai.claude-haiku-4-5",
            },
            "claude_default_model": "system.ai.claude-opus-5",
        }

    def test_manifest_wins_over_local_per_family(self):
        state = _state(claude_models={"opus": "system.ai.claude-opus-4-8"})
        resolved = resolve_state(MANAGED, state, "claude")
        assert resolved["claude_models"]["opus"] == "system.ai.claude-opus-5"

    def test_family_absent_from_manifest_is_dropped(self):
        # The manifest is the whole allowlist: a family the admin didn't pin is left unset rather
        # than inheriting the developer's model, so a launch can't reach models the admin didn't
        # sanction. Nothing is written for it, so the agent uses its own default.
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"models": {"default_opus_model": "managed-opus"}}}
            }
        }
        state = _state(claude_models={"opus": "local-opus", "fable": "local-fable"})
        assert resolve_state(managed, state, "claude")["claude_models"] == {"opus": "managed-opus"}

    def test_unset_families_do_not_inherit_the_default_model(self):
        # An admin who names only opus is steering people off the other families, so filling them in
        # from `default_model` would quietly re-enable what they left out.
        managed = {
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "managed-default",
                        "models": {"default_opus_model": "managed-opus"},
                    }
                }
            }
        }
        state = _state(claude_models={"sonnet": "local-sonnet"})
        assert resolve_state(managed, state, "claude")["claude_models"] == {"opus": "managed-opus"}

    def test_no_manifest_models_leaves_local_state_alone(self):
        # No override means resolve_state never touches the key, so the developer's own models stand.
        state = _state(claude_models={"sonnet": "local-sonnet"})
        assert resolve_state({}, state, "claude")["claude_models"] == {"sonnet": "local-sonnet"}


class TestListModels:
    def test_manifest_list_replaces_local(self):
        # A flat list has no per-key identity to merge on, so the manifest's list wins outright.
        state = _state(codex_models=["local-codex"])
        assert resolve_state(MANAGED, state, "codex")["codex_models"] == [
            "databricks-gpt-5-3-codex",
            "databricks-gpt-5-2-codex",
        ]

    def test_local_list_stands_when_manifest_silent(self):
        state = _state(codex_models=["local-codex"])
        assert resolve_state({}, state, "codex")["codex_models"] == ["local-codex"]

    def test_blank_entries_dropped(self):
        managed = {"enabled_agents": {"codex": {"model_config": {"models": ["  ", "real", ""]}}}}
        assert managed_state_overrides(managed, "codex") == {"codex_models": ["real"]}


class TestManagedProviderService:
    """The manifest-only read: needed to attribute a provider to the admin, not to local state."""

    def test_returns_the_manifest_provider(self):
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"model_provider_service": "main.default.managed"}}
            }
        }
        assert managed_provider_service(managed, "claude") == "main.default.managed"

    def test_ignores_locally_persisted_provider(self):
        # No fallback to local state: otherwise a developer's own provider would be misreported as
        # the admin's when rejecting a conflicting --provider.
        assert managed_provider_service({}, "claude") is None

    def test_none_for_agent_not_in_manifest(self):
        assert managed_provider_service(MANAGED, "gemini") is None


class TestResolveState:
    def test_does_not_mutate_input_state(self):
        # managed-state.json and state.json stay separate files: resolution is per-write and
        # in-memory, so the developer's own state must come back untouched.
        state = _state(claude_models={"opus": "local-opus"})
        before = json.dumps(state, sort_keys=True)
        resolve_state(MANAGED, state, "claude")
        assert json.dumps(state, sort_keys=True) == before

    def test_layers_managed_models_onto_copy(self):
        resolved = resolve_state(MANAGED, _state(), "claude")
        assert resolved["claude_models"]["opus"] == "system.ai.claude-opus-5"

    def test_preserves_unrelated_state_keys(self):
        resolved = resolve_state(MANAGED, _state(profile="my-profile"), "claude")
        assert resolved["profile"] == "my-profile"
        assert resolved["workspace"] == WORKSPACE

    def test_layers_provider_without_dropping_other_tools(self):
        managed = {
            "enabled_agents": {
                "codex": {"model_config": {"model_provider_service": "main.default.managed"}}
            }
        }
        state = _state(provider_services={"claude": "main.default.keep"})
        resolved = resolve_state(managed, state, "codex")
        assert resolved["provider_services"] == {
            "claude": "main.default.keep",
            "codex": "main.default.managed",
        }


class TestStateFileIsNotRewritten:
    """The managed config must win by precedence, not by overwriting the developer's state file.

    managed-state.json and state.json stay separate on disk: resolution happens in memory and only
    the generated agent settings file reflects it. These tests deliberately let the real
    ``save_state`` run against a temp ``state.json`` — stubbing it out is what let this regress,
    because the overwrite happens inside ``write_tool_config``, one layer below the resolver.
    """

    @pytest.fixture
    def real_state_file(self, tmp_path, monkeypatch):
        """Redirect state.json and both Claude settings files into tmp_path, unstubbed."""
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", tmp_path / "ucode-settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "backup.json")
        managed_settings_path = tmp_path / "managed-settings.json"
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: managed_settings_path)
        monkeypatch.setattr(claude, "managed_writes_allowed", lambda: True)

        def reconcile_managed_file(path, desired_text, **kwargs):
            path.write_text(desired_text, encoding="utf-8")
            return "written"

        monkeypatch.setattr(claude, "reconcile_managed_file", reconcile_managed_file)
        # Seed a developer whose own opus choice differs from the manifest's.
        state_mod.save_state(
            {
                "workspace": WORKSPACE,
                "managed_configs": {"claude": {"keys": []}},
                "claude_models": {"opus": "system.ai.claude-opus-4-8"},
            }
        )
        return tmp_path

    @staticmethod
    def _persisted_claude_models(tmp_path) -> dict:
        full = json.loads((tmp_path / "state.json").read_text())
        return full["workspaces"][WORKSPACE].get("claude_models") or {}

    def test_developers_state_file_keeps_their_own_model(self, real_state_file):
        # The developer picked opus-4-8; the manifest says opus-5. After configuring under the
        # managed config, state.json must still say opus-4-8 — the admin's value belongs only in
        # the generated settings file, so removing the managed config restores their own choice.
        assert self._persisted_claude_models(real_state_file)["opus"] == "system.ai.claude-opus-4-8"

        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        claude.write_tool_config(resolved_state, None)

        assert self._persisted_claude_models(real_state_file)["opus"] == "system.ai.claude-opus-4-8"

    def test_settings_file_gets_the_managed_model(self, real_state_file):
        # The other half of the contract: precedence must actually reach the generated file.
        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        claude.write_tool_config(
            resolved_state,
            None,
            coding_agent_config_defaults={
                "opus": "system.ai.claude-opus-5",
                "sonnet": "system.ai.claude-sonnet-4-6",
                "haiku": "system.ai.claude-haiku-4-5",
            },
        )

        env = json.loads((real_state_file / "ucode-settings.json").read_text())["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"].startswith("system.ai.claude-opus-5")
        managed_env = json.loads((real_state_file / "managed-settings.json").read_text())["env"]
        assert managed_env["ANTHROPIC_DEFAULT_OPUS_MODEL"].startswith("system.ai.claude-opus-5")

    def test_overlay_bookkeeping_never_lands_on_disk(self, real_state_file):
        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        claude.write_tool_config(resolved_state, None)

        raw = (real_state_file / "state.json").read_text()
        assert MANAGED_OVERLAY_KEY not in raw

    def test_repeated_saves_still_restore_the_developers_value(self, real_state_file):
        # A launch can save twice from the same dict (the relayed proxy rewrites its port after
        # configure), so the swap-back must be idempotent rather than consuming the overlay.
        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        state_mod.save_state(resolved_state)
        state_mod.save_state(resolved_state)

        assert self._persisted_claude_models(real_state_file)["opus"] == "system.ai.claude-opus-4-8"
        # The in-memory dict still carries the managed value for rendering.
        assert resolved_state["claude_models"]["opus"] == "system.ai.claude-opus-5"

    def test_managed_provider_does_not_overwrite_the_developers_provider(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        state_mod.save_state(
            {"workspace": WORKSPACE, "provider_services": {"claude": "main.default.mine"}}
        )
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"model_provider_service": "main.default.admin"}}
            }
        }
        resolved_state = resolve_state(managed, state_mod.load_state(), "claude")
        state_mod.save_state(resolved_state)

        full = json.loads((tmp_path / "state.json").read_text())
        assert full["workspaces"][WORKSPACE]["provider_services"] == {"claude": "main.default.mine"}
        assert resolved_state["provider_services"]["claude"] == "main.default.admin"

    def test_developer_with_no_prior_value_is_not_given_one(self, tmp_path, monkeypatch):
        # The developer never configured claude models; the manifest supplies them for this launch
        # only, so state.json must not gain a key recording the admin's choice as theirs.
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        state_mod.save_state({"workspace": WORKSPACE})

        resolved_state = resolve_state(MANAGED, state_mod.load_state(), "claude")
        state_mod.save_state(resolved_state)

        full = json.loads((tmp_path / "state.json").read_text())
        assert not full["workspaces"][WORKSPACE].get("claude_models")

    @pytest.mark.parametrize(
        ("tool", "models_key", "managed_models"),
        [
            ("codex", "codex_models", ["managed-codex"]),
            ("gemini", "gemini_models", ["managed-gemini"]),
        ],
    )
    def test_other_agents_state_is_also_preserved(
        self, tmp_path, monkeypatch, tool, models_key, managed_models
    ):
        # Every agent's write_tool_config calls save_state, so the swap-back has to hold for all of
        # them — not just claude.
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        state_mod.save_state({"workspace": WORKSPACE, models_key: ["mine"]})
        managed = {"enabled_agents": {tool: {"model_config": {"models": managed_models}}}}

        resolved_state = resolve_state(managed, state_mod.load_state(), tool)
        state_mod.save_state(resolved_state)

        full = json.loads((tmp_path / "state.json").read_text())
        assert full["workspaces"][WORKSPACE][models_key] == ["mine"]
        assert resolved_state[models_key] == managed_models


class TestManagedDefaultModel:
    """The model a launch starts on, which is separate from the family slots."""

    def test_returns_the_manifest_default_model(self):
        assert managed_default_model(MANAGED, "claude") == "system.ai.claude-opus-5"

    def test_none_when_the_manifest_names_no_default(self):
        managed = {"enabled_agents": {"claude": {"model_config": {"models": {}}}}}
        assert managed_default_model(managed, "claude") is None

    def test_none_for_agent_not_in_manifest(self):
        assert managed_default_model({}, "codex") is None

    def test_survives_a_config_with_no_model_list(self):
        # CodexModelConfig has no `models` field at all, so default_model is the only model an
        # admin can set — it has to be usable on its own or a codex launch can't honor the config.
        managed = {"enabled_agents": {"codex": {"model_config": {"default_model": "admin-codex"}}}}
        state = {"workspace": WORKSPACE, "managed_configs": {"codex": {"keys": []}}}
        assert managed_default_model(managed, "codex") == "admin-codex"
        # Nothing lands in the model list, so the launch path must pass the default model into
        # resolve_launch_model rather than relying on state having one.
        assert resolve_state(managed, state, "codex").get("codex_models") is None


class TestManagedSuppliesModels:
    """Whether the config already says which models an agent uses, so discovery can be skipped."""

    def test_true_when_a_family_slot_is_pinned(self):
        managed = {
            "enabled_agents": {
                "claude": {
                    "model_config": {"models": {"default_opus_model": "system.ai.claude-opus-5"}}
                }
            }
        }
        assert managed_supplies_models(managed, "claude") is True

    def test_true_for_a_default_model(self):
        managed = {"enabled_agents": {"codex": {"model_config": {"default_model": "gpt"}}}}
        assert managed_supplies_models(managed, "codex") is True

    def test_true_for_a_provider(self):
        # A provider routes by header and pins no Databricks model, so discovery is moot.
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"model_provider_service": "main.default.mps"}}
            }
        }
        assert managed_supplies_models(managed, "claude") is True

    def test_true_for_a_flat_model_list(self):
        managed = {"enabled_agents": {"opencode": {"model_config": {"models": ["a", "b"]}}}}
        assert managed_supplies_models(managed, "opencode") is True

    def test_false_when_the_config_names_no_models(self):
        # Discovery still has to run, or the launch has nothing to pin.
        managed = {"enabled_agents": {"claude": {}}}
        assert managed_supplies_models(managed, "claude") is False

    def test_false_for_an_agent_the_config_does_not_cover(self):
        assert managed_supplies_models(MANAGED, "gemini") is False

    def test_false_for_no_config_at_all(self):
        # First launch has no persisted copy yet, so discovery runs exactly as it always did.
        assert managed_supplies_models(None, "claude") is False

    def test_false_when_slots_are_present_but_blank(self):
        managed = {
            "enabled_agents": {
                "claude": {"model_config": {"models": {"default_opus_model": "   "}}}
            }
        }
        assert managed_supplies_models(managed, "claude") is False


class TestManagedStateOverrides:
    """Each agent reads its models from a different shape, so the manifest has to be translated."""

    def test_opencode_gets_provider_buckets_not_a_flat_list(self):
        # OpenCode's state is `{provider: [models]}` and its writer calls `.get()` on it, so handing
        # it the manifest's flat list raises AttributeError.
        managed = {
            "enabled_agents": {
                "opencode": {
                    "model_config": {
                        "models": [
                            "system.ai.claude-opus-4-8",
                            "system.ai.gemini-3-flash",
                            "system.ai.kimi-k2-7-code",
                        ]
                    }
                }
            }
        }
        assert managed_state_overrides(managed, "opencode") == {
            "opencode_models": {
                "anthropic": ["system.ai.claude-opus-4-8"],
                "gemini": ["system.ai.gemini-3-flash"],
                "oss": ["system.ai.kimi-k2-7-code"],
            }
        }

    def test_opencode_buckets_are_usable_by_its_own_writer(self):
        managed = {
            "enabled_agents": {
                "opencode": {"model_config": {"models": ["system.ai.claude-opus-4-8"]}}
            }
        }
        buckets = managed_state_overrides(managed, "opencode")["opencode_models"]
        assert opencode._resolve_model_selector("system.ai.claude-opus-4-8", buckets) == (
            "databricks-anthropic/system.ai.claude-opus-4-8"
        )

    @pytest.mark.parametrize("tool", ["pi", "copilot"])
    def test_pi_and_copilot_get_their_own_key(self, tool):
        # They compose from claude_models/codex_models/gemini_models, which claude, codex, and gemini
        # also read — writing a per-agent policy there would let one agent's config change another's.
        managed = {
            "enabled_agents": {tool: {"model_config": {"models": ["system.ai.claude-opus-4-8"]}}}
        }
        assert managed_state_overrides(managed, tool) == {
            f"{tool}_models": ["system.ai.claude-opus-4-8"]
        }

    def test_no_overrides_when_the_manifest_names_no_models(self):
        assert managed_state_overrides({}, "claude") == {}

    def test_unclassifiable_models_are_dropped_from_buckets(self):
        # A model whose family can't be identified has no provider to route through, so guessing a
        # bucket would produce a selector OpenCode can't resolve. It is dropped, but the models that
        # do classify still apply.
        managed = {
            "enabled_agents": {
                "opencode": {
                    "model_config": {"models": ["mystery-model", "system.ai.claude-opus-4-8"]}
                }
            }
        }
        assert managed_state_overrides(managed, "opencode") == {
            "opencode_models": {"anthropic": ["system.ai.claude-opus-4-8"]}
        }

    def test_no_override_when_nothing_is_servable(self):
        # An all-unservable list must not replace the developer's buckets with an empty dict —
        # that would leave OpenCode with no models at all.
        managed = {"enabled_agents": {"opencode": {"model_config": {"models": ["mystery-model"]}}}}
        state = _state(opencode_models={"anthropic": ["local-opus"]})
        assert managed_state_overrides(managed, "opencode") == {}
        assert resolve_state(managed, state, "opencode")["opencode_models"] == {
            "anthropic": ["local-opus"]
        }
        assert managed_unservable_models(managed, "opencode") == ["mystery-model"]


class TestManagedDefaultModelStateOverrides:
    """The managed default_model should be layered into state for each agent."""

    @pytest.mark.parametrize("tool", ["pi", "copilot", "gemini", "opencode", "codex"])
    def test_emits_a_per_agent_default_model_key(self, tool):
        managed = {"enabled_agents": {tool: {"model_config": {"default_model": "admin-default"}}}}
        assert managed_state_overrides(managed, tool) == {f"{tool}_default_model": "admin-default"}

    def test_emits_default_model_alongside_the_allowlist(self):
        managed = {
            "enabled_agents": {
                "pi": {
                    "model_config": {
                        "default_model": "admin-default",
                        "models": ["model-a", "model-b"],
                    }
                }
            }
        }
        assert managed_state_overrides(managed, "pi") == {
            "pi_default_model": "admin-default",
            "pi_models": ["model-a", "model-b"],
        }

    def test_codex_only_default_model_no_models_field(self):
        # CodexModelConfig has no `models` field, so only default_model can be set.
        managed = {"enabled_agents": {"codex": {"model_config": {"default_model": "admin-codex"}}}}
        state = _state()
        resolved = resolve_state(managed, state, "codex")
        assert resolved.get("codex_default_model") == "admin-codex"
        assert resolved.get("codex_models") is None


class TestManagedEnabledTools:
    def test_lists_the_configs_agents(self):
        managed = {"enabled_agents": {"claude": {}, "opencode": {}}}
        assert managed_enabled_tools(managed) == ["claude", "opencode"]

    def test_empty_when_the_config_names_no_agents(self):
        # Callers treat this as "no opinion", so a budget-only config blocks nothing.
        assert managed_enabled_tools({"budget_policy": {}}) == []


class TestManagedUnservableModels:
    """Warn when the admin's list names only models the agent has no provider for."""

    @staticmethod
    def _managed(tool, models):
        return {"enabled_agents": {tool: {"model_config": {"models": models}}}}

    def test_pi_oss_only_is_unservable(self):
        # Pi has no OSS provider block.
        assert managed_unservable_models(
            self._managed("pi", ["system.ai.kimi-k2-7-code"]), "pi"
        ) == ["system.ai.kimi-k2-7-code"]

    def test_opencode_gpt_only_is_unservable(self):
        # OpenCode has no OpenAI provider block.
        managed = self._managed("opencode", ["system.ai.gpt-5"])
        assert managed_unservable_models(managed, "opencode") == ["system.ai.gpt-5"]

    @pytest.mark.parametrize(
        ("tool", "models"),
        [
            ("pi", ["system.ai.kimi-k2-7-code", "system.ai.claude-opus-4-8"]),
            ("opencode", ["system.ai.gpt-5", "system.ai.claude-opus-4-8"]),
        ],
    )
    def test_no_warning_when_anything_is_servable(self, tool, models):
        assert managed_unservable_models(self._managed(tool, models), tool) == []

    def test_agents_that_pass_models_through_never_warn(self):
        assert managed_unservable_models(self._managed("codex", ["anything"]), "codex") == []


class TestRecommendedAgent:
    """A tier can move the org to a cheaper agent; with none named, default_agent stands."""

    def test_tier_agent_wins(self):
        assert recommended_agent({"agent": "opencode"}, {"default_agent": "claude"}) == "opencode"

    def test_falls_back_to_default_agent(self):
        assert recommended_agent({"agent": None}, {"default_agent": "claude"}) == "claude"
        assert recommended_agent(None, {"default_agent": "claude"}) == "claude"

    def test_none_when_neither_is_set(self):
        assert recommended_agent(None, {}) is None


class TestManagedLaunchModel:
    """A tier's model supersedes the config default, but only for the tier's own agent."""

    MANAGED = {
        "enabled_agents": {
            "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}},
            "opencode": {"model_config": {"default_model": "system.ai.claude-sonnet-4-6"}},
        }
    }

    def test_the_recommended_agent_gets_the_recommended_model(self):
        rec = {"agent": "opencode", "model": "system.ai.kimi-k2-7-code"}
        assert managed_launch_model(self.MANAGED, rec, "opencode") == "system.ai.kimi-k2-7-code"

    def test_other_agents_keep_their_own_default(self):
        # opencode's Kimi model is not servable by claude's Anthropic-dialect endpoint.
        rec = {"agent": "opencode", "model": "system.ai.kimi-k2-7-code"}
        assert managed_launch_model(self.MANAGED, rec, "claude") == "system.ai.claude-opus-4-8"

    def test_a_model_without_an_agent_applies_to_any_tool(self):
        rec = {"agent": None, "model": "system.ai.claude-haiku-4-5"}
        assert managed_launch_model(self.MANAGED, rec, "claude") == "system.ai.claude-haiku-4-5"

    def test_default_model_stands_without_a_recommendation(self):
        assert managed_launch_model(self.MANAGED, None, "claude") == "system.ai.claude-opus-4-8"

    def test_none_when_neither_names_a_model(self):
        assert managed_launch_model({}, None, "pi") is None
