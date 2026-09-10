"""Tests for the interactive `ug setup` flow and its CLI wiring.

The wizard is mostly orchestration, so these focus on the parts where it can silently produce a
wrong manifest: reading tracing/MCP/skills back out of ``state.json``, classifying MCP URLs into
managed-config types, the admin gate, and the per-agent model-config shapes.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import patch

import pytest
import typer.main
from typer.testing import CliRunner

import ucode.cli as cli_mod
import ucode.config_io as config_io_mod
import ucode.managed_config as managed_config_mod
import ucode.managed_wizard as wizard
from ucode.cli import app
from ucode.managed_setup import serialize_managed_config, validate_manifest

runner = CliRunner()

WORKSPACE = "https://ws.example.com"

# `list_workspace_budgets` returns real `budget_configuration_id`s, and validation requires a
# parseable UUID, so the fixtures use one rather than a readable placeholder.
BUDGET_ID = "c6563b45-df9a-4b19-afb2-d42dc2b52576"

STATE = {
    "workspace": WORKSPACE,
    "claude_models": {
        "opus": "system.ai.claude-opus-4-8",
        "sonnet": "system.ai.claude-sonnet-4-6",
    },
    "codex_models": ["system.ai.gpt-5-6"],
    "gemini_models": ["system.ai.gemini-3-flash"],
    "oss_models": ["system.ai.kimi-k2-6"],
}


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    """Point the managed-config file at a tmp dir so no test touches the real ~/.ucode."""
    monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
    monkeypatch.setattr(managed_config_mod, "MANAGED_STATE_PATH", tmp_path / "managed-state.json")
    monkeypatch.setattr(config_io_mod, "_dry_run", False)


class TestTracingReadback:
    def test_reads_uc_destination(self):
        state = {"tracing": {"enabled": True, "uc_destination": "main.default.ucode-traces"}}
        assert wizard._tracing_table_from_state(state) == "main.default.ucode-traces"

    def test_disabled_tracing_yields_none(self):
        state = {"tracing": {"enabled": False, "uc_destination": "main.default.t"}}
        assert wizard._tracing_table_from_state(state) is None

    def test_enabled_without_destination_yields_none(self):
        # A non-UC-backed experiment has no table to publish, so the manifest must omit tracing
        # rather than carry an empty value the server would reject.
        assert wizard._tracing_table_from_state({"tracing": {"enabled": True}}) is None

    def test_missing_tracing_yields_none(self):
        assert wizard._tracing_table_from_state({}) is None

    def test_malformed_tracing_yields_none(self):
        assert wizard._tracing_table_from_state({"tracing": "on"}) is None


class TestMcpServerFromUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # mcp-service stores the dash form the launch path rebuilds the dotted URL from.
            (
                "https://ws.example.com/ai-gateway/mcp-services/system.ai.github",
                ("system-ai-github", "mcp-service"),
            ),
            ("https://ws.example.com/api/2.0/mcp/external/jira-prod", ("jira-prod", "external")),
            ("https://ws.example.com/api/2.0/mcp/genie/01ef", ("01ef", "genie-space")),
            # vector-search / uc-functions store `<catalog>.<schema>`, not the local display slug.
            (
                "https://ws.example.com/api/2.0/mcp/vector-search/my_cat/my_schema",
                ("my_cat.my_schema", "vector-search"),
            ),
            (
                "https://ws.example.com/api/2.0/mcp/functions/dev_cat/dev_fixture",
                ("dev_cat.dev_fixture", "uc-functions"),
            ),
            ("https://ws.example.com/api/2.0/mcp/sql", ("databricks-sql", "sql")),
        ],
    )
    def test_known_urls(self, url, expected):
        assert wizard._mcp_server_from_url(url) == expected

    def test_apps_are_not_publishable(self):
        # An app's host isn't reconstructable from the workspace + an id, so it can't be published.
        assert (
            wizard._mcp_server_from_url("https://mcp-myapp-123.aws.databricksapps.com/mcp") is None
        )

    def test_unknown_url_yields_none(self):
        assert wizard._mcp_server_from_url("https://example.com/something/else") is None

    def test_vector_search_needs_both_catalog_and_schema(self):
        assert (
            wizard._mcp_server_from_url("https://ws.example.com/api/2.0/mcp/functions/onlycat")
            is None
        )


class TestMcpServersFromState:
    def test_maps_registered_servers_to_name_and_type(self):
        state = {
            "mcp_servers": [
                {
                    "name": "databricks-github",
                    "url": f"{WORKSPACE}/ai-gateway/mcp-services/system.ai.github",
                },
                {"name": "databricks-sql", "url": f"{WORKSPACE}/api/2.0/mcp/sql"},
            ]
        }
        # The published name comes from the URL (the identifier the server field holds), not the
        # local display name.
        assert wizard._mcp_servers_from_state(state) == [
            {"name": "system-ai-github", "type": "mcp-service"},
            {"name": "databricks-sql", "type": "sql"},
        ]

    def test_publishes_catalog_schema_for_uc_functions(self):
        # The lossy local slug is replaced with the dotted catalog.schema the launch path can split.
        state = {
            "mcp_servers": [
                {
                    "name": "databricks-functions-dev-cat-dev-fixture",
                    "url": f"{WORKSPACE}/api/2.0/mcp/functions/dev_cat/dev_fixture",
                },
            ]
        }
        assert wizard._mcp_servers_from_state(state) == [
            {"name": "dev_cat.dev_fixture", "type": "uc-functions"},
        ]

    def test_skips_the_skills_registry_entry(self):
        # Skills are published under the manifest's own `skills` field; including the MCP entry too
        # would configure them twice.
        from ucode.mcp import SKILLS_MCP_KIND

        state = {
            "mcp_servers": [
                {
                    "name": "databricks-skill-registry",
                    "kind": SKILLS_MCP_KIND,
                    "url": f"{WORKSPACE}/api/2.0/mcp/sql",
                },
                {"name": "databricks-sql", "url": f"{WORKSPACE}/api/2.0/mcp/sql"},
            ]
        }
        assert wizard._mcp_servers_from_state(state) == [{"name": "databricks-sql", "type": "sql"}]

    def test_skips_apps_and_unclassifiable_servers(self):
        state = {
            "mcp_servers": [
                {"name": "mystery", "url": "https://example.com/nope"},
                {"name": "databricks-app-x", "url": "https://x-1.databricksapps.com/mcp"},
            ]
        }
        assert wizard._mcp_servers_from_state(state) == []

    def test_skips_entries_missing_name_or_url(self):
        state = {
            "mcp_servers": [
                {"url": f"{WORKSPACE}/api/2.0/mcp/sql"},
                {"name": "no-url"},
                "not-a-dict",
            ]
        }
        assert wizard._mcp_servers_from_state(state) == []

    def test_empty_state_yields_nothing(self):
        assert wizard._mcp_servers_from_state({}) == []

    def test_output_validates_as_a_manifest(self):
        state = {
            "mcp_servers": [
                {"name": "databricks-sql", "url": f"{WORKSPACE}/api/2.0/mcp/sql"},
            ]
        }
        servers = wizard._mcp_servers_from_state(state)
        assert validate_manifest({"mcp_servers": servers}) == []


class TestAdminGate:
    def test_non_admin_is_rejected(self):
        with patch.object(wizard, "is_workspace_admin", return_value=False):
            with pytest.raises(RuntimeError, match="not an admin"):
                wizard._require_admin(WORKSPACE, "token")

    def test_admin_passes(self):
        with patch.object(wizard, "is_workspace_admin", return_value=True):
            wizard._require_admin(WORKSPACE, "token")  # must not raise

    def test_unverifiable_check_warns_and_continues(self):
        # A failed SCIM call must not block a legitimate admin — the API enforces the same rule.
        with (
            patch.object(wizard, "is_workspace_admin", return_value=None),
            patch.object(wizard, "print_warning") as warn,
        ):
            wizard._require_admin(WORKSPACE, "token")
        assert warn.called


class TestExistingConfigHandling:
    RICH_CONFIG = {
        "name": "coding-agent-configs/abc",
        "enabled_agents": {"claude": {}, "opencode": {}, "pi": {}},
        "mcp_servers": [{"name": "a", "type": "sql"}],
        "skills": {"names": ["main.default"]},
        "tracing_table": "main.default.traces",
        "budget_policy": {"display_name": "lillys_budget", "budget_id": "abc"},
    }

    def test_continue_when_no_config_exists(self):
        # Nothing published, so there is no prompt — the wizard just proceeds.
        with (
            patch.object(wizard, "get_managed_config", return_value=(None, None)),
            patch.object(wizard, "prompt_for_selection") as select,
            patch.object(wizard, "print_warning") as warn,
        ):
            # No published config, so nothing to carry forward.
            assert wizard._handle_existing_config(WORKSPACE, "token") == (True, None)
        assert not select.called
        assert not warn.called

    def test_read_failure_continues_with_a_note(self):
        # Can't check isn't the same as "there is one"; don't imply data loss or block the wizard.
        with (
            patch.object(wizard, "get_managed_config", return_value=(None, "HTTP 403 Forbidden")),
            patch.object(wizard, "prompt_for_selection") as select,
            patch.object(wizard, "print_note") as note,
        ):
            assert wizard._handle_existing_config(WORKSPACE, "token") == (True, None)
        assert not select.called
        assert note.called

    def test_feature_disabled_blocks_setup_with_an_actionable_error(self):
        # When the coding-agent-config APIs aren't enabled, the read fails with a FEATURE_DISABLED
        # 404. Stop before authoring a draft that can never be published.
        reason = (
            'HTTP 404 Not Found: {"error_code":"FEATURE_DISABLED",'
            '"message":"Coding agent config APIs are not enabled for this workspace."}'
        )
        with (
            patch.object(wizard, "get_managed_config", return_value=(None, reason)),
            patch.object(wizard, "prompt_for_selection") as select,
            patch.object(wizard, "print_note") as note,
            pytest.raises(RuntimeError) as exc_info,
        ):
            wizard._handle_existing_config(WORKSPACE, "token")
        assert not select.called
        message = str(exc_info.value)
        assert message == wizard.CODING_AGENT_CONFIGS_DISABLED_MESSAGE
        assert "`ug configure`" in message
        # The raw 404 / JSON body must not leak into the message.
        assert "404" not in message
        assert "FEATURE_DISABLED" not in message
        assert not note.called

    def test_choosing_create_continues_authoring(self):
        with (
            patch.object(
                wizard,
                "get_managed_config",
                return_value=({"name": "x", "enabled_agents": {}}, None),
            ),
            patch.object(wizard, "prompt_for_selection", return_value="create"),
        ):
            # Continues authoring, and hands back the published config to carry its sections forward.
            assert wizard._handle_existing_config(WORKSPACE, "token") == (
                True,
                {"name": "x", "enabled_agents": {}},
            )

    def test_warning_does_not_itemize_the_existing_config(self):
        # The warning is the same whatever the config holds: an inventory doesn't change what the
        # admin should do, and `ug setup show` prints the real thing for comparison.
        with (
            patch.object(wizard, "get_managed_config", return_value=(self.RICH_CONFIG, None)),
            patch.object(wizard, "prompt_for_selection", return_value="create"),
            patch.object(wizard, "print_warning") as warn,
        ):
            wizard._handle_existing_config(WORKSPACE, "token")
        message = warn.call_args[0][0]
        assert "one config covers every agent" in message
        for leaked in ("Claude Code", "OpenCode", "lillys_budget", "main.default"):
            assert leaked not in message, leaked

    def test_choosing_delete_stops_and_deletes(self):
        with (
            patch.object(
                wizard,
                "get_managed_config",
                return_value=({"name": "cfg/1", "enabled_agents": {}}, None),
            ),
            patch.object(wizard, "prompt_for_selection", return_value="delete"),
            patch.object(wizard, "prompt_yes_no_default", return_value=True),
            patch.object(wizard, "delete_coding_agent_config", return_value=None) as delete,
        ):
            keep_going, _ = wizard._handle_existing_config(WORKSPACE, "token")
            assert keep_going is False
        delete.assert_called_once_with(WORKSPACE, "token", "cfg/1")

    def test_choosing_adopt_confirms_and_stops(self):
        existing = {"name": "cfg/1", "enabled_agents": {"claude": {}}}
        with (
            patch.object(wizard, "get_managed_config", return_value=(existing, None)),
            patch.object(wizard, "prompt_for_selection", return_value="adopt"),
            patch("ucode.cli._confirm_managed_config_applied") as confirm,
        ):
            # Adopting just confirms the config is in force (the launch path applies it) and stops
            # the wizard — no re-authoring, no local writes.
            keep_going, published = wizard._handle_existing_config(WORKSPACE, "token")
            assert keep_going is False
            assert published == existing
        confirm.assert_called_once_with(existing, WORKSPACE)

    def test_delete_declined_leaves_config_intact(self):
        with (
            patch.object(
                wizard,
                "get_managed_config",
                return_value=({"name": "cfg/1", "enabled_agents": {}}, None),
            ),
            patch.object(wizard, "prompt_for_selection", return_value="delete"),
            patch.object(wizard, "prompt_yes_no_default", return_value=False),
            patch.object(wizard, "delete_coding_agent_config") as delete,
        ):
            # Still stops the wizard: the admin chose the delete path, not the author path.
            keep_going, _ = wizard._handle_existing_config(WORKSPACE, "token")
            assert keep_going is False
        assert not delete.called

    def test_delete_failure_raises(self):
        with (
            patch.object(
                wizard,
                "get_managed_config",
                return_value=({"name": "cfg/1", "enabled_agents": {}}, None),
            ),
            patch.object(wizard, "prompt_for_selection", return_value="delete"),
            patch.object(wizard, "prompt_yes_no_default", return_value=True),
            patch.object(wizard, "delete_coding_agent_config", return_value="HTTP 500"),
            pytest.raises(RuntimeError, match="Could not delete"),
        ):
            wizard._handle_existing_config(WORKSPACE, "token")

    def test_cancelling_the_picker_aborts(self):
        with (
            patch.object(
                wizard,
                "get_managed_config",
                return_value=({"name": "x", "enabled_agents": {}}, None),
            ),
            patch.object(wizard, "prompt_for_selection", return_value=None),
            pytest.raises(KeyboardInterrupt),
        ):
            wizard._handle_existing_config(WORKSPACE, "token")


class TestStepBanner:
    """The step headers brand themselves to the invoking command, so a `ug configure` run
    doesn't show `ug setup` headers."""

    def test_defaults_to_ucode_setup(self):
        with patch.object(wizard, "print_section") as section:
            wizard._step_banner(1, "Agents")
        assert section.call_args.args[0].startswith("ug setup · step 1 of ")

    def test_uses_the_command_label_when_given(self):
        with patch.object(wizard, "print_section") as section:
            wizard._step_banner(2, "Models", "ug configure")
        assert section.call_args.args[0].startswith("ug configure · step 2 of ")


class TestSetupCommandToken:
    """A caller (e.g. `ug configure`) can hand setup a token so its admin gate uses the same
    identity as the routing decision, instead of fetching a second time."""

    def test_reuses_a_passed_token_and_skips_a_second_fetch(self):
        seen: list[tuple[str, str]] = []
        with (
            patch.object(
                wizard,
                "get_databricks_token",
                side_effect=AssertionError("must not fetch a token when one was passed"),
            ),
            patch.object(
                wizard,
                "ensure_databricks_auth",
                side_effect=AssertionError("must not re-authenticate when a token was passed"),
            ),
            patch.object(
                wizard, "_require_admin", side_effect=lambda ws, tok: seen.append((ws, tok))
            ),
            # Stop right after the admin gate so the heavy discovery/picker path doesn't run.
            patch.object(wizard, "_handle_existing_config", return_value=(False, None)),
        ):
            code = wizard.setup_command(workspace="https://w", profile=None, token="tok")
        assert code == 0
        assert seen == [("https://w", "tok")]

    def test_fetches_its_own_token_when_none_passed(self):
        seen: list[tuple[str, str]] = []
        with (
            patch.object(wizard, "ensure_databricks_auth", return_value=None),
            patch.object(wizard, "get_databricks_token", return_value="fetched"),
            patch.object(
                wizard, "_require_admin", side_effect=lambda ws, tok: seen.append((ws, tok))
            ),
            patch.object(wizard, "_handle_existing_config", return_value=(False, None)),
        ):
            code = wizard.setup_command(workspace="https://w", profile=None)
        assert code == 0
        assert seen == [("https://w", "fetched")]


class TestModelPrompting:
    def test_codex_takes_a_single_model(self):
        with patch.object(wizard, "prompt_for_selection", return_value="system.ai.gpt-5-6"):
            config = wizard._prompt_models_for_agent("codex", STATE, None)
        # CodexModelConfig has no model list, so the wizard must not build one.
        assert config == {"default_model": "system.ai.gpt-5-6"}

    def test_claude_prompts_one_slot_per_family(self):
        # Claude Code selects by family alias, so each `ClaudeDefaultModels` slot gets its own
        # prompt — and each shows that family's real alternatives, not just the newest.
        candidates = {
            "opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }
        asked: list[str] = []

        def fake_sel(prompt, options, **kwargs):
            asked.append(prompt)
            return [v for v, _ in options][0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert config["models"] == {
            "default_opus_model": "system.ai.claude-opus-5",
            "default_sonnet_model": "system.ai.claude-sonnet-5",
        }
        assert any("opus" in p for p in asked) and any("sonnet" in p for p in asked)

    def test_claude_offers_every_version_in_a_family(self):
        candidates = {"opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"]}
        offered: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            offered.append(values)
            return values[1]  # pick the older opus on purpose

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        # Pinning a known-good older version has to be expressible.
        assert config["models"] == {"default_opus_model": "system.ai.claude-opus-4-8"}
        assert "system.ai.claude-opus-4-8" in offered[0]

    def test_claude_families_can_be_skipped(self):
        candidates = {
            "opus": ["system.ai.claude-opus-5"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            return wizard._SKIP_FAMILY if "sonnet" in prompt else values[0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        # Every slot is optional in the proto; a skipped one must simply be absent.
        assert config["models"] == {"default_opus_model": "system.ai.claude-opus-5"}

    def test_claude_overall_default_comes_from_the_filled_slots(self):
        candidates = {
            "opus": ["system.ai.claude-opus-5"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }
        prompts: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            prompts.append(values)
            return values[0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        # The last prompt is the overall default, offering only what the slots hold — so it can
        # never name a model the config doesn't carry.
        assert set(prompts[-1]) == {"system.ai.claude-opus-5", "system.ai.claude-sonnet-5"}
        assert config["default_model"] in config["models"].values()

    def test_claude_single_slot_skips_the_default_prompt(self):
        candidates = {"opus": ["system.ai.claude-opus-5"]}
        calls = {"n": 0}

        def fake_sel(prompt, options, **kwargs):
            calls["n"] += 1
            return [v for v, _ in options][0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert calls["n"] == 1  # only the opus prompt; no redundant default question
        assert config["default_model"] == "system.ai.claude-opus-5"

    def test_claude_falls_back_to_text_when_nothing_discovered(self):
        with (
            patch.object(wizard, "_claude_candidates", return_value={}),
            patch.object(wizard, "prompt_for_text", return_value="some-claude"),
            patch.object(wizard, "print_warning"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)
        assert config == {"default_model": "some-claude"}

    def test_single_slot_announces_the_inferred_default(self):
        # The one-option prompt is skipped, but silence reads as a dropped step — the admin has to
        # learn that the default was inferred. Announced as a note; the per-agent ✔ (`_confirm_agent`)
        # in the setup loop carries the final confirmation.
        candidates = {"opus": ["system.ai.claude-opus-4-8", "system.ai.claude-opus-5"]}

        def fake_sel(prompt, options, **kwargs):
            if prompt.startswith("Default opus"):
                return "system.ai.claude-opus-4-8"
            return wizard._SKIP_FAMILY

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "print_note") as note,
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert config["default_model"] == "system.ai.claude-opus-4-8"
        notes = " ".join(str(call.args[0]) for call in note.call_args_list)
        assert "overall default" in notes

    def test_claude_all_families_skipped_still_picks_from_the_candidates(self):
        # Skipping every slot is a legitimate minimal config — `models` is optional and each unset
        # slot falls back to `default_model`, so one model covers every family. The admin shouldn't
        # have to type an id we already have.
        candidates = {
            "opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"],
            "sonnet": ["system.ai.claude-sonnet-5"],
        }
        offered: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            offered.append(values)
            if prompt.startswith("Default "):
                return wizard._SKIP_FAMILY
            return "system.ai.claude-opus-4-8"

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "prompt_for_text") as text,
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)

        assert config == {"default_model": "system.ai.claude-opus-4-8"}
        assert "models" not in config
        assert not text.called, "should pick from candidates, not ask for free text"
        # The final prompt offers every candidate across all families (all fit under the picker
        # limit here), plus the custom-entry row.
        assert set(offered[-1]) == {
            "system.ai.claude-opus-5",
            "system.ai.claude-opus-4-8",
            "system.ai.claude-sonnet-5",
            wizard._CUSTOM_MODEL,
        }

    def test_older_claude_version_passes_validation(self):
        # The picker offers every version in a family, but `claude_models` holds only the newest —
        # so validation has to learn about the rest or it rejects a legitimate pick.
        candidates = {"opus": ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"]}
        state = {
            "workspace": "https://ws.example.com",
            "claude_models": {"opus": "system.ai.claude-opus-5"},
        }

        def fake_unbucketed(workspace, token):
            return ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"], None

        with (
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "discover_claude_models_unbucketed", fake_unbucketed),
            patch.object(wizard, "claude_family_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", return_value="system.ai.claude-opus-4-8"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", state, None)

        manifest = {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": config}},
        }
        assert validate_manifest(manifest, state) == []

    def test_claude_cancelled_family_prompt_aborts(self):
        with (
            patch.object(
                wizard, "_claude_candidates", return_value={"opus": ["system.ai.claude-opus-5"]}
            ),
            patch.object(wizard, "prompt_for_selection", return_value=None),
            patch.object(wizard, "print_note"),
        ):
            with pytest.raises(KeyboardInterrupt):
                wizard._prompt_models_for_agent("claude", STATE, None)

    def test_single_model_agents_get_one_prompt(self):
        # Gemini and Copilot declare `repeated string models` in the proto, but their config writers
        # take one model and write one env var — a published list would be read by nothing.
        for tool in ("codex", "gemini", "copilot"):
            options = wizard.model_options_for_agent(tool, STATE)
            with (
                patch.object(wizard, "prompt_for_selection", return_value=options[0]) as select,
                patch.object(wizard, "prompt_for_multi_selection") as multi,
            ):
                config = wizard._prompt_models_for_agent(tool, STATE, None)
            assert select.called, tool
            assert not multi.called, tool
            assert config == {"default_model": options[0]}, tool
            assert "models" not in config, tool

    def test_claude_catalog_is_fetched_once(self):
        # `configure_shared_state` already paged the whole catalog; re-fetching per claude prompt
        # pages it again for no new information.
        calls = {"n": 0}

        def fake_fetch(workspace, token):
            calls["n"] += 1
            return ["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"], None

        state = {"workspace": "https://ws.example.com", "profile": "p"}
        with (
            patch.object(wizard, "discover_claude_models_unbucketed", side_effect=fake_fetch),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "prompt_for_selection", return_value="system.ai.claude-opus-5"),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            wizard._prompt_models_for_agent("claude", state, None)
            wizard._prompt_models_for_agent("claude", state, None)

        assert calls["n"] == 1
        assert state["all_claude_models"]

    def test_nothing_is_prechecked(self):
        # The first option is whatever discovery sorted first, not a recommendation — for pi it is a
        # Claude model, for codex the oldest GPT. Pre-checking it made "hit Enter" produce an
        # arbitrary config.
        captured: dict = {}

        def fake_multi(prompt, options, preselected=None, **kwargs):
            captured["preselected"] = preselected
            return [v for v, _ in options][:1]

        with (
            patch.object(wizard, "prompt_for_multi_selection", side_effect=fake_multi),
            patch.object(wizard, "prompt_for_selection", return_value="x"),
        ):
            wizard._prompt_models_for_agent("pi", STATE, None)

        assert not captured["preselected"]

    def test_list_agents_still_multi_select(self):
        # OpenCode and Pi really do show a model picker, so their lists are honoured.
        for tool in ("opencode", "pi"):
            options = wizard.model_options_for_agent(tool, STATE)
            with (
                patch.object(wizard, "prompt_for_multi_selection", return_value=options[:2]),
                patch.object(wizard, "prompt_for_selection", return_value=options[0]),
            ):
                config = wizard._prompt_models_for_agent(tool, STATE, None)
            assert config["models"] == options[:2], tool

    def test_flat_list_agents_keep_the_picked_list(self):
        picked = ["system.ai.claude-opus-4-8", "system.ai.kimi-k2-6"]
        with (
            patch.object(wizard, "prompt_for_multi_selection", return_value=picked),
            patch.object(wizard, "prompt_for_selection", return_value=picked[0]),
        ):
            config = wizard._prompt_models_for_agent("opencode", STATE, None)
        assert config["models"] == picked

    def test_single_pick_skips_the_default_prompt(self):
        with (
            patch.object(
                wizard, "prompt_for_multi_selection", return_value=["system.ai.claude-opus-4-8"]
            ),
            patch.object(wizard, "prompt_for_selection") as select,
        ):
            config = wizard._prompt_models_for_agent("pi", STATE, None)
        assert config["default_model"] == "system.ai.claude-opus-4-8"
        assert not select.called

    def test_provider_service_offers_its_targets(self):
        # The service's own targets are the model vocabulary the manifest must use, so the admin
        # picks from them rather than typing an id from memory. Uses codex here: it takes one model
        # from any service (Claude with explicit targets is prompted per family — see the Bedrock/
        # Anthropic per-family tests below).
        service = {
            "name": "main.default.openai-mps",
            "provider_type": "openai",
            "targets": ["gpt-5-6", "gpt-5-6-sol"],
            "allow_all_targets": False,
        }
        with (
            patch.object(wizard, "prompt_for_selection", return_value="gpt-5-6") as select,
            patch.object(wizard, "prompt_for_text") as text,
        ):
            config = wizard._prompt_models_for_agent("codex", STATE, service)
        assert config == {
            "model_provider_service": "main.default.openai-mps",
            "default_model": "gpt-5-6",
        }
        assert not text.called, "should not fall back to free text when targets are known"
        # Offered sorted, so the picker order is stable run to run.
        assert [value for value, _ in select.call_args[0][1]] == ["gpt-5-6", "gpt-5-6-sol"]

    def test_provider_service_falls_back_to_text_when_targets_unknown(self):
        # allow_all_targets passes the provider's whole catalog through; there is nothing to list.
        service = {
            "name": "main.default.anthropic-mps",
            "provider_type": "anthropic",
            "targets": [],
            "allow_all_targets": True,
        }
        with (
            patch.object(wizard, "prompt_for_text", return_value="claude-sonnet-5"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config == {
            "model_provider_service": "main.default.anthropic-mps",
            "default_model": "claude-sonnet-5",
        }

    def test_relayed_service_falls_back_to_text(self):
        # A relayed Anthropic subscription service routes by canonical name, with no target list.
        service = {
            "name": "main.default.lilly-anthropic",
            "provider_type": "anthropic",
            "targets": [],
            "allow_all_targets": False,
            "relayed": True,
        }
        with (
            patch.object(wizard, "prompt_for_text", return_value="claude-sonnet-4-6"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config["default_model"] == "claude-sonnet-4-6"

    def test_falls_back_to_free_text_when_nothing_discovered(self):
        with (
            patch.object(wizard, "prompt_for_text", return_value="some-model"),
            patch.object(wizard, "print_warning"),
        ):
            config = wizard._prompt_models_for_agent("pi", {}, None)
        assert config == {"default_model": "some-model"}

    def test_bedrock_claude_prompts_per_family(self):
        # render_overlay pins ANTHROPIC_DEFAULT_<FAMILY>_MODEL from a Bedrock service's provider-side
        # ids (Claude Code can't route its canonical names there), so Claude needs a default per
        # family — a single overall default would leave the other families unpinned.
        service = {
            "name": "main.default.bedrock",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": False,
            "targets": [
                "anthropic.claude-opus-4-1",
                "anthropic.claude-opus-4-8",
                "anthropic.claude-sonnet-4-6",
                "anthropic.claude-haiku-4-5",
            ],
        }
        asked: list[str] = []

        def fake_selection(prompt, options, **kwargs):
            asked.append(prompt)
            return options[0][0]

        with (
            # Decline quick setup to exercise the per-family prompts.
            patch.object(wizard, "prompt_yes_no_default", return_value=False),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_selection),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert [p for p in asked if p.startswith("Default ")] == [
            "Default opus model:",
            "Default sonnet model:",
            "Default haiku model:",
        ]
        assert config["model_provider_service"] == "main.default.bedrock"
        assert config["models"] == {
            "default_opus_model": "anthropic.claude-opus-4-1",
            "default_sonnet_model": "anthropic.claude-sonnet-4-6",
            "default_haiku_model": "anthropic.claude-haiku-4-5",
        }
        # Slots carry the service's own provider-side ids, not workspace `system.ai.*` ids.
        assert all(m.startswith("anthropic.") for m in config["models"].values())
        assert config["default_model"] in service["targets"]

    def test_bedrock_claude_skipped_family_is_omitted(self):
        service = {
            "name": "main.default.bedrock",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": False,
            "targets": ["anthropic.claude-opus-4-8", "anthropic.claude-sonnet-4-6"],
        }
        # pick=-1 selects the trailing "(skip <family>)" option for every family.
        with (
            patch.object(wizard, "prompt_yes_no_default", return_value=False),
            patch.object(wizard, "prompt_for_selection", side_effect=lambda p, o, **k: o[-1][0]),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert "models" not in config
        assert config["default_model"] in service["targets"]

    def test_bedrock_codex_still_takes_a_single_model(self):
        # Only Claude pins per family; codex through any provider takes one model.
        service = {
            "name": "main.default.bedrock-oai",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": False,
            "targets": ["gpt-5-6", "gpt-5-6-sol"],
        }
        asked: list[str] = []
        with (
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=lambda p, o, **k: (asked.append(p), o[0][0])[1],
            ),
        ):
            config = wizard._prompt_models_for_agent("codex", STATE, service)
        assert len(asked) == 1
        assert "models" not in config

    def test_anthropic_claude_with_explicit_targets_prompts_per_family(self):
        # An Anthropic service that publishes explicit canonical targets IS pinned per family by
        # render_overlay (each ANTHROPIC_DEFAULT_<FAMILY>_MODEL to a chosen version), so the wizard
        # prompts per family — same as Bedrock, just canonical ids instead of provider slugs.
        service = {
            "name": "main.default.ant",
            "provider_type": "anthropic",
            "allow_all_targets": False,
            "targets": ["claude-opus-4-8", "claude-sonnet-4-6"],
        }
        asked: list[str] = []
        with (
            patch.object(wizard, "prompt_yes_no_default", return_value=False),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=lambda p, o, **k: (asked.append(p), o[0][0])[1],
            ),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert [p for p in asked if p.startswith("Default ")] == [
            "Default opus model:",
            "Default sonnet model:",
        ]
        assert config["models"] == {
            "default_opus_model": "claude-opus-4-8",
            "default_sonnet_model": "claude-sonnet-4-6",
        }

    def test_quick_setup_fills_newest_per_family_without_per_family_prompts(self):
        # Quick setup (default yes) auto-fills each family with the service's newest id — same pick
        # map_claude_family_models makes — and asks no per-family questions.
        service = {
            "name": "main.default.bedrock",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": False,
            "targets": [
                "anthropic.claude-opus-4-1",
                "anthropic.claude-opus-4-8",
                "anthropic.claude-sonnet-4-6",
            ],
        }
        selections: list[str] = []
        with (
            patch.object(wizard, "prompt_yes_no_default", return_value=True),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=lambda p, o, **k: selections.append(p) or o[0][0],
            ),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        # No per-family (or overall-default) picker was shown — quick setup asked nothing.
        assert selections == []
        assert config["models"] == {
            "default_opus_model": "anthropic.claude-opus-4-8",  # newest opus, not 4-1
            "default_sonnet_model": "anthropic.claude-sonnet-4-6",
        }
        # Overall default is opus (the flagship), not sonnet.
        assert config["default_model"] == "anthropic.claude-opus-4-8"

    def test_quick_setup_default_is_flagship_not_target_order(self):
        # Regression: the overall default must be the highest-tier family (opus), not whichever
        # target sorts first — here haiku leads the list but opus must still win.
        service = {
            "name": "main.default.bedrock",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": False,
            "targets": [
                "us.anthropic.claude-haiku-4-5",
                "us.anthropic.claude-sonnet-5",
                "us.anthropic.claude-opus-5",
            ],
        }
        with (
            patch.object(wizard, "prompt_yes_no_default", return_value=True),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config["default_model"] == "us.anthropic.claude-opus-5"

    def test_quick_setup_default_falls_to_sonnet_without_opus(self):
        service = {
            "name": "main.default.bedrock",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": False,
            "targets": ["us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5"],
        }
        with (
            patch.object(wizard, "prompt_yes_no_default", return_value=True),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert config["default_model"] == "us.anthropic.claude-sonnet-5"

    def test_anthropic_claude_allow_all_takes_a_single_default(self):
        # No explicit targets (allow_all) means nothing is pinned — Claude Code's canonical names
        # route fine — so it falls back to a single free-text default, not per-family slots.
        service = {
            "name": "main.default.ant-all",
            "provider_type": "anthropic",
            "allow_all_targets": True,
            "targets": [],
        }
        with (
            patch.object(wizard, "prompt_for_text", return_value="claude-sonnet-5"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        assert "models" not in config
        assert config["default_model"] == "claude-sonnet-5"

    def test_allow_all_with_explicit_claude_targets_does_not_abort(self):
        # Regression: a service that is allow_all_targets AND lists Claude targets. The family
        # decision must key on the enumerated targets (which allow_all zeroes), not the raw service —
        # otherwise it takes the per-family branch with an empty list and aborts the wizard.
        service = {
            "name": "main.default.weird",
            "provider_type": "amazon_bedrock",
            "allow_all_targets": True,
            "targets": ["us.anthropic.claude-opus-5", "us.anthropic.claude-sonnet-5"],
        }
        with (
            patch.object(wizard, "prompt_for_text", return_value="typed-model"),
            patch.object(wizard, "print_note"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, service)
        # Falls through to the single free-text default; no per-family slots, no KeyboardInterrupt.
        assert "models" not in config
        assert config["default_model"] == "typed-model"

    def test_empty_selection_is_re_prompted(self):
        # An agent with no default_model can't be the config's default_agent (the server rejects
        # it) and gives developers nothing to launch, so "none" is re-asked rather than accepted.
        with (
            patch.object(
                wizard,
                "prompt_for_multi_selection",
                side_effect=[[], ["system.ai.claude-opus-4-8"]],
            ) as picker,
            patch.object(wizard, "print_err") as err,
        ):
            config = wizard._prompt_models_for_agent("pi", STATE, None)
        assert picker.call_count == 2
        assert err.called
        assert config["default_model"] == "system.ai.claude-opus-4-8"

    def test_every_agent_always_gets_a_default_model(self):
        # The invariant the late-validation bug violated: no agent can come back model-less.
        for tool in ("claude", "codex", "gemini", "opencode", "pi", "copilot"):
            options = wizard.model_options_for_agent(tool, STATE)
            with (
                patch.object(wizard, "prompt_for_multi_selection", return_value=[options[0]]),
                patch.object(wizard, "prompt_for_selection", return_value=options[0]),
                # Without this the claude pass reaches for the real catalog, which shells out to
                # `databricks auth token` and depends on the machine's CLI and credentials.
                patch.object(wizard, "discover_claude_models_unbucketed", return_value=([], None)),
                patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
                patch.object(wizard, "print_note"),
                patch.object(wizard, "print_success"),
            ):
                config = wizard._prompt_models_for_agent(tool, STATE, None)
            assert config.get("default_model"), tool

    def test_claude_candidates_survive_a_missing_databricks_cli(self):
        # `get_databricks_token` shells out, so a machine without the CLI on PATH raises
        # FileNotFoundError, not RuntimeError. That must degrade to the bucketed per-family picks
        # rather than aborting the wizard mid-flow.
        def no_cli(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "databricks")

        with patch.object(wizard, "get_databricks_token", side_effect=no_cli):
            candidates = wizard._claude_candidates(dict(STATE))

        # STATE's bucketed `claude_models` still supplies one model per family.
        assert candidates["opus"] == ["system.ai.claude-opus-4-8"]
        assert candidates["sonnet"] == ["system.ai.claude-sonnet-4-6"]

    def test_default_model_is_a_bare_uc_id(self):
        # Provider prefixes (e.g. opencode's `databricks-anthropic/`) are added by each agent's own
        # writer, so the manifest stays agent-neutral.
        with (
            patch.object(
                wizard, "prompt_for_multi_selection", return_value=["system.ai.claude-opus-4-8"]
            ),
        ):
            config = wizard._prompt_models_for_agent("opencode", STATE, None)
        assert config["default_model"] == "system.ai.claude-opus-4-8"
        assert "/" not in config["default_model"]

    def test_dismissed_single_select_aborts_instead_of_re_prompting(self):
        # This used to re-prompt. questionary's `ask` swallows Ctrl-C and returns None, so "empty
        # submission" and "user aborted" are the same value here — and re-asking spun forever.
        # Asserted on the helper directly: the codex path only reaches the picker when discovery
        # found models, and falling through to the free-text branch would read real stdin.
        with (
            patch.object(wizard, "prompt_for_selection", return_value=None) as picker,
            patch.object(wizard, "print_err"),
        ):
            with pytest.raises(KeyboardInterrupt):
                wizard._require_selection("Select the model:", [("a", "A")])
        assert picker.call_count == 1

    def test_empty_free_text_is_re_prompted(self):
        with (
            patch.object(wizard, "prompt_for_text", side_effect=[None, "some-model"]) as text,
            patch.object(wizard, "print_err"),
            patch.object(wizard, "print_warning"),
        ):
            config = wizard._prompt_models_for_agent("pi", {}, None)
        assert text.call_count == 2
        assert config == {"default_model": "some-model"}

    def test_cancelled_picker_aborts(self):
        with patch.object(wizard, "prompt_for_multi_selection", return_value=None):
            with pytest.raises(KeyboardInterrupt):
                wizard._prompt_models_for_agent("pi", STATE, None)


ANTHROPIC_SERVICE = {
    "name": "main.default.lilly-anthropic",
    "provider_type": "anthropic",
    "targets": ["claude-sonnet-4-6"],
    "allow_all_targets": False,
    "relayed": False,
}
OPENAI_SERVICE = {
    "name": "main.default.openai-mps",
    "provider_type": "openai",
    "targets": ["gpt-5-6"],
    "allow_all_targets": False,
    "relayed": False,
}


class TestCustomModelEntry:
    """The hosted-model pickers let an admin type a model service discovery didn't list."""

    def test_single_agent_can_enter_a_verified_custom_model(self):
        # Selecting the custom row prompts for a path, verifies it exists, and records it so
        # validation won't reject a model outside the discovered inventory.
        with (
            patch.object(wizard, "prompt_for_selection", return_value=wizard._CUSTOM_MODEL),
            patch.object(wizard, "prompt_for_text", return_value="main.aarushi.gpt-5-custom"),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "model_service_exists", return_value=(True, None)) as exists,
        ):
            config = wizard._prompt_models_for_agent("codex", STATE, None)
        assert config == {
            "default_model": "main.aarushi.gpt-5-custom",
            "custom_models": ["main.aarushi.gpt-5-custom"],
        }
        exists.assert_called_once_with(WORKSPACE, "tok", "main.aarushi.gpt-5-custom")

    def test_custom_model_reprompts_until_it_exists(self):
        # A typo shouldn't get baked into a published config — a miss re-asks.
        with (
            patch.object(wizard, "prompt_for_selection", return_value=wizard._CUSTOM_MODEL),
            patch.object(
                wizard, "prompt_for_text", side_effect=["main.typo.model", "main.real.model"]
            ),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "model_service_exists", side_effect=[(False, None), (True, None)]),
            patch.object(wizard, "print_err") as err,
        ):
            config = wizard._prompt_models_for_agent("codex", STATE, None)
        assert config["default_model"] == "main.real.model"
        assert err.called

    def test_multi_select_agent_can_add_several_custom_models(self):
        # opencode/pi carry a model list, so the custom row keeps prompting until the admin declines.
        with (
            patch.object(wizard, "prompt_for_multi_selection", return_value=[wizard._CUSTOM_MODEL]),
            patch.object(
                wizard, "prompt_for_text", side_effect=["main.default.a", "main.default.b"]
            ),
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "prompt_for_selection", return_value="main.default.a"),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "model_service_exists", return_value=(True, None)),
        ):
            config = wizard._prompt_models_for_agent("pi", STATE, None)
        assert config["models"] == ["main.default.a", "main.default.b"]
        assert config["custom_models"] == ["main.default.a", "main.default.b"]
        assert config["default_model"] == "main.default.a"

    def test_short_reason_drops_the_json_body(self):
        # The warning for an inconclusive check should show the status, not the raw error blob.
        assert (
            wizard._short_reason('HTTP 500 Server Error: {"error_code":"INTERNAL"}')
            == "HTTP 500 Server Error"
        )
        assert wizard._short_reason("network error: timed out") == "network error: timed out"
        assert wizard._short_reason(None) == "unknown error"

    def test_unverifiable_custom_model_is_accepted_with_a_warning(self):
        # An inconclusive check (transient error / no token) must not block a possibly-valid model.
        with (
            patch.object(wizard, "prompt_for_selection", return_value=wizard._CUSTOM_MODEL),
            patch.object(wizard, "prompt_for_text", return_value="main.aarushi.maybe"),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "model_service_exists", return_value=(None, "HTTP 500")),
            patch.object(wizard, "print_warning") as warn,
        ):
            config = wizard._prompt_models_for_agent("codex", STATE, None)
        assert config["default_model"] == "main.aarushi.maybe"
        assert warn.called

    def test_picker_offers_all_models_plus_custom(self):
        # Every discovered model is offered (the searchable picker scrolls), with the custom row last.
        models = [f"system.ai.gpt-5-{i}" for i in range(10)]
        state = {**STATE, "codex_models": list(models)}
        offered: list[list[str]] = []

        def fake_sel(prompt, options, **kwargs):
            offered.append([v for v, _ in options])
            return options[0][0]

        with patch.object(wizard, "prompt_for_selection", side_effect=fake_sel):
            wizard._prompt_models_for_agent("codex", state, None)
        rows = offered[0]
        assert rows[-1] == wizard._CUSTOM_MODEL
        # All 10 discovered ids are offered — no truncation to the old top-few cap.
        assert set(models) <= set(rows[:-1])

    def test_claude_family_custom_model_slots_and_validates(self):
        # A custom id chosen for a family lands in that slot, is marked custom, and survives
        # validation even though discovery never listed it.
        candidates = {"opus": ["system.ai.claude-opus-5"]}

        def fake_sel(prompt, options, **kwargs):
            values = [v for v, _ in options]
            if wizard._CUSTOM_MODEL in values:
                return wizard._CUSTOM_MODEL
            return values[0]

        with (
            patch.object(wizard, "_claude_candidates", return_value=candidates),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "prompt_for_text", return_value="main.aarushi.claude-opus-4-5"),
            patch.object(wizard, "get_databricks_token", lambda *a, **k: "tok"),
            patch.object(wizard, "model_service_exists", return_value=(True, None)),
            patch.object(wizard, "print_note"),
            patch.object(wizard, "print_success"),
        ):
            config = wizard._prompt_models_for_agent("claude", STATE, None)
        assert config["models"]["default_opus_model"] == "main.aarushi.claude-opus-4-5"
        assert config["custom_models"] == ["main.aarushi.claude-opus-4-5"]
        assert (
            validate_manifest(
                {
                    "default_agent": "claude",
                    "enabled_agents": {"claude": {"model_config": config}},
                },
                STATE,
            )
            == []
        )


class TestProviderServiceSpinner:
    """The MPS listing is cached per workspace, so only the first agent's lookup does any I/O."""

    SERVICES = [
        {
            "name": "main.j.ant",
            "provider_type": "anthropic",
            "targets": ["claude-opus-5"],
            "allow_all_targets": False,
            "relayed": False,
        }
    ]

    def test_spinner_shows_once_not_once_per_agent(self):
        # The reported symptom: "Checking for model provider services for <agent>..." appeared for
        # every configured agent even though the listing had already been fetched.
        spins: list[str] = []
        cached = {"yes": False}

        def fake_list(workspace, token, **kwargs):
            cached["yes"] = True
            return list(self.SERVICES), None

        def fake_spinner(message):
            spins.append(message)
            from contextlib import nullcontext

            return nullcontext()

        with (
            patch.object(wizard, "list_model_provider_services", side_effect=fake_list),
            patch.object(
                wizard, "has_cached_model_provider_services", side_effect=lambda ws: cached["yes"]
            ),
            patch.object(wizard, "spinner", side_effect=fake_spinner),
            patch.object(wizard, "prompt_for_selection", return_value="databricks"),
        ):
            wizard._select_provider_service("claude", WORKSPACE, "tok")
            wizard._select_provider_service("codex", WORKSPACE, "tok")

        listing_spins = [m for m in spins if "provider service" in m]
        assert len(listing_spins) == 1, listing_spins
        # And it doesn't name an agent, since one lookup covers them all.
        assert "Claude Code" not in listing_spins[0]


class TestProviderServiceSelection:
    def test_agents_without_provider_support_skip_the_prompt(self):
        with patch.object(wizard, "list_model_provider_services") as listing:
            assert wizard._select_provider_service("opencode", WORKSPACE, "token") is None
        assert not listing.called

    def test_feature_disabled_is_silent(self):
        # The common case on most workspaces; a warning here would be noise.
        with (
            patch.object(wizard, "list_model_provider_services", return_value=([], "HTTP 404")),
            patch.object(wizard, "is_model_provider_feature_unavailable", return_value=True),
            patch.object(wizard, "print_warning") as warn,
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert not warn.called

    def test_unexpected_listing_failure_warns(self):
        # Without this the admin silently loses the MPS option with no idea why.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([], "HTTP 403 Forbidden")
            ),
            patch.object(wizard, "is_model_provider_feature_unavailable", return_value=False),
            patch.object(wizard, "print_warning") as warn,
            patch.object(wizard, "print_note"),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert warn.called
        assert "403" in warn.call_args[0][0]

    def test_services_exist_but_none_match_the_agent_explains_why(self):
        # An openai-only workspace offers claude nothing; say so rather than showing no picker.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([OPENAI_SERVICE], None)
            ),
            patch.object(wizard, "print_note") as note,
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert note.called
        assert "API dialect" in note.call_args[0][0]

    def test_choosing_databricks_returns_none(self):
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(wizard, "prompt_for_selection", return_value="databricks"),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None

    def test_choosing_mps_returns_the_whole_service(self):
        # The dict (not just the name) is returned so the model prompt can offer its targets.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["mps", "main.default.lilly-anthropic"],
            ),
            patch.object(wizard, "all_users_can_use_schema", return_value=True),
        ):
            service = wizard._select_provider_service("claude", WORKSPACE, "token")
        assert service == ANTHROPIC_SERVICE

    def test_only_matching_services_are_offered(self):
        with (
            patch.object(
                wizard,
                "list_model_provider_services",
                return_value=([ANTHROPIC_SERVICE, OPENAI_SERVICE], None),
            ),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["mps", "main.default.lilly-anthropic"],
            ) as select,
            patch.object(wizard, "all_users_can_use_schema", return_value=True),
        ):
            wizard._select_provider_service("claude", WORKSPACE, "token")
        offered = [value for value, _ in select.call_args_list[1][0][1]]
        assert offered == ["main.default.lilly-anthropic"]

    def test_relayed_services_are_not_offered_for_claude(self):
        relayed = {
            **ANTHROPIC_SERVICE,
            "name": "main.default.claude-enterprise",
            "targets": [],
            "relayed": True,
        }
        with (
            patch.object(
                wizard,
                "list_model_provider_services",
                return_value=([relayed, ANTHROPIC_SERVICE], None),
            ),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=["mps", "main.default.lilly-anthropic"],
            ) as select,
            patch.object(wizard, "all_users_can_use_schema", return_value=True),
        ):
            service = wizard._select_provider_service("claude", WORKSPACE, "token")
        assert service == ANTHROPIC_SERVICE
        offered = [value for value, _ in select.call_args_list[1][0][1]]
        assert offered == ["main.default.lilly-anthropic"]

    def test_only_relayed_services_falls_back_to_databricks_for_claude(self):
        relayed = {**ANTHROPIC_SERVICE, "targets": [], "relayed": True}
        with (
            patch.object(wizard, "list_model_provider_services", return_value=([relayed], None)),
            patch.object(wizard, "prompt_for_selection") as select,
            patch.object(wizard, "print_note"),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None
        assert not select.called

    def test_warns_when_all_users_lack_schema_access(self):
        # The picked MPS's schema isn't granted to all workspace users, so developers who pull the
        # config may hit "does not have USE_SCHEMA"; warn but still return the service (never block).
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(
                wizard, "prompt_for_selection", side_effect=["mps", "main.default.lilly-anthropic"]
            ),
            patch.object(wizard, "all_users_can_use_schema", return_value=False),
            patch.object(wizard, "print_warning") as warn,
        ):
            service = wizard._select_provider_service("claude", WORKSPACE, "token")
        assert service == ANTHROPIC_SERVICE
        assert warn.called
        assert "main.default" in warn.call_args[0][0]

    def test_no_warning_when_access_check_is_inconclusive(self):
        # A None result (API unreachable / unexpected shape) must not cry wolf.
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(
                wizard, "prompt_for_selection", side_effect=["mps", "main.default.lilly-anthropic"]
            ),
            patch.object(wizard, "all_users_can_use_schema", return_value=None),
            patch.object(wizard, "print_warning") as warn,
        ):
            service = wizard._select_provider_service("claude", WORKSPACE, "token")
        assert service == ANTHROPIC_SERVICE
        assert not warn.called

    def test_cancelling_the_service_picker_returns_none(self):
        with (
            patch.object(
                wizard, "list_model_provider_services", return_value=([ANTHROPIC_SERVICE], None)
            ),
            patch.object(wizard, "prompt_for_selection", side_effect=["mps", None]),
        ):
            assert wizard._select_provider_service("claude", WORKSPACE, "token") is None


class TestProviderServiceModelOptions:
    def test_returns_sorted_targets(self):
        service = {"targets": ["b-model", "a-model"], "allow_all_targets": False}
        assert wizard.provider_service_model_options(service) == ["a-model", "b-model"]

    def test_deduplicates(self):
        service = {"targets": ["m", "m"], "allow_all_targets": False}
        assert wizard.provider_service_model_options(service) == ["m"]

    def test_allow_all_targets_yields_nothing(self):
        service = {"targets": ["m"], "allow_all_targets": True}
        assert wizard.provider_service_model_options(service) == []

    def test_missing_targets_yields_nothing(self):
        assert wizard.provider_service_model_options({}) == []

    def test_malformed_targets_yield_nothing(self):
        assert wizard.provider_service_model_options({"targets": "m"}) == []


# Agents as the wizard configures them: the tier picker must offer these, not the workspace catalog.
CLAUDE_ONLY = {"claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}}


class TestBudgetPolicy:
    def test_no_up_front_gate(self):
        # Running `ug setup spend-tiers` is the consent, so the flow asks no "set up a policy?"
        # question — it goes straight to listing budgets. (The only yes/no it asks is "add another
        # tier?", after a tier is built.)
        with (
            patch.object(wizard, "prompt_yes_no_default") as ask,
            patch.object(wizard, "list_workspace_budgets", return_value=([], "none found")),
            patch.object(wizard, "print_warning_panel"),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        assert not ask.called

    def test_no_budgets_warns_in_a_box_and_skips_the_blurb(self):
        # No attachable budget: show the dead-end warning as a box and don't explain a feature the
        # workspace can't use yet.
        with (
            patch.object(wizard, "list_workspace_budgets", return_value=([], "none found")),
            patch.object(wizard, "print_warning_panel") as warn_box,
            patch.object(wizard, "print_note") as note,
        ):
            assert wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE) is None
        warning = warn_box.call_args.args[0]
        assert "only AI Gateway budgets with hard blocks" in warning
        assert "eligible to be associated with Tiered Spend Policies" in warning
        assert not note.called  # the BUDGET_POLICY_BLURB note is skipped

    def test_no_per_user_block_budgets_warns_and_yields_none(self):
        # Spend routing needs a per-user threshold that hard-blocks; a workspace whose only budgets
        # lack one has nothing usable to attach a policy to.
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": False}]
        with (
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(wizard, "print_warning_panel") as warn_box,
        ):
            assert wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE) is None
        assert warn_box.called

    def test_only_per_user_block_budgets_are_offered(self):
        # The picker hides budgets without a per-user hard block rather than letting the admin pick
        # one that would leave every tier inert or unenforced.
        budgets = [
            {"id": "no-block", "display_name": "email-only", "has_per_user_block": False},
            {"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True},
        ]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "claude", "system.ai.claude-opus-4-8"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        # First selection call is the budget picker; only the per-user budget is offered.
        offered = [value for value, _ in select.call_args_list[0][0][1]]
        assert offered == [BUDGET_ID]
        assert policy is not None and policy["budget_id"] == BUDGET_ID

    def test_percentages_are_stored_as_fractions(self):
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "claude", "system.ai.claude-opus-4-8"],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            # prompt_for_percentage already converts; it returns the fraction.
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        assert policy is not None
        assert policy["budget_id"] == BUDGET_ID
        # The picked budget's name is remembered for the summary.
        assert policy["budget_display_name"] == "eng"
        assert policy["tiers"] == [
            {
                "spending_percentage": 0.8,
                "default_agent": "claude",
                "default_model": "system.ai.claude-opus-4-8",
            }
        ]

    def test_shows_per_user_threshold_and_tier_dollars(self):
        # The admin picks tiers as percentages, so surface the budget's per-user monthly cap and what
        # each percentage works out to in dollars — otherwise a percentage is a number in a vacuum.
        budgets = [
            {
                "id": BUDGET_ID,
                "display_name": "eng",
                "has_per_user_block": True,
                "per_user_threshold": Decimal("500.00"),
            }
        ]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "claude", "system.ai.claude-opus-4-8"],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
            patch.object(wizard, "print_note") as note,
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        notes = " ".join(str(call.args[0]) for call in note.call_args_list)
        assert "$500" in notes  # the per-user monthly cap
        assert "$400" in notes  # 80% of $500

    def test_missing_threshold_skips_the_dollar_hints(self):
        # A budget whose threshold couldn't be read still works; the prompt just omits the dollars.
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "claude", "system.ai.claude-opus-4-8"],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
            patch.object(wizard, "print_note") as note,
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        notes = " ".join(str(call.args[0]) for call in note.call_args_list)
        assert "$" not in notes
        assert policy is not None and policy["tiers"][0]["spending_percentage"] == 0.8

    def test_offers_only_the_models_the_agent_was_configured_with(self):
        # Pi's catalog spans every family, so offering the workspace catalog would present four
        # models it was never given — and a tier naming one of them silently misroutes developers.
        enabled = {
            "pi": {
                "model_config": {
                    "default_model": "system.ai.kimi-k2-6",
                    "models": ["system.ai.kimi-k2-6"],
                }
            }
        }
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "pi", "system.ai.kimi-k2-6"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", enabled, STATE)
        # Third call is the model picker.
        offered = [value for value, _ in select.call_args_list[2][0][1]]
        assert offered == ["system.ai.kimi-k2-6"]

    def test_claude_family_slots_are_flattened_for_the_picker(self):
        enabled = {
            "claude": {
                "model_config": {
                    "default_model": "system.ai.claude-opus-4-8",
                    "models": {
                        "default_opus_model": "system.ai.claude-opus-4-8",
                        "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    },
                }
            }
        }
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "claude", "system.ai.claude-opus-4-8"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", enabled, STATE)
        offered = [value for value, _ in select.call_args_list[2][0][1]]
        assert set(offered) == {"system.ai.claude-opus-4-8", "system.ai.claude-sonnet-4-6"}

    def test_falls_back_to_the_catalog_when_an_agent_lists_nothing(self):
        # An agent configured through a provider service has no enumerable list; better to offer the
        # catalog than nothing at all.
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "gemini", "system.ai.gemini-3-flash"],
            ) as select,
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", {"gemini": {}}, STATE)
        offered = [value for value, _ in select.call_args_list[2][0][1]]
        assert offered == ["system.ai.gemini-3-flash"]

    def test_authored_policy_validates(self):
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[BUDGET_ID, "claude", "system.ai.claude-opus-4-8"],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        manifest = {
            "default_agent": "claude",
            "enabled_agents": CLAUDE_ONLY,
            "budget_policy": policy,
        }
        assert validate_manifest(manifest, STATE) == []

    def test_a_repeated_agent_model_pair_is_rejected_and_re_prompted(self):
        # The highest crossed tier wins, so a second tier on the same agent+model is inert. The loop
        # rejects the repeat and re-asks only the agent/model — the percentage already entered for
        # this tier is kept, not re-prompted.
        two_models = {
            "claude": {
                "model_config": {
                    "default_model": "system.ai.claude-opus-4-8",
                    "models": {
                        "default_opus_model": "system.ai.claude-opus-4-8",
                        "default_sonnet_model": "system.ai.claude-sonnet-4-6",
                    },
                }
            }
        }
        # `has_per_user_block` is required since the budget-threshold gate landed on main: spend
        # routing needs a per-user threshold that hard-blocks, so a budget without one is filtered
        # out and the policy flow returns before the tier loop this test exercises.
        budgets = [{"id": BUDGET_ID, "display_name": "eng", "has_per_user_block": True}]
        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[True, False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(
                wizard,
                "prompt_for_selection",
                side_effect=[
                    BUDGET_ID,
                    # Tier 1: claude / opus.
                    "claude",
                    "system.ai.claude-opus-4-8",
                    # Tier 2 first attempt: claude / opus again — rejected, so just agent/model re-ask.
                    "claude",
                    "system.ai.claude-opus-4-8",
                    # Tier 2 retry: a genuine step-down.
                    "claude",
                    "system.ai.claude-sonnet-4-6",
                ],
            ),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            # Percentage asked once per tier: 0.5 for tier 1, 0.9 for tier 2. The combo retry does
            # not re-ask it.
            patch.object(wizard, "prompt_for_percentage", side_effect=[0.5, 0.9]),
            patch.object(wizard, "print_err") as err,
        ):
            policy = wizard._prompt_budget_policy(WORKSPACE, "token", two_models, STATE)
        assert [(t["default_agent"], t["default_model"]) for t in policy["tiers"]] == [
            ("claude", "system.ai.claude-opus-4-8"),
            ("claude", "system.ai.claude-sonnet-4-6"),
        ]
        assert any("do nothing" in call.args[0] for call in err.call_args_list)


class TestConfiguredModelsForAgent:
    def test_flat_list_plus_default(self):
        agent = {"model_config": {"default_model": "b", "models": ["a", "b"]}}
        assert wizard.configured_models_for_agent(agent) == ["a", "b"]

    def test_claude_slots_are_flattened(self):
        agent = {
            "model_config": {
                "default_model": "opus",
                "models": {"default_opus_model": "opus", "default_sonnet_model": "sonnet"},
            }
        }
        assert set(wizard.configured_models_for_agent(agent)) == {"opus", "sonnet"}

    def test_codex_has_only_a_default(self):
        # CodexModelConfig carries no model list, so the default is the whole set.
        assert wizard.configured_models_for_agent({"model_config": {"default_model": "gpt-5"}}) == [
            "gpt-5"
        ]

    def test_no_model_config_yields_nothing(self):
        assert wizard.configured_models_for_agent({}) == []


class TestSummary:
    def test_lists_claude_family_slots(self, capsys):
        # The one-line default hides which families were configured, which is most of the choice.
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {
                    "model_config": {
                        "default_model": "system.ai.claude-opus-4-8",
                        "models": {
                            "default_opus_model": "system.ai.claude-opus-4-8",
                            "default_haiku_model": "system.ai.claude-haiku-4-5",
                        },
                    }
                }
            },
        }
        wizard._render_summary(WORKSPACE, manifest)
        out = capsys.readouterr().out
        assert "opus" in out and "haiku" in out
        assert "system.ai.claude-haiku-4-5" in out

    def test_lists_a_multi_model_agents_models(self, capsys):
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
        }
        wizard._render_summary(WORKSPACE, manifest)
        assert "system.ai.gpt-5-6" in capsys.readouterr().out

    def test_single_model_agent_needs_no_extra_line(self, capsys):
        manifest = {
            "default_agent": "gemini",
            "enabled_agents": {
                "gemini": {"model_config": {"default_model": "system.ai.gemini-3-flash"}}
            },
        }
        wizard._render_summary(WORKSPACE, manifest)
        out = capsys.readouterr().out
        assert "system.ai.gemini-3-flash" in out
        assert "models:" not in out

    def test_summary_has_no_settings_scope_choice(self, capsys):
        manifest = {
            "default_agent": "codex",
            "enabled_agents": {
                "codex": {"model_config": {"default_model": "system.ai.gpt-5"}},
                "gemini": {"model_config": {"default_model": "system.ai.gemini-3-flash"}},
            },
        }
        wizard._render_summary(WORKSPACE, manifest)
        out = capsys.readouterr().out
        assert "global settings" not in out
        assert "ucode-only" not in out


class TestSetupFromFile:
    def _write(self, tmp_path, payload):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _valid(self):
        return {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
            },
        }

    def test_valid_manifest_is_saved(self, tmp_path):
        path = self._write(tmp_path, self._valid())
        with patch.object(wizard, "load_state", return_value=STATE):
            assert wizard.setup_from_file(str(path)) == 0
        assert managed_config_mod.load_managed_state(WORKSPACE) == self._valid()

    def test_invalid_manifest_returns_1_and_saves_nothing(self, tmp_path):
        path = self._write(tmp_path, {"enabled_agents": {"claude": {}}})
        with patch.object(wizard, "load_state", return_value=STATE):
            assert wizard.setup_from_file(str(path)) == 1
        assert managed_config_mod.load_managed_state(WORKSPACE) is None

    def test_missing_file_is_actionable(self, tmp_path):
        with patch.object(wizard, "load_state", return_value=STATE):
            with pytest.raises(RuntimeError, match="Could not read manifest file"):
                wizard.setup_from_file(str(tmp_path / "nope.json"))

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{oops", encoding="utf-8")
        with patch.object(wizard, "load_state", return_value=STATE):
            with pytest.raises(RuntimeError, match="not valid JSON"):
                wizard.setup_from_file(str(path))

    def test_non_object_json_is_rejected(self, tmp_path):
        path = self._write(tmp_path, ["not", "an", "object"])
        with patch.object(wizard, "load_state", return_value=STATE):
            with pytest.raises(RuntimeError, match="must contain a JSON object"):
                wizard.setup_from_file(str(path))

    def test_unconfigured_workspace_is_actionable(self, tmp_path):
        path = self._write(tmp_path, self._valid())
        with patch.object(wizard, "load_state", return_value={}):
            with pytest.raises(RuntimeError, match="No workspace is configured"):
                wizard.setup_from_file(str(path))


class TestShowCommand:
    def test_reports_nothing_when_unauthored(self):
        with patch.object(wizard, "load_state", return_value={"workspace": WORKSPACE}):
            assert wizard.show_command() == 0

    def test_prints_the_publish_payload(self, capsys):
        manifest = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
            },
        }
        managed_config_mod.save_managed_state(WORKSPACE, manifest)
        with patch.object(wizard, "load_state", return_value={"workspace": WORKSPACE}):
            assert wizard.show_command() == 0
        out = capsys.readouterr().out
        # The proto enum spelling is what `publish` sends, so it must appear verbatim.
        assert "CODING_AGENT_CLAUDE_CODE" in out


class TestSummaryPanel:
    def test_summary_is_boxed(self, capsys):
        # The summary is the one block an admin reads as a whole to check against what they
        # intended, and it lands after a long flow of prompts — so it gets a box rather than loose
        # lines that blend into the preceding output.
        wizard._render_summary(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                },
            },
        )
        out = capsys.readouterr().out
        assert "Configuration summary" in out
        # Rich box-drawing characters: the panel border.
        assert "╭" in out and "╰" in out
        assert "system.ai.claude-opus-5" in out

    def test_a_bracketed_policy_name_survives_the_summary(self, capsys):
        # Rich reads bracketed text as a style tag and renders nothing for it, so an unescaped
        # `[prod] tiered routing` displayed as `tiered routing` — in the block whose whole purpose
        # is confirming what the admin is about to publish workspace-wide.
        wizard._render_summary(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                },
                "budget_policy": {
                    "budget_id": "19165ea4-ff8d-4fbb-b6ce-fc5abe7e1c57",
                    "display_name": "[prod] tiered routing",
                    "tiers": [],
                },
            },
        )
        assert "[prod] tiered routing" in capsys.readouterr().out

    def test_shows_both_budget_and_policy_names(self, capsys):
        # An admin checks the policy against two distinct things: which budget it tracks and what the
        # policy itself is called. The summary must surface both, not collapse to one.
        wizard._render_summary(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                },
                "budget_policy": {
                    "budget_id": "19165ea4-ff8d-4fbb-b6ce-fc5abe7e1c57",
                    "budget_display_name": "eng-budget",
                    "display_name": "tiered routing",
                    "tiers": [],
                },
            },
        )
        out = capsys.readouterr().out
        assert "eng-budget" in out
        assert "tiered routing" in out

    def test_falls_back_to_budget_id_without_a_budget_name(self, capsys):
        # `--from-file` and server-read manifests carry no `budget_display_name`, so the budget id is
        # all there is to show.
        wizard._render_summary(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                },
                "budget_policy": {
                    "budget_id": "19165ea4-ff8d-4fbb-b6ce-fc5abe7e1c57",
                    "display_name": "tiered routing",
                    "tiers": [],
                },
            },
        )
        assert "19165ea4-ff8d-4fbb-b6ce-fc5abe7e1c57" in capsys.readouterr().out


class TestCancelledPromptsAbort:
    """A dismissed prompt must abort, not re-ask an input that can't answer."""

    def test_require_selection_aborts_when_the_picker_is_dismissed(self):
        # questionary's Question.ask catches KeyboardInterrupt and returns None (v2.1.1), so Ctrl-C
        # is indistinguishable from an empty submission here. Re-asking looped forever.
        with patch.object(wizard, "prompt_for_selection", return_value=None) as sel:
            with pytest.raises(KeyboardInterrupt):
                wizard._require_selection("pick", [("a", "A")])
        assert sel.call_count == 1

    def test_require_text_asks_for_a_required_answer(self):
        # `required=True` is what makes closed stdin raise instead of returning None; without it a
        # piped/CI run spins re-asking an exhausted stream.
        with patch.object(wizard, "prompt_for_text", return_value="m") as text:
            assert wizard._require_text("Default model") == "m"
        assert text.call_args.kwargs.get("required") is True

    def test_require_text_aborts_on_closed_stdin(self):
        with patch("ucode.ui.console.input", side_effect=EOFError):
            with pytest.raises(KeyboardInterrupt):
                wizard._require_text("Default model")


class TestClaudeCandidatesStayValidatable:
    """Whatever the Claude prompts offer, `validate_manifest` must accept."""

    def _manifest(self, model: str) -> dict:
        return {
            "default_agent": "claude",
            "enabled_agents": {"claude": {"model_config": {"default_model": model}}},
        }

    def test_listing_path_caches_so_older_versions_validate(self):
        # The unbucketed listing widens the candidates past `claude_models`, so it must also cache
        # them — otherwise picking an older Opus is rejected at the end of the flow.
        state = {
            "workspace": "https://ws.example.com",
            "claude_models": {"opus": "system.ai.claude-opus-5"},
        }
        with (
            patch.object(wizard, "get_databricks_token", return_value="t"),
            patch.object(
                wizard,
                "discover_claude_models_unbucketed",
                return_value=(["system.ai.claude-opus-5", "system.ai.claude-opus-4-8"], None),
            ),
        ):
            candidates = wizard._claude_candidates(state)
        offered = [m for models in candidates.values() for m in models]
        assert "system.ai.claude-opus-4-8" in offered
        for model in offered:
            assert validate_manifest(self._manifest(model), state) == []

    def test_fallback_path_only_offers_what_already_validates(self):
        # The fallback caches nothing, so it must not offer anything beyond `claude_models`.
        state = {
            "workspace": "https://ws.example.com",
            "claude_models": {"opus": "system.ai.claude-opus-5"},
        }
        with (
            patch.object(wizard, "get_databricks_token", return_value="t"),
            patch.object(
                wizard, "discover_claude_models_unbucketed", side_effect=RuntimeError("boom")
            ),
        ):
            candidates = wizard._claude_candidates(state)
        assert "all_claude_models" not in state
        for models in candidates.values():
            for model in models:
                assert validate_manifest(self._manifest(model), state) == []


class TestSearchablePickers:
    """Long lists (models, provider services, budgets) filter as you type."""

    def test_model_pickers_are_searchable(self):
        seen: list[dict] = []

        def fake_multi(prompt, options, preselected=None, **kwargs):
            seen.append(kwargs)
            return [options[0][0]]

        with patch.object(wizard, "prompt_for_multi_selection", side_effect=fake_multi):
            wizard._require_multi_selection("pick", [("a", "a"), ("b", "b")])
        assert seen[0].get("searchable") is True

    def test_single_select_pickers_are_searchable(self):
        seen: list[dict] = []

        def fake_sel(prompt, options, **kwargs):
            seen.append(kwargs)
            return options[0][0]

        with patch.object(wizard, "prompt_for_selection", side_effect=fake_sel):
            wizard._require_selection("pick", [("a", "a"), ("b", "b")])
        assert seen[0].get("searchable") is True

    def test_budget_and_tier_pickers_are_searchable(self):
        budgets = [{"id": "budget-1", "display_name": "eng", "has_per_user_block": True}]
        searchable_prompts: list[str] = []

        def fake_sel(prompt, options, **kwargs):
            if kwargs.get("searchable"):
                searchable_prompts.append(prompt)
            if "budget" in prompt:
                return "budget-1"
            if "agent" in prompt:
                return "claude"
            return "system.ai.claude-opus-4-8"

        with (
            patch.object(wizard, "prompt_yes_no_default", side_effect=[False]),
            patch.object(wizard, "list_workspace_budgets", return_value=(budgets, None)),
            patch.object(wizard, "prompt_for_selection", side_effect=fake_sel),
            patch.object(wizard, "prompt_for_text", return_value="tiered"),
            patch.object(wizard, "prompt_for_percentage", return_value=0.8),
        ):
            wizard._prompt_budget_policy(WORKSPACE, "token", CLAUDE_ONLY, STATE)
        # Both the budget list and the tier's model list filter as you type.
        assert any("budget" in p for p in searchable_prompts), searchable_prompts
        assert any("model" in p for p in searchable_prompts), searchable_prompts


# A minimal authored manifest (agents + models only), the shape `ug setup` now writes.
AGENTS_ONLY = {
    "default_agent": "claude",
    "enabled_agents": {"claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}},
}


class TestCarryForwardSections:
    def test_all_optional_sections_survive_a_rerun(self):
        previous = {
            **AGENTS_ONLY,
            "mcp_servers": [{"name": "system.ai.github", "type": "mcp-service"}],
            "skills": {"names": ["main.default"]},
            "tracing_table": "main.default.traces",
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.8,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    }
                ],
            },
        }
        manifest = dict(AGENTS_ONLY)
        wizard._carry_forward_sections(previous, manifest)
        assert manifest["mcp_servers"] == previous["mcp_servers"]
        assert manifest["skills"] == previous["skills"]
        assert manifest["tracing_table"] == previous["tracing_table"]
        assert manifest["budget_policy"] == previous["budget_policy"]

    def test_empty_previous_adds_nothing(self):
        manifest = dict(AGENTS_ONLY)
        wizard._carry_forward_sections({}, manifest)
        assert set(manifest) == set(AGENTS_ONLY)

    def test_budget_policy_naming_a_dropped_agent_is_left_out_with_a_warning(self):
        # The admin re-ran `setup` and de-selected codex; the saved policy still routes to it, which
        # would fail `validate_manifest` and block the whole save. Drop just the policy, loudly.
        previous = {
            **AGENTS_ONLY,
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.9,
                        "default_agent": "codex",
                        "default_model": "system.ai.gpt-5",
                    }
                ],
            },
        }
        manifest = dict(AGENTS_ONLY)
        with patch.object(wizard, "print_warning") as warn:
            wizard._carry_forward_sections(previous, manifest)
        assert "budget_policy" not in manifest
        assert warn.called
        # The rest of the manifest is untouched and still valid.
        assert validate_manifest(manifest, None) == []


class TestNextSteps:
    def test_marks_configured_and_unconfigured_sections(self, capsys):
        manifest = {**AGENTS_ONLY, "skills": {"names": ["main.default"]}}
        wizard._print_next_steps(manifest)
        out = capsys.readouterr().out
        assert "ug setup mcps" in out
        assert "ug setup skills" in out
        assert "ug setup spend-tiers" in out
        assert "ug publish" in out

    def test_dry_run_says_nothing_was_saved(self, capsys, monkeypatch):
        monkeypatch.setattr(config_io_mod, "_dry_run", True)
        wizard._print_next_steps(AGENTS_ONLY)
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "ug publish" not in out


class TestSectionCommands:
    """The `ug setup mcps` / `skills` / `spend-tiers` section commands."""

    @staticmethod
    def _admin(**overrides):
        """Patch the auth/admin boundary the section commands resolve through."""
        defaults = {
            "load_state": lambda: {"workspace": WORKSPACE, "profile": "p", **STATE},
            "ensure_databricks_auth": lambda *a, **k: None,
            "get_databricks_token": lambda *a, **k: "tok",
            "is_workspace_admin": lambda *a, **k: True,
            # Decline the end-of-section "publish now?" offer so a section run saves the draft without
            # trying to publish; the publish path is exercised in TestPublishCommand.
            "prompt_yes_no_default": lambda *a, **k: False,
        }
        defaults.update(overrides)
        return [patch.object(wizard, name, value) for name, value in defaults.items()]

    def _run(self, fn, *, admin_overrides=None, **patches):
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in self._admin(**(admin_overrides or {})):
                stack.enter_context(p)
            for name, value in patches.items():
                stack.enter_context(patch.object(wizard, name, value))
            return fn()

    def test_mcp_requires_an_authored_config(self):
        # No manifest on disk → the command can't edit a section that doesn't exist.
        with pytest.raises(RuntimeError, match="ug setup"):
            self._run(wizard.setup_mcp_command)

    def test_mcp_requires_enabled_agents(self):
        # A launch stores `{}` to mean "no managed config"; that must not count as authored.
        managed_config_mod.save_managed_state(WORKSPACE, {})
        with pytest.raises(RuntimeError, match="ug setup"):
            self._run(wizard.setup_mcp_command)

    def test_mcp_writes_only_its_section(self):
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        servers = [{"name": "system.ai.github", "type": "mcp-service"}]
        # The picker (imported lazily inside the command) registers servers into local state; fake
        # that by having the before/after reads bracket a change.
        reads = iter([[], servers])
        with (
            patch("ucode.mcp.configure_mcp_command", return_value=0) as picker,
            patch.object(wizard, "_mcp_servers_from_state", side_effect=lambda *_: next(reads)),
        ):
            code = self._run(wizard.setup_mcp_command)
        assert code == 0
        assert picker.call_args.kwargs == {"exclude_sources": {"apps"}}
        saved = managed_config_mod.load_managed_state(WORKSPACE)
        assert saved["mcp_servers"] == servers
        assert saved["enabled_agents"] == AGENTS_ONLY["enabled_agents"]

    def test_mcp_cancel_is_a_no_op(self):
        # Picker cancelled / nothing changed → the section is left exactly as it was.
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        with (
            patch("ucode.mcp.configure_mcp_command", return_value=0),
            patch.object(wizard, "_mcp_servers_from_state", return_value=[]),
            patch.object(wizard, "save_managed_state") as save,
        ):
            code = self._run(wizard.setup_mcp_command)
        assert code == 0
        assert not save.called

    def test_mcp_carries_forward_preregistered_servers(self):
        # An admin who ran `ug configure mcp` first arrives with those servers already registered,
        # so the picker leaves local state unchanged (before == after). The manifest doesn't carry them
        # yet, so `setup mcps` must still save them rather than report "no changes" and drop them.
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        servers = [{"name": "system.ai.github", "type": "mcp-service"}]
        with (
            patch("ucode.mcp.configure_mcp_command", return_value=0),
            patch.object(wizard, "_mcp_servers_from_state", return_value=servers),
        ):
            code = self._run(wizard.setup_mcp_command)
        assert code == 0
        assert managed_config_mod.load_managed_state(WORKSPACE)["mcp_servers"] == servers

    def test_mcp_not_admin_raises(self):
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        with pytest.raises(RuntimeError, match="not an admin"):
            self._run(
                wizard.setup_mcp_command,
                admin_overrides={"is_workspace_admin": lambda *a, **k: False},
            )

    def test_skills_location_bypasses_the_prompt(self):
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        with (
            patch("ucode.mcp.configure_skills_mcp_command", return_value=0) as configure,
            patch.object(wizard, "_skill_names_from_state", return_value=["main.default"]),
            patch.object(wizard, "prompt_for_text") as prompt,
        ):
            code = self._run(lambda: wizard.setup_skills_command(["main.default"]))
        assert code == 0
        assert not prompt.called
        configure.assert_called_once_with(["main.default"])
        assert managed_config_mod.load_managed_state(WORKSPACE)["skills"] == {
            "names": ["main.default"]
        }

    def test_skills_blank_answer_writes_nothing(self):
        # A blank answer must not delegate: `configure_skills_mcp_command([])` is not a no-op.
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        with (
            patch("ucode.mcp.configure_skills_mcp_command") as configure,
            patch.object(wizard, "prompt_for_text", return_value=""),
            patch.object(wizard, "save_managed_state") as save,
        ):
            code = self._run(wizard.setup_skills_command)
        assert code == 0
        assert not configure.called
        assert not save.called

    def test_budget_policy_offers_only_the_manifests_agents(self):
        managed_config_mod.save_managed_state(WORKSPACE, AGENTS_ONLY)
        captured = {}

        def fake_prompt(workspace, token, enabled_agents, state, **kwargs):
            captured["agents"] = enabled_agents
            return None  # decline / dead end → leave the policy unchanged

        with patch.object(wizard, "_prompt_budget_policy", side_effect=fake_prompt):
            code = self._run(wizard.setup_budget_policy_command)
        assert code == 0
        assert captured["agents"] == AGENTS_ONLY["enabled_agents"]

    def test_budget_policy_none_leaves_existing_untouched(self):
        # A transient budget-listing failure returns None; it must never delete a saved policy.
        seeded = {
            **AGENTS_ONLY,
            "budget_policy": {
                "budget_id": BUDGET_ID,
                "tiers": [
                    {
                        "spending_percentage": 0.8,
                        "default_agent": "claude",
                        "default_model": "system.ai.claude-opus-4-8",
                    }
                ],
            },
        }
        managed_config_mod.save_managed_state(WORKSPACE, seeded)
        with patch.object(wizard, "_prompt_budget_policy", return_value=None):
            code = self._run(wizard.setup_budget_policy_command)
        assert code == 0
        assert (
            managed_config_mod.load_managed_state(WORKSPACE)["budget_policy"]
            == seeded["budget_policy"]
        )


class TestSetupHelp:
    def test_lists_every_setup_command(self, capsys):
        wizard.setup_help_command()
        out = capsys.readouterr().out
        for command in (
            "ug setup",
            "ug setup mcps",
            "ug setup skills",
            "ug setup spend-tiers",
            "ug setup show",
            "ug publish",
        ):
            assert command in out


class TestPublishDiff:
    def test_lists_added_removed_and_changed(self, capsys):
        existing = {
            "name": "cfg/1",
            **AGENTS_ONLY,
            "mcp_servers": [{"name": "system.ai.slack", "type": "mcp-service"}],
        }
        incoming = {
            "default_agent": "claude",
            "enabled_agents": {
                "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-9"}}
            },
            "mcp_servers": [{"name": "system.ai.github", "type": "mcp-service"}],
        }
        changed = wizard._render_config_diff(existing, incoming, WORKSPACE)
        out = capsys.readouterr().out
        assert changed is True
        assert "CHANGE" in out and "claude-opus-4-8" in out and "claude-opus-4-9" in out
        assert "ADD" in out and "system.ai.github" in out  # added server
        assert "DELETE" in out and "system.ai.slack" in out  # removed server

    def test_identical_configs_report_no_change(self, capsys):
        assert wizard._render_config_diff(AGENTS_ONLY, AGENTS_ONLY, WORKSPACE) is False

    def test_display_name_change_is_detected(self, capsys):
        existing = {"display_name": "old-name", **AGENTS_ONLY}
        incoming = {"display_name": "new-name", **AGENTS_ONLY}
        changed = wizard._render_config_diff(existing, incoming, WORKSPACE)
        out = capsys.readouterr().out
        assert changed is True
        assert "old-name" in out and "new-name" in out


class TestPublishCommand:
    MANIFEST = {
        "default_agent": "claude",
        "enabled_agents": {
            "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}}
        },
    }

    @staticmethod
    def _patches(**overrides):
        """The network/auth boundary `publish_command` sits behind, with per-test overrides."""
        defaults = {
            "load_state": lambda: {"workspace": WORKSPACE, "profile": "p", **STATE},
            "ensure_databricks_auth": lambda *a, **k: None,
            "get_databricks_token": lambda *a, **k: "tok",
            "is_workspace_admin": lambda *a, **k: True,
            "get_managed_config": lambda *a, **k: (None, None),
            "create_coding_agent_config": lambda *a, **k: (
                {"name": "coding-agent-configs/new"},
                None,
            ),
            "update_coding_agent_config": lambda *a, **k: (
                {"name": "coding-agent-configs/old"},
                None,
            ),
            "prompt_yes_no_default": lambda *a, **k: True,
        }
        defaults.update(overrides)
        return [patch.object(wizard, name, value) for name, value in defaults.items()]

    def _run(self, *, yes=False, file_path=None, **overrides):
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in self._patches(**overrides):
                stack.enter_context(p)
            return wizard.publish_command(file_path=file_path, yes=yes)

    @staticmethod
    def _config_file(tmp_path, manifest, *, workspace=WORKSPACE, spec_version=1, **extra):
        config = serialize_managed_config(manifest)
        config.pop("name", None)
        payload = {"workspace": workspace, "spec_version": spec_version, **config, **extra}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_unauthored_config_is_an_actionable_error(self):
        with patch.object(wizard, "load_state", return_value={"workspace": WORKSPACE}):
            with pytest.raises(RuntimeError, match="ug setup"):
                wizard.publish_command()

    def test_creates_when_no_config_exists(self):
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        created = {}

        def fake_create(workspace, token, payload):
            created.update(workspace=workspace, payload=payload)
            return {"name": "coding-agent-configs/new"}, None

        assert self._run(create_coding_agent_config=fake_create) == 0
        assert created["workspace"] == WORKSPACE
        # What goes over the wire is proto-JSON, not ug's manifest shape.
        assert created["payload"]["default_agent"] == "CODING_AGENT_CLAUDE_CODE"

    def test_updates_in_place_when_a_config_exists(self):
        # Delete-then-create would leave the workspace with no config if the create failed, so an
        # existing config must be PATCHed rather than replaced.
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        existing = {"name": "coding-agent-configs/abc", "enabled_agents": {"codex": {}}}
        updated = {}
        created = {"called": False}

        def fake_update(workspace, token, name, payload):
            updated.update(name=name, payload=payload)
            return {"name": name}, None

        def fake_create(*a, **k):
            created["called"] = True
            return {}, None

        assert (
            self._run(
                get_managed_config=lambda *a, **k: (existing, None),
                update_coding_agent_config=fake_update,
                create_coding_agent_config=fake_create,
            )
            == 0
        )
        assert updated["name"] == "coding-agent-configs/abc"
        assert created["called"] is False

    def test_no_publish_when_the_published_config_already_matches(self):
        # Publishing a config identical to what's live is a no-op; say so and skip the write rather
        # than PATCH the same bytes back.
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        # What's live is the manifest normalized the same way `publish` will send it.
        existing = {
            "name": "coding-agent-configs/abc",
            **managed_config_mod.normalize_managed_config(serialize_managed_config(self.MANIFEST)),
        }
        updated = {"called": False}

        def fake_update(*a, **k):
            updated["called"] = True
            return {}, None

        assert (
            self._run(
                get_managed_config=lambda *a, **k: (existing, None),
                update_coding_agent_config=fake_update,
            )
            == 0
        )
        assert updated["called"] is False

    def test_invalid_manifest_is_not_published(self):
        managed_config_mod.save_managed_state(
            WORKSPACE, {"default_agent": "codex", "enabled_agents": {"claude": {}}}
        )
        created = {"called": False}

        def fake_create(*a, **k):
            created["called"] = True
            return {}, None

        with pytest.raises(RuntimeError, match="not valid"):
            self._run(create_coding_agent_config=fake_create)
        assert created["called"] is False

    def test_an_older_family_version_the_wizard_offered_still_publishes(self):
        # `setup` offers every version of a Claude family, but `claude_models` keeps only the newest
        # per family and the wizard's `all_claude_models` stash is never persisted — so a separate
        # `publish` process used to reject a model it had just offered:
        #   claude: model 'system.ai.claude-opus-4-1' is not available on this workspace.
        # `publish` re-fetches the full listing rather than trusting what `setup` left in state.
        managed_config_mod.save_managed_state(
            WORKSPACE,
            {
                "default_agent": "claude",
                "enabled_agents": {
                    "claude": {
                        "model_config": {
                            "default_model": "system.ai.claude-opus-4-1",
                            "models": {"default_opus_model": "system.ai.claude-opus-4-1"},
                        }
                    }
                },
            },
        )
        published: dict = {}

        def fake_create(workspace, token, payload):
            published["payload"] = payload
            return {"name": "coding-agent-configs/new"}, None

        # State carries only the newest Opus, as a fresh `load_state()` would.
        narrow = {"workspace": WORKSPACE, "profile": "p", "claude_models": {"opus": "newest"}}
        assert (
            self._run(
                load_state=lambda: dict(narrow),
                discover_claude_models_unbucketed=lambda *a, **k: (
                    ["system.ai.claude-opus-4-1", "newest"],
                    None,
                ),
                create_coding_agent_config=fake_create,
            )
            == 0
        )
        assert published, "the manifest should have been published"

    def test_a_failed_inventory_fetch_does_not_block_publishing(self):
        # The re-fetch is best-effort: a transient listing failure must not turn into a refusal to
        # publish a manifest that validates against what state already knows.
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        published: dict = {}

        def fake_create(workspace, token, payload):
            published["payload"] = payload
            return {"name": "coding-agent-configs/new"}, None

        assert (
            self._run(
                discover_claude_models_unbucketed=lambda *a, **k: ([], "HTTP 500"),
                create_coding_agent_config=fake_create,
            )
            == 0
        )
        assert published

    def test_declining_the_prompt_publishes_nothing(self):
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        created = {"called": False}

        def fake_create(*a, **k):
            created["called"] = True
            return {}, None

        code = self._run(
            prompt_yes_no_default=lambda *a, **k: False, create_coding_agent_config=fake_create
        )
        assert code == 1
        assert created["called"] is False

    def test_yes_skips_the_prompt(self):
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)

        def refuse(*a, **k):
            raise AssertionError("--yes must not prompt")

        assert self._run(yes=True, prompt_yes_no_default=refuse) == 0

    def test_non_admin_is_rejected_before_publishing(self):
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        created = {"called": False}

        def fake_create(*a, **k):
            created["called"] = True
            return {}, None

        with pytest.raises(RuntimeError, match="not an admin"):
            self._run(
                is_workspace_admin=lambda *a, **k: False, create_coding_agent_config=fake_create
            )
        assert created["called"] is False

    def test_unreadable_existing_config_refuses_to_publish(self):
        # Publishing without knowing whether a config exists risks silently overwriting one.
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        created = {"called": False}

        def fake_create(*a, **k):
            created["called"] = True
            return {}, None

        with pytest.raises(RuntimeError, match="Refusing to publish"):
            self._run(
                get_managed_config=lambda *a, **k: (None, "HTTP 500 Server Error"),
                create_coding_agent_config=fake_create,
            )
        assert created["called"] is False

    def test_feature_disabled_read_uses_the_shared_blocking_message(self):
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        created = {"called": False}

        def fake_create(*a, **k):
            created["called"] = True
            return {}, None

        reason = 'HTTP 404 Not Found: {"error_code":"FEATURE_DISABLED"}'
        with pytest.raises(RuntimeError) as exc_info:
            self._run(
                get_managed_config=lambda *a, **k: (None, reason),
                create_coding_agent_config=fake_create,
            )
        assert str(exc_info.value) == wizard.CODING_AGENT_CONFIGS_DISABLED_MESSAGE
        assert "FEATURE_DISABLED" not in str(exc_info.value)
        assert created["called"] is False

    def test_existing_config_without_a_resource_name_is_an_error(self):
        managed_config_mod.save_managed_state(WORKSPACE, self.MANIFEST)
        with pytest.raises(RuntimeError, match="resource name"):
            self._run(get_managed_config=lambda *a, **k: ({"enabled_agents": {}}, None))

    def test_file_input_creates_and_sends_spec_version_without_workspace(self, tmp_path):
        path = self._config_file(tmp_path, self.MANIFEST)
        created = {}

        def fake_create(workspace, token, payload):
            created.update(workspace=workspace, payload=payload)
            return {"name": "coding-agent-configs/new"}, None

        assert self._run(file_path=path, create_coding_agent_config=fake_create) == 0
        assert created["workspace"] == WORKSPACE
        assert created["payload"]["spec_version"] == 1
        assert "workspace" not in created["payload"]
        assert "name" not in created["payload"]
        assert created["payload"]["default_agent"] == "CODING_AGENT_CLAUDE_CODE"

    def test_file_input_updates_in_place_and_sends_spec_version(self, tmp_path):
        path = self._config_file(tmp_path, self.MANIFEST)
        existing = {"name": "coding-agent-configs/abc", "enabled_agents": {"codex": {}}}
        updated = {}

        def fake_update(workspace, token, name, payload):
            updated.update(name=name, payload=payload)
            return {"name": name}, None

        assert (
            self._run(
                file_path=path,
                get_managed_config=lambda *a, **k: (existing, None),
                update_coding_agent_config=fake_update,
            )
            == 0
        )
        assert updated["name"] == "coding-agent-configs/abc"
        assert updated["payload"]["spec_version"] == 1
        assert "workspace" not in updated["payload"]

    def test_file_workspace_mismatch_aborts_before_auth_or_mutation(self, tmp_path):
        path = self._config_file(tmp_path, self.MANIFEST, workspace="https://other.example.com")
        called = {"auth": False, "create": False}

        def fake_auth(*a, **k):
            called["auth"] = True

        def fake_create(*a, **k):
            called["create"] = True
            return {}, None

        with pytest.raises(RuntimeError, match="configured workspace"):
            self._run(
                file_path=path,
                ensure_databricks_auth=fake_auth,
                create_coding_agent_config=fake_create,
            )
        assert called == {"auth": False, "create": False}

    def test_file_bad_spec_version_aborts_before_auth(self, tmp_path):
        path = self._config_file(tmp_path, self.MANIFEST, spec_version=2)
        called = {"auth": False}

        def fake_auth(*a, **k):
            called["auth"] = True

        with pytest.raises(RuntimeError, match="spec_version"):
            self._run(file_path=path, ensure_databricks_auth=fake_auth)
        assert called["auth"] is False

    def test_missing_file_is_actionable(self, tmp_path):
        with pytest.raises(RuntimeError, match="No config file"):
            self._run(file_path=str(tmp_path / "absent.json"))

    def test_no_file_publishes_a_hand_entered_custom_model(self):
        managed_config_mod.save_managed_state(
            WORKSPACE,
            {
                "default_agent": "codex",
                "enabled_agents": {
                    "codex": {
                        "model_config": {
                            "default_model": "main.custom.model",
                            "custom_models": ["main.custom.model"],
                        }
                    }
                },
            },
        )
        created = {"called": False}

        def fake_create(*a, **k):
            created["called"] = True
            return {"name": "coding-agent-configs/new"}, None

        assert self._run(create_coding_agent_config=fake_create) == 0
        assert created["called"] is True


class TestPublishFailureMessages:
    """The server's error codes, turned into something an admin can act on."""

    def test_feature_disabled_uses_the_shared_message(self):
        message = wizard._explain_publish_failure(
            'HTTP 400 Bad Request: {"error_code":"FEATURE_DISABLED","message":"..."}'
        )
        assert message == wizard.CODING_AGENT_CONFIGS_DISABLED_MESSAGE
        assert "`ug configure`" in message

    def test_permission_denied_says_admin_is_required(self):
        message = wizard._explain_publish_failure(
            'HTTP 403 Forbidden: {"error_code":"PERMISSION_DENIED"}'
        )
        assert "workspace admin" in message

    def test_invalid_parameter_value_is_passed_through_verbatim(self):
        # The server names the offending field, which is more useful than any paraphrase.
        reason = (
            'HTTP 400 Bad Request: {"error_code":"INVALID_PARAMETER_VALUE",'
            '"message":"budget_policy.tiers[0].spending_percentage must be between 0 and 1"}'
        )
        message = wizard._explain_publish_failure(reason)
        assert "budget_policy.tiers[0].spending_percentage" in message

    def test_unknown_failure_still_surfaces_the_reason(self):
        message = wizard._explain_publish_failure("network error: timed out")
        assert "timed out" in message


class TestCliWiring:
    def test_setup_is_registered(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "setup" in result.output

    def test_setup_help_lists_from_file(self):
        # Assert on the declared option rather than the rendered help text: Rich ellipsizes option
        # names to fit the terminal ("--fro…" below ~40 columns), and CI runners report no width, so
        # grepping `--from-file` out of the output fails there while passing on a wide local one.
        group = typer.main.get_command(app).commands["setup"]  # type: ignore[attr-defined]
        declared = {opt for param in group.params for opt in param.opts}
        assert "--from-file" in declared
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0

    def test_setup_show_is_registered(self):
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output

    def test_publish_is_registered(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "publish" in result.output

    def test_publish_declares_yes_and_no_dry_run(self):
        # `--dry-run` was removed: publish always validates before publishing, so a separate
        # validate-only mode is redundant. Asserted on declared options rather than rendered help,
        # which Rich ellipsizes at narrow widths (see test_setup_help_lists_from_file).
        command = typer.main.get_command(app).commands["publish"]  # type: ignore[attr-defined]
        declared = {opt for param in command.params for opt in param.opts}
        assert "--yes" in declared
        assert "--dry-run" not in declared

    def test_publish_declares_file_option(self):
        command = typer.main.get_command(app).commands["publish"]  # type: ignore[attr-defined]
        declared = {opt for param in command.params for opt in param.opts}
        assert "--file" in declared
        assert "-f" in declared

    def test_publish_file_flag_is_forwarded(self):
        for flag in ("-f", "--file"):
            with (
                patch("ucode.cli.install_databricks_cli"),
                patch.object(cli_mod, "publish_command", return_value=0) as publish,
            ):
                runner.invoke(app, ["publish", flag, "/tmp/cfg.json"])
            assert publish.call_args.kwargs["file_path"] == "/tmp/cfg.json"

    def test_publish_error_exits_nonzero_with_a_message(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch.object(
                cli_mod, "publish_command", side_effect=RuntimeError("no config authored")
            ),
        ):
            result = runner.invoke(app, ["publish"])
        assert result.exit_code == 1

    def test_successful_publish_exits_zero(self):
        # Same trap as `setup`: `typer.Exit` subclasses RuntimeError, so raising it inside the
        # command's try block would report success as "ERROR 0".
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch.object(cli_mod, "publish_command", return_value=0),
        ):
            result = runner.invoke(app, ["publish"])
        assert result.exit_code == 0
        assert "ERROR" not in result.output

    def test_successful_setup_exits_zero(self):
        # `typer.Exit` subclasses RuntimeError, so a success code must not be caught and reported
        # as an error by the command's own RuntimeError handler.
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=0) as setup,
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert setup.called
        assert "ERROR" not in _out(result)

    def test_nonzero_setup_propagates(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=1),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1

    def test_runtime_error_is_reported_and_exits_1(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", side_effect=RuntimeError("you are not an admin")),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1
        assert "not an admin" in _out(result)

    def test_interrupt_exits_130(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", side_effect=KeyboardInterrupt),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 130

    def test_from_file_is_forwarded(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_command", return_value=0) as setup,
        ):
            runner.invoke(app, ["setup", "--from-file", "/tmp/x.json"])
        assert setup.call_args.kwargs["from_file"] == "/tmp/x.json"

    def test_show_exits_zero(self):
        with patch("ucode.cli.show_command", return_value=0):
            result = runner.invoke(app, ["setup", "show"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        ("command", "target"),
        [
            ("mcps", "setup_mcp_command"),
            ("skills", "setup_skills_command"),
            ("spend-tiers", "setup_budget_policy_command"),
        ],
    )
    def test_section_subcommands_are_registered_and_called(self, command, target):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch(f"ucode.cli.{target}", return_value=0) as fn,
        ):
            result = runner.invoke(app, ["setup", command])
        assert result.exit_code == 0
        assert fn.called
        assert "ERROR" not in _out(result)

    def test_setup_skills_declares_location(self):
        group = typer.main.get_command(app).commands["setup"]  # type: ignore[attr-defined]
        skills = group.commands["skills"]  # type: ignore[attr-defined]
        declared = {opt for param in skills.params for opt in param.opts}
        assert "--location" in declared

    def test_setup_skills_location_is_parsed_to_a_list(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_skills_command", return_value=0) as fn,
        ):
            runner.invoke(app, ["setup", "skills", "--location", "main.a,main.b"])
        assert fn.call_args.args[0] == ["main.a", "main.b"]

    def test_setup_help_needs_no_auth(self):
        # `ug setup help` reads the local draft only — it must not shell out to install the CLI.
        with (
            patch("ucode.cli.install_databricks_cli") as install,
            patch("ucode.cli.setup_help_command", return_value=0) as fn,
        ):
            result = runner.invoke(app, ["setup", "help"])
        assert result.exit_code == 0
        assert fn.called
        assert not install.called

    def test_section_command_runtime_error_exits_1(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_mcp_command", side_effect=RuntimeError("run `ug setup` first")),
        ):
            result = runner.invoke(app, ["setup", "mcps"])
        assert result.exit_code == 1
        assert "ug setup" in _out(result)

    def test_section_command_interrupt_exits_130(self):
        with (
            patch("ucode.cli.install_databricks_cli"),
            patch("ucode.cli.setup_budget_policy_command", side_effect=KeyboardInterrupt),
        ):
            result = runner.invoke(app, ["setup", "spend-tiers"])
        assert result.exit_code == 130


def _out(result) -> str:
    """CliRunner output with stderr folded in, since print_err writes to a stderr console."""
    return result.output + (result.stderr if result.stderr_bytes else "")


class TestSetupExcludedAgents:
    def test_omp_is_excluded_from_the_setup_picker(self):
        # The server proto has no omp agent variant, so the managed-setup
        # picker must never offer it even when the workspace serves omp's
        # models (the picker's `available` filter honors this set).
        assert "omp" in wizard.SETUP_EXCLUDED_AGENTS
        assert "gemini" in wizard.SETUP_EXCLUDED_AGENTS

    def test_omp_would_otherwise_pass_the_gateway_gate(self):
        # Guard against a vacuous exclusion: with models discovered, omp
        # clears the gateway check, so the exclusion set is what keeps it
        # out of the picker.
        from ucode.agents import check_gateway_endpoint

        assert check_gateway_endpoint(STATE, "omp") is True
