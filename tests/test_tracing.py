"""Tests for ucode.tracing and per-agent tracing wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ucode.databricks as databricks
from ucode import tracing
from ucode.agents import claude

WS = "https://example.databricks.com"


SHARED_EXPERIMENT_ID = "111"


@pytest.fixture(autouse=True)
def _avoid_real_managed_settings(monkeypatch):
    monkeypatch.setattr(claude, "_managed_settings_path", lambda: None)


def _enabled_state(profile: str | None = None) -> dict:
    return {
        "workspace": WS,
        "profile": profile,
        "tracing": {
            "enabled": True,
            "tracking_uri": f"databricks://{profile}" if profile else "databricks",
            "experiment_id": SHARED_EXPERIMENT_ID,
            "experiment_name": "/Shared/ucode-traces",
            "uc_destination": "main.default.ucode_traces",
            "sql_warehouse_id": "wh123",
        },
    }


class TestTrackingUri:
    def test_with_profile(self):
        assert tracing.tracking_uri_for_state({"profile": "myprof"}) == "databricks://myprof"

    def test_without_profile(self):
        assert tracing.tracking_uri_for_state({}) == "databricks"


class TestTracingConfig:
    def test_none_when_absent(self):
        assert tracing.tracing_config({}) is None

    def test_none_when_disabled(self):
        state = {"tracing": {"enabled": False, "agents": {}}}
        assert tracing.tracing_config(state) is None

    def test_returns_cfg_when_enabled(self):
        state = _enabled_state()
        assert tracing.tracing_config(state) is state["tracing"]


class TestAgentTracing:
    def test_returns_shared_entry(self):
        entry = tracing.agent_tracing(_enabled_state(), "claude")
        assert entry["experiment_id"] == "111"

    def test_none_for_non_tracing_agents(self):
        # Claude is the only tracing-capable agent now.
        state = _enabled_state()
        for tool in ("codex", "opencode", "gemini"):
            assert tracing.agent_tracing(state, tool) is None

    def test_none_when_disabled(self):
        assert tracing.agent_tracing({}, "claude") is None

    def test_none_when_experiment_unresolved(self):
        state = {"tracing": {"enabled": True, "tracking_uri": "databricks"}}
        assert tracing.agent_tracing(state, "claude") is None


class TestTracingEnv:
    def test_empty_when_disabled(self):
        assert tracing.tracing_env({}, "claude") == {}

    def test_uri_and_experiment(self):
        env = tracing.tracing_env(_enabled_state("p"), "claude")
        assert env == {
            "MLFLOW_TRACKING_URI": "databricks://p",
            "MLFLOW_EXPERIMENT_ID": "111",
            "MLFLOW_TRACING_SQL_WAREHOUSE_ID": "wh123",
            "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false",
        }

    def test_empty_for_non_claude_agents(self):
        # Only Claude is tracing-capable; codex/opencode get nothing.
        state = _enabled_state()
        assert tracing.tracing_env(state, "codex") == {}
        assert tracing.tracing_env(state, "opencode") == {}

    def test_includes_sql_warehouse_id(self):
        env = tracing.tracing_env(_enabled_state(), "claude")
        assert env["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "wh123"

    def test_omits_warehouse_when_absent(self):
        state = _enabled_state()
        del state["tracing"]["sql_warehouse_id"]
        assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" not in tracing.tracing_env(state, "claude")


class TestExperimentName:
    def test_leaf_name(self):
        assert tracing.experiment_name() == "ucode-traces"


def _experiment(name: str, exp_id: str, uc_destination: str | None) -> dict:
    tags = [{"key": "mlflow.experiment.sourceName", "value": name}]
    if uc_destination is not None:
        tags.append({"key": databricks.UC_TRACE_DESTINATION_TAG, "value": uc_destination})
    return {"experiment_id": exp_id, "name": name, "tags": tags}


class TestFindUcBackedExperiment:
    def test_returns_uc_backed_match(self):
        payload = {
            "experiments": [
                _experiment("/Users/me@example.com/ucode-traces", "42", "main.default.ucode_traces")
            ]
        }
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert reason is None
        assert exp == {
            "experiment_id": "42",
            "experiment_name": "/Users/me@example.com/ucode-traces",
            "uc_destination": "main.default.ucode_traces",
        }

    def test_any_catalog_schema_table_qualifies(self):
        payload = {"experiments": [_experiment("/Shared/ucode-traces", "7", "cat.sch.tbl")]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, _ = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp["uc_destination"] == "cat.sch.tbl"

    def test_none_when_no_experiment(self):
        with patch.object(databricks, "_http_post_json", return_value=({"experiments": []}, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "no experiment named 'ucode-traces'" in reason

    def test_none_when_match_not_uc_backed(self):
        payload = {"experiments": [_experiment("/Shared/ucode-traces", "9", None)]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "not backed by Unity Catalog" in reason

    def test_rejects_non_three_part_destination(self):
        payload = {"experiments": [_experiment("/Shared/ucode-traces", "9", "main.default")]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "not backed by Unity Catalog" in reason

    def test_leaf_match_excludes_substring_names(self):
        # "team-ucode-traces" ends with the leaf as a substring but is a
        # different experiment — only an exact final path segment counts.
        payload = {"experiments": [_experiment("/Shared/team-ucode-traces", "1", "c.s.t")]}
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "no experiment named 'ucode-traces'" in reason

    def test_prefers_uc_backed_over_plain_duplicate(self):
        payload = {
            "experiments": [
                _experiment("/Users/a@x.com/ucode-traces", "1", None),
                _experiment("/Shared/ucode-traces", "2", "main.default.tbl"),
            ]
        }
        with patch.object(databricks, "_http_post_json", return_value=(payload, None)):
            exp, _ = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp["experiment_id"] == "2"

    def test_returns_reason_on_search_failure(self):
        with patch.object(databricks, "_http_post_json", return_value=(None, "HTTP 403 Forbidden")):
            exp, reason = databricks.find_uc_backed_experiment(WS, "tok", "ucode-traces")
        assert exp is None
        assert "403" in reason


class TestResolveSqlWarehouseId:
    def test_prefers_running_warehouse(self):
        payload = {
            "warehouses": [
                {"id": "stopped1", "state": "STOPPED"},
                {"id": "running1", "state": "RUNNING"},
            ]
        }
        with patch.object(databricks, "_http_get_json", return_value=(payload, None)):
            wh, reason = databricks.resolve_sql_warehouse_id(WS, "tok")
        assert wh == "running1"
        assert reason is None

    def test_falls_back_to_first_when_none_running(self):
        payload = {
            "warehouses": [
                {"id": "stopped1", "state": "STOPPED"},
                {"id": "stopped2", "state": "STOPPED"},
            ]
        }
        with patch.object(databricks, "_http_get_json", return_value=(payload, None)):
            wh, reason = databricks.resolve_sql_warehouse_id(WS, "tok")
        assert wh == "stopped1"
        assert reason is None

    def test_none_when_no_warehouses(self):
        with patch.object(databricks, "_http_get_json", return_value=({"warehouses": []}, None)):
            wh, reason = databricks.resolve_sql_warehouse_id(WS, "tok")
        assert wh is None
        assert "no SQL warehouse" in reason

    def test_returns_reason_on_failure(self):
        with patch.object(databricks, "_http_get_json", return_value=(None, "HTTP 403 Forbidden")):
            wh, reason = databricks.resolve_sql_warehouse_id(WS, "tok")
        assert wh is None
        assert "403" in reason


class TestClaudeTracingEnv:
    STOP_HOOK_CMD = "/uv/bin/mlflow autolog claude stop-hook"

    def _write(self, state: dict, tmp_path, monkeypatch) -> dict:
        settings = tmp_path / "ucode-settings.json"
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings)
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "backup.json")
        # Pin the resolved hook command so tests don't depend on a real uv/mlflow.
        monkeypatch.setattr(claude, "claude_tracing_stop_hook_command", lambda: self.STOP_HOOK_CMD)
        claude.write_tool_config(state, "databricks-claude-opus-4-7")
        return json.loads(settings.read_text())

    def test_injects_mlflow_env_when_enabled(self, tmp_path, monkeypatch):
        state = {**_enabled_state(), "claude_models": {}}
        env = self._write(state, tmp_path, monkeypatch)["env"]
        assert env["MLFLOW_CLAUDE_TRACING_ENABLED"] == "true"
        assert env["MLFLOW_TRACKING_URI"] == "databricks"
        assert env["MLFLOW_EXPERIMENT_ID"] == "111"
        assert env["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "wh123"

    def test_writes_stop_hook_when_enabled(self, tmp_path, monkeypatch):
        state = {**_enabled_state(), "claude_models": {}}
        settings = self._write(state, tmp_path, monkeypatch)
        hooks = settings["hooks"]["Stop"]
        assert hooks[0]["hooks"][0]["command"] == self.STOP_HOOK_CMD

    def test_preserves_user_hooks_when_enabled(self, tmp_path, monkeypatch):
        settings = tmp_path / "ucode-settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [{"hooks": [{"type": "command", "command": "user-stop"}]}],
                        "PreToolUse": [{"hooks": [{"type": "command", "command": "user-pre"}]}],
                    }
                }
            )
        )
        state = {**_enabled_state(), "claude_models": {}}
        doc = self._write(state, tmp_path, monkeypatch)

        stop_commands = [
            hook["command"] for entry in doc["hooks"]["Stop"] for hook in entry["hooks"]
        ]
        assert stop_commands == ["user-stop", self.STOP_HOOK_CMD]
        assert doc["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "user-pre"

    def test_updates_existing_tracing_hook_when_enabled(self, tmp_path, monkeypatch):
        settings = tmp_path / "ucode-settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "/old/bin/mlflow autolog claude stop-hook",
                                    }
                                ]
                            }
                        ]
                    }
                }
            )
        )
        state = {**_enabled_state(), "claude_models": {}}
        doc = self._write(state, tmp_path, monkeypatch)

        stop_commands = [
            hook["command"] for entry in doc["hooks"]["Stop"] for hook in entry["hooks"]
        ]
        assert stop_commands == [self.STOP_HOOK_CMD]

    def test_no_mlflow_env_when_disabled(self, tmp_path, monkeypatch):
        state = {"workspace": WS, "claude_models": {}}
        settings = self._write(state, tmp_path, monkeypatch)
        assert "MLFLOW_TRACKING_URI" not in settings.get("env", {})
        assert "hooks" not in settings

    def test_strips_stale_keys_when_disabled(self, tmp_path, monkeypatch):
        settings = tmp_path / "ucode-settings.json"
        settings.write_text(
            json.dumps(
                {
                    "env": {
                        "MLFLOW_CLAUDE_TRACING_ENABLED": "true",
                        "MLFLOW_TRACKING_URI": "databricks",
                        "MLFLOW_EXPERIMENT_ID": "1",
                        "MLFLOW_TRACING_SQL_WAREHOUSE_ID": "old-wh",
                    },
                    "hooks": {
                        "Stop": [
                            {
                                "hooks": [
                                    {"type": "command", "command": "user-stop"},
                                    {
                                        "type": "command",
                                        "command": "/old/bin/mlflow autolog claude stop-hook",
                                    },
                                ]
                            }
                        ],
                        "PreToolUse": [{"hooks": [{"type": "command", "command": "user-pre"}]}],
                    },
                }
            )
        )
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", settings)
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "backup.json")
        claude.write_tool_config(
            {"workspace": WS, "claude_models": {}}, "databricks-claude-opus-4-7"
        )
        doc = json.loads(settings.read_text())
        env = doc["env"]
        assert "MLFLOW_TRACKING_URI" not in env
        assert "MLFLOW_EXPERIMENT_ID" not in env
        assert "MLFLOW_CLAUDE_TRACING_ENABLED" not in env
        assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" not in env
        assert doc["hooks"]["Stop"][0]["hooks"][0]["command"] == "user-stop"
        assert doc["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "user-pre"


class TestSelectTracingWorkspace:
    def _full(self) -> dict:
        return {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"], "profile": "pa"},
                "https://b.databricks.com": {"available_tools": ["claude"], "profile": "pb"},
                # codex/gemini aren't tracing-capable agents → excluded from candidates
                "https://c.databricks.com": {"available_tools": ["codex", "gemini"]},
            },
        }

    def test_raises_when_none_configured(self):
        with patch.object(tracing, "load_full_state", return_value={"workspaces": {}}):
            with pytest.raises(RuntimeError, match="Claude Code is not configured"):
                tracing._select_tracing_workspace()

    def test_lists_current_first_and_excludes_non_tracing(self):
        captured: dict = {}

        def fake_prompt(desc, profiles):
            captured["profiles"] = profiles
            return ("https://a.databricks.com", "pa")

        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(tracing, "prompt_for_workspace", side_effect=fake_prompt),
        ):
            state = tracing._select_tracing_workspace()

        hosts = [host for host, _ in captured["profiles"]]
        assert hosts[0] == "https://a.databricks.com"
        assert "https://c.databricks.com" not in hosts
        assert state["workspace"] == "https://a.databricks.com"

    def test_returns_picked_workspace_state(self):
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://b.databricks.com", "pb")
            ),
        ):
            state = tracing._select_tracing_workspace()
        assert state["workspace"] == "https://b.databricks.com"
        assert state["profile"] == "pb"
        assert "claude" in state["available_tools"]

    def test_raises_when_picked_workspace_unconfigured(self):
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://x.databricks.com", None)
            ),
        ):
            with pytest.raises(RuntimeError, match="no tracing-capable agents"):
                tracing._select_tracing_workspace()

    def test_single_candidate_skips_prompt(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"], "profile": "pa"},
            },
        }
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(tracing, "prompt_for_workspace") as prompt,
        ):
            state = tracing._select_tracing_workspace()
        prompt.assert_not_called()
        assert state["workspace"] == "https://a.databricks.com"


class TestSelectTracingWorkspaceOnlyEnabled:
    def test_empty_when_none_enabled(self):
        full = {
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"]},
            },
        }
        with patch.object(tracing, "load_full_state", return_value=full):
            assert tracing._select_tracing_workspace(only_enabled=True) == {}

    def test_auto_selects_lone_enabled_workspace(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"]},
                "https://b.databricks.com": {
                    "available_tools": ["claude"],
                    "tracing": {"enabled": True, "agents": {}},
                },
            },
        }
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(tracing, "prompt_for_workspace") as prompt,
        ):
            state = tracing._select_tracing_workspace(only_enabled=True)
        prompt.assert_not_called()
        assert state["workspace"] == "https://b.databricks.com"

    def test_prompts_when_multiple_enabled(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pa",
                    "tracing": {"enabled": True, "agents": {}},
                },
                "https://b.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pb",
                    "tracing": {"enabled": True, "agents": {}},
                },
            },
        }
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://b.databricks.com", "pb")
            ) as prompt,
        ):
            state = tracing._select_tracing_workspace(only_enabled=True)
        prompt.assert_called_once()
        assert state["workspace"] == "https://b.databricks.com"


class TestConfigureTracingPreservesCurrentWorkspace:
    """``save_state`` flips ``current_workspace`` on every call. The tracing
    command must not change which workspace ``ucode launch`` targets, even
    when configuring tracing for a non-current workspace."""

    def _full(self) -> dict:
        return {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pa",
                    "tracing": {"enabled": True, "agents": {}},
                },
                "https://b.databricks.com": {
                    "available_tools": ["claude"],
                    "profile": "pb",
                    "tracing": {"enabled": True, "agents": {}},
                },
            },
        }

    def test_disable_restores_original_current(self):
        captured: dict = {}
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(
                tracing, "prompt_for_workspace", return_value=("https://b.databricks.com", "pb")
            ),
            patch.object(tracing, "ensure_databricks_auth"),
            patch.object(tracing, "_rewrite_agent_configs", side_effect=lambda s: s),
            patch.object(tracing, "save_state"),
            patch.object(
                tracing,
                "set_current_workspace",
                side_effect=lambda ws: captured.setdefault("restored_to", ws),
            ),
        ):
            tracing.configure_tracing_command(disable=True)
        assert captured["restored_to"] == "https://a.databricks.com"

    def test_disable_with_none_enabled_still_calls_restore(self):
        full = {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {"https://a.databricks.com": {"available_tools": ["claude"]}},
        }
        captured: dict = {}
        with (
            patch.object(tracing, "load_full_state", return_value=full),
            patch.object(
                tracing,
                "set_current_workspace",
                side_effect=lambda ws: captured.setdefault("restored_to", ws),
            ),
        ):
            rc = tracing.configure_tracing_command(disable=True)
        assert rc == 0
        assert captured["restored_to"] == "https://a.databricks.com"


class TestEnableTracingForWorkspaces:
    """``configure --tracing`` enables tracing for explicit workspaces without
    prompting, and skips workspaces with no tracing-capable agent."""

    def _full(self) -> dict:
        return {
            "current_workspace": "https://a.databricks.com",
            "workspaces": {
                "https://a.databricks.com": {"available_tools": ["claude"], "profile": "pa"},
                # no tracing-capable agent → skipped, not an error
                "https://b.databricks.com": {"available_tools": ["gemini"], "profile": "pb"},
            },
        }

    def test_enables_without_prompting(self):
        enabled: list[str] = []
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(tracing, "prompt_for_workspace") as prompt,
            patch.object(tracing, "set_current_workspace"),
            patch.object(
                tracing,
                "_enable_tracing_for_state",
                side_effect=lambda s: enabled.append(s["workspace"]) or s,
            ),
        ):
            rc = tracing.configure_tracing_command(workspaces=[("https://a.databricks.com", None)])
        prompt.assert_not_called()
        assert rc == 0
        assert enabled == ["https://a.databricks.com"]

    def test_skips_workspace_without_tracing_agent(self):
        enabled: list[str] = []
        with (
            patch.object(tracing, "load_full_state", return_value=self._full()),
            patch.object(tracing, "set_current_workspace"),
            patch.object(
                tracing,
                "_enable_tracing_for_state",
                side_effect=lambda s: enabled.append(s["workspace"]) or s,
            ),
        ):
            rc = tracing.configure_tracing_command(workspaces=[("https://b.databricks.com", None)])
        assert enabled == []
        assert rc == 1


class TestInstallAgentTracingDeps:
    """Only Claude has a tracing runtime; it installs when Claude is configured
    with tracing on, and is skipped otherwise."""

    def test_installs_claude_runtime_when_configured(self):
        state = {**_enabled_state(), "available_tools": ["claude"]}
        with patch("ucode.agents.claude.ensure_tracing_runtime") as claude_dep:
            tracing._install_agent_tracing_deps(state)
        claude_dep.assert_called_once()

    def test_skips_when_claude_not_configured(self):
        # Claude isn't configured on this workspace, so its runtime is skipped.
        state = {**_enabled_state(), "available_tools": ["codex"]}
        with patch("ucode.agents.claude.ensure_tracing_runtime") as claude_dep:
            tracing._install_agent_tracing_deps(state)
        claude_dep.assert_not_called()


class TestDisableTracing:
    def test_sets_disabled_and_rewrites_configs(self):
        state = _enabled_state()
        with (
            patch.object(tracing, "_rewrite_agent_configs", side_effect=lambda s: s) as rewrite,
            patch.object(tracing, "save_state"),
        ):
            out = tracing.disable_tracing(state)
        assert out["tracing"]["enabled"] is False
        rewrite.assert_called_once()


class TestStopHookCommand:
    def test_builds_command_from_resolved_path(self, monkeypatch):
        monkeypatch.setattr(claude, "_uv_tool_mlflow_path", lambda: "/uv/bin/mlflow")
        assert (
            claude.claude_tracing_stop_hook_command() == "/uv/bin/mlflow autolog claude stop-hook"
        )

    def test_none_when_mlflow_missing(self, monkeypatch):
        monkeypatch.setattr(claude, "_uv_tool_mlflow_path", lambda: None)
        assert claude.claude_tracing_stop_hook_command() is None


class TestParseMlflowVersion:
    def test_parses_full_version(self):
        assert claude._parse_mlflow_version("mlflow, version 3.12.0") == (3, 12)

    def test_parses_major_minor(self):
        assert claude._parse_mlflow_version("mlflow version 3.4") == (3, 4)

    def test_returns_none_on_garbage(self):
        assert claude._parse_mlflow_version("not a version") is None


class TestEnsureMlflowCli:
    def test_noop_when_already_in_range(self):
        with (
            patch.object(claude, "_installed_mlflow_version", return_value=(3, 11)),
            patch.object(claude.subprocess, "run") as run,
        ):
            assert claude._ensure_mlflow_cli() is True
        run.assert_not_called()

    def test_installs_when_missing(self, monkeypatch):
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: None)
        monkeypatch.setattr(claude.shutil, "which", lambda binary: f"/bin/{binary}")
        monkeypatch.setattr(claude, "_uv_tool_mlflow_path", lambda: "/bin/mlflow")
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is True
        cmd = run.call_args[0][0]
        assert cmd[:3] == ["uv", "tool", "install"]
        assert claude.MLFLOW_CLI_SPEC in cmd
        # Always --force so an unparseable-version mlflow on disk doesn't trip
        # uv's "Executable already exists" error.
        assert "--force" in cmd

    def test_force_replaces_when_below_minimum(self, monkeypatch):
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: (3, 4))
        monkeypatch.setattr(claude.shutil, "which", lambda binary: f"/bin/{binary}")
        monkeypatch.setattr(claude, "_uv_tool_mlflow_path", lambda: "/bin/mlflow")
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is True
        assert "--force" in run.call_args[0][0]

    def test_force_replaces_when_above_maximum(self, monkeypatch):
        # 3.12 dropped UC trace writes — it must be replaced, not left alone.
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: (3, 12))
        monkeypatch.setattr(claude.shutil, "which", lambda binary: f"/bin/{binary}")
        monkeypatch.setattr(claude, "_uv_tool_mlflow_path", lambda: "/bin/mlflow")
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is True
        assert "--force" in run.call_args[0][0]

    def test_warns_when_uv_missing(self, monkeypatch):
        monkeypatch.setattr(claude, "_installed_mlflow_version", lambda: None)
        monkeypatch.setattr(claude.shutil, "which", lambda binary: None)
        with patch.object(claude.subprocess, "run") as run:
            assert claude._ensure_mlflow_cli() is False
        run.assert_not_called()
