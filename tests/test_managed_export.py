"""Tests for `ucode export` and its :mod:`ucode.managed_export` backing module.

`export` is read-only and offline: it serializes the local managed config to the external
proto-JSON `CodingAgentConfig` that `ucode publish -f <path>` consumes. These focus on the parts
that must not regress — a clean machine-readable stdout stream, byte-identical file output, atomic
replacement that never truncates on failure, exclusion of server-owned fields, and the absence of
any auth/admin/network call.
"""

from __future__ import annotations

import contextlib
import json
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import ucode.config_io as config_io_mod
import ucode.managed_config as managed_config_mod
import ucode.managed_export as export_mod
from ucode.cli import app
from ucode.managed_config import normalize_managed_config
from ucode.managed_setup import serialize_managed_config, validate_manifest

runner = CliRunner()

WORKSPACE = "https://ws.example.com"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

FULL_MANIFEST = {
    "name": "coding-agent-configs/abc123",
    "default_agent": "claude",
    "enabled_agents": {
        "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}},
        "codex": {"model_config": {"default_model": "system.ai.gpt-5-6"}},
    },
    "mcp_servers": [{"name": "system.ai.slack", "type": "mcp-service"}],
    "skills": {"names": ["main.default"]},
}


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    """Point the managed-config file at a tmp dir so no test touches the real ~/.ucode."""
    monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
    monkeypatch.setattr(managed_config_mod, "MANAGED_STATE_PATH", tmp_path / "managed-state.json")
    monkeypatch.setattr(config_io_mod, "_dry_run", False)


@contextlib.contextmanager
def _with_manifest(manifest: dict | None, workspace: str | None = WORKSPACE):
    """Patch the module's local reads so a test controls the source config without disk or network."""
    with (
        patch.object(export_mod, "load_state", return_value={"workspace": workspace}),
        patch.object(export_mod, "load_managed_state", return_value=manifest),
    ):
        yield


class TestBuildPayload:
    def test_excludes_server_owned_resource_name(self):
        with _with_manifest(FULL_MANIFEST):
            payload = export_mod.build_export_payload()
        assert "name" not in payload
        assert payload["default_agent"] == "CODING_AGENT_CLAUDE_CODE"
        assert payload["mcp_servers"] == [
            {"name": "system.ai.slack", "type": "MCP_SERVER_TYPE_UC_SERVICE"}
        ]

    def test_envelope_workspace_first_then_spec_version(self):
        with _with_manifest(FULL_MANIFEST):
            payload = export_mod.build_export_payload()
        assert list(payload)[:2] == ["workspace", "spec_version"]
        assert payload["workspace"] == WORKSPACE
        assert payload["spec_version"] == 1

    def test_matches_serialize_minus_name_under_envelope(self):
        config = serialize_managed_config(FULL_MANIFEST)
        config.pop("name", None)
        expected = {"workspace": WORKSPACE, "spec_version": 1, **config}
        with _with_manifest(FULL_MANIFEST):
            assert export_mod.build_export_payload() == expected

    def test_config_roundtrips_through_parser_and_validator(self):
        with _with_manifest(FULL_MANIFEST):
            payload = export_mod.build_export_payload()
        config = {k: v for k, v in payload.items() if k not in ("workspace", "spec_version")}
        reparsed = normalize_managed_config(config)
        assert validate_manifest(reparsed, None) == []
        assert serialize_managed_config(reparsed) == config

    def test_no_config_is_actionable(self):
        with (
            patch.object(export_mod, "load_state", return_value={}),
            patch.object(export_mod, "load_managed_state", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="No managed coding-agent config found"):
                export_mod.build_export_payload()

    def test_invalid_config_is_rejected(self):
        invalid = {"enabled_agents": {"claude": {}}}
        with _with_manifest(invalid):
            with pytest.raises(RuntimeError, match="not valid"):
                export_mod.build_export_payload()

    def test_falls_back_to_managed_state_workspace(self):
        managed_config_mod.save_managed_state(WORKSPACE, FULL_MANIFEST)
        with patch.object(export_mod, "load_state", return_value={}):
            payload = export_mod.build_export_payload()
        assert payload["default_agent"] == "CODING_AGENT_CLAUDE_CODE"


class TestExportCommandStdout:
    def test_emits_valid_json_with_single_trailing_newline(self, capsys):
        with _with_manifest(FULL_MANIFEST):
            export_mod.export_command()
        captured = capsys.readouterr()
        out = captured.out
        assert out.endswith("}\n")
        assert not out.endswith("}\n\n")
        json.loads(out)
        assert captured.err == ""

    def test_stdout_has_no_rich_or_human_output(self, capsys):
        with _with_manifest(FULL_MANIFEST):
            export_mod.export_command()
        out = capsys.readouterr().out
        assert _ANSI_RE.search(out) is None
        with _with_manifest(FULL_MANIFEST):
            payload = export_mod.build_export_payload()
        assert json.loads(out) == payload


class TestExportCommandFile:
    def test_file_bytes_identical_to_stdout_and_stdout_empty(self, capsys, tmp_path):
        with _with_manifest(FULL_MANIFEST):
            export_mod.export_command()
        stdout_bytes = capsys.readouterr().out

        dest = tmp_path / "config.json"
        with _with_manifest(FULL_MANIFEST):
            export_mod.export_command(file_path=str(dest))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert dest.read_text(encoding="utf-8") == stdout_bytes

    def test_expands_user_home_in_output_path(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with _with_manifest(FULL_MANIFEST):
            export_mod.export_command(file_path="~/config.json")
        capsys.readouterr()
        assert (tmp_path / "config.json").exists()

    def test_replaces_existing_destination(self, tmp_path):
        dest = tmp_path / "config.json"
        dest.write_text("stale contents", encoding="utf-8")
        with _with_manifest(FULL_MANIFEST):
            export_mod.export_command(file_path=str(dest))
        assert json.loads(dest.read_text(encoding="utf-8"))["default_agent"] == (
            "CODING_AGENT_CLAUDE_CODE"
        )

    def test_invalid_config_leaves_existing_destination_unchanged(self, tmp_path):
        dest = tmp_path / "config.json"
        dest.write_text("original", encoding="utf-8")
        invalid = {"enabled_agents": {"claude": {}}}
        with _with_manifest(invalid):
            with pytest.raises(RuntimeError):
                export_mod.export_command(file_path=str(dest))
        assert dest.read_text(encoding="utf-8") == "original"

    def test_invalid_config_does_not_create_destination(self, tmp_path):
        dest = tmp_path / "config.json"
        invalid = {"enabled_agents": {"claude": {}}}
        with _with_manifest(invalid):
            with pytest.raises(RuntimeError):
                export_mod.export_command(file_path=str(dest))
        assert not dest.exists()

    def test_missing_parent_directory_fails_without_creating_it(self, tmp_path):
        missing_parent = tmp_path / "nope"
        dest = missing_parent / "config.json"
        with _with_manifest(FULL_MANIFEST):
            with pytest.raises(RuntimeError, match="parent directory does not exist"):
                export_mod.export_command(file_path=str(dest))
        assert not missing_parent.exists()

    def test_write_failure_is_actionable_and_leaves_no_temp_file(self, tmp_path):
        dest = tmp_path / "adir"
        dest.mkdir()
        with _with_manifest(FULL_MANIFEST):
            with pytest.raises(RuntimeError, match="Failed to write"):
                export_mod.export_command(file_path=str(dest))
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".ucode-export-")]
        assert leftovers == []


class TestNoAuthOrAdmin:
    def test_no_admin_or_token_lookup_occurs(self, capsys):
        with (
            _with_manifest(FULL_MANIFEST),
            patch("ucode.databricks.is_workspace_admin") as admin,
            patch("ucode.databricks.get_databricks_token") as token,
        ):
            export_mod.export_command()
        capsys.readouterr()
        admin.assert_not_called()
        token.assert_not_called()

    def test_admin_and_non_admin_produce_identical_output(self, capsys):
        outputs = []
        for admin_value in (True, False):
            with (
                _with_manifest(FULL_MANIFEST),
                patch("ucode.databricks.is_workspace_admin", MagicMock(return_value=admin_value)),
            ):
                export_mod.export_command()
            outputs.append(capsys.readouterr().out)
        assert outputs[0] == outputs[1]


class TestExportCLI:
    def test_help_documents_command_and_flags(self):
        top = runner.invoke(app, ["--help"])
        assert top.exit_code == 0
        assert "export" in _ANSI_RE.sub("", top.output)

        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        cleaned = _ANSI_RE.sub("", result.output)
        assert "--file" in cleaned
        assert "-f" in cleaned

    def test_long_and_short_file_flags_both_write_the_file(self, tmp_path):
        for flag in ("--file", "-f"):
            dest = tmp_path / f"cfg{flag.strip('-')}.json"
            with _with_manifest(FULL_MANIFEST):
                result = runner.invoke(app, ["export", flag, str(dest)])
            assert result.exit_code == 0
            assert json.loads(dest.read_text(encoding="utf-8"))["default_agent"] == (
                "CODING_AGENT_CLAUDE_CODE"
            )

    def test_no_config_exits_nonzero(self):
        with (
            patch.object(export_mod, "load_state", return_value={}),
            patch.object(export_mod, "load_managed_state", return_value=None),
        ):
            result = runner.invoke(app, ["export"])
        assert result.exit_code == 1
