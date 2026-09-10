"""Tests for :mod:`ucode.managed_publish` — the input handling behind `ucode publish`.

Cover the two config sources (in-process export payload and a `-f` file), the envelope checks
(workspace match, spec_version type and value), and the canonicalization that rejects server-owned,
unknown, and lossy fields before anything reaches the workspace.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ucode.managed_publish as publish_mod
from ucode.managed_publish import load_publish_payload, parse_publish_payload
from ucode.managed_setup import serialize_managed_config

WORKSPACE = "https://ws.example.com"

MANIFEST = {
    "default_agent": "claude",
    "enabled_agents": {
        "claude": {"model_config": {"default_model": "system.ai.claude-opus-4-8"}},
        "codex": {"model_config": {"default_model": "system.ai.gpt-5-6"}},
    },
    "mcp_servers": [{"name": "system.ai.slack", "type": "mcp-service"}],
    "skills": {"names": ["main.default"]},
}


def _payload(manifest=MANIFEST, *, workspace=WORKSPACE, spec_version=1, **extra):
    config = serialize_managed_config(manifest)
    config.pop("name", None)
    return {"workspace": workspace, "spec_version": spec_version, **config, **extra}


class TestLoadPublishPayload:
    def test_no_file_uses_build_export_payload_in_process(self):
        sentinel = {"workspace": WORKSPACE, "spec_version": 1}
        with patch.object(publish_mod, "build_export_payload", return_value=sentinel) as build:
            assert load_publish_payload(None) is sentinel
        build.assert_called_once_with()

    def test_reads_json_file(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(_payload()), encoding="utf-8")
        assert load_publish_payload(str(path)) == _payload()

    def test_expands_user_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "config.json").write_text(json.dumps(_payload()), encoding="utf-8")
        assert load_publish_payload("~/config.json")["workspace"] == WORKSPACE

    def test_missing_file_is_actionable(self, tmp_path):
        with pytest.raises(RuntimeError, match="No config file"):
            load_publish_payload(str(tmp_path / "absent.json"))

    def test_malformed_json_is_actionable(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            load_publish_payload(str(path))

    def test_non_object_root_is_rejected(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(RuntimeError, match="JSON object at the top level"):
            load_publish_payload(str(path))

    def test_non_utf8_is_actionable(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_bytes(b"\xff\xfe not utf-8")
        with pytest.raises(RuntimeError, match="UTF-8"):
            load_publish_payload(str(path))


class TestParsePublishPayload:
    def test_returns_manifest_and_api_payload(self):
        manifest, api_payload = parse_publish_payload(_payload(), WORKSPACE)
        assert manifest["default_agent"] == "claude"
        assert set(manifest["enabled_agents"]) == {"claude", "codex"}
        assert api_payload["spec_version"] == 1
        assert api_payload["default_agent"] == "CODING_AGENT_CLAUDE_CODE"
        assert "workspace" not in api_payload
        assert "name" not in api_payload

    def test_api_payload_matches_serialize_of_manifest(self):
        manifest, api_payload = parse_publish_payload(_payload(), WORKSPACE)
        canonical = serialize_managed_config(manifest)
        assert api_payload == {"spec_version": 1, **canonical}

    def test_non_object_payload_is_rejected(self):
        with pytest.raises(RuntimeError, match="JSON object"):
            parse_publish_payload([1, 2], WORKSPACE)

    def test_missing_workspace_is_rejected(self):
        payload = _payload()
        del payload["workspace"]
        with pytest.raises(RuntimeError, match="workspace"):
            parse_publish_payload(payload, WORKSPACE)

    def test_mismatched_workspace_is_rejected(self):
        with pytest.raises(RuntimeError, match="configured workspace"):
            parse_publish_payload(_payload(workspace="https://other.example.com"), WORKSPACE)

    def test_workspace_match_ignores_scheme_and_trailing_slash(self):
        manifest, _ = parse_publish_payload(_payload(workspace="ws.example.com/"), WORKSPACE)
        assert manifest["default_agent"] == "claude"

    def test_missing_spec_version_is_rejected(self):
        payload = _payload()
        del payload["spec_version"]
        with pytest.raises(RuntimeError, match="spec_version"):
            parse_publish_payload(payload, WORKSPACE)

    def test_boolean_spec_version_is_rejected(self):
        with pytest.raises(RuntimeError, match="spec_version"):
            parse_publish_payload(_payload(spec_version=True), WORKSPACE)

    def test_fractional_spec_version_is_rejected(self):
        with pytest.raises(RuntimeError, match="spec_version"):
            parse_publish_payload(_payload(spec_version=1.0), WORKSPACE)

    def test_unsupported_spec_version_is_rejected(self):
        with pytest.raises(RuntimeError, match="spec_version 2"):
            parse_publish_payload(_payload(spec_version=2), WORKSPACE)

    def test_server_owned_name_is_rejected(self):
        with pytest.raises(RuntimeError, match="server-owned"):
            parse_publish_payload(_payload(name="coding-agent-configs/abc"), WORKSPACE)

    def test_server_owned_workspace_id_is_rejected(self):
        with pytest.raises(RuntimeError, match="does not recognize"):
            parse_publish_payload(_payload(workspace_id="123456"), WORKSPACE)

    def test_unknown_top_level_field_is_rejected(self):
        with pytest.raises(RuntimeError, match="does not recognize"):
            parse_publish_payload(_payload(bogus="value"), WORKSPACE)

    def test_nested_unknown_field_is_rejected(self):
        payload = _payload()
        payload["enabled_agents"][0]["config"] = {"unknown_setting": True}
        with pytest.raises(RuntimeError, match="enabled_agents"):
            parse_publish_payload(payload, WORKSPACE)

    def test_key_order_does_not_matter(self):
        config = serialize_managed_config(MANIFEST)
        config.pop("name", None)
        reordered = {**config, "spec_version": 1, "workspace": WORKSPACE}
        manifest, api_payload = parse_publish_payload(reordered, WORKSPACE)
        assert api_payload["spec_version"] == 1
        assert manifest["default_agent"] == "claude"

    def test_display_name_round_trips(self):
        manifest_with_name = {**MANIFEST, "display_name": "paved-path"}
        manifest, api_payload = parse_publish_payload(_payload(manifest_with_name), WORKSPACE)
        assert manifest["display_name"] == "paved-path"
        assert api_payload["display_name"] == "paved-path"
