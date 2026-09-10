"""Tests for the per-connection MCP login helpers (mcp_connection_login).

Network-free: URL/connection parsing and the login runner with the subprocess
monkeypatched.
"""

from __future__ import annotations

import subprocess

from ucode import mcp_connection_login as mcl

WS = "https://ws.staging.cloud.databricks.com"
AIGW_URL = f"{WS}/ai-gateway/mcp-services/system.ai.github"


class TestConnectionFromUrl:
    def test_plain_endpoint(self):
        assert mcl.connection_from_url(AIGW_URL) == "system.ai.github"

    def test_with_trailing_path_and_query(self):
        assert mcl.connection_from_url(f"{AIGW_URL}/tools/list?x=1") == "system.ai.github"

    def test_non_aigw_url_is_none(self):
        assert mcl.connection_from_url(f"{WS}/api/2.0/mcp/functions/system/ai") is None

    def test_missing_service_is_none(self):
        assert mcl.connection_from_url(f"{WS}/ai-gateway/mcp-services/") is None


class TestRunConnectionLogin:
    def _fake_run(self, captured, *, returncode, stderr=""):
        def _run(argv, **kwargs):
            captured.append(argv)
            return subprocess.CompletedProcess(argv, returncode, stdout="", stderr=stderr)

        return _run

    def test_success_sends_resource_and_host_without_client_id(self, monkeypatch):
        captured: list[list[str]] = []
        monkeypatch.setattr(mcl.subprocess, "run", self._fake_run(captured, returncode=0))

        ok, message = mcl.run_connection_login(AIGW_URL, WS, profile="p")

        assert ok and message == "signed in"
        argv = captured[0]
        assert argv[:3] == ["databricks", "auth", "login"]
        assert "--resource" in argv and AIGW_URL in argv
        assert "--host" in argv and WS in argv
        assert "--profile" in argv and "p" in argv
        # Uses the CLI's default client (its own registered redirect), so no --client-id.
        assert "--client-id" not in argv

    def test_failure_returns_cli_detail(self, monkeypatch):
        captured: list[list[str]] = []
        monkeypatch.setattr(
            mcl.subprocess, "run", self._fake_run(captured, returncode=1, stderr="nope")
        )
        ok, message = mcl.run_connection_login(AIGW_URL, WS)
        assert not ok and message == "nope"

    def test_timeout_is_reported(self, monkeypatch):
        def _run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 1)

        monkeypatch.setattr(mcl.subprocess, "run", _run)
        ok, message = mcl.run_connection_login(AIGW_URL, WS)
        assert not ok and "timed out" in message

    def test_binary_missing_is_reported(self, monkeypatch):
        def _run(argv, **kwargs):
            raise OSError("not found")

        monkeypatch.setattr(mcl.subprocess, "run", _run)
        ok, message = mcl.run_connection_login(AIGW_URL, WS)
        assert not ok and "could not run" in message
