"""End-to-end test that the User-Agent header ucode injects actually reaches the wire.

We don't talk to a real Databricks workspace here — instead we stand up a
tiny HTTP capture server on localhost, point each agent's *_BASE_URL at it,
launch the agent, and assert on the User-Agent the server saw.

The server returns a canned error so the agent itself fails; we don't care
about the agent's exit code, only the headers that arrived before it bailed.
This is the cheapest way to verify "ucode wired the UA into the request"
end-to-end without TLS, real models, or workspace credentials.

Skipped per-agent when the binary isn't installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ucode.telemetry import agent_version, ucode_version


def _require_binary(binary: str):
    if not shutil.which(binary):
        pytest.skip(f"`{binary}` is not installed")


class _CapturedRequest:
    """Bag of fields recorded by the capture server for one inbound request."""

    def __init__(self, method: str, path: str, headers: dict[str, str]):
        self.method = method
        self.path = path
        self.headers = headers


class _CaptureServer:
    """HTTP server that records every inbound request's method/path/headers
    and replies 401 with a JSON error so the agent fails fast and exits."""

    def __init__(self):
        self.requests: list[_CapturedRequest] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        captured = self.requests

        class Handler(BaseHTTPRequestHandler):
            def _record_and_reply(self):
                # Drain any request body so the client doesn't block on write.
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    try:
                        self.rfile.read(length)
                    except Exception:
                        pass
                captured.append(
                    _CapturedRequest(
                        method=self.command,
                        path=self.path,
                        headers=dict(self.headers.items()),
                    )
                )
                body = json.dumps(
                    {"error": {"type": "invalid_api_key", "message": "ucode test capture"}}
                ).encode()
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                self._record_and_reply()

            def do_POST(self):  # noqa: N802
                self._record_and_reply()

            def log_message(self, format, *args):  # noqa: A002
                # Silence the default stderr access log.
                pass

        # Bind to an ephemeral port on localhost.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def first_request_with_path_prefix(self, prefix: str) -> _CapturedRequest | None:
        for req in self.requests:
            if req.path.startswith(prefix):
                return req
        return None


@pytest.fixture
def capture_server():
    server = _CaptureServer()
    server.start()
    yield server
    server.stop()


def _expected_ua(agent_name: str, binary: str) -> str:
    return f"ucode/{ucode_version()} {agent_name}/{agent_version(binary)}"


def _assert_ua(req: _CapturedRequest, expected: str) -> None:
    # http.server lowercases header lookup keys; check both common spellings.
    ua = req.headers.get("User-Agent") or req.headers.get("user-agent")
    assert ua == expected, f"User-Agent mismatch.\n  got:      {ua!r}\n  expected: {expected!r}"


def _run_until_first_request(
    cmd: list[str], env: dict[str, str], timeout: int = 20
) -> subprocess.CompletedProcess | None:
    """Spawn the agent. We only need it to fire its first HTTP request; some
    agents retry on 401 indefinitely. Swallow timeouts — the capture server
    has what we need by then. Returns the CompletedProcess (or None on
    timeout) so callers can surface stderr on failure."""
    try:
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return None


def _no_request_msg(server: _CaptureServer, result: subprocess.CompletedProcess | None) -> str:
    if result is None:
        return "Agent timed out before any request reached the capture server."
    stderr = (result.stderr or b"").decode(errors="replace")[:600]
    stdout = (result.stdout or b"").decode(errors="replace")[:300]
    return (
        f"No request reached the capture server.\n"
        f"  paths: {[r.path for r in server.requests]}\n"
        f"  rc:    {result.returncode}\n"
        f"  stderr: {stderr!r}\n"
        f"  stdout: {stdout!r}"
    )


# ---------------------------------------------------------------------------
# Per-agent tests
# ---------------------------------------------------------------------------


class TestClaudeUserAgent:
    def test_user_agent_arrives_at_gateway(self, tmp_path, monkeypatch, capture_server):
        import ucode.config_io as config_io_mod
        from ucode.agents import claude

        _require_binary("claude")
        config_dir = tmp_path / "claude_config"
        config_dir.mkdir()
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", config_dir / "settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "claude.backup.json")

        # Render the overlay against the capture server (treated as the workspace).
        # render_overlay just builds the env block; we write it ourselves to
        # avoid the apiKeyHelper / save_state plumbing.
        overlay, _ = claude.render_overlay(capture_server.base_url, "test-model")
        claude.CLAUDE_SETTINGS_PATH.write_text(json.dumps(overlay), encoding="utf-8")
        env = {
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "ANTHROPIC_API_KEY": "test-key-not-real",
            **overlay["env"],
        }

        result = _run_until_first_request(claude.validate_cmd("claude"), env)

        req = capture_server.first_request_with_path_prefix("/ai-gateway/anthropic")
        assert req is not None, _no_request_msg(capture_server, result)
        _assert_ua(req, _expected_ua("claude", "claude"))


class TestCodexUserAgent:
    def test_user_agent_arrives_at_gateway(self, tmp_path, monkeypatch, capture_server):
        import ucode.config_io as config_io_mod
        from ucode.agents import codex

        _require_binary("codex")
        config_dir = tmp_path / "codex_home" / ".codex"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "ucode.config.toml"

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "codex.backup.toml")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            codex.write_tool_config({"workspace": capture_server.base_url})

        # Point codex at the redirected config dir and let $OPENAI_API_KEY
        # bypass the auth command.
        env = {
            **os.environ,
            "CODEX_HOME": str(config_dir),
            "OPENAI_API_KEY": "test-key-not-real",
        }
        result = _run_until_first_request(codex.validate_cmd("codex"), env)

        req = capture_server.first_request_with_path_prefix("/ai-gateway/codex")
        assert req is not None, _no_request_msg(capture_server, result)
        _assert_ua(req, _expected_ua("codex", "codex"))


class TestOpencodeUserAgent:
    def test_user_agent_arrives_at_gateway(self, tmp_path, monkeypatch, capture_server):
        import ucode.config_io as config_io_mod
        from ucode.agents import opencode

        _require_binary("opencode")
        # Redirect via XDG_CONFIG_HOME so the spawned opencode reads from
        # tmp_path instead of the developer's real ~/.config/opencode.
        xdg = tmp_path / "xdg"
        opencode_dir = xdg / "opencode"
        opencode_dir.mkdir(parents=True)
        config_path = opencode_dir / "opencode.json"
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(opencode, "OPENCODE_CONFIG_PATH", config_path)
        monkeypatch.setattr(opencode, "OPENCODE_BACKUP_PATH", tmp_path / "opencode.backup.json")

        # Construct a state with localhost base URLs so render_overlay points
        # both providers at the capture server.
        state = {
            "workspace": capture_server.base_url,
            "opencode_models": {"anthropic": ["test-claude-model"]},
            "base_urls": {
                "opencode": {
                    "anthropic": f"{capture_server.base_url}/ai-gateway/anthropic/v1",
                    "gemini": f"{capture_server.base_url}/ai-gateway/gemini/v1beta",
                },
            },
        }
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            mp.setattr(
                "ucode.agents.opencode.get_databricks_token",
                lambda ws, profile=None, **kwargs: "test-token",
            )
            opencode.write_tool_config(state, "test-claude-model", token="test-token")

        env = {**os.environ, "OAUTH_TOKEN": "test-token", "XDG_CONFIG_HOME": str(xdg)}
        result = _run_until_first_request(opencode.validate_cmd("opencode"), env)

        req = capture_server.first_request_with_path_prefix("/ai-gateway/anthropic")
        assert req is not None, _no_request_msg(capture_server, result)
        # The Vercel AI SDK appends its own suffix to UA; ucode's prefix
        # appears at the front. Per upstream investigation, the AI SDK
        # prepends our value then suffixes "ai-sdk/anthropic/X
        # ai-sdk/provider-utils/X runtime/bun/X". We assert the prefix only.
        ua = req.headers.get("User-Agent") or req.headers.get("user-agent") or ""
        expected_prefix = _expected_ua("opencode", "opencode")
        assert ua.startswith(expected_prefix), (
            f"OpenCode UA missing ucode prefix.\n  got:    {ua!r}\n  prefix: {expected_prefix!r}"
        )


class TestGeminiUserAgent:
    def test_user_agent_arrives_at_gateway(self, tmp_path, monkeypatch, capture_server):
        import ucode.config_io as config_io_mod
        from ucode.agents import gemini

        _require_binary("gemini")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / "ucode.env")
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "gemini-ucode-env.backup")
        monkeypatch.setattr(gemini, "GEMINI_HOME_DIR", tmp_path / ".gemini-home")
        monkeypatch.setattr(
            gemini, "GEMINI_SETTINGS_PATH", tmp_path / ".gemini-home" / ".gemini" / "settings.json"
        )
        # Run from tmp_path so Gemini sees an untrusted folder (the trust env
        # var built into build_runtime_env handles it).
        monkeypatch.chdir(tmp_path)

        env = gemini.build_runtime_env(capture_server.base_url, "test-model", "test-token")
        result = _run_until_first_request(gemini.validate_cmd("gemini"), env)

        req = capture_server.first_request_with_path_prefix("/ai-gateway/gemini")
        assert req is not None, _no_request_msg(capture_server, result)
        _assert_ua(req, _expected_ua("gemini", "gemini"))


class TestPiUserAgent:
    def test_user_agent_arrives_at_gateway(self, tmp_path, monkeypatch, capture_server):
        import ucode.config_io as config_io_mod
        from ucode.agents import pi

        _require_binary("pi")
        pi_home = tmp_path / "pi-home"
        pi_dir = pi_home / ".pi" / "agent"
        config_path = pi_dir / "models.json"

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(pi, "PI_UCODE_HOME", pi_home)
        monkeypatch.setattr(pi, "PI_CONFIG_DIR", pi_dir)
        monkeypatch.setattr(pi, "PI_CONFIG_PATH", config_path)
        monkeypatch.setattr(pi, "PI_SETTINGS_PATH", pi_dir / "settings.json")
        monkeypatch.setattr(pi, "PI_SETTINGS_BACKUP_PATH", tmp_path / "pi-settings.backup.json")
        monkeypatch.setattr(pi, "PI_BACKUP_PATH", tmp_path / "pi.backup.json")

        state = {
            "workspace": capture_server.base_url,
            "claude_models": {"sonnet": "test-claude-model"},
            "codex_models": [],
            "gemini_models": [],
            "base_urls": {
                "pi": {
                    "claude": f"{capture_server.base_url}/ai-gateway/anthropic",
                    "openai": f"{capture_server.base_url}/ai-gateway/codex/v1",
                    "gemini": f"{capture_server.base_url}/ai-gateway/gemini/v1beta",
                },
            },
        }
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            mp.setattr(
                "ucode.agents.pi.get_databricks_token",
                lambda ws, profile=None, **kwargs: "test-token",
            )
            pi.write_tool_config(state, "test-claude-model", token="test-token")

        env = pi.build_runtime_env("test-token")
        # Pi now resolves its apiKey by running `ucode auth-token`. The static
        # bearer short-circuit gives that command something to print without a
        # real workspace behind the capture server.
        env["DATABRICKS_BEARER"] = "test-token"
        result = _run_until_first_request(pi.validate_cmd("pi"), env)

        req = capture_server.first_request_with_path_prefix("/ai-gateway/anthropic")
        assert req is not None, _no_request_msg(capture_server, result)
        _assert_ua(req, _expected_ua("pi", "pi"))
