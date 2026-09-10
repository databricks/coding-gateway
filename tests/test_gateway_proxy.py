"""Tests for the relayed refresh proxy header handling."""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time

import httpx

from ucode import gateway_proxy


def _make_jwt(exp: float | None) -> str:
    """A minimal JWT-shaped token whose payload carries `exp` (or none)."""
    claims = {"exp": exp} if exp is not None else {"sub": "x"}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{payload}.sig"


class _FakeHandler:
    """Minimal stand-in exposing a `.headers` mapping like BaseHTTPRequestHandler."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class TestForwardedRequestHeaders:
    def test_injects_swap_header_with_bearer(self):
        handler = _FakeHandler({"Authorization": "Bearer anthropic-oauth"})
        out = gateway_proxy.forwarded_request_headers(handler, "dbx-token")
        assert out["X-Databricks-AI-Gateway-Token"] == "Bearer dbx-token"

    def test_passes_authorization_through_untouched(self):
        # The caller's Anthropic OAuth must survive verbatim — the proxy never
        # reads or rewrites it.
        handler = _FakeHandler({"Authorization": "Bearer anthropic-oauth"})
        out = gateway_proxy.forwarded_request_headers(handler, "dbx-token")
        assert out["Authorization"] == "Bearer anthropic-oauth"

    def test_overwrites_client_supplied_swap_header(self):
        # A stale settings.json value must not survive; the proxy replaces it.
        handler = _FakeHandler({"X-Databricks-AI-Gateway-Token": "Bearer stale"})
        out = gateway_proxy.forwarded_request_headers(handler, "fresh")
        assert out["X-Databricks-AI-Gateway-Token"] == "Bearer fresh"

    def test_overwrites_authorization_header(self):
        handler = _FakeHandler({"Authorization": "Bearer stale"})
        out = gateway_proxy.forwarded_request_headers(
            handler, "fresh", gateway_proxy.AUTHORIZATION_HEADER
        )
        assert out["Authorization"] == "Bearer fresh"

    def test_strips_hop_by_hop_headers(self):
        handler = _FakeHandler(
            {"Host": "localhost:9", "Content-Length": "5", "Connection": "keep-alive"}
        )
        out = gateway_proxy.forwarded_request_headers(handler, "t")
        assert "Host" not in out
        assert "Content-Length" not in out
        assert "Connection" not in out


class _FakeResponse:
    """Stand-in for httpx.Response exposing only what `_relay_response` reads."""

    def __init__(self, status_code: int, headers: dict[str, str], chunks):
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks
        self.chunk_sizes = []

    def iter_raw(self, chunk_size=None):
        self.chunk_sizes.append(chunk_size)
        yield from self._chunks


class _BrokenPipeWriter(io.RawIOBase):
    """A wfile stand-in that raises BrokenPipeError on write, mimicking a client
    (Claude Code) that closed the connection mid-response."""

    def write(self, _data):  # type: ignore[override]
        raise BrokenPipeError(32, "Broken pipe")


def _relay_handler(wfile) -> gateway_proxy._ProxyHandler:
    # Bypass BaseHTTPRequestHandler.__init__ (which would service a socket);
    # we only exercise _relay_response's write path. Set the few attributes the
    # send_response/send_header machinery reads (normally populated by __init__).
    handler = object.__new__(gateway_proxy._ProxyHandler)
    handler.wfile = wfile
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST /v1/messages HTTP/1.1"
    handler.command = "POST"
    handler._headers_buffer = []
    return handler


class TestRelayResponseClientDisconnect:
    def test_relay_swallows_broken_pipe_on_headers(self):
        # Client gone before headers flush: end_headers write raises BrokenPipe.
        handler = _relay_handler(_BrokenPipeWriter())
        resp = _FakeResponse(200, {}, [b'{"ok":true}'])
        # Must not raise — a dead client is a routine teardown, not an error.
        handler._relay_response(resp)

    def test_relay_swallows_connection_reset_mid_stream(self):
        # Headers flush ok, then the client resets while streaming body chunks.
        writes: list[bytes] = []

        class _ResetAfterHeaders(io.RawIOBase):
            def write(self, data):  # type: ignore[override]
                writes.append(bytes(data))
                if b"chunk" in bytes(data):
                    raise ConnectionResetError(54, "Connection reset by peer")
                return len(data)

            def flush(self):
                return None

        handler = _relay_handler(_ResetAfterHeaders())
        resp = _FakeResponse(200, {}, [b"chunk-of-sse-data"])
        handler._relay_response(resp)

    def test_relay_swallows_upstream_error_mid_stream(self):
        # Upstream drops mid-body after headers are already sent — we can't signal
        # a fresh error, so stop quietly rather than corrupt the response framing.
        class _Ok(io.RawIOBase):
            def write(self, data):  # type: ignore[override]
                return len(data)

            def flush(self):
                return None

        def _chunks():
            yield b"partial"
            raise httpx.ReadError("upstream dropped")

        handler = _relay_handler(_Ok())
        resp = _FakeResponse(200, {}, _chunks())
        handler._relay_response(resp)  # must not raise

    def test_relay_forwards_status_and_skips_hop_by_hop_headers(self):
        # A non-200 status (e.g. 429 rate limit) and content headers are relayed;
        # hop-by-hop framing headers are dropped.
        chunks_written: list[bytes] = []

        class _Collect(io.RawIOBase):
            def write(self, data):  # type: ignore[override]
                chunks_written.append(bytes(data))
                return len(data)

            def flush(self):
                return None

        handler = _relay_handler(_Collect())
        resp = _FakeResponse(
            429,
            {"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
            [b'{"type":"error"}'],
        )
        handler._relay_response(resp)
        blob = b"".join(chunks_written)
        assert b"429" in blob
        assert b"Content-Type: application/json" in blob
        assert b"Transfer-Encoding" not in blob  # hop-by-hop, stripped
        assert resp.chunk_sizes == [None]  # relay each upstream SSE/network chunk immediately

    def test_diagnostics_identify_upstream_mid_stream_drop(self, monkeypatch, capsys):
        monkeypatch.setenv(gateway_proxy._DIAGNOSTICS_ENV, "1")

        class _Ok(io.RawIOBase):
            def write(self, data):  # type: ignore[override]
                return len(data)

            def flush(self):
                return None

        def _chunks():
            yield b"partial"
            raise httpx.ReadError("upstream dropped")

        handler = _relay_handler(_Ok())
        handler._relay_response(
            _FakeResponse(200, {}, _chunks()),
            diagnostic_id="local-id",
            started=time.monotonic(),
        )

        line = capsys.readouterr().err.strip()
        assert line.startswith("[ucode-relay] ")
        event = json.loads(line.removeprefix("[ucode-relay] "))
        assert event["event"] == "upstream_stream_error"
        assert event["request_id"] == "local-id"
        assert event["error_type"] == "ReadError"
        assert event["bytes"] == len(b"partial")
        assert "upstream dropped" not in line

    def test_diagnostics_are_silent_by_default(self, monkeypatch, capsys):
        monkeypatch.delenv(gateway_proxy._DIAGNOSTICS_ENV, raising=False)
        handler = _relay_handler(_Collect())
        handler._relay_response(_FakeResponse(200, {}, [b"ok"]))
        assert capsys.readouterr().err == ""


class TestJwtExp:
    def test_extracts_exp(self):
        assert gateway_proxy._jwt_exp(_make_jwt(1234567890.0)) == 1234567890.0

    def test_none_on_missing_exp(self):
        assert gateway_proxy._jwt_exp(_make_jwt(None)) is None

    def test_none_on_garbage(self):
        assert gateway_proxy._jwt_exp("not-a-jwt") is None


def _install_fake_token(monkeypatch, exp_offsets, delay=0.0):
    """Patch get_databricks_token to hand out JWTs whose exp is now+offset, one
    per successive mint (last offset repeats). Records the force flag of each."""
    state = {"i": 0, "forces": []}

    def fake(_ws, _profile, force_refresh=False):
        if delay:
            time.sleep(delay)
        off = exp_offsets[min(state["i"], len(exp_offsets) - 1)]
        state["i"] += 1
        state["forces"].append(force_refresh)
        return _make_jwt(time.time() + off)

    monkeypatch.setattr(gateway_proxy, "get_databricks_token", fake)
    return state


class TestTokenCache:
    def test_initial_mint_preserves_default_nonforce_refresh(self, monkeypatch):
        state = _install_fake_token(monkeypatch, [5000])
        gateway_proxy.TokenCache("ws", None)
        assert state["forces"] == [False]

    def test_fresh_token_is_not_refreshed(self, monkeypatch):
        state = _install_fake_token(monkeypatch, [5000])
        cache = gateway_proxy.TokenCache("ws", None)
        _ = cache.token
        _ = cache.token
        assert state["forces"] == [False]  # no extra mint while fresh

    def test_near_expiry_preserves_default_nonforce_refresh(self, monkeypatch):
        state = _install_fake_token(monkeypatch, [100, 5000])
        cache = gateway_proxy.TokenCache("ws", None)
        _ = cache.token
        assert state["forces"] == [False, False]
        _ = cache.token  # now fresh again
        assert state["forces"] == [False, False]

    def test_near_expiry_can_force_refresh(self, monkeypatch):
        state = _install_fake_token(monkeypatch, [100, 5000])
        cache = gateway_proxy.TokenCache("ws", None, force_refresh_near_expiry=True)
        _ = cache.token
        assert state["forces"] == [True, True]

    def test_refresh_is_single_flighted(self, monkeypatch):
        # A burst of concurrent requests at the expiry boundary must trigger ONE
        # refresh, not a thundering herd on the shared token cache.
        state = _install_fake_token(monkeypatch, [100, 5000], delay=0.05)
        cache = gateway_proxy.TokenCache("ws", None, force_refresh_near_expiry=True)
        threads = [threading.Thread(target=lambda: cache.token) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 1 forced init + exactly 1 forced refresh shared by all 10 readers.
        assert state["forces"] == [True, True]

    def test_ensure_fresh_keeps_token_when_refresh_fails(self, monkeypatch):
        _install_fake_token(monkeypatch, [5000])
        cache = gateway_proxy.TokenCache("ws", None)
        good = cache.token

        def boom(*_a, **_k):
            raise RuntimeError("mint failed")

        monkeypatch.setattr(gateway_proxy, "get_databricks_token", boom)
        # Force staleness so _ensure_fresh attempts a refresh, which now fails.
        cache._expiry = time.time()
        assert cache.token == good  # last good token retained, no exception

    def test_refresher_loop_survives_unexpected_error(self, monkeypatch):
        _install_fake_token(monkeypatch, [5000])
        cache = gateway_proxy.TokenCache("ws", None)
        monkeypatch.setattr(gateway_proxy, "_REFRESHER_POLL_S", 0.01)
        ticks = []

        def boom():
            ticks.append(1)
            cache.stop()  # let the loop exit after this tick
            raise ValueError("unexpected, non-RuntimeError")

        monkeypatch.setattr(cache, "_ensure_fresh", boom)
        cache.run_refresher()  # must RETURN, not propagate — else the thread dies
        assert ticks == [1]


class _FakeResp:
    def __init__(self, status: int, body: bytes = b"", headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.read_called = False

    def read(self):
        self.read_called = True
        return self._body

    def iter_raw(self, _n=None):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent_tokens: list[str | None] = []

    def stream(self, _method, _url, headers, content):
        self.sent_tokens.append(headers.get(gateway_proxy.AI_GATEWAY_TOKEN_HEADER))
        return self._responses.pop(0)


class _FakeCache:
    def __init__(self):
        self._t = "Bearer-tok1"
        self.refreshed = 0

    @property
    def token(self):
        return self._t.replace("Bearer-", "")

    def refresh(self):
        self.refreshed += 1
        self._t = "Bearer-tok2"


class _RefreshFailsCache:
    """A cache whose forced refresh raises, mimicking a dead Databricks OAuth
    session that can't be re-minted non-interactively."""

    def __init__(self):
        self.refreshed = 0

    @property
    def token(self):
        return "tok1"

    def refresh(self):
        self.refreshed += 1
        raise RuntimeError("mint failed")


def _handle_handler(client, cache, wfile) -> gateway_proxy._ProxyHandler:
    h = object.__new__(gateway_proxy._ProxyHandler)
    h.client = client
    h.cache = cache
    h.headers = {}
    h.rfile = io.BytesIO(b"")
    h.path = "/v1/messages"
    h.command = "POST"
    h.wfile = wfile
    h.request_version = "HTTP/1.1"
    h.requestline = "POST /v1/messages HTTP/1.1"
    h._headers_buffer = []
    return h


class _Collect(io.RawIOBase):
    def __init__(self):
        self.data = bytearray()

    def write(self, b):  # type: ignore[override]
        self.data += bytes(b)
        return len(b)

    def flush(self):
        return None


class TestRetryOn401:
    def test_401_forces_refresh_and_retries(self):
        # A stale swap token yields 401; the proxy force-refreshes and retries,
        # this time succeeding, so Claude Code never sees the 401.
        client = _FakeClient([_FakeResp(401, b'{"e":1}'), _FakeResp(200, b"ok")])
        cache = _FakeCache()
        out = _Collect()
        _handle_handler(client, cache, out)._handle()
        assert cache.refreshed == 1
        assert client.sent_tokens == ["Bearer tok1", "Bearer tok2"]  # retried with fresh token
        assert b"200" in bytes(out.data)
        assert b"ok" in bytes(out.data)

    def test_persistent_401_is_relayed(self):
        # If the retry also 401s, it's genuinely the Anthropic layer — relay it so
        # Claude Code re-auths Anthropic (correct), rather than looping forever.
        client = _FakeClient([_FakeResp(401, b"a"), _FakeResp(401, b"b")])
        cache = _FakeCache()
        out = _Collect()
        _handle_handler(client, cache, out)._handle()
        assert cache.refreshed == 1
        assert b"401" in bytes(out.data)

    def test_success_first_try_does_not_refresh(self):
        client = _FakeClient([_FakeResp(200, b"hi")])
        cache = _FakeCache()
        out = _Collect()
        _handle_handler(client, cache, out)._handle()
        assert cache.refreshed == 0
        assert client.sent_tokens == ["Bearer tok1"]
        assert b"hi" in bytes(out.data)

    def test_failed_refresh_surfaces_reauth_hint(self, capsys):
        # When the forced refresh itself fails (the Databricks OAuth session is
        # dead, not just the access token), the user must be told to run
        # `databricks auth login` rather than left with a bare 401 that reads as
        # an Anthropic `/login` prompt. We still retry + relay whatever comes.
        client = _FakeClient([_FakeResp(401, b"a"), _FakeResp(401, b"b")])
        cache = _RefreshFailsCache()
        out = _Collect()
        _handle_handler(client, cache, out)._handle()
        assert cache.refreshed == 1  # a refresh was attempted
        assert "databricks auth login" in capsys.readouterr().err
        assert b"401" in bytes(out.data)  # the response is still relayed


class TestStartProxyPortFallback:
    def test_falls_back_to_free_port_when_cached_port_busy(self, monkeypatch):
        # A stale proxy from a killed session can still hold the cached port; the
        # bind must fall back to an OS-assigned free port rather than crash.
        class _StubCache:
            def run_refresher(self):
                return None

        monkeypatch.setattr(
            gateway_proxy,
            "TokenCache",
            lambda workspace, profile, **_kwargs: _StubCache(),
        )
        # Occupy a port to simulate the leftover proxy holding it.
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = occupied.getsockname()[1]
        try:
            server, _cache, client = gateway_proxy.start_proxy(
                "https://x.staging.cloud.databricks.com",
                None,
                busy_port,
                token_header=gateway_proxy.AI_GATEWAY_TOKEN_HEADER,
                force_refresh_near_expiry=False,
            )
            try:
                bound = server.server_address[1]
                assert bound != busy_port  # fell back to a different, free port
                assert bound != 0
            finally:
                server.server_close()
                client.close()
        finally:
            occupied.close()
