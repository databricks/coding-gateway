"""End-to-end tests for the relayed refresh proxy as a running server.

`test_gateway_proxy.py` covers the proxy's pieces with handler-level fakes
(`object.__new__(_ProxyHandler)`, `_FakeClient`, `_FakeResponse`) — the header
builder, the relay path, the token cache, and the retry-on-401 logic. What none
of those exercise is the whole thing wired together over real sockets: the
`ThreadingHTTPServer` bind, the `do_<METHOD>` dispatch, reading the body off a
real `rfile`, the pooled `httpx` client streaming to a real upstream and back,
and `_relay_response` writing to a real `wfile`.

These tests stand up a fake AI Gateway upstream, start the *real* proxy via
`start_proxy` pointed at it (patching only the Databricks token mint), and drive
it with a real HTTP client — so a regression anywhere in that chain is caught.
Fully hermetic: no agent binary, no network, no workspace credentials.
"""

from __future__ import annotations

import contextlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from ucode import gateway_proxy


class _CapturedRequest:
    def __init__(self, method: str, path: str, headers: dict[str, str], body: bytes):
        self.method = method
        self.path = path
        # Keyed lowercase so assertions don't depend on header-case normalization.
        self.headers = headers
        self.body = body

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


class _FakeGateway:
    """Real HTTP server standing in for the workspace AI Gateway upstream.

    Records each forwarded request and returns a scripted response: a status code
    plus a list of body byte-chunks (flushed with an optional inter-chunk delay so
    a streaming relay is exercised, not just a single write)."""

    def __init__(
        self, status: int = 200, chunks: list[bytes] | None = None, sse_delay: float = 0.0
    ):
        self.requests: list[_CapturedRequest] = []
        self._status = status
        self._chunks = chunks if chunks is not None else [b'{"ok":true}']
        self._sse_delay = sse_delay
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        captured = self.requests
        status, chunks, sse_delay = self._status, self._chunks, self._sse_delay

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                captured.append(
                    _CapturedRequest(
                        method=self.command,
                        path=self.path,
                        headers={k.lower(): v for k, v in self.headers.items()},
                        body=body,
                    )
                )
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if sse_delay:
                        time.sleep(sse_delay)

            def do_GET(self):  # noqa: N802
                self._serve()

            def do_POST(self):  # noqa: N802
                self._serve()

            def log_message(self, format, *args):  # noqa: A002
                pass

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


@pytest.fixture
def make_gateway():
    """Factory that starts fake upstreams and tears them all down at test end."""
    gateways: list[_FakeGateway] = []

    def _make(**kwargs) -> _FakeGateway:
        gw = _FakeGateway(**kwargs)
        gw.start()
        gateways.append(gw)
        return gw

    yield _make
    for gw in gateways:
        gw.stop()


def _counting_token(value: str = "dbx-swap-token"):
    """A get_databricks_token stand-in that records the force flag of each mint.

    Returns a plain (non-JWT) token, so `_jwt_exp` yields None and the cache falls
    back to the default TTL — the background refresher then never re-mints, keeping
    the mint count deterministic (one on init, one per forced retry-refresh)."""
    calls: list[bool] = []

    def fn(_workspace, _profile, force_refresh=False):
        calls.append(force_refresh)
        return value

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


@contextlib.contextmanager
def _running_proxy(gateway: _FakeGateway, monkeypatch, token_fn=None):
    """Start the real proxy pointed at `gateway`, yield its loopback URL, tear down."""
    monkeypatch.setattr(gateway_proxy, "get_databricks_token", token_fn or _counting_token())
    server, cache, client = gateway_proxy.start_proxy(
        gateway.base_url,
        None,
        0,
        token_header=gateway_proxy.AI_GATEWAY_TOKEN_HEADER,
        force_refresh_near_expiry=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        cache.stop()
        client.close()
        thread.join(timeout=2)


class TestRelayedProxyEndToEnd:
    def test_forwards_request_with_swap_header_and_passthrough(self, make_gateway, monkeypatch):
        # The whole relayed data-plane over real sockets: the proxy injects a fresh
        # swap token, passes the caller's Anthropic OAuth + the MPS routing header
        # through untouched, forwards the body verbatim, and composes the upstream
        # path under /ai-gateway/anthropic/.
        gw = make_gateway()
        with _running_proxy(gw, monkeypatch, _counting_token("swap-tok")) as proxy_url:
            resp = httpx.post(
                f"{proxy_url}/v1/messages",
                headers={
                    "Authorization": "Bearer anthropic-oauth",
                    "Databricks-Model-Provider-Service": "main.mcao.anthropic-mps",
                },
                content=b'{"model":"claude","stream":true}',
                timeout=10,
            )
        assert resp.status_code == 200
        assert resp.content == b'{"ok":true}'
        req = gw.requests[-1]
        assert req.method == "POST"
        assert req.path == "/ai-gateway/anthropic/v1/messages"
        assert req.header("X-Databricks-AI-Gateway-Token") == "Bearer swap-tok"
        assert req.header("Authorization") == "Bearer anthropic-oauth"
        assert req.header("Databricks-Model-Provider-Service") == "main.mcao.anthropic-mps"
        assert req.body == b'{"model":"claude","stream":true}'

    def test_streams_sse_chunks_back_in_order(self, make_gateway, monkeypatch):
        # A relayed model turn streams SSE; the proxy must relay chunks through
        # rather than buffering the whole response. Assert the client receives the
        # full stream, in order, over a real socket.
        chunks = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n", b"event: c\ndata: 3\n\n"]
        gw = make_gateway(chunks=chunks, sse_delay=0.02)
        with _running_proxy(gw, monkeypatch) as proxy_url:
            with httpx.Client(timeout=10) as client:
                with client.stream(
                    "POST",
                    f"{proxy_url}/v1/messages",
                    headers={"Authorization": "Bearer oauth"},
                    content=b"{}",
                ) as resp:
                    assert resp.status_code == 200
                    body = b"".join(resp.iter_raw())
        assert body == b"".join(chunks)

    def test_upstream_401_triggers_refresh_and_relays_over_socket(self, make_gateway, monkeypatch):
        # A 401 may be a stale swap token, so the proxy force-refreshes and retries
        # once; when the retry still 401s it's genuinely the Anthropic layer and the
        # 401 is relayed verbatim (Claude Code then re-auths Anthropic). Exercised
        # here end-to-end over real sockets, not just the _handle fake path.
        token_fn = _counting_token("swap-tok")
        gw = make_gateway(status=401, chunks=[b'{"type":"error"}'])
        with _running_proxy(gw, monkeypatch, token_fn) as proxy_url:
            resp = httpx.post(
                f"{proxy_url}/v1/messages",
                headers={"Authorization": "Bearer oauth"},
                content=b"{}",
                timeout=10,
            )
        assert resp.status_code == 401
        # Init mint isn't forced (force_refresh_near_expiry=False); the first 401 is
        # what forces a fresh mint before the single retry.
        assert token_fn.calls == [False, True]  # type: ignore[attr-defined]
        assert len(gw.requests) == 2  # original attempt + one retry
