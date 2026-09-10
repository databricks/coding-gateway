"""Loopback refresh proxy for Claude gateway requests.

A relayed Model Provider Service authenticates the caller's own Anthropic
subscription OAuth (which Claude Code owns in the `Authorization` header) and
carries a Databricks credential in the `X-Databricks-AI-Gateway-Token` swap
header. Native gateway discovery instead carries the Databricks credential in
`Authorization`. The proxy refreshes the applicable header and streams responses
back verbatim.

Security invariants (mirroring `databricks.py` token handling):
  - Binds 127.0.0.1 only; never exposed off-host.
  - Never logs header values or bodies. The Databricks token lives in memory,
    refreshed off the request path; the Anthropic OAuth in `Authorization` is
    passed through untouched in relayed mode and never logged.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from ucode.databricks import get_databricks_token

# Header we overwrite with the freshly-minted Databricks credential. Any
# client-supplied value is replaced, so a stale settings.json value can't leak.
AI_GATEWAY_TOKEN_HEADER = "X-Databricks-AI-Gateway-Token"
AUTHORIZATION_HEADER = "Authorization"
# Hop-by-hop headers must not be forwarded across a proxy.
HOP_BY_HOP_HEADERS = frozenset(
    h.lower()
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    )
)
# Per-operation upstream timeouts. `read` is generous because model turns stream
# over a single response and Anthropic emits SSE pings, so inter-chunk gaps stay
# small; `connect`/`pool` fail fast when the gateway is unreachable.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)
# Refresh once the token has less than this many seconds of life left. Databricks
# access tokens live ~1h; a 10-min buffer leaves ample headroom for a retry.
_REFRESH_BUFFER_S = 600
# How often the background thread re-checks freshness. Cheap: it only shells out
# to the CLI when actually within the buffer, otherwise it's a bare clock compare.
_REFRESHER_POLL_S = 120
# Assumed lifetime when a token carries no decodable `exp` (defensive fallback).
_DEFAULT_TTL_S = 3600
# Opt-in transport diagnostics for intermittent streaming failures. Events only
# contain locally-generated request ids, timings, status codes, byte counts,
# and exception class names — never headers, bodies, or credentials.
_DIAGNOSTICS_ENV = "UCODE_RELAYED_PROXY_DIAGNOSTICS"
_DIAGNOSTICS_TRUE = frozenset({"1", "true", "yes", "on"})


def _diagnostics_enabled() -> bool:
    return os.environ.get(_DIAGNOSTICS_ENV, "").strip().lower() in _DIAGNOSTICS_TRUE


def log_proxy_diagnostic(event: str, **fields: object) -> None:
    if not _diagnostics_enabled():
        return
    payload = {"event": event, **fields}
    sys.stderr.write(
        f"[ucode-relay] {json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n"
    )
    sys.stderr.flush()


def _jwt_exp(token: str) -> float | None:
    """Best-effort `exp` (epoch seconds) from a JWT access token, else None."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore base64 padding
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except (IndexError, ValueError, KeyError, binascii.Error, json.JSONDecodeError):
        return None


def log_token_refresh_failure(exc: BaseException) -> None:
    """Surface (never silently swallow) a refresh failure, without leaking any
    token or header value."""
    sys.stderr.write(
        f"[ucode] Databricks token refresh failed: {exc}. If the session stalls, "
        "run `databricks auth login` for your workspace profile.\n"
    )


class TokenCache:
    """Holds the current Databricks token and its expiry, refreshing lazily as it
    nears expiry.

    A background thread refreshes proactively so the request path rarely blocks,
    but the request path also refreshes on demand — which is what carries the
    token across events the timer can't (laptop sleep suspends the monotonic
    clock, so a fixed interval silently stops advancing). All refreshes are
    single-flighted through ``_refresh_lock`` so a burst of requests at the expiry
    boundary triggers exactly one CLI call, not a thundering herd on the shared
    token cache."""

    def __init__(
        self,
        workspace: str,
        profile: str | None,
        *,
        force_refresh_near_expiry: bool = False,
    ) -> None:
        self._workspace = workspace
        self._profile = profile
        self._force_refresh_near_expiry = force_refresh_near_expiry
        self._state_lock = threading.Lock()  # guards _token / _expiry (brief)
        self._refresh_lock = threading.Lock()  # single-flights the CLI refresh
        self._stop = threading.Event()
        self._token = ""
        self._expiry = 0.0
        # Preserve the existing non-forced relayed-auth fetch. Gateway discovery
        # opts into a forced fetch so its static client token starts with a full TTL.
        self._refresh(force=force_refresh_near_expiry)

    def _refresh(self, *, force: bool) -> None:
        """Mint a token and record its expiry."""
        token = get_databricks_token(self._workspace, self._profile, force_refresh=force)
        expiry = _jwt_exp(token) or (time.time() + _DEFAULT_TTL_S)
        with self._state_lock:
            self._token = token
            self._expiry = expiry

    def _fresh_enough(self) -> bool:
        with self._state_lock:
            return bool(self._token) and time.time() < self._expiry - _REFRESH_BUFFER_S

    def _ensure_fresh(self) -> None:
        if self._fresh_enough():
            return
        with self._refresh_lock:
            if self._fresh_enough():  # another thread refreshed while we waited
                return
            try:
                self._refresh(force=self._force_refresh_near_expiry)
            except RuntimeError as exc:
                # Keep serving the current token; a request that then 401s triggers
                # a forced refresh + retry (see _ProxyHandler._handle).
                log_token_refresh_failure(exc)

    @property
    def token(self) -> str:
        self._ensure_fresh()
        with self._state_lock:
            return self._token

    def refresh(self) -> None:
        """Force a fresh mint now (used by the retry-on-401 path)."""
        with self._refresh_lock:
            self._refresh(force=True)

    def run_refresher(self) -> None:
        while not self._stop.wait(_REFRESHER_POLL_S):
            try:
                self._ensure_fresh()
            except Exception as exc:  # noqa: BLE001 - a stray error must NOT kill the thread
                # If this thread dies, nothing refreshes and the session lapses at
                # the ~1h mark until restart. Log and keep looping instead.
                log_token_refresh_failure(exc)

    def stop(self) -> None:
        self._stop.set()


def forwarded_request_headers(
    handler: BaseHTTPRequestHandler,
    token: str,
    token_header: str = AI_GATEWAY_TOKEN_HEADER,
) -> dict[str, str]:
    strip_on_forward = HOP_BY_HOP_HEADERS | {token_header.lower()}
    headers = {
        key: value for key, value in handler.headers.items() if key.lower() not in strip_on_forward
    }
    headers[token_header] = f"Bearer {token}"
    return headers


class _ProxyHandler(BaseHTTPRequestHandler):
    # Set by the server factory.
    cache: TokenCache
    client: httpx.Client
    token_header = AI_GATEWAY_TOKEN_HEADER

    def log_message(self, format: str, *args: object) -> None:
        return

    def _safe_send_error(self, code: int, message: str) -> None:
        # The client (Claude Code) may already have disconnected, in which case
        # reporting the error writes to a dead socket and raises again; swallow it.
        try:
            self.send_error(code, message)
        except OSError:
            pass

    def _handle(self) -> None:
        diagnostic_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        url = self.path.lstrip("/")
        log_proxy_diagnostic(
            "request_start",
            request_id=diagnostic_id,
            method=self.command,
            path=self.path.split("?", 1)[0],
        )
        try:
            # First attempt with the current token.
            headers = forwarded_request_headers(self, self.cache.token, self.token_header)
            with self.client.stream(self.command, url, headers=headers, content=body) as resp:
                log_proxy_diagnostic(
                    "upstream_headers",
                    request_id=diagnostic_id,
                    attempt=1,
                    status=resp.status_code,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
                if resp.status_code not in (401, 403):
                    self._relay_response(resp, diagnostic_id=diagnostic_id, started=started)
                    return
                # Auth rejected. Drain the (small) error body so the pooled
                # connection can be reused, then fall through to one retry.
                resp.read()
            # A relayed 401/403 may be a stale Databricks swap token rather than a
            # bad Anthropic OAuth — the two are indistinguishable from the status
            # alone. Force-refresh the Databricks token and retry once. If it was the
            # Anthropic layer, the retry still 401s and we relay it verbatim, so a
            # genuine re-auth is triggered; a stale-Databricks 401 self-heals here
            # instead of surfacing to Claude Code as a spurious Anthropic prompt.
            try:
                self.cache.refresh()
            except RuntimeError as exc:
                # Refresh failed: the Databricks OAuth session is dead (not just the
                # access token) and can't be re-minted non-interactively. Surface the
                # `databricks auth login` hint rather than silently relaying a bare 401,
                # which otherwise reads as an Anthropic `/login` prompt and sends the
                # user to the wrong re-auth. Still retry + relay with the existing token.
                log_token_refresh_failure(exc)
            headers = forwarded_request_headers(self, self.cache.token, self.token_header)
            with self.client.stream(self.command, url, headers=headers, content=body) as resp:
                log_proxy_diagnostic(
                    "upstream_headers",
                    request_id=diagnostic_id,
                    attempt=2,
                    status=resp.status_code,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
                self._relay_response(resp, diagnostic_id=diagnostic_id, started=started)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed before/while we relayed headers — routine on cancel.
            log_proxy_diagnostic(
                "client_disconnect",
                request_id=diagnostic_id,
                phase="request",
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return
        except httpx.HTTPError as exc:
            # Upstream failed before any bytes reached the client; a 502 is still
            # sendable. (An HTTP *status* like 429 is not an error here — httpx
            # only raises for transport failures — so real gateway errors are
            # relayed verbatim by `_relay_response`.)
            log_proxy_diagnostic(
                "upstream_request_error",
                request_id=diagnostic_id,
                error_type=type(exc).__name__,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            self._safe_send_error(502, "gateway proxy upstream error")

    # Streaming passthrough: forward chunks as they arrive so SSE token streaming
    # is not buffered (buffering would add full-response latency to first token).
    # `iter_raw` preserves any Content-Encoding verbatim (we relay that header),
    # so the proxy stays byte-transparent.
    def _relay_response(
        self,
        resp: httpx.Response,
        *,
        diagnostic_id: str | None = None,
        started: float | None = None,
    ) -> None:
        started = started if started is not None else time.monotonic()
        chunks = 0
        bytes_relayed = 0
        first_byte_ms: int | None = None
        try:
            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.end_headers()
            # Do not pass a fixed chunk size here. httpx accumulates bytes until
            # that size is reached, which can hide small SSE heartbeat frames
            # from Claude Code for minutes during a slow artifact/tool call.
            # With ``chunk_size=None`` (the default), raw upstream chunks are
            # yielded as they arrive and pings keep the downstream connection
            # alive even before the model produces a large content block.
            for chunk in resp.iter_raw():
                if chunk:
                    if first_byte_ms is None:
                        first_byte_ms = round((time.monotonic() - started) * 1000)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    chunks += 1
                    bytes_relayed += len(chunk)
            log_proxy_diagnostic(
                "response_complete",
                request_id=diagnostic_id,
                status=resp.status_code,
                chunks=chunks,
                bytes=bytes_relayed,
                first_byte_ms=first_byte_ms,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        except (BrokenPipeError, ConnectionResetError):
            # Client (Claude Code) closed the connection mid-response — routine on
            # cancelled turns / SSE teardown. Nothing left to relay to, so stop
            # quietly rather than crashing the handler thread.
            log_proxy_diagnostic(
                "client_disconnect",
                request_id=diagnostic_id,
                phase="response",
                chunks=chunks,
                bytes=bytes_relayed,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return
        except httpx.HTTPError as exc:
            # Upstream dropped mid-stream. Headers (and status) may already be
            # sent, so we can't reliably signal a fresh error — stop and let the
            # client see a truncated stream rather than corrupt the framing.
            log_proxy_diagnostic(
                "upstream_stream_error",
                request_id=diagnostic_id,
                error_type=type(exc).__name__,
                status=resp.status_code,
                chunks=chunks,
                bytes=bytes_relayed,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            return

    # Forward every method: this is a transparent pass-through, so routing any
    # `do_<METHOD>` lookup to `_handle` lets the gateway reject unsupported methods.
    def __getattr__(self, name: str):
        if name.startswith("do_"):
            return self._handle
        raise AttributeError(name)


def start_proxy(
    workspace: str,
    profile: str | None,
    port: int,
    token_header: str,
    force_refresh_near_expiry: bool,
) -> tuple[ThreadingHTTPServer, TokenCache, httpx.Client]:
    """Start the loopback refresh proxy + its background token refresher.

    Binds ``port``, falling back to a fresh OS-assigned port when it is already
    in use (e.g. a prior session's proxy that was killed before its teardown ran
    still holds the socket). The caller reads ``server.server_address[1]`` for the
    actual port and points Claude Code at it.

    Returns (server, cache, client); the caller runs the server (e.g. in a
    thread) and calls shutdown()/cache.stop()/client.close() on exit.
    """
    upstream_base = f"{workspace.rstrip('/')}/ai-gateway/anthropic/"
    cache = TokenCache(
        workspace,
        profile,
        force_refresh_near_expiry=force_refresh_near_expiry,
    )
    # One pooled, keep-alive client shared across handler threads: reuses TCP+TLS
    # to the gateway instead of a fresh handshake per request. Don't follow
    # redirects — a proxy relays 3xx verbatim.
    client = httpx.Client(base_url=upstream_base, timeout=UPSTREAM_TIMEOUT, follow_redirects=False)

    handler = type(
        "BoundProxyHandler",
        (_ProxyHandler,),
        {"cache": cache, "client": client, "token_header": token_header},
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        # Cached port is occupied (stale proxy from a killed session). Port 0 lets
        # the OS pick any free port; the caller reconciles the base URL to it.
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    refresher = threading.Thread(target=cache.run_refresher, daemon=True)
    refresher.start()
    return server, cache, client
