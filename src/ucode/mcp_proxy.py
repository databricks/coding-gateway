"""`ucode mcp-proxy`: a stdio MCP server that bridges to a Databricks
streamable-HTTP MCP endpoint, injecting a freshly-minted OAuth bearer.

Every coding agent ucode configures points its Databricks MCP servers at this
one command (``ucode mcp-proxy --url <endpoint> --profile <profile>``) as a
local **stdio** server. The agent spawns and reaps the proxy as a child process
— ucode owns no long-lived process and no background refresh thread. The proxy
speaks stdio to the agent and streamable-HTTP to Databricks, and mints a fresh
token from the Databricks CLI profile on **every** upstream HTTP request via an
httpx ``Auth`` hook, so the bearer never goes stale mid-session.

This replaces the previous per-client header auth (static ``Bearer
${OAUTH_TOKEN}``, Claude ``headersHelper``, Cursor literal-token rewrites): one
uniform mechanism, token refresh in a single place, and the proxy is an
invisible implementation detail baked into each client's config.

MCP SDK compatibility (1.x and 2.x). The proxy uses the SDK's *2.x-native* call
shape — ``streamable_http_client(url, http_client=<AsyncClient>)`` — which both
mcp 1.28+ and mcp 2.x export (1.x's older ``streamablehttp_client`` is a thin
deprecated shim over it). The only thing that differs across the major versions
is the HTTP library: mcp 1.x builds on ``httpx``, mcp 2.x on ``httpx2``. We
resolve whichever the installed SDK uses (see ``_httpx``) and build the client
and auth from that, so a single code path works against both — no version cap.

Auth failures are terminal and are reported *fast*. When the Databricks CLI
can't mint a token (expired refresh token, logged-out profile), the proxy prints
the CLI's own message to stderr and exits ``AUTH_FAILURE_EXIT_CODE`` rather than
letting the client wait out its MCP startup timeout. Every server registered
against the same profile fails at once in that state, so a silent hang is
especially confusing -- the user needs to be told to re-run
``databricks auth login``.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType, TracebackType
from typing import Protocol, Self

import anyio
from mcp.client.streamable_http import streamable_http_client
from mcp.server.stdio import stdio_server

from ucode.databricks import ensure_pat_bearer, get_databricks_token

# Exit code used when the proxy cannot continue. MCP clients surface a non-zero
# exit far more usefully than a timeout, so bail out instead of hanging.
AUTH_FAILURE_EXIT_CODE = 2


def _httpx() -> ModuleType:
    """Return the httpx module the installed MCP SDK is built on.

    mcp 2.x moved from ``httpx`` to ``httpx2`` and passes an ``AsyncClient`` of
    that flavor into ``streamable_http_client``. We must construct our client and
    ``Auth`` from the *same* module the SDK uses, or the transport rejects it.
    Prefer ``httpx2`` (mcp 2.x) and fall back to ``httpx`` (mcp 1.x)."""
    # Imported dynamically: httpx2 ships only with mcp 2.x, so a static `import
    # httpx2` is unresolvable under the mcp 1.x that's pinned for type-checking
    # and CI. importlib keeps the type checker out of it while the runtime picks
    # the right flavor.
    from importlib import import_module

    try:
        return import_module("httpx2")
    except ImportError:
        return import_module("httpx")


class ProxyAuthError(RuntimeError):
    """The proxy could not mint a Databricks token, so it cannot serve requests.

    Kept distinct from a transport error: this one is terminal and actionable
    (the user must re-run `databricks auth login`), so `serve` reports it on
    stderr and exits rather than retrying."""


class ProxyTransportError(RuntimeError):
    """The upstream MCP transport failed or closed unexpectedly."""


class _ReceiveStream[T](Protocol):
    def __aiter__(self) -> AsyncIterator[T]: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


class _SendStream[T](Protocol):
    async def send(self, item: T, /) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


def _fail_fast(message: str) -> None:
    """Report a terminal proxy failure on stderr and exit non-zero.

    stdout is the MCP wire, so diagnostics must go to stderr — MCP clients
    surface a child's stderr when it fails to start."""
    print(f"ucode mcp-proxy: {message}", file=sys.stderr, flush=True)
    raise SystemExit(AUTH_FAILURE_EXIT_CODE)


def _build_token_auth(workspace: str, profile: str | None):
    """Build an httpx ``Auth`` that injects a fresh bearer on every request.

    The base class comes from whichever httpx the SDK uses (see ``_httpx``), so
    the returned auth is accepted by that SDK's ``AsyncClient``. Behaviour is
    identical across flavours — ``Auth.auth_flow`` has the same generator
    contract in httpx and httpx2."""
    httpx = _httpx()

    class _DatabricksTokenAuth(httpx.Auth):
        def auth_flow(self, request):
            # get_databricks_token honors the DATABRICKS_BEARER short-circuit and
            # PAT profiles internally; --use-pat is surfaced via the env ucode set.
            # A RuntimeError here means auth is dead (expired refresh token,
            # logged-out profile). Raising it from inside auth_flow would tear
            # through the transport's task group and stall the process until the
            # client times out, so translate it into a terminal ProxyAuthError the
            # caller reports cleanly.
            try:
                token = get_databricks_token(workspace, profile)
            except RuntimeError as exc:
                raise ProxyAuthError(str(exc)) from exc
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    return _DatabricksTokenAuth()


async def _pump[T](
    source: _ReceiveStream[T],
    dest: _SendStream[T],
) -> None:
    """Forward every message from ``source`` to ``dest``.

    The proxy is transport-level: it never inspects or rewrites MCP method
    payloads, so new methods and capabilities pass through untouched."""
    async with source, dest:
        async for message in source:
            await dest.send(message)


async def _pump_upstream[T](
    source: _ReceiveStream[T | Exception],
    dest: _SendStream[T],
) -> None:
    """Forward upstream messages, failing if the transport closes first."""
    async with source, dest:
        async for message in source:
            if isinstance(message, Exception):
                detail = " ".join(str(message).split()) or type(message).__name__
                raise ProxyTransportError(f"upstream MCP transport failed: {detail}") from message
            await dest.send(message)
    raise ProxyTransportError("upstream MCP transport closed unexpectedly")


async def _run(url: str, workspace: str, profile: str | None) -> None:
    httpx = _httpx()
    auth = _build_token_auth(workspace, profile)
    # 2.x-native shape: hand the transport a pre-built AsyncClient carrying our
    # per-request auth. Works on mcp 1.28+ and 2.x; `streamable_http_client`
    # yields a (read, write) pair in both.
    async with httpx.AsyncClient(
        auth=auth,
        timeout=httpx.Timeout(
            connect=30.0,
            read=300.0,
            write=30.0,
            pool=30.0,
        ),
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            # mcp 1.x yields (read, write, get_session_id); mcp 2.x drops the
            # trailing callback and yields (read, write). Take the first two
            # positionally so both arities work — we don't use get_session_id.
            http_read, http_write = streams[0], streams[1]
            async with stdio_server() as (stdio_read, stdio_write):
                # Bidirectional bridge: client stdin -> Databricks, Databricks -> client stdout.
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_pump_upstream, http_read, stdio_write)
                    await _pump(stdio_read, http_write)
                    tg.cancel_scope.cancel()


def _preflight_token(workspace: str, profile: str | None) -> None:
    """Verify a Databricks token can be minted before opening the bridge.

    Raises ``RuntimeError`` (with the CLI's own message) when auth is dead. This
    is a plain synchronous call: ``get_databricks_token`` already bounds itself
    with subprocess timeouts, so it returns or fails on its own — the point here
    is only to *locate* the failure before the transport starts, where it can be
    reported instead of stalling the session."""
    get_databricks_token(workspace, profile)


def _unwrap_proxy_error(exc: BaseException) -> ProxyAuthError | ProxyTransportError | None:
    """Find a known proxy error in an exception (or ExceptionGroup) tree.

    anyio task groups wrap failures in ExceptionGroups, so failures raised
    inside the transport arrive nested rather than as themselves."""
    if isinstance(exc, (ProxyAuthError, ProxyTransportError)):
        return exc
    for nested in getattr(exc, "exceptions", ()) or ():
        found = _unwrap_proxy_error(nested)
        if found is not None:
            return found
    return None


def serve(url: str, workspace: str, profile: str | None = None, *, use_pat: bool = False) -> None:
    """Run the stdio<->streamable-HTTP MCP proxy until the client closes stdin.

    Authentication is checked up front: a dead profile is a terminal condition,
    and failing here (fast, with the CLI's own message) is far better than
    letting the client wait out its MCP startup timeout with no explanation.

    ``use_pat`` selects static personal-access-token auth: ``databricks auth
    token`` only reads OAuth caches, so a PAT profile's token must be exported as
    ``DATABRICKS_BEARER`` first (``ensure_pat_bearer``) — then every per-request
    mint takes that short-circuit. OAuth needs no such step."""
    if use_pat and not ensure_pat_bearer(profile):
        _fail_fast(
            "--use-pat is set but no personal access token was found for profile "
            f"'{profile or '<none>'}' in ~/.databrickscfg (expected auth_type = pat). "
            "Set DATABRICKS_BEARER, or reconfigure the profile."
        )

    # Pre-flight the token before opening the bridge. Without this, the first
    # token failure surfaces from inside the transport's task group, where it can
    # stall the process instead of erroring out.
    try:
        _preflight_token(workspace, profile)
    except RuntimeError as exc:
        _fail_fast(str(exc))

    try:
        anyio.run(_run, url, workspace, profile)
    except BaseException as exc:  # noqa: BLE001 - re-raised unless it's a known proxy failure
        # Errors raised inside the transport arrive wrapped by its task group.
        # Report expected auth/transport failures without hiding programming bugs.
        proxy_error = _unwrap_proxy_error(exc)
        if proxy_error is None:
            raise
        _fail_fast(str(proxy_error))


__all__ = ["AUTH_FAILURE_EXIT_CODE", "ProxyAuthError", "ProxyTransportError", "serve"]
