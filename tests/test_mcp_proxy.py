"""Tests for the `ucode mcp-proxy` stdio<->streamable-HTTP bridge."""

from __future__ import annotations

import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
import pytest

from ucode import mcp_proxy

WS = "https://example.databricks.com"
URL = f"{WS}/api/2.0/mcp/functions/system/ai"


def _runtime_dependencies() -> list[str]:
    project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())[
        "project"
    ]
    return project["dependencies"]


def test_httpx_is_a_direct_runtime_dependency():
    assert any(dependency.startswith("httpx") for dependency in _runtime_dependencies())


def test_mcp_dependency_is_uncapped():
    # The proxy works against both mcp 1.x (httpx) and mcp 2.x (httpx2) via a
    # single code path (see mcp_proxy._httpx / _run), so `mcp` must NOT be capped
    # below 2 — a fresh install resolving mcp 2.x should work, not break (#307).
    mcp_specs = [d for d in _runtime_dependencies() if d.replace(" ", "").startswith("mcp")]
    assert mcp_specs, "mcp must be a direct dependency"
    assert not any("<2" in spec.replace(" ", "") for spec in mcp_specs), mcp_specs


def test_selected_httpx_matches_the_installed_mcp_sdk():
    # `_httpx()` must return the HTTP flavor the installed SDK is built on:
    # httpx2 for mcp 2.x, httpx for mcp 1.x. Using the wrong one makes the
    # transport reject our AsyncClient/Auth.
    from importlib.metadata import version

    from packaging.version import Version

    selected = mcp_proxy._httpx().__name__
    if Version(version("mcp")) >= Version("2"):
        assert selected == "httpx2"
    else:
        assert selected == "httpx"


def test_proxy_imports_the_streamable_http_client_shared_by_both_majors():
    # `streamable_http_client` (2.x-native, and re-exported by mcp 1.28+) is the
    # symbol the single code path relies on; a rename in a resolved SDK fails here.
    assert mcp_proxy.streamable_http_client is not None
    assert mcp_proxy.stdio_server is not None


class TestDatabricksTokenAuth:
    def test_injects_bearer_from_minted_token(self, monkeypatch):
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "tok-123")
        auth = mcp_proxy._build_token_auth(WS, "uc-dogfood")

        request = httpx.Request("POST", URL)
        # auth_flow is a generator that yields the (mutated) request.
        list(auth.auth_flow(request))

        assert request.headers["Authorization"] == "Bearer tok-123"

    def test_auth_is_an_instance_of_the_selected_httpx_auth(self, monkeypatch):
        # The auth must subclass the *same* httpx flavor's Auth as the transport,
        # or the SDK's AsyncClient won't accept it.
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "t")
        auth = mcp_proxy._build_token_auth(WS, None)

        assert isinstance(auth, mcp_proxy._httpx().Auth)

    def test_calls_get_token_with_workspace_and_profile(self, monkeypatch):
        calls: list[tuple[str, str | None]] = []
        monkeypatch.setattr(
            mcp_proxy,
            "get_databricks_token",
            lambda ws, profile: calls.append((ws, profile)) or "t",
        )
        auth = mcp_proxy._build_token_auth(WS, "myprofile")

        list(auth.auth_flow(httpx.Request("POST", URL)))

        assert calls == [(WS, "myprofile")]

    def test_mints_a_fresh_token_per_request(self, monkeypatch):
        # Each request re-invokes get_databricks_token, so a rotated token is
        # picked up mid-session without the proxy tracking expiry itself.
        tokens = iter(["first", "second"])
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: next(tokens))
        auth = mcp_proxy._build_token_auth(WS, None)

        r1 = httpx.Request("POST", URL)
        r2 = httpx.Request("POST", URL)
        list(auth.auth_flow(r1))
        list(auth.auth_flow(r2))

        assert r1.headers["Authorization"] == "Bearer first"
        assert r2.headers["Authorization"] == "Bearer second"

    def test_auth_flow_yields_the_same_request(self, monkeypatch):
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "t")
        auth = mcp_proxy._build_token_auth(WS, None)

        request = httpx.Request("POST", URL)
        yielded = list(auth.auth_flow(request))

        assert yielded == [request]

    def test_dead_auth_becomes_a_terminal_proxy_auth_error(self, monkeypatch):
        # A raw RuntimeError escaping auth_flow tears through the transport's task
        # group and stalls the proxy until the client's startup timeout.
        # Translating it keeps the failure reportable by `serve`.
        def boom(ws, profile):
            raise RuntimeError("no access token; run `databricks auth login`")

        monkeypatch.setattr(mcp_proxy, "get_databricks_token", boom)
        auth = mcp_proxy._build_token_auth(WS, "p")

        with pytest.raises(mcp_proxy.ProxyAuthError, match="databricks auth login"):
            list(auth.auth_flow(httpx.Request("POST", URL)))


class TestPump:
    def test_forwards_all_messages_in_order(self):
        async def scenario() -> list[str]:
            src_send, src_recv = anyio.create_memory_object_stream(10)
            dst_send, dst_recv = anyio.create_memory_object_stream(10)
            # Preload the source, then close its send end so _pump's `async for`
            # terminates once drained.
            for msg in ["a", "b", "c"]:
                await src_send.send(msg)
            await src_send.aclose()

            await mcp_proxy._pump(src_recv, dst_send)

            received: list[str] = []
            # _pump closed dst_send on exit, so this drains then stops.
            async with dst_recv:
                async for msg in dst_recv:
                    received.append(msg)
            return received

        assert anyio.run(scenario) == ["a", "b", "c"]

    def test_closes_destination_when_source_exhausts(self):
        # A closed dest send-stream is what lets the *other* pump's reader
        # terminate, so the bridge tears down cleanly when one side hangs up.
        async def scenario() -> bool:
            src_send, src_recv = anyio.create_memory_object_stream(1)
            dst_send, dst_recv = anyio.create_memory_object_stream(1)
            await src_send.aclose()

            await mcp_proxy._pump(src_recv, dst_send)

            with pytest.raises(anyio.EndOfStream):
                dst_recv.receive_nowait()
            return True

        assert anyio.run(scenario) is True

    def test_client_errors_are_forwarded(self):
        async def scenario() -> Exception:
            src_send, src_recv = anyio.create_memory_object_stream(1)
            dst_send, dst_recv = anyio.create_memory_object_stream(1)
            error = ValueError("malformed client message")
            await src_send.send(error)
            await src_send.aclose()

            await mcp_proxy._pump(src_recv, dst_send)
            return await dst_recv.receive()

        error = anyio.run(scenario)
        assert isinstance(error, ValueError)
        assert str(error) == "malformed client message"

    def test_upstream_errors_are_raised(self):
        async def scenario() -> None:
            src_send, src_recv = anyio.create_memory_object_stream(1)
            dst_send, _ = anyio.create_memory_object_stream(1)
            await src_send.send(httpx.ReadTimeout("upstream timed out"))
            await src_send.aclose()

            with pytest.raises(mcp_proxy.ProxyTransportError, match="upstream timed out"):
                await mcp_proxy._pump_upstream(src_recv, dst_send)

        anyio.run(scenario)

    def test_upstream_eof_is_an_error(self):
        async def scenario() -> None:
            src_send, src_recv = anyio.create_memory_object_stream(1)
            dst_send, _ = anyio.create_memory_object_stream(1)
            await src_send.aclose()

            with pytest.raises(
                mcp_proxy.ProxyTransportError, match="upstream MCP transport closed"
            ):
                await mcp_proxy._pump_upstream(src_recv, dst_send)

        anyio.run(scenario)


def test_run_uses_mcp_http_defaults(monkeypatch):
    httpx_module = mcp_proxy._httpx()
    captured: dict = {}

    class CapturingClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class StopBridge(Exception):
        pass

    @asynccontextmanager
    async def stop_bridge(*args, **kwargs):
        raise StopBridge
        yield

    monkeypatch.setattr(httpx_module, "AsyncClient", CapturingClient)
    monkeypatch.setattr(mcp_proxy, "_build_token_auth", lambda *args: object())
    monkeypatch.setattr(mcp_proxy, "streamable_http_client", stop_bridge)

    with pytest.raises(StopBridge):
        anyio.run(mcp_proxy._run, URL, WS, None)

    timeout = captured["timeout"]
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (30.0, 300.0, 30.0, 30.0)


class TestServe:
    def test_runs_the_bridge_with_parsed_args(self, monkeypatch):
        captured: dict = {}

        def fake_run(func, *args):
            captured["func"] = func
            captured["args"] = args

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", fake_run)

        mcp_proxy.serve(URL, WS, "uc-dogfood")

        assert captured["func"] is mcp_proxy._run
        # PAT is resolved inside the token mint, so _run takes no use_pat arg.
        assert captured["args"] == (URL, WS, "uc-dogfood")

    def test_defaults_profile_none(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: captured.update(args=args))

        mcp_proxy.serve(URL, WS)

        assert captured["args"] == (URL, WS, None)

    def test_use_pat_exports_the_bearer_before_serving(self, monkeypatch):
        # PAT auth: the profile's static token must be exported (ensure_pat_bearer)
        # before the token preflight, so the per-request mint's DATABRICKS_BEARER
        # short-circuit returns it. `databricks auth token` can't read a PAT itself.
        order: list[str] = []
        monkeypatch.setattr(
            mcp_proxy, "ensure_pat_bearer", lambda profile: order.append(f"pat:{profile}") or True
        )
        monkeypatch.setattr(
            mcp_proxy, "_preflight_token", lambda ws, profile: order.append("preflight")
        )
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: order.append("bridge"))

        mcp_proxy.serve(URL, WS, "patprof", use_pat=True)

        assert order == ["pat:patprof", "preflight", "bridge"]

    def test_use_pat_without_a_resolvable_pat_exits_before_serving(self, monkeypatch, capsys):
        started: list[str] = []
        monkeypatch.setattr(mcp_proxy, "ensure_pat_bearer", lambda profile: False)
        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: started.append("bridge"))

        with pytest.raises(SystemExit) as excinfo:
            mcp_proxy.serve(URL, WS, "nopat", use_pat=True)

        assert excinfo.value.code == mcp_proxy.AUTH_FAILURE_EXIT_CODE
        assert started == []  # never opened the bridge
        assert "no personal access token" in capsys.readouterr().err

    def test_oauth_path_never_touches_pat(self, monkeypatch):
        # Without use_pat, ensure_pat_bearer must not be consulted at all.
        called: list[str] = []
        monkeypatch.setattr(
            mcp_proxy, "ensure_pat_bearer", lambda profile: called.append(profile) or True
        )
        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: None)

        mcp_proxy.serve(URL, WS, "oauthprof", use_pat=False)

        assert called == []

    def test_preflights_auth_before_opening_the_bridge(self, monkeypatch):
        # Order matters: a dead profile must be caught before the stdio bridge
        # starts, so the failure is a fast exit rather than a stalled session.
        order: list[str] = []
        monkeypatch.setattr(
            mcp_proxy, "_preflight_token", lambda ws, profile: order.append("preflight")
        )
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: order.append("bridge"))

        mcp_proxy.serve(URL, WS, "p")

        assert order == ["preflight", "bridge"]

    def test_dead_auth_exits_fast_without_starting_the_bridge(self, monkeypatch, capsys):
        # The regression this fix targets: previously the token failure surfaced
        # from inside the transport and the proxy hung until the MCP client's
        # startup timeout (~30s) with no explanation.
        started: list[str] = []

        def dead_auth(ws, profile):
            raise RuntimeError("no access token for " + ws + "; run `databricks auth login`")

        monkeypatch.setattr(mcp_proxy, "_preflight_token", dead_auth)
        monkeypatch.setattr(mcp_proxy.anyio, "run", lambda func, *args: started.append("bridge"))

        with pytest.raises(SystemExit) as excinfo:
            mcp_proxy.serve(URL, WS, "p")

        assert excinfo.value.code == mcp_proxy.AUTH_FAILURE_EXIT_CODE
        assert started == []  # the bridge never opened
        # Diagnostics go to stderr; stdout is the MCP wire and must stay clean.
        captured = capsys.readouterr()
        assert "databricks auth login" in captured.err
        assert captured.out == ""

    def test_auth_expiring_mid_session_exits_with_the_actionable_message(self, monkeypatch, capsys):
        # A ProxyAuthError raised once the bridge is running arrives wrapped in an
        # anyio ExceptionGroup; it must still be reported, not surface as a crash.
        def raise_group(func, *args):
            raise BaseExceptionGroup(
                "transport",
                [mcp_proxy.ProxyAuthError("token expired; run `databricks auth login`")],
            )

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", raise_group)

        with pytest.raises(SystemExit) as excinfo:
            mcp_proxy.serve(URL, WS, "p")

        assert excinfo.value.code == mcp_proxy.AUTH_FAILURE_EXIT_CODE
        assert "token expired" in capsys.readouterr().err

    def test_transport_failure_exits_with_a_one_line_message(self, monkeypatch, capsys):
        def raise_group(func, *args):
            raise BaseExceptionGroup(
                "transport",
                [mcp_proxy.ProxyTransportError("upstream MCP transport closed unexpectedly")],
            )

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", raise_group)

        with pytest.raises(SystemExit) as excinfo:
            mcp_proxy.serve(URL, WS, "p")

        assert excinfo.value.code == mcp_proxy.AUTH_FAILURE_EXIT_CODE
        captured = capsys.readouterr()
        assert captured.err == "ucode mcp-proxy: upstream MCP transport closed unexpectedly\n"
        assert captured.out == ""

    def test_non_auth_failures_still_propagate(self, monkeypatch):
        # Only expected proxy failures are converted to a clean exit; programming
        # bugs must keep their traceback so they stay debuggable.
        def raise_other(func, *args):
            raise ValueError("some transport bug")

        monkeypatch.setattr(mcp_proxy, "_preflight_token", lambda ws, profile: None)
        monkeypatch.setattr(mcp_proxy.anyio, "run", raise_other)

        with pytest.raises(ValueError, match="some transport bug"):
            mcp_proxy.serve(URL, WS, "p")


class TestPreflightToken:
    def test_passes_through_when_a_token_is_available(self, monkeypatch):
        monkeypatch.setattr(mcp_proxy, "get_databricks_token", lambda ws, profile: "tok")
        mcp_proxy._preflight_token(WS, "p")  # no exception

    def test_surfaces_the_cli_error_message(self, monkeypatch):
        def boom(ws, profile):
            raise RuntimeError("profile is stale; run `databricks auth logout`")

        monkeypatch.setattr(mcp_proxy, "get_databricks_token", boom)

        with pytest.raises(RuntimeError, match="databricks auth logout"):
            mcp_proxy._preflight_token(WS, "p")

    def test_checks_the_same_workspace_and_profile_the_bridge_will_use(self, monkeypatch):
        # The preflight must validate the exact credentials the request-time auth
        # hook uses, or it could pass while the bridge still fails.
        calls: list[tuple[str, str | None]] = []
        monkeypatch.setattr(
            mcp_proxy,
            "get_databricks_token",
            lambda ws, profile: calls.append((ws, profile)) or "tok",
        )

        mcp_proxy._preflight_token(WS, "myprofile")

        assert calls == [(WS, "myprofile")]
