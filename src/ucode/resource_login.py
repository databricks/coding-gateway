"""Interactive OAuth login for a specific MCP connection (RFC 8707 resource).

When an AI Gateway MCP service is backed by a per-user connection (e.g.
``system.ai.github``) and the user hasn't logged in to that connection yet, the
gateway answers a tools call with an RFC 9728 challenge: HTTP 401 +
``WWW-Authenticate: Bearer resource_metadata="…"``. A plain Databricks workspace
token (what ``databricks auth token`` mints) is not enough — the *connection*
still needs a login.

This module drives that login. It runs a standard OAuth authorization-code + PKCE
flow against the workspace ``/oidc`` server using the published ``databricks-cli``
app, but adds the **RFC 8707 ``resource`` indicator** naming the MCP service. That
indicator is what makes ``/oidc`` route the browser through the connection's
``/mcp-service-login`` page: the user signs in to the backing SaaS (GitHub, …),
the connection credential is stored server-side, and the flow returns here with an
authorization code. After that the caller's normal Databricks token works, because
the connection credential now exists.

stdlib only (no new dependency): ``http.server`` for the loopback callback,
``urllib`` for the token exchange, ``secrets``/``hashlib`` for PKCE.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import sys
import threading
import webbrowser
from urllib import request as urllib_request
from urllib.parse import parse_qs, urlencode, urlparse

# The published "Databricks CLI" OAuth app. It is a public (PKCE) client with
# loopback redirect URIs registered, so no client secret is needed. ``databricks``
# itself authenticates with this same client id.
DATABRICKS_CLI_CLIENT_ID = "databricks-cli"
# One of the app's registered redirect URIs. Must match exactly what OIDC has on
# file for the client, so this is not freely configurable.
DEFAULT_CALLBACK_PORT = 8020
# `all-apis` for the workspace token; `offline_access` for a refresh token.
DEFAULT_SCOPES = ("all-apis", "offline_access")
# How long to wait for the user to finish the browser login before giving up.
LOGIN_TIMEOUT_SECONDS = 300


class ConnectionLoginError(RuntimeError):
    """The interactive connection login could not be completed."""


def _b64url(raw: bytes) -> str:
    """base64url without padding, per RFC 7636."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE S256."""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(
    workspace: str,
    resource: str,
    *,
    client_id: str = DATABRICKS_CLI_CLIENT_ID,
    callback_port: int = DEFAULT_CALLBACK_PORT,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    code_challenge: str,
    state: str,
) -> str:
    """Build the ``/oidc/v1/authorize`` URL carrying the RFC 8707 resource.

    Split out from the flow so it can be unit-tested without any network or
    browser. ``resource`` is the MCP endpoint URL the proxy is bridging to."""
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "redirect_uri": f"http://localhost:{callback_port}",
            "scope": " ".join(scopes),
            "state": state,
            # RFC 8707: names the connection-backed MCP service so /oidc redirects
            # through its /mcp-service-login page instead of issuing a bare token.
            "resource": resource,
            "prompt": "consent",
        }
    )
    return f"{workspace.rstrip('/')}/oidc/v1/authorize?{query}"


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures the ``code``/``state`` the OIDC server redirects back with."""

    # Set by the server instance before handling.
    expected_state: str = ""
    result: dict[str, str] = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        params = parse_qs(urlparse(self.path).query)
        code = params.get("code", [""])[0]
        state = params.get("state", [""])[0]
        server_result = type(self).result
        if code and state == type(self).expected_state:
            server_result["code"] = code
            body = b"<html><body>Login complete. You can close this tab.</body></html>"
        else:
            server_result["error"] = params.get("error", ["state_mismatch_or_no_code"])[0]
            body = b"<html><body>Login failed. You can close this tab.</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - match base signature
        # Silence the default stderr access log — stderr is surfaced to the MCP client.
        return


def _await_authorization_code(callback_port: int, state: str) -> str:
    """Serve the loopback redirect until the auth code arrives (or timeout)."""
    _CallbackHandler.expected_state = state
    _CallbackHandler.result = {}
    server = http.server.HTTPServer(("127.0.0.1", callback_port), _CallbackHandler)
    server.timeout = LOGIN_TIMEOUT_SECONDS

    done = threading.Event()

    def _serve() -> None:
        # One real redirect ends the flow; loop so a stray request (favicon, etc.)
        # doesn't consume the single handle_request budget.
        while not done.is_set():
            server.handle_request()
            if _CallbackHandler.result:
                done.set()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    if not done.wait(timeout=LOGIN_TIMEOUT_SECONDS):
        server.server_close()
        raise ConnectionLoginError(
            f"timed out after {LOGIN_TIMEOUT_SECONDS}s waiting for the connection login"
        )
    server.server_close()

    result = _CallbackHandler.result
    if "code" not in result:
        raise ConnectionLoginError(f"connection login failed: {result.get('error', 'unknown')}")
    return result["code"]


def _exchange_code(
    workspace: str,
    code: str,
    *,
    client_id: str,
    callback_port: int,
    code_verifier: str,
    resource: str,
) -> str:
    """Exchange the authorization code for an access token at ``/oidc/v1/token``."""
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"http://localhost:{callback_port}",
            "client_id": client_id,
            "code_verifier": code_verifier,
            "resource": resource,
        }
    ).encode("ascii")
    request = urllib_request.Request(
        f"{workspace.rstrip('/')}/oidc/v1/token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib_request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any exchange failure uniformly
        raise ConnectionLoginError(f"token exchange failed: {exc}") from exc
    token = payload.get("access_token")
    if not token:
        raise ConnectionLoginError("token exchange returned no access_token")
    return token


def login_for_connection(
    workspace: str,
    resource: str,
    *,
    client_id: str = DATABRICKS_CLI_CLIENT_ID,
    callback_port: int = DEFAULT_CALLBACK_PORT,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> str:
    """Run the resource-scoped OAuth login for an MCP connection.

    Opens the browser to the workspace ``/oidc`` authorize endpoint with the RFC
    8707 ``resource`` indicator, serves the loopback redirect, and exchanges the
    returned code for a token. On success the connection credential is stored
    server-side, so the caller's subsequent MCP requests succeed.

    Blocking (browser + local HTTP server); call it from a worker thread when on
    an event loop. Returns the freshly minted access token.
    """
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)
    authorize_url = build_authorize_url(
        workspace,
        resource,
        client_id=client_id,
        callback_port=callback_port,
        scopes=scopes,
        code_challenge=challenge,
        state=state,
    )

    # stdout is the MCP wire — every human-facing hint goes to stderr. Print the URL
    # too: on a remote/SSH host the auto-open may not reach the user's browser.
    print(
        f"ucode mcp-proxy: connection login required — opening browser to sign in.\n"
        f"  If it doesn't open, visit:\n  {authorize_url}",
        file=sys.stderr,
        flush=True,
    )
    try:
        webbrowser.open(authorize_url)
    except Exception:  # noqa: BLE001 - a headless open failure is non-fatal; URL was printed
        pass

    code = _await_authorization_code(callback_port, state)
    return _exchange_code(
        workspace,
        code,
        client_id=client_id,
        callback_port=callback_port,
        code_verifier=verifier,
        resource=resource,
    )


__all__ = [
    "ConnectionLoginError",
    "DATABRICKS_CLI_CLIENT_ID",
    "DEFAULT_CALLBACK_PORT",
    "build_authorize_url",
    "login_for_connection",
]
