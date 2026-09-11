"""Per-connection login for AI Gateway MCP services, driven by the proxy on a 401.

A connection-backed AI Gateway MCP service (e.g. ``system.ai.github``) needs a
per-user connection credential before its tools can be used. Until the user has
logged in to the underlying SaaS, AI Gateway answers requests with an HTTP 401
(RFC 9728 ``WWW-Authenticate``).

The ``ug mcp-proxy`` bridge (see ``mcp_proxy``) sees that 401 in its httpx auth
flow and, for a connection-backed service, runs the Databricks CLI U2M login
with an RFC 8707 ``resource`` indicator naming the service, then retries the
request. A resource-aware ``/oidc`` drives the connection's own SaaS login
before minting the token, so the credential exists on retry — transparently to
the coding agent, which just sees the connection authenticate and succeed. This
is the behaviour of a generic OAuth MCP bridge (e.g. ``mcp-remote``), done in
ucode with the Databricks CLI so no extra library or per-agent OAuth app is
needed. Requires the CLI ``--resource`` flag (databricks/cli#6621).
"""

from __future__ import annotations

import subprocess
import sys

# AI Gateway MCP service endpoints look like
# ``https://<ws>/ai-gateway/mcp-services/<catalog>.<schema>.<service>``.
AIGW_MCP_SERVICES_SEGMENT = "/ai-gateway/mcp-services/"

# Login can pop a browser and wait for the user to complete the SaaS login, so
# allow generously more than a token refresh would take.
_LOGIN_TIMEOUT_SECONDS = 300


def connection_from_url(url: str) -> str | None:
    """Return the connection FQN of an AI Gateway MCP service URL, or ``None``.

    ``https://ws/ai-gateway/mcp-services/system.ai.github`` -> ``system.ai.github``.
    A URL that is not an mcp-services endpoint (or names no service) returns
    ``None`` — only connection-backed services get the login-on-401 treatment.
    """
    marker = url.find(AIGW_MCP_SERVICES_SEGMENT)
    if marker == -1:
        return None
    tail = url[marker + len(AIGW_MCP_SERVICES_SEGMENT) :]
    # Strip any trailing path (``/tools/list``), query, or fragment.
    connection = tail.split("/")[0].split("?")[0].split("#")[0]
    return connection or None


def run_connection_login(
    resource_url: str,
    workspace: str,
    *,
    profile: str | None = None,
    login_binary: str = "databricks",
) -> tuple[bool, str]:
    """Run the CLI U2M login with an RFC 8707 resource indicator for this service.

    ``resource_url`` is the MCP service endpoint (also the proxy's upstream URL);
    it is sent as ``--resource`` so a resource-aware ``/oidc`` drives the
    connection's SaaS login before issuing the token. Uses the Databricks CLI's
    own default client, whose loopback redirect is already registered — no
    ``--client-id`` needed.

    The CLI opens the browser to complete the login and prints the authorize URL.
    We route its output to **stderr** (never stdout — that is the proxy's MCP
    JSON-RPC wire), so a coding agent surfaces it in the server's log and the URL
    stays visible when the browser can't open (e.g. a headless remote). Returns
    ``(ok, message)``; on failure ``message`` points at that log.
    """
    argv = [
        login_binary,
        "auth",
        "login",
        "--host",
        workspace.rstrip("/"),
        "--resource",
        resource_url,
    ]
    if profile:
        argv += ["--profile", profile]
    connection = connection_from_url(resource_url) or resource_url
    print(
        f"ucode mcp-proxy: '{connection}' needs a one-time connection sign-in. Opening your "
        "browser to complete it — if it doesn't open, use the authorization URL printed below.",
        file=sys.stderr,
        flush=True,
    )
    try:
        # stdout -> stderr: the CLI's prompts and authorize URL reach the agent's
        # MCP log (fd 2) without corrupting this process's stdout (fd 1, the MCP
        # JSON-RPC stream). stdin is closed since the flow is browser-driven.
        result = subprocess.run(
            argv,
            check=False,
            timeout=_LOGIN_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
    except OSError as exc:
        return False, f"could not run '{login_binary} auth login': {exc}"
    except subprocess.TimeoutExpired:
        return False, "connection sign-in timed out waiting for the browser flow to complete"
    if result.returncode == 0:
        return True, "signed in"
    return (
        False,
        f"connection sign-in did not complete (CLI exited {result.returncode}; see the log above)",
    )


__all__ = [
    "AIGW_MCP_SERVICES_SEGMENT",
    "connection_from_url",
    "run_connection_login",
]
