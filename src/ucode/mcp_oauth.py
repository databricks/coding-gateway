"""Discovery of the OAuth client used for direct-HTTP MCP registration.

A connection-backed AI Gateway MCP service (e.g. ``system.ai.github``) needs a
per-user connection login before its tools can be called. When a coding agent
registers the service as a **direct HTTP** MCP server, the agent itself drives
that login via OAuth against the workspace ``/oidc`` — and `/oidc` has no dynamic
client registration, so the agent must present a **pre-registered public client**.

``claude-code`` is that published client for Claude Code: it has the loopback
``/callback`` redirect Claude Code uses registered (``databricks-cli`` does not,
so Claude's direct-HTTP OAuth is rejected against it). Where a workspace has
``claude-code``, ucode registers these services as direct HTTP so ``/mcp`` shows
"needs authentication" and Authenticate drives the login natively. Not every
workspace has it yet, so we probe — and cache the answer per workspace.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

from ucode.config_io import APP_DIR

# Published public OAuth client Claude Code authenticates with. Any loopback
# callback port works — `/oidc` strips the port when matching loopback redirects
# (RFC 8252 §8.4) — but the redirect *path* (`/callback`) must be registered,
# which this client has and `databricks-cli` does not.
CLAUDE_CODE_OAUTH_CLIENT_ID = "claude-code"
MCP_OAUTH_CALLBACK_PORT = 3118

# Published apps rarely appear/disappear, so a per-workspace probe result is good
# for a while; delete the cache file to force a re-probe.
_CACHE_PATH = APP_DIR / "oauth_client_cache.json"
_CACHE_TTL_SECONDS = 7 * 24 * 3600


def _probe_oauth_client(workspace: str, client_id: str) -> bool:
    """True if ``client_id`` is a registered OAuth app on the workspace's ``/oidc``.

    Back-channel and unauthenticated: POST a throwaway ``authorization_code`` grant
    to ``/oidc/v1/token``. An **unknown** client fails client authentication (HTTP
    401 ``invalid_client``); a **known** client gets past that to a grant error
    (HTTP 400 ``invalid_request`` — "Invalid authorization code"). We only read
    which of the two it is; the dummy code always fails, harmlessly. This is the
    only user-level check available — the authorize endpoint redirects to SSO
    before validating the client, and the published-app API needs account-admin."""
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": "ucode-probe-not-a-real-code",
            "redirect_uri": f"http://localhost:{MCP_OAUTH_CALLBACK_PORT}/callback",
            "client_id": client_id,
            # `/oidc` rejects a code_verifier shorter than 43 chars *before* it
            # validates the client, so pad past that to reach the client check.
            "code_verifier": "u" * 43,
        }
    ).encode("ascii")
    request = urllib.request.Request(
        f"{workspace.rstrip('/')}/oidc/v1/token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        urllib.request.urlopen(request, timeout=10)  # noqa: S310 - fixed https workspace URL
        return True  # a 2xx for a dummy code is unexpected, but means the client is valid
    except urllib.error.HTTPError as exc:
        # 401 invalid_client => not registered; any other error (400 for the bad
        # code) => the client IS registered.
        return exc.code != 401
    except OSError:
        # Network failure: don't claim availability — the caller falls back to the
        # stdio proxy, which is always safe.
        return False


def _read_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort: a write failure just means we re-probe next time


def oauth_client_available(workspace: str, client_id: str) -> bool:
    """Whether the workspace has ``client_id`` as a registered OAuth app, cached
    per (workspace, client_id).

    Cached in ``APP_DIR`` with a weekly TTL so ``ug mcp add`` doesn't probe every
    run. Negative results are cached too (workspaces that don't have it yet)."""
    ws = workspace.rstrip("/")
    cache = _read_cache()
    entry = cache.get(ws, {}).get(client_id)
    if entry and (time.time() - entry.get("checked_at", 0)) < _CACHE_TTL_SECONDS:
        return bool(entry.get("available"))
    available = _probe_oauth_client(ws, client_id)
    cache.setdefault(ws, {})[client_id] = {
        "available": available,
        "checked_at": time.time(),
    }
    _write_cache(cache)
    return available


__all__ = [
    "CLAUDE_CODE_OAUTH_CLIENT_ID",
    "MCP_OAUTH_CALLBACK_PORT",
    "oauth_client_available",
]
