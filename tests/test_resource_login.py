"""Tests for resource-scoped MCP connection login (RFC 8707) and the proxy's
detection of the AI Gateway connection-login challenge.

Network-free: the browser + loopback + token-exchange steps are not exercised
here; these cover the pure URL / PKCE construction and challenge classification.
"""

from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

from ucode import mcp_proxy, resource_login

WS = "https://example.staging.cloud.databricks.com"
RESOURCE = f"{WS}/ai-gateway/mcp-services/system.ai.github"


def test_build_authorize_url_carries_resource_and_pkce():
    url = resource_login.build_authorize_url(
        WS,
        RESOURCE,
        code_challenge="test-challenge",
        state="test-state",
    )
    parsed = urlparse(url)
    assert parsed.path == "/oidc/v1/authorize"
    q = parse_qs(parsed.query)
    assert q["response_type"] == ["code"]
    assert q["client_id"] == [resource_login.DATABRICKS_CLI_CLIENT_ID]
    assert q["code_challenge"] == ["test-challenge"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["redirect_uri"] == [f"http://localhost:{resource_login.DEFAULT_CALLBACK_PORT}"]
    assert q["state"] == ["test-state"]
    # The RFC 8707 resource indicator is what makes /oidc route through the
    # connection's /mcp-service-login page.
    assert q["resource"] == [RESOURCE]
    assert "all-apis" in q["scope"][0]
    assert "offline_access" in q["scope"][0]


def test_build_authorize_url_trims_trailing_slash():
    url = resource_login.build_authorize_url(WS + "/", RESOURCE, code_challenge="c", state="s")
    assert url.startswith(f"{WS}/oidc/v1/authorize?")


def test_pkce_pair_is_valid_s256():
    verifier, challenge = resource_login._pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge == expected
    # base64url, no padding.
    assert "=" not in verifier and "=" not in challenge


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers


def test_is_connection_login_challenge_true_on_401_with_resource_metadata():
    resp = _FakeResponse(
        401,
        {
            "www-authenticate": 'Bearer resource_metadata="https://ws/.well-known/oauth-protected-resource/x"'
        },
    )
    assert mcp_proxy._is_connection_login_challenge(resp) is True


def test_is_connection_login_challenge_false_without_resource_metadata():
    # A generic 401 (e.g. workspace auth) is not a connection-login challenge.
    assert mcp_proxy._is_connection_login_challenge(_FakeResponse(401, {})) is False
    assert (
        mcp_proxy._is_connection_login_challenge(
            _FakeResponse(401, {"www-authenticate": 'Bearer error="invalid_token"'})
        )
        is False
    )


def test_is_connection_login_challenge_false_on_non_401():
    resp = _FakeResponse(200, {"www-authenticate": 'Bearer resource_metadata="https://ws/x"'})
    assert mcp_proxy._is_connection_login_challenge(resp) is False
