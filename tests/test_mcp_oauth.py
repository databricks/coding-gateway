"""Tests for OAuth-client discovery + per-workspace caching (mcp_oauth).

Network-free: the token-endpoint probe is monkeypatched; these cover the
known/unknown/error classification and the cache behaviour.
"""

from __future__ import annotations

import urllib.error

from ucode import mcp_oauth

WS = "https://ws.staging.cloud.databricks.com"


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url="x", code=code, msg="m", hdrs=None, fp=None)


class TestProbeOauthClient:
    def test_known_client_400_is_available(self, monkeypatch):
        # 400 invalid_request ("Invalid authorization code") => client is registered.
        monkeypatch.setattr(
            mcp_oauth.urllib.request,
            "urlopen",
            lambda req, timeout=0: (_ for _ in ()).throw(_http_error(400)),
        )
        assert mcp_oauth._probe_oauth_client(WS, "claude-code") is True

    def test_unknown_client_401_is_absent(self, monkeypatch):
        # 401 invalid_client => client not registered on this workspace.
        monkeypatch.setattr(
            mcp_oauth.urllib.request,
            "urlopen",
            lambda req, timeout=0: (_ for _ in ()).throw(_http_error(401)),
        )
        assert mcp_oauth._probe_oauth_client(WS, "claude-code") is False

    def test_network_error_is_absent(self, monkeypatch):
        # A network failure must not claim availability — caller falls back to the proxy.
        monkeypatch.setattr(
            mcp_oauth.urllib.request,
            "urlopen",
            lambda req, timeout=0: (_ for _ in ()).throw(OSError("down")),
        )
        assert mcp_oauth._probe_oauth_client(WS, "claude-code") is False


class TestOauthClientAvailableCache:
    def test_probes_once_then_serves_from_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mcp_oauth, "_CACHE_PATH", tmp_path / "cache.json")
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            mcp_oauth, "_probe_oauth_client", lambda ws, cid: calls.append((ws, cid)) or True
        )
        assert mcp_oauth.oauth_client_available(WS, "claude-code") is True
        assert mcp_oauth.oauth_client_available(WS, "claude-code") is True
        assert len(calls) == 1  # second call served from cache, no re-probe

    def test_negative_result_is_cached_too(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mcp_oauth, "_CACHE_PATH", tmp_path / "cache.json")
        calls: list[int] = []
        monkeypatch.setattr(
            mcp_oauth, "_probe_oauth_client", lambda ws, cid: calls.append(1) or False
        )
        assert mcp_oauth.oauth_client_available(WS, "claude-code") is False
        assert mcp_oauth.oauth_client_available(WS, "claude-code") is False
        assert len(calls) == 1

    def test_expired_entry_reprobes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mcp_oauth, "_CACHE_PATH", tmp_path / "cache.json")
        monkeypatch.setattr(mcp_oauth, "_CACHE_TTL_SECONDS", -1)  # every entry immediately stale
        calls: list[int] = []
        monkeypatch.setattr(
            mcp_oauth, "_probe_oauth_client", lambda ws, cid: calls.append(1) or True
        )
        mcp_oauth.oauth_client_available(WS, "claude-code")
        mcp_oauth.oauth_client_available(WS, "claude-code")
        assert len(calls) == 2  # re-probed because the cached entry expired
