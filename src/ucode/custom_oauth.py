"""Custom-client OAuth, separate from production Databricks CLI auth."""

from __future__ import annotations

import platform
import shlex
import subprocess
from collections.abc import Sequence
from typing import TypedDict
from urllib.parse import urlparse

try:
    from databricks.sdk import oauth
except ModuleNotFoundError:
    oauth = None

from ucode.constants import LOCALHOST, LOOPBACK_HOST
from ucode.databricks import build_auth_token_argv
from ucode.ui import err_console, normalize_workspace_url, print_warning_err

DEFAULT_REDIRECT_URL = f"http://{LOCALHOST}:8020"
_MISSING_SDK_ERROR = (
    "Custom OAuth requires the optional Databricks SDK dependency. "
    'Install it with `uv tool install "ucode[custom-oauth]"` and retry.'
)


class CustomOAuthConfig(TypedDict):
    client_id: str
    redirect_url: str
    scopes: list[str]


def _require_oauth():
    if oauth is None:
        raise RuntimeError(_MISSING_SDK_ERROR)
    return oauth


def _normalize_scopes(scopes: Sequence[str]) -> list[str]:
    if isinstance(scopes, str):
        raise RuntimeError("OAuth scopes must be provided as a sequence of scope names.")
    requested_scopes = list(dict.fromkeys(scope.strip() for scope in scopes if scope.strip()))
    if "offline_access" not in requested_scopes:
        raise RuntimeError("OAuth scopes must include offline_access to support token refresh.")
    if not any(scope != "offline_access" for scope in requested_scopes):
        raise RuntimeError("At least one API OAuth scope must be provided.")
    return requested_scopes


def _validate_redirect_url(redirect_url: str) -> None:
    try:
        redirect = urlparse(redirect_url)
        valid_redirect = (
            redirect.scheme == "http"
            and redirect.hostname in {LOCALHOST, LOOPBACK_HOST}
            and redirect.port is not None
            and redirect.port > 0
            and not (redirect.username or redirect.password or redirect.query or redirect.fragment)
        )
    except ValueError:
        valid_redirect = False
    if not valid_redirect:
        raise RuntimeError("--redirect-url must be a local HTTP callback with a port.")


def create_custom_oauth_config(
    client_id: str,
    scopes: Sequence[str],
    redirect_url: str = DEFAULT_REDIRECT_URL,
) -> CustomOAuthConfig:
    client_id = client_id.strip()
    if not client_id:
        raise RuntimeError("--client-id must not be empty.")
    _validate_redirect_url(redirect_url)
    return {
        "client_id": client_id,
        "redirect_url": redirect_url,
        "scopes": _normalize_scopes(scopes),
    }


def build_custom_auth_token_argv(workspace: str, config: CustomOAuthConfig) -> list[str]:
    normalized = create_custom_oauth_config(
        config["client_id"], config["scopes"], config["redirect_url"]
    )
    return [
        *build_auth_token_argv(workspace),
        "--client-id",
        normalized["client_id"],
        "--redirect-url",
        normalized["redirect_url"],
        "--scopes",
        ",".join(normalized["scopes"]),
    ]


def build_custom_auth_shell_command(workspace: str, config: CustomOAuthConfig) -> str:
    argv = build_custom_auth_token_argv(workspace, config)
    if platform.system() == "Windows":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def get_custom_client_token(
    workspace: str,
    client_id: str,
    redirect_url: str = DEFAULT_REDIRECT_URL,
    *,
    scopes: Sequence[str],
    force_refresh: bool = False,
) -> str:
    """Reuse the SDK's PKCE flow and per-workspace/client token cache."""
    config = create_custom_oauth_config(client_id, scopes, redirect_url)
    sdk_oauth = _require_oauth()
    workspace = normalize_workspace_url(workspace)
    try:
        endpoints = sdk_oauth.get_workspace_endpoints(workspace)
        cache = sdk_oauth.TokenCache(
            host=workspace,
            oidc_endpoints=endpoints,
            client_id=config["client_id"],
            redirect_url=config["redirect_url"],
            scopes=config["scopes"],
        )
        credentials = cache.load()
        if credentials is not None:
            try:
                if force_refresh:
                    credentials = sdk_oauth.SessionCredentials(
                        token=credentials.refresh(),
                        token_endpoint=endpoints.token_endpoint,
                        client_id=config["client_id"],
                        redirect_url=config["redirect_url"],
                    )
                credentials.token()
            except Exception:
                print_warning_err("Cached OAuth token could not be refreshed. Sign in again.")
                credentials = None
        if credentials is None:
            client = sdk_oauth.OAuthClient(
                oidc_endpoints=endpoints,
                client_id=config["client_id"],
                redirect_url=config["redirect_url"],
                scopes=config["scopes"],
            )
            consent = client.initiate_consent()
            err_console.print(
                f"Sign in using your browser: {consent.authorization_url}",
                markup=False,
                soft_wrap=True,
            )
            credentials = consent.launch_external_browser()
        token = credentials.token().access_token
        if not token:
            raise ValueError("OAuth returned no access token")
        cache.save(credentials)
        return token
    except Exception as exc:
        raise RuntimeError(
            "Custom-client OAuth failed. Check the workspace, client ID, and registered "
            f"redirect URL ({config['redirect_url']}); ensure its local port is available and "
            "the SDK token cache is writable, then retry."
        ) from exc
