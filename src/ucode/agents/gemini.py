"""Gemini CLI agent: writes ~/.gemini/.env and runs with periodic token refresh."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from pathlib import Path

from ucode.agent_updates import latest_version_below
from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    parse_dotenv,
    read_json_safe,
    write_dotenv,
    write_json_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_tool_base_url,
    get_databricks_token,
)
from ucode.state import (
    get_provider_service,
    mark_tool_managed,
    save_state,
    set_provider_service,
)
from ucode.telemetry import agent_version, ucode_version

from .args import LaunchOptions

GEMINI_CONFIG_DIR = Path.home() / ".gemini"
GEMINI_ENV_PATH = GEMINI_CONFIG_DIR / "ucode.env"
GEMINI_BACKUP_PATH = APP_DIR / "gemini-ucode-env.backup"
GEMINI_HOME_DIR = APP_DIR / ".gemini-home"
GEMINI_SETTINGS_PATH = GEMINI_HOME_DIR / ".gemini" / "settings.json"

SPEC: ToolSpec = {
    "binary": "gemini",
    "package": "@google/gemini-cli",
    "display": "Gemini CLI",
    "config_path": GEMINI_ENV_PATH,
    "backup_path": GEMINI_BACKUP_PATH,
}

MANAGED_KEYS: list[str] = [
    "GEMINI_MODEL",
    "GOOGLE_GEMINI_BASE_URL",
    "GEMINI_API_KEY_AUTH_MECHANISM",
    "GEMINI_API_KEY",
    "GEMINI_CLI_CUSTOM_HEADERS",
    "OAUTH_TOKEN",
]


# Gemini CLI 0.45 introduced a "Gemini 3.5 Flash GA" router that rewrites any
# forced flash model id (e.g. `databricks-gemini-3-5-flash`) to Google's
# canonical `gemini-3.5-flash`, which the Databricks AI Gateway rejects as an
# invalid Unity Catalog endpoint name. Until that regression is fixed upstream
# we cap the supported version below 0.45 and steer clients onto the newest
# release that still passes the configured model through verbatim.
MAX_GEMINI_VERSION = (0, 45, 0)
MAX_GEMINI_VERSION_TEXT = "0.45.0"


def _parse_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def latest_working_version() -> str | None:
    """Newest published Gemini CLI release below the broken-version ceiling."""
    return latest_version_below(SPEC["package"], MAX_GEMINI_VERSION)


def too_new_version() -> str | None:
    """Return the installed version string when it exceeds the safe ceiling.

    Used by the install flow to warn the client and offer a downgrade.
    Returns None when the version is safe or cannot be determined.
    """
    raw = agent_version(SPEC["binary"])
    parsed = _parse_version(raw)
    if parsed is None:
        return None
    if parsed >= MAX_GEMINI_VERSION:
        return raw
    return None


def too_new_downgrade() -> tuple[str, str] | None:
    """Return (installed_version, downgrade_target) when a downgrade is needed.

    `downgrade_target` is the newest published release below the broken
    ceiling. Returns None when the installed version is safe, npm is
    unavailable, or no working release can be resolved.
    """
    installed = too_new_version()
    if installed is None:
        return None
    target = latest_working_version()
    if target is None:
        return None
    return installed, target


def _ensure_local_settings_selected_type() -> None:
    settings = read_json_safe(GEMINI_SETTINGS_PATH)
    deep_merge_dict(
        settings,
        {"security": {"auth": {"selectedType": "gemini-api-key"}}},
    )
    write_json_file(GEMINI_SETTINGS_PATH, settings)


def render_env_overlay(
    workspace: str, model: str, token: str, *, provider: str | None = None
) -> dict[str, str]:
    # Gemini CLI parses GEMINI_CLI_CUSTOM_HEADERS as comma-separated
    # `Key:Value` pairs and spreads them after the SDK's default User-Agent,
    # so a key named `User-Agent` overrides the default. Resolved via
    # upstream issue google-gemini/gemini-cli#10088.
    custom_headers = f"User-Agent:ucode/{ucode_version()} gemini/{agent_version('gemini')}"
    if provider:
        # A Model Provider Service routes by this header; the request still names
        # the service's target model in `GEMINI_MODEL` (pinned by the launch path).
        custom_headers += f",Databricks-Model-Provider-Service:{provider}"
    return {
        "GEMINI_MODEL": model,
        "GOOGLE_GEMINI_BASE_URL": build_tool_base_url("gemini", workspace),
        "GEMINI_API_KEY_AUTH_MECHANISM": "bearer",
        "GEMINI_API_KEY": token,
        "GEMINI_CLI_CUSTOM_HEADERS": custom_headers,
        "OAUTH_TOKEN": token,
    }


def build_runtime_env(
    workspace: str, model: str, token: str, *, provider: str | None = None
) -> dict[str, str]:
    _ensure_local_settings_selected_type()
    env = os.environ.copy()
    env.update(render_env_overlay(workspace, model, token, provider=provider))
    # Newer Gemini CLI releases refuse to run in untrusted directories;
    # opt every launch into trust so `ucode gemini` works in any folder.
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    env["GEMINI_CLI_HOME"] = str(GEMINI_HOME_DIR)
    return env


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
    provider: str | None = None,
) -> tuple[dict, str]:
    backup_existing_file(GEMINI_ENV_PATH, GEMINI_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    overlay = render_env_overlay(state["workspace"], model, token, provider=provider)
    existing = parse_dotenv(GEMINI_ENV_PATH)
    existing.update(overlay)
    write_dotenv(GEMINI_ENV_PATH, existing)
    if provider:
        # Persist so the token-refresh thread and later bare `ucode gemini` re-emit
        # the routing header; `--provider` on a launch overrides this saved choice.
        state = set_provider_service(state, "gemini", provider)
    state = mark_tool_managed(state, "gemini", MANAGED_KEYS)
    save_state(state)
    return state, token


def default_model(state: dict) -> str | None:
    if isinstance(state.get("gemini_default_model"), str):
        return state.get("gemini_default_model")
    gemini_models = state.get("gemini_models") or []
    return gemini_models[0] if gemini_models else None


def persisted_provider_model() -> str | None:
    """The Gemini model currently pinned in the env file, if any.

    A provider launch writes the service's target here; reusing it lets a bare
    ``ucode gemini`` relaunch (or reconfigure) run without re-passing ``--model``.
    """
    return parse_dotenv(GEMINI_ENV_PATH).get("GEMINI_MODEL")


def _launch_model(state: dict, provider: str | None) -> str | None:
    """The model this session runs on.

    Under a provider it is the service's target model, pinned into the env file
    by ``write_tool_config`` (``default_model`` would return a Databricks id the
    gateway can't resolve behind the provider header). Otherwise the usual default.
    """
    if provider:
        written = persisted_provider_model()
        if written:
            return written
    return default_model(state)


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    provider = get_provider_service(state, "gemini")
    model = _launch_model(state, provider)
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh, provider=provider)
    return token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def launch(state: dict, tool_args: list[str], *, options: LaunchOptions) -> None:
    provider = get_provider_service(state, "gemini")
    token = _refresh_token_once(state)
    model = _launch_model(state, provider)
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    env = build_runtime_env(state["workspace"], model, token, provider=provider)

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen([SPEC["binary"], *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "-p", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    """Inject env vars for the validation subprocess.

    The Gemini CLI's .env auto-loading skips ~/.gemini/.env when run from an
    untrusted folder, so we cannot rely on it during validation.
    """
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    provider = get_provider_service(state, "gemini")
    model = _launch_model(state, provider)
    if not model:
        raise RuntimeError("No Gemini model is configured.")
    token = get_databricks_token(workspace, state.get("profile"))
    return build_runtime_env(workspace, model, token, provider=provider)
