"""Databricks workspace integration: CLI auth, token retrieval, model
discovery, AI Gateway checks, SQL warehouse discovery, URL builders."""

from __future__ import annotations

import configparser
import functools
import json
import logging
import logging.handlers
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, NamedTuple, NoReturn, cast, overload
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, urlencode, urlparse

from databricks.sql.exc import ServerOperationError

from ucode.config_io import APP_DIR
from ucode.ui import (
    err_console,
    normalize_workspace_url,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    spinner,
)

UNIX_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh"
)
WINDOWS_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.ps1"
)
AI_GATEWAY_DOCS_URL = "https://docs.databricks.com/aws/en/ai-gateway/overview-beta"
ANTHROPIC_MODELS_PATH = "/ai-gateway/anthropic/v1/models"
# v1.0.0 is the release that ships `databricks aitools`.
MIN_DATABRICKS_CLI_VERSION = (1, 0, 0)
TOKEN_REFRESH_INTERVAL_SECONDS = 1800
# Substrings the Databricks CLI emits when it loses the token-cache write lock
# to a concurrent `databricks auth token` (e.g. another ucode helper process or
# MLflow tracing refreshing the shared ~/.databricks/token-cache.json at the same
# instant). These are transient — the credential is fine, only the local write
# raced — so we retry rather than treat them as an expired session.
_TOKEN_CACHE_LOCK_MARKERS = ("cache update", "exit status 45")
_TOKEN_FETCH_MAX_ATTEMPTS = 4
_HTTP_GET_RETRYABLE_STATUS_CODES = frozenset({429})
_HTTP_GET_RETRY_BASE_SECONDS = 1.0
_HTTP_GET_RETRY_MAX_SECONDS = 5.0
_HTTP_GET_RETRY_AFTER_JITTER_SECONDS = 0.25
_ANTHROPIC_MODEL_DISCOVERY_SETUP_MAX_RETRIES = 2


@dataclass(frozen=True)
class AnthropicModelCatalog:
    """Models advertised by AI Gateway's Anthropic endpoint."""

    model_ids: list[str]
    model_id_to_display_name: dict[str, str]
    error_msg: str | None = None


def _debug_enabled() -> bool:
    return os.environ.get("UCODE_DEBUG") == "1"


_DEBUG_LOGGER: logging.Logger | None = None


def _get_debug_logger() -> logging.Logger | None:
    """Lazily configure a rotating file logger when UCODE_DEBUG=1.

    Returns the logger on first call (and caches it), or None if debug is
    disabled or the log file could not be opened. A one-time breadcrumb is
    printed to stderr so the user knows where to tail."""
    global _DEBUG_LOGGER
    if _DEBUG_LOGGER is not None or not _debug_enabled():
        return _DEBUG_LOGGER

    log_path = APP_DIR / "debug.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )
    except OSError:
        return None

    logger = logging.getLogger("ucode.debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    _DEBUG_LOGGER = logger
    err_console.print(f"[dim]\\[ucode debug] logging to {log_path}[/dim]")
    return _DEBUG_LOGGER


def _debug(label: str, detail: str) -> None:
    """When UCODE_DEBUG=1, append a timestamped entry to ~/.ucode/debug.log."""
    logger = _get_debug_logger()
    if logger is not None:
        logger.debug("%s: %s", label, detail)


_SECRET_KEY_PATTERN = re.compile(r"(token|secret|password|bearer|api_key|apikey)", re.IGNORECASE)


def _format_subprocess_result(
    result: subprocess.CompletedProcess[str],
) -> str:
    """Format a CompletedProcess for the debug log without leaking tokens.

    On success, stdout is suppressed (it often contains the access token).
    On failure, stdout/stderr are included truncated."""
    stderr = (result.stderr or "").strip()[:500]
    if result.returncode == 0:
        return f"rc=0 stderr={stderr!r}"
    stdout = (result.stdout or "").strip()[:500]
    return f"rc={result.returncode} stdout={stdout!r} stderr={stderr!r}"


def _scrub_databrickscfg(text: str) -> str:
    """Redact value of any INI key that looks secret-bearing."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if "=" in stripped and not stripped.startswith(("#", ";")):
            key = stripped.split("=", 1)[0].strip()
            if _SECRET_KEY_PATTERN.search(key):
                indent = line[: len(line) - len(stripped)]
                out.append(f"{indent}{key} = <redacted>")
                continue
        out.append(line)
    return "\n".join(out)


def _scrub_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: (
                "<redacted>"
                if isinstance(k, str) and _SECRET_KEY_PATTERN.search(k)
                else _scrub_json(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_scrub_json(v) for v in value]
    return value


@functools.cache
def _log_auth_diagnostics() -> None:
    """Dump CLI version, profiles, and ~/.databrickscfg (scrubbed) to the debug log.

    No-op unless UCODE_DEBUG=1; cached so it runs at most once per process."""
    if not _debug_enabled():
        return

    try:
        version_result = subprocess.run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (version_result.stdout or version_result.stderr or "").strip()
        _debug("databricks --version", version[:200])
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("databricks --version", f"exception: {type(exc).__name__}: {exc}")

    try:
        profiles_result = subprocess.run(
            ["databricks", "auth", "profiles", "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        _debug(
            "databricks auth profiles",
            f"rc={profiles_result.returncode} "
            f"stderr={(profiles_result.stderr or '').strip()[:300]!r}",
        )
        if profiles_result.returncode == 0 and profiles_result.stdout:
            try:
                payload = json.loads(profiles_result.stdout)
                _debug("profiles json", json.dumps(_scrub_json(payload))[:2000])
            except json.JSONDecodeError as exc:
                _debug("profiles json", f"decode error: {exc}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("databricks auth profiles", f"exception: {type(exc).__name__}: {exc}")

    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg").expanduser()
    try:
        if cfg_path.is_file():
            raw = cfg_path.read_text(encoding="utf-8", errors="replace")
            _debug(f"databrickscfg ({cfg_path})", _scrub_databrickscfg(raw)[:4000])
        else:
            _debug(f"databrickscfg ({cfg_path})", "not present")
    except OSError as exc:
        _debug(f"databrickscfg ({cfg_path})", f"read error: {exc}")


def _http_get_retry_delay(retry_after: str | None, retry_index: int) -> float:
    if retry_after is not None:
        try:
            retry_after_seconds = float(retry_after)
        except ValueError:
            pass
        else:
            if retry_after_seconds >= 0:
                return min(retry_after_seconds, _HTTP_GET_RETRY_MAX_SECONDS) + random.uniform(
                    0, _HTTP_GET_RETRY_AFTER_JITTER_SECONDS
                )

    backoff = min(
        _HTTP_GET_RETRY_BASE_SECONDS * (2 ** min(retry_index, 10)),
        _HTTP_GET_RETRY_MAX_SECONDS,
    )
    return backoff + random.uniform(0, min(backoff * 0.25, 0.5))


def _http_get_json(
    url: str,
    token: str,
    *,
    timeout: int = 10,
    max_retries: int = 0,
) -> tuple[dict | list | None, str | None]:
    """GET a JSON endpoint. Returns (payload, None) on success, (None, reason) on failure.

    ``max_retries`` opts individual callers into bounded retries for rate limits
    and network failures. Other callers retain the original single-attempt
    behavior.

    Honors UCODE_DEBUG=1 to append status + truncated body to ~/.ucode/debug.log.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    request = urllib_request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            _debug(f"GET {url}", f"HTTP 200, {len(body)} bytes")
            if _debug_enabled():
                _debug("body", body[:4000])
            try:
                return json.loads(body), None
            except json.JSONDecodeError as exc:
                return None, f"response was not valid JSON ({exc.msg})"
        except urllib_error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            except Exception:
                body = ""
            _debug(f"GET {url}", f"HTTP {exc.code} {exc.reason}")
            if _debug_enabled() and body:
                _debug("body", body[:4000])
            reason = f"HTTP {exc.code} {exc.reason}"
            # Surface the response body too — gateway auth failures return 400
            # with body `Invalid Token`, which is invisible without this.
            body_excerpt = body.strip()[:200]
            if body_excerpt:
                reason = f"{reason}: {body_excerpt}"
            if exc.code not in _HTTP_GET_RETRYABLE_STATUS_CODES or attempt == max_retries:
                return None, reason
            retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
        except urllib_error.URLError as exc:
            _debug(f"GET {url}", f"URLError: {exc.reason}")
            reason = f"network error: {exc.reason}"
            if attempt == max_retries:
                return None, reason
            retry_after = None
        except OSError as exc:
            # A socket read timeout raises a bare TimeoutError (an OSError), not a
            # URLError, so it must be caught explicitly or it escapes the whole
            # discovery flow. Surface it as a reason like every other failure.
            _debug(f"GET {url}", f"OSError: {exc}")
            reason = f"network error: {exc}"
            if attempt == max_retries:
                return None, reason
            retry_after = None

        delay = _http_get_retry_delay(retry_after, attempt)
        _debug(
            f"GET {url}",
            f"attempt {attempt + 1} failed: {reason}; retrying in {delay:.2f}s",
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


def _http_send_json(
    method: str,
    url: str,
    token: str,
    payload: dict | None,
    *,
    timeout: int = 10,
    allow_empty_body: bool = False,
) -> tuple[dict | list | None, str | None]:
    """Send a request that may carry a JSON body, and decode a JSON response.

    Shared by `_http_post_json`, `_http_patch_json`, and `_http_delete` — the three differ only in
    verb, whether they send a body, and whether an empty response is success. Returns
    ``(payload, None)`` on success and ``(None, reason)`` on failure, like `_http_get_json`.

    ``allow_empty_body`` is for DELETE, whose success response is ``google.protobuf.Empty`` — an
    empty body there is the expected result, not a decode failure.
    """
    body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if body_bytes is not None:
        headers["Content-Type"] = "application/json"
    request = urllib_request.Request(url, data=body_bytes, method=method, headers=headers)
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        _debug(f"{method} {url}", f"HTTP {response.status}, {len(body)} bytes")
        if _debug_enabled():
            _debug("body", body[:4000])
        if allow_empty_body and not body.strip():
            return None, None
        try:
            return json.loads(body), None
        except json.JSONDecodeError as exc:
            return None, f"response was not valid JSON ({exc.msg})"
    except urllib_error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        except Exception:
            body = ""
        _debug(f"{method} {url}", f"HTTP {exc.code} {exc.reason}")
        if _debug_enabled() and body:
            _debug("body", body[:4000])
        reason = f"HTTP {exc.code} {exc.reason}"
        body_excerpt = body.strip()[:200]
        if body_excerpt:
            reason = f"{reason}: {body_excerpt}"
        return None, reason
    except urllib_error.URLError as exc:
        _debug(f"{method} {url}", f"URLError: {exc.reason}")
        return None, f"network error: {exc.reason}"
    except OSError as exc:
        # See `_http_get_json`: a bare socket timeout is an OSError, not a
        # URLError, and would otherwise escape the caller's error handling.
        _debug(f"{method} {url}", f"OSError: {exc}")
        return None, f"network error: {exc}"


def _http_post_json(
    url: str, token: str, payload: dict, *, timeout: int = 10
) -> tuple[dict | list | None, str | None]:
    """POST a JSON body to an endpoint. Returns (payload, None) on success,
    (None, reason) on failure. Mirrors `_http_get_json`."""
    return _http_send_json("POST", url, token, payload, timeout=timeout)


def _http_patch_json(
    url: str, token: str, payload: dict, *, timeout: int = 10
) -> tuple[dict | list | None, str | None]:
    """PATCH a JSON body to an endpoint. Returns (payload, None) on success,
    (None, reason) on failure."""
    return _http_send_json("PATCH", url, token, payload, timeout=timeout)


def _http_delete(
    url: str, token: str, *, timeout: int = 10
) -> tuple[dict | list | None, str | None]:
    """DELETE a resource. Returns (payload, None) on success, (None, reason) on failure.

    A successful delete returns ``google.protobuf.Empty``, which serializes as ``{}`` or an empty
    body depending on the gateway, so both count as success and yield ``(None, None)``. Callers
    should test ``reason`` rather than the payload.
    """
    return _http_send_json("DELETE", url, token, None, timeout=timeout, allow_empty_body=True)


def _http_get_bytes(url: str, token: str, *, timeout: int = 10) -> tuple[bytes | None, str | None]:
    """GET raw bytes. Returns (body, None) on success, (None, reason) on failure.

    Like `_http_get_json` but leaves the body undecoded, since skill bundles can
    contain binary files.
    """
    request = urllib_request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        _debug(f"GET {url}", f"HTTP 200, {len(body)} bytes")
        return body, None
    except urllib_error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        except Exception:
            detail = ""
        _debug(f"GET {url}", f"HTTP {exc.code} {exc.reason}")
        reason = f"HTTP {exc.code} {exc.reason}"
        excerpt = detail.strip()[:200]
        if excerpt:
            reason = f"{reason}: {excerpt}"
        return None, reason
    except urllib_error.URLError as exc:
        _debug(f"GET {url}", f"URLError: {exc.reason}")
        return None, f"network error: {exc.reason}"


# Workspace group whose members are workspace admins. `ucode setup` / `ucode publish` are restricted
# to this group because the coding-agent-config CRUD API enforces the same check server-side.
WORKSPACE_ADMIN_GROUP = "admins"


def _scim_me(workspace: str, token: str) -> dict | None:
    """Return the SCIM `Me` payload for the caller, or None on failure."""
    hostname = workspace_hostname(workspace)
    payload, _ = _http_get_json(f"https://{hostname}/api/2.0/preview/scim/v2/Me", token)
    return payload if isinstance(payload, dict) else None


def is_workspace_admin(workspace: str, token: str) -> bool | None:
    """Whether the caller is a workspace admin, via their SCIM `Me` group membership.

    Returns True/False, or None when the check itself could not be made (SCIM unreachable or a
    malformed response). Callers should treat None as "unknown" and proceed optimistically rather
    than blocking: the API enforces the same check server-side, so a false negative here would
    needlessly stop a legitimate admin, while a false positive just surfaces the server's
    PERMISSION_DENIED later.
    """
    payload = _scim_me(workspace, token)
    if payload is None:
        return None
    groups = payload.get("groups")
    if not isinstance(groups, list):
        # A well-formed `Me` for a user in no groups omits `groups` entirely, so this is a
        # definitive "not an admin" rather than a failed check.
        return False
    return any(
        isinstance(group, dict) and group.get("display") == WORKSPACE_ADMIN_GROUP
        for group in groups
    )


# Workspace-scoped budget listing. Account-level budget APIs need account auth, which ucode does not
# have; this endpoint resolves the workspace server-side from the caller's token.
_WORKSPACE_BUDGETS_API_PATH = "/api/ai-gateway/v2/workspace-metrics/budgets"


_PER_USER_ALERT_SCOPE = "ALERT_CONFIGURATION_SCOPE_TYPE_PER_USER"
_BLOCK_ACTION_TYPE = "BLOCK_USAGE"


def _has_per_user_block(entry: dict) -> bool:
    """Whether a raw budget entry has a per-user alert threshold that hard-blocks usage.

    True only when some alert is per-user scoped *and* carries a ``BLOCK_USAGE`` action; a per-user
    alert with only an email notification does not enforce spend routing.
    """
    for alert in entry.get("alert_configurations") or []:
        if not isinstance(alert, dict) or alert.get("scope_type") != _PER_USER_ALERT_SCOPE:
            continue
        for action in alert.get("action_configurations") or []:
            if isinstance(action, dict) and action.get("action_type") == _BLOCK_ACTION_TYPE:
                return True
    return False


def _per_user_block_threshold(entry: dict) -> Decimal | None:
    """The dollar amount a per-user hard block trips at, or None when there isn't one.

    Reads ``quantity_threshold`` off the same per-user ``BLOCK_USAGE`` alert that
    :func:`_has_per_user_block` gates on (``quantity_type`` is ``LIST_PRICE_DOLLARS_USD`` and
    ``time_period`` is ``MONTH``, so it's a per-user monthly dollar cap). Budget tiers are picked as
    percentages of this, so surfacing it lets the admin see the dollars a percentage stands for. A
    block alert without a parseable threshold yields None — the wizard just omits the dollar hint.
    """
    for alert in entry.get("alert_configurations") or []:
        if not isinstance(alert, dict) or alert.get("scope_type") != _PER_USER_ALERT_SCOPE:
            continue
        blocks = any(
            isinstance(action, dict) and action.get("action_type") == _BLOCK_ACTION_TYPE
            for action in alert.get("action_configurations") or []
        )
        if blocks:
            return _parse_decimal(alert.get("quantity_threshold"))
    return None


def list_workspace_budgets(workspace: str, token: str) -> tuple[list[dict], str | None]:
    """List the AI Gateway budgets that apply to this workspace.

    Returns ``(budgets, reason)`` where each budget is
    ``{"id", "display_name", "has_per_user_block", "per_user_threshold"}``. ``reason`` is None on
    success, otherwise it explains why the list is empty. ucode never creates budgets — an admin picks
    an existing one to attach a spend-routing policy to. ``has_per_user_block`` lets the picker hide
    budgets that can't enforce spend routing (see ``_has_per_user_block``); ``per_user_threshold`` is
    the per-user monthly dollar cap (a ``Decimal``, or None when it can't be read) so the tier prompt
    can show what a tier percentage works out to in dollars.
    """
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}{_WORKSPACE_BUDGETS_API_PATH}"
    payload, reason = _http_get_json(url, token, timeout=30)
    if reason is not None:
        return [], reason
    if not isinstance(payload, dict):
        return [], "workspace budget listing returned an unexpected response shape"
    raw = payload.get("workspace_ai_gateway_budgets")
    if not isinstance(raw, list):
        return [], "workspace budget listing returned no budgets"
    budgets: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        budget_id = entry.get("budget_configuration_id")
        if not isinstance(budget_id, str) or not budget_id:
            continue
        display_name = entry.get("display_name")
        budgets.append(
            {
                "id": budget_id,
                "display_name": display_name if isinstance(display_name, str) else "",
                "has_per_user_block": _has_per_user_block(entry),
                "per_user_threshold": _per_user_block_threshold(entry),
            }
        )
    if not budgets:
        return [], "workspace budget listing returned no budgets"
    return budgets, None


def get_current_user_name(workspace: str, token: str) -> str | None:
    """Return the current user's login (email) via SCIM `Me`, or None on failure.

    Databricks puts the workspace login in `userName`; fall back to the first
    `emails` entry for workspaces that diverge."""
    payload = _scim_me(workspace, token)
    if payload is None:
        return None
    user_name = payload.get("userName")
    if isinstance(user_name, str) and user_name.strip():
        return user_name.strip()
    emails = payload.get("emails")
    if isinstance(emails, list):
        for entry in emails:
            if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                return entry["value"].strip()
    return None


# Experiment tag Databricks sets when an experiment's traces are written to a
# Unity Catalog table. Its value is the UC destination, e.g.
# "my_catalog.my_schema.my_table". A plain (file/DBFS-backed) experiment does
# not carry this tag, so its presence is our signal that traces land in UC.
UC_TRACE_DESTINATION_TAG = "mlflow.experiment.databricksTraceDestinationPath"


def _experiment_tags(experiment: dict) -> dict[str, str | None]:
    """Flatten an experiment's ``tags`` list ([{key, value}, ...]) into a dict."""
    out: dict[str, str | None] = {}
    tags = experiment.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and isinstance(tag.get("key"), str):
                out[tag["key"]] = tag.get("value")
    return out


def _uc_trace_destination(experiment: dict) -> str | None:
    """The Unity Catalog destination (``catalog.schema.table``) an experiment
    logs traces to, or None when it isn't UC-backed. Any three-part UC name
    qualifies — the specific catalog/schema/table is not constrained."""
    value = _experiment_tags(experiment).get(UC_TRACE_DESTINATION_TAG)
    if isinstance(value, str):
        parts = value.split(".")
        if len(parts) == 3 and all(parts):
            return value
    return None


def find_uc_backed_experiment(
    workspace: str, token: str, leaf_name: str
) -> tuple[dict | None, str | None]:
    """Find an existing experiment whose final path segment is ``leaf_name`` and
    whose traces are backed by Unity Catalog.

    Returns (experiment, reason). On success ``experiment`` is
    ``{"experiment_id", "experiment_name", "uc_destination"}`` and reason is
    None. On failure ``experiment`` is None and reason explains why (no such
    experiment, or it exists but isn't UC-backed) so the caller can tell the
    user to create one."""
    hostname = workspace_hostname(workspace)
    # Leaf-match in the filter (anything ending in the name), then confirm the
    # exact leaf segment in Python so "/Users/<me>/ucode-traces" matches but
    # "team-ucode-traces" does not.
    safe_leaf = leaf_name.replace("'", "")
    payload, reason = _http_post_json(
        f"https://{hostname}/api/2.0/mlflow/experiments/search",
        token,
        {"filter": f"name LIKE '%{safe_leaf}'", "max_results": 1000},
    )
    if not isinstance(payload, dict):
        return None, reason or "could not search MLflow experiments"

    experiments = payload.get("experiments")
    named = [
        exp
        for exp in (experiments if isinstance(experiments, list) else [])
        if isinstance(exp, dict)
        and str(exp.get("name") or "").rsplit("/", 1)[-1] == leaf_name
        and exp.get("experiment_id")
    ]
    if not named:
        return None, f"no experiment named '{leaf_name}' exists on this workspace"

    for exp in named:
        dest = _uc_trace_destination(exp)
        if dest:
            return {
                "experiment_id": str(exp["experiment_id"]),
                "experiment_name": str(exp.get("name") or leaf_name),
                "uc_destination": dest,
            }, None

    return (
        None,
        f"experiment '{leaf_name}' exists but its traces are not backed by Unity Catalog",
    )


def resolve_sql_warehouse_id(workspace: str, token: str) -> tuple[str | None, str | None]:
    """Pick a SQL warehouse for writing traces to a UC-backed experiment.

    Writing traces to a Unity Catalog table requires a SQL warehouse
    (``MLFLOW_TRACING_SQL_WAREHOUSE_ID``); without one the MLflow exporter
    silently drops them. We prefer a RUNNING warehouse so the first trace isn't
    blocked on a cold start, falling back to any existing warehouse (a stopped
    one auto-starts on first query). Returns (warehouse_id, reason); reason is
    None on success, else explains why none could be resolved."""
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(f"https://{hostname}/api/2.0/sql/warehouses", token)
    if not isinstance(payload, dict):
        return None, reason or "could not list SQL warehouses"

    warehouses = payload.get("warehouses")
    warehouses = (
        [w for w in warehouses if isinstance(w, dict) and w.get("id")]
        if isinstance(warehouses, list)
        else []
    )
    if not warehouses:
        return None, "no SQL warehouse exists on this workspace"

    running = next((w for w in warehouses if str(w.get("state")).upper() == "RUNNING"), None)
    chosen = running or warehouses[0]
    return str(chosen["id"]), None


@overload
def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: Literal[True],
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]: ...


@overload
def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: Literal[False] = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[bytes]: ...


def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
        timeout=timeout,
    )


def build_databricks_cli_env(workspace: str, profile: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = workspace
    if profile is None:
        env.pop("DATABRICKS_CONFIG_PROFILE", None)
    return env


def workspace_hostname(workspace: str) -> str:
    parsed = urlparse(normalize_workspace_url(workspace))
    if not parsed.hostname:
        raise RuntimeError(f"Unable to derive hostname from workspace URL: {workspace}")
    return parsed.hostname


def _parse_databricks_cli_version(output: str) -> tuple[int, int, int] | None:
    # Example output: "Databricks CLI v0.299.2"
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", output)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _run_databricks_cli_installer(brew_subcommand: str = "install") -> None:
    system = platform.system()
    try:
        if system == "Windows":
            run(
                ["powershell", "-Command", f"irm {WINDOWS_DATABRICKS_INSTALL_URL} | iex"],
                timeout=240,
            )
        elif system == "Darwin" and shutil.which("brew"):
            run(["brew", brew_subcommand, "databricks/tap/databricks"], timeout=240)
        elif shutil.which("curl"):
            run(["sh", "-c", f"curl -fsSL {UNIX_DATABRICKS_INSTALL_URL} | sudo sh"], timeout=240)
        elif shutil.which("wget"):
            run(["sh", "-c", f"wget -qO- {UNIX_DATABRICKS_INSTALL_URL} | sudo sh"], timeout=240)
        else:
            raise RuntimeError("Neither curl nor wget is available.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        raise RuntimeError("Failed to install/upgrade Databricks CLI automatically.") from exc


def ensure_databricks_cli_version() -> None:
    try:
        result = run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Failed to read Databricks CLI version.") from exc

    raw = result.stdout or result.stderr or ""
    output = (raw if isinstance(raw, str) else raw.decode(errors="replace")).strip()
    version = _parse_databricks_cli_version(output)
    if version is None:
        raise RuntimeError(
            f"Could not parse Databricks CLI version from `databricks --version` output: {output!r}"
        )
    if version < MIN_DATABRICKS_CLI_VERSION:
        current = ".".join(str(n) for n in version)
        required = ".".join(str(n) for n in MIN_DATABRICKS_CLI_VERSION)
        print_warning(
            f"Databricks CLI v{current} is too old (need v{required} or newer). Upgrading..."
        )
        _run_databricks_cli_installer(brew_subcommand="upgrade")
        ensure_databricks_cli_version()


def databricks_cli_version() -> tuple[int, int, int] | None:
    """Return the installed Databricks CLI's (major, minor, patch), or None if
    the CLI is absent or its version can't be read/parsed. Unlike
    ``ensure_databricks_cli_version`` this only reports — it never upgrades — so
    ``ucode doctor`` can decide what to recommend."""
    if not shutil.which("databricks"):
        return None
    try:
        result = run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = result.stdout or result.stderr or ""
    output = (raw if isinstance(raw, str) else raw.decode(errors="replace")).strip()
    return _parse_databricks_cli_version(output)


def upgrade_databricks_cli() -> bool:
    """Upgrade an already-installed Databricks CLI to the latest release.
    Returns True on success, False if the installer failed."""
    try:
        _run_databricks_cli_installer(brew_subcommand="upgrade")
    except RuntimeError:
        return False
    return True


def install_databricks_cli() -> None:
    if shutil.which("databricks"):
        ensure_databricks_cli_version()
        return

    print_section("Bootstrap")
    print_warning("`databricks` was not found. Installing Databricks CLI...")
    _run_databricks_cli_installer(brew_subcommand="install")

    if not shutil.which("databricks"):
        raise RuntimeError(
            "Databricks CLI install completed, but `databricks` is still not on PATH."
        )
    ensure_databricks_cli_version()


def install_ai_tools(agent_tokens: list[str], profile: str | None = None) -> None:
    """Install Databricks AI Tools for the given agents (e.g. ``claude-code``).

    Databricks AI Tools is the set of skills and plugins that teach coding
    agents how to work with Databricks (installed via ``databricks aitools``).
    Idempotent and best-effort: any failure only warns (surfacing the CLI's
    own error), since AI Tools aren't required to launch an agent."""
    if not agent_tokens:
        return

    agents_arg = ",".join(agent_tokens)
    try:
        with spinner(f"Installing Databricks AI Tools for {agents_arg}..."):
            run(
                ["databricks", "aitools", "install", "--agents", agents_arg, "--scope", "global"]
                + _profile_args(profile),
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        # The CLI version is already guaranteed by ensure_databricks_cli_version,
        # so any failure here is something else (e.g. an agent binary missing
        # from PATH). Surface the CLI's own error rather than guessing a cause.
        detail = getattr(exc, "stderr", None) or ""
        if isinstance(detail, bytes):  # TimeoutExpired.stderr is bytes even with text=True
            detail = detail.decode(errors="replace")
        detail = detail.strip()
        reason = detail.splitlines()[-1] if detail else str(exc)
        print_warning(f"Could not install Databricks AI Tools: {reason}")
    else:
        print_success("Databricks AI Tools installed")


def _profile_args(profile: str | None) -> list[str]:
    """Return ``["--profile", profile]`` when set, otherwise an empty list.

    Centralizing this keeps every `databricks` CLI invocation in this module
    consistent when a workspace's `~/.databrickscfg` has more than one profile
    pointing at the same host."""
    return ["--profile", profile] if profile else []


def has_valid_databricks_auth(workspace: str, profile: str | None = None) -> bool:
    # Honor the CI short-circuit (see ``get_databricks_token``): if a
    # pre-fetched bearer is available, treat auth as valid and skip the
    # `databricks auth token` shell-out (which only knows user-OAuth).
    if os.environ.get("DATABRICKS_BEARER", "").strip():
        return True
    _log_auth_diagnostics()
    # Mirror run_databricks_login: when ~/.databrickscfg has multiple
    # profiles for the same host, `databricks auth token --host …` refuses
    # to disambiguate without --profile, so resolve it from the host here.
    profile = profile or find_profile_name_for_host(workspace)
    try:
        env = build_databricks_cli_env(workspace, profile)
        result = run(
            [
                "databricks",
                "auth",
                "token",
                "--host",
                workspace,
                *_profile_args(profile),
                "--output",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        _debug(
            "has_valid_databricks_auth",
            _format_subprocess_result(result),
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout or "{}")
        return bool(data.get("access_token"))
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as exc:
        _debug("has_valid_databricks_auth", f"exception: {type(exc).__name__}: {exc}")
        return False


def list_profile_entries() -> list[dict]:
    """Return raw profile dicts ({"name", "host", "auth_type", ...}) from
    `databricks auth profiles`.

    Each non-PAT profile is returned individually — duplicate hosts (multiple
    profiles pointing at the same workspace) appear as separate entries so the
    workspace picker can offer each profile by name. Order matches the CLI's
    own ordering.

    Returns ``[]`` on any failure (CLI missing, timeout, non-zero exit, JSON
    decode error). When ``UCODE_DEBUG=1`` each dropout path logs *why* the
    result was empty so a silently-disappearing workspace picker is
    diagnosable from ``~/.ucode/debug.log``.
    """
    try:
        result = run(
            ["databricks", "auth", "profiles", "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("list_profile_entries", f"subprocess error: {type(exc).__name__}: {exc}")
        return []
    if result.returncode != 0:
        _debug("list_profile_entries", _format_subprocess_result(result))
        return []
    try:
        profiles = json.loads(result.stdout or "{}").get("profiles") or []
    except json.JSONDecodeError as exc:
        _debug("list_profile_entries", f"json decode error: {exc.msg}")
        return []
    return [p for p in profiles if isinstance(p, dict)]


def get_databricks_profiles() -> list[tuple[str, str]]:
    """Return [(host_url, profile_name), ...] from Databricks CLI profiles."""
    profiles = list_profile_entries()

    out: list[tuple[str, str]] = []
    pat = 0
    for p in profiles:
        host = (p.get("host") or "").rstrip("/")
        name = p.get("name")
        if not host or not name:
            continue
        if p.get("auth_type") == "pat":
            pat += 1
            continue
        out.append((host, name))

    _debug(
        "get_databricks_profiles",
        f"returned={len(out)} total={len(profiles)} pat={pat}",
    )
    return out


def find_profile_name_for_host(workspace: str) -> str | None:
    """Find the Databricks CLI profile name matching a workspace URL."""
    normalized = workspace.rstrip("/")
    for host, name in get_databricks_profiles():
        if host == normalized:
            return name
    return None


def profile_auth_type(profile: str) -> str | None:
    """Return the auth_type of a Databricks CLI profile (e.g. "pat"), or None."""
    for p in list_profile_entries():
        if p.get("name") == profile:
            auth_type = p.get("auth_type")
            return auth_type if isinstance(auth_type, str) else None
    return None


def _read_databrickscfg_token(profile: str) -> str | None:
    """Read the static ``token`` value for a profile from ``~/.databrickscfg``.

    `databricks auth token` only knows OAuth caches; for PAT profiles the PAT
    itself is the credential, stored in the config file. The parser's default
    section is pointed at a name that never appears in the file so a token in
    ``[DEFAULT]`` does not leak into every named profile."""
    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg").expanduser()
    parser = configparser.ConfigParser(default_section="@ucode-no-defaults@", interpolation=None)
    try:
        if not parser.read(cfg_path, encoding="utf-8"):
            return None
    except (configparser.Error, OSError):
        return None
    if not parser.has_section(profile):
        return None
    token = (parser.get(profile, "token", fallback="") or "").strip()
    return token or None


def resolve_pat_token(profile: str | None) -> str | None:
    """Return the static PAT of a PAT-type Databricks CLI profile, or None.

    Only consulted when the user explicitly opted in via
    ``ucode configure --profiles <name> --use-pat`` — ucode never picks up a
    PAT implicitly."""
    if profile and profile_auth_type(profile) == "pat":
        return _read_databrickscfg_token(profile)
    return None


def ensure_pat_bearer(profile: str | None, pat: str | None = None) -> bool:
    """Ensure ``DATABRICKS_BEARER`` holds a usable token for a ``--use-pat`` profile.

    If a non-empty bearer is already in the environment it wins (the CI escape
    hatch). Otherwise the profile's static PAT is exported — callers that have
    already resolved it (e.g. ``configure_shared_state``) pass it via ``pat`` to
    skip a redundant ``~/.databrickscfg`` read; everyone else lets this resolve
    it. An exported-but-*empty* ``DATABRICKS_BEARER`` is treated as absent —
    matching ``get_databricks_token``'s own ``.strip()`` check — so a stray
    ``export DATABRICKS_BEARER=`` does not shadow the PAT and silently force the
    OAuth path (which fails for PAT-only profiles).

    Returns ``True`` iff a usable bearer is now present in the environment."""
    if os.environ.get("DATABRICKS_BEARER", "").strip():
        return True
    pat = pat or resolve_pat_token(profile)
    if pat:
        os.environ["DATABRICKS_BEARER"] = pat
        return True
    return False


def apply_pat_environment(state: dict) -> None:
    """Export the configured profile's PAT as ``DATABRICKS_BEARER`` when the
    workspace was configured with ``--use-pat``.

    Every token fetch in this process (and in launched agent subprocesses,
    which inherit the environment) then takes the existing static-bearer
    short-circuit instead of the OAuth-only `databricks auth token` path.
    A non-empty bearer already present in the environment is left untouched."""
    if not state.get("use_pat"):
        return
    ensure_pat_bearer(state.get("profile"))


def run_databricks_login(workspace: str, profile: str | None = None) -> None:
    """Run databricks auth login unconditionally.

    When ``profile`` is provided, it is passed via ``--profile``. Otherwise we
    fall back to looking up an existing profile by host so a stored session is
    refreshed in place rather than overwriting another profile's tokens."""
    print_section("Databricks Login")
    print_kv("Workspace", workspace)
    print_note("A browser may open for `databricks auth login`.")
    try:
        profile_name = profile or find_profile_name_for_host(workspace)
        cmd = [
            "databricks",
            "auth",
            "login",
            "--host",
            workspace,
            *_profile_args(profile_name),
        ]
        run(cmd, env=build_databricks_cli_env(workspace, profile_name), timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`databricks auth login` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`databricks auth login` timed out.") from exc
    print_success("Databricks authentication complete")


def ensure_databricks_auth(
    workspace: str, profile: str | None = None, *, quiet: bool = False
) -> None:
    """Check auth and login only if needed (used by launch path).

    ``quiet`` suppresses the "already available" line for a caller that only needs a token before
    some later step re-authenticates and reports it — otherwise the same success prints twice. A
    login that actually runs is never silent.
    """
    with spinner("Checking Databricks auth..."):
        auth_is_valid = has_valid_databricks_auth(workspace, profile)
    if auth_is_valid:
        if not quiet:
            print_success(f"Databricks auth already available for {workspace}")
        return
    run_databricks_login(workspace, profile)


def get_databricks_token(
    workspace: str,
    profile: str | None = None,
    *,
    force_refresh: bool = False,
) -> str:
    # ``DATABRICKS_BEARER`` is the CI escape hatch: when set, skip the
    # `databricks auth token` subprocess entirely and return the pre-fetched
    # bearer directly. Used by the e2e job, where the protected runner has
    # no `databricks auth login` cache and `databricks auth token` only knows
    # how to read user-OAuth caches (not M2M client_credentials). Mirrors the
    # same short-circuit baked into ``build_auth_shell_command``.
    bearer = os.environ.get("DATABRICKS_BEARER", "").strip()
    if bearer:
        _debug("get_databricks_token", "using DATABRICKS_BEARER env var")
        return bearer

    _log_auth_diagnostics()
    # See has_valid_databricks_auth: resolve the profile from the host when
    # the caller didn't supply one, so duplicate-host cfgs don't break us.
    profile = profile or find_profile_name_for_host(workspace)
    env = build_databricks_cli_env(workspace, profile)
    cmd = [
        "databricks",
        "auth",
        "token",
        "--host",
        workspace,
        *_profile_args(profile),
        "--output",
        "json",
    ]
    if force_refresh:
        cmd.append("--force-refresh")

    _debug(
        "get_databricks_token.env",
        "set="
        + ",".join(sorted(k for k in env if k.startswith("DATABRICKS_") or k in {"BUNDLE_PROFILE"}))
        + f" profile={profile or '<none>'}",
    )

    def _fetch() -> tuple[str, str]:
        """Return (access_token, stderr). token is '' on any failure."""
        try:
            result = run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            _debug("auth token", _format_subprocess_result(result))
            if result.returncode == 0:
                return json.loads(result.stdout or "{}").get("access_token", ""), ""
            return "", result.stderr or ""
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            _debug("auth token", f"exception: {type(exc).__name__}: {exc}")
            return "", str(exc)

    def _fetch_with_lock_retry() -> str:
        """Mint a token, retrying transient token-cache lock contention.

        Concurrent `databricks auth token` calls racing on the shared cache fail
        fast with a lock error (see ``_TOKEN_CACHE_LOCK_MARKERS``). The lock is
        held only for the brief cache write, so a short jittered backoff almost
        always wins the next attempt. A non-lock failure returns '' immediately
        so the caller can fall through to the re-auth path."""
        for attempt in range(_TOKEN_FETCH_MAX_ATTEMPTS):
            token, stderr = _fetch()
            if token:
                return token
            if not any(marker in stderr.lower() for marker in _TOKEN_CACHE_LOCK_MARKERS):
                return ""
            _debug("auth token", f"cache-lock contention (attempt {attempt + 1}); retrying")
            if attempt < _TOKEN_FETCH_MAX_ATTEMPTS - 1:
                time.sleep(random.uniform(0.05, 0.1 * (2**attempt)))
        return ""

    token = _fetch_with_lock_retry()
    if not token:
        # Session may have expired — attempt non-interactive re-auth and retry once.
        _debug("auth token", "empty on first fetch; attempting auth login --no-browser")
        try:
            reauth = run(
                [
                    "databricks",
                    "auth",
                    "login",
                    "--host",
                    workspace,
                    *_profile_args(profile),
                    "--no-browser",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            _debug("auth login", _format_subprocess_result(reauth))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _debug("auth login", f"exception: {type(exc).__name__}: {exc}")
        token = _fetch_with_lock_retry()

    if not token:
        profile_name = profile or find_profile_name_for_host(workspace)
        stale_profile_hint = ""
        if profile_name:
            stale_profile_hint = (
                " The saved Databricks CLI profile may be stale or invalid. Try:\n"
                f"  databricks auth logout --profile {profile_name}\n"
                f"  databricks auth login --host {workspace} --profile {profile_name}"
            )
        raise RuntimeError(
            f"Databricks CLI returned no access token for {workspace}. "
            "Run `databricks auth login` to re-authenticate."
            f"{stale_profile_hint}"
        )
    return token


def _extract_connection_page(payload: object) -> tuple[list[dict], str | None]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], None
    if not isinstance(payload, dict):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    payload_dict = cast(dict[str, object], payload)
    raw_connections = payload_dict.get("connections") or []
    if not isinstance(raw_connections, list):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    next_page_token = payload_dict.get("next_page_token")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    return [item for item in raw_connections if isinstance(item, dict)], next_page_token


def list_databricks_connections(workspace: str, profile: str | None = None) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    connections: list[dict] = []
    page_token: str | None = None
    seen_page_tokens: set[str] = set()

    try:
        while True:
            cmd = [
                "databricks",
                "connections",
                "list",
                *_profile_args(profile),
                "--max-results",
                "0",
                "--output",
                "json",
            ]
            if page_token:
                cmd.extend(["--page-token", page_token])

            result = run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
            page_connections, page_token = _extract_connection_page(payload)
            connections.extend(page_connections)

            if not page_token:
                return connections
            if page_token in seen_page_tokens:
                raise RuntimeError("Databricks connections listing returned a repeated page token.")
            seen_page_tokens.add(page_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Databricks connections via `databricks connections list`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks connections.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks connections listing returned invalid JSON.") from exc


def _extract_genie_spaces_page(payload: object) -> tuple[list[dict], str | None]:
    if not isinstance(payload, dict):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    payload_dict = cast(dict[str, object], payload)
    raw_spaces = payload_dict.get("spaces") or []
    if not isinstance(raw_spaces, list):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    next_page_token = payload_dict.get("next_page_token")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    return [item for item in raw_spaces if isinstance(item, dict)], next_page_token


def list_genie_spaces(workspace: str, profile: str | None = None) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    spaces: list[dict] = []
    page_token: str | None = None
    seen_page_tokens: set[str] = set()

    try:
        while True:
            cmd = [
                "databricks",
                "genie",
                "list-spaces",
                *_profile_args(profile),
                "--page-size",
                "100",
                "--output",
                "json",
            ]
            if page_token:
                cmd.extend(["--page-token", page_token])

            result = run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
            page_spaces, page_token = _extract_genie_spaces_page(payload)
            spaces.extend(page_spaces)

            if not page_token:
                return spaces
            if page_token in seen_page_tokens:
                raise RuntimeError(
                    "Databricks Genie spaces listing returned a repeated page token."
                )
            seen_page_tokens.add(page_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Databricks Genie spaces via `databricks genie list-spaces`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks Genie spaces.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.") from exc


def _extract_apps_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        payload_dict = cast(dict[str, object], payload)
        raw_apps = payload_dict.get("apps") or []
        if isinstance(raw_apps, list):
            return [item for item in raw_apps if isinstance(item, dict)]
    raise RuntimeError("Databricks apps listing returned invalid JSON.")


def list_databricks_apps(workspace: str, profile: str | None = None) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    try:
        result = run(
            [
                "databricks",
                "apps",
                "list",
                *_profile_args(profile),
                "--limit",
                "1000",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        return _extract_apps_payload(json.loads(result.stdout or "[]"))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to list Databricks apps via `databricks apps list`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks apps.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks apps listing returned invalid JSON.") from exc


def _ucode_binary() -> str:
    """Resolve the absolute path to the running `ucode` executable.

    Agents persist the auth command into config files and re-run it on every
    token refresh, possibly from launchers without a full PATH (desktop GUIs).
    An absolute path keeps the helper working regardless of PATH. Falls back to
    the bare name when resolution fails."""
    return shutil.which("ucode") or "ucode"


def build_auth_token_argv(
    workspace: str, profile: str | None = None, *, use_pat: bool = False
) -> list[str]:
    """Argv for the cross-platform token helper: `ucode auth-token ...`.

    Unlike the previous POSIX `databricks ... | jq` pipeline, this is a single
    executable with plain arguments — no `sh`, no `jq`, no shell quoting — so it
    runs identically on macOS, Linux, and Windows (issue #116). The DATABRICKS_BEARER
    short-circuit and the PAT path both live inside `auth-token` itself."""
    argv = [_ucode_binary(), "auth-token", "--host", workspace.rstrip("/")]
    if profile:
        argv += ["--profile", profile]
    if use_pat:
        argv.append("--use-pat")
    return argv


def build_mcp_proxy_argv(
    url: str, workspace: str, profile: str | None = None, *, use_pat: bool = False
) -> list[str]:
    """Argv for the stdio MCP bridge: `ucode mcp-proxy --url ... --host ...`.

    Every coding agent registers this single command as a local stdio MCP
    server instead of a per-client HTTP endpoint with a bearer header. The proxy
    forwards to ``url`` and mints a fresh OAuth token on each upstream request,
    so tokens never expire mid-session — the client only ever spawns a process,
    which keeps registration uniform across CLIs that disagree on HTTP-auth
    syntax. Like `build_auth_token_argv`, this resolves the absolute `ucode`
    path and passes plain arguments (no shell), so it runs identically on every
    platform."""
    argv = [_ucode_binary(), "mcp-proxy", "--url", url, "--host", workspace.rstrip("/")]
    if profile:
        argv += ["--profile", profile]
    if use_pat:
        argv.append("--use-pat")
    return argv


def build_auth_shell_command(
    workspace: str, profile: str | None = None, *, use_pat: bool = False
) -> str:
    """Single-line, shell-quoted form of :func:`build_auth_token_argv`.

    Used where a tool wants the helper as one command *string* (Claude Code's
    `apiKeyHelper`). On every platform this resolves to the `ucode auth-token`
    executable rather than a POSIX shell pipeline, so no `sh`/`jq` is required."""
    argv = build_auth_token_argv(workspace, profile, use_pat=use_pat)
    if platform.system() == "Windows":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


# A model-service's `name` is `model-services/system.ai.<model-name>`; the
# part after the prefix is exactly the model string agents send (no
# `databricks-` infix — that only appears on the inner destination name).
_MODEL_SERVICE_NAME_PREFIX = "model-services/"
# The listing can return services from EVERY schema (e.g. `main.user.foo`,
# `temp.*`, internal DLT schemas). We only want the Databricks-managed
# foundation models under `system.ai`.
_MODEL_SERVICE_REQUIRED_PREFIX = "system.ai."
# Scope the listing to the `system.ai` schema via the `parent` query param
# (`schemas/{catalog}.{schema}`). Without it the endpoint walks the ENTIRE
# metastore — hundreds of unrelated services across dozens of ~2s pages, then
# discards all but `system.ai.*` client-side (a ~50s walk on a busy workspace).
# Parent-scoped, the same set comes back in a single page (~1s). The endpoint
# ignores the other filters (`catalog_name`/`schema_name`/`filter`), so `parent`
# is the only server-side narrowing that works.
_MODEL_SERVICE_PARENT_SCHEMA = "schemas/system.ai"

# Supported OSS chat families, matched by name substring. Add an entry to
# support a new family.
_OSS_MODEL_FAMILIES = ("kimi-", "glm-", "deepseek-")

# Claude model families ucode buckets, newest tier first. Each maps to a
# Claude Code family alias (ANTHROPIC_DEFAULT_<FAMILY>_MODEL). Add an entry to
# support a new family in both discovery paths (`claude-<family>-*` via the
# model-services listing and either `databricks-claude-<family>-*` or
# `system.ai.claude-<family>-*` via the AI Gateway).
ANTHROPIC_FAMILIES = ("fable", "opus", "sonnet", "haiku")


def classify_model_family(model_id: str) -> str | None:
    """Bucket a model FQN into the family ucode keys its state by, or None if unrecognized.

    Mirrors how discovery buckets a model-services listing (see `discover_model_services`), so a
    model named in a managed config lands in the same bucket it would have from discovery. Returns
    one of ``ANTHROPIC_FAMILIES``, ``"codex"``, ``"gemini"``, or ``"oss"``. Matching is by name
    substring because neither the listing nor the config records a model's API dialect.
    """
    for family in ANTHROPIC_FAMILIES:
        if f"claude-{family}-" in model_id:
            return family
    if "gpt-" in model_id:
        return "codex"
    if "gemini-" in model_id:
        return "gemini"
    if any(oss in model_id for oss in _OSS_MODEL_FAMILIES):
        return "oss"
    return None


# Per-family token limits (context window + max output tokens). These are a
# property of the model + its `/ai-gateway/mlflow/v1` route (the gateway rejects
# requests whose output exceeds the cap), not of any one agent — so every agent
# that serves OSS models reads this single table and translates it into its own
# config dialect. Both fields are provided because agents like OpenCode require
# context and output together. Keyed by family substring; add an entry to bound
# a new model.
_MODEL_TOKEN_LIMITS: dict[str, dict[str, int]] = {
    # GLM-4.6: 200k context, but the gateway caps output well below the model's
    # native 128k — pin 25k so requests aren't rejected.
    "glm": {"context": 200_000, "output": 25_000},
}


def model_token_limits(model_id: str) -> dict[str, int] | None:
    """Return ``{"context": ..., "output": ...}`` limits for ``model_id``, or None.

    Matches by family substring (e.g. any ``*glm*`` id). None means the model
    has no known limits and the agent should not pin any."""
    for family, limits in _MODEL_TOKEN_LIMITS.items():
        if family in model_id:
            return dict(limits)
    return None


def _model_service_id(service: dict) -> str | None:
    """Extract the `system.ai.<model-name>` id from one model-service entry.

    Returns None for services in any other schema, so user/internal model
    services don't leak into the family buckets."""
    name = service.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name.startswith(_MODEL_SERVICE_NAME_PREFIX):
        name = name[len(_MODEL_SERVICE_NAME_PREFIX) :]
    if not name.startswith(_MODEL_SERVICE_REQUIRED_PREFIX):
        return None
    return name or None


# The model-services metastore listing REQUIRES a bounded `page_size`:
# unparameterized or large-page requests (verified against
# eng-ml-agent-platform.staging 2026-06-14) return `HTTP 499` with an empty
# body, while pages of 10–100 come back reliably. A page can still 499
# intermittently under load, so each gets a few retries before we give up.
_MODEL_SERVICES_PAGE_SIZE = 100
_MODEL_SERVICES_PAGE_RETRIES = 4

# Substrings that mark a failure reason (`HTTP <code> <phrase>: <body>`) as a 404 / NOT_FOUND:
# the HTTP status line and the Databricks `error_code` carried in the response body.
_NOT_FOUND_REASON_MARKERS = ("http 404", "not_found")


def _get_model_services_page(
    url: str, token: str, *, retries: int = _MODEL_SERVICES_PAGE_RETRIES
) -> tuple[dict | list | None, str | None]:
    """GET one model-services page, retrying on failure.

    The endpoint intermittently 499/504s under load; a retry usually succeeds.
    Returns the same (payload, reason) shape as ``_http_get_json`` — the last
    attempt's result when all retries are exhausted."""
    payload: dict | list | None = None
    reason: str | None = None
    for attempt in range(retries):
        payload, reason = _http_get_json(url, token, timeout=30)
        if payload is not None:
            return payload, None
        _debug("model-services page", f"attempt {attempt + 1}/{retries} failed: {reason}")
    return payload, reason


# Successful model-service listings for this process, keyed by workspace. The listing is a paginated
# walk of the whole metastore catalog, and several callers want different views of the same result
# (`discover_model_services` buckets it per family, `discover_claude_models_unbucketed` keeps the raw
# Claude ids), so a single `ucode setup` run would otherwise page it twice. Cached per process, not
# persisted: a long-lived process is not a thing here, and a new model appearing mid-command is not
# worth a second walk. Failures are never cached, so a transient error still retries.
_MODEL_SERVICES_CACHE: dict[str, list[str]] = {}

# Same idea for the Model Provider Service listing (a different endpoint). It is workspace-wide and
# filtered per agent afterwards, so `ucode setup` would otherwise re-list it once per MPS-capable
# agent. Keyed by ``(workspace, parent)`` — a schema-scoped listing is a different result set than
# the metastore-wide one, so they must not share an entry.
_MODEL_PROVIDER_SERVICES_CACHE: dict[tuple[str, str], list[dict]] = {}


def clear_model_services_cache() -> None:
    """Forget cached model-service listings (used by tests, and after a workspace switch)."""
    _MODEL_SERVICES_CACHE.clear()
    _MODEL_PROVIDER_SERVICES_CACHE.clear()


def has_cached_model_provider_services(workspace: str, parent: str | None = None) -> bool:
    """True when :func:`list_model_provider_services` will answer from cache.

    Lets a caller skip a progress spinner it doesn't need: the cold listing takes over a second, so
    it deserves one, but repeating it per agent on an instant cache hit is just noise. Takes
    ``parent`` for the same reason the cache is keyed on it — a scoped listing is a separate entry.
    """
    return (workspace, parent or "") in _MODEL_PROVIDER_SERVICES_CACHE


def list_model_services(
    workspace: str,
    token: str,
    *,
    page_size: int = _MODEL_SERVICES_PAGE_SIZE,
    max_pages: int = 100,
    use_cache: bool = True,
) -> tuple[list[str], str | None]:
    """List all `system.ai.*` model ids via the UC model-services API.

    Pages through ``/api/2.1/unity-catalog/model-services`` scoped to the
    ``system.ai`` schema (``parent=schemas/system.ai``) with a bounded
    ``page_size`` (the endpoint 499s without one) and returns the de-duplicated,
    sorted list of ``system.ai.<model-name>`` ids. Returns (ids, reason); reason
    is None on success, otherwise it describes why the list is empty (HTTP/network
    error or no services). Scoping matters: the unscoped metastore listing walks
    every schema across dozens of ~2s pages (~50s on a busy workspace) only to
    keep the same ``system.ai.*`` subset — see ``_MODEL_SERVICE_PARENT_SCHEMA``.

    A successful result is memoized per workspace for the life of the process; pass
    ``use_cache=False`` to force a fresh walk.
    """
    if use_cache:
        cached = _MODEL_SERVICES_CACHE.get(workspace)
        if cached is not None:
            return list(cached), None

    hostname = workspace_hostname(workspace)
    ids: list[str] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    last_reason: str | None = None
    for _ in range(max_pages):
        params: dict[str, str] = {
            "parent": _MODEL_SERVICE_PARENT_SCHEMA,
            "page_size": str(page_size),
        }
        if page_token:
            params["page_token"] = page_token
        url = f"https://{hostname}/api/2.1/unity-catalog/model-services?{urlencode(params)}"
        payload, reason = _get_model_services_page(url, token)
        if payload is None:
            # Surface the failure only if we have nothing yet; a mid-pagination
            # blip still returns whatever we collected.
            last_reason = reason
            break
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        for service in data.get("model_services", []):
            if isinstance(service, dict):
                model_id = _model_service_id(service)
                if model_id:
                    ids.append(model_id)
        page_token = data.get("next_page_token") or None
        if not page_token:
            last_reason = None
            break
        if page_token in seen_tokens:
            break
        seen_tokens.add(page_token)

    deduped = sorted(set(ids))
    if deduped:
        if use_cache:
            _MODEL_SERVICES_CACHE[workspace] = list(deduped)
        return deduped, None
    return [], last_reason or "model-services listing returned no models"


def _is_not_found_reason(reason: str | None) -> bool:
    """True when an HTTP reason describes a 404 / NOT_FOUND (a resource that isn't there)."""
    if not reason:
        return False
    lowered = reason.lower()
    return any(marker in lowered for marker in _NOT_FOUND_REASON_MARKERS)


def model_service_exists(
    workspace: str, token: str, full_name: str, *, max_pages: int = 100
) -> tuple[bool | None, str | None]:
    """Whether ``<catalog>.<schema>.<model>`` is a UC model service on this workspace.

    Used to quick-check a hand-typed custom model before an admin pins a config to it. Lists the
    model services in the typed name's own schema (``parent=schemas/<catalog>.<schema>``, the same
    scoped listing :func:`list_model_services` uses for ``system.ai``) and checks for the name.

    Returns ``(exists, reason)``:

    - ``True`` — the name is a model service in that schema.
    - ``False`` — the schema exists but has no such service, or the API returned 404/NOT_FOUND (the
      catalog or schema in the name doesn't exist, so the model can't either). Both are a definitive
      "no" the caller can re-prompt on.
    - ``None`` — the check couldn't run: a name that isn't a three-part UC path, or a non-404
      HTTP/network error. The caller treats this as "couldn't verify" rather than "doesn't exist" so
      a transient failure never blocks a valid model.

    Never cached: it targets a user schema, not the memoized ``system.ai`` walk.
    """
    parts = [part.strip() for part in full_name.split(".")]
    if len(parts) != 3 or not all(parts):
        return None, "a model service is named <catalog>.<schema>.<model>"
    catalog, schema, _model = parts
    normalized = f"{catalog}.{schema}.{_model}"
    parent = f"schemas/{catalog}.{schema}"
    hostname = workspace_hostname(workspace)
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for _ in range(max_pages):
        params: dict[str, str] = {"parent": parent, "page_size": str(_MODEL_SERVICES_PAGE_SIZE)}
        if page_token:
            params["page_token"] = page_token
        url = f"https://{hostname}/api/2.1/unity-catalog/model-services?{urlencode(params)}"
        payload, reason = _get_model_services_page(url, token)
        if payload is None:
            # A 404 means the catalog/schema in the typed name doesn't exist on this workspace, so
            # neither can the model — a definitive "no". Every other failure (auth, 5xx, network) is
            # inconclusive: don't block a possibly-valid model on a blip.
            return (False, reason) if _is_not_found_reason(reason) else (None, reason)
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        for service in data.get("model_services", []):
            if not isinstance(service, dict):
                continue
            name = service.get("name")
            if not isinstance(name, str):
                continue
            name = name.strip()
            if name.startswith(_MODEL_SERVICE_NAME_PREFIX):
                name = name[len(_MODEL_SERVICE_NAME_PREFIX) :]
            if name == normalized:
                return True, None
        page_token = data.get("next_page_token") or None
        if not page_token or page_token in seen_tokens:
            break
        seen_tokens.add(page_token)
    return False, None


def discover_claude_models_unbucketed(workspace: str, token: str) -> tuple[list[str], str | None]:
    """Every `system.ai.claude-*` id on the workspace, unbucketed.

    `discover_model_services` keeps only the newest id per family because the launch path pins one
    model per Claude family alias. An admin authoring a managed config needs the alternatives too
    (see `managed_setup.claude_family_candidates`), so this returns the full set without disturbing
    that shape.
    """
    ids, reason = list_model_services(workspace, token)
    if not ids:
        return [], reason
    return [m for m in ids if "claude-" in m.lower()], None


def _prefer_opus_4_8(models: dict[str, str], all_ids: list[str]) -> None:
    """Swap the opus slot to claude-opus-4-8 when it's available.

    Discovery picks the newest opus (opus-5) but smart routing's
    CLAUDE_ROUTE_ARMS require claude-opus-4-8. Pin to 4-8 when both
    exist so the routing availability check passes.
    """
    opus = models.get("opus")
    if opus and "claude-opus-5" in opus:
        opus_48 = next((m for m in all_ids if "claude-opus-4-8" in m), None)
        if opus_48:
            models["opus"] = opus_48


def discover_model_services(
    workspace: str, token: str
) -> tuple[dict[str, str], list[str], list[str], list[str], str | None]:
    """Discover models via UC model-services and bucket them by family name.

    Returns (claude_models, codex_models, gemini_models, oss_models, reason):

    - ``claude_models`` maps ``fable``/``opus``/``sonnet``/``haiku`` to the
      newest matching ``system.ai.claude-*`` id (mirrors
      ``discover_claude_models``).
    - ``codex_models`` is the list of ``system.ai.*gpt-*`` ids, newest first.
    - ``gemini_models`` is the list of ``system.ai.*gemini-*`` ids, newest first.
    - ``oss_models`` is the list of OSS-model ``system.ai.*`` ids.

    ``reason`` is None on success, else explains why nothing was found. Family
    bucketing is by name substring because the model-services API does not
    expose per-model API dialects.
    """
    ids, reason = list_model_services(workspace, token)
    if not ids:
        return {}, [], [], [], reason

    claude_models: dict[str, str] = {}
    for family in ANTHROPIC_FAMILIES:
        candidates = sorted(
            [m for m in ids if f"claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            claude_models[family] = candidates[0]
    # Smart routing's CLAUDE_ROUTE_ARMS require claude-opus-4-8, but the
    # newest-wins sort above picks opus-5 when both exist — making the
    # routing availability check fail. Pin opus-4-8 when it's available so
    # routing works with the current task_v2 router. Revert to
    # newest-wins once the router accepts opus-5 (PR databricks-eng/universe#2365446).
    _prefer_opus_4_8(claude_models, ids)

    codex_models = sorted([m for m in ids if "gpt-" in m], key=model_version_sort_key)
    gemini_models = sorted([m for m in ids if "gemini-" in m], key=model_version_sort_key)

    oss_models = [m for m in ids if any(family in m for family in _OSS_MODEL_FAMILIES)]

    if not (claude_models or codex_models or gemini_models or oss_models):
        sample = ", ".join(ids[:5])
        return (
            {},
            [],
            [],
            [],
            (
                "model-services returned model ids but none matched "
                f"claude/gpt/gemini/oss families (got: {sample})"
            ),
        )
    return claude_models, codex_models, gemini_models, oss_models, None


# --- Managed coding-agent config (admin-authored, developer-read) -----------

# The workspace-admin authors a CodingAgentConfig via the AI Gateway; developers read it
# (non-admin) through the List endpoint and apply it locally.
_CODING_AGENT_CONFIGS_API_PATH = "/api/ai-gateway/v2/coding-agent-configs"


def fetch_managed_coding_agent_configs(workspace: str, token: str) -> tuple[list[dict], str | None]:
    """List the workspace's managed CodingAgentConfig(s) via the AI Gateway."""
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}{_CODING_AGENT_CONFIGS_API_PATH}"
    payload, reason = _http_get_json(url, token, timeout=30)
    if reason is not None:
        return [], reason
    if isinstance(payload, dict):
        configs = payload.get("coding_agent_configs") or []
    elif isinstance(payload, list):
        configs = payload
    else:
        return [], "coding-agent-configs listing returned an unexpected response shape"
    if not isinstance(configs, list):
        return [], "coding-agent-configs listing returned an unexpected response shape"
    return [c for c in configs if isinstance(c, dict)], None


def fetch_model_recommendation(workspace: str, token: str) -> tuple[dict, str | None]:
    """Ask the AI Gateway which agent and model the caller's budget tier allows.

    The request takes no parameters: the server matches the caller's live spend against the managed
    config's budget tiers and resolves the agent first, then that agent's model.
    """
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}{_CODING_AGENT_CONFIGS_API_PATH}:recommendModel"
    payload, reason = _http_post_json(url, token, {}, timeout=30)
    if reason is not None:
        return {}, reason
    if not isinstance(payload, dict):
        return {}, "recommendModel returned an unexpected response shape"
    return payload, None


# The gateway's per-model price catalog (USD per million tokens), sourced from the same Zippy data
# the server uses to bill external-model spend. We read it to estimate per-model cost from token
# counts, since no API returns per-model dollars directly.
_EXTERNAL_PROVIDER_MODELS_API_PATH = "/api/ai-gateway/v2/external-provider-models"
_EXTERNAL_PROVIDER_MODELS_PAGE_SIZE = 1000
_EXTERNAL_PROVIDER_MODELS_MAX_PAGES = 20


def fetch_external_model_prices(workspace: str, token: str) -> tuple[list[dict], str | None]:
    """List external-provider models and their `base_pricing` (USD per million tokens) via the gateway.

    Returns ``(models, reason)`` with each model the raw API entry; ``reason`` is non-None on failure
    (callers omit cost rather than fail).
    """
    hostname = workspace_hostname(workspace)
    base_url = f"https://{hostname}{_EXTERNAL_PROVIDER_MODELS_API_PATH}"
    models: list[dict] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for _ in range(_EXTERNAL_PROVIDER_MODELS_MAX_PAGES):
        params: dict[str, str] = {"page_size": str(_EXTERNAL_PROVIDER_MODELS_PAGE_SIZE)}
        if page_token:
            params["page_token"] = page_token
        payload, reason = _http_get_json(f"{base_url}?{urlencode(params)}", token, timeout=30)
        if payload is None:
            # Return what we have if a later page blips; only the first-page failure is fatal.
            return (models, None) if models else ([], reason or "unknown error")
        if not isinstance(payload, dict):
            return [], "external-provider-models returned an unexpected response shape"
        for entry in payload.get("models") or []:
            if isinstance(entry, dict) and entry.get("model_name"):
                models.append(entry)
        page_token = payload.get("next_page_token") or None
        if not page_token or page_token in seen_tokens:
            break
        seen_tokens.add(page_token)
    if not models:
        return [], "external-provider-models listing returned no models"
    return models, None


# The `update_mask` paths a config PATCH sends. The server rejects paths outside its mutable set,
# so this omits `spec_version` (an estore-internal format marker, still sent in the body; naming it
# in the mask is the 400 this fixes) and the deprecated `budget_id`/`default_options`/`tiers`.
# Sending all owned paths lets a re-run clear an admin-removed field, since the server merges per path.
MANAGED_CONFIG_UPDATE_MASK_PATHS: tuple[str, ...] = (
    "display_name",
    "default_agent",
    "enabled_agents",
    "mcp_servers",
    "skills",
    "tracing",
    "budget_policy",
)


def _coding_agent_config_url(workspace: str, name: str | None = None) -> str:
    """The collection URL, or one config's resource URL when ``name`` is given.

    ``name`` is the server-assigned resource name (``coding-agent-configs/{id}``), which the Get and
    Update paths template directly, so it is appended as-is rather than rebuilt from an id.
    """
    hostname = workspace_hostname(workspace)
    base = f"https://{hostname}{_CODING_AGENT_CONFIGS_API_PATH}"
    if name is None:
        return base
    # The resource name already carries the collection segment, so join on the API root.
    root = base.rsplit("/coding-agent-configs", 1)[0]
    return f"{root}/{name.strip().strip('/')}"


def create_coding_agent_config(
    workspace: str, token: str, config: dict
) -> tuple[dict | None, str | None]:
    """Create the workspace's managed CodingAgentConfig.

    v0 allows at most one config per workspace, so this fails with ALREADY_EXISTS when one is
    already defined; callers should update that one instead of creating a second.
    """
    url = _coding_agent_config_url(workspace)
    payload, reason = _http_post_json(url, token, config, timeout=30)
    if reason is not None:
        return None, reason
    if not isinstance(payload, dict):
        return None, "coding-agent-config create returned an unexpected response shape"
    return payload, None


def update_coding_agent_config(
    workspace: str,
    token: str,
    name: str,
    config: dict,
    *,
    update_mask: tuple[str, ...] = MANAGED_CONFIG_UPDATE_MASK_PATHS,
) -> tuple[dict | None, str | None]:
    """Update an existing managed CodingAgentConfig in place.

    Preferred over delete-then-create: the server applies the mask inside a single entity-store
    update, so the workspace is never left without a config if the write fails partway. ``name``
    identifies the config and is echoed in the body, which is what the API's path template expects.

    ``update_mask`` goes in the query string, not the body. The RPC's HTTP binding is
    ``patch: "…/{coding_agent_config.name=coding-agent-configs/*}"`` with ``body:
    "coding_agent_config"`` — the config *is* the whole body, so a mask nested inside it is parsed
    as an unknown config field and the server reports the mask as missing:

        Field 'update_mask' is required and must contain at least one subfield with a non-default
        value!

    It is also a ``google.protobuf.FieldMask``, whose JSON/query form is one comma-separated string
    rather than a ``{"paths": [...]}`` object.
    """
    query = urlencode({"update_mask": ",".join(update_mask)})
    url = f"{_coding_agent_config_url(workspace, name)}?{query}"
    body = {**config, "name": name}
    payload, reason = _http_patch_json(url, token, body, timeout=30)
    if reason is not None:
        return None, reason
    if not isinstance(payload, dict):
        return None, "coding-agent-config update returned an unexpected response shape"
    return payload, None


def delete_coding_agent_config(workspace: str, token: str, name: str) -> str | None:
    """Delete a managed CodingAgentConfig by resource name. Returns None on success, else a reason.

    Returns only the failure reason: a successful delete responds with ``Empty``, so there is no
    payload worth handing back.
    """
    url = _coding_agent_config_url(workspace, name)
    _, reason = _http_delete(url, token, timeout=30)
    return reason


# --- MCP services (parallel to model services) -----------------------------


_MCP_SERVICE_NAME_PREFIX = "mcp-services/"


def _mcp_service_full_name(service: dict, required_prefix: str) -> str | None:
    """Extract the full UC name from one mcp-service entry, or None if it
    doesn't live under ``required_prefix`` or isn't ACTIVE."""
    name = service.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip().removeprefix(_MCP_SERVICE_NAME_PREFIX)
    if not name.startswith(required_prefix):
        return None
    status = ((service.get("config") or {}).get("connection") or {}).get("status")
    if status is not None and status != "ACTIVE":
        return None
    return name


def list_mcp_services(
    workspace: str, token: str, parent: str = "system.ai"
) -> tuple[list[str], str | None]:
    """List UC MCP services under ``parent`` (a ``<catalog>.<schema>`` ref).

    A non-None string indicates the listing call itself failed. Callers can inspect
    ``error`` for ``HTTP 404`` to distinguish "invalid location" from other failures.
    """
    hostname = workspace_hostname(workspace)
    url = (
        f"https://{hostname}/api/2.1/unity-catalog/mcp-services"
        f"?{urlencode({'parent': f'schemas/{parent}'})}"
    )
    payload, reason = _http_get_json(url, token, timeout=30)
    if payload is None:
        return [], reason
    expected_prefix = parent + "."
    data = cast(dict, payload) if isinstance(payload, dict) else {}
    names: list[str] = []
    for service in data.get("mcp_services") or []:
        if not isinstance(service, dict):
            continue
        full_name = _mcp_service_full_name(service, expected_prefix)
        if full_name:
            names.append(full_name)
    return sorted(set(names)), None


def build_mcp_service_url(workspace: str, full_name: str) -> str:
    return f"{workspace}/ai-gateway/mcp-services/{full_name}"


def build_skills_mcp_url(workspace: str, locations: list[str]) -> str:
    """Skills route with one ``?schema=`` scope per location. The trailing slash
    is required by the Envoy prefix even with no query params.

        []                        -> ``.../ai-gateway/skills/``
        ["main.default", "ml.a"]  -> ``.../ai-gateway/skills/?schema=main.default&schema=ml.a``
    """
    base = f"{workspace}/ai-gateway/skills/"
    if not locations:
        return base
    return base + "?" + urlencode([("schema", loc) for loc in locations])


# Maps the gateway routing dialect a coding tool speaks to the Model Provider
# Service `provider_type`s it can be backed by. claude speaks Anthropic's API,
# which both the `anthropic` and `amazon_bedrock` provider types serve (Bedrock
# just exposes different model ids); codex speaks OpenAI's; gemini speaks
# Google's, served by a Gemini Enterprise provider. Tags are the short form
# produced by `_provider_type_tag` (e.g. `amazon_bedrock`).
_TOOL_PROVIDER_TYPES: dict[str, tuple[str, ...]] = {
    "claude": ("anthropic", "amazon_bedrock"),
    "codex": ("openai",),
    "gemini": ("gemini_enterprise",),
}

# Provider types that expose Bedrock-style model ids (e.g.
# `us.anthropic.claude-sonnet-4-6`) instead of the agent's canonical model
# names, so ucode must pin them explicitly.
BEDROCK_PROVIDER_TYPES: tuple[str, ...] = ("amazon_bedrock",)


def tool_supports_provider_type(tool: str, provider_type: str) -> bool:
    """True when ``tool``'s API dialect can be backed by ``provider_type``."""
    return provider_type in _TOOL_PROVIDER_TYPES.get(tool, ())


def _provider_type_tag(provider_type: str | None) -> str:
    """Shorten `EXTERNAL_MODEL_PROVIDER_TYPE_ANTHROPIC` to `anthropic`."""
    if not isinstance(provider_type, str):
        return ""
    prefix = "EXTERNAL_MODEL_PROVIDER_TYPE_"
    tag = provider_type[len(prefix) :] if provider_type.startswith(prefix) else provider_type
    return tag.lower()


# The listing is paginated; a metastore with more services than one page silently truncated before
# this was honored, making services on later pages look nonexistent.
_PROVIDER_SERVICES_PAGE_SIZE = 100
_PROVIDER_SERVICES_MAX_PAGES = 50


def list_model_provider_services(
    workspace: str, token: str, *, parent: str | None = None, use_cache: bool = True
) -> tuple[list[dict], str | None]:
    """List Unity Catalog Model Provider Services on the workspace.

    Returns ``(services, reason)`` where each service is
    ``{"name": "<catalog>.<schema>.<service>", "provider_type": "anthropic"|...,
    "targets": [model_id, ...], "allow_all_targets": bool, "relayed": bool}``.
    ``targets`` is the provider-side model ids the service exposes (used to pin
    Bedrock model names). ``relayed`` is True for a credential-less Anthropic
    service (Claude Max/Team/Enterprise subscription relay). A non-None
    ``reason`` means the listing call itself failed.

    Pages through the endpoint: a metastore with more services than fit on one page used to have the
    remainder silently dropped, so a service that plainly existed looked absent. ``parent`` scopes
    the listing to one ``catalog.schema`` — the metastore-wide default is documented as an internal,
    likely-to-be-deprecated scope, so prefer passing it when the schema is known.

    A successful result is memoized per workspace for the life of the process, like the
    model-services listing: the listing is workspace-wide (filtered per agent afterwards by
    :func:`service_usable_for_tool`), so without the memo `ucode setup` re-lists it once per
    MPS-capable agent. Pass ``use_cache=False`` to force a fresh call.
    """
    # Keyed by workspace *and* parent: a `parent`-scoped listing holds only that schema's services,
    # so caching it under the workspace alone would serve a partial list to an unscoped caller (and
    # vice versa) — a service that plainly exists would look absent, the same failure pagination was
    # added to fix.
    cache_key = (workspace, parent or "")
    if use_cache:
        cached = _MODEL_PROVIDER_SERVICES_CACHE.get(cache_key)
        if cached is not None:
            # A fresh list of fresh dicts each time: callers treat the result as theirs (the wizard
            # filters it per agent), so handing out the cached objects would let one caller's edit
            # reach the next.
            return [dict(service) for service in cached], None

    hostname = workspace_hostname(workspace)
    services: list[dict] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    last_reason: str | None = None
    for _ in range(_PROVIDER_SERVICES_MAX_PAGES):
        params: dict[str, str] = {"page_size": str(_PROVIDER_SERVICES_PAGE_SIZE)}
        if parent:
            params["parent"] = f"schemas/{parent}"
        if page_token:
            params["page_token"] = page_token
        url = (
            f"https://{hostname}/api/2.1/unity-catalog/model-provider-services?{urlencode(params)}"
        )
        payload, reason = _http_get_json(url, token, timeout=30)
        if payload is None:
            # Surface the failure only if we have nothing yet; a mid-pagination blip still
            # returns whatever was collected.
            last_reason = reason
            break
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        for service in data.get("model_provider_services") or []:
            entry = _provider_service_entry(service)
            if entry is not None:
                services.append(entry)
        page_token = data.get("next_page_token") or None
        if not page_token:
            last_reason = None
            break
        if page_token in seen_tokens:
            break
        seen_tokens.add(page_token)

    if not services and last_reason is not None:
        return [], last_reason
    services.sort(key=lambda s: s["name"])
    if use_cache:
        _MODEL_PROVIDER_SERVICES_CACHE[cache_key] = [dict(service) for service in services]
    return services, None


def _provider_service_entry(raw_service: object) -> dict | None:
    """Normalize one listing entry, or None when it isn't usable."""
    if not isinstance(raw_service, dict):
        return None
    # A bare isinstance narrows to dict[Never, Never], which rejects string keys.
    service = cast("dict[str, object]", raw_service)
    raw_name = service.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        return None
    # The API returns `model-provider-services/<catalog>.<schema>.<name>`.
    full_name = raw_name.split("/", 1)[1] if "/" in raw_name else raw_name
    raw_config = service.get("config")
    config = cast("dict[str, object]", raw_config) if isinstance(raw_config, dict) else {}
    targets: list[str] = []
    raw_targets = config.get("targets")
    for target in raw_targets if isinstance(raw_targets, list) else []:
        if not isinstance(target, dict):
            continue
        model_id = cast("dict[str, object]", target).get("model")
        if isinstance(model_id, str) and model_id:
            targets.append(model_id)
    # Relayed = credential-less Anthropic (subscription relay). Only whether
    # it's relayed matters here; the tier (Max vs Team/Enterprise) is governed
    # server-side, so both launch identically.
    anthropic_cfg = config.get("anthropic")
    relayed = isinstance(anthropic_cfg, dict) and "relayed" in anthropic_cfg
    raw_type = config.get("provider_type")
    return {
        "name": full_name,
        "provider_type": _provider_type_tag(raw_type if isinstance(raw_type, str) else None),
        "targets": targets,
        "allow_all_targets": bool(config.get("allow_all_targets")),
        "relayed": relayed,
    }


def get_model_provider_service(
    service_name: str, workspace: str, token: str
) -> tuple[dict | None, str | None]:
    """Fetch one provider service by its full `catalog.schema.name`, bypassing the listing.

    The listing is paginated and metastore-wide, so any gap in it (a page we failed to fetch, a
    server-side filter) makes a service that plainly exists look absent. Addressing it directly
    removes that whole class of false negative.
    """
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}/api/2.1/unity-catalog/model-provider-services/{service_name}"
    payload, reason = _http_get_json(url, token, timeout=30)
    if payload is None:
        return None, reason
    entry = _provider_service_entry(payload)
    if entry is None:
        return None, "model-provider-service response had an unexpected shape"
    return entry, None


# The group every workspace user belongs to. USE_SCHEMA granted to it (directly or inherited from
# the catalog) is what lets an arbitrary developer pull a config that routes through an MPS in that
# schema; without it they hit "User does not have USE_SCHEMA on Schema <catalog>.<schema>".
_ALL_WORKSPACE_USERS_GROUP = "account users"


def all_users_can_use_schema(workspace: str, token: str, schema_full_name: str) -> bool | None:
    """Whether the `account users` group has USE_SCHEMA on ``<catalog>.<schema>``.

    Uses UC's effective-permissions API, so a USE_SCHEMA inherited from a catalog-level grant counts.
    Returns True/False, or None when the check itself could not be made (API unreachable or an
    unexpected shape) — callers treat None as "unknown" and skip the warning rather than cry wolf.

    A False here is only a heuristic: a workspace may instead grant access through team groups or
    individual users, so callers must warn rather than block on it.
    """
    hostname = workspace_hostname(workspace)
    principal = quote(_ALL_WORKSPACE_USERS_GROUP)
    url = (
        f"https://{hostname}/api/2.1/unity-catalog/effective-permissions/"
        f"schema/{schema_full_name}?principal={principal}"
    )
    payload, reason = _http_get_json(url, token, timeout=30)
    if reason is not None or not isinstance(payload, dict):
        return None
    for assignment in payload.get("privilege_assignments") or []:
        if not isinstance(assignment, dict):
            continue
        for entry in assignment.get("privileges") or []:
            if isinstance(entry, dict) and entry.get("privilege") in (
                "USE_SCHEMA",
                "ALL_PRIVILEGES",
            ):
                return True
    return False


def is_model_provider_feature_unavailable(reason: str | None) -> bool:
    """True when a model-provider-services API failure means the workspace
    simply hasn't enabled the feature (HTTP 400 "feature is not available"),
    as opposed to a transient or auth error. Callers use this to fall back to
    Databricks models silently rather than surfacing a scary error.
    """
    return bool(reason) and "feature is not available" in reason.lower()


def list_tool_provider_services(
    tool: str, workspace: str, token: str
) -> tuple[list[str], str | None]:
    """Provider-service names whose provider type matches ``tool``'s API dialect.

    Returns ``(names, reason)``; ``reason`` is non-None when the listing failed.
    """
    services, reason = list_model_provider_services(workspace, token)
    if reason is not None:
        return [], reason
    names = [s["name"] for s in services if service_usable_for_tool(tool, s)]
    return names, None


def service_usable_for_tool(tool: str, service: dict) -> bool:
    """True when ``tool`` can actually route through ``service``.

    Beyond the provider-type match, a Bedrock service is only usable for claude
    if it exposes at least one Claude model in its targets — otherwise there's no
    routable model id to pin. (Anthropic services use canonical names, so any
    match is usable.)
    """
    provider_type = service.get("provider_type", "")
    if not tool_supports_provider_type(tool, provider_type):
        return False
    if provider_type in BEDROCK_PROVIDER_TYPES:
        return bool(map_claude_family_models(service.get("targets") or []))
    return True


def resolve_provider_service(
    tool: str, service_name: str, workspace: str, token: str
) -> tuple[dict | None, str | None]:
    """Validate that ``service_name`` exists and is usable by ``tool``.

    Returns ``(service, error)``. On success ``service`` is the full service
    dict (``name``/``provider_type``/``targets``/``allow_all_targets``) and
    ``error`` is None. On failure ``service`` is None and ``error`` is an
    actionable message: the feature is off, the listing failed, the service
    doesn't exist, or its provider type isn't one ``tool`` can route to (e.g.
    pointing claude at an OpenAI service).
    """
    services, reason = list_model_provider_services(workspace, token)
    if is_model_provider_feature_unavailable(reason):
        return None, "Model Provider Service feature is not available yet for this workspace."
    if reason is not None:
        return None, f"Could not list model provider services: {reason}"
    match = next((s for s in services if s["name"] == service_name), None)
    if match is None:
        # Don't conclude "not found" from a listing that may be incomplete — a named service can be
        # fetched directly. Only when that 404s is it really absent.
        match, get_reason = get_model_provider_service(service_name, workspace, token)
        if match is None:
            usable = [
                s["name"] for s in services if tool_supports_provider_type(tool, s["provider_type"])
            ]
            suffix = f" Available for {tool}: {', '.join(usable)}." if usable else ""
            detail = f" ({get_reason})" if get_reason and "404" not in get_reason else ""
            return None, f"Model provider service '{service_name}' was not found.{detail}{suffix}"
    provider_type = match["provider_type"]
    if not tool_supports_provider_type(tool, provider_type):
        supported = ", ".join(_TOOL_PROVIDER_TYPES.get(tool, ())) or "none"
        return None, (
            f"Model provider service '{service_name}' is a '{provider_type}' provider, "
            f"which {tool} can't route to (supported: {supported})."
        )
    if provider_type in BEDROCK_PROVIDER_TYPES and not map_claude_family_models(
        match.get("targets") or []
    ):
        return None, (
            f"Model provider service '{service_name}' exposes no Claude models — "
            f"add Claude targets to it or pick a different service."
        )
    return match, None


# A Model Provider Service exposes Claude under per-family target ids: Bedrock as provider-side
# slugs (`us.anthropic.claude-sonnet-4-6`, `global.anthropic.claude-opus-4-8`, region-less
# `anthropic.claude-opus-4-8`), Anthropic as canonical names (`claude-sonnet-5`). Either way we map
# each target to a Claude family and keep the best id per family. Claude Code takes one default per
# family; users switch to any other listed id at runtime with `/model <full-id>` or `--model`.
_BEDROCK_CLAUDE_FAMILIES = ("opus", "sonnet", "haiku")
# When the same model/version is offered under several cross-region inference
# profiles, prefer the broadest-routing one as the pinned default.
_BEDROCK_REGION_RANK = {"global": 5, "us": 4, "eu": 3, "apac": 2, "": 1}


def _bedrock_target_family(model_id: str) -> str | None:
    lowered = model_id.lower()
    if "claude" not in lowered:
        return None
    return next((fam for fam in _BEDROCK_CLAUDE_FAMILIES if fam in lowered), None)


def _bedrock_region_rank(model_id: str) -> int:
    """Rank a target's cross-region inference profile (`us.`/`eu.`/`global.`/
    region-less) so ties on model version resolve deterministically."""
    head = model_id.lower().split("anthropic.", 1)[0].rstrip(".")
    return _BEDROCK_REGION_RANK.get(head, 0)


def _bedrock_sort_key(model_id: str) -> tuple:
    """Order targets best-first: highest model version, then preferred region."""
    version = tuple(int(n) for n in re.findall(r"\d+", model_id))
    return (version, _bedrock_region_rank(model_id))


def map_claude_family_models(targets: list[str]) -> dict[str, str]:
    """Map a service's Claude targets to ``{family: model_id}`` for opus/sonnet/haiku.

    Chooses the highest-versioned id per family and, on a version tie, the broadest-routing region
    profile (Bedrock targets carry a region prefix; canonical Anthropic ids rank equal, so version
    alone decides). Targets that don't name a Claude family are ignored, so a mixed catalog (e.g. a
    Bedrock service also exposing Titan embeddings) yields only the Claude families.
    """
    best_key: dict[str, tuple] = {}
    result: dict[str, str] = {}
    for model_id in targets:
        family = _bedrock_target_family(model_id)
        if not family:
            continue
        key = _bedrock_sort_key(model_id)
        if family not in best_key or key > best_key[family]:
            best_key[family] = key
            result[family] = model_id
    return result


# Claude Code starts every session on its opus tier, which the gateway 403s when a Model Provider
# Service declares no opus target. When opus is missing, fall back to the most capable tier the
# service does offer. opus > sonnet > haiku.
_CLAUDE_LAUNCH_TIER_PREFERENCE = ("opus", "sonnet", "haiku")


def resolve_provider_launch_model(model: str | None, provider_models: dict[str, str]) -> str | None:
    """Pick the model a provider-routed Claude session starts on, or None to keep Claude Code's default.

    ``provider_models`` maps the Claude families a service declares to their target ids (see
    ``map_claude_family_models``). With an explicit ``model`` (``ucode claude --model``) the user's
    choice wins: a family alias resolves to that tier's declared target (erroring when the service
    doesn't offer it), any other value is trusted as a raw target id the service allows. Without one,
    return None when the service offers opus — Claude Code's own default already works, so we avoid
    setting ANTHROPIC_MODEL and the duplicate ``/model`` picker row it produces — else the most
    capable tier the service does offer, so the launch doesn't dead-end on an unservable opus.
    """
    if model:
        if model in ANTHROPIC_FAMILIES:
            target = provider_models.get(model)
            if not target:
                available = ", ".join(sorted(provider_models)) or "none"
                raise RuntimeError(
                    f"This Model Provider Service does not offer a '{model}' model "
                    f"(available families: {available})."
                )
            return target
        return model
    if provider_models.get("opus"):
        return None
    return next(
        (
            provider_models[fam]
            for fam in _CLAUDE_LAUNCH_TIER_PREFERENCE
            if provider_models.get(fam)
        ),
        None,
    )


# `list_vector_search_catalog_schemas` walks Vector Search endpoints+indexes.
# `list_uc_functions_catalog_schemas` walks UC catalogs+schemas in parallel and
# keeps only schemas with at least one user function.

_UC_LIST_PAGE_SIZE = 200
_UC_LIST_MAX_PAGES = 50
_UC_FUNCTION_PROBE_WORKERS = 16
_UC_LIST_HTTP_TIMEOUT = 10
_UC_FUNCTION_PROBE_TIMEOUT = 5
_VECTOR_SEARCH_DEADLINE_SECONDS = 15.0
_UC_FUNCTIONS_DEADLINE_SECONDS = 20.0
# Most MCP services live outside `system.ai`, so this workspace-wide walk needs
# enough time to enumerate them; a slow workspace still degrades to partial
# results once the budget is exceeded instead of hanging indefinitely.
_MCP_SERVICES_WALK_DEADLINE_SECONDS = 30.0
# Skip UC catalogs whose schemas almost never carry user-callable functions
# you'd want to expose as agent tools.
_UC_FUNCTIONS_SKIP_CATALOGS = frozenset(
    {"__databricks_internal", "hive_metastore", "samples", "system"}
)


def _drain_with_deadline(futures: dict, deadline: float, on_result) -> None:
    """Iterate `futures` via `as_completed`, calling `on_result(value, key)` per
    completed future, until either all are done or `deadline` passes. Per-task
    exceptions are swallowed so one failure doesn't stop the rest."""
    remaining = max(0.0, deadline - time.monotonic())
    try:
        for future in as_completed(futures, timeout=remaining):
            try:
                value = future.result()
            except Exception:  # noqa: BLE001
                continue
            on_result(value, futures[future])
            if time.monotonic() > deadline:
                break
    except FutureTimeoutError:
        pass


def _paginated_json_items(
    base_url: str,
    token: str,
    *,
    items_key: str,
    extra_params: dict[str, str] | None = None,
    page_size: int = _UC_LIST_PAGE_SIZE,
    max_pages: int = _UC_LIST_MAX_PAGES,
    timeout: int = 30,
) -> tuple[list[dict], str | None]:
    """Walk a Databricks `next_page_token` listing and return all items.

    Returns (items, reason). Items are dicts; reason is None on success or a
    short description of why the walk stopped early.
    """
    items: list[dict] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    last_reason: str | None = None
    for _ in range(max_pages):
        params: dict[str, str] = {"max_results": str(page_size)}
        if extra_params:
            params.update(extra_params)
        if page_token:
            params["page_token"] = page_token
        url = f"{base_url}?{urlencode(params)}"
        payload, reason = _http_get_json(url, token, timeout=timeout)
        if payload is None:
            last_reason = reason
            break
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        raw = data.get(items_key) or []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    items.append(item)
        page_token = data.get("next_page_token") or None
        if not page_token or page_token in seen_tokens:
            break
        seen_tokens.add(page_token)
    return items, last_reason


def _vector_index_catalog_schema(index: dict) -> tuple[str, str] | None:
    """Pull (catalog, schema) from one vector-search index entry."""
    catalog = index.get("catalog_name")
    schema = index.get("schema_name")
    if isinstance(catalog, str) and isinstance(schema, str) and catalog and schema:
        return catalog, schema
    # Fallback: `name` is the fully-qualified UC name `catalog.schema.index`.
    name = index.get("name")
    if isinstance(name, str):
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] and parts[1]:
            return parts[0], parts[1]
    return None


def list_vector_search_catalog_schemas(
    workspace: str,
    token: str,
    *,
    deadline_seconds: float = _VECTOR_SEARCH_DEADLINE_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return sorted unique `(catalog, schema)` pairs that contain at least
    one Databricks Vector Search index. Walks the per-endpoint index listings
    in parallel under a wall-clock budget; returns partial results once
    `deadline_seconds` is exceeded.

    `on_progress`, if given, is called as each endpoint's listing completes with
    `(endpoints_done, endpoints_total, pairs_found)` for live count reporting.
    It is invoked serially from the draining thread (not the workers)."""
    hostname = workspace_hostname(workspace)
    deadline = time.monotonic() + deadline_seconds
    endpoints, reason = _paginated_json_items(
        f"https://{hostname}/api/2.0/vector-search/endpoints",
        token,
        items_key="endpoints",
        timeout=_UC_LIST_HTTP_TIMEOUT,
    )
    if not endpoints:
        return [], reason or "no vector search endpoints found"

    endpoint_names = [e["name"] for e in endpoints if isinstance(e.get("name"), str) and e["name"]]
    if not endpoint_names:
        return [], "no vector search endpoints with names"

    pairs: set[tuple[str, str]] = set()
    endpoints_total = len(endpoint_names)
    endpoints_done = 0
    workers = max(1, min(_UC_FUNCTION_PROBE_WORKERS, endpoints_total))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _paginated_json_items,
                f"https://{hostname}/api/2.0/vector-search/indexes",
                token,
                items_key="vector_indexes",
                extra_params={"endpoint_name": name},
                timeout=_UC_LIST_HTTP_TIMEOUT,
            ): name
            for name in endpoint_names
        }

        def collect(result, _endpoint):
            nonlocal endpoints_done
            indexes, _ = result
            for index in indexes:
                pair = _vector_index_catalog_schema(index)
                if pair:
                    pairs.add(pair)
            endpoints_done += 1
            if on_progress is not None:
                on_progress(endpoints_done, endpoints_total, len(pairs))

        _drain_with_deadline(futures, deadline, collect)
        pool.shutdown(wait=False, cancel_futures=True)

    if not pairs:
        return [], "no vector search indexes found"
    return sorted(pairs), None


def _schema_has_user_function(hostname: str, token: str, catalog: str, schema: str) -> bool:
    """One-shot probe: does `{catalog}.{schema}` expose any UC function?"""
    url = (
        f"https://{hostname}/api/2.1/unity-catalog/functions"
        f"?{urlencode({'catalog_name': catalog, 'schema_name': schema, 'max_results': '1'})}"
    )
    payload, _reason = _http_get_json(url, token, timeout=_UC_FUNCTION_PROBE_TIMEOUT)
    if not isinstance(payload, dict):
        return False
    functions = payload.get("functions") or []
    return isinstance(functions, list) and any(isinstance(item, dict) for item in functions)


def list_uc_functions_catalog_schemas(
    workspace: str,
    token: str,
    *,
    deadline_seconds: float = _UC_FUNCTIONS_DEADLINE_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return sorted unique `(catalog, schema)` pairs containing at least one
    user-defined UC function.

    `on_progress`, if given, is called during the function-probe phase with
    `(schemas_done, schemas_total, pairs_found)` for live count reporting. It is
    invoked serially from the draining thread (not the workers)."""
    hostname = workspace_hostname(workspace)
    deadline = time.monotonic() + deadline_seconds

    catalogs, catalogs_reason = _paginated_json_items(
        f"https://{hostname}/api/2.1/unity-catalog/catalogs",
        token,
        items_key="catalogs",
        timeout=_UC_LIST_HTTP_TIMEOUT,
    )
    if not catalogs:
        return [], catalogs_reason or "no UC catalogs found"

    catalog_names = [
        c["name"]
        for c in catalogs
        if isinstance(c.get("name"), str)
        and c["name"]
        and c["name"] not in _UC_FUNCTIONS_SKIP_CATALOGS
    ]
    if not catalog_names:
        return [], "no user UC catalogs found"
    if time.monotonic() > deadline:
        return [], "deadline exceeded while listing UC catalogs"

    # Parallel per-catalog schema listing.
    candidate_pairs: list[tuple[str, str]] = []
    schema_workers = max(1, min(_UC_FUNCTION_PROBE_WORKERS, len(catalog_names)))
    with ThreadPoolExecutor(max_workers=schema_workers) as pool:
        schema_futures = {
            pool.submit(
                _paginated_json_items,
                f"https://{hostname}/api/2.1/unity-catalog/schemas",
                token,
                items_key="schemas",
                extra_params={"catalog_name": cat},
                timeout=_UC_LIST_HTTP_TIMEOUT,
            ): cat
            for cat in catalog_names
        }

        def collect_schemas(result, catalog):
            schemas, _ = result
            for schema in schemas:
                schema_name = schema.get("name")
                # `information_schema` is auto-attached to every catalog and
                # never holds user functions.
                if (
                    isinstance(schema_name, str)
                    and schema_name
                    and schema_name != "information_schema"
                ):
                    candidate_pairs.append((catalog, schema_name))

        _drain_with_deadline(schema_futures, deadline, collect_schemas)
        pool.shutdown(wait=False, cancel_futures=True)

    if not candidate_pairs:
        if time.monotonic() > deadline:
            return [], "deadline exceeded while listing UC schemas"
        return [], "no UC schemas found"

    # Parallel function-existence probes.
    pairs: set[tuple[str, str]] = set()
    schemas_total = len(candidate_pairs)
    schemas_done = 0
    with ThreadPoolExecutor(max_workers=_UC_FUNCTION_PROBE_WORKERS) as pool:
        probe_futures = {
            pool.submit(_schema_has_user_function, hostname, token, cat, schema): (cat, schema)
            for cat, schema in candidate_pairs
        }

        def collect_pair(has_fn, pair):
            nonlocal schemas_done
            if has_fn:
                pairs.add(pair)
            schemas_done += 1
            if on_progress is not None:
                on_progress(schemas_done, schemas_total, len(pairs))

        _drain_with_deadline(probe_futures, deadline, collect_pair)
        pool.shutdown(wait=False, cancel_futures=True)

    if not pairs:
        if time.monotonic() > deadline:
            return [], "deadline exceeded probing UC schemas for functions"
        return [], "no UC schemas with user functions found"
    return sorted(pairs), None


def list_all_mcp_services(
    workspace: str,
    token: str,
    *,
    deadline_seconds: float = _MCP_SERVICES_WALK_DEADLINE_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[str], str | None]:
    """Return sorted unique MCP-service full names across every `<catalog>.<schema>`
    in the workspace. The mcp-services API is one-schema-per-call, so this walks
    catalogs -> schemas -> mcp-services in parallel under a wall-clock budget,
    returning partial results once `deadline_seconds` is exceeded.

    `on_progress`, if given, is called as each schema's listing completes with
    `(schemas_done, schemas_total, services_found)` so callers can render a live
    count. It is invoked serially from the draining thread (not the workers).

    This walk is the slow, workspace-wide counterpart to `list_mcp_services`
    (single schema)."""
    hostname = workspace_hostname(workspace)
    deadline = time.monotonic() + deadline_seconds

    catalogs, catalogs_reason = _paginated_json_items(
        f"https://{hostname}/api/2.1/unity-catalog/catalogs",
        token,
        items_key="catalogs",
        timeout=_UC_LIST_HTTP_TIMEOUT,
    )
    if not catalogs:
        return [], catalogs_reason or "no UC catalogs found"

    catalog_names = [
        c["name"]
        for c in catalogs
        if isinstance(c.get("name"), str)
        and c["name"]
        and c["name"] not in _UC_FUNCTIONS_SKIP_CATALOGS
    ]
    if not catalog_names:
        return [], "no user UC catalogs found"
    if time.monotonic() > deadline:
        return [], "deadline exceeded while listing UC catalogs"

    # Parallel per-catalog schema listing.
    schema_refs: list[str] = []
    schema_workers = max(1, min(_UC_FUNCTION_PROBE_WORKERS, len(catalog_names)))
    with ThreadPoolExecutor(max_workers=schema_workers) as pool:
        schema_futures = {
            pool.submit(
                _paginated_json_items,
                f"https://{hostname}/api/2.1/unity-catalog/schemas",
                token,
                items_key="schemas",
                extra_params={"catalog_name": cat},
                timeout=_UC_LIST_HTTP_TIMEOUT,
            ): cat
            for cat in catalog_names
        }

        def collect_schemas(result, catalog):
            schemas, _ = result
            for schema in schemas:
                schema_name = schema.get("name")
                if (
                    isinstance(schema_name, str)
                    and schema_name
                    and schema_name != "information_schema"
                ):
                    schema_refs.append(f"{catalog}.{schema_name}")

        _drain_with_deadline(schema_futures, deadline, collect_schemas)
        pool.shutdown(wait=False, cancel_futures=True)

    if not schema_refs:
        if time.monotonic() > deadline:
            return [], "deadline exceeded while listing UC schemas"
        return [], "no UC schemas found"

    # Parallel per-schema mcp-services listing.
    names: set[str] = set()
    schemas_total = len(schema_refs)
    schemas_done = 0
    probe_workers = max(1, min(_UC_FUNCTION_PROBE_WORKERS, schemas_total))
    with ThreadPoolExecutor(max_workers=probe_workers) as pool:
        service_futures = {
            pool.submit(list_mcp_services, workspace, token, ref): ref for ref in schema_refs
        }

        def collect_services(result, _ref):
            nonlocal schemas_done
            found, _ = result
            names.update(found)
            schemas_done += 1
            if on_progress is not None:
                on_progress(schemas_done, schemas_total, len(names))

        _drain_with_deadline(service_futures, deadline, collect_services)
        pool.shutdown(wait=False, cancel_futures=True)

    if not names:
        if time.monotonic() > deadline:
            return [], "deadline exceeded while listing MCP services"
        return [], "no MCP services found"
    return sorted(names), None


def _get_anthropic_models_json(workspace: str, token: str) -> tuple[dict | list | None, str | None]:
    hostname = workspace_hostname(workspace)
    return _http_get_json(
        f"https://{hostname}{ANTHROPIC_MODELS_PATH}",
        token,
        max_retries=_ANTHROPIC_MODEL_DISCOVERY_SETUP_MAX_RETRIES,
    )


def list_anthropic_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    """List every model id advertised by AI Gateway's Anthropic endpoint.

    Claude Code's native gateway discovery consumes this same catalog, so callers
    using that mode must not apply ucode's legacy ``databricks-claude-*`` family
    validation.
    """
    catalog = list_anthropic_model_catalog(workspace, token)
    return catalog.model_ids, catalog.error_msg


def list_anthropic_model_catalog(workspace: str, token: str) -> AnthropicModelCatalog:
    """Return advertised Anthropic model ids and their optional display names."""
    payload, reason = _get_anthropic_models_json(workspace, token)
    if payload is None:
        return AnthropicModelCatalog(model_ids=[], model_id_to_display_name={}, error_msg=reason)

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    model_ids: list[str] = []
    display_names: dict[str, str] = {}
    seen: set[str] = set()
    for model in data.get("data", []):
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id and model_id not in seen:
            seen.add(model_id)
            model_ids.append(model_id)
            display_name = model.get("display_name")
            if isinstance(display_name, str) and display_name:
                display_names[model_id] = display_name
    if model_ids:
        return AnthropicModelCatalog(model_ids=model_ids, model_id_to_display_name=display_names)
    return AnthropicModelCatalog(
        model_ids=[],
        model_id_to_display_name={},
        error_msg="AI Gateway returned no Anthropic model ids",
    )


def discover_claude_models(workspace: str, token: str) -> tuple[dict[str, str], str | None]:
    """Discover Claude families on this workspace's AI Gateway.

    Returns (models_by_family, reason). reason is None on success; otherwise it
    describes why the dict is empty (HTTP error, network error, or no models
    matching the expected naming convention).
    """
    payload, reason = _get_anthropic_models_json(workspace, token)
    if payload is None:
        return {}, reason

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    raw_ids = [
        m["id"]
        for m in data.get("data", [])
        if isinstance(m.get("id"), str) and not m["id"].endswith("-anthropic")
    ]

    result: dict[str, str] = {}
    for family in ANTHROPIC_FAMILIES:
        candidates = sorted(
            [m for m in raw_ids if f"claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            result[family] = candidates[0]
    # Same opus-4-8 pin as discover_model_services — see comment there.
    _prefer_opus_4_8(result, raw_ids)
    if result:
        return result, None
    if not raw_ids:
        return {}, "AI Gateway returned no Claude model ids"
    sample = ", ".join(raw_ids[:5])
    families = ",".join(ANTHROPIC_FAMILIES)
    return {}, (
        "AI Gateway returned model ids but none matched "
        f"`*-claude-{{{families}}}-*` (got: {sample})"
    )


def fetch_ai_gateway_claude_models(workspace: str, token: str) -> dict[str, str]:
    """Backwards-compatible wrapper that discards the diagnostic reason."""
    models, _ = discover_claude_models(workspace, token)
    return models


def model_version_sort_key(name: str) -> tuple:
    """Sort endpoint names so newer model versions come first.

    Endpoint names embed a dotted version as dash-separated digits, e.g.
    `databricks-gemini-3-5-flash` (3.5) or `databricks-gemini-3-flash` (3.0).
    Plain alphabetical sorting buries `3-5-flash` below `2-5-flash`; this key
    groups by the non-numeric prefix, orders by version descending, then falls
    back to the remaining text so ties stay stable and deterministic.
    """
    tokens = name.split("-")
    start = next((i for i, tok in enumerate(tokens) if tok.isdigit()), None)
    if start is None:
        # No version segment — sort these after versioned ones, alphabetically.
        # The leading 1 keeps the whole group below every versioned name (0).
        return (1, name, (), "")
    end = start
    while end < len(tokens) and tokens[end].isdigit():
        end += 1
    version = tuple(int(tok) for tok in tokens[start:end])
    # Pad to a fixed width so (3,) compares as (3, 0) — i.e. 3.0 < 3.5.
    padded = (version + (0, 0, 0))[:3]
    prefix = "-".join(tokens[:start])
    suffix = "-".join(tokens[end:])
    # Negate version components for descending order within a prefix group.
    return (0, prefix, tuple(-v for v in padded), suffix)


def discover_endpoints_with_api_type(
    workspace: str,
    token: str,
    api_type: str,
    *,
    sort_key=None,
) -> tuple[list[str], str | None]:
    """List endpoint names whose served_entities expose api_type with v2 support.

    Returns (endpoints, reason). reason is None on success; otherwise it
    describes why the list is empty. `sort_key` overrides the default
    alphabetical ordering of the returned names.
    """
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(
        f"https://{hostname}/api/2.0/serving-endpoints:foundation-models", token
    )
    if payload is None:
        return [], reason

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    endpoints = data.get("endpoints", [])
    out: list[str] = []
    saw_endpoint_without_v2 = False
    for ep in endpoints:
        name = ep.get("name", "")
        entities = ep.get("config", {}).get("served_entities", [])
        api_types: set[str] = set()
        any_v2 = False
        for se in entities:
            fm = se.get("foundation_model", {})
            if fm.get("ai_gateway_v2_supported") is True:
                any_v2 = True
                api_types.update(fm.get("api_types", []))
        if not any_v2 and entities:
            saw_endpoint_without_v2 = True
        if api_type in api_types:
            out.append(name)
    if out:
        return sorted(out, key=sort_key), None
    if not endpoints:
        return [], "foundation-models listing returned no endpoints"
    if saw_endpoint_without_v2:
        return [], (
            f"no endpoint exposes api_type `{api_type}` with "
            "`ai_gateway_v2_supported=true` (workspace has v1-only endpoints)"
        )
    return [], f"no endpoint exposes api_type `{api_type}`"


def _fetch_endpoints_with_api_type(workspace: str, token: str, api_type: str) -> list[str]:
    """Backwards-compatible wrapper that discards the diagnostic reason."""
    endpoints, _ = discover_endpoints_with_api_type(workspace, token, api_type)
    return endpoints


def discover_gemini_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    # Order newest model version first so `default_model()` (which picks the
    # first entry) launches e.g. gemini-3.5-flash rather than gemini-2.5-flash.
    return discover_endpoints_with_api_type(
        workspace, token, "gemini/v1/generateContent", sort_key=model_version_sort_key
    )


def discover_codex_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    # Order newest model version first (like `discover_gemini_models`), so the picker's top choice
    # and default is e.g. gpt-5-4 rather than the alphabetically-first gpt-5.
    return discover_endpoints_with_api_type(
        workspace, token, "openai/v1/responses", sort_key=model_version_sort_key
    )


def fetch_gemini_models(workspace: str, token: str) -> list[str]:
    models, _ = discover_gemini_models(workspace, token)
    return models


def fetch_codex_models(workspace: str, token: str) -> list[str]:
    models, _ = discover_codex_models(workspace, token)
    return models


class GatewayProbe(NamedTuple):
    reachable: bool
    detail: str
    resource_available: bool = False
    conclusive: bool = True


_MODEL_SERVICE_PROBE_PAGE_SIZE = 50
_MODEL_SERVICE_PROBE_MAX_PAGES = 20
_MODEL_SERVICE_EMPTY_DETAIL = (
    "reachable, no accessible model services returned; "
    "check USE CATALOG on system, and USE SCHEMA and EXECUTE on system.ai"
)


def _probe_model_services(workspace: str, token: str) -> GatewayProbe:
    hostname = workspace_hostname(workspace)
    base = f"https://{hostname}/api/2.1/unity-catalog/model-services"
    page_token: str | None = None
    for page in range(_MODEL_SERVICE_PROBE_MAX_PAGES):
        params: dict[str, object] = {"page_size": _MODEL_SERVICE_PROBE_PAGE_SIZE}
        if page_token:
            params["page_token"] = page_token
        payload, reason = _http_get_json(f"{base}?{urlencode(params)}", token)
        if payload is None:
            if page == 0:
                return GatewayProbe(False, reason or "unknown error")
            return GatewayProbe(True, "reachable", conclusive=False)
        if isinstance(payload, dict) and payload.get("model_services"):
            return GatewayProbe(True, "reachable, accessible model service returned", True)
        page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
        if not page_token:
            return GatewayProbe(True, _MODEL_SERVICE_EMPTY_DETAIL)
    return GatewayProbe(True, "reachable", conclusive=False)


def _raise_ai_gateway_auth_failure(workspace: str, reason: str) -> NoReturn:
    raise RuntimeError(
        f"Databricks rejected the access token for {workspace} ({reason}). "
        f"Try:\n"
        f"  databricks auth logout --host {workspace}\n"
        f"  databricks auth login --host {workspace}"
    )


def _raise_ai_gateway_scope_failure(workspace: str, reason: str) -> NoReturn:
    raise RuntimeError(
        f"The access token for {workspace} is missing an OAuth scope required by the "
        f"AI Gateway APIs ({reason}). Re-authenticate to mint a token with the needed "
        f"scopes:\n"
        f"  databricks auth login --host {workspace}"
    )


def _raise_model_service_permission_failure(workspace: str, model_service_reason: str) -> NoReturn:
    raise RuntimeError(
        "Databricks Unity AI Gateway model service access could not be verified on "
        f"{workspace} ({model_service_reason}). Listing Unity Catalog model services requires "
        "USE CATALOG on `system`, and USE SCHEMA and EXECUTE on `system.ai`."
    )


def probe_unity_gateway_capabilities(workspace: str, token: str) -> GatewayProbe:
    """Return the model service probe, raising if model service access can't be verified."""
    model_service_probe = _probe_model_services(workspace, token)
    if model_service_probe.reachable:
        # resource_available, reachable-but-empty, and inconclusive are all non-fatal: the
        # caller surfaces the detail as a warning when no accessible model service came back.
        return model_service_probe

    reason = model_service_probe.detail
    if _looks_like_definitive_auth_failure(reason):
        _raise_ai_gateway_auth_failure(workspace, reason)
    if _looks_like_scope_failure(reason):
        _raise_ai_gateway_scope_failure(workspace, reason)
    if _looks_like_permission_failure(reason):
        _raise_model_service_permission_failure(workspace, reason)

    raise RuntimeError(
        "Databricks Unity AI Gateway is not enabled on this workspace: model services "
        f"({reason}) are not available. See {AI_GATEWAY_DOCS_URL}"
    )


def _looks_like_definitive_auth_failure(reason: str) -> bool:
    """True when the token itself is rejected (401, or an invalid-token 400).

    A 403 is left to the scope and permission routing, since it can mean a
    missing OAuth scope or missing Unity Catalog grants rather than a bad token.
    """
    if "HTTP 401" in reason:
        return True
    return "HTTP 400" in reason and "invalid token" in reason.lower()


def _looks_like_scope_failure(reason: str) -> bool:
    """True for a 403 that reports the OAuth *token* is missing a required scope.

    Matched to the OAuth-token wording so a PAT's permission 403 -- which
    re-login cannot fix -- is not misrouted to the re-login hint and instead
    falls through to the grant guidance.
    """
    lowered = reason.lower()
    return "http 403" in lowered and "oauth token" in lowered and "required scopes" in lowered


def _looks_like_permission_failure(reason: str) -> bool:
    return "HTTP 403" in reason


CODING_AGENT_RECOMMEND_MODEL_PATH = "/api/ai-gateway/v2/coding-agent-configs:recommendModel"


def resolve_current_budget_spend(
    workspace: str,
    token: str,
    *,
    timeout: int = 10,
) -> tuple[tuple[Decimal, Decimal] | None, str | None]:
    """Fetch the caller's coding-agent budget spend and alert threshold.

    Reads them off `recommendModel`, which returns the spend its model
    recommendation was based on. `available_models` is empty since we want the
    spend, not the recommendation.

    Returns `((spend, threshold), None)` or `(None, reason)`. Absence is
    routine — the endpoint needs a per-org SAFE flag (default off) and a
    coding-agent config — so it never raises.
    """
    url = f"https://{workspace_hostname(workspace)}{CODING_AGENT_RECOMMEND_MODEL_PATH}"
    payload, reason = _http_post_json(url, token, {"available_models": []}, timeout=timeout)
    if payload is None:
        return None, reason or "unknown error"
    if not isinstance(payload, dict):
        return None, "response was not a JSON object"

    # The threshold is what anchors the spend: a caller with a per-user threshold but no spend yet
    # this period gets `effective_threshold` set and `current_spend` omitted (see the server's
    # per-user spend resolution). Treat an *absent* spend as $0 rather than "no budget", so a
    # developer who hasn't spent anything still sees their budget instead of a blank. With no
    # threshold there is nothing to measure against, so that genuinely counts as no spend.
    threshold = _parse_decimal(payload.get("effective_threshold"))
    if threshold is None:
        return None, "workspace reported no coding-agent budget spend"
    raw_spend = payload.get("current_spend")
    if raw_spend is None:
        spend: Decimal | None = Decimal(0)
    else:
        # Present but unparseable is corrupt data, not zero spend — don't silently mask it.
        spend = _parse_decimal(raw_spend)
        if spend is None:
            return None, "workspace reported no coding-agent budget spend"
    return (spend, threshold), None


def _parse_decimal(value: object) -> Decimal | None:
    if isinstance(value, str) and value.strip():
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            return None
    if isinstance(value, int):
        return Decimal(value)
    return None


class SqlWarehouse(NamedTuple):
    http_path: str
    label: str
    state: str


def discover_sql_warehouses(
    workspace: str,
    token: str,
    *,
    warehouse_id: str | None = None,
) -> list[SqlWarehouse]:
    """Candidate warehouses to run the usage query against, RUNNING ones first.

    Several are returned because a warehouse can report RUNNING and still refuse
    connections, so callers fall through to the next one. An explicit
    `warehouse_id` skips discovery entirely.
    """
    if warehouse_id:
        return [SqlWarehouse(_warehouse_http_path(warehouse_id), warehouse_id, "REQUESTED")]

    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/api/2.0/sql/warehouses",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = body.strip() or f"HTTP {exc.code}"
        raise RuntimeError(f"Failed to list SQL warehouses: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Could not reach workspace hostname {hostname}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks warehouse discovery returned invalid JSON.") from exc

    warehouses = payload.get("warehouses")
    if not isinstance(warehouses, list) or not warehouses:
        raise RuntimeError(
            "No SQL warehouses found in this workspace. Create one or pass `--warehouse-id`."
        )

    candidates: list[SqlWarehouse] = []
    for entry in warehouses:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            continue
        name = entry.get("name")
        state = entry.get("state", "UNKNOWN")
        label = name if isinstance(name, str) and name else entry_id
        candidates.append(SqlWarehouse(_warehouse_http_path(entry_id), label, str(state)))

    if not candidates:
        raise RuntimeError("No usable SQL warehouse was returned by Databricks.")
    # Stopped warehouses work too, but cold-starting one costs minutes.
    candidates.sort(key=lambda w: w.state != "RUNNING")
    return candidates


def _warehouse_http_path(warehouse_id: str) -> str:
    return f"/sql/1.0/warehouses/{warehouse_id.strip()}"


def run_usage_query(
    workspace: str,
    http_path: str,
    token: str,
    query: str,
    on_connected: Callable[[], None] | None = None,
) -> tuple[list[str], list[tuple]]:
    """Run `query` on one warehouse.

    `on_connected` fires once the connection opens — the point a stopped
    warehouse has finished starting — so callers can update their progress
    message.
    """
    try:
        logging.getLogger("databricks.sql").setLevel(logging.ERROR)
        from databricks import sql
    except ImportError as exc:
        raise RuntimeError(
            "`databricks-sql-connector` is not installed. "
            "Install it with `pip install databricks-sql-connector`."
        ) from exc

    try:
        with sql.connect(
            server_hostname=workspace_hostname(workspace),
            http_path=http_path,
            access_token=token,
        ) as connection:
            if on_connected is not None:
                on_connected()
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in (cursor.description or [])]
                rows = cast(list[tuple], cursor.fetchall())
    except ServerOperationError as exc:
        if _is_usage_table_access_error(exc):
            raise RuntimeError(
                "Unable to read `system.ai_gateway.usage`. Ask your workspace admin "
                "to enable READ access to `system.ai_gateway.usage` for your account."
            ) from exc
        raise RuntimeError(f"Usage query failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Usage query failed: {exc}") from exc

    return columns, rows


def _is_usage_table_access_error(exc: BaseException) -> bool:
    """Return True when a `ServerOperationError` blocks reads of
    `system.ai_gateway.usage` — gated on one of the bracketed error codes
    `INSUFFICIENT_PERMISSIONS` plus a `system.ai_gateway` substring (identifier quoting
    stripped first)."""
    normalized = str(exc).lower().translate(str.maketrans("", "", """`[]"'"""))
    if "system.ai_gateway" not in normalized:
        return False
    return "insufficient_permissions" in normalized


# ---------------------------------------------------------------------------
# URL builders (AI Gateway v2 only — no fallback to /serving-endpoints)
# ---------------------------------------------------------------------------


def build_tool_base_url(tool: str, workspace: str) -> str:
    if tool == "codex":
        return f"{workspace}/ai-gateway/codex/v1"
    if tool == "claude":
        return f"{workspace}/ai-gateway/anthropic"
    if tool == "gemini":
        return f"{workspace}/ai-gateway/gemini"
    if tool == "opencode":
        raise RuntimeError(
            "OpenCode has multiple base URLs — use build_opencode_base_urls() instead."
        )
    if tool == "copilot":
        raise RuntimeError(
            "Copilot has multiple base URLs — use build_copilot_base_urls() instead."
        )
    if tool == "pi":
        raise RuntimeError("Pi has multiple base URLs — use build_pi_base_urls() instead.")
    if tool == "omp":
        raise RuntimeError("Oh My Pi has multiple base URLs — use build_pi_base_urls() instead.")
    raise RuntimeError(f"Unsupported tool '{tool}'.")


def build_opencode_base_urls(workspace: str) -> dict[str, str]:
    return {
        "anthropic": build_tool_base_url("claude", workspace) + "/v1",
        "gemini": build_tool_base_url("gemini", workspace) + "/v1beta",
        "oss": f"{workspace}/ai-gateway/mlflow/v1",
    }


def build_pi_base_urls(workspace: str) -> dict[str, str]:
    # Pi speaks each model family's native API dialect to its dedicated gateway
    # path (verified end-to-end). Each `api` type appends its own path suffix:
    #
    # - anthropic-messages       appends `/v1/messages`
    # - openai-responses         appends `/responses`
    # - google-generative-ai     appends `/v1beta/models/{id}:streamGenerateContent`
    # - openai-completions       appends `/chat/completions`
    #
    # So the baseUrls below stop just before the suffix Pi will tack on.
    # Compat flags applied per-provider in agents/pi.py; required for `oss`
    # only (MLflow rejects `store` and `tools[].function.strict`).
    return {
        "claude": build_tool_base_url("claude", workspace),
        "openai": build_tool_base_url("codex", workspace),
        "gemini": build_tool_base_url("gemini", workspace) + "/v1beta",
    }


def build_copilot_base_url(workspace: str) -> str:
    # Copilot CLI's `openai` provider appends `/chat/completions` to the
    # configured base URL. The Databricks MLflow chat-completions gateway is
    # OpenAI-compatible and serves Claude, codex (gpt-5), and gemini models
    # behind one URL.
    return f"{workspace}/ai-gateway/mlflow/v1"


def build_shared_base_urls(workspace: str) -> dict[str, str | dict[str, str]]:
    urls: dict[str, str | dict[str, str]] = {
        "codex": build_tool_base_url("codex", workspace),
        "claude": build_tool_base_url("claude", workspace),
        "gemini": build_tool_base_url("gemini", workspace),
        "opencode": build_opencode_base_urls(workspace),
        "copilot": build_copilot_base_url(workspace),
        "pi": build_pi_base_urls(workspace),
        "omp": build_pi_base_urls(workspace),
    }
    return urls
