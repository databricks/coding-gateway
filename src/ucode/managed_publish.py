"""Input handling for ``ucode publish``: turn a config source into a publishable payload.

``ucode publish`` publishes either the locally authored managed config (no ``-f``) or a config file
produced by ``ucode export`` (``-f <path>``). Both routes converge here: a source dict is validated
against the configured workspace and the ``spec_version`` envelope, canonicalized through the same
normalize/serialize path the rest of ucode uses, and returned as the internal manifest (for
validation, summaries, and diffs) plus the API payload (the canonical ``CodingAgentConfig`` with
``spec_version`` but without ``workspace``).

Pure input handling: no auth, no admin check, no network, no publish. Invalid input raises
RuntimeError with an actionable message so ``publish`` fails before it touches the workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from ucode.managed_config import normalize_managed_config
from ucode.managed_export import EXPORT_SPEC_VERSION, build_export_payload
from ucode.managed_setup import serialize_managed_config
from ucode.ui import normalize_workspace_url

_ENVELOPE_FIELDS = ("workspace", "spec_version")


def load_publish_payload(file_path: str | None) -> dict:
    """Return the source config dict for ``ucode publish``.

    With no ``file_path`` the locally authored config is serialized in-process via
    :func:`ucode.managed_export.build_export_payload` (no subprocess, no captured stdout). With a
    ``file_path`` the file is read as UTF-8 JSON; a missing file, non-UTF-8 bytes, malformed JSON, or
    a non-object root each raise RuntimeError with an actionable message.
    """
    if file_path is None:
        return build_export_payload()

    path = Path(file_path).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(
            f"No config file at {path}. Pass an existing file to `ucode publish -f`, or run "
            "`ucode publish` with no file to publish the locally authored config."
        ) from None
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{path} is not valid UTF-8 text: {exc}.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not read {path}: {exc}.") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{path} must contain a JSON object at the top level, not a {type(payload).__name__}."
        )
    return payload


def parse_publish_payload(payload: object, workspace: str) -> tuple[dict, dict]:
    """Validate and canonicalize a source config into ``(manifest, api_payload)``.

    ``manifest`` is ucode's internal normalized shape (for validation, summaries, and diffs).
    ``api_payload`` is the canonical proto-JSON ``CodingAgentConfig`` carrying ``spec_version`` but
    not ``workspace``, ready for the configuration API.

    Enforces, before any of it reaches the workspace: an object root; a top-level ``workspace`` that
    normalizes to the configured one (the file can never redirect publication elsewhere); a
    ``spec_version`` that is a JSON integer (not a boolean or float) equal to the supported version;
    no server-owned ``name``; and no unknown or lossy fields (anything normalization would silently
    drop is rejected instead).
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"The config must be a JSON object, not a {type(payload).__name__}.")
    payload = cast("dict[str, object]", payload)

    file_workspace = payload.get("workspace")
    if not isinstance(file_workspace, str) or not file_workspace.strip():
        raise RuntimeError(
            'The config is missing a top-level "workspace". Export it with `ucode export` so it '
            "records the workspace it belongs to."
        )
    if normalize_workspace_url(file_workspace) != normalize_workspace_url(workspace):
        raise RuntimeError(
            f"The config is for {file_workspace}, but the configured workspace is {workspace}. "
            "`ucode publish` only publishes to the configured workspace; re-run `ucode configure` "
            "to switch workspaces."
        )

    spec_version = payload.get("spec_version")
    if isinstance(spec_version, bool) or not isinstance(spec_version, int):
        raise RuntimeError(
            f'The config "spec_version" must be the integer {EXPORT_SPEC_VERSION}. Export it with '
            "`ucode export` to get a config this ucode can publish."
        )
    if spec_version != EXPORT_SPEC_VERSION:
        raise RuntimeError(
            f"The config is spec_version {spec_version}, but this ucode publishes version "
            f"{EXPORT_SPEC_VERSION}. Upgrade ucode, or re-export the config."
        )

    config = {key: value for key, value in payload.items() if key not in _ENVELOPE_FIELDS}
    if "name" in config:
        raise RuntimeError(
            'The config includes a server-owned "name" field. Remove it — the workspace assigns the '
            "resource name when the config is created."
        )

    manifest = normalize_managed_config(config)
    canonical = serialize_managed_config(manifest)
    canonical.pop("name", None)
    if canonical != config:
        dropped = sorted(key for key in config if key not in canonical)
        altered = sorted(
            key for key in config if key in canonical and config[key] != canonical[key]
        )
        offending = ", ".join(dropped + altered)
        raise RuntimeError(
            f"The config has fields ucode does not recognize or cannot publish: {offending}. "
            "Server-owned fields (workspace ids, timestamps, user ids) and unknown fields must be "
            "removed. Re-export a clean config with `ucode export`."
        )

    api_payload = {"spec_version": spec_version, **canonical}
    return manifest, api_payload


__all__ = ["load_publish_payload", "parse_publish_payload"]
