"""`ucode export`: serialize the workspace's managed coding-agent config to portable JSON.

Reads the local managed config (the one file :mod:`ucode.managed_config` owns, authored by
``ucode setup`` and refreshed by a launch), validates and serializes it through the same path
``ucode publish`` uses, and writes the external proto-JSON ``CodingAgentConfig`` — prefixed with the
source ``workspace`` and a ``spec_version`` envelope, the format ``ucode publish -f <path>`` consumes
— to stdout or a file.

Deliberately read-only and offline: no auth, no admin check, no discovery, no publish, and no write
except the explicitly requested ``--file`` output. That makes it role-agnostic (any developer can
run it) and keeps the machine-readable stream on stdout uncontaminated by Rich output.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from ucode.managed_config import load_managed_state, managed_state_workspace
from ucode.managed_setup import serialize_managed_config, validate_manifest
from ucode.state import load_state

_SERVER_OWNED_FIELDS = ("name",)

EXPORT_SPEC_VERSION = 1


def build_export_payload() -> dict:
    """Validate the local managed config and return the export payload.

    The payload is the source ``workspace`` URL and a ``spec_version`` (identifying the export
    format), followed by the external proto-JSON ``CodingAgentConfig`` with its server-owned resource
    ``name`` stripped. Reads only local state — no network, no auth. Raises RuntimeError with an
    actionable message when no config is authored locally or the config fails structural validation.
    """
    workspace = load_state().get("workspace") or managed_state_workspace()
    manifest = load_managed_state(workspace)
    if not manifest:
        raise RuntimeError(
            "No managed coding-agent config found locally. Run `ug setup` to author one, or run "
            "`ug` against a workspace that publishes one, then re-run `ug export`."
        )
    errors = validate_manifest(manifest, None)
    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(f"The managed config is not valid, so it was not exported:\n{detail}")
    config = serialize_managed_config(manifest)
    for field in _SERVER_OWNED_FIELDS:
        config.pop(field, None)
    return {"workspace": workspace, "spec_version": EXPORT_SPEC_VERSION, **config}


def export_command(file_path: str | None = None) -> None:
    """Serialize the managed config once and write it to ``file_path`` or stdout.

    The complete payload is built and serialized before the destination is touched, so a validation
    or serialization failure never creates or truncates it. With no ``file_path`` the JSON goes to
    stdout with exactly one trailing newline and nothing else; with ``file_path`` the identical bytes
    are written atomically and stdout stays empty. Raises RuntimeError on failure.
    """
    payload = build_export_payload()
    json_text = json.dumps(payload, indent=2) + "\n"
    if file_path is None:
        sys.stdout.write(json_text)
        return
    _write_atomic(Path(file_path).expanduser(), json_text)


def _write_atomic(destination: Path, json_text: str) -> None:
    """Write ``json_text`` to ``destination`` via a temp file in its directory, then ``os.replace``.

    The parent directory must already exist — a missing one is a clear error rather than a silent
    ``mkdir``. A partial write leaves the temp file behind, so it is removed on any failure and only
    the atomic replace makes the new content visible; an existing destination is replaced without a
    ``--force``.
    """
    directory = destination.parent
    if not directory.is_dir():
        raise RuntimeError(
            f"Cannot write {destination}: its parent directory does not exist. Create it first "
            "(ucode export does not create parent directories)."
        )
    try:
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".ucode-export-", suffix=".tmp")
    except OSError as exc:
        raise RuntimeError(f"Failed to write {destination}: {exc}") from exc
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json_text)
        os.replace(tmp_path, destination)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {destination}: {exc}") from exc
    finally:
        _cleanup(tmp_path)


def _cleanup(tmp_path: Path) -> None:
    """Best-effort removal of a leftover temp file (a no-op once os.replace has consumed it)."""
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
