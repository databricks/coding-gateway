"""Safely manage root-owned, highest-precedence agent settings files.

Interactive updates preserve unrelated policy, retain a private baseline for ``ucode revert``, and
verify the privileged atomic replacement. Non-interactive runs only check whether existing managed
values are compatible with ucode's local settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ucode.config_io import APP_DIR, is_dry_run
from ucode.ui import console, print_note, print_success, print_warning

# Absolute path so a stripped PATH (desktop/GUI launchers) still finds it.
_SUDO = "/usr/bin/sudo"
MANAGED_BACKUP_DIR = APP_DIR / "managed-backups"
MANAGED_BACKUP_MANIFEST_PATH = MANAGED_BACKUP_DIR / "manifest.json"
MANAGED_FINGERPRINT_VERSION = 1
_MISSING = object()
_managed_write_batch: tuple[str, ...] = ()
_managed_write_notice_shown = False

ManagedParser = Callable[[str], dict]
ManagedDumper = Callable[[dict], str]


class OS(Enum):
    """The host OS families this module distinguishes, off `sys.platform`."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    OTHER = "other"


def current_os() -> OS:
    """Map `sys.platform` onto :class:`OS` (lowercased, so a mixed-case value can't slip through)."""
    platform = sys.platform.lower()
    if platform.startswith("linux"):
        return OS.LINUX
    if platform == "darwin":
        return OS.MACOS
    if platform.startswith("win"):
        return OS.WINDOWS
    return OS.OTHER


def managed_files_supported() -> bool:
    """True on the platforms whose managed-settings write path is implemented (Linux, macOS).

    The write path needs `sudo` (`sudo cp`, `chattr`/`chflags`), which is Unix-only — so Windows and
    any other platform are unsupported.
    """
    return current_os() in (OS.LINUX, OS.MACOS)


def read_managed_file(path: Path) -> str | None:
    """Read a managed file strictly, returning ``None`` only when it is absent."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"Cannot read managed settings at {path}: {exc}") from exc


def managed_file_fingerprint(path: Path) -> dict[str, int | bool]:
    """Return metadata sufficient to detect normal MDM replacement or in-place edits."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False}
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect managed settings at {path}: {exc}") from exc
    return {
        "exists": True,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def managed_file_is_verified(
    state: dict, tool: str, path: Path, *, required_scope: str | None = None
) -> bool:
    """Fast cached-launch guard: one stat and no content parsing when unchanged."""
    records = state.get("managed_file_fingerprints")
    if not isinstance(records, dict):
        return False
    record = records.get(tool)
    if not isinstance(record, dict):
        return False
    if record.get("version") != MANAGED_FINGERPRINT_VERSION or record.get("path") != str(path):
        return False
    if required_scope is not None and record.get("scope") != required_scope:
        return False
    try:
        return record.get("fingerprint") == managed_file_fingerprint(path)
    except RuntimeError:
        return False


def mark_managed_file_verified(
    state: dict, tool: str, path: Path, *, scope: str = "managed"
) -> None:
    records = dict(state.get("managed_file_fingerprints") or {})
    records[tool] = {
        "version": MANAGED_FINGERPRINT_VERSION,
        "path": str(path),
        "scope": scope,
        "fingerprint": managed_file_fingerprint(path),
    }
    state["managed_file_fingerprints"] = records


def managed_writes_allowed() -> bool:
    """Managed writes are interactive setup work; scripts and CI use local settings."""
    return sys.stdin.isatty()


@contextmanager
def managed_write_batch(displays: list[str]) -> Iterator[None]:
    """Group setup messaging for agents configured in one command."""
    global _managed_write_batch, _managed_write_notice_shown

    previous_batch = _managed_write_batch
    previous_notice = _managed_write_notice_shown
    _managed_write_batch = tuple(dict.fromkeys(displays))
    _managed_write_notice_shown = False
    try:
        yield
        if _managed_write_notice_shown:
            print_success(f"Settings configured for {' and '.join(_managed_write_batch)}")
    finally:
        _managed_write_batch = previous_batch
        _managed_write_notice_shown = previous_notice


def _print_managed_write_permission(display: str) -> None:
    global _managed_write_notice_shown

    if not _managed_write_batch:
        print_note(f"Enter password to configure settings for {display}.")
        return
    if _managed_write_notice_shown:
        return

    displays = " and ".join(_managed_write_batch)
    print_note(f"Enter password to configure settings for {displays}.")
    _managed_write_notice_shown = True


def managed_file_conflicts(
    existing: dict, desired: dict, owned_paths: list[list[str]]
) -> list[str]:
    """Return managed leaves that would override ucode's local settings."""
    conflicts: list[str] = []
    for path in owned_paths:
        existing_value = _path_value(existing, path)
        if existing_value is _MISSING:
            continue
        if existing_value != _path_value(desired, path):
            conflicts.append(".".join(path))
    return conflicts


def managed_file_status(
    state: dict,
    tool: str,
    path: Path | None,
    *,
    parser: ManagedParser | None = None,
) -> tuple[str, str]:
    """Return a read-only status and backup label for ``ucode status``."""
    if path is None:
        return "unsupported", "none"
    try:
        fingerprint = managed_file_fingerprint(path)
    except RuntimeError:
        return "unreadable", _backup_label(tool)
    records = state.get("managed_file_fingerprints")
    record = records.get(tool) if isinstance(records, dict) else None
    if not fingerprint.get("exists"):
        status = "missing" if isinstance(record, dict) else "not configured"
    elif not isinstance(record, dict):
        status = "not configured"
    elif managed_file_is_verified(state, tool, path):
        scope = record.get("scope")
        if scope == "local-compatible":
            status = "compatible (local settings)"
        elif scope == "relay-compatible":
            status = "compatible (relay settings)"
        else:
            status = "current"
    else:
        status = "drifted"
    if parser is not None and status in {"current", "drifted", "not configured"}:
        try:
            text = read_managed_file(path)
        except RuntimeError:
            status = "unreadable"
        else:
            try:
                if text is not None:
                    parser(text)
            except RuntimeError:
                status = "invalid"
    return status, _backup_label(tool)


def reconcile_managed_file(
    path: Path,
    desired_text: str,
    *,
    tool: str,
    display: str,
    owned_paths: list[list[str]],
) -> str:
    """Back up, atomically write, and verify one OS-managed settings file.

    The first pre-ucode contents are retained until ``ucode revert``. Subsequent writes update only
    the last-applied snapshot used for drift-safe three-way restoration.
    """
    if not managed_files_supported():
        print_warning(
            f"{display}: OS-managed settings aren't supported on this platform; skipped {path}."
        )
        return "unsupported"
    if not managed_writes_allowed() and not is_dry_run():
        raise RuntimeError(
            f"Refusing to update {display} managed settings at {path} non-interactively. "
            "Run the command from an interactive terminal."
        )
    if path.is_symlink():
        raise RuntimeError(
            f"Refusing to update {display} managed settings through symlink {path}. "
            "Replace it with a regular file or contact your administrator."
        )
    current_text = read_managed_file(path)
    if current_text == desired_text:
        return "unchanged"
    if is_dry_run():
        console.print(f"\n[bold]\\[dry run] {path} (via sudo)[/bold]\n{desired_text}")
        return "written"

    created = current_text is None
    _ensure_backup(tool, path, current_text)
    _print_managed_write_permission(display)
    if read_managed_file(path) != current_text:
        raise RuntimeError(
            f"{display} managed settings changed while ucode was preparing the update. "
            "ucode preserved the newer file; run the command again."
        )
    for attempt in range(2):
        try:
            _sudo_replace(path, desired_text)
        except PermissionError as exc:
            raise RuntimeError(
                f"{display} cannot start because ucode could not update {path}: {exc}. "
                "Run the ucode command from an interactive terminal and approve the administrator "
                "prompt, or contact your administrator."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(_sudo_failure_message(path, display, exc)) from exc

        written_text = read_managed_file(path)
        if written_text == desired_text:
            break
        if attempt == 0 and written_text == current_text:
            print_warning(
                f"{display} managed settings were restored during the update; retrying once."
            )
            continue
        if written_text == current_text:
            raise RuntimeError(
                f"{display} managed settings at {path} were updated but immediately restored by "
                "device management. Contact your administrator."
            )
        raise RuntimeError(
            f"{display} managed settings changed concurrently at {path}. ucode will not overwrite "
            "the newer policy; run the command again or contact your administrator."
        )
    _record_last_applied(tool, path, desired_text, owned_paths)
    if not _managed_write_batch:
        print_success(f"Settings configured for {display}")
    return "created" if created else "written"


def revert_managed_file(
    tool: str,
    *,
    display: str,
    parser: ManagedParser,
    dumper: ManagedDumper,
) -> str:
    """Restore one managed file from its baseline while preserving later external edits."""
    manifest = _load_manifest()
    entry = _manifest_files(manifest).get(tool)
    if not isinstance(entry, dict):
        return "unchanged"
    path = Path(str(entry.get("path") or ""))
    if not path.is_absolute():
        raise RuntimeError(f"Invalid managed-settings backup path for {display}.")
    if path.is_symlink():
        raise RuntimeError(
            f"Refusing to restore {display} managed settings through symlink {path}."
        )
    current_text = read_managed_file(path)
    original_text = _original_text(entry)
    last_text = _snapshot_text(entry, "last_applied_file")

    if current_text == last_text:
        desired_text = original_text
    elif current_text is None or last_text is None:
        desired_text = current_text
    else:
        try:
            current_doc = parser(current_text)
            original_doc = parser(original_text) if original_text is not None else {}
            last_doc = parser(last_text)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot safely revert {display} managed settings at {path}: {exc}"
            ) from exc
        owned_paths = entry.get("owned_paths")
        paths = owned_paths if isinstance(owned_paths, list) else []
        reverted = _three_way_revert(current_doc, original_doc, last_doc, paths)
        desired_text = dumper(reverted)

    if desired_text != current_text:
        if not managed_writes_allowed():
            raise RuntimeError(
                f"Cannot restore {display} managed settings non-interactively. Run `ucode revert` "
                "from an interactive terminal."
            )
        try:
            if desired_text is None:
                _sudo_remove(path)
            else:
                _sudo_replace(path, desired_text)
        except (PermissionError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Could not restore {display} managed settings at {path}. The backup was retained "
                f"under {MANAGED_BACKUP_DIR}. Resolve the permission issue and run `ucode revert` "
                "again."
            ) from exc
        if read_managed_file(path) != desired_text:
            raise RuntimeError(
                f"Could not verify restored {display} managed settings at {path}. The backup was "
                f"retained under {MANAGED_BACKUP_DIR}."
            )

    _delete_backup(tool, manifest, entry)
    if original_text is None and desired_text is None:
        return "removed"
    if current_text != last_text:
        return "ucode entries removed; external changes preserved"
    return "restored"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_files(manifest: dict) -> dict:
    files = manifest.get("files")
    if not isinstance(files, dict):
        files = {}
        manifest["files"] = files
    return files


def _load_manifest() -> dict:
    try:
        if MANAGED_BACKUP_MANIFEST_PATH.is_symlink():
            raise RuntimeError(
                f"Refusing to read symlinked managed-settings backup manifest at "
                f"{MANAGED_BACKUP_MANIFEST_PATH}."
            )
        if not MANAGED_BACKUP_MANIFEST_PATH.exists():
            return {"version": 1, "files": {}}
        manifest = json.loads(MANAGED_BACKUP_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read managed-settings backup manifest at {MANAGED_BACKUP_MANIFEST_PATH}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise RuntimeError(
            f"Unsupported managed-settings backup manifest at {MANAGED_BACKUP_MANIFEST_PATH}."
        )
    return manifest


def _write_private_file(path: Path, text: str) -> None:
    if MANAGED_BACKUP_DIR.is_symlink():
        raise RuntimeError(f"Refusing to use symlinked backup directory {MANAGED_BACKUP_DIR}.")
    MANAGED_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(MANAGED_BACKUP_DIR, 0o700)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=MANAGED_BACKUP_DIR, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_manifest(manifest: dict) -> None:
    _write_private_file(MANAGED_BACKUP_MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")


def _backup_filename(tool: str, path: Path) -> str:
    suffix = path.suffix or ".txt"
    return f"{tool}-managed-settings.backup{suffix}"


def _last_applied_filename(tool: str, path: Path) -> str:
    suffix = path.suffix or ".txt"
    return f"{tool}-managed-settings.last-applied{suffix}"


def _ensure_backup(tool: str, path: Path, current_text: str | None) -> bool:
    manifest = _load_manifest()
    files = _manifest_files(manifest)
    existing = files.get(tool)
    if isinstance(existing, dict):
        if existing.get("path") != str(path):
            raise RuntimeError(
                f"The saved {tool} managed-settings backup targets {existing.get('path')}, not "
                f"{path}. Run `ucode revert` before configuring this path."
            )
        if existing.get("original_existed"):
            _original_text(existing)
        return False

    entry: dict[str, Any] = {
        "path": str(path),
        "original_existed": current_text is not None,
        "owned_paths": [],
    }
    if current_text is not None:
        backup_file = _backup_filename(tool, path)
        _write_private_file(MANAGED_BACKUP_DIR / backup_file, current_text)
        entry["backup_file"] = backup_file
        entry["original_sha256"] = _sha256(current_text)
    files[tool] = entry
    _write_manifest(manifest)
    return True


def _record_last_applied(
    tool: str, path: Path, desired_text: str, owned_paths: list[list[str]]
) -> None:
    manifest = _load_manifest()
    entry = _manifest_files(manifest).get(tool)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Missing managed-settings backup metadata for {tool}.")
    last_file = _last_applied_filename(tool, path)
    _write_private_file(MANAGED_BACKUP_DIR / last_file, desired_text)
    entry["last_applied_file"] = last_file
    entry["last_applied_sha256"] = _sha256(desired_text)
    known_paths = entry.get("owned_paths") if isinstance(entry.get("owned_paths"), list) else []
    for owned_path in owned_paths:
        if owned_path not in known_paths:
            known_paths.append(list(owned_path))
    entry["owned_paths"] = known_paths
    _write_manifest(manifest)


def _snapshot_text(entry: dict, key: str) -> str | None:
    filename = entry.get(key)
    if not isinstance(filename, str):
        return None
    path = _snapshot_path(filename)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read managed-settings snapshot at {path}: {exc}") from exc
    hash_key = "original_sha256" if key == "backup_file" else "last_applied_sha256"
    expected_hash = entry.get(hash_key)
    if not isinstance(expected_hash, str) or _sha256(text) != expected_hash:
        raise RuntimeError(f"Managed-settings snapshot failed integrity verification at {path}.")
    return text


def _original_text(entry: dict) -> str | None:
    if not entry.get("original_existed"):
        return None
    return _snapshot_text(entry, "backup_file")


def _backup_label(tool: str) -> str:
    try:
        entry = _manifest_files(_load_manifest()).get(tool)
    except RuntimeError:
        return "invalid"
    return "available" if isinstance(entry, dict) else "none"


def _delete_backup(tool: str, manifest: dict, entry: dict) -> None:
    for key in ("backup_file", "last_applied_file"):
        filename = entry.get(key)
        if isinstance(filename, str):
            try:
                _snapshot_path(filename).unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError(f"Could not remove managed-settings backup: {exc}") from exc
    _manifest_files(manifest).pop(tool, None)
    _write_manifest(manifest)


def _snapshot_path(filename: str) -> Path:
    if Path(filename).name != filename:
        raise RuntimeError(f"Invalid managed-settings snapshot filename: {filename}")
    return MANAGED_BACKUP_DIR / filename


def _path_value(doc: dict, path: list[str]) -> object:
    node: object = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = cast(dict, node)[key]
    return node


def _set_path_value(doc: dict, path: list[str], value: object) -> None:
    node = doc
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = deepcopy(value)


def _delete_path_value(doc: dict, path: list[str]) -> None:
    parents: list[tuple[dict, str]] = []
    node = doc
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            return
        parents.append((node, key))
        node = child
    node.pop(path[-1], None)
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)


def _owned_path(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value or not all(isinstance(part, str) for part in value):
        return None
    return cast(list[str], value)


def _three_way_revert(current: dict, original: dict, last: dict, paths: list) -> dict:
    reverted = deepcopy(current)
    for raw_path in paths:
        path = _owned_path(raw_path)
        if path is None:
            continue
        current_value = _path_value(reverted, path)
        original_value = _path_value(original, path)
        last_value = _path_value(last, path)
        if current_value == last_value:
            if original_value is _MISSING:
                _delete_path_value(reverted, path)
            else:
                _set_path_value(reverted, path, original_value)
            continue
        if not isinstance(current_value, list) or not isinstance(last_value, list):
            continue
        original_list = original_value if isinstance(original_value, list) else []
        additions = [item for item in last_value if item not in original_list]
        cleaned = [item for item in current_value if item not in additions]
        if cleaned:
            _set_path_value(reverted, path, cleaned)
        else:
            _delete_path_value(reverted, path)
    return reverted


def _sudo_remove(path: Path) -> None:
    original_flags = _clear_immutable(path)
    try:
        subprocess.run(
            _sudo_command("rm", "-f", str(path)), capture_output=True, text=True, check=True
        )
    finally:
        if original_flags and path.exists():
            _restore_immutable(path, original_flags)


def _sudo_replace(path: Path, desired_text: str) -> None:
    """Atomically replace ``path`` via sudo while preserving metadata and file flags."""
    if not managed_writes_allowed():
        raise RuntimeError("Refusing to invoke sudo for managed settings non-interactively.")
    try:
        parent_existed = path.parent.exists()
    except OSError:
        parent_existed = True
    subprocess.run(_sudo_command("mkdir", "-p", str(path.parent)), check=True)
    if not parent_existed:
        subprocess.run(_sudo_command("chown", "0:0", str(path.parent)), check=True)
        subprocess.run(_sudo_command("chmod", "755", str(path.parent)), check=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=path.suffix or ".tmp", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(desired_text)
        tmp_path = tmp.name
    staging_path: str | None = None
    original_flags: tuple[str, ...] = ()
    try:
        result = subprocess.run(
            _sudo_command("mktemp", str(path.parent / f".{path.name}.ucode.XXXXXX")),
            capture_output=True,
            text=True,
            check=True,
        )
        staging_path = result.stdout.strip()
        if not staging_path or Path(staging_path).parent != path.parent:
            raise RuntimeError(f"sudo mktemp returned an invalid staging path for {path}.")

        path_exists = path.exists()
        if path_exists:
            original_flags = _clear_immutable(path)
            preserve_args = ["-p"] if current_os() is OS.MACOS else ["--preserve=all"]
            subprocess.run(
                _sudo_command("cp", *preserve_args, str(path), staging_path),
                capture_output=True,
                text=True,
                check=True,
            )
            _clear_immutable(Path(staging_path))

        subprocess.run(
            _sudo_command("cp", tmp_path, staging_path),
            capture_output=True,
            text=True,
            check=True,
        )
        if not path_exists:
            subprocess.run(_sudo_command("chown", "0:0", staging_path), check=True)
            subprocess.run(_sudo_command("chmod", "644", staging_path), check=True)

        subprocess.run(
            _sudo_command("mv", "-f", staging_path, str(path)),
            capture_output=True,
            text=True,
            check=True,
        )
        staging_path = None
        if original_flags:
            _restore_immutable(path, original_flags)
            original_flags = ()
    finally:
        if original_flags and path.exists():
            _restore_immutable(path, original_flags)
        os.unlink(tmp_path)
        if staging_path:
            subprocess.run(
                _sudo_command("rm", "-f", staging_path),
                capture_output=True,
                text=True,
                check=False,
            )


def _clear_immutable(path: Path) -> tuple[str, ...]:
    """Clear immutable/append-only flags and return the flags that must be restored."""
    try:
        if not path.exists():
            return ()
    except OSError:
        return ()
    if current_os() is OS.MACOS:
        result = subprocess.run(
            ["/usr/bin/stat", "-f", "%Sf", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return ()
        supported = {"schg", "uchg", "sappnd", "uappnd"}
        flags = tuple(flag for flag in result.stdout.strip().split(",") if flag in supported)
        if flags:
            subprocess.run(
                _sudo_command("chflags", ",".join(f"no{flag}" for flag in flags), str(path)),
                capture_output=True,
                text=True,
                check=True,
            )
        return flags
    result = subprocess.run(
        _sudo_command("lsattr", "-d", str(path)), capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ()
    attributes = result.stdout.split()[0]
    flags = tuple(flag for flag in ("i", "a") if flag in attributes)
    if flags:
        subprocess.run(
            _sudo_command("chattr", f"-{''.join(flags)}", str(path)),
            capture_output=True,
            text=True,
            check=True,
        )
    return flags


def _restore_immutable(path: Path, flags: tuple[str, ...]) -> None:
    """Restore immutable/append-only flags after replacing a managed file."""
    if not flags:
        return
    if current_os() is OS.MACOS:
        command = _sudo_command("chflags", ",".join(flags), str(path))
    else:
        command = _sudo_command("chattr", f"+{''.join(flags)}", str(path))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print_warning(f"Could not restore the immutable flag on {path}.")


def _sudo_failure_message(path: Path, display: str, exc: subprocess.CalledProcessError) -> str:
    stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
    cmd = exc.cmd or []
    cp_failed = "cp" in cmd[1:3]
    if cp_failed and "Operation not permitted" in stderr:
        return (
            f"{display} managed settings at {path} are immutable and could not be updated. "
            "Contact your administrator."
        )
    return (
        f"{display} cannot start because ucode could not update {path}: {stderr or exc}. "
        "Run the ucode command from an interactive terminal and approve the administrator prompt, "
        "or contact your administrator."
    )


def _sudo_command(*args: str) -> list[str]:
    """Build a sudo command only for an explicitly interactive managed-file operation."""
    if not managed_writes_allowed():
        raise RuntimeError("Refusing to invoke sudo for managed settings non-interactively.")
    return [_SUDO, *args]
