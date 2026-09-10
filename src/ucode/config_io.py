"""File I/O, dry-run flag, backup/restore, deep-merge, dotenv parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import tomlkit
import tomlkit.exceptions
import yaml

from ucode.ui import console


def read_yaml_safe(path: Path) -> dict:
    # Same contract as read_json_safe: missing/unreadable/invalid → {}.
    try:
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def write_yaml_file(path: Path, payload: dict) -> None:
    content = yaml.safe_dump(payload, sort_keys=False)
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


class ToolSpec(TypedDict):
    binary: str
    package: str
    display: str
    config_path: Path
    backup_path: Path


APP_DIR = Path.home() / ".ucode"

_dry_run = False


def set_dry_run(value: bool) -> None:
    global _dry_run
    _dry_run = bool(value)


def is_dry_run() -> bool:
    return _dry_run


def ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory for {path}") from exc


def backup_existing_file(config_path: Path, backup_path: Path) -> bool:
    if _dry_run:
        return False
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            return True
        if not config_path.exists():
            return False
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    except OSError as exc:
        raise RuntimeError(f"Failed to back up config from {config_path}") from exc


def restore_file(config_path: Path, backup_path: Path, managed: bool) -> bool:
    try:
        if backup_path.exists():
            ensure_parent_dir(config_path)
            config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink()
            return True
        if managed and config_path.exists():
            config_path.unlink()
            return True
        return False
    except OSError as exc:
        raise RuntimeError(f"Failed to restore config at {config_path}") from exc


def write_text_file(path: Path, content: str) -> None:
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def write_json_file(path: Path, payload: dict) -> None:
    content = json.dumps(payload, indent=2) + "\n"
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def deep_merge_dict(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base; overlay wins for conflicting leaves.

    Mutates and returns base. Nested dicts are merged; everything else is replaced.
    """
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_merge_dict(base[key], val)
        else:
            base[key] = val
    return base


def prune_key_paths(doc: dict, key_paths: list[list[str]]) -> bool:
    """Surgically remove each ``key_path`` from a nested dict, dropping emptied parents.

    ``key_paths`` is a list of paths like ``[["env", "ANTHROPIC_BASE_URL"], ["apiKeyHelper"]]``.
    Only the exact leaves are removed; sibling keys the user set themselves stay untouched, and a
    parent dict left empty by the removal is dropped too. Returns True if anything changed.

    Used to undo ucode's writes to an agent's *native* config file (which it shares with the
    user's own settings), where restoring a backup would clobber edits made since ucode first ran.
    """
    changed = False
    for path in key_paths:
        if _prune_one(doc, list(path)):
            changed = True
    return changed


def _prune_one(node: object, path: list[str]) -> bool:
    if not path or not isinstance(node, dict):
        return False
    mapping = cast("dict[str, object]", node)
    key = path[0]
    if key not in mapping:
        return False
    if len(path) == 1:
        mapping.pop(key, None)
        return True
    child = mapping.get(key)
    removed = _prune_one(child, path[1:])
    if removed and isinstance(child, dict) and not child:
        mapping.pop(key, None)
    return removed


def read_json_safe(path: Path) -> dict:
    # `path.exists()` is inside the try: stat-ing a file under a root-locked dir (e.g. a
    # root-owned /etc/codex) raises PermissionError, which must read as "absent/unreadable → {}"
    # rather than crash a launch that only wanted to merge into it.
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_toml_safe(path: Path) -> tomlkit.TOMLDocument:
    # See read_json_safe: keep `path.exists()` inside the try so a PermissionError on a locked
    # parent directory is treated as an empty document rather than propagating.
    try:
        if not path.exists():
            return tomlkit.document()
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, tomlkit.exceptions.TOMLKitError):
        return tomlkit.document()


def write_toml_file(path: Path, doc: tomlkit.TOMLDocument) -> None:
    content = tomlkit.dumps(doc)
    if _dry_run:
        console.print(f"\n[bold]\\[dry run] {path}[/bold]\n{content}")
        return
    ensure_parent_dir(path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write config file: {path}") from exc


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE / KEY="VALUE" .env file, preserving insertion order.

    Comments and blank lines are dropped on round-trip. Lines that don't look
    like KEY=... are skipped. Leading whitespace (line indentation and spacing
    after the ``=``) is trimmed, but the exact characters up to the end of the
    line are preserved — including any trailing spaces — so a value such as
    ``token = abc123 `` keeps its trailing space.
    """
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw_line in text.splitlines():
        # Detect blank / comment lines on the fully-stripped form so trailing
        # spaces on value lines don't change which lines are skipped.
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        # Only strip *leading* whitespace from the line so the characters
        # between "=" and end-of-line (including trailing spaces) are preserved.
        key, _, val = raw_line.lstrip().partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.lstrip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        env[key] = val
    return env


def write_dotenv(path: Path, env: dict[str, str]) -> None:
    content = "".join(f'{key}="{val}"\n' for key, val in env.items())
    write_text_file(path, content)
