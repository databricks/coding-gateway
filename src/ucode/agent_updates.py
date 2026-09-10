"""Update checks for npm-installed coding agent CLIs."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

_BASE_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_STABLE_VERSION_RE = re.compile(r"v?\d+\.\d+\.\d+$")


def _base_version(value: str) -> tuple[int, int, int] | None:
    """Return the leading (major, minor, patch) of a version, ignoring any
    prerelease/build suffix (e.g. `-nightly.20260515.g928a311fb`)."""
    match = _BASE_VERSION_RE.search(value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _is_stable(value: str) -> bool:
    """True for plain `X.Y.Z` releases (no prerelease/nightly/preview suffix)."""
    return bool(_STABLE_VERSION_RE.fullmatch(value.strip()))


def published_versions(package: str) -> list[str]:
    """Return every published version of an npm package, in npm's ascending
    order, or an empty list if the registry can't be reached."""
    if not shutil.which("npm"):
        return []
    try:
        result = subprocess.run(
            ["npm", "view", package, "versions", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        versions = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(versions, str):
        versions = [versions]
    if not isinstance(versions, list):
        return []
    return [v for v in versions if isinstance(v, str)]


def latest_version_below(package: str, ceiling: tuple[int, int, int]) -> str | None:
    """Return the newest published version whose base (major, minor, patch) is
    strictly below `ceiling`, preferring a stable release over a prerelease at
    the same base. Returns None when nothing qualifies or npm is unavailable."""
    candidates = [(v, _base_version(v)) for v in published_versions(package)]
    eligible = [(v, base) for v, base in candidates if base is not None and base < ceiling]
    if not eligible:
        return None
    max_base = max(base for _, base in eligible)
    at_max = [v for v, base in eligible if base == max_base]
    stable = [v for v in at_max if _is_stable(v)]
    pool = stable or at_max
    # npm returns versions in ascending order, so the last entry is newest.
    return pool[-1]


def available_npm_package_update(package: str) -> tuple[str, str] | None:
    """Return ``(current, latest)`` when a globally installed npm package has a
    newer release published, else None. Read-only — for ``ucode doctor``."""
    if not shutil.which("npm"):
        return None
    try:
        result = subprocess.run(
            ["npm", "outdated", "-g", "--json", package],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    # npm outdated exits 1 when it finds outdated packages.
    if result.returncode not in (0, 1) or not result.stdout.strip():
        return None
    try:
        outdated = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    package_update = outdated.get(package)
    if not isinstance(package_update, dict):
        return None
    current = package_update.get("current")
    latest = package_update.get("latest")
    if not isinstance(current, str) or not isinstance(latest, str):
        return None
    return current, latest
