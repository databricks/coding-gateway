"""Tests for npm-backed agent update checks."""

from __future__ import annotations

import json
import subprocess

from ucode.agent_updates import latest_version_below, published_versions

_GEMINI_VERSIONS = [
    "0.43.0",
    "0.44.0-nightly.20260515.g928a311fb",
    "0.44.0",
    "0.44.1",
    "0.45.0-nightly.20260602.g665228e98",
    "0.45.0-preview.0",
]


def _fake_published(monkeypatch, versions):
    monkeypatch.setattr("ucode.agent_updates.shutil.which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        "ucode.agent_updates.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(versions), stderr=""
        ),
    )


class TestPublishedVersions:
    def test_returns_empty_when_npm_missing(self, monkeypatch):
        monkeypatch.setattr("ucode.agent_updates.shutil.which", lambda _: None)
        assert published_versions("@google/gemini-cli") == []

    def test_parses_version_list(self, monkeypatch):
        _fake_published(monkeypatch, _GEMINI_VERSIONS)
        assert published_versions("@google/gemini-cli") == _GEMINI_VERSIONS

    def test_wraps_single_string_response(self, monkeypatch):
        _fake_published(monkeypatch, "0.44.1")
        assert published_versions("@google/gemini-cli") == ["0.44.1"]


class TestLatestVersionBelow:
    def test_picks_newest_stable_below_ceiling(self, monkeypatch):
        _fake_published(monkeypatch, _GEMINI_VERSIONS)
        # 0.44.1 is the newest base < 0.45.0, and it is stable.
        assert latest_version_below("@google/gemini-cli", (0, 45, 0)) == "0.44.1"

    def test_excludes_versions_at_or_above_ceiling(self, monkeypatch):
        _fake_published(monkeypatch, _GEMINI_VERSIONS)
        result = latest_version_below("@google/gemini-cli", (0, 45, 0))
        assert result is not None
        assert not result.startswith("0.45")

    def test_prefers_stable_over_prerelease_at_same_base(self, monkeypatch):
        _fake_published(
            monkeypatch,
            ["0.44.0-nightly.20260515.g928a311fb", "0.44.0", "0.44.0-preview.0"],
        )
        assert latest_version_below("@google/gemini-cli", (0, 45, 0)) == "0.44.0"

    def test_falls_back_to_prerelease_when_no_stable(self, monkeypatch):
        _fake_published(monkeypatch, ["0.44.0-nightly.20260515.g928a311fb"])
        assert (
            latest_version_below("@google/gemini-cli", (0, 45, 0))
            == "0.44.0-nightly.20260515.g928a311fb"
        )

    def test_returns_none_when_nothing_qualifies(self, monkeypatch):
        _fake_published(monkeypatch, ["0.45.0", "0.46.0"])
        assert latest_version_below("@google/gemini-cli", (0, 45, 0)) is None
