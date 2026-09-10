"""Tests for config_io.py — file I/O helpers, dry-run flag, deep_merge_dict."""

from __future__ import annotations

import json

import pytest
import tomlkit

import ucode.config_io as config_io
from ucode.config_io import (
    backup_existing_file,
    deep_merge_dict,
    ensure_parent_dir,
    is_dry_run,
    parse_dotenv,
    prune_key_paths,
    read_json_safe,
    read_toml_safe,
    read_yaml_safe,
    restore_file,
    set_dry_run,
    write_dotenv,
    write_json_file,
    write_text_file,
    write_toml_file,
    write_yaml_file,
)


@pytest.fixture(autouse=True)
def reset_dry_run():
    """Ensure dry-run flag is reset after every test."""
    set_dry_run(False)
    yield
    set_dry_run(False)


# ---------------------------------------------------------------------------
# dry-run flag
# ---------------------------------------------------------------------------


class TestDryRunFlag:
    def test_default_is_false(self):
        assert is_dry_run() is False

    def test_set_true(self):
        set_dry_run(True)
        assert is_dry_run() is True

    def test_reset_to_false(self):
        set_dry_run(True)
        set_dry_run(False)
        assert is_dry_run() is False


# ---------------------------------------------------------------------------
# ensure_parent_dir
# ---------------------------------------------------------------------------


class TestEnsureParentDir:
    def test_creates_missing_parents(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        ensure_parent_dir(target)
        assert target.parent.exists()

    def test_existing_dir_is_ok(self, tmp_path):
        ensure_parent_dir(tmp_path / "file.txt")  # tmp_path already exists


# ---------------------------------------------------------------------------
# backup_existing_file / restore_file
# ---------------------------------------------------------------------------


class TestBackupAndRestore:
    def test_backup_copies_file(self, tmp_path, monkeypatch):
        config = tmp_path / "config.json"
        backup = tmp_path / "backup.json"
        config.write_text('{"key": "value"}', encoding="utf-8")
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)

        result = backup_existing_file(config, backup)

        assert result is True
        assert backup.exists()
        assert backup.read_text() == config.read_text()

    def test_backup_skipped_when_config_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        result = backup_existing_file(tmp_path / "missing.json", tmp_path / "backup.json")
        assert result is False

    def test_backup_idempotent_when_backup_exists(self, tmp_path, monkeypatch):
        config = tmp_path / "config.json"
        backup = tmp_path / "backup.json"
        config.write_text("new", encoding="utf-8")
        backup.write_text("old", encoding="utf-8")
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)

        backup_existing_file(config, backup)

        assert backup.read_text() == "old"  # original backup preserved

    def test_backup_skipped_in_dry_run(self, tmp_path, monkeypatch):
        config = tmp_path / "config.json"
        backup = tmp_path / "backup.json"
        config.write_text("data", encoding="utf-8")
        monkeypatch.setattr(config_io, "APP_DIR", tmp_path)
        set_dry_run(True)

        result = backup_existing_file(config, backup)

        assert result is False
        assert not backup.exists()

    def test_restore_from_backup(self, tmp_path):
        config = tmp_path / "config.json"
        backup = tmp_path / "backup.json"
        backup.write_text("original", encoding="utf-8")

        result = restore_file(config, backup, managed=True)

        assert result is True
        assert config.read_text() == "original"
        assert not backup.exists()

    def test_restore_deletes_managed_config_when_no_backup(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("managed", encoding="utf-8")

        result = restore_file(config, tmp_path / "no-backup.json", managed=True)

        assert result is True
        assert not config.exists()

    def test_restore_returns_false_when_nothing_to_do(self, tmp_path):
        result = restore_file(
            tmp_path / "missing.json",
            tmp_path / "also-missing.json",
            managed=False,
        )
        assert result is False


# ---------------------------------------------------------------------------
# write_text_file / write_json_file / write_toml_file / write_dotenv
# ---------------------------------------------------------------------------


class TestWriteHelpers:
    def test_write_text_file(self, tmp_path):
        p = tmp_path / "out.txt"
        write_text_file(p, "hello")
        assert p.read_text() == "hello"

    def test_write_text_file_dry_run_no_write(self, tmp_path):
        set_dry_run(True)
        p = tmp_path / "out.txt"
        write_text_file(p, "hello")
        assert not p.exists()

    def test_write_json_file(self, tmp_path):
        p = tmp_path / "out.json"
        write_json_file(p, {"a": 1})
        data = json.loads(p.read_text())
        assert data == {"a": 1}

    def test_write_json_file_dry_run_no_write(self, tmp_path):
        set_dry_run(True)
        p = tmp_path / "out.json"
        write_json_file(p, {"a": 1})
        assert not p.exists()

    def test_write_yaml_file(self, tmp_path):
        p = tmp_path / "out.yml"
        write_yaml_file(p, {"a": 1})
        assert "a: 1" in p.read_text()

    def test_write_yaml_file_dry_run_no_write(self, tmp_path):
        set_dry_run(True)
        p = tmp_path / "out.yml"
        write_yaml_file(p, {"a": 1})
        assert not p.exists()

    def test_write_toml_file(self, tmp_path):
        p = tmp_path / "out.toml"
        doc = tomlkit.document()
        doc.add("key", "val")
        write_toml_file(p, doc)
        assert "key" in p.read_text()

    def test_write_toml_file_dry_run_no_write(self, tmp_path):
        set_dry_run(True)
        p = tmp_path / "out.toml"
        doc = tomlkit.document()
        doc.add("key", "val")
        write_toml_file(p, doc)
        assert not p.exists()

    def test_write_dotenv(self, tmp_path):
        p = tmp_path / ".env"
        write_dotenv(p, {"KEY": "value", "OTHER": "123"})
        text = p.read_text()
        assert 'KEY="value"' in text
        assert 'OTHER="123"' in text


# ---------------------------------------------------------------------------
# read_json_safe / read_toml_safe
# ---------------------------------------------------------------------------


class TestReadHelpers:
    def test_read_json_safe_missing_file(self, tmp_path):
        result = read_json_safe(tmp_path / "missing.json")
        assert result == {}

    def test_read_json_safe_valid(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"x": 1}', encoding="utf-8")
        assert read_json_safe(p) == {"x": 1}

    def test_read_json_safe_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        assert read_json_safe(p) == {}

    def test_read_json_safe_non_dict(self, tmp_path):
        p = tmp_path / "arr.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_json_safe(p) == {}

    def test_read_yaml_safe_missing_file(self, tmp_path):
        result = read_yaml_safe(tmp_path / "missing.yml")
        assert result == {}

    def test_read_yaml_safe_valid(self, tmp_path):
        p = tmp_path / "data.yml"
        p.write_text("x: 1\n", encoding="utf-8")
        assert read_yaml_safe(p) == {"x": 1}

    def test_read_yaml_safe_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yml"
        p.write_text("not: [unclosed\n  bad indent: }", encoding="utf-8")
        assert read_yaml_safe(p) == {}

    def test_read_yaml_safe_non_dict(self, tmp_path):
        p = tmp_path / "arr.yml"
        p.write_text("- 1\n- 2\n", encoding="utf-8")
        assert read_yaml_safe(p) == {}

    def test_read_toml_safe_missing_file(self, tmp_path):
        doc = read_toml_safe(tmp_path / "missing.toml")
        assert dict(doc) == {}

    def test_read_toml_safe_valid(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text('[section]\nkey = "val"\n', encoding="utf-8")
        doc = read_toml_safe(p)
        assert doc["section"]["key"] == "val"

    def test_read_toml_safe_invalid(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("[[[ broken", encoding="utf-8")
        doc = read_toml_safe(p)
        assert dict(doc) == {}


# ---------------------------------------------------------------------------
# parse_dotenv
# ---------------------------------------------------------------------------


class TestParseDotenv:
    def test_simple_pairs(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("KEY=value\nOTHER=123\n", encoding="utf-8")
        assert parse_dotenv(p) == {"KEY": "value", "OTHER": "123"}

    def test_quoted_values(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("KEY=\"quoted\"\nSINGLE='also'\n", encoding="utf-8")
        assert parse_dotenv(p) == {"KEY": "quoted", "SINGLE": "also"}

    def test_skips_comments_and_blank_lines(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# comment\n\nKEY=value\n", encoding="utf-8")
        assert parse_dotenv(p) == {"KEY": "value"}

    def test_skips_lines_without_equals(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("NOEQUALSSIGN\nKEY=value\n", encoding="utf-8")
        assert parse_dotenv(p) == {"KEY": "value"}

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_dotenv(tmp_path / "missing.env") == {}

    def test_value_with_equals(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("URL=http://example.com?a=1\n", encoding="utf-8")
        assert parse_dotenv(p) == {"URL": "http://example.com?a=1"}

    def test_preserves_trailing_spaces_in_value(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("TOKEN=abc123 \n", encoding="utf-8")
        result = parse_dotenv(p)
        assert result == {"TOKEN": "abc123 "}
        assert result["TOKEN"].endswith(" ")

    def test_preserves_trailing_spaces_around_delimiter(self, tmp_path):
        p = tmp_path / ".env"
        # Leading whitespace after "=" is still trimmed, trailing spaces survive.
        p.write_text("TOKEN = abc123   \n", encoding="utf-8")
        assert parse_dotenv(p) == {"TOKEN": "abc123   "}


# ---------------------------------------------------------------------------
# deep_merge_dict
# ---------------------------------------------------------------------------


class TestDeepMergeDict:
    def test_flat_overlay_wins(self):
        base = {"a": 1, "b": 2}
        result = deep_merge_dict(base, {"b": 99, "c": 3})
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"env": {"X": "1", "Y": "2"}}
        overlay = {"env": {"Y": "new", "Z": "3"}}
        result = deep_merge_dict(base, overlay)
        assert result == {"env": {"X": "1", "Y": "new", "Z": "3"}}

    def test_overlay_replaces_non_dict_with_dict(self):
        base = {"key": "scalar"}
        overlay = {"key": {"nested": True}}
        result = deep_merge_dict(base, overlay)
        assert result == {"key": {"nested": True}}

    def test_overlay_replaces_dict_with_scalar(self):
        base = {"key": {"nested": True}}
        overlay = {"key": "scalar"}
        result = deep_merge_dict(base, overlay)
        assert result == {"key": "scalar"}

    def test_empty_overlay_leaves_base_unchanged(self):
        base = {"a": 1}
        result = deep_merge_dict(base, {})
        assert result == {"a": 1}

    def test_empty_base_returns_overlay(self):
        result = deep_merge_dict({}, {"a": 1})
        assert result == {"a": 1}

    def test_mutates_and_returns_base(self):
        base = {"a": 1}
        result = deep_merge_dict(base, {"b": 2})
        assert result is base


# ---------------------------------------------------------------------------
# prune_key_paths
# ---------------------------------------------------------------------------


class _StatDeniedPath:
    """Path stand-in whose existence check raises PermissionError, as stat-ing a file under a
    root-locked directory (e.g. a root-owned /etc/codex) does for a non-root process."""

    def exists(self):
        raise PermissionError(13, "Permission denied")

    def read_text(self, encoding="utf-8"):
        raise AssertionError("read_text must not be reached once exists() denies")


class TestReadSafePermissionDenied:
    def test_read_json_safe_returns_empty_on_permission_error(self):
        # Regression: a locked parent dir made path.exists() raise and crashed the whole launch.
        assert read_json_safe(_StatDeniedPath()) == {}

    def test_read_toml_safe_returns_empty_on_permission_error(self):
        result = read_toml_safe(_StatDeniedPath())
        assert dict(result) == {}


class TestPruneKeyPaths:
    def test_removes_leaf_and_leaves_siblings(self):
        doc = {"env": {"MANAGED": "1", "USER": "2"}, "apiKeyHelper": "x"}
        changed = prune_key_paths(doc, [["env", "MANAGED"], ["apiKeyHelper"]])
        assert changed is True
        assert doc == {"env": {"USER": "2"}}

    def test_drops_emptied_parent(self):
        doc = {"env": {"ONLY": "1"}}
        assert prune_key_paths(doc, [["env", "ONLY"]]) is True
        assert doc == {}

    def test_keeps_parent_with_remaining_siblings(self):
        doc = {"env": {"A": "1", "B": "2"}}
        assert prune_key_paths(doc, [["env", "A"]]) is True
        assert doc == {"env": {"B": "2"}}

    def test_missing_path_is_noop(self):
        doc = {"env": {"A": "1"}}
        assert prune_key_paths(doc, [["env", "NOPE"], ["absent"]]) is False
        assert doc == {"env": {"A": "1"}}

    def test_removes_top_level_key(self):
        doc = {"model_provider": "ucode", "model": "gpt-5", "other": True}
        assert prune_key_paths(doc, [["model_provider"], ["model"]]) is True
        assert doc == {"other": True}

    def test_partial_path_through_non_dict_is_noop(self):
        doc = {"model": "scalar"}
        assert prune_key_paths(doc, [["model", "nested"]]) is False
        assert doc == {"model": "scalar"}
