"""Tests for managed settings without real ``sudo`` or ``/etc`` writes."""

from __future__ import annotations

import json
import subprocess

import pytest

import ucode.config_io as config_io
from ucode import managed_files


@pytest.fixture(autouse=True)
def _reset_dry_run():
    config_io.set_dry_run(False)
    yield
    config_io.set_dry_run(False)


@pytest.fixture(autouse=True)
def _supported(monkeypatch):
    # Pin platform support on so tests are deterministic on any host.
    monkeypatch.setattr(managed_files, "managed_files_supported", lambda: True)
    monkeypatch.setattr(managed_files.sys.stdin, "isatty", lambda: True)


@pytest.fixture
def backup_dir(tmp_path, monkeypatch):
    path = tmp_path / "managed-backups"
    monkeypatch.setattr(managed_files, "MANAGED_BACKUP_DIR", path)
    monkeypatch.setattr(managed_files, "MANAGED_BACKUP_MANIFEST_PATH", path / "manifest.json")
    return path


class TestClearImmutableStatDenied:
    def test_stat_denied_path_returns_no_flags_without_raising(self, monkeypatch):
        # Regression: `_clear_immutable` ran an unguarded path.exists() inside the sudo write; under a
        # root-locked /etc/codex that raised PermissionError and aborted the write ("without root").
        class _StatDenied:
            def exists(self):
                raise PermissionError(13, "Permission denied")

        # Ensure no sudo subprocess is attempted if the guard ever regresses.
        monkeypatch.setattr(
            managed_files.subprocess, "run", lambda *a, **k: pytest.fail("should not shell out")
        )
        assert managed_files._clear_immutable(_StatDenied()) == ()


class TestImmutableFlags:
    def test_macos_flags_are_cleared_and_restored(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("{}", encoding="utf-8")
        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            stdout = "schg,uchg\n" if command[0] == "/usr/bin/stat" else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(managed_files, "current_os", lambda: managed_files.OS.MACOS)
        monkeypatch.setattr(managed_files.subprocess, "run", run)

        flags = managed_files._clear_immutable(path)
        managed_files._restore_immutable(path, flags)

        assert flags == ("schg", "uchg")
        assert ["/usr/bin/sudo", "chflags", "noschg,nouchg", str(path)] in calls
        assert ["/usr/bin/sudo", "chflags", "schg,uchg", str(path)] in calls

    def test_linux_flags_are_cleared_and_restored(self, tmp_path, monkeypatch):
        path = tmp_path / "managed.toml"
        path.write_text("", encoding="utf-8")
        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            stdout = "----ia------- managed.toml\n" if command[1] == "lsattr" else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(managed_files, "current_os", lambda: managed_files.OS.LINUX)
        monkeypatch.setattr(managed_files.subprocess, "run", run)

        flags = managed_files._clear_immutable(path)
        managed_files._restore_immutable(path, flags)

        assert flags == ("i", "a")
        assert ["/usr/bin/sudo", "chattr", "-ia", str(path)] in calls
        assert ["/usr/bin/sudo", "chattr", "+ia", str(path)] in calls


class TestManagedFileLifecycle:
    def test_dry_run_does_not_write_or_backup(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        config_io.set_dry_run(True)
        monkeypatch.setattr(
            managed_files, "_sudo_replace", lambda *args: pytest.fail("must not write")
        )

        result = managed_files.reconcile_managed_file(
            path,
            '{"ucode": true}\n',
            tool="claude",
            display="Claude Code",
            owned_paths=[["ucode"]],
        )

        assert result == "written"
        assert not backup_dir.exists()

    def test_unsupported_platform_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(managed_files, "managed_files_supported", lambda: False)
        monkeypatch.setattr(
            managed_files, "_sudo_replace", lambda *args: pytest.fail("must not write")
        )

        result = managed_files.reconcile_managed_file(
            tmp_path / "managed.json",
            '{"ucode": true}\n',
            tool="claude",
            display="Claude Code",
            owned_paths=[["ucode"]],
        )

        assert result == "unsupported"

    def test_permission_failure_is_actionable(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"

        def deny_write(path, text):
            raise PermissionError("no root")

        monkeypatch.setattr(managed_files, "_sudo_replace", deny_write)

        with pytest.raises(RuntimeError, match="could not update"):
            managed_files.reconcile_managed_file(
                path,
                '{"ucode": true}\n',
                tool="claude",
                display="Claude Code",
                owned_paths=[["ucode"]],
            )
        assert (backup_dir / "manifest.json").exists()

    def test_reconcile_refuses_symlink_target(self, tmp_path, backup_dir, monkeypatch):
        target = tmp_path / "real.json"
        target.write_text("{}", encoding="utf-8")
        path = tmp_path / "managed.json"
        path.symlink_to(target)
        monkeypatch.setattr(
            managed_files, "_sudo_replace", lambda *args: pytest.fail("must not write")
        )

        with pytest.raises(RuntimeError, match="Refusing to update"):
            managed_files.reconcile_managed_file(
                path,
                '{"ucode": true}\n',
                tool="claude",
                display="Claude Code",
                owned_paths=[["ucode"]],
            )

    def test_reconcile_backs_up_before_write(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text('{"enterprise": true}\n', encoding="utf-8")

        def replace(target, text):
            assert (backup_dir / "claude-managed-settings.backup.json").exists()
            target.write_text(text, encoding="utf-8")

        monkeypatch.setattr(managed_files, "_sudo_replace", replace)
        result = managed_files.reconcile_managed_file(
            path,
            '{"enterprise": true, "ucode": true}\n',
            tool="claude",
            display="Claude Code",
            owned_paths=[["ucode"]],
        )

        assert result == "written"
        assert (backup_dir / "claude-managed-settings.backup.json").read_text() == (
            '{"enterprise": true}\n'
        )
        manifest = json.loads((backup_dir / "manifest.json").read_text())
        assert manifest["files"]["claude"]["original_existed"] is True

    def test_batch_messages_name_all_agents_once(self, tmp_path, backup_dir, monkeypatch):
        notes: list[str] = []
        successes: list[str] = []

        monkeypatch.setattr(managed_files, "print_note", notes.append)
        monkeypatch.setattr(managed_files, "print_success", successes.append)
        monkeypatch.setattr(
            managed_files,
            "_sudo_replace",
            lambda target, text: target.write_text(text, encoding="utf-8"),
        )

        with managed_files.managed_write_batch(["Codex", "Claude Code"]):
            for tool in ("codex", "claude"):
                managed_files.reconcile_managed_file(
                    tmp_path / f"{tool}.json",
                    '{"ucode": true}\n',
                    tool=tool,
                    display=tool.title(),
                    owned_paths=[["ucode"]],
                )

        assert notes == ["Enter password to configure settings for Codex and Claude Code."]
        assert successes == ["Settings configured for Codex and Claude Code"]

    def test_unchanged_file_never_creates_backup(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text("same", encoding="utf-8")
        monkeypatch.setattr(
            managed_files, "_sudo_replace", lambda *args: pytest.fail("must not write")
        )

        result = managed_files.reconcile_managed_file(
            path,
            "same",
            tool="claude",
            display="Claude Code",
            owned_paths=[["env"]],
        )

        assert result == "unchanged"
        assert not backup_dir.exists()

    def test_verified_check_uses_fingerprint(self, tmp_path):
        path = tmp_path / "managed.json"
        path.write_text("current", encoding="utf-8")
        state: dict = {}
        managed_files.mark_managed_file_verified(state, "claude", path)

        assert managed_files.managed_file_is_verified(state, "claude", path) is True
        path.write_text("changed-content", encoding="utf-8")
        assert managed_files.managed_file_is_verified(state, "claude", path) is False

    def test_revert_restores_exact_original(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text('{"enterprise": true}\n', encoding="utf-8")
        monkeypatch.setattr(
            managed_files,
            "_sudo_replace",
            lambda target, text: target.write_text(text, encoding="utf-8"),
        )
        managed_files.reconcile_managed_file(
            path,
            '{"enterprise": true, "ucode": true}\n',
            tool="claude",
            display="Claude Code",
            owned_paths=[["ucode"]],
        )

        result = managed_files.revert_managed_file(
            "claude",
            display="Claude Code",
            parser=json.loads,
            dumper=lambda doc: json.dumps(doc) + "\n",
        )

        assert result == "restored"
        assert path.read_text() == '{"enterprise": true}\n'
        assert json.loads((backup_dir / "manifest.json").read_text())["files"] == {}

    def test_revert_removes_file_created_by_ucode(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        monkeypatch.setattr(
            managed_files,
            "_sudo_replace",
            lambda target, text: target.write_text(text, encoding="utf-8"),
        )
        monkeypatch.setattr(managed_files, "_sudo_remove", lambda target: target.unlink())
        managed_files.reconcile_managed_file(
            path,
            '{"ucode": true}\n',
            tool="claude",
            display="Claude Code",
            owned_paths=[["ucode"]],
        )

        result = managed_files.revert_managed_file(
            "claude",
            display="Claude Code",
            parser=json.loads,
            dumper=lambda doc: json.dumps(doc) + "\n",
        )

        assert result == "removed"
        assert not path.exists()

    def test_revert_preserves_external_changes(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text('{"enterprise": "original"}\n', encoding="utf-8")
        monkeypatch.setattr(
            managed_files,
            "_sudo_replace",
            lambda target, text: target.write_text(text, encoding="utf-8"),
        )
        managed_files.reconcile_managed_file(
            path,
            '{"enterprise": "original", "ucode": "gateway"}\n',
            tool="claude",
            display="Claude Code",
            owned_paths=[["ucode"]],
        )
        path.write_text(
            '{"enterprise": "new-policy", "ucode": "gateway", "new": true}\n',
            encoding="utf-8",
        )

        result = managed_files.revert_managed_file(
            "claude",
            display="Claude Code",
            parser=json.loads,
            dumper=lambda doc: json.dumps(doc, sort_keys=True) + "\n",
        )

        assert result == "ucode entries removed; external changes preserved"
        assert json.loads(path.read_text()) == {"enterprise": "new-policy", "new": True}

    def test_reconcile_retries_exact_mdm_restore_once(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text('{"enterprise": true}\n', encoding="utf-8")
        calls = 0

        def restore_original(target, text):
            nonlocal calls
            calls += 1
            target.write_text('{"enterprise": true}\n', encoding="utf-8")

        monkeypatch.setattr(managed_files, "_sudo_replace", restore_original)

        with pytest.raises(RuntimeError, match="immediately restored by device management"):
            managed_files.reconcile_managed_file(
                path,
                '{"enterprise": true, "ucode": true}\n',
                tool="claude",
                display="Claude Code",
                owned_paths=[["ucode"]],
            )

        assert calls == 2

    def test_reconcile_preserves_concurrent_policy_change(self, tmp_path, backup_dir, monkeypatch):
        path = tmp_path / "managed.json"
        path.write_text('{"enterprise": "old"}\n', encoding="utf-8")

        def external_update(target, text):
            target.write_text('{"enterprise": "new"}\n', encoding="utf-8")

        monkeypatch.setattr(managed_files, "_sudo_replace", external_update)

        with pytest.raises(RuntimeError, match="changed concurrently"):
            managed_files.reconcile_managed_file(
                path,
                '{"enterprise": "old", "ucode": true}\n',
                tool="claude",
                display="Claude Code",
                owned_paths=[["ucode"]],
            )

        assert json.loads(path.read_text()) == {"enterprise": "new"}


def test_managed_writes_disabled_without_tty(monkeypatch):
    monkeypatch.setattr(managed_files.sys.stdin, "isatty", lambda: False)

    assert managed_files.managed_writes_allowed() is False


def test_sudo_command_refuses_noninteractive_execution(monkeypatch):
    monkeypatch.setattr(managed_files, "managed_writes_allowed", lambda: False)

    with pytest.raises(RuntimeError, match="Refusing to invoke sudo"):
        managed_files._sudo_command("cp", "a", "b")
