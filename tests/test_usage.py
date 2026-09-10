"""Tests for the budget-only usage command."""

from __future__ import annotations

from decimal import Decimal

import pytest

import ucode.usage as usage_mod
from ucode.usage import render_budget_summary, usage


class TestRenderBudgetSummary:
    def test_shows_original_spend_summary_and_meter(self):
        result = render_budget_summary((Decimal("110.70"), Decimal("5000")))

        assert "[bold]Budget spend:[/bold] [cyan]$110.70 of $5,000.00 (2%)[/cyan]" in result
        assert "[dim][█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░][/dim]" in result
        assert "Usage Budget" not in result
        assert "warehouse" not in result.lower()
        assert "token" not in result.lower()

    def test_shows_zero_total_budget(self):
        result = render_budget_summary((Decimal("5"), Decimal("0")))

        assert "$5.00 of $0.00 (0%)" in result
        assert "[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]" in result


class TestUsageCommand:
    @staticmethod
    def _stub_auth(monkeypatch, state):
        monkeypatch.setattr(usage_mod, "load_state", lambda: state)
        monkeypatch.setattr(usage_mod, "apply_pat_environment", lambda value: None)
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *args: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *args: "token")

    def test_fetches_and_prints_budget(self, monkeypatch):
        state = {"workspace": "https://workspace", "profile": "test-profile"}
        calls: list[tuple[str, object]] = []
        printed: list[str] = []

        monkeypatch.setattr(usage_mod, "load_state", lambda: state)
        monkeypatch.setattr(
            usage_mod, "apply_pat_environment", lambda value: calls.append(("pat", value))
        )
        monkeypatch.setattr(
            usage_mod,
            "ensure_databricks_auth",
            lambda workspace, profile: calls.append(("auth", (workspace, profile))),
        )
        monkeypatch.setattr(
            usage_mod,
            "get_databricks_token",
            lambda workspace, profile: calls.append(("token", (workspace, profile))) or "token",
        )
        monkeypatch.setattr(
            usage_mod,
            "resolve_current_budget_spend",
            lambda workspace, token: ((Decimal("12.34"), Decimal("100")), None),
        )
        monkeypatch.setattr(
            usage_mod,
            "console",
            type("Console", (), {"print": lambda _, text: printed.append(text)})(),
        )

        assert usage() == 0
        assert calls == [
            ("pat", state),
            ("auth", ("https://workspace", "test-profile")),
            ("token", ("https://workspace", "test-profile")),
        ]
        assert len(printed) == 1
        assert "$12.34 of $100.00 (12%)" in printed[0]

    def test_reports_unavailable_budget(self, monkeypatch):
        self._stub_auth(monkeypatch, {"workspace": "https://workspace"})
        notes: list[str] = []
        monkeypatch.setattr(
            usage_mod,
            "resolve_current_budget_spend",
            lambda workspace, token: (None, "disabled"),
        )
        monkeypatch.setattr(usage_mod, "print_note", notes.append)
        monkeypatch.setattr(
            usage_mod,
            "console",
            type("Console", (), {"print": lambda *_: pytest.fail("should not print a summary")})(),
        )

        assert usage() == 0
        assert notes == [
            "Usage information is unavailable. Ask your workspace admin to configure "
            "Unity Gateway Budgets and add it to the workspace configuration."
        ]

    def test_requires_a_configured_workspace(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "load_state", lambda: {})

        with pytest.raises(RuntimeError, match="Workspace is not configured"):
            usage()
