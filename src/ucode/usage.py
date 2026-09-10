"""Show the current user's coding-agent budget usage."""

from __future__ import annotations

from decimal import Decimal

from ucode.databricks import (
    apply_pat_environment,
    ensure_databricks_auth,
    get_databricks_token,
    resolve_current_budget_spend,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    format_meter,
    format_usd,
    label,
    muted,
    print_note,
    spinner,
    value,
)


def render_budget_summary(budget_spend: tuple[Decimal, Decimal]) -> str:
    """Render budget spend against the total budget with a spend meter."""
    spend, total_budget = budget_spend
    fraction = float(spend / total_budget) if total_budget > 0 else 0.0
    summary = f"{format_usd(spend)} of {format_usd(total_budget)} ({fraction:.0%})"
    return "\n".join(
        [
            f"{label('Budget spend:')} {value(summary)}",
            muted(format_meter(fraction)),
        ]
    )


def usage() -> int:
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    profile = state.get("profile")
    apply_pat_environment(state)
    ensure_databricks_auth(workspace, profile)
    with spinner("Retrieving Databricks access token..."):
        token = get_databricks_token(workspace, profile)

    with spinner("Checking budget spend..."):
        budget_spend, _ = resolve_current_budget_spend(workspace, token)

    if budget_spend is None:
        print_note(
            "Usage information is unavailable. Ask your workspace admin to configure "
            "Unity Gateway Budgets and add it to the workspace configuration."
        )
        return 0

    console.print(render_budget_summary(budget_spend))
    return 0
