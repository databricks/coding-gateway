#!/usr/bin/env python3
"""CLI entry point for ``ug``."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from typing import Annotated, Any

import typer
from rich.panel import Panel
from typer.core import TyperCommand

from ucode import custom_oauth
from ucode.agents import (
    TOOL_SPECS,
    LaunchOptions,
    check_gateway_endpoint,
    configure_selected_tools,
    configure_single_tool,
    configure_tool,
    ensure_bootstrap_dependencies,
    ensure_provider_state,
    explicit_model_arg_value,
    install_databricks_ai_tools_for_agents,
    install_tool_binary,
    normalize_tool,
    provider_permission_error,
    resolve_gemini_provider_model,
    resolve_launch_model,
    resolve_provider_models,
    validate_all_tools,
    validate_tool,
)
from ucode.agents import claude as claude_agent
from ucode.agents import codex as codex_agent
from ucode.agents import (
    launch as launch_agent,
)
from ucode.agents.args import has_explicit_model_arg
from ucode.agents.codex import revert_legacy_shared_config
from ucode.agents.pi import PI_SETTINGS_BACKUP_PATH, PI_SETTINGS_PATH
from ucode.config_io import is_dry_run, restore_file, set_dry_run
from ucode.databricks import (
    apply_pat_environment,
    build_shared_base_urls,
    discover_claude_models,
    discover_codex_models,
    discover_gemini_models,
    discover_model_services,
    ensure_databricks_auth,
    ensure_pat_bearer,
    find_profile_name_for_host,
    get_databricks_profiles,
    get_databricks_token,
    install_databricks_cli,
    is_model_provider_feature_unavailable,
    is_workspace_admin,
    list_profile_entries,
    list_tool_provider_services,
    normalize_workspace_url,
    probe_unity_gateway_capabilities,
    resolve_pat_token,
    resolve_provider_launch_model,
    run_databricks_login,
)
from ucode.managed_budget import (
    budget_usage_percent,
    recommendation_line,
    render_budget_panel,
)
from ucode.managed_config import (
    ManagedConfigResult,
    get_model_recommendation,
    load_managed_state,
    refresh_managed_config,
)
from ucode.managed_resolve import (
    managed_claude_family_models,
    managed_default_model,
    managed_enabled_tools,
    managed_launch_model,
    managed_provider_family_models,
    managed_provider_service,
    managed_supplies_models,
    managed_unservable_models,
    recommended_agent,
    resolve_state,
)
from ucode.managed_wizard import (
    publish_command,
    setup_budget_policy_command,
    setup_command,
    setup_help_command,
    setup_mcp_command,
    setup_skills_command,
    show_command,
)
from ucode.mcp import (
    MCP_CLIENTS,
    SKILLS_MCP_KIND,
    add_mcp_command,
    add_skills_command,
    apply_managed_mcp_servers,
    configure_mcp_command,
    configure_skills_mcp_command,
    purge_cross_workspace_mcp_residue,
    remove_mcp_command,
    revert_mcp_configs,
)
from ucode.skills_download import (
    configure_skills_download_command,
    download_managed_skills_on_launch,
)
from ucode.smart_routing import v2 as smart_routing_v2
from ucode.smart_routing.claude_hooks import FIRST_PROMPT_SOCKET_ENV, ROUTE_FIRST_PROMPT_EVENT
from ucode.state import (
    STATE_PATH,
    clear_state,
    get_provider_service,
    load_full_state,
    load_state,
    save_state,
    set_current_workspace,
    set_provider_service,
)
from ucode.tracing import configure_tracing_command
from ucode.ui import (
    console,
    heading,
    print_err,
    print_heading,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    prompt_for_selection,
    prompt_for_tools,
    prompt_for_workspace,
    prompt_yes_no,
    set_verbosity,
    spinner,
    status_badge,
)
from ucode.usage import usage as usage_report

CustomOAuthConfig = custom_oauth.CustomOAuthConfig

_DISCOVERY_CONSUMERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "opencode", "copilot", "pi"),
    "codex": ("codex", "copilot", "pi"),
    "gemini": ("gemini", "opencode", "pi"),
    "oss": ("opencode",),
}


def _policy_summary_lines(managed: dict) -> list[str]:
    """Rich-markup lines describing the admin's tiered spend policy, or empty when it sets none."""
    policy = managed.get("budget_policy")
    if not isinstance(policy, dict):
        return []
    name = str(policy.get("display_name") or "coding-agents-default")
    lines = [f"[bold]Policy:[/bold] [cyan]{name}[/cyan]"]
    tiers = policy.get("tiers")
    for tier in tiers if isinstance(tiers, list) else []:
        if not isinstance(tier, dict):
            continue
        pct_raw = tier.get("spending_percentage")
        pct = (
            f"{float(pct_raw) * 100:g}%"
            if isinstance(pct_raw, int | float) and not isinstance(pct_raw, bool)
            else "?"
        )
        # A tier whose agent enum this build doesn't know is dropped during normalization, so it
        # arrives unset rather than as a tool name TOOL_SPECS could resolve.
        agent = tier.get("default_agent")
        agent_display = TOOL_SPECS[agent]["display"] if agent in TOOL_SPECS else "?"
        model = str(tier.get("default_model") or "?")
        lines.append(
            f"  [dim]·[/dim] [bold]at {pct}[/bold] → {agent_display} · [magenta]{model}[/magenta]"
        )
    return lines


def _print_managed_summary(
    managed: dict, state: dict, tool: str | None, *, abridged: bool = False
) -> None:
    """Show which of the admin's settings are in force.

    With ``tool`` set (launch path) the per-agent Agent/Provider/Model lines are included;
    with ``tool=None`` (e.g. ``ug configure`` under a managed config) they are skipped
    since no single agent has been chosen yet.

    ``abridged`` prints only what changes launch-to-launch — the agent and model this run will use,
    and the policy in force — with a pointer to ``ug status`` for the rest. Bare ``ucode`` runs
    every session, so re-enumerating the workspace's full MCP/skills/tier list each time is noise;
    the full box stays for ``status`` and ``configure``, where the reader asked to see it.
    """
    if abridged:
        _print_managed_summary_abridged(managed, state, tool)
        return
    lines = [f"[bold]Workspace:[/bold] [cyan]{state.get('workspace', '?')}[/cyan]"]
    if tool is not None:
        lines.append(f"[bold]Agent:[/bold] [green]{TOOL_SPECS[tool]['display']}[/green]")
    enabled = [t for t in (managed.get("enabled_agents") or {}) if t in TOOL_SPECS]
    if enabled:
        lines.append(
            f"[bold]Enabled agents:[/bold] {', '.join(TOOL_SPECS[t]['display'] for t in enabled)}"
        )
    if tool is not None:
        provider = managed_provider_service(managed, tool)
        if provider:
            lines.append(f"[bold]Provider:[/bold] [magenta]{provider}[/magenta]")
        model = managed_default_model(managed, tool)
        if model:
            lines.append(f"[bold]Model:[/bold] [magenta]{model}[/magenta]")
    # Always listed, including when empty: "none configured" tells a developer their admin set none,
    # which a missing row leaves ambiguous. Shown as the admin configured them — registering them
    # locally is a separate change, hence "pending".
    mcp_names = [
        str(server.get("name"))
        for server in (managed.get("mcp_servers") or [])
        if isinstance(server, dict) and server.get("name")
    ]
    if mcp_names:
        lines.append(f"[bold]MCPs:[/bold] {', '.join(mcp_names)} [dim](pending)[/dim]")
    else:
        lines.append("[bold]MCPs:[/bold] [dim]none configured[/dim]")
    skill_names = [str(name) for name in ((managed.get("skills") or {}).get("names") or []) if name]
    if skill_names:
        lines.append(f"[bold]Skills:[/bold] {', '.join(skill_names)} [dim](pending)[/dim]")
    else:
        lines.append("[bold]Skills:[/bold] [dim]none configured[/dim]")
    lines.extend(_policy_summary_lines(managed))
    console.print(
        Panel("\n".join(lines), title="Workspace-managed config", style="green", expand=False)
    )


def _print_managed_summary_abridged(managed: dict, state: dict, tool: str | None) -> None:
    """One-line launch banner: which agent (and model) this managed run is launching.

    Bare ``ucode`` runs every session, so the full box's MCP/skills/policy enumeration is noise
    each time; ``ug status`` still shows all of it. See ``_print_managed_summary``'s ``abridged``
    note. ``tool`` is always set on the launch path, but is guarded for callers that pass None."""
    if tool is None:
        print_note("Using managed config.")
        return
    agent = TOOL_SPECS[tool]["display"]
    model = managed_default_model(managed, tool)
    model_suffix = f" with [magenta]{model}[/magenta]" if model else ""
    # "as the default agent" only when this really is the config's default: a budget tier can
    # override the default and launch a different agent, and the tier note in `_launch_tool` already
    # explains that case — so claiming "default" here would contradict it.
    role = " as the default agent" if tool == managed.get("default_agent") else ""
    console.print(
        f"[dim]•[/dim] Using managed config — launching [green]{agent}[/green]{role}{model_suffix}"
    )


def _confirm_managed_config_applied(managed: dict, workspace: str) -> None:
    print_success("A managed config is published for your workspace — you're all set.")
    _print_managed_summary(managed, {"workspace": workspace}, tool=None)
    print_note("Run `ug` to launch with your managed settings.")


def _print_discovery_diagnostics(state: dict) -> None:
    """Surface per-source reasons after a failed discovery so the user knows
    which API call returned what — instead of the generic 'no agents' line."""
    reasons = state.get("_discovery_reasons") or {}
    if not reasons:
        return
    labels = {
        "claude": "Claude models",
        "codex": "Codex models",
        "gemini": "Gemini models",
        "oss": "OSS models",
    }
    for source, reason in reasons.items():
        consumers = ", ".join(_DISCOVERY_CONSUMERS.get(source, ()))
        label = labels.get(source, source)
        if reason:
            print_note(f"{label} (needed for: {consumers}): {reason}")
        else:
            print_note(f"{label} (needed for: {consumers}): no models returned")
    print_note("Re-run with `UCODE_DEBUG=1` to log raw discovery responses to ~/.ucode/debug.log.")


def _prompt_for_configuration(tool: str | None = None) -> tuple[str, str | None]:
    if tool is None:
        desc = "Configure your Databricks workspace"
    else:
        desc = f"Configure {TOOL_SPECS[tool]['display']} to use your Databricks endpoint."
    with spinner("Loading Databricks workspaces and profiles..."):
        profiles = get_databricks_profiles()
    return prompt_for_workspace(desc, profiles)


def _custom_oauth_config(
    client_id: str | None,
    redirect_url: str | None,
    scopes: str | None,
) -> CustomOAuthConfig | None:
    if client_id is None and redirect_url is None and scopes is None:
        return None
    if client_id is None:
        raise RuntimeError("--redirect-url and --scopes require --client-id.")
    if scopes is None:
        raise RuntimeError("--scopes is required with --client-id.")

    return custom_oauth.create_custom_oauth_config(
        client_id,
        scopes.split(","),
        redirect_url or custom_oauth.DEFAULT_REDIRECT_URL,
    )


def _parse_agents_option(agents: str) -> list[str]:
    tools: list[str] = []
    for raw_tool in agents.split(","):
        raw_tool = raw_tool.strip()
        if not raw_tool:
            continue
        tool = normalize_tool(raw_tool)
        if tool not in tools:
            tools.append(tool)
    if not tools:
        raise RuntimeError(
            "No agents provided for --agents. Use a comma-separated list like `--agents claude,codex`."
        )
    return tools


def _parse_skill_locations(location: str | None) -> list[str]:
    """Parse a comma-separated `--location` into `<catalog>.<schema>` refs,
    dropping duplicates while preserving order. `None`/empty yields `[]` (the
    schema-less, utility-tools-only connection)."""
    locations: list[str] = []
    for raw in (location or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(".")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise RuntimeError(f"--location entries must be `<catalog>.<schema>`, got `{raw}`.")
        if raw not in locations:
            locations.append(raw)
    return locations


def _parse_workspaces_option(workspaces: str) -> list[tuple[str, str | None]]:
    """Parse `--workspaces` into [(url, profile_name | None), ...].

    `--workspaces` supplies bare URLs; the matching profile (if any) is
    resolved later via `find_profile_name_for_host`.
    """
    workspace_entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for raw_workspace in workspaces.split(","):
        raw_workspace = raw_workspace.strip()
        if not raw_workspace:
            continue
        try:
            workspace = normalize_workspace_url(raw_workspace)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if workspace not in seen:
            seen.add(workspace)
            workspace_entries.append((workspace, None))
    if not workspace_entries:
        raise RuntimeError(
            "No workspaces provided for --workspaces. Use a comma-separated list like "
            "`--workspaces https://workspace.databricks.com`."
        )
    return workspace_entries


def _parse_profiles_option(profiles: str) -> list[tuple[str, str | None]]:
    """Parse `--profiles` into [(url, profile_name), ...].

    Each name must be an existing Databricks CLI profile; its host supplies
    the workspace URL. Auth behaves the same as `--workspaces`: OAuth login is
    forced unless `--use-pat` is also passed."""
    available = {str(p.get("name")): p for p in list_profile_entries() if p.get("name")}
    workspace_entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for raw_name in profiles.split(","):
        name = raw_name.strip()
        if not name:
            continue
        entry = available.get(name)
        if entry is None:
            known = ", ".join(sorted(available)) or "none"
            raise RuntimeError(
                f"Databricks CLI profile '{name}' was not found (available: {known}). "
                "Check `databricks auth profiles` or add the profile to ~/.databrickscfg."
            )
        host = str(entry.get("host") or "").strip()
        if not host:
            raise RuntimeError(
                f"Databricks CLI profile '{name}' has no host configured in ~/.databrickscfg."
            )
        try:
            workspace = normalize_workspace_url(host)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if workspace not in seen:
            seen.add(workspace)
            workspace_entries.append((workspace, name))
    if not workspace_entries:
        raise RuntimeError(
            "No profiles provided for --profiles. Use a comma-separated list like "
            "`--profiles DEFAULT`."
        )
    return workspace_entries


def configure_shared_state(
    workspace: str,
    profile: str | None = None,
    tools: list[str] | None = None,
    force_login: bool = False,
    use_pat: bool | None = None,
    skip_model_discovery: bool = False,
    skip_preflight: bool = False,
    fable_enabled: bool | None = None,
    databricks_ai_tools_enabled: bool | None = None,
    custom_oauth: CustomOAuthConfig | None = None,
    clear_custom_oauth: bool = False,
) -> dict:
    """Log into Databricks, verify AI Gateway, fetch model lists, persist state.

    If tools is provided, only fetch models for those tools. Otherwise fetch all.
    If force_login is True, always run databricks auth login (used by explicit configure).
    If use_pat is True (explicit `configure --profiles <name> --use-pat`), the
    profile's personal access token from ~/.databrickscfg is used instead of
    OAuth and no interactive login ever runs. ``None`` means "inherit": a
    launch re-run keeps the mode the workspace was configured with.
    ``profile`` is the Databricks CLI profile name to address — passed via
    ``--profile`` to every CLI invocation so ambiguous `~/.databrickscfg`
    entries (e.g. DEFAULT and a named profile both pointing at the same host)
    don't error out. If ``None``, we resolve it from the host after login.
    If skip_preflight is True, skip the entire preflight block below — auth
    validation, the AI Gateway probe, and model discovery — trusting a prior
    ``ug configure``. The PAT/bearer is already exported (``apply_pat_environment``
    in ``_launch_tool``) and the gateway was verified by that earlier configure.
    Only the local profile resolution and the shared state assembly still run;
    the saved model lists are preserved.
    ``fable_enabled`` opts the premium Claude Fable family into Claude Code's
    ``ANTHROPIC_DEFAULT_FABLE_MODEL`` pin (default off). ``None`` means "inherit":
    a launch re-run keeps whatever the workspace was configured with; ``True``/
    ``False`` come from an explicit ``configure --enable-fable``/``--disable-fable``.
    """
    workspace = normalize_workspace_url(workspace)
    prior_state = load_state()
    previous_workspace = prior_state.get("workspace")
    if use_pat is None:
        use_pat = bool(prior_state.get("use_pat")) and previous_workspace == workspace
    if fable_enabled is None:
        fable_enabled = bool(prior_state.get("fable_enabled")) and previous_workspace == workspace
    if databricks_ai_tools_enabled is None:
        # Opt-out: on by default. With no flag, keep this workspace's prior
        # choice but don't inherit another workspace's opt-out.
        disabled = (
            prior_state.get("databricks_ai_tools_enabled") is False
            and previous_workspace == workspace
        )
        databricks_ai_tools_enabled = not disabled
    fetch_all = tools is None

    # Assemble the shared workspace state that doesn't depend on model discovery:
    # workspace, profile, auth mode, base URLs. `profile` may still be None here;
    # each path below resolves it once, where a host->profile lookup is reliable
    # (the skip branch trusts the prior configure; the preflight resolves after
    # login). --skip-preflight persists exactly this and returns, trusting a prior
    # `ug configure` — it already validated auth + the AI Gateway and saved the
    # model lists (carried over by load_state, left untouched).
    state = load_state()
    state["workspace"] = workspace
    if profile:
        state["profile"] = profile
    else:
        state.pop("profile", None)
    # UC discovery is now always-on; drop any flag persisted by older versions.
    state.pop("uc_enabled", None)
    # Persist the auth mode so launches rebuild the same (PAT-based) agent
    # auth command; an explicit re-configure without --use-pat clears it.
    if use_pat:
        state["use_pat"] = True
    else:
        state.pop("use_pat", None)
    # Persist the Fable opt-in so launches keep pinning the family; an explicit
    # `configure --disable-fable` (fable_enabled=False) clears it.
    if fable_enabled:
        state["fable_enabled"] = True
    else:
        state.pop("fable_enabled", None)
    state["databricks_ai_tools_enabled"] = databricks_ai_tools_enabled
    if clear_custom_oauth:
        state.pop("custom_oauth", None)
    elif custom_oauth is not None:
        state["custom_oauth"] = dict(custom_oauth)
    elif previous_workspace != workspace:
        state.pop("custom_oauth", None)
    state["base_urls"] = build_shared_base_urls(workspace)

    if skip_preflight:
        # A prior `ug configure` created the profile; resolve it locally (no
        # login needed) and persist it so launches disambiguate.
        if profile is None:
            profile = find_profile_name_for_host(workspace)
            if profile:
                state["profile"] = profile
        save_state(state)
        # Scrub MCP entries ucode wrote for a previous workspace.
        if previous_workspace and previous_workspace != workspace:
            purge_cross_workspace_mcp_residue(state, workspace)
        # Diagnostic reasons are transient (attached after save_state so they
        # don't land on disk). No discovery ran, so there is nothing to report.
        state["_discovery_reasons"] = {"claude": None, "gemini": None, "codex": None, "oss": None}
        return state

    # ── Preflight (bypassed above under --skip-preflight): validate Databricks
    #    auth + the AI Gateway, then discover the available models. ──
    if use_pat:
        if not profile:
            raise RuntimeError(
                "--use-pat requires a Databricks CLI profile. Pass one via `--profiles <name>`."
            )
        pat = resolve_pat_token(profile)
        if not pat:
            raise RuntimeError(
                f"--use-pat: profile '{profile}' has no personal access token in "
                "~/.databrickscfg (its auth_type must be `pat`). Add a `token = <PAT>` "
                f"entry under [{profile}], or re-run without --use-pat to use OAuth."
            )
        # Export the PAT for this process and launched agent subprocesses so
        # every token fetch takes the static-bearer path. ensure_pat_bearer
        # keeps a non-empty pre-set bearer (CI escape hatch) but treats an
        # empty one as absent, so it never shadows the PAT. Pass the validated
        # token to avoid re-reading ~/.databrickscfg.
        ensure_pat_bearer(profile, pat)
        ensure_databricks_auth(workspace, profile)
    elif force_login:
        run_databricks_login(workspace, profile)
    else:
        ensure_databricks_auth(workspace, profile)
    # After login the profile exists in ~/.databrickscfg, so a host->profile
    # lookup is reliable even when it returned nothing above.
    if profile is None:
        profile = find_profile_name_for_host(workspace)
        if profile:
            state["profile"] = profile
    with spinner("Verifying Unity AI Gateway..."):
        token = get_databricks_token(workspace, profile)
        model_service_probe = probe_unity_gateway_capabilities(workspace, token)
    if model_service_probe.resource_available:
        print_success("Unity AI Gateway connected")
    else:
        print_warning(f"Model service: {model_service_probe.detail}")

    want_claude = (
        fetch_all or "claude" in tools or "opencode" in tools or "copilot" in tools or "pi" in tools
    )
    want_gemini = fetch_all or "gemini" in tools or "opencode" in tools or "pi" in tools
    want_codex = fetch_all or "codex" in tools or "copilot" in tools or "pi" in tools
    # Codex smart routing can select OSS models such as GLM, so a Codex-only
    # configure must persist that discovered family too.
    want_oss = fetch_all or "opencode" in tools or "codex" in tools

    claude_reason: str | None = None
    gemini_reason: str | None = None
    codex_reason: str | None = None
    oss_reason: str | None = None
    claude_models = {}
    gemini_models = []
    codex_models = []
    oss_models = []
    opencode_models: dict[str, list[str]] = {}
    web_search_model: str | None = None
    if skip_model_discovery:
        # Provider mode: the agent routes through a Model Provider Service and
        # pins no Databricks model, so the full family discovery is unused. Web
        # search (claude only) still needs one Responses-capable model, so fetch
        # just that with a single call.
        if want_claude:
            with spinner("Fetching web search model..."):
                ws_models, _ = discover_codex_models(workspace, token)
            if ws_models:
                web_search_model = ws_models[0]
    else:
        # UC-first, best-effort: one UC model-services call yields all families
        # as `system.ai.<model-name>` ids, bucketed by name. If a family comes
        # back empty (workspace without UC model-services, or the listing
        # failed), fall back to the per-family AI Gateway listing for that
        # family only.
        with spinner("Fetching available models..."):
            ms_claude, ms_codex, ms_gemini, ms_oss, ms_reason = discover_model_services(
                workspace, token
            )
            if want_claude:
                claude_models, claude_reason = ms_claude, ms_reason
                if not claude_models:
                    claude_models, claude_reason = discover_claude_models(workspace, token)
                # Fable is opt-in (`configure --enable-fable`). Unless enabled,
                # drop it from the discovered bundle entirely so it never becomes
                # part of any agent's config — not claude's family pins, nor the
                # opencode/pi/copilot model lists built from claude_models.
                if not fable_enabled:
                    claude_models.pop("fable", None)
            if want_gemini:
                gemini_models, gemini_reason = ms_gemini, ms_reason
                if not gemini_models:
                    gemini_models, gemini_reason = discover_gemini_models(workspace, token)
            if want_codex:
                codex_models, codex_reason = ms_codex, ms_reason
                if not codex_models:
                    codex_models, codex_reason = discover_codex_models(workspace, token)
            if want_oss:
                oss_models, oss_reason = ms_oss, ms_reason
        if claude_models:
            opencode_models["anthropic"] = list(claude_models.values())
        if gemini_models:
            opencode_models["gemini"] = gemini_models
        if oss_models:
            opencode_models["oss"] = oss_models

    if skip_model_discovery:
        # Don't clobber any previously-discovered Databricks model lists; provider
        # mode just doesn't refresh or use them. Persist the web-search model so
        # claude's web_search MCP keeps working through the normal gateway.
        if web_search_model:
            state["web_search_model"] = web_search_model
    else:
        if want_claude:
            state["claude_models"] = claude_models
        if want_gemini:
            state["gemini_models"] = gemini_models
        if want_codex:
            state["codex_models"] = codex_models
        if want_oss:
            state["oss_models"] = oss_models
        if fetch_all or "opencode" in tools:
            state["opencode_models"] = opencode_models
    save_state(state)
    # Scrub MCP entries that ucode wrote for the previous workspace so the new
    # workspace's agent configs aren't stale.
    if previous_workspace and previous_workspace != workspace:
        purge_cross_workspace_mcp_residue(state, workspace)
    # Diagnostic reasons are transient — attach after save_state so they don't
    # land on disk but are available to the caller for this run.
    state["_discovery_reasons"] = {
        "claude": claude_reason,
        "gemini": gemini_reason,
        "codex": codex_reason,
        "oss": oss_reason,
    }
    return state


def _configure_shared_workspace_states(
    workspaces: list[tuple[str, str | None]],
    tools: list[str] | None,
    *,
    force_login: bool,
    use_pat: bool = False,
    fable_enabled: bool | None = None,
    databricks_ai_tools_enabled: bool | None = None,
    custom_oauth: CustomOAuthConfig | None = None,
    clear_custom_oauth: bool = False,
) -> list[dict]:
    if not workspaces:
        raise RuntimeError("At least one workspace must be provided.")
    states: list[dict] = []
    for workspace, profile in workspaces:
        custom_oauth_kwargs = {"custom_oauth": custom_oauth} if custom_oauth is not None else {}
        if clear_custom_oauth:
            custom_oauth_kwargs["clear_custom_oauth"] = True
        states.append(
            configure_shared_state(
                workspace,
                profile=profile,
                tools=tools,
                force_login=force_login,
                use_pat=use_pat,
                fable_enabled=fable_enabled,
                databricks_ai_tools_enabled=databricks_ai_tools_enabled,
                **custom_oauth_kwargs,
            )
        )
    return states


def _provider_summary(tool: str, state: dict) -> str:
    """Short label for the Configuration Complete box: 'Databricks' when no
    Model Provider Service is configured, otherwise the external provider type
    backing this tool (claude routes to Anthropic, codex to OpenAI)."""
    if not get_provider_service(state, tool):
        return "Databricks"
    return {"claude": "Anthropic", "codex": "OpenAI"}.get(tool, "Model Provider Service")


def _maybe_select_provider_service(tool: str, state: dict) -> dict:
    """Interactively let the user route claude/codex through a Model Provider
    Service instead of Databricks models, and persist (or clear) the choice.

    No-op for tools other than claude/codex/gemini. Falls back to Databricks when no
    matching provider services are found or the listing fails.
    """
    if tool not in ("claude", "codex", "gemini"):
        return state
    display = TOOL_SPECS[tool]["display"]

    def _use_databricks() -> dict:
        new_state = set_provider_service(state, tool, None)
        save_state(new_state)
        return new_state

    # Probe first so we only offer the picker when it's actually usable. The
    # interactive path always reaches here, so explain any fallback rather than
    # silently dropping back to Databricks.
    token = get_databricks_token(state["workspace"], state.get("profile"))
    with spinner("Checking for model provider services..."):
        names, reason = list_tool_provider_services(tool, state["workspace"], token)
    if reason is not None:
        # Most workspaces don't have the feature enabled — that's the common case,
        # so fall back to Databricks silently. Only surface unexpected failures.
        if not is_model_provider_feature_unavailable(reason):
            print_warning(f"Could not list model provider services: {reason}")
            print_note("Falling back to Databricks models.")
        return _use_databricks()
    if not names:
        # Feature is on but no service matches this tool's provider type.
        print_note(f"Using Databricks models for {display}.")
        return _use_databricks()

    choice = prompt_for_selection(
        f"How should {display} get its models?",
        [
            ("databricks", "Databricks Hosted"),
            ("mps", "External Models"),
        ],
    )
    if choice is None:
        raise KeyboardInterrupt
    if choice == "databricks":
        return _use_databricks()

    selected = prompt_for_selection(
        "Select a model provider service:", [(name, name) for name in names]
    )
    if selected is None:
        raise KeyboardInterrupt
    state = set_provider_service(state, tool, selected)
    save_state(state)
    print_success(f"{display} will route through {selected}")
    return state


def configure_workspace_command(
    tool: str | None = None,
    selected_tools: list[str] | None = None,
    workspaces: list[tuple[str, str | None]] | None = None,
    *,
    prompt_optional_updates: bool = True,
    use_pat: bool = False,
    skip_validate: bool = False,
    skip_unavailable: bool = False,
    fable_enabled: bool | None = None,
    databricks_ai_tools_enabled: bool | None = None,
    custom_oauth: CustomOAuthConfig | None = None,
    offer_optional_setup: bool = False,
) -> int:
    if tool is not None and selected_tools is not None:
        raise RuntimeError("Use either --agent or --agents, not both.")

    # The Databricks-vs-Model-Provider-Service picker is shown only on the fully
    # interactive path (`ug configure` with no --agent/--agents). Naming agents
    # explicitly signals the non-interactive flow, which stays on Databricks.
    offer_provider = tool is None and selected_tools is None

    workspace_entries = workspaces or [_prompt_for_configuration(tool)]

    if tool is not None:
        states = _configure_shared_workspace_states(
            workspace_entries,
            [tool],
            force_login=True,
            use_pat=use_pat,
            fable_enabled=fable_enabled,
            databricks_ai_tools_enabled=databricks_ai_tools_enabled,
            custom_oauth=custom_oauth,
            clear_custom_oauth=custom_oauth is None,
        )
        state = states[0]
        state = configure_single_tool(tool, state)
        install_databricks_ai_tools_for_agents([tool], state)
        spec = TOOL_SPECS[tool]
        console.print(
            Panel(
                f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]\n"
                f"[bold]{spec['display']}:[/bold] [green]configured[/green] "
                f"[dim](Provider: {_provider_summary(tool, state)})[/dim]",
                title="Configuration Complete",
                style="green",
                expand=False,
            )
        )
        if skip_validate:
            print_note(f"Skipping {spec['display']} validation (--skip-validate).")
            return 0
        with spinner(f"Validating {spec['display']}..."):
            ok, err = validate_tool(tool)
        if ok:
            print_success(f"{spec['display']} is working")
        else:
            print_err(f"{spec['display']}: {provider_permission_error(tool, state, err)}")
            managed = bool(state.get("managed_configs", {}).get(tool))
            restore_file(spec["config_path"], spec["backup_path"], managed)
            available_tools = [t for t in (state.get("available_tools") or []) if t != tool]
            state["available_tools"] = available_tools
            save_state(state)
            raise RuntimeError(f"{spec['display']} validation failed — config reverted.")
        return 0

    states = _configure_shared_workspace_states(
        workspace_entries,
        selected_tools,
        force_login=True,
        use_pat=use_pat,
        fable_enabled=fable_enabled,
        databricks_ai_tools_enabled=databricks_ai_tools_enabled,
        custom_oauth=custom_oauth,
        clear_custom_oauth=custom_oauth is None,
    )
    state = states[0]
    save_state(state)

    available_on_workspace: list[str] = []
    tools_to_check = selected_tools or list(TOOL_SPECS)
    for tool_name in tools_to_check:
        with spinner(f"Checking {TOOL_SPECS[tool_name]['display']} availability..."):
            if check_gateway_endpoint(state, tool_name):
                available_on_workspace.append(tool_name)

    if not available_on_workspace:
        print_err("No coding agents are available on this workspace.")
        _print_discovery_diagnostics(state)
        return 1

    if selected_tools is None:
        picked = prompt_for_tools([(t, TOOL_SPECS[t]["display"]) for t in available_on_workspace])
    else:
        unavailable_tools = [
            tool_name for tool_name in selected_tools if tool_name not in available_on_workspace
        ]
        if unavailable_tools:
            _print_discovery_diagnostics(state)
            displays = ", ".join(
                TOOL_SPECS[tool_name]["display"] for tool_name in unavailable_tools
            )
            if not skip_unavailable:
                raise RuntimeError(
                    f"Requested agent(s) not available on this workspace: {displays}. "
                    "Pass --skip-unavailable to configure the available ones instead."
                )
            print_warning(f"Skipping agent(s) not available on this workspace: {displays}.")
        picked = [tool_name for tool_name in selected_tools if tool_name in available_on_workspace]

    if not picked:
        print_note("No coding agents selected — nothing to configure.")
        return 0

    for tool_name in picked:
        install_tool_binary(
            tool_name,
            strict=False,
            update_existing=True,
            prompt_optional_updates=prompt_optional_updates,
        )

    # Offer the provider picker for the chosen claude/codex tools only on the
    # interactive path (no --agents); otherwise stay on the Databricks path.
    if offer_provider:
        for tool_name in picked:
            state = _maybe_select_provider_service(tool_name, state)

    if offer_optional_setup:
        state = configure_selected_tools(state, picked, install_ai_tools=False)
    else:
        state = configure_selected_tools(state, picked)

    summary_lines = [f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]"]
    for tool_name in picked:
        spec = TOOL_SPECS[tool_name]
        summary_lines.append(
            f"[bold]{spec['display']}:[/bold] [green]configured[/green] "
            f"[dim](Provider: {_provider_summary(tool_name, state)})[/dim]"
        )
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

    if skip_validate:
        print_note("Skipping agent validation (--skip-validate).")
    else:
        # Limit validation to just-configured tools so we don't re-validate
        # previously-configured tools the user didn't touch this run.
        validate_state = {**state, "available_tools": picked}
        validate_all_tools(validate_state)
    if offer_optional_setup and not is_dry_run():
        _configure_optional_setup(state, picked)
    return 0


def status() -> int:
    state = load_state()
    workspace = state.get("workspace")
    managed_configs = state.get("managed_configs") or {}
    mcp_servers = state.get("mcp_servers") or []
    configured_tools = set(state.get("available_tools") or managed_configs.keys())

    console.print(heading("ug status"))
    console.print(
        f"  {status_badge('Configured', 'ok') if workspace else status_badge('Not Configured', 'warn')}"
    )

    print_heading("Provider")
    print_kv("Workspace URL", workspace or "not configured")
    profile = state.get("profile")
    if profile:
        print_kv("CLI profile", profile)

    if workspace:
        managed = load_managed_state(workspace)
        if managed:
            _print_managed_summary(managed, state, None)

    print_heading("Coding Agents")
    for tool, spec in TOOL_SPECS.items():
        configured = tool in configured_tools
        base_url = (
            state.get("base_urls", {}).get(tool, "not configured")
            if configured
            else "not configured"
        )
        config_path = spec["config_path"]
        print_kv("Coding Agent", spec["display"])
        print_kv("Configured", "yes" if configured else "no")
        provider_service = get_provider_service(state, tool)
        if configured and provider_service:
            print_kv("Model Provider Service", provider_service)
        print_kv("Base URL", base_url)
        if configured and tool in MCP_CLIENTS:
            tool_mcp_servers = [
                str(server.get("name"))
                for server in mcp_servers
                if tool in (server.get("clients") or [])
                and server.get("name")
                and server.get("kind") != SKILLS_MCP_KIND
            ]
            print_kv("MCP list command", str(MCP_CLIENTS[tool]["list_command"]))
            print_kv(
                "MCP servers",
                ", ".join(tool_mcp_servers) if tool_mcp_servers else "none saved by ug",
            )
        print_kv("Config file", str(config_path) if config_path.exists() else "missing")
        if tool == "claude":
            managed_path, managed_status, backup_status = claude_agent.managed_settings_status(
                state
            )
            print_kv("OS-managed settings", managed_status)
            print_kv("Managed settings file", str(managed_path) if managed_path else "unsupported")
            print_kv("Managed settings backup", backup_status)
        elif tool == "codex":
            managed_path, managed_status, backup_status = codex_agent.managed_config_status(state)
            print_kv("OS-managed settings", managed_status)
            print_kv("Managed settings file", str(managed_path) if managed_path else "unsupported")
            print_kv("Managed settings backup", backup_status)
        console.print()

    print_heading("Skills")
    skill_mcp_entry = next((s for s in mcp_servers if s.get("kind") == SKILLS_MCP_KIND), None)
    if not skill_mcp_entry:
        print_kv("Skills", "not configured")
    else:
        locations = skill_mcp_entry.get("skill_locations") or []
        print_kv(
            "Skill MCP Locations",
            ", ".join(locations) if locations else "none — utility tools only",
        )
        configured_agents = [
            str(MCP_CLIENTS[client]["display"])
            for client in (skill_mcp_entry.get("clients") or [])
            if client in MCP_CLIENTS
        ]
        print_kv("Configured", ", ".join(configured_agents) if configured_agents else "none")

    print_heading("Tracing")
    tracing = state.get("tracing") or {}
    if tracing.get("enabled"):
        print_kv("MLflow tracing", "enabled")
        print_kv("Tracking URI", str(tracing.get("tracking_uri") or "unknown"))
        print_kv(
            "Experiment",
            f"{tracing.get('experiment_name')} (id {tracing.get('experiment_id')})",
        )
        uc_destination = tracing.get("uc_destination")
        if uc_destination:
            print_kv("Unity Catalog", str(uc_destination))
        sql_warehouse_id = tracing.get("sql_warehouse_id")
        if sql_warehouse_id:
            print_kv("SQL warehouse", str(sql_warehouse_id))
    else:
        print_kv("MLflow tracing", "disabled")

    print_heading("State")
    print_kv("State file", str(STATE_PATH) if STATE_PATH.exists() else "missing")
    print_note("Use `ug configure` to update workspace settings or configure new tools.")
    print_note("Use `ug configure mcp` to add Databricks MCP servers to configured coding tools.")
    print_note(
        "Use `ug configure skills` to set up Unity Catalog Skills for configured coding tools."
    )
    print_note("Use `ug configure tracing` to log coding sessions to an MLflow experiment.")
    print_note("Use `ug revert` to clear managed configs and restore prior files.")
    return 0


def revert() -> int:
    state = load_state()
    managed_configs = state.get("managed_configs") or {}
    mcp_results = revert_mcp_configs(state)
    claude_managed_result = claude_agent.revert_managed_settings()
    codex_managed_result = codex_agent.revert_managed_config()

    results: dict[str, bool] = {
        tool: restore_file(
            spec["config_path"], spec["backup_path"], bool(managed_configs.get(tool))
        )
        for tool, spec in TOOL_SPECS.items()
    }
    pi_settings_restored = restore_file(
        PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH, bool(managed_configs.get("pi"))
    )
    # Older Codex (< 0.134.0) had ucode edit the shared ~/.codex/config.toml in
    # place; restoring the per-profile file above does not undo that.
    legacy_codex_stripped = revert_legacy_shared_config()
    clear_state()

    print_heading("Revert")
    print_kv("Workspace", state.get("workspace") or "none")
    for tool, spec in TOOL_SPECS.items():
        print_kv(f"{spec['display']} config", "restored" if results[tool] else "unchanged")
    if legacy_codex_stripped:
        print_kv("Codex shared config", "ucode entries removed")
    print_kv("Claude Code OS-managed settings", claude_managed_result)
    print_kv("Codex OS-managed settings", codex_managed_result)
    print_kv("Pi settings", "restored" if pi_settings_restored else "unchanged")
    for client, spec in MCP_CLIENTS.items():
        print_kv(
            f"{spec['display']} MCP config",
            "restored" if mcp_results.get(client) else "unchanged",
        )
    print_success("ug state cleared")
    return 0


# ---------------------------------------------------------------------------
# typer app
# ---------------------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
configure_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(configure_app, name="configure", help="Configure workspace and tool settings.")
mcp_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(mcp_app, name="mcp", help="MCP servers exposed by ug.")
skill_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(skill_app, name="skill", help="Databricks Skills for your coding tools.")
setup_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(
    setup_app,
    name="setup",
    help="Author the workspace's managed coding config (admins only). See `ug setup help`.",
)


def _version_callback(value: bool) -> None:
    if value:
        from ucode.telemetry import ucode_version

        print(ucode_version())
        raise typer.Exit()


def _configure_agents_for_mcp(
    requested: list[str], *, prompt_optional_updates: bool = True
) -> set[str]:
    """Ensure the named coding agents are set up (workspace + models) so a
    subsequent `ug mcp add` has them as targets, and return their canonical
    names. Mirrors `ug configure --agents`: model agents go through
    configure_workspace_command (which installs binaries and configures models);
    Cursor is MCP-only, so it just needs workspace state established and rides
    along via MCP_ONLY_CLIENTS. Interactive — prompts for the workspace URL on
    first run."""
    wants_cursor = "cursor" in requested
    model_agent_names = ",".join(a for a in requested if a != "cursor")
    configured: set[str] = set()
    if model_agent_names:
        selected_tools = _parse_agents_option(model_agent_names)
        configure_workspace_command(
            selected_tools=selected_tools, prompt_optional_updates=prompt_optional_updates
        )
        configured.update(selected_tools)
    if wants_cursor:
        # Establish workspace state for a Cursor-only run; when model agents were
        # configured above the workspace is already set, so Cursor just rides along.
        if not model_agent_names:
            _configure_shared_workspace_states(
                [_prompt_for_configuration(None)], tools=[], force_login=True
            )
        configured.add("cursor")
    return configured


def _configure_optional_setup(state: dict, tools: list[str]) -> None:
    enabled = prompt_yes_no("Configure MCP servers, skills, and plugins?")
    state["databricks_ai_tools_enabled"] = enabled
    save_state(state)
    if not enabled:
        return

    install_databricks_ai_tools_for_agents(tools, state)
    configure_mcp_command()


@mcp_app.command("add")
def mcp_add(
    location: Annotated[
        str | None,
        typer.Option(
            "--location",
            help="Non-interactive: register the MCP services in the given Unity Catalog "
            "`<catalog>.<schema>` (e.g. `system.ai`) and exit without showing the picker. "
            "Servers already configured outside this location are kept.",
        ),
    ] = None,
    services: Annotated[
        str | None,
        typer.Option(
            "--services",
            help="Register this comma-separated subset of MCP services (additively). Full names "
            "like `system.ai.github` work on their own; bare short names like `github` need "
            "--location to locate them. Omit --services to register the whole --location schema; "
            'an empty `--services ""` adds nothing (no-op).',
        ),
    ] = None,
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents",
            help="Comma-separated coding agents to register the server(s) for (e.g. "
            "claude,codex,cursor). Any that aren't configured yet are set up first "
            "(workspace + models), so this works as a one-command setup. Without --agents, "
            "the server is registered for every already-configured agent.",
        ),
    ] = None,
) -> None:
    """Add Databricks MCP servers to installed coding tools.

    Like `ug configure mcp`, but purely additive: it never removes MCP servers
    that are already configured, only registers new ones. Pass --agents to target
    (and, if needed, set up) specific agents.
    """
    selected = None if services is None else {s.strip() for s in services.split(",") if s.strip()}
    requested_agents = (
        None
        if agents is None
        else ({a.strip().lower() for a in agents.split(",") if a.strip()} or None)
    )
    try:
        scope = _configure_agents_for_mcp(sorted(requested_agents)) if requested_agents else None
        add_mcp_command(location=location, services=selected, agents=scope)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@mcp_app.command("remove")
def mcp_remove(
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents",
            help="Comma-separated coding agents to remove the server(s) from (e.g. "
            "claude,codex). A server registered on several agents is unregistered only "
            "from the named ones and kept on the rest. Without --agents, a selected server "
            "is removed from every agent it's on.",
        ),
    ] = None,
) -> None:
    """Remove configured Databricks MCP servers from your coding tools.

    Interactive: shows the servers you currently have configured and unregisters the
    ones you select. Needs no Databricks login.
    """
    requested_agents = (
        None
        if agents is None
        else ({a.strip().lower() for a in agents.split(",") if a.strip()} or None)
    )
    try:
        remove_mcp_command(agents=requested_agents)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@mcp_app.command("web-search")
def mcp_web_search_cmd() -> None:
    """Run the web_search MCP server over stdio. Invoked as a subprocess by Claude Code."""
    from ucode.mcp_web_search import serve

    serve()


@skill_app.command("add")
def skills_add(
    location: Annotated[
        str | None,
        typer.Option(
            "--location", help="Comma-separated `<catalog>.<schema>` skill scopes to add."
        ),
    ] = None,
    mcp: Annotated[
        bool,
        typer.Option(
            "--mcp",
            help="Add the schemas to the skills MCP connection's scope instead of downloading.",
        ),
    ] = False,
    path: Annotated[
        str | None,
        typer.Option(
            "--path",
            help="(download) Existing absolute project directory to download into; defaults "
            "to user-level skill directories.",
        ),
    ] = None,
    skills: Annotated[
        str | None,
        typer.Option(
            "--skills",
            help="(download) Download only this comma-separated subset of skills instead of "
            "every skill in the schema. Bare securable names (e.g. `my-skill`) need a single "
            "--location; fully-qualified `<catalog>.<schema>.<name>` names work on their own. "
            "Not valid with --mcp.",
        ),
    ] = None,
) -> None:
    """Add Databricks Skills to your coding tools, keeping any already configured.

    With ``--mcp``, adds the given schemas to the skills MCP connection's scope.
    Otherwise downloads each schema's skills to project-level skill directories under
    ``--path``, or to user-level skill directories when omitted, keeping
    already-downloaded skills. ``--skills`` narrows a download to a subset of one
    schema's skills, by bare name (with ``--location``) or fully-qualified
    ``<catalog>.<schema>.<name>``.
    """
    try:
        locations = _parse_skill_locations(location)
        requested_skills = (
            None if skills is None else {s.strip() for s in skills.split(",") if s.strip()}
        )
        if mcp and path is not None:
            raise RuntimeError("--path is not supported when using --mcp")
        if mcp and requested_skills is not None:
            raise RuntimeError("--skills is not supported when using --mcp")
        qualified_skill_parts: dict[str, list[str]] = {}
        invalid_skills: list[str] = []
        for skill in requested_skills or set():
            if "." not in skill:
                continue
            parts = skill.split(".")
            if len(parts) != 3 or any(not part or part != part.strip() for part in parts):
                invalid_skills.append(skill)
            else:
                qualified_skill_parts[skill] = parts
        if invalid_skills:
            raise RuntimeError(
                "--skills entries must be bare names or fully qualified "
                "`<catalog>.<schema>.<name>` values "
                f"(invalid: {', '.join(sorted(invalid_skills))})."
            )
        if requested_skills is not None and not locations:
            schemas = {".".join(parts[:2]) for parts in qualified_skill_parts.values()}
            bare = sorted(skill for skill in requested_skills if skill not in qualified_skill_parts)
            if bare:
                raise RuntimeError(
                    "--skills short names need --location (or pass full names like "
                    f"`<catalog>.<schema>.<name>`): {', '.join(bare)}"
                )
            if len(schemas) != 1:
                raise RuntimeError(
                    "--skills without --location must all share one `<catalog>.<schema>` "
                    f"(got: {', '.join(sorted(schemas)) or 'none'}); pass --location instead."
                )
            locations = list(schemas)
        if not locations:
            raise RuntimeError("--location is required for `ucode skill add`.")
        if requested_skills is not None and len(locations) != 1:
            raise RuntimeError(
                f"--skills requires a single --location (got: {', '.join(locations)})."
            )
        mismatched_skills = sorted(
            skill
            for skill, parts in qualified_skill_parts.items()
            if ".".join(parts[:2]) != locations[0]
        )
        if mismatched_skills:
            raise RuntimeError(
                f"--skills entries must match --location `{locations[0]}` "
                f"(got: {', '.join(mismatched_skills)})."
            )
        selected_skills = (
            None if requested_skills is None else {s.split(".")[-1] for s in requested_skills}
        )
        if mcp:
            add_skills_command(locations)
        else:
            configure_skills_download_command(locations, path=path, skills=selected_skills)
    except (RuntimeError, ValueError) as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@app.command("mcp-proxy", hidden=True)
def mcp_proxy_cmd(
    url: Annotated[
        str,
        typer.Option("--url", help="Databricks streamable-HTTP MCP endpoint to forward to."),
    ],
    host: Annotated[
        str | None,
        typer.Option(
            "--host", help="Workspace URL for token minting. Defaults to the saved workspace."
        ),
    ] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Databricks CLI profile.")
    ] = None,
    use_pat: Annotated[
        bool,
        typer.Option(
            "--use-pat",
            help="Authenticate with the profile's static personal access token (from "
            "~/.databrickscfg) instead of OAuth. Set automatically for workspaces configured "
            "with `ug configure --profiles <name> --use-pat`.",
        ),
    ] = False,
) -> None:
    """Bridge a coding agent's stdio MCP transport to a Databricks MCP endpoint.

    Each configured client spawns this as a local stdio MCP server (see
    `ug configure mcp`); it forwards messages to ``--url`` and injects a
    freshly-minted token on every upstream request, so it never expires
    mid-session. Not meant for interactive use — the agent manages this
    process's lifecycle."""
    from ucode.mcp_proxy import serve

    state = load_state()
    workspace = host or state.get("workspace")
    if not workspace:
        print_err("No workspace configured. Run `ug configure` first.")
        raise typer.Exit(1)
    profile = profile or state.get("profile")
    serve(url, workspace, profile, use_pat=use_pat or bool(state.get("use_pat")))


@app.command("auth-token", hidden=True)
def auth_token_cmd(
    host: Annotated[
        str | None, typer.Option("--host", help="Workspace URL. Defaults to the saved workspace.")
    ] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Databricks CLI profile.")
    ] = None,
    use_pat: Annotated[
        bool, typer.Option("--use-pat", help="Read the profile's static PAT instead of OAuth.")
    ] = False,
    force_refresh: Annotated[
        bool,
        typer.Option("--force-refresh", help="Force the Databricks CLI to mint a new token."),
    ] = False,
    client_id: Annotated[
        str | None,
        typer.Option(
            "--client-id", hidden=True, help="Experimental: custom public OAuth client ID."
        ),
    ] = None,
    redirect_url: Annotated[
        str | None,
        typer.Option(
            "--redirect-url",
            hidden=True,
            help="Registered OAuth callback URL. Defaults to http://localhost:8020.",
        ),
    ] = None,
    scopes: Annotated[
        str | None,
        typer.Option(
            "--scopes",
            hidden=True,
            help="Comma-separated OAuth scopes for the custom client.",
        ),
    ] = None,
) -> None:
    """Print a Databricks bearer token to stdout, then exit.

    This is the cross-platform helper invoked by Claude Code's `apiKeyHelper`
    and Codex's auth command on every token refresh. It is not meant for
    interactive use. All token logic (DATABRICKS_BEARER short-circuit, PAT
    profiles, OAuth refresh) lives in `get_databricks_token`, so the same
    binary works on macOS, Linux, and Windows without any POSIX shell."""
    import sys

    if client_id is not None and use_pat:
        print_err("--client-id cannot be combined with --use-pat.")
        raise typer.Exit(1)
    if redirect_url is not None and client_id is None:
        print_err("--redirect-url requires --client-id.")
        raise typer.Exit(1)
    if scopes is not None and client_id is None:
        print_err("--scopes requires --client-id.")
        raise typer.Exit(1)
    if client_id is not None and scopes is None:
        print_err("--scopes is required with --client-id.")
        raise typer.Exit(1)
    state = load_state()
    workspace = host or state.get("workspace")
    if not workspace:
        print_err("No workspace configured. Run `ug configure` first.")
        raise typer.Exit(1)
    profile = profile or state.get("profile")
    if client_id is None and (use_pat or state.get("use_pat")):
        # --use-pat explicitly means "serve the profile's static PAT". Fail
        # closed if it can't be read rather than falling through to OAuth —
        # `auth token` cannot serve a PAT-only profile, so that path would
        # surface a misleading stale-login error instead of the real cause.
        if not ensure_pat_bearer(profile):
            print_err(
                f"--use-pat: no personal access token available for profile "
                f"'{profile or '<none>'}'. Add a `token = <PAT>` entry under "
                f"[{profile or 'your-profile'}] in ~/.databrickscfg, or re-run "
                "`ug configure` without --use-pat to use OAuth."
            )
            raise typer.Exit(1)
    try:
        if client_id is not None:
            assert scopes is not None
            token = custom_oauth.get_custom_client_token(
                workspace,
                client_id=client_id,
                redirect_url=(
                    redirect_url if redirect_url is not None else custom_oauth.DEFAULT_REDIRECT_URL
                ),
                scopes=scopes.split(","),
                force_refresh=force_refresh,
            )
        else:
            token = get_databricks_token(workspace, profile, force_refresh=force_refresh)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    # Write the bare token (with trailing newline) to stdout — nothing else may
    # land on stdout or the consuming agent will treat it as part of the token.
    sys.stdout.write(token + "\n")


def _oauth_token_is_fresh(token: str, buffer_seconds: float = 120) -> bool:
    import base64
    import binascii
    import json
    import time

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        expires_at = float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except (IndexError, KeyError, TypeError, ValueError, binascii.Error, json.JSONDecodeError):
        return False
    return time.time() < expires_at - buffer_seconds


@app.command("codex-router-hook", hidden=True)
def codex_router_hook_cmd(
    event: str,
    host: Annotated[str | None, typer.Option("--host")] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    use_pat: Annotated[bool, typer.Option("--use-pat")] = False,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
) -> None:
    """Run a Codex smart-routing lifecycle hook."""
    import json
    import sys

    if not smart_routing_v2.enabled():
        return

    from ucode.smart_routing.codex_routing import (
        record_session_start,
        record_subagent_start,
        route_pre_tool_use,
    )

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return
    if not isinstance(payload, dict):
        return
    if event == "session-start":
        record_session_start(payload)
        return
    if event == "record-subagent":
        record = record_subagent_start(payload)
        matched = record.get("matches_router_decision")
        if matched is True:
            sys.stdout.write(
                json.dumps(
                    {
                        "systemMessage": "Smart Routing verified. "
                        f"Subagent is using {record.get('model')}."
                    }
                )
            )
        elif matched is False:
            sys.stdout.write(
                json.dumps(
                    {
                        "systemMessage": "Smart Routing mismatch: router requested "
                        f"{record.get('requested_model')}, but Codex started "
                        f"{record.get('model')}."
                    }
                )
            )
        # When matched is None the harness didn't report the subagent model —
        # the PreToolUse hook already injected the routed model, so emit nothing.
        return
    if event != "route-subagent" or not host:
        return
    if use_pat and not ensure_pat_bearer(profile):
        return
    token = os.environ.get("DATABRICKS_BEARER", "").strip()
    if not token:
        token = os.environ.get("OAUTH_TOKEN", "").strip()
        if not _oauth_token_is_fresh(token):
            try:
                token = get_databricks_token(host, profile, force_refresh=True)
            except RuntimeError:
                return
    output = route_pre_tool_use(
        payload,
        workspace=host,
        token=token,
        available_models=model or [],
        audit_decision=True,
    )
    if output is not None:
        sys.stdout.write(json.dumps(output))


@app.command("claude-router-hook", hidden=True)
def claude_router_hook_cmd(
    event: str,
    host: Annotated[str | None, typer.Option("--host")] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    use_pat: Annotated[bool, typer.Option("--use-pat")] = False,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
    socket_path: Annotated[str | None, typer.Option("--socket")] = None,
) -> None:
    """Run a Claude Code smart-routing lifecycle hook."""
    import json
    import sys

    if not smart_routing_v2.enabled():
        return

    from ucode.smart_routing.claude_routing import (
        record_session_start,
        record_subagent_start,
    )

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return
    if not isinstance(payload, dict):
        return
    if event == ROUTE_FIRST_PROMPT_EVENT:
        if not socket_path:
            socket_path = os.environ.get(FIRST_PROMPT_SOCKET_ENV)
        if not socket_path:
            return
        from pathlib import Path

        from ucode.smart_routing.claude_pty import (
            first_prompt_hook_output,
            request_first_prompt_route,
        )

        output = first_prompt_hook_output(request_first_prompt_route(Path(socket_path), payload))
        if output is not None:
            sys.stdout.write(json.dumps(output))
        return
    if event == "session-start":
        record_session_start(payload)
        return
    if event == "record-subagent":
        record = record_subagent_start(payload)
        matched = record.get("matches_router_decision")
        if matched is True:
            sys.stdout.write(
                json.dumps(
                    {
                        "systemMessage": "Smart Routing verified. "
                        f"Subagent is using {record.get('model')}."
                    }
                )
            )
        elif matched is False:
            sys.stdout.write(
                json.dumps(
                    {
                        "systemMessage": "Smart Routing mismatch: router requested "
                        f"{record.get('requested_model')}, but Claude Code started "
                        f"{record.get('model')}."
                    }
                )
            )
        # When matched is None the harness didn't report the subagent model —
        # the PreToolUse hook already injected the routed model, so emit nothing.
        return
    if event != "route-subagent" or not host:
        return
    token = os.environ.get("OAUTH_TOKEN") or os.environ.get("DATABRICKS_BEARER")
    if not token:
        if use_pat and not ensure_pat_bearer(profile):
            return
        try:
            token = get_databricks_token(host, profile)
        except RuntimeError:
            return
    output = smart_routing_v2.route_claude_pre_tool_use(
        payload,
        workspace=host,
        token=token,
        available_models=model or [],
        audit_decision=True,
    )
    if output is not None:
        sys.stdout.write(json.dumps(output))


def _auto_configure_tool(tool: str, custom_oauth: CustomOAuthConfig | None = None) -> None:
    """First-time setup for a single tool — mirrors configure_workspace_command."""
    existing = load_state()
    workspace = existing.get("workspace")
    profile = existing.get("profile")
    if not workspace:
        workspace, profile = _prompt_for_configuration(tool)
    configure_kwargs = {"custom_oauth": custom_oauth} if custom_oauth is not None else {}
    state = configure_shared_state(workspace, profile=profile, tools=[tool], **configure_kwargs)

    state = configure_single_tool(tool, state)

    spec = TOOL_SPECS[tool]
    console.print(
        Panel(
            f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]\n"
            f"[bold]{spec['display']}:[/bold] [green]configured[/green] "
            f"[dim](Provider: {_provider_summary(tool, state)})[/dim]",
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

    with spinner(f"Validating {spec['display']}..."):
        ok, err = validate_tool(tool)
    if ok:
        print_success(f"{spec['display']} is working")
    else:
        print_err(f"{spec['display']}: {provider_permission_error(tool, state, err)}")
        managed = bool(state.get("managed_configs", {}).get(tool))
        restore_file(spec["config_path"], spec["backup_path"], managed)
        available_tools = [t for t in (state.get("available_tools") or []) if t != tool]
        state["available_tools"] = available_tools
        save_state(state)
        raise RuntimeError(f"{spec['display']} validation failed — config reverted.")


CAN_USE_CACHED_CONFIG_AGENTS = frozenset({"claude", "codex"})


@contextmanager
def _smart_routing_v2_flag(enabled: bool) -> Iterator[None]:
    """Enable V2 for this launch without leaking into an embedding process."""
    if not enabled:
        yield
        return
    previous = os.environ.get(smart_routing_v2.ENV_VAR)
    os.environ[smart_routing_v2.ENV_VAR] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(smart_routing_v2.ENV_VAR, None)
        else:
            os.environ[smart_routing_v2.ENV_VAR] = previous


def _migrate_legacy_smart_routing(state: dict) -> dict:
    """Remove the former persisted opt-in and its permanent routing hooks."""
    if smart_routing_v2.LEGACY_STATE_KEY not in state:
        return state
    # The legacy key was shared by both agents. Clean both sets of files and
    # artifacts before the first helper removes the marker from state.
    codex_agent.disable_smart_routing(state)
    claude_agent.disable_smart_routing(state)
    return state


def _reject_disabled_agent(managed: dict | None, tool: str) -> None:
    """Refuse to launch ``tool`` when the managed config enables other agents but not this one.

    ``enabled_agents`` is an allowlist: launching an agent the admin didn't enable would run
    unmanaged, with none of their models or provider applied. A config that names no agents at all
    expresses no opinion, so it blocks nothing.
    """
    enabled = managed_enabled_tools(managed or {})
    if enabled and tool not in enabled:
        names = ", ".join(TOOL_SPECS[name]["display"] for name in enabled)
        raise RuntimeError(
            f"Your workspace's managed config doesn't enable {TOOL_SPECS[tool]['display']}. "
            f"Enabled: {names}."
        )


def _fetch_managed_config(state: dict) -> ManagedConfigResult:
    """The workspace's managed config for this launch, plus whether the feature is disabled.

    ``ManagedConfigResult(None, True)`` when the workspace has the feature disabled server-side;
    ``ManagedConfigResult(None, False)`` when the feature is on but no config is published.
    """
    with spinner("Loading..."):
        return refresh_managed_config(state)


def _note_recommended_agent(recommendation: dict | None, tool: str) -> None:
    """Say when the budget tier points at a different agent than the one being launched.

    Launching any enabled agent is allowed, so this informs rather than blocks — and explains why
    the session is not on the tier's model.
    """
    # The tier's own agent, not `recommended_agent`'s default_agent fallback: there is nothing to
    # say when the config's baseline simply differs from what the developer asked for.
    agent = (recommendation or {}).get("agent")
    if agent == tool or agent not in TOOL_SPECS:
        return
    model = (recommendation or {}).get("model")
    suffix = f" with {model}" if isinstance(model, str) and model else ""
    print_note(
        f"Your budget tier recommends {TOOL_SPECS[agent]['display']}{suffix}; "
        f"launching {TOOL_SPECS[tool]['display']} as requested."
    )


def _fetch_budget_recommendation(state: dict, managed: dict | None) -> dict | None:
    """The agent and model the caller's budget tier allows, or None when there is no budget to read.

    Enforcement is server-side, so a failed read only costs the recommendation: the config's own
    ``default_model`` still applies and the launch proceeds.
    """
    if managed is None or is_dry_run():
        return None
    reason: str | None = None
    recommendation = None
    with spinner("Checking your budget..."):
        try:
            recommendation, reason = get_model_recommendation(
                state["workspace"],
                get_databricks_token(state["workspace"], state.get("profile")),
            )
        except (RuntimeError, OSError) as exc:
            # A token that lapsed since the config refresh — or a Databricks CLI that isn't
            # installed or reachable — must not block the launch; the config's default_model stands.
            reason = str(exc)
    if reason is not None:
        print_warning(
            f"Could not check your budget ({reason}); "
            "using the default model from your workspace's config."
        )
    return recommendation


def _launch_title(tool: str) -> str:
    return f"Launching {TOOL_SPECS[tool]['display']} with Unity Gateway"


def _print_budget_panel(recommendation: dict, tool: str, managed: dict | None = None) -> None:
    """Show the workspace budget this launch spends against, when one is configured."""
    agent = recommendation.get("agent")
    display_agent = TOOL_SPECS[agent]["display"] if agent in TOOL_SPECS else None
    percent = budget_usage_percent(
        float(recommendation.get("current_spend") or 0.0),
        float(recommendation.get("effective_threshold") or 0.0),
    )
    line = recommendation_line(display_agent, recommendation.get("model"), percent)
    panel = render_budget_panel(
        recommendation,
        title=_launch_title(tool),
        extra_lines=[line] if line else None,
        managed=managed,
    )
    if panel is not None:
        console.print(panel)


def _register_managed_mcp_servers(managed: dict, tool: str, state: dict) -> None:
    """Apply the managed config's MCP servers to ``tool`` and persist what was registered.

    Persisting under ``managed_mcp_servers`` lets the next launch diff against it, so a server the
    admin later removes from the config is unregistered rather than left behind. A failure here never
    blocks the launch — the agent still starts, just without the workspace's MCP servers.
    """
    try:
        registered = apply_managed_mcp_servers(
            managed,
            tool,
            state["workspace"],
            state.get("profile"),
            use_pat=bool(state.get("use_pat")),
        )
    except RuntimeError as exc:
        print_warning(f"Could not register your workspace's MCP servers: {exc}")
        return
    # Persist even when empty so a config that dropped its last server clears the prior registration.
    others = [
        server
        for server in (state.get("managed_mcp_servers") or [])
        if isinstance(server, dict) and tool not in (server.get("clients") or [])
    ]
    state["managed_mcp_servers"] = others + registered
    save_state(state)
    if registered:
        names = ", ".join(str(server["name"]) for server in registered)
        print_note(f"Registered workspace MCP server(s) for {TOOL_SPECS[tool]['display']}: {names}")


def _managed_skill_locations(managed: dict) -> list[str]:
    """The ``<catalog>.<schema>`` skill locations the admin published, or ``[]``."""
    return [
        loc
        for loc in ((managed.get("skills") or {}).get("names") or [])
        if isinstance(loc, str) and loc
    ]


def _download_managed_skills(managed: dict, state: dict) -> None:
    """Download the admin-published skill schemas to disk (user scope).

    Managed skills are delivered by download only. The agent's ``/skills`` picker reads skill bundles
    from ``~/.claude/skills`` / ``~/.agents/skills`` on disk, so without this download a
    workspace-published skill never shows up in ``/skills``. Skills already on disk are left
    untouched, so a steady-state launch only lists each schema and writes nothing. Best-effort: a
    failure here never blocks the launch.
    """
    locations = _managed_skill_locations(managed)
    if not locations:
        return
    try:
        token = get_databricks_token(state["workspace"], state.get("profile"))
        written = download_managed_skills_on_launch(state["workspace"], token, locations)
    except RuntimeError as exc:
        print_warning(f"Could not download your workspace's skills: {exc}")
        return
    if written:
        print_note(f"Downloaded workspace skill(s) to disk: {', '.join(written)}")


def _should_launch_smart_routing(
    tool: str,
    tool_args: list[str],
    *,
    explicit_prompt: bool,
    model: str | None,
) -> bool:
    if model is not None or has_explicit_model_arg(tool_args):
        return False
    if not tool_args or explicit_prompt:
        return True
    return tool == "claude" and tool_args[0].startswith("-")


def _launch_options(
    tool: str,
    tool_args: list[str],
    *,
    smart_routing_enabled: bool,
    explicit_prompt: bool,
    model: str | None,
    provider: str | None,
) -> LaunchOptions:
    return LaunchOptions(
        claude_launch_model=model if tool == "claude" and provider is None else None,
        launch_smart_routing=(
            # Smart routing is enabled globally.
            smart_routing_enabled
            # Only Claude Code and Codex currently support smart routing.
            and tool in CAN_USE_CACHED_CONFIG_AGENTS
            # Smart routing does not currently support Model Provider Services.
            and provider is None
            # Route a supported interactive launch shape.
            and _should_launch_smart_routing(
                tool,
                tool_args,
                explicit_prompt=explicit_prompt,
                model=model,
            )
        ),
    )


def _launch_tool(
    tool_name: str,
    ctx: typer.Context,
    provider: str | None = None,
    refresh: bool = False,
    skip_preflight: bool = False,
    workspace_url: str | None = None,
    managed: dict | None = None,
    recommendation: dict | None = None,
    model: str | None = None,
    custom_oauth: CustomOAuthConfig | None = None,
) -> None:
    try:
        tool = normalize_tool(tool_name)
        explicit_prompt = _has_explicit_prompt(ctx)
        smart_routing_enabled = smart_routing_v2.enabled()
        # Launchers such as isaac put their harness arguments after `--`, so the harness's own
        # `--model` lands in ctx.args instead of a ucode option. It still determines the effective
        # launch model and should therefore win in the launch summary.
        forwarded_model = (
            explicit_model_arg_value(ctx.args) if tool in {"claude", "codex"} else None
        )
        # `--model` is exposed by the claude and gemini launch commands. Under a provider it selects
        # which of the service's targets/tiers to launch on, rather than being rejected — see the
        # provider branch below.
        # An explicit --workspace targets that workspace for this launch (and
        # auto-configures it if unseen), so `ug claude --provider ... --workspace ...`
        # works without a prior `ug configure`.
        if workspace_url:
            set_current_workspace(normalize_workspace_url(workspace_url))
        existing = load_state()
        # Workspaces configured with --use-pat export the profile's PAT as
        # DATABRICKS_BEARER up front so every auth check below (and the
        # launched agent itself) uses the static token instead of OAuth.
        apply_pat_environment(existing)
        needs_auto_configure = not existing.get("workspace") or tool not in (
            existing.get("available_tools") or []
        )
        ensure_bootstrap_dependencies(tool, update_existing=needs_auto_configure)
        if needs_auto_configure:
            if custom_oauth is None:
                _auto_configure_tool(tool)
            else:
                _auto_configure_tool(tool, custom_oauth=custom_oauth)
        state = ensure_provider_state(tool)
        # Remembered before the fallback below collapses the two cases: a managed config may not
        # silently override a provider the user typed on the command line (it errors instead).
        explicit_provider = provider
        # An explicit --provider overrides the persisted choice; otherwise fall
        # back to whatever `ug configure` saved for this tool.
        provider = provider or get_provider_service(state, tool)
        state = _migrate_legacy_smart_routing(state)
        # Fetched before `configure_shared_state` because it decides whether this agent may launch
        # at all and whether the model discovery below can be skipped.
        # Bare `ucode` already fetched one to choose the agent; refetching would double the
        # control-plane round trip and any fallback warning it printed.
        coding_agent_config_feature_disabled = False
        if managed is None:
            managed, coding_agent_config_feature_disabled = _fetch_managed_config(state)
        # Checked before discovery, which can take tens of seconds, so a blocked launch fails fast.
        _reject_disabled_agent(managed, tool)
        # Discovery exists to find models and isn't needed for managed config that already names them.
        managed_models_known = managed_supplies_models(managed, tool)
        # Re-fetch model lists on every launch so newly-added Databricks
        # endpoints show up without a manual `ug configure` (and so that
        # tools like pi which read multiple model bundles never run on
        # stale state from before a tool added a new bundle). Under a provider
        # this heavy discovery is skipped (only a web-search model is fetched).
        configure_kwargs = {"custom_oauth": custom_oauth} if custom_oauth is not None else {}
        state = configure_shared_state(
            state["workspace"],
            profile=state.get("profile"),
            tools=[tool],
            skip_model_discovery=bool(provider) or managed_models_known,
            skip_preflight=skip_preflight,
            **configure_kwargs,
        )
        # An admin-published managed config wins over the developer's own settings. Layered on after
        # `configure_shared_state`, whose returned state it overrides, and before the provider and
        # model are settled below — the two state files are never merged on disk.
        # Bare `ucode` already read one to choose the agent; refetching would double the round trip.
        if recommendation is None:
            recommendation = _fetch_budget_recommendation(state, managed)
        _note_recommended_agent(recommendation, tool)
        if managed is not None:
            state = resolve_state(managed, state, tool)
            print_success("Applied your workspace's managed coding agent config")
            unservable = managed_unservable_models(managed, tool)
            if unservable:
                print_warning(
                    f"Your workspace's managed config lists no {TOOL_SPECS[tool]['display']}-servable "
                    f"models ({', '.join(unservable)}); using your discovered models instead."
                )
        elif not coding_agent_config_feature_disabled:
            print_note("No managed coding agent config found; using your own settings")
        if managed is not None:
            managed_provider = managed_provider_service(managed, tool)
            if explicit_provider and managed_provider and managed_provider != explicit_provider:
                # An explicit --provider that disagrees with the admin's is a hard error rather
                # than a silent override: the user asked for something the managed config forbids,
                # and quietly routing them elsewhere would hide it.
                raise RuntimeError(
                    f"You cannot launch {TOOL_SPECS[tool]['display']} with provider "
                    f"{explicit_provider} because your admin has specified managed provider "
                    f"{managed_provider}."
                )
            if managed_provider:
                provider = managed_provider
        # Checked after the managed config settles `provider`: an admin-set provider must trip this
        # guard too, or routing would be persisted as on while a provider is active.
        if tool in CAN_USE_CACHED_CONFIG_AGENTS and smart_routing_enabled and provider:
            raise RuntimeError(
                f"{TOOL_SPECS[tool]['display']} smart routing cannot be enabled with "
                "--provider. Launch without a Model Provider Service and try again."
            )
        # Validate the provider service before launching — it must exist, be a
        # provider type this tool can route to (e.g. claude can't use an OpenAI
        # or Foundry service), and, for Bedrock, expose Claude models to pin.
        # Gemini is exempt: it validates the service and resolves its target in a single
        # lookup via resolve_gemini_provider_model (below), and uses no family model map.
        provider_models = None
        relayed = False
        coding_agent_config_defaults = (
            managed_claude_family_models(managed) or {}
            if tool == "claude" and managed is not None
            else {}
        )
        if provider and tool != "gemini":
            provider_models, error, relayed = resolve_provider_models(tool, state, provider)
            if error:
                if managed is not None and provider == managed_provider_service(managed, tool):
                    # Clear error if the admin has Unity Catalog grants the developer doesn't.
                    raise RuntimeError(
                        f"Your admin's managed config specifies provider {provider} for "
                        f"{TOOL_SPECS[tool]['display']}, which can't be used: {error}"
                    )
                raise RuntimeError(error)
            # A managed config launch uses exactly what the admin authored: pin Claude's family
            # models from the manifest's slots rather than the versions resolve_provider_models
            # re-derived from the service's live targets. The developer-configured path keeps that
            # re-derivation (see resolve_provider_models). Only when the manifest actually selected
            # this provider for claude, and authored something to pin.
            if (
                tool == "claude"
                and managed is not None
                and provider == managed_provider_service(managed, tool)
            ):
                authored = managed_provider_family_models(managed)
                if authored:
                    provider_models = authored
                    coding_agent_config_defaults = authored
        # The router's per-launch pick for the root session. Codex pins it as the
        # resolved model; claude pins it via ANTHROPIC_MODEL (route_root_model).
        route_root_model = None
        if provider:
            # Routing through a Model Provider Service pins no Databricks model;
            # the agent uses its own canonical model names (header selects the
            # provider). Skip model resolution, which would otherwise fail when
            # the workspace has no matching Databricks models.
            resolved_model = None
            # Relayed services forward --model to Claude Code's own flag at launch (below), not env.
            if tool == "claude" and not relayed and (model or provider_models):
                route_root_model = resolve_provider_launch_model(model, provider_models or {})
            if tool == "gemini":
                # Gemini is the exception: the request still names a concrete model
                # in the URL, so pin one of the service's targets (--model or default).
                resolved_model, gemini_error = resolve_gemini_provider_model(state, provider, model)
                if gemini_error:
                    raise RuntimeError(gemini_error)
        else:
            # A managed default_model is the model the admin wants sessions to start on, so it goes
            # in as the explicit model rather than being applied afterwards: for codex the proto has
            # no model list at all, so passing it here is the only way a launch succeeds when the
            # workspace's own discovery turned up nothing.
            managed_model = (
                managed_launch_model(managed, recommendation, tool) if managed is not None else None
            )
            state, resolved_model = resolve_launch_model(tool, state, managed_model)
            # The admin's model outranks a smart-routing pick too. Claude only launches on it when
            # pinned as ANTHROPIC_MODEL (route_root_model); other agents take `resolved_model`,
            # which already holds it from resolve_launch_model above.
            if managed_model:
                if tool == "claude":
                    route_root_model = managed_model
                else:
                    resolved_model = managed_model
            # An explicit `--model` is the user's own choice and outranks everything above (managed
            # default, smart-routing pick). Non-claude agents take it as the resolved model, which
            # Codex keeps an explicit --model in ctx.args and passes it to its CLI verbatim.
            if model and tool != "claude":
                resolved_model = model
        state = configure_tool(
            tool,
            state,
            resolved_model,
            provider=provider,
            provider_models=provider_models,
            relayed=relayed,
            route_root_model=route_root_model,
            # Claude's explicit model is launch-scoped and is passed through LaunchOptions below.
            custom_model=None,
            coding_agent_config_defaults=coding_agent_config_defaults,
        )
        # Relayed = a Claude subscription: forward --model to Claude Code's own flag, like `-- --model X`.
        if tool == "claude" and provider and relayed and model and not forwarded_model:
            ctx.args = ["--model", model, *ctx.args]
            forwarded_model = model
        print_section(_launch_title(tool))
        if managed is not None:
            print_kv("Config", "workspace-managed")
        if provider:
            print_kv("Provider", provider)
            # The tier the session will start on when it isn't Claude Code's own opus default.
            if forwarded_model:
                print_kv("Model", forwarded_model)
            elif route_root_model:
                print_kv("Model", route_root_model)
            # Gemini pins a concrete target under a provider (held in resolved_model).
            elif resolved_model:
                print_kv("Model", resolved_model)
        elif forwarded_model:
            print_kv("Model", forwarded_model)
        elif model and tool == "claude":
            # Claude's --model is pinned via the family aliases, not resolved_model/route_root_model.
            print_kv("Model", model)
        elif route_root_model:
            print_kv("Model", route_root_model)
        elif resolved_model:
            print_kv("Model", resolved_model)
        if tool in CAN_USE_CACHED_CONFIG_AGENTS and smart_routing_enabled and not provider:
            print_kv("Smart routing", "enabled")
            print_note(
                f"{TOOL_SPECS[tool]['display']} may require one-time hook review. Open "
                "`/hooks` and trust the ug routing hooks if prompted."
            )
        if tool in ("gemini", "opencode", "copilot", "pi"):
            print_note(
                f"{TOOL_SPECS[tool]['display']} token refresh is managed automatically "
                f"every 30 minutes while the session is running."
            )
        if recommendation is not None:
            _print_budget_panel(recommendation, tool, managed)
        # Register the managed config's MCP servers so they reach the agent's `/mcp` list. Nothing
        # else on this path does it — the config only lists them — so without this a
        # workspace-published server never shows up. Skipped on --dry-run, which writes nothing.
        if managed is not None and not is_dry_run():
            _register_managed_mcp_servers(managed, tool, state)
            _download_managed_skills(managed, state)
        if tool == "claude":
            if smart_routing_v2.enabled():
                # Transient launch precedence for the v2 PTY's initial --model flag.
                # An explicit choice wins, followed by a routed/managed root pick;
                # neither value is persisted into workspace state.
                launch_model = model or route_root_model
                if launch_model:
                    state["_claude_launch_model"] = launch_model
            if provider:
                state["_claude_launch_provider"] = provider
        launch_options = _launch_options(
            tool,
            ctx.args,
            smart_routing_enabled=smart_routing_enabled,
            explicit_prompt=explicit_prompt,
            model=model or (route_root_model if tool == "claude" else None),
            provider=provider,
        )
        print_success(f"Starting {TOOL_SPECS[tool]['display']}")
        launch_agent(tool, state, ctx.args, options=launch_options)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


# Launch-only escape hatch for managed/headless launchers (e.g. omnigent) that
# have already run `ug configure`: skip the ~5-10s per-launch auth + AI
# Gateway re-validation. Distinct from the configure-only `--skip-validate`,
# which skips the model smoke test.
SkipPreflightOption = Annotated[
    bool,
    typer.Option(
        "--skip-preflight",
        help="Skip the per-launch Databricks auth + AI Gateway re-validation, trusting a "
        "prior `ug configure`.",
    ),
]

REFRESH_HELP = (
    "Refresh Databricks auth, gateway, models, managed config, and agent configuration before "
    "launching."
)

# Target this launch at a specific workspace, auto-configuring (and logging in)
# if it hasn't been set up yet — so a launch needs no prior `ug configure`.
WorkspaceOption = Annotated[
    str | None,
    typer.Option(
        "--workspace",
        help="Databricks workspace URL to launch against; sets up and authenticates it "
        "if not already configured.",
    ),
]


_PROMPT_SUFFIX_KEY = "ucode_explicit_prompt_suffix"


class _PromptAwareCommand(TyperCommand):
    """Record an agent's ``--`` prompt separator before Click removes it."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        try:
            separator = args.index("--")
        except ValueError:
            pass
        else:
            ctx.meta[_PROMPT_SUFFIX_KEY] = tuple(args[separator + 1 :])
        return super().parse_args(ctx, args)


def _has_explicit_prompt(ctx: typer.Context) -> bool:
    suffix = ctx.meta.get(_PROMPT_SUFFIX_KEY)
    if not isinstance(suffix, tuple):
        return False
    suffix_args = list(suffix)
    return ctx.args == suffix_args and len(suffix_args) <= 1


@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the ug version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print config files without writing them. Uses the last saved managed "
            "config instead of fetching a fresh one.",
        ),
    ] = False,
    skip_preflight: SkipPreflightOption = False,
    workspace: WorkspaceOption = None,
) -> None:
    """Configure and launch coding agents through Databricks AI Gateway.

    The primary command is `ug`; `ucode` remains supported as an alias.

    With no subcommand, launches the agent your workspace's managed config selects.
    """
    if ctx.invoked_subcommand is not None:
        return
    set_dry_run(dry_run)
    try:
        _launch_managed_default(
            ctx, dry_run=dry_run, skip_preflight=skip_preflight, workspace=workspace
        )
    except typer.Exit:
        # `typer.Exit` subclasses RuntimeError, so it has to be re-raised ahead of the handler
        # below. Otherwise a launch that already reported its own error is followed by
        # `print_err(str(exc))` printing the exit code — a bare, meaningless "ERROR 1".
        raise
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


def _launch_managed_default(
    ctx: typer.Context,
    *,
    dry_run: bool,
    skip_preflight: bool,
    workspace: str | None,
) -> None:
    """Route bare ``ucode`` by whether the workspace publishes a managed config."""
    if workspace:
        set_current_workspace(normalize_workspace_url(workspace))
    state = load_state()
    current = state.get("workspace")
    if not current:
        console.print(ctx.get_help())
        return
    install_databricks_cli()
    apply_pat_environment(state)
    coding_agent_config_feature_disabled = False
    if dry_run:
        managed = load_managed_state(current)
    else:
        with spinner("Loading..."):
            managed, coding_agent_config_feature_disabled = refresh_managed_config(state)
    if coding_agent_config_feature_disabled:
        print_note(
            "Run `ug configure` to set up your coding agents, then launch one with "
            "`ug <agent>` (for example `ug claude`)."
        )
        return
    if not managed:
        _print_no_managed_config_guidance(current, state.get("profile"))
        return
    # The budget tier can move the org to a cheaper agent, so it outranks the config's
    # default_agent. Fetched here and handed to _launch_tool so it is read once per launch.
    recommendation = _fetch_budget_recommendation(state, managed)
    tool = recommended_agent(recommendation, managed) or next(
        iter(managed.get("enabled_agents") or {}), None
    )
    if not isinstance(tool, str) or not tool:
        raise RuntimeError(
            "Your workspace's managed config names no agent to launch. Ask an admin to set a "
            "default agent, or run `ug <agent>` directly."
        )
    _print_managed_summary(managed, state, tool, abridged=True)
    _launch_tool(
        tool,
        ctx,
        skip_preflight=skip_preflight,
        workspace_url=workspace,
        managed=managed,
        recommendation=recommendation,
    )


def _print_no_managed_config_guidance(workspace: str, profile: str | None) -> None:
    """Tell an admin how to publish a config, and everyone else who to ask."""
    print_warning(
        "No managed coding agent config was found for this workspace; using your local settings."
    )
    try:
        token = get_databricks_token(workspace, profile)
    except RuntimeError:
        return
    with spinner("Checking your workspace permissions..."):
        is_admin = is_workspace_admin(workspace, token)
    if is_admin is False:
        print_note("Ask a workspace admin to set one up with `ug setup`.")
    else:
        # None means the admin check itself failed; point at setup rather than a dead end.
        print_note("Run `ug setup` to configure one for your workspace, then `ug publish`.")


@app.command(
    "codex",
    cls=_PromptAwareCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def codex_cmd(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Route through a Unity Catalog Model Provider Service "
            "(<catalog>.<schema>.<name>). Skips Databricks model pinning; pass "
            "before any `--` separator.",
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help=REFRESH_HELP,
        ),
    ] = False,
    skip_preflight: SkipPreflightOption = False,
    workspace: WorkspaceOption = None,
    client_id: Annotated[
        str | None,
        typer.Option("--client-id", hidden=True, help="Custom OAuth client ID for Codex auth."),
    ] = None,
    redirect_url: Annotated[
        str | None,
        typer.Option(
            "--redirect-url", hidden=True, help="Custom OAuth callback URL for Codex auth."
        ),
    ] = None,
    scopes: Annotated[
        str | None,
        typer.Option("--scopes", hidden=True, help="Comma-separated custom OAuth scopes."),
    ] = None,
    enable_smart_routing_flag: Annotated[
        bool,
        typer.Option(
            "--enable-smart-routing",
            help="Enable AI Gateway model routing for Codex sessions and subagents.",
        ),
    ] = False,
    disable_smart_routing_flag: Annotated[
        bool,
        typer.Option(
            "--disable-smart-routing",
            hidden=True,
            help="Disable smart routing and remove ug's Codex routing hooks.",
        ),
    ] = False,
) -> None:
    """Launch Codex via Databricks."""
    try:
        custom_oauth = _custom_oauth_config(client_id, redirect_url, scopes)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from exc
    if enable_smart_routing_flag and disable_smart_routing_flag:
        print_err("Use only one of --enable-smart-routing or --disable-smart-routing.")
        raise typer.Exit(1)
    if disable_smart_routing_flag:
        codex_agent.disable_smart_routing(load_state())
        print_success("Codex smart routing disabled; ug routing hooks removed")
        return
    with _smart_routing_v2_flag(enable_smart_routing_flag):
        _launch_tool(
            "codex",
            ctx,
            provider=provider,
            refresh=refresh,
            skip_preflight=skip_preflight,
            workspace_url=workspace,
            custom_oauth=custom_oauth,
        )


@app.command(
    "claude",
    cls=_PromptAwareCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def claude_cmd(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Route through a Unity Catalog Model Provider Service "
            "(<catalog>.<schema>.<name>). Skips Databricks model pinning; pass "
            "before any `--` separator.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Launch on a specific Databricks model id (e.g. a UC "
            "`<catalog>.<schema>.<name>`). Pinned via ANTHROPIC_MODEL so the gateway "
            "resolves it — unlike Claude Code's own --model, which rejects non-catalog ids. "
            "With --provider, pass a family (opus/sonnet/haiku) or a target the service allows to "
            "start on that tier instead of Claude Code's opus default. Pass before any `--` separator.",
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help=REFRESH_HELP,
        ),
    ] = False,
    skip_preflight: SkipPreflightOption = False,
    workspace: WorkspaceOption = None,
    client_id: Annotated[
        str | None,
        typer.Option("--client-id", hidden=True, help="Custom OAuth client ID for apiKeyHelper."),
    ] = None,
    redirect_url: Annotated[
        str | None,
        typer.Option(
            "--redirect-url", hidden=True, help="Custom OAuth callback URL for apiKeyHelper."
        ),
    ] = None,
    scopes: Annotated[
        str | None,
        typer.Option("--scopes", hidden=True, help="Comma-separated custom OAuth scopes."),
    ] = None,
    enable_model_discovery: Annotated[
        bool,
        typer.Option(
            "--enable-model-discovery",
            hidden=True,
            help="Enable AI Gateway models in Claude Code's model picker.",
        ),
    ] = False,
    enable_smart_routing_flag: Annotated[
        bool,
        typer.Option(
            "--enable-smart-routing",
            help="Enable AI Gateway model routing for Claude Code sessions and subagents.",
        ),
    ] = False,
    disable_smart_routing_flag: Annotated[
        bool,
        typer.Option(
            "--disable-smart-routing",
            hidden=True,
            help="Disable smart routing and remove ug's Claude Code routing hooks.",
        ),
    ] = False,
) -> None:
    """Launch Claude Code via Databricks."""
    try:
        custom_oauth = _custom_oauth_config(client_id, redirect_url, scopes)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from exc
    if enable_smart_routing_flag and disable_smart_routing_flag:
        print_err("Use only one of --enable-smart-routing or --disable-smart-routing.")
        raise typer.Exit(1)
    if disable_smart_routing_flag:
        claude_agent.disable_smart_routing(load_state())
        print_success("Claude Code smart routing disabled; ug routing hooks removed")
        return
    if enable_model_discovery:
        os.environ[claude_agent.GATEWAY_MODEL_DISCOVERY_ENV_VAR] = "1"
    with _smart_routing_v2_flag(enable_smart_routing_flag):
        _launch_tool(
            "claude",
            ctx,
            provider=provider,
            model=model,
            refresh=refresh,
            skip_preflight=skip_preflight,
            workspace_url=workspace,
            custom_oauth=custom_oauth,
        )


@app.command("gemini", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def gemini_cmd(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Route through a Unity Catalog Model Provider Service "
            "(<catalog>.<schema>.<name>) that serves a Gemini model. Pass before any "
            "`--` separator.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model to launch on. Under --provider, selects which of the service's "
            "target models to use. Pass before any `--` separator.",
        ),
    ] = None,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Launch Gemini CLI via Databricks."""
    _launch_tool("gemini", ctx, provider=provider, model=model, skip_preflight=skip_preflight)


@app.command(
    "opencode", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def opencode_cmd(
    ctx: typer.Context,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Launch OpenCode via Databricks."""
    _launch_tool("opencode", ctx, skip_preflight=skip_preflight)


@app.command("copilot", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def copilot_cmd(
    ctx: typer.Context,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Launch GitHub Copilot CLI via Databricks."""
    _launch_tool("copilot", ctx, skip_preflight=skip_preflight)


@app.command("pi", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def pi_cmd(
    ctx: typer.Context,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Launch Pi coding agent via Databricks."""
    _launch_tool("pi", ctx, skip_preflight=skip_preflight)


@app.command("cursor", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cursor_cmd(ctx: typer.Context) -> None:
    """Launch Cursor Agent.

    Cursor is MCP-only: `cursor-agent` runs models on your own Cursor account, so
    ug configures no models for it. Its Databricks MCP servers (added via
    `ug configure mcp`) run `ug mcp-proxy`, which authenticates itself — so
    this command is a thin convenience wrapper over `cursor-agent`, kept for
    symmetry with the other `ug <agent>` launchers.
    """
    from ucode.agents import cursor

    try:
        if not shutil.which(cursor.CURSOR_BINARY):
            raise RuntimeError(
                f"`{cursor.CURSOR_BINARY}` was not found on PATH. Install Cursor Agent "
                "(https://cursor.com/cli), then re-run `ug cursor`."
            )
        print_section("Unity Gateway with Cursor")
        print_note(
            "Cursor runs models on your Cursor account; its Databricks MCP servers "
            "authenticate through `ug mcp-proxy`."
        )
        print_success("Starting Cursor Agent")
        cursor.launch(load_state(), ctx.args)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.callback(invoke_without_command=True)
def configure(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print config files without writing them.")
    ] = False,
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Configure only the named agent (e.g. claude, codex, gemini, opencode, copilot, pi).",
        ),
    ] = None,
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents",
            help="Configure a comma-separated list of agents without prompting (e.g. claude,codex).",
        ),
    ] = None,
    workspaces: Annotated[
        str | None,
        typer.Option(
            "--workspaces",
            help="Configure a comma-separated list of workspaces without prompting.",
        ),
    ] = None,
    profiles: Annotated[
        str | None,
        typer.Option(
            "--profiles",
            help="Configure a comma-separated list of existing Databricks CLI profiles "
            "without the workspace prompt. Each profile's host from ~/.databrickscfg "
            "supplies the workspace URL. Auth behaves like --workspaces: OAuth login "
            "is forced unless --use-pat is also passed.",
        ),
    ] = None,
    use_pat: Annotated[
        bool,
        typer.Option(
            "--use-pat",
            help="Authenticate with the personal access token stored in "
            "~/.databrickscfg for the selected profile(s) instead of OAuth. "
            "Requires --profiles; no interactive login is run. Intended for "
            "CI / headless environments.",
        ),
    ] = False,
    client_id: Annotated[
        str | None,
        typer.Option("--client-id", hidden=True, help="Custom OAuth client ID for apiKeyHelper."),
    ] = None,
    redirect_url: Annotated[
        str | None,
        typer.Option(
            "--redirect-url",
            hidden=True,
            help="Custom OAuth callback URL for apiKeyHelper.",
        ),
    ] = None,
    scopes: Annotated[
        str | None,
        typer.Option(
            "--scopes",
            hidden=True,
            help="Comma-separated custom OAuth scopes for apiKeyHelper.",
        ),
    ] = None,
    skip_validate: Annotated[
        bool,
        typer.Option(
            "--skip-validate",
            help="Skip the post-configure validation step that sends a quick test "
            "message through each agent. Config files are still written with the "
            "freshly discovered models.",
        ),
    ] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option(
            "--skip-unavailable",
            help="With --agents, configure the agents that are available on the workspace "
            "and skip (with a warning) any that aren't, instead of failing the whole run. "
            "Useful in CI against heterogeneous workspaces — e.g. requesting "
            "claude,codex,pi where the workspace exposes no OpenAI models still "
            "configures claude and pi. Exits non-zero only if none are available.",
        ),
    ] = False,
    enable_fable: Annotated[
        bool | None,
        typer.Option(
            "--enable-fable/--disable-fable",
            help="Pin the premium Claude Fable family via ANTHROPIC_DEFAULT_FABLE_MODEL "
            "for Claude Code (opt-in; off by default). Only takes effect when the "
            "workspace's AI Gateway actually advertises a Claude Fable model. "
            "--disable-fable clears a prior opt-in. Omitting both keeps the "
            "workspace's existing setting. Passed on its own (no --agent/--agents), "
            "it configures Claude Code directly since Fable is Claude-only.",
        ),
    ] = None,
    enable_databricks_ai_tools: Annotated[
        bool | None,
        typer.Option(
            "--enable-databricks-ai-tools/--disable-databricks-ai-tools",
            help="Install Databricks AI Tools (skills + plugins that teach agents to use "
            "Databricks) for the configured agents. Installation is configure-only; pass "
            "--disable-databricks-ai-tools to opt out.",
        ),
    ] = None,
    mcp: Annotated[
        str | None,
        typer.Option(
            "--mcp",
            help="Also register the given Databricks MCP service(s) for the configured "
            "coding agents, in one command. Pass a comma-separated list of fully-qualified "
            "names like `system.ai.slack`. Combine with --agents to set up an agent and its "
            "MCP servers together (e.g. `--agents claude --mcp system.ai.slack`); use without "
            "--agents for MCP-only clients such as Cursor.",
        ),
    ] = None,
    tracing: Annotated[
        bool,
        typer.Option(
            "--tracing",
            help="Also enable MLflow tracing for the configured workspace(s).",
        ),
    ] = False,
    skip_upgrade: Annotated[
        bool,
        typer.Option(
            "--skip-upgrade",
            help="Don't prompt to upgrade already-installed agent CLIs to a newer version. "
            "Required updates (when an agent is below its minimum supported version) are "
            "still applied.",
        ),
    ] = False,
    verbose: Annotated[
        str,
        typer.Option(
            "--verbose",
            help="Output verbosity: 'normal' (default) renders decorative panels; "
            "'low' prints terse single-line status instead.",
        ),
    ] = "normal",
) -> None:
    """Configure workspace URL and AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    if verbose not in ("normal", "low"):
        print_err("--verbose must be one of: normal, low.")
        raise typer.Exit(2)
    set_dry_run(dry_run)
    set_verbosity(verbose)
    prompt_optional_updates = not skip_upgrade
    try:
        custom_oauth = _custom_oauth_config(client_id, redirect_url, scopes)
        if custom_oauth is not None and use_pat:
            raise RuntimeError("--client-id cannot be combined with --use-pat.")
        install_databricks_cli()
        if agent is not None and agents is not None:
            raise RuntimeError("Use either --agent or --agents, not both.")
        if workspaces is not None and profiles is not None:
            raise RuntimeError("Use either --workspaces or --profiles, not both.")
        if use_pat and profiles is None:
            raise RuntimeError(
                "--use-pat requires --profiles. Pass the PAT-backed Databricks CLI "
                "profile(s) explicitly, e.g. `ug configure --profiles DEFAULT --use-pat`."
            )
        # Skipping only has meaning against an explicit agent list: the interactive
        # picker already offers just the available agents, and --agent names a
        # single agent whose absence is the whole answer.
        if skip_unavailable and agents is None:
            raise RuntimeError(
                "--skip-unavailable requires --agents. It selects the available subset "
                "of an explicit agent list, e.g. `ug configure --agents claude,codex,pi "
                "--skip-unavailable`."
            )
        workspace_entries = _parse_workspaces_option(workspaces) if workspaces is not None else None
        if profiles is not None:
            workspace_entries = _parse_profiles_option(profiles)
        flag_driven_workspace = workspace_entries is not None
        # Only forward the opt-in flags when set so existing call expectations
        # (and defaults) stay unchanged for the common interactive path.
        skip_kwargs: dict = {}
        if use_pat:
            skip_kwargs["use_pat"] = True
        if skip_validate:
            skip_kwargs["skip_validate"] = True
        # Only forward the Fable opt-in when the user passed the flag; `None`
        # (neither flag given) lets configure_shared_state inherit the prior
        # workspace setting instead of clobbering it.
        if enable_fable is not None:
            skip_kwargs["fable_enabled"] = enable_fable
        # Fable is a Claude-only model family, so `--enable-fable`/`--disable-fable`
        # only makes sense for Claude Code. When passed on its own, implicitly
        # target claude instead of dropping into the interactive agent picker.
        if enable_fable is not None and agent is None and agents is None:
            agent = "claude"
        if enable_databricks_ai_tools is not None:
            skip_kwargs["databricks_ai_tools_enabled"] = enable_databricks_ai_tools
        if custom_oauth is not None:
            skip_kwargs["custom_oauth"] = custom_oauth
        # Set True only in the fully-interactive branch below; gates the optional
        # MCP setup prompt so flag-driven / scripted runs are never interrupted.
        fully_interactive = False
        combined_optional_setup = False
        if agent is not None:
            tool = normalize_tool(agent)
            install_tool_binary(
                tool,
                strict=True,
                update_existing=True,
                prompt_optional_updates=prompt_optional_updates,
            )
            if workspace_entries is None:
                configure_workspace_command(tool, **skip_kwargs)
            else:
                configure_workspace_command(
                    tool,
                    workspaces=workspace_entries,
                    **skip_kwargs,
                )
        elif agents is not None:
            # Cursor is MCP-only (no model routing), so it can't go through the
            # model-agent configure path. Split it out: model agents configure
            # normally; cursor only needs workspace state established here, and
            # its MCP servers are added separately via `ug configure mcp`
            # (which picks cursor up through MCP_ONLY_CLIENTS). If cursor is the
            # only agent, do a workspace-only configure so that later `configure
            # mcp` run has a current workspace to target.
            requested = [a.strip().lower() for a in agents.split(",") if a.strip()]
            wants_cursor = "cursor" in requested
            model_agent_names = ",".join(a for a in requested if a != "cursor")
            if model_agent_names:
                selected_tools = _parse_agents_option(model_agent_names)
                agents_kwargs = dict(skip_kwargs)
                if skip_unavailable:
                    agents_kwargs["skip_unavailable"] = True
                if workspace_entries is None:
                    configure_workspace_command(
                        selected_tools=selected_tools,
                        prompt_optional_updates=prompt_optional_updates,
                        **agents_kwargs,
                    )
                else:
                    configure_workspace_command(
                        selected_tools=selected_tools,
                        workspaces=workspace_entries,
                        prompt_optional_updates=prompt_optional_updates,
                        **agents_kwargs,
                    )
            elif wants_cursor:
                # Cursor-only: establish workspace state without the model picker.
                _configure_shared_workspace_states(
                    workspace_entries or [_prompt_for_configuration(None)],
                    tools=[],
                    force_login=not use_pat,
                    use_pat=use_pat,
                    custom_oauth=custom_oauth,
                )
            else:
                # Neither model agents nor cursor -> empty/invalid --agents list.
                _parse_agents_option(agents)
        elif mcp is not None:
            # MCP-only: `--mcp` without --agent(s) (e.g. Cursor, which isn't a
            # model agent, or adding MCP servers to an already-configured setup).
            # Configure just the workspace — no interactive agent picker — so the
            # `--mcp` registration below has a current workspace to target.
            if workspace_entries is None:
                workspace_entries = [_prompt_for_configuration(None)]
            _configure_shared_workspace_states(
                workspace_entries,
                tools=[],
                force_login=not use_pat,
                use_pat=use_pat,
                custom_oauth=custom_oauth,
            )
        else:
            # Tool binaries are installed after the user picks which agents
            # they want, in configure_workspace_command.
            combined_optional_setup = (
                not flag_driven_workspace and enable_databricks_ai_tools is None
            )
            if combined_optional_setup:
                skip_kwargs["offer_optional_setup"] = True
            if workspace_entries is None:
                configure_workspace_command(
                    prompt_optional_updates=prompt_optional_updates,
                    **skip_kwargs,
                )
            else:
                configure_workspace_command(
                    workspaces=workspace_entries,
                    prompt_optional_updates=prompt_optional_updates,
                    **skip_kwargs,
                )
            # Only the no-agent, no-workspace path is truly interactive (the user
            # picked agents/workspace via prompts); that's where we offer the MCP
            # step below. Flag-driven runs stay scriptable.
            fully_interactive = not flag_driven_workspace
        if tracing:
            # The workspaces were just configured, so enable tracing for them
            # directly instead of re-prompting. Fall back to the workspace that
            # `configure_workspace_command` made current (the interactive pick).
            tracing_workspaces: list[tuple[str, str | None]] | None = workspace_entries
            if tracing_workspaces is None:
                current = load_full_state().get("current_workspace")
                tracing_workspaces = (
                    [(current, None)] if isinstance(current, str) and current else None
                )
            if tracing_workspaces:
                configure_tracing_command(workspaces=tracing_workspaces)
        if mcp is not None:
            # The workspace + agents were just configured above, so the current
            # workspace state now lists the agents whose MCP configs we should
            # write. `--mcp` takes fully-qualified service names, which
            # `configure_mcp_command` locates and registers without a picker
            # (bare short names would need --location, which we don't accept here).
            services = {name.strip() for name in mcp.split(",") if name.strip()}
            if not services:
                raise RuntimeError(
                    "--mcp needs at least one fully-qualified MCP service name, e.g. "
                    "`--mcp system.ai.slack`."
                )
            bare = sorted(name for name in services if name.count(".") < 2)
            if bare:
                raise RuntimeError(
                    "--mcp names must be fully qualified `<catalog>.<schema>.<name>` "
                    f"(got: {', '.join(bare)}). Use `ug configure mcp` for the "
                    "interactive picker."
                )
            configure_mcp_command(services=services)
        if (
            fully_interactive
            and not combined_optional_setup
            and not dry_run
            and prompt_yes_no("Configure MCP servers now?")
        ):
            configure_mcp_command()
    except typer.Exit:
        # `typer.Exit` subclasses RuntimeError, so it has to be re-raised ahead of the handler
        # below. Otherwise a clean exit (e.g. `_reject_configure_under_managed_config` under a
        # managed config) is followed by `print_err(str(exc))` printing the exit code — a bare,
        # meaningless "ERROR 0".
        raise
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("mcp")
def configure_mcp(
    location: Annotated[
        str | None,
        typer.Option(
            "--location",
            help="Non-interactive: replace registered MCPs with exactly the services "
            "in the given Unity Catalog `<catalog>.<schema>` (e.g. `system.ai`) and "
            "exit without showing the picker. Any previously-registered MCPs outside "
            "this location are removed.",
        ),
    ] = None,
    services: Annotated[
        str | None,
        typer.Option(
            "--services",
            help="Configure exactly this comma-separated subset of MCP services (adding and "
            "removing to match) instead of a whole schema. Full names like `system.ai.github` "
            "work on their own; bare short names like `github` need --location to locate them. "
            "Omit --services to configure the whole --location schema; pass an empty string "
            "(with --location) to remove all.",
        ),
    ] = None,
) -> None:
    """Add Databricks MCP servers to installed coding tools."""
    # `--services` absent -> None (whole schema); present (even empty) -> the
    # explicit subset, so `--services ""` deselects everything.
    selected = None if services is None else {s.strip() for s in services.split(",") if s.strip()}
    try:
        configure_mcp_command(location=location, services=selected)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("skills")
def configure_skills(
    location: Annotated[
        str | None,
        typer.Option("--location", help="Comma-separated `<catalog>.<schema>` skill scopes."),
    ] = None,
    mcp: Annotated[
        bool,
        typer.Option("--mcp", help="Mutate the skills MCP connection instead of downloading."),
    ] = False,
    path: Annotated[
        str | None,
        typer.Option(
            "--path",
            help="(download) Existing absolute dir to download into; defaults to your home dir.",
        ),
    ] = None,
    skill: Annotated[
        str | None,
        typer.Option(
            "--skill",
            help="(download) Download only this comma-separated subset of skills (by "
            "securable name, e.g. `my-skill`) from the schema, instead of every skill. "
            "Requires a single --location; not valid with --mcp.",
        ),
    ] = None,
) -> None:
    """Configure Databricks Skills for your coding tools.

    When ``--location`` is not provided, registers the skills MCP connection with
    utility tools only.

    When ``--location`` is provided: with ``--mcp``, sets the connection's scope to
    exactly the listed schemas (no download); otherwise, downloads every skill in
    each schema to disk (under ``--path``, or your home dir when omitted) and
    registers the MCP connection with utility tools only. ``--skill`` narrows a
    download to a named subset of a single schema's skills (requires exactly one
    ``--location``).
    """
    try:
        locations = _parse_skill_locations(location)
        # `--skill` absent -> None (whole schema); present (even empty) -> the
        # explicit subset, so `--skill ""` downloads nothing.
        selected_skills = (
            None if skill is None else {s.strip() for s in skill.split(",") if s.strip()}
        )
        if mcp and path is not None:
            raise RuntimeError("--path is not valid with --mcp.")
        if mcp and selected_skills is not None:
            raise RuntimeError("--skill is not valid with --mcp; it only applies when downloading.")
        if path is not None and not locations:
            raise RuntimeError("--path only applies when downloading with --location.")
        if selected_skills is not None and not locations:
            raise RuntimeError("--skill only applies when downloading with --location.")
        if selected_skills is not None and len(locations) != 1:
            raise RuntimeError(
                f"--skill requires a single --location (got: {', '.join(locations)})."
            )
        if mcp or not locations:
            configure_skills_mcp_command(locations)
        else:
            configure_skills_download_command(locations, path=path, skills=selected_skills)
    except (RuntimeError, ValueError) as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("tracing")
def configure_tracing(
    disable: Annotated[
        bool, typer.Option("--disable", help="Turn off MLflow tracing for configured agents.")
    ] = False,
) -> None:
    """Send coding-session traces to an MLflow experiment in your workspace."""
    try:
        install_databricks_cli()
        configure_tracing_command(disable=disable)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@setup_app.callback(invoke_without_command=True)
def setup(
    ctx: typer.Context,
    from_file: Annotated[
        str | None,
        typer.Option(
            "--from-file",
            help="Skip the interactive flow and load a hand-written managed config (JSON, in "
            "ug's manifest shape) instead. Validated before it is saved.",
        ),
    ] = None,
) -> None:
    """Choose the agents and models for your workspace's managed config (admins only).

    MCP servers, skills, and the tiered spend policy have their own commands — see `ug setup help`.
    """
    if ctx.invoked_subcommand is not None:
        return
    # `typer.Exit` subclasses RuntimeError, so it must be raised outside the try — inside, the
    # `except RuntimeError` below would swallow it and report the exit code as an error message.
    try:
        install_databricks_cli()
        code = setup_command(from_file=from_file)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None
    if code:
        raise typer.Exit(code)


@setup_app.command("mcps")
def setup_mcp_cmd() -> None:
    """Choose the MCP servers the managed config gives developers (admins only)."""
    # Same `typer.Exit`/RuntimeError ordering trap as the `setup` callback above.
    try:
        install_databricks_cli()
        code = setup_mcp_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None
    if code:
        raise typer.Exit(code)


@setup_app.command("skills")
def setup_skills_cmd(
    location: Annotated[
        str | None,
        typer.Option(
            "--location",
            help="Skill schemas to publish as `<catalog>.<schema>` (comma-separated for several). "
            "Skips the prompt.",
        ),
    ] = None,
) -> None:
    """Choose the skills the managed config gives developers (admins only)."""
    try:
        install_databricks_cli()
        # None means "prompt"; an explicit `--location` is parsed to the list to publish.
        locations = None if location is None else _parse_skill_locations(location)
        code = setup_skills_command(locations)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None
    if code:
        raise typer.Exit(code)


@setup_app.command("spend-tiers")
def setup_budget_policy_cmd() -> None:
    """Route developers to cheaper agents as the workspace spends its budget (admins only)."""
    try:
        install_databricks_cli()
        code = setup_budget_policy_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None
    if code:
        raise typer.Exit(code)


@setup_app.command("help")
def setup_help_cmd() -> None:
    """Walk through the managed-config setup: every command, in order, and what's already done."""
    # No auth and no CLI install: this reads the local draft only, so it works before `ucode
    # configure` and on a machine without the Databricks CLI.
    try:
        code = setup_help_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    if code:
        raise typer.Exit(code)


@setup_app.command("show")
def setup_show_cmd() -> None:
    """Print the authored managed config and the payload `ug publish` would publish."""
    try:
        code = show_command()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    if code:
        raise typer.Exit(code)


@app.command("publish")
def publish_cmd(
    file_path: Annotated[
        str | None,
        typer.Option(
            "--file",
            "-f",
            help="Publish a config file exported with `ug export` instead of the locally "
            "authored config. Its `workspace` must match the configured workspace.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Publish without the confirmation prompt."),
    ] = False,
) -> None:
    """Publish this workspace's managed coding config (workspace admins only).

    Always validates the manifest before publishing (and shows what would change, then confirms), so
    there is no separate dry-run: `ug setup` only ever writes a valid manifest, and a
    hand-editing admin sees any error here before anything reaches the workspace.
    """
    # See the `setup` callback: `typer.Exit` subclasses RuntimeError, so it must be raised after
    # the try block or the handler below would report a successful exit as an error.
    try:
        install_databricks_cli()
        code = publish_command(file_path=file_path, yes=yes)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None
    if code:
        raise typer.Exit(code)


@app.command("export")
def export_cmd(
    file_path: Annotated[
        str | None,
        typer.Option(
            "--file",
            "-f",
            help="Write the exported config JSON to this file (atomically) instead of stdout. "
            "The parent directory must already exist.",
        ),
    ] = None,
) -> None:
    """Export this workspace's managed coding-agent config as portable JSON.

    Serializes the local managed config to the external `CodingAgentConfig` format that
    `ug publish -f <path>` consumes, with credentials and server-owned fields (resource name,
    workspace id, timestamps, user ids) excluded. Any user can run it; it makes no network calls
    and mutates no workspace or local state. Without --file the JSON is printed to stdout;
    diagnostics and errors go to stderr.
    """
    from ucode.managed_export import export_command

    try:
        export_command(file_path=file_path)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("status")
def status_cmd() -> None:
    """Show current workspace, tool configs, and saved model selections."""
    try:
        status()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("revert")
def revert_cmd() -> None:
    """Clear ug state and restore backed-up agent config files."""
    try:
        revert()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("doctor")
def doctor_cmd() -> None:
    """Diagnose the local ug setup and offer to fix any problems found."""
    from ucode.doctor import doctor

    try:
        doctor()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("usage")
def usage_cmd(
    warehouse_id: Annotated[
        str | None,
        typer.Option("--warehouse-id", help="SQL warehouse to query, instead of discovering one."),
    ] = None,
) -> None:
    """Show Databricks AI Gateway usage summary (last 7 days)."""
    try:
        install_databricks_cli()
        usage_report(warehouse_id=warehouse_id)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("upgrade")
def upgrade_cmd() -> None:
    """Upgrade ug to the latest version from GitHub."""
    legacy_distribution = "ucode"
    current_distribution = "unity-gateway"
    git_url = "git+https://github.com/databricks/unity-gateway"
    installed_distribution = _installed_cli_distribution()
    upgrade_requirement = f"{installed_distribution} @ {git_url}"
    migrated = False
    legacy_removed = False

    print_section("Upgrade")
    print_kv("Source", git_url)
    print_kv("Installed distribution", installed_distribution)
    try:
        result = subprocess.run(
            ["uv", "tool", "install", "--reinstall", upgrade_requirement],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if installed_distribution == legacy_distribution and _is_distribution_cutover(result):
                print_note(
                    "The package is now distributed as `unity-gateway`; migrating this installation."
                )
                subprocess.run(
                    ["uv", "tool", "uninstall", legacy_distribution],
                    check=True,
                )
                legacy_removed = True
                subprocess.run(
                    ["uv", "tool", "install", "--force", git_url],
                    check=True,
                )
                migrated = True
                installed_distribution = current_distribution
            else:
                detail = _upgrade_failure_detail(result)
                print_err(
                    f"Upgrade failed (exit code {result.returncode})"
                    f"{f': {detail}' if detail else '.'}"
                )
                print_note("The existing installation was left unchanged.")
                raise typer.Exit(1)

        if migrated:
            _verify_upgraded_commands()
    except typer.Exit:
        raise
    except FileNotFoundError:
        print_err("`uv` was not found on PATH. Install uv to upgrade ug.")
        raise typer.Exit(1) from None
    except subprocess.CalledProcessError as exc:
        if legacy_removed:
            print_err(
                "The legacy `ucode` tool was removed, but installing `unity-gateway` failed "
                f"(exit code {exc.returncode})."
            )
            print_note(f"Recover by running `uv tool install --force {git_url}`.")
        else:
            print_err(f"Upgrade failed (exit code {exc.returncode}); `ucode` was not removed.")
        raise typer.Exit(1) from None
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None

    if migrated:
        print_success("Migrated to `unity-gateway`; both `ug` and `ucode` are working")
    else:
        print_success(f"{installed_distribution} upgraded; both `ug` and `ucode` are working")


def _installed_cli_distribution() -> str:
    """Return the uv tool identity, preferring the post-cutover distribution."""
    for distribution_name in ("unity-gateway", "ucode"):
        try:
            metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        return distribution_name
    # Source checkouts and unusual installers may expose neither distribution.
    # The distribution remains named ucode until the Phase 3 cutover, so use the
    # non-destructive legacy upgrade path and let uv report an actionable error.
    return "ucode"


def _is_distribution_cutover(result: subprocess.CompletedProcess[str]) -> bool:
    """Recognize uv's specific error when source metadata changes distribution name."""
    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return all(
        marker in output
        for marker in (
            "metadata name",
            "unity-gateway",
            "does not match given name",
            "ucode",
        )
    )


def _upgrade_failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "").strip()


def _verify_upgraded_commands() -> None:
    """Ensure both compatibility entry points were installed and can start."""
    for command in ("ug", "ucode"):
        executable = shutil.which(command)
        if executable is None:
            raise RuntimeError(
                f"Upgrade completed, but `{command}` is not available on PATH. "
                "Reinstall Unity Gateway and ensure the uv tool bin directory is on PATH."
            )
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = _upgrade_failure_detail(result)
            raise RuntimeError(
                f"Upgrade completed, but `{command} --version` failed"
                f"{f': {detail}' if detail else '.'}"
            )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
