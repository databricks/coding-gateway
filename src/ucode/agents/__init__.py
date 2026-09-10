"""Per-agent modules + dispatch helpers.

Each `agents.<tool>` module owns its own config layout, overlay rendering,
config-file writer, default-model selection, launch logic, and validation
command. This `__init__` aggregates the registry and exposes uniform
dispatchers for the rest of the codebase.

Adding a new agent: create `agents/<name>.py` exposing `SPEC`, `write_tool_config`,
`default_model`, `launch`, `validate_cmd`. Then add an entry to `_MODULES`
below and to `TOOL_ALIASES` if needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from ucode.agent_updates import available_npm_package_update
from ucode.config_io import ToolSpec
from ucode.databricks import (
    get_databricks_token,
    install_ai_tools,
    install_databricks_cli,
    map_claude_family_models,
    resolve_provider_service,
)
from ucode.managed_files import managed_write_batch
from ucode.state import get_provider_service, load_state, save_state
from ucode.telemetry import agent_version
from ucode.ui import (
    console,
    is_low_verbosity,
    print_err,
    print_note,
    print_section,
    print_success,
    print_warning,
    prompt_yes_no,
    spinner,
)

from . import claude, codex, copilot, gemini, opencode, pi
from .args import LaunchOptions as LaunchOptions
from .args import explicit_model_arg_value as explicit_model_arg_value

_MODULES = {
    "codex": codex,
    "claude": claude,
    "gemini": gemini,
    "opencode": opencode,
    "copilot": copilot,
    "pi": pi,
}

TOOL_SPECS: dict[str, ToolSpec] = {name: module.SPEC for name, module in _MODULES.items()}


# Model-routing agents ucode configures end to end. Cursor is deliberately NOT
# here: it runs models on the user's own Cursor account, so `normalize_tool`
# rejects it and the model-config paths never see it. The `configure`/MCP flows
# handle "cursor" separately as an MCP-only client (see MCP_ONLY_CLIENTS).
TOOL_ALIASES = {
    "codex": "codex",
    "claude": "claude",
    "claude-code": "claude",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "opencode": "opencode",
    "copilot": "copilot",
    "pi": "pi",
}

DEFAULT_TOOL = "codex"
BUNDLE_VERSION = 1
_MANAGED_SETTINGS_TOOLS = {"claude", "codex"}
_NATIVE_UPGRADE_COMMANDS = {
    "claude": ["claude", "upgrade"],
    "codex": ["codex", "update"],
}

# ucode tool -> `databricks aitools` agent id. gemini/pi aren't supported.
AITOOLS_AGENT_TOKENS = {
    "claude": "claude-code",
    "codex": "codex",
    "opencode": "opencode",
    "copilot": "copilot",
}


def install_databricks_ai_tools_for_agents(tools: list[str], state: dict) -> None:
    """Install Databricks AI Tools for supported agents.

    Gemini and Pi have no ``aitools`` support and are dropped.
    """
    if state.get("databricks_ai_tools_enabled", True) is False:
        return
    agents = [AITOOLS_AGENT_TOKENS[tool] for tool in tools if tool in AITOOLS_AGENT_TOKENS]
    if not agents:
        return
    install_ai_tools(agents, state.get("profile"))


def normalize_tool(tool: str) -> str:
    normalized = TOOL_ALIASES.get(tool.strip().lower())
    if not normalized:
        raise RuntimeError(
            f"Unsupported tool '{tool}'. Use one of: codex, claude, gemini, opencode, copilot, pi."
        )
    return normalized


def _update_installed_tool_binary(tool: str, version: str | None = None) -> bool:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    package = spec["package"]
    target = f"{package}@{version}" if version else package

    if tool in _NATIVE_UPGRADE_COMMANDS and version is None and shutil.which(binary):
        command = _NATIVE_UPGRADE_COMMANDS[tool]
    else:
        if not shutil.which("npm"):
            print_warning(f"`npm` is not available to update {spec['display']}; continuing.")
            return False
        command = ["npm", "install", "-g", target]

    print_note(f"Upgrading {spec['display']}...")
    try:
        subprocess.run(command, check=True, timeout=300)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print_warning(f"Could not update {spec['display']}; continuing.")
        return False

    print_success(f"{spec['display']} is up to date")
    agent_version.cache_clear()
    return bool(shutil.which(binary))


def _minimum_version_error(tool: str) -> str | None:
    checker = getattr(_MODULES[tool], "minimum_version_error", None)
    if not callable(checker):
        return None
    return checker()


def _too_new_downgrade(tool: str) -> tuple[str, str] | None:
    """Return (installed_version, downgrade_target) when the installed tool is
    too new to work, or None. Agents opt in by defining `too_new_downgrade`."""
    checker = getattr(_MODULES[tool], "too_new_downgrade", None)
    if not callable(checker):
        return None
    return checker()


def _maybe_downgrade_too_new_tool(tool: str, *, prompt: bool) -> bool:
    """Warn when the installed tool exceeds its supported version and offer to
    downgrade to the latest working release. Returns True when the tool was too
    new (regardless of whether the client accepted the downgrade).

    Unlike a required *upgrade*, a too-new build may still launch (it just
    misbehaves), so we never force the change — we warn and, when prompting is
    enabled, let the client press `y` to downgrade.
    """
    downgrade = _too_new_downgrade(tool)
    if not downgrade:
        return False
    spec = TOOL_SPECS[tool]
    installed, target = downgrade
    print_warning(
        f"{spec['display']} {installed} is newer than the latest version known to work "
        f"with the Databricks AI Gateway ({target})."
    )
    if prompt and prompt_yes_no(f"Downgrade {spec['display']} from {installed} to {target}?"):
        _update_installed_tool_binary(tool, version=target)
    return True


def install_tool_binary(
    tool: str,
    *,
    strict: bool = True,
    update_existing: bool = False,
    prompt_optional_updates: bool = True,
) -> bool:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    package = spec["package"]

    if shutil.which(binary):
        # A too-new build is a correctness blocker (the tool runs but misbehaves
        # against the gateway), so check it on every launch — not just when
        # auto-configuring — mirroring the minimum-version gate below.
        too_new = _maybe_downgrade_too_new_tool(tool, prompt=prompt_optional_updates)
        version_error = _minimum_version_error(tool)

        should_update = update_existing or tool in _NATIVE_UPGRADE_COMMANDS
        if should_update and not too_new and version_error:
            print_warning(version_error)
            if (
                tool in _NATIVE_UPGRADE_COMMANDS
                and prompt_optional_updates
                and not prompt_yes_no(f"Upgrade {spec['display']} if available?")
            ):
                raise RuntimeError(version_error)
            if not _update_installed_tool_binary(tool):
                raise RuntimeError(version_error)
            version_error = _minimum_version_error(tool)
        if version_error:
            raise RuntimeError(version_error)
        return True

    if not shutil.which("npm"):
        message = f"`{binary}` is not installed and npm is not available to install it."
        if strict:
            raise RuntimeError(message)
        print_warning(message)
        return False

    print_section("Bootstrap")
    print_warning(f"`{binary}` was not found. Installing {spec['display']}...")
    try:
        subprocess.run(["npm", "install", "-g", package], check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        message = f"Failed to install {spec['display']} automatically."
        if strict:
            raise RuntimeError(message) from exc
        print_warning(f"{message} Continuing without it.")
        return False

    if not shutil.which(binary):
        message = f"{spec['display']} install completed, but `{binary}` is still not on PATH."
        if strict:
            raise RuntimeError(message)
        print_warning(f"{message} Continuing without it.")
        return False

    return True


def ensure_tool_binary_available(tool: str) -> None:
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    if shutil.which(binary):
        return
    raise RuntimeError(
        f"{spec['display']} is not installed (`{binary}` was not found on PATH). "
        f"Install it with `npm install -g {spec['package']}` or run "
        f"`ucode configure` to try automatic installation."
    )


def tool_binary_installed(tool: str) -> bool:
    """True when the agent's CLI binary is on PATH. Read-only — for ``ucode doctor``."""
    return bool(shutil.which(TOOL_SPECS[tool]["binary"]))


def tool_update_available(tool: str) -> tuple[str, str] | None:
    """Return ``(current, latest)`` when a newer agent CLI is published, else None.
    Read-only wrapper over the npm update check — for ``ucode doctor``."""
    if tool in _NATIVE_UPGRADE_COMMANDS:
        return None
    checker = getattr(_MODULES[tool], "is_update_available", None)
    if callable(checker):
        return checker()
    return available_npm_package_update(TOOL_SPECS[tool]["package"])


def update_tool_binary(tool: str) -> bool:
    """Install the latest agent CLI, returning True on success. Public entry
    point over the internal updater so ``ucode doctor`` can apply the fix."""
    return _update_installed_tool_binary(tool)


def tool_uses_native_updater(tool: str) -> bool:
    """Whether upgrades are resolved and installed entirely by the agent CLI."""
    return tool in _NATIVE_UPGRADE_COMMANDS


def tool_version_error(tool: str) -> str | None:
    """Return an active minimum-version blocker for a configured agent."""
    return _minimum_version_error(tool)


def tracing_mlflow_ok() -> bool:
    """True when the `mlflow` CLI that Claude tracing needs is installed and in
    the supported version range. Read-only — for ``ucode doctor``."""
    current = claude._installed_mlflow_version()
    return bool(
        current and claude.MINIMUM_MLFLOW_VERSION <= current < claude.MAXIMUM_MLFLOW_VERSION
    )


def ensure_tracing_mlflow_cli() -> bool:
    """Install/repair the pinned `mlflow` CLI for Claude tracing, returning True
    on success. Public entry point so ``ucode doctor`` can apply the fix."""
    return claude._ensure_mlflow_cli()


def ensure_bootstrap_dependencies(
    tool: str,
    *,
    update_existing: bool = False,
    prompt_optional_updates: bool = True,
) -> None:
    install_databricks_cli()
    install_tool_binary(
        tool,
        strict=True,
        update_existing=update_existing,
        prompt_optional_updates=prompt_optional_updates,
    )


def default_model_for_tool(tool: str, state: dict) -> str | None:
    return _MODULES[tool].default_model(state)


def resolve_launch_model(
    tool: str,
    state: dict,
    explicit_model: str | None,
) -> tuple[dict, str | None]:
    model = explicit_model or default_model_for_tool(tool, state)
    # if model is not specified for codex, then launch with harness's default model.
    if not model and tool != "codex":
        raise RuntimeError(
            f"No models available for {tool}. Run `ucode configure` to set up your workspace."
        )
    return state, model


def resolve_provider_models(
    tool: str, state: dict, provider: str | None
) -> tuple[dict | None, str | None, bool]:
    """Validate ``provider`` for ``tool`` and return the model ids to pin.

    Returns ``(provider_models, error, relayed)``. ``provider_models`` is a ``{family: model_id}``
    dict re-derived from the service's live targets for a non-relayed claude service — both Bedrock
    (provider-side slugs) and API-key Anthropic (canonical ids) — so the client sends exactly the ids
    the MPS allows rather than Claude Code's defaults, which may not match the declared targets. It is
    None when ``provider`` is None, for a relayed subscription (see below), or for a non-Claude (e.g.
    codex) service. ``relayed`` is True for a credential-less Anthropic subscription relay, which the
    launch path wires with the relayed overlay + refresh proxy. A non-None ``error`` means the
    provider is invalid for the tool and the caller should not launch.

    This is the *developer-configured* path (``ucode configure`` then ``ucode claude``). The *managed*
    path pins from the admin's authored manifest slots instead — see
    ``managed_resolve.managed_provider_family_models`` and its launch call site — so an admin's chosen
    versions win rather than being re-derived here.
    """
    if not provider:
        return None, None, False
    token = get_databricks_token(state["workspace"], state.get("profile"))
    service, error = resolve_provider_service(tool, provider, state["workspace"], token)
    if error or service is None:
        return None, error, False
    relayed = bool(service.get("relayed"))
    # Relayed (Claude Max/Enterprise subscription) is exempt: the gateway disables
    # model selection server-side for that tier, so there's nothing to reconcile.
    if relayed:
        return None, None, relayed
    # Only Claude pins per-family model ids. Codex ignores this map, and gemini resolves
    # its target through resolve_gemini_provider_model instead — so mapping their targets
    # through Claude-family logic would be meaningless (see docstring).
    if tool != "claude":
        return None, None, relayed
    return map_claude_family_models(service.get("targets") or []) or None, None, relayed


def resolve_gemini_provider_model(
    state: dict,
    provider: str,
    explicit_model: str | None,
    *,
    service: dict | None = None,
) -> tuple[str | None, str | None]:
    """Pick the Gemini model to pin for a provider-service launch.

    A Gemini Enterprise service routes by header but the request still names a
    concrete model in the URL, so one of the service's declared targets must be
    pinned. In precedence order: ``explicit_model`` (from ``--model``) when it
    names a target; the model already pinned in the env file when it is still a
    target (so a bare relaunch or reconfigure needn't re-pass ``--model``); the
    sole target when the service declares exactly one; otherwise ask the user to
    choose. Returns ``(model, error)``.

    Pass ``service`` to reuse an already-fetched service dict and skip the
    control-plane lookup (the launch/configure paths hold one).
    """
    if service is None:
        token = get_databricks_token(state["workspace"], state.get("profile"))
        service, error = resolve_provider_service("gemini", provider, state["workspace"], token)
        if error or service is None:
            return None, error or f"Model provider service '{provider}' was not found."
    targets = [t for t in (service.get("targets") or []) if isinstance(t, str) and t]
    if explicit_model:
        if explicit_model in targets:
            return explicit_model, None
        available = ", ".join(targets) or "none"
        return None, (
            f"Model '{explicit_model}' is not a target of provider service '{provider}'. "
            f"Available: {available}."
        )
    if not targets:
        return None, f"Provider service '{provider}' exposes no models to launch."
    # Reuse a previously pinned target so a bare relaunch/reconfigure keeps working without
    # --model — but only when it is still one of the service's declared targets.
    persisted = gemini.persisted_provider_model()
    if persisted in targets:
        return persisted, None
    if len(targets) == 1:
        return targets[0], None
    return None, (
        f"Provider service '{provider}' exposes several models "
        f"({', '.join(targets)}); pass --model to choose one."
    )


def configure_tool(
    tool: str,
    state: dict,
    model: str | None = None,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
    relayed: bool = False,
    route_root_model: str | None = None,
    custom_model: str | None = None,
    coding_agent_config_defaults: dict[str, str] | None = None,
) -> dict:
    result: dict | tuple[dict, str]
    if tool == "codex":
        result = codex.write_tool_config(state, model, provider=provider)
    elif tool == "claude":
        # A Model Provider Service routes by header and pins no Databricks
        # model, so the usual "model required" guard doesn't apply to claude.
        if not model and not provider:
            raise RuntimeError(f"A {tool} model must be selected before configuration.")
        result = claude.write_tool_config(
            state,
            model,
            provider=provider,
            provider_models=provider_models,
            relayed=relayed,
            route_root_model=route_root_model,
            custom_model=custom_model,
            coding_agent_config_defaults=coding_agent_config_defaults,
        )
    else:
        # Every tool in this branch needs a model — including gemini under a provider,
        # which still pins the service's target model in the URL.
        if not model:
            raise RuntimeError(f"A {tool} model must be selected before configuration.")
        if tool == "gemini":
            result = gemini.write_tool_config(state, model, provider=provider)
        elif tool == "copilot":
            result = copilot.write_tool_config(state, model)
        elif tool == "pi":
            result = pi.write_tool_config(state, model)
        else:
            result = opencode.write_tool_config(state, model)
    # gemini/opencode/copilot/pi return (state, token); codex/claude return state
    if isinstance(result, tuple):
        return result[0]
    return result


def launch(
    tool: str,
    state: dict,
    tool_args: list[str],
    *,
    options: LaunchOptions,
) -> None:
    _MODULES[tool].launch(state, tool_args, options=options)


def check_gateway_endpoint(state: dict, tool: str) -> bool:
    """V2-only: a tool is available iff we discovered models for it."""
    if tool == "claude":
        return bool(state.get("claude_models"))
    if tool == "opencode":
        return bool(state.get("opencode_models"))
    if tool == "codex":
        return bool(state.get("codex_models"))
    if tool == "gemini":
        return bool(state.get("gemini_models"))
    if tool == "copilot":
        return bool(state.get("claude_models")) or bool(state.get("codex_models"))
    if tool == "pi":
        return (
            bool(state.get("claude_models"))
            or bool(state.get("codex_models"))
            or bool(state.get("gemini_models"))
        )
    return False


_TOOL_DISCOVERY_SOURCES: dict[str, tuple[str, ...]] = {
    "claude": ("claude",),
    "opencode": ("claude", "gemini", "oss"),
    "codex": ("codex",),
    "gemini": ("gemini",),
    "copilot": ("claude", "codex"),
    "pi": ("claude", "codex", "gemini"),
}


def _availability_failure_detail(tool: str, state: dict) -> str:
    reasons = state.get("_discovery_reasons") or {}
    if not reasons:
        return ""
    sources = _TOOL_DISCOVERY_SOURCES.get(tool, ())
    parts = [f"{source} discovery: {reasons[source]}" for source in sources if reasons.get(source)]
    if not parts:
        return ""
    return " (" + "; ".join(parts) + ")"


def configure_single_tool(tool: str, state: dict) -> dict:
    """Check availability, configure, and persist state for one tool only."""
    provider = get_provider_service(state, tool)
    # A Model Provider Service routes through the same gateway and pins no
    # Databricks model, so the per-tool model availability check doesn't apply.
    if not provider:
        with spinner(f"Checking {TOOL_SPECS[tool]['display']} availability..."):
            ok = check_gateway_endpoint(state, tool)
        if not ok:
            detail = _availability_failure_detail(tool, state)
            raise RuntimeError(
                f"{TOOL_SPECS[tool]['display']} is not available on this workspace.{detail}"
            )
    with managed_write_batch(_managed_settings_displays([tool])):
        state = _configure_one(tool, state, provider)
    available_tools = list(set((state.get("available_tools") or []) + [tool]))
    state["available_tools"] = available_tools
    save_state(state)
    return state


def _configure_one(tool: str, state: dict, provider: str | None) -> dict:
    """Write one tool's config, routing through ``provider`` when set."""
    if provider:
        if tool == "gemini":
            # Gemini pins a concrete target in the URL, so configure must resolve one now —
            # unlike claude/codex, its config writer requires a model. This also validates the
            # service in a single lookup (no resolve_provider_models family map for gemini).
            model, error = resolve_gemini_provider_model(state, provider, None)
            if error:
                raise RuntimeError(error)
            return configure_tool(tool, state, model, provider=provider)
        provider_models, error, relayed = resolve_provider_models(tool, state, provider)
        if error:
            raise RuntimeError(error)
        return configure_tool(
            tool, state, None, provider=provider, provider_models=provider_models, relayed=relayed
        )
    if tool == "codex":
        return configure_tool("codex", state)
    state, model = resolve_launch_model(tool, state, None)
    return configure_tool(tool, state, model)


def configure_selected_tools(
    state: dict, tools: list[str], *, install_ai_tools: bool = True
) -> dict:
    """Configure the given tools. Caller is responsible for ensuring each tool
    is available on the workspace.

    Merges newly-configured tools into state['available_tools'] rather than
    replacing it, so a previously-configured tool the user didn't pick this
    run is preserved.
    """
    with managed_write_batch(_managed_settings_displays(tools)):
        for tool in tools:
            state = _configure_one(tool, state, get_provider_service(state, tool))

    existing = state.get("available_tools") or []
    state["available_tools"] = sorted(set(existing) | set(tools))
    save_state(state)
    if install_ai_tools:
        install_databricks_ai_tools_for_agents(tools, state)
    return state


def _managed_settings_displays(tools: list[str]) -> list[str]:
    return [TOOL_SPECS[tool]["display"] for tool in tools if tool in _MANAGED_SETTINGS_TOOLS]


def configure_all_tools(state: dict) -> dict:
    """Discover available tools on the workspace and configure all of them.

    Thin wrapper retained for callers that want the legacy "configure
    everything that works" behavior.
    """
    available_tools: list[str] = []
    unavailable_tools: list[str] = []

    for tool in TOOL_SPECS:
        with spinner(f"Checking {TOOL_SPECS[tool]['display']} availability..."):
            ok = check_gateway_endpoint(state, tool)
        if ok:
            available_tools.append(tool)
        else:
            unavailable_tools.append(tool)

    for tool in unavailable_tools:
        print_err(f"{TOOL_SPECS[tool]['display']} is not available on this workspace")

    return configure_selected_tools(state, available_tools)


def ensure_provider_state(tool: str) -> dict:
    """Validate that workspace + tool are configured. Caller is expected to
    handle auth (typically via `configure_shared_state` immediately after)."""
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured. Run `ucode configure` first.")
    available_tools = state.get("available_tools") or []
    if tool not in available_tools:
        raise RuntimeError(
            f"{TOOL_SPECS[tool]['display']} is not available on this workspace. "
            f"Run `ucode configure` to set up your agents."
        )
    return state


def validate_tool(tool: str) -> tuple[bool, str]:
    """Invoke a tool with a simple prompt to verify it works. Returns (ok, error_msg)."""
    spec = TOOL_SPECS[tool]
    binary = spec["binary"]
    module = _MODULES[tool]
    # Some configs (e.g. claude relayed) can't be probed with a live message —
    # the proxy + subscription login only exist at launch. Trust the written config.
    if hasattr(module, "skip_validation") and module.skip_validation(load_state()):
        return True, ""
    cmd = module.validate_cmd(binary)
    env = None
    if hasattr(module, "validate_env"):
        try:
            env = module.validate_env(load_state())
        except RuntimeError:
            env = None
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True, ""
        output = (result.stderr or result.stdout or "").strip()
        for line in output.splitlines():
            if "error" in line.lower() and ("message" in line.lower() or ":" in line):
                msg = line.strip()
                if "error_code" in msg:
                    try:
                        payload = json.loads(msg[msg.index("{") : msg.rindex("}") + 1])
                        return False, payload.get("message", msg)
                    except (json.JSONDecodeError, ValueError):
                        pass
                return False, msg
        last_line = output.splitlines()[-1] if output else "unknown error"
        return False, last_line
    except OSError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired:
        return False, "timed out"


def provider_permission_error(tool: str, state: dict, err: str) -> str:
    """Rewrite the opaque gateway connection-permission failure into an
    actionable message naming the Model Provider Service the user must be
    granted access to. Returns ``err`` unchanged when it doesn't apply.
    """
    provider = get_provider_service(state, tool)
    if provider and "USE CONNECTION on SCHEMA_CONNECTION" in err:
        return (
            f"You don't have EXECUTE permission on the model provider service "
            f"'{provider}'. Ask its owner to grant you access, then re-run "
            f"`ucode configure`."
        )
    return err


def validate_all_tools(state: dict) -> None:
    from rich.panel import Panel  # local to avoid bumping module-level deps

    from ucode.agents.pi import PI_SETTINGS_BACKUP_PATH, PI_SETTINGS_PATH
    from ucode.config_io import restore_file

    low_verbosity = is_low_verbosity()
    console.print()
    if low_verbosity:
        console.print("[bold blue]Validating...[/bold blue]")
    else:
        console.print(
            Panel(
                "Testing each tool with a quick message...",
                title="Validating",
                style="bold blue",
                expand=False,
            )
        )
    results: list[tuple[str, bool]] = []
    available_tools = list(state.get("available_tools") or [])
    for tool, spec in TOOL_SPECS.items():
        if tool not in available_tools:
            continue
        with spinner(f"Validating {spec['display']}..."):
            ok, err = validate_tool(tool)
        results.append((tool, ok))
        if ok:
            print_success(f"{spec['display']} is working")
        else:
            print_err(f"{spec['display']}: {provider_permission_error(tool, state, err)}")
            managed = bool(state.get("managed_configs", {}).get(tool))
            restore_file(spec["config_path"], spec["backup_path"], managed)
            # Rollback settings.json for Pi
            if tool == "pi":
                restore_file(PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH, managed)
            available_tools.remove(tool)
    state["available_tools"] = available_tools
    save_state(state)

    success_tools = [(t, s) for t, s in results if s]
    if success_tools and not low_verbosity:
        console.print()
        lines = []
        for tool, _ in success_tools:
            spec = TOOL_SPECS[tool]
            lines.append(
                f"[green]✓[/green] [bold]{spec['display']}[/bold] — "
                f"run with [cyan]ucode {tool}[/cyan]"
            )
        console.print(Panel("\n".join(lines), title="Ready", style="green", expand=False))
