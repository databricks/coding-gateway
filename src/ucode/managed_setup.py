"""Admin-authored managed coding-agent config: model catalogs, validation, and serialization.

This module owns the admin-write half of the managed config, mirroring the developer-read half in
:mod:`ucode.managed_config`:

- which models an admin may pick for each agent (:func:`model_options_for_agent`),
- validating a manifest before it is published (:func:`validate_manifest`), and
- serializing ucode's internal manifest shape into proto-JSON ``CodingAgentConfig``
  (:func:`serialize_managed_config`).

The manifest shape here is exactly the one :func:`ucode.managed_config.normalize_managed_config`
produces, so ``serialize`` then ``normalize`` round-trips to the input. The enum maps are derived by
inverting that module's maps rather than restated, so a new agent or MCP type only has to be added
once.

Local persistence is not duplicated here: the authored manifest is saved to and loaded from the one
local file, ``~/.ucode/managed-state.json``, via :func:`ucode.managed_config.save_managed_state` and
:func:`ucode.managed_config.load_managed_state` — the same file the launch path pulls into.

The interactive wizard that calls these helpers, and the publish step, live in
:mod:`ucode.managed_wizard`.
"""

from __future__ import annotations

import uuid
from typing import cast

from ucode.databricks import (
    ANTHROPIC_FAMILIES,
    model_version_sort_key,
    tool_supports_provider_type,
)
from ucode.managed_config import (
    AGENT_ENUM_TO_TOOL,
    MCP_TYPE_ENUM_TO_TAG,
)

# ucode tool name -> CodingAgent proto enum, and ucode MCP type tag -> McpServerType proto enum.
# Inverted from the read side's maps so the two directions cannot drift: adding an agent to
# `managed_config._AGENT_ENUM_TO_TOOL` makes it serializable here automatically.
AGENT_TOOL_TO_ENUM: dict[str, str] = {tool: enum for enum, tool in AGENT_ENUM_TO_TOOL.items()}
MCP_TAG_TO_TYPE_ENUM: dict[str, str] = {tag: enum for enum, tag in MCP_TYPE_ENUM_TO_TAG.items()}

# Agents whose model config carries a flat `models` list. Claude instead uses per-family slots
# (`ClaudeDefaultModels`), and Codex has no model list at all — it selects exactly one model.
_FLAT_MODEL_LIST_AGENTS = frozenset({"opencode", "pi", "gemini", "copilot"})

# Claude family slot names in `ClaudeDefaultModels`, keyed by ucode's family name. Public because
# the wizard prompts one slot at a time.
CLAUDE_SLOT_FOR_FAMILY: dict[str, str] = {
    family: f"default_{family}_model" for family in ANTHROPIC_FAMILIES
}

# Which discovered model families each agent may be configured with, on the Databricks-hosted path.
# Claude Code only speaks the Anthropic dialect, Gemini CLI only Gemini; Codex's `/v1/models` route
# serves GPT plus the OSS models; the multi-provider harnesses can use anything discovered.
_AGENT_MODEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "claude": ("claude",),
    "gemini": ("gemini",),
    "codex": ("codex", "oss"),
    "opencode": ("claude", "codex", "gemini", "oss"),
    "pi": ("claude", "codex", "gemini", "oss"),
    "copilot": ("claude", "codex", "gemini", "oss"),
    # omp mirrors pi (the multi-provider harness); it is still excluded from the
    # setup picker (see SETUP_EXCLUDED_AGENTS) since the server has no omp variant.
    "omp": ("claude", "codex", "gemini", "oss"),
}


def _as_dict(value: object) -> dict[str, object]:
    """Return ``value`` as a ``dict[str, object]`` when it is a dict, else an empty dict.

    Mirrors the read side's helper: a bare ``isinstance(x, dict)`` narrows to ``dict[Never, Never]``,
    which rejects string keys, so ``.get("name")`` on an untyped manifest fails type-checking without
    this.
    """
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def model_families_for_agent(tool: str) -> tuple[str, ...]:
    """The discovered model families ``tool`` can be configured with."""
    return _AGENT_MODEL_FAMILIES.get(tool, ())


def supports_provider_service(tool: str, provider_type: str) -> bool:
    """True when ``tool`` can route through a ``provider_type`` Model Provider Service.

    Thin pass-through to :func:`ucode.databricks.tool_supports_provider_type` so the wizard has one
    obvious place to ask. Only claude (anthropic / amazon_bedrock) and codex (openai) have MPS
    support today; the other harnesses are Databricks-hosted only.
    """
    return tool_supports_provider_type(tool, provider_type)


def model_options_for_agent(tool: str, state: dict) -> list[str]:
    """Models an admin may pick for ``tool``, drawn from the workspace's discovered inventory.

    ``state`` is a hydrated workspace state (as ``configure_shared_state`` produces): ``claude_models``
    is a family->id dict, while ``codex_models`` / ``gemini_models`` / ``oss_models`` are lists.
    Returns a de-duplicated list in a stable order (Claude newest-family first, then the family
    order in :data:`_AGENT_MODEL_FAMILIES`) — empty when nothing was discovered for those families,
    in which case the caller should fall back to free-text entry.
    """
    options: list[str] = []
    for family in model_families_for_agent(tool):
        if family == "claude":
            claude_models = state.get("claude_models")
            if isinstance(claude_models, dict):
                # ANTHROPIC_FAMILIES is newest-tier-first, so slot order is meaningful.
                for name in ANTHROPIC_FAMILIES:
                    model = claude_models.get(name)
                    if isinstance(model, str) and model:
                        options.append(model)
            continue
        key = {"codex": "codex_models", "gemini": "gemini_models", "oss": "oss_models"}[family]
        models = state.get(key)
        if isinstance(models, list):
            options.extend(m for m in models if isinstance(m, str) and m)
    # dict.fromkeys de-duplicates while preserving first-seen order (a model can match more than
    # one family bucket, e.g. an OSS id that also appears in the codex listing).
    return list(dict.fromkeys(options))


def claude_family_for_model(model: str) -> str | None:
    """The Claude family (``opus``/``sonnet``/``haiku``/``fable``) a model id belongs to, or None.

    Used to place picked Claude models into their `ClaudeDefaultModels` slots. Matches on the
    family segment so both discovery spellings work (``system.ai.claude-opus-4-8`` and
    ``databricks-claude-opus-4-8``).
    """
    lowered = model.lower()
    return next((family for family in ANTHROPIC_FAMILIES if f"claude-{family}-" in lowered), None)


def claude_family_candidates(
    all_claude_models: list[str], state: dict | None = None
) -> dict[str, list[str]]:
    """Group Claude model ids by family, newest first.

    ``state["claude_models"]`` holds only one id per family — the newest, chosen by
    ``discover_model_services`` for the launch path, which pins exactly one model per family alias.
    An admin authoring a managed config needs the alternatives too: pinning ``default_opus_model``
    to a known-good ``claude-opus-4-8`` rather than whatever happens to be newest is a normal thing
    to want, and impossible if only the newest is offered.

    ``all_claude_models`` is the unbucketed listing (see
    :func:`ucode.databricks.discover_claude_family_candidates`). When it is empty, falls back to the
    per-family picks already in ``state`` so the per-slot prompts still work — with one candidate
    each. Families with no models are omitted.
    """
    models = list(all_claude_models)
    if not models and state:
        claude_models = state.get("claude_models")
        if isinstance(claude_models, dict):
            models = [m for m in claude_models.values() if isinstance(m, str) and m]

    candidates: dict[str, list[str]] = {}
    for model in models:
        family = claude_family_for_model(model)
        if family:
            candidates.setdefault(family, []).append(model)
    for family, found in candidates.items():
        # model_version_sort_key negates version components, so plain ascending is newest-first.
        candidates[family] = sorted(set(found), key=model_version_sort_key)
    return candidates


def claude_model_slots(models: list[str]) -> dict[str, str]:
    """Group picked Claude model ids into ``ClaudeDefaultModels`` slots.

    Claude Code addresses models by family alias rather than by list, so the wizard's multi-select
    has to be bucketed into ``default_opus_model`` / ``default_sonnet_model`` / etc. Ids whose family
    can't be identified are skipped; when two ids share a family the first wins (the caller's list
    order is the admin's preference order).
    """
    slots: dict[str, str] = {}
    for model in models:
        family = claude_family_for_model(model)
        if family is None:
            continue
        slot = CLAUDE_SLOT_FOR_FAMILY[family]
        slots.setdefault(slot, model)
    return slots


def _model_config_payload(tool: str, model_config: dict) -> dict:
    """Build one ``AgentModelConfig`` oneof variant body for ``tool``.

    Shapes per the proto: claude gets `models` as a `ClaudeDefaultModels` slot object, codex gets
    no model list at all, and the rest get a flat repeated `models`.
    """
    body: dict = {}
    mps = model_config.get("model_provider_service")
    if isinstance(mps, str) and mps:
        body["model_provider_service"] = mps
    default_model = model_config.get("default_model")
    if isinstance(default_model, str) and default_model:
        body["default_model"] = default_model

    models = model_config.get("models")
    if tool == "claude":
        if isinstance(models, dict):
            slots = {
                slot: value
                for slot, value in models.items()
                if isinstance(slot, str) and isinstance(value, str) and value
            }
            if slots:
                body["models"] = slots
    elif tool in _FLAT_MODEL_LIST_AGENTS:
        if isinstance(models, list):
            model_list = [m for m in models if isinstance(m, str) and m]
            if model_list:
                body["models"] = model_list
    # codex intentionally carries no model list — CodexModelConfig has only
    # model_provider_service + default_model.
    return body


def _enabled_agent_payload(tool: str, agent_config: dict) -> dict:
    """Build one ``EnabledAgent`` entry (agent enum + its ``AgentConfig``)."""
    config: dict = {}
    headers = agent_config.get("custom_headers")
    if isinstance(headers, dict):
        clean = {k: v for k, v in headers.items() if isinstance(k, str) and isinstance(v, str)}
        if clean:
            config["custom_headers"] = clean
    tracing_table = agent_config.get("tracing_table")
    if isinstance(tracing_table, str) and tracing_table:
        config["tracing_config"] = {"table": tracing_table}
    model_config = agent_config.get("model_config")
    if isinstance(model_config, dict):
        body = _model_config_payload(tool, model_config)
        if body:
            # The `AgentModelConfig` oneof field names are ucode's tool names verbatim (claude,
            # codex, opencode, pi, gemini, copilot), so the tool doubles as the variant key. The
            # server rejects a variant that doesn't match its agent (`validateAgentModelConfig`),
            # and the round-trip through `normalize_managed_config` pins that alignment in tests.
            config["model_config"] = {tool: body}

    entry: dict = {"agent": AGENT_TOOL_TO_ENUM[tool]}
    if config:
        entry["config"] = config
    return entry


def _budget_policy_payload(budget_policy: dict) -> dict:
    """Build the ``BudgetPolicy`` body, dropping tiers that name an unknown agent.

    ``spending_percentage`` is passed through as-is: it is a fraction in [0, 1] both in ucode's
    manifest and in the proto (the server validates that range). Callers prompting an admin in
    percent must divide before building the manifest.
    """
    payload: dict = {}
    display_name = budget_policy.get("display_name")
    if isinstance(display_name, str) and display_name:
        payload["display_name"] = display_name
    budget_id = budget_policy.get("budget_id")
    if isinstance(budget_id, str) and budget_id:
        payload["budget_id"] = budget_id

    tiers: list[dict] = []
    raw_tiers = budget_policy.get("tiers")
    for tier in raw_tiers if isinstance(raw_tiers, list) else []:
        if not isinstance(tier, dict):
            continue
        pct = tier.get("spending_percentage")
        if not isinstance(pct, (int, float)) or isinstance(pct, bool):
            continue
        tier_payload: dict = {"spending_percentage": float(pct)}
        agent_enum = AGENT_TOOL_TO_ENUM.get(str(tier.get("default_agent") or ""))
        if agent_enum:
            tier_payload["default_agent"] = agent_enum
        default_model = tier.get("default_model")
        if isinstance(default_model, str) and default_model:
            tier_payload["default_model"] = default_model
        tiers.append(tier_payload)
    if tiers:
        payload["tiers"] = tiers
    return payload


def serialize_managed_config(manifest: dict) -> dict:
    """Serialize ucode's internal manifest into a proto-JSON ``CodingAgentConfig``.

    The exact inverse of :func:`ucode.managed_config.normalize_managed_config`: tool names become
    ``CODING_AGENT_*`` enums, MCP type tags become ``MCP_SERVER_TYPE_*``, and each agent's model
    config is wrapped in its matching ``AgentModelConfig`` oneof variant. Agents and MCP types this
    build doesn't recognize are dropped, mirroring the read side.

    Output-only proto fields (``workspace_id``, timestamps, user ids) are never emitted. ``name`` is
    carried through when present so an update path can address an existing resource; ``ucode publish``
    omits it on create and lets the server assign one.
    """
    payload: dict = {}

    name = manifest.get("name")
    if isinstance(name, str) and name:
        payload["name"] = name
    display_name = manifest.get("display_name")
    if isinstance(display_name, str) and display_name:
        payload["display_name"] = display_name

    default_agent = AGENT_TOOL_TO_ENUM.get(str(manifest.get("default_agent") or ""))
    if default_agent:
        payload["default_agent"] = default_agent

    enabled_agents = manifest.get("enabled_agents")
    if isinstance(enabled_agents, dict):
        entries = [
            _enabled_agent_payload(tool, agent_config)
            for tool, agent_config in enabled_agents.items()
            if tool in AGENT_TOOL_TO_ENUM and isinstance(agent_config, dict)
        ]
        if entries:
            payload["enabled_agents"] = entries

    mcp_servers = manifest.get("mcp_servers")
    if isinstance(mcp_servers, list):
        servers: list[dict] = []
        for server in mcp_servers:
            if not isinstance(server, dict):
                continue
            server_name = server.get("name")
            type_enum = MCP_TAG_TO_TYPE_ENUM.get(str(server.get("type") or ""))
            if isinstance(server_name, str) and server_name and type_enum:
                servers.append({"name": server_name, "type": type_enum})
        if servers:
            payload["mcp_servers"] = servers

    skills = manifest.get("skills")
    if isinstance(skills, dict):
        names = skills.get("names")
        if isinstance(names, list):
            skill_names = [n for n in names if isinstance(n, str) and n]
            if skill_names:
                payload["skills"] = {"names": skill_names}

    tracing_table = manifest.get("tracing_table")
    if isinstance(tracing_table, str) and tracing_table:
        payload["tracing"] = {"table": tracing_table}

    budget_policy = manifest.get("budget_policy")
    if isinstance(budget_policy, dict):
        policy = _budget_policy_payload(budget_policy)
        if policy:
            payload["budget_policy"] = policy

    return payload


def _manifest_default_model(agent_config: dict) -> str | None:
    """The ``default_model`` on an agent's model config, or None when unset/empty."""
    model_config = agent_config.get("model_config")
    if not isinstance(model_config, dict):
        return None
    default_model = model_config.get("default_model")
    return default_model if isinstance(default_model, str) and default_model else None


def _known_models(state: dict) -> set[str]:
    """Every model id discovered on the workspace, across all families.

    Empty when discovery found nothing (or no state was passed), which callers treat as "can't
    check" rather than "nothing is valid".

    ``claude_models`` holds only the newest id per family (the launch path pins one model per family
    alias), so on its own it would reject the older versions the per-family prompts legitimately
    offer. ``all_claude_models`` carries the full listing when the caller has it.
    """
    known: set[str] = set()
    claude_models = state.get("claude_models")
    if isinstance(claude_models, dict):
        known.update(m for m in claude_models.values() if isinstance(m, str) and m)
    for key in ("codex_models", "gemini_models", "oss_models", "all_claude_models"):
        models = state.get(key)
        if isinstance(models, list):
            known.update(m for m in models if isinstance(m, str) and m)
    return known


def _validate_agent_models(tool: str, agent_config: dict, known: set[str]) -> list[str]:
    """Check one agent's configured models against the workspace inventory.

    Skipped entirely when the agent routes through a Model Provider Service: those model ids come
    from the provider's own catalog, not from UC model services, so the workspace inventory says
    nothing about them.

    ``model_config.custom_models`` lists ids the admin typed by hand for a model discovery didn't
    surface (a model service outside ``system.ai``). Those were verified to exist when entered (see
    ``managed_wizard._prompt_custom_model``), so they're excluded from the inventory check — the
    discovered inventory legitimately doesn't contain them.
    """
    model_config = agent_config.get("model_config")
    if not isinstance(model_config, dict):
        return []
    if model_config.get("model_provider_service"):
        return []

    custom = {m for m in model_config.get("custom_models", []) if isinstance(m, str) and m}
    referenced: list[str] = []
    default_model = model_config.get("default_model")
    if isinstance(default_model, str) and default_model:
        referenced.append(default_model)
    models = model_config.get("models")
    if isinstance(models, dict):
        referenced.extend(m for m in models.values() if isinstance(m, str) and m)
    elif isinstance(models, list):
        referenced.extend(m for m in models if isinstance(m, str) and m)

    return [
        f"{tool}: model '{model}' is not available on this workspace."
        for model in dict.fromkeys(referenced)
        if model not in known and model not in custom
    ]


def validate_manifest(manifest: dict, state: dict | None = None) -> list[str]:
    """Validate a manifest before publishing; returns human-readable errors (empty when valid).

    Mirrors the server's ``validateStoredConfig`` so an admin sees problems locally instead of via
    an ``INVALID_PARAMETER_VALUE`` round-trip:

    - ``default_agent`` is required once any agent configuration is present, must appear in
      ``enabled_agents``, and that agent must have a non-empty ``default_model``;
    - every ``enabled_agents`` key must be an agent this ucode build knows;
    - each MCP server needs a name and a recognized type; skill names must be non-empty;
    - ``tracing_table`` must be non-empty when the key is present;
    - a ``budget_policy`` needs a ``budget_id``, and each tier needs a ``spending_percentage`` in
      [0, 1] (unique across tiers), a ``default_agent`` that appears in ``enabled_agents``, and a
      ``default_model``.

    When ``state`` is provided, configured models are additionally checked against the workspace's
    discovered inventory — skipped for agents routing through a Model Provider Service, and skipped
    entirely when discovery returned nothing.
    """
    errors: list[str] = []

    enabled_agents_raw = manifest.get("enabled_agents")
    enabled_agents: dict[str, dict] = {}
    if isinstance(enabled_agents_raw, dict):
        for tool, agent_config in enabled_agents_raw.items():
            if tool not in AGENT_TOOL_TO_ENUM:
                valid = ", ".join(sorted(AGENT_TOOL_TO_ENUM))
                errors.append(
                    f"enabled_agents: '{tool}' is not a supported agent (valid: {valid})."
                )
                continue
            if not isinstance(agent_config, dict):
                errors.append(f"enabled_agents: '{tool}' config must be an object.")
                continue
            enabled_agents[tool] = agent_config

    default_agent = manifest.get("default_agent")
    budget_policy = manifest.get("budget_policy")
    has_agent_selection = bool(default_agent or enabled_agents_raw or budget_policy)
    if has_agent_selection:
        if not default_agent:
            errors.append("default_agent is required when agent configuration is present.")
        elif default_agent not in enabled_agents:
            errors.append(f"default_agent '{default_agent}' must appear in enabled_agents.")
        elif not _manifest_default_model(enabled_agents[default_agent]):
            errors.append(
                f"default_agent '{default_agent}' must have a non-empty model_config.default_model."
            )

    known = _known_models(state or {})
    if known:
        for tool, agent_config in enabled_agents.items():
            errors.extend(_validate_agent_models(tool, agent_config, known))

    mcp_servers = manifest.get("mcp_servers")
    if isinstance(mcp_servers, list):
        for index, raw_server in enumerate(mcp_servers, start=1):
            if not isinstance(raw_server, dict):
                errors.append(f"mcp_servers[{index}] must be an object.")
                continue
            server = _as_dict(raw_server)
            if not server.get("name"):
                errors.append(f"mcp_servers[{index}]: name is required.")
            server_type = str(server.get("type") or "")
            if server_type not in MCP_TAG_TO_TYPE_ENUM:
                valid = ", ".join(sorted(MCP_TAG_TO_TYPE_ENUM))
                errors.append(
                    f"mcp_servers[{index}]: type '{server_type}' is not recognized "
                    f"(valid: {valid})."
                )

    skills = manifest.get("skills")
    if isinstance(skills, dict):
        names = skills.get("names")
        if isinstance(names, list) and any(not isinstance(name, str) or not name for name in names):
            errors.append("skills.names must not contain empty names.")

    if "tracing_table" in manifest and not manifest.get("tracing_table"):
        errors.append("tracing_table must not be empty.")

    if isinstance(budget_policy, dict):
        errors.extend(_validate_budget_policy(budget_policy, enabled_agents))

    return errors


def _agent_model_ids(agent_config: dict) -> set[str]:
    """Every model id an agent is configured with — its list plus its default.

    Claude's ``models`` is a family-slot dict and the others' a flat list; codex has no list at all,
    only ``default_model``. Returns an empty set when nothing is configured, which callers treat as
    "can't check" rather than "nothing is allowed".
    """
    model_config = agent_config.get("model_config")
    if not isinstance(model_config, dict):
        return set()
    ids: set[str] = set()
    raw = model_config.get("models")
    if isinstance(raw, dict):
        ids.update(v for v in raw.values() if isinstance(v, str) and v)
    elif isinstance(raw, list):
        ids.update(m for m in raw if isinstance(m, str) and m)
    default_model = model_config.get("default_model")
    if isinstance(default_model, str) and default_model:
        ids.add(default_model)
    return ids


def _validate_budget_policy(budget_policy: dict, enabled_agents: dict[str, dict]) -> list[str]:
    """Validate a ``budget_policy`` against the agents the manifest enables.

    Tier positions are reported 0-based to match the server's own messages, which index with
    ``zipWithIndex`` — an admin comparing the two error sources should see the same number.
    """
    errors: list[str] = []
    budget_id = budget_policy.get("budget_id")
    if not budget_id:
        errors.append("budget_policy.budget_id is required.")
    else:
        # The server requires a parseable UUID here. The wizard can only offer real
        # `budget_configuration_id`s, but `--from-file` and hand-edited manifests can carry
        # anything, and catching it locally beats an INVALID_PARAMETER_VALUE round-trip.
        try:
            uuid.UUID(str(budget_id))
        except ValueError:
            errors.append(
                f"budget_policy.budget_id must be a UUID (got '{budget_id}'). Use the "
                "budget_configuration_id from the workspace's AI Gateway budgets."
            )

    percentages: list[float] = []
    combos: list[tuple[str, str]] = []
    tiers = budget_policy.get("tiers")
    for index, tier in enumerate(tiers if isinstance(tiers, list) else []):
        if not isinstance(tier, dict):
            errors.append(f"budget_policy.tiers[{index}] must be an object.")
            continue
        pct = tier.get("spending_percentage")
        if not isinstance(pct, (int, float)) or isinstance(pct, bool):
            errors.append(f"budget_policy.tiers[{index}]: spending_percentage is required.")
        elif not 0 <= float(pct) <= 1:
            errors.append(
                f"budget_policy.tiers[{index}]: spending_percentage must be a fraction "
                f"between 0 and 1 (got {pct})."
            )
        else:
            percentages.append(float(pct))
        tier_agent = tier.get("default_agent")
        tier_model = tier.get("default_model")
        if not tier_agent:
            errors.append(f"budget_policy.tiers[{index}]: default_agent is required.")
        elif tier_agent not in enabled_agents:
            errors.append(
                f"budget_policy.tiers[{index}]: default_agent '{tier_agent}' must appear "
                "in enabled_agents."
            )
        elif tier_model:
            # The server only checks that the tier's agent is enabled, not that it has the model —
            # so without this a tier can activate and hand the developer a model their agent was
            # never configured with. Skipped when the agent lists no models (it then has only a
            # default, or routes through a provider service whose catalog isn't enumerable).
            available = _agent_model_ids(enabled_agents[tier_agent])
            if available and tier_model not in available:
                errors.append(
                    f"budget_policy.tiers[{index}]: default_model '{tier_model}' is not one of the "
                    f"models configured for '{tier_agent}' ({', '.join(sorted(available))})."
                )
        if not tier_model:
            errors.append(f"budget_policy.tiers[{index}]: default_model is required.")
        if tier_agent and tier_model:
            combos.append((str(tier_agent), str(tier_model)))

    if len(set(percentages)) != len(percentages):
        errors.append("budget_policy tier spending_percentage values must be unique.")
    # Two tiers with the same agent+model are a no-op: the server picks the highest crossed tier, so
    # the second never changes what the lower one already selected. Flagging it catches a tier the
    # admin meant to be a real step-down but left unchanged.
    if len(set(combos)) != len(combos):
        errors.append(
            "budget_policy tiers must each route to a different agent/model — two tiers with the "
            "same pair make the higher one a no-op."
        )
    return errors
