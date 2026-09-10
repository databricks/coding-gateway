"""Resolve the effective agent settings from the managed config plus local ucode state.

The managed config (``~/.ucode/managed-state.json`` — authored by ``ucode setup`` and refreshed
from the workspace at launch, both through :mod:`ucode.managed_config`) and the developer's own
ucode state (``~/.ucode/state.json``) stay separate files — they are never merged on disk. Instead
this module resolves them *per key* at config-write time: whatever the manifest specifies wins, and
anything it leaves unset falls back to
the developer's ucode state. The resolved view is what gets rendered into the agent config files
(e.g. ``~/.claude/ucode-settings.json``), so managed settings take precedence for every ``ucode``
command without either file being rewritten.

Only settings the developer set *through* ucode participate in the fallback. Settings they wrote by
hand outside ucode (``~/.claude/settings.json``, etc.) are not read here — Claude Code merges
those scopes itself at launch, underneath the file ucode passes via ``--settings``.

Everything here is pure: no I/O, no mutation of the inputs. Fetching and persisting the manifest,
and handing the resolved state to the agent config writers, live in :mod:`ucode.managed_config`.
"""

from __future__ import annotations

from typing import cast

from ucode.databricks import ANTHROPIC_FAMILIES, classify_model_family
from ucode.state import MANAGED_OVERLAY_KEY

# Proto model-config slot -> the family key `claude.py`'s render_overlay reads. The manifest keeps
# the proto spelling (`default_opus_model`), while ucode state and render_overlay both key claude
# models by bare family (`opus`), so the two have to be bridged before the settings file is written.
_CLAUDE_FAMILY_SLOTS = {
    "default_opus_model": "opus",
    "default_sonnet_model": "sonnet",
    "default_haiku_model": "haiku",
    "default_fable_model": "fable",
}


def _as_dict(value: object) -> dict[str, object]:
    """Return ``value`` as a ``dict[str, object]`` when it is a dict, else an empty dict."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _str(value: object) -> str | None:
    """Return a non-empty stripped string, or None."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _agent_entry(managed: dict, tool: str) -> dict[str, object]:
    """Return the manifest's config for ``tool``, or an empty dict when it isn't enabled."""
    enabled = _as_dict(_as_dict(managed).get("enabled_agents"))
    return _as_dict(enabled.get(tool))


def _agent_model_config(managed: dict, tool: str) -> dict[str, object]:
    """Return the manifest's normalized ``model_config`` for ``tool``, if any."""
    return _as_dict(_agent_entry(managed, tool).get("model_config"))


def managed_state_overrides(managed: dict, tool: str) -> dict[str, object]:
    """The state keys to layer over local state so ``tool``'s writer sees the admin's models.

    Each agent reads its models from a different shape, so the manifest's list has to be translated
    rather than dropped into one key: opencode wants provider-bucketed lists, and pi/copilot compose
    from their own per-agent keys. Returns ``{state_key: value}`` — empty when the manifest names
    nothing for ``tool``, in which case the developer's own state stands.
    """
    overrides: dict[str, object] = {}
    models = _manifest_models(managed, tool)
    if models:
        if tool == "claude":
            overrides["claude_models"] = models
        elif tool == "opencode" and isinstance(models, list):
            # OpenCode selects `provider/model`, so its state is bucketed by provider rather than flat.
            # No override when nothing buckets: an empty dict would replace the developer's own
            # models, leaving opencode with none at all.
            buckets = _bucket_by_provider(models)
            if buckets:
                overrides["opencode_models"] = buckets
        else:
            overrides[f"{tool}_models"] = models
    default_model = _str(_agent_model_config(managed, tool).get("default_model"))
    if default_model:
        overrides[f"{tool}_default_model"] = default_model
    return overrides


def managed_unservable_models(managed: dict, tool: str) -> list[str]:
    """The models the manifest names for ``tool`` when it has no provider to serve any of them.

    Only non-empty when *every* named model is unservable, which is when the translation yields
    nothing and the developer's own models stand — so the caller can say why the admin's list had no
    effect. opencode has no OpenAI provider and pi has no OSS provider, so each can be handed a
    valid model FQN it cannot route.
    """
    if tool not in ("opencode", "pi"):
        return []
    models = _manifest_models(managed, tool)
    if not isinstance(models, list):
        return []
    servable = (
        _bucket_by_provider(models)
        if tool == "opencode"
        else [
            m
            for m in models
            if classify_model_family(m) in (*ANTHROPIC_FAMILIES, "codex", "gemini")
        ]
    )
    return [] if servable else models


def _manifest_models(managed: dict, tool: str) -> dict | list | None:
    """The manifest's models for ``tool`` in its own vocabulary, or None when it names none."""
    manifest_models = _agent_model_config(managed, tool).get("models")
    if tool == "claude":
        slots: dict[str, str] = {}
        for slot, family in _CLAUDE_FAMILY_SLOTS.items():
            model = _str(_as_dict(manifest_models).get(slot))
            if model:
                slots[family] = model
        return slots or None
    if isinstance(manifest_models, list):
        listed = [model for model in (_str(item) for item in manifest_models) if model]
        return listed or None
    return None


def _bucket_by_provider(models: list[str]) -> dict[str, list[str]]:
    """Group model FQNs into OpenCode's provider buckets, mirroring how discovery builds them.

    Discovery derives these from the per-family lists (claude -> anthropic, and gemini/oss as-is), so
    the same family classification recovers them from a flat manifest list. Models whose family
    can't be identified are dropped.
    """
    buckets: dict[str, list[str]] = {}
    for model in models:
        family = classify_model_family(model)
        if family in ANTHROPIC_FAMILIES:
            buckets.setdefault("anthropic", []).append(model)
        elif family in ("gemini", "oss"):
            buckets.setdefault(family, []).append(model)
    return buckets


def managed_enabled_tools(managed: dict) -> list[str]:
    """The tools the managed config enables, in the config's own order.

    Every entry is an agent ucode recognizes: ``normalize_managed_config`` drops enum values this
    build doesn't know, so an unrecognized agent never reaches here."""
    return list(_as_dict(_as_dict(managed).get("enabled_agents")))


def managed_supplies_models(managed: dict | None, tool: str) -> bool:
    """True when the managed config already says which models ``tool`` should use.

    Lets the launch path skip Databricks model discovery, whose whole purpose is to find the models
    the config has now specified. Any of the three counts: a provider (the agent routes by header and
    pins no Databricks model), a ``default_model``, or at least one entry in ``models``.
    """
    model_config = _agent_model_config(managed or {}, tool)
    if _str(model_config.get("model_provider_service")) or _str(model_config.get("default_model")):
        return True
    models = model_config.get("models")
    if isinstance(models, dict):
        return any(_str(value) for value in models.values())
    if isinstance(models, list):
        return any(_str(item) for item in models)
    return False


def managed_provider_service(managed: dict, tool: str) -> str | None:
    """Return only the provider the managed config specifies for ``tool``, ignoring local state."""
    return _str(_agent_model_config(managed, tool).get("model_provider_service"))


def managed_default_model(managed: dict, tool: str) -> str | None:
    """Return the model the managed config wants ``tool`` to launch on, if it names one.

    Distinct from the family slots :func:`managed_state_overrides` resolves: those set what each
    family shortcut maps to, while this is the model the session actually starts on. The launch path
    pins it explicitly, so the admin's choice holds even for agents that would otherwise pick their
    own default."""
    return _str(_agent_model_config(managed, tool).get("default_model"))


def managed_claude_family_models(managed: dict) -> dict[str, str] | None:
    """Claude family models explicitly authored by Coding Agent Config."""

    models = _manifest_models(managed, "claude")
    return cast("dict[str, str]", models) if isinstance(models, dict) else None


def managed_provider_family_models(managed: dict) -> dict[str, str] | None:
    """Claude's authored per-family models for launch, when a managed config routes it through a
    Model Provider Service.

    The launch path pins each ``ANTHROPIC_DEFAULT_<FAMILY>_MODEL`` from this so a *managed* launch
    uses exactly the versions the admin chose in ``ucode setup`` — rather than
    ``resolve_provider_models`` re-deriving "newest per family" from the service's live targets. It
    returns the manifest's own family slots (``{opus: id, sonnet: id, ...}``), i.e. what the wizard's
    per-family prompt authored.

    Falls back to the single ``default_model`` (mapped to its family) when the manifest carries no
    slots — the case where the service is ``allow_all_targets`` so setup could only ask for one
    overall default. Returns None when neither is present, leaving the launch path to its usual
    provider handling.

    TODO: when the service is ``allow_all_targets`` an admin can't enumerate a per-family choice yet.
    A list-models API for provider services would let the wizard offer the full catalog per family;
    until then the single default is the best the manifest can express.
    """
    from ucode.managed_setup import claude_family_for_model

    config = _agent_model_config(managed, "claude")
    slots: dict[str, str] = {}
    raw_slots = _as_dict(config.get("models"))
    for slot, family in _CLAUDE_FAMILY_SLOTS.items():
        model = _str(raw_slots.get(slot))
        if model:
            slots[family] = model
    if slots:
        return slots
    default_model = _str(config.get("default_model"))
    if default_model:
        family = claude_family_for_model(default_model)
        if family:
            return {family: default_model}
    return None


def recommended_agent(recommendation: dict | None, managed: dict) -> str | None:
    """The agent the budget tier recommends, or the config's ``default_agent`` when it names none.

    The server resolves the agent before the model, so a tier can move a developer to a cheaper
    agent without restating a model.
    """
    agent = _str(_as_dict(recommendation).get("agent"))
    return agent or _str(_as_dict(managed).get("default_agent"))


def managed_launch_model(managed: dict, recommendation: dict | None, tool: str) -> str | None:
    """The model the admin's policy wants ``tool`` to start on, or None.

    A budget recommendation supersedes the config's own ``default_model``, since it additionally
    reflects which spend tier the developer has reached — but only for the agent it was recommended
    for. A tier that moves the org to another agent names that agent's model, which the one being
    launched may not be able to serve.
    """
    recommended = _as_dict(recommendation)
    agent = _str(recommended.get("agent"))
    if agent is None or agent == tool:
        model = _str(recommended.get("model"))
        if model:
            return model
    return managed_default_model(managed, tool)


def resolve_state(managed: dict, state: dict, tool: str) -> dict:
    """Return a copy of ``state`` with ``tool``'s managed values layered on top.

    ``write_tool_config`` reads its models and provider out of the state dict it is handed, so
    handing it this resolved copy is what makes managed settings win. Each key the managed config
    displaces is recorded under :data:`~ucode.state.MANAGED_OVERLAY_KEY` with the developer's own
    value (None when they had none), which ``save_state`` swaps back before writing — so the admin's
    settings reach the generated agent config files without ``state.json`` losing what the developer
    configured. The two files are never merged on disk.
    """
    resolved = dict(state)
    overlay: dict[str, object] = {}
    for key, value in managed_state_overrides(managed, tool).items():
        if value != state.get(key):
            overlay[key] = state.get(key)
            resolved[key] = value
    provider = managed_provider_service(managed, tool)
    if provider:
        providers = dict(_as_dict(state.get("provider_services")))
        if providers.get(tool) != provider:
            overlay["provider_services"] = state.get("provider_services")
            providers[tool] = provider
            resolved["provider_services"] = providers
    if overlay:
        resolved[MANAGED_OVERLAY_KEY] = overlay
    return resolved
