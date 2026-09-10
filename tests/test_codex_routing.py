"""Tests for Codex smart-routing hooks."""

from __future__ import annotations

import json
import urllib.error

from ucode.smart_routing import codex_routing
from ucode.smart_routing.codex_hooks import routing_models

WS = "https://example.databricks.com"


def test_routing_models_combines_and_deduplicates_codex_and_oss_models():
    assert routing_models(
        {
            "codex_models": ["system.ai.gpt-5-6-sol", "system.ai.gpt-oss-120b"],
            "oss_models": ["system.ai.gpt-oss-120b", "system.ai.glm-5-2"],
        }
    ) == [
        "system.ai.gpt-5-6-sol",
        "system.ai.gpt-oss-120b",
        "system.ai.glm-5-2",
    ]


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_routes_with_models_from_stored_state(monkeypatch):
    monkeypatch.delenv("SMART_ROUTER_NAME", raising=False)
    captured = {}
    task = "Refactor the parser" + "x" * 5000

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            {
                "route_selection": [{"route_option": {"model": "gpt-5-6-sol", "harness": "codex"}}],
                "rationale": "Needs the strongest coding model.",
            }
        )

    monkeypatch.setattr(codex_routing.urllib.request, "urlopen", fake_urlopen)

    decision, error = codex_routing.request_routing_decision(
        WS,
        "token",
        task,
        ["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
    )

    assert error is None
    assert decision == codex_routing.RoutingDecision(
        model="system.ai.gpt-5-6-sol",
        raw_model="gpt-5-6-sol",
        rationale="Needs the strongest coding model.",
    )
    assert captured["url"] == f"{WS}/ai-gateway/routing/v1/routes:select"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["body"] == {
        "route_options": [
            {"model": "gpt-5-6-luna", "harness": "codex"},
            {"model": "gpt-5-6-sol", "harness": "codex"},
        ],
        "task": {"prompt": task},
        "route_selector": {"router_name": codex_routing.routing.ROUTER_NAME},
    }


def test_router_name_can_be_overridden_with_environment_variable(monkeypatch):
    captured = {}
    monkeypatch.setenv("SMART_ROUTER_NAME", "  custom_router  ")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return _Response({"route_selection": [{"route_option": {"model": "gpt-5-6-sol"}}]})

    monkeypatch.setattr(codex_routing.urllib.request, "urlopen", fake_urlopen)

    decision, error = codex_routing.request_routing_decision(
        WS, "token", "Refactor the parser", ["system.ai.gpt-5-6-sol"]
    )

    assert error is None
    assert decision is not None
    assert captured["body"]["route_selector"] == {"router_name": "custom_router"}


def test_blank_router_name_environment_variable_uses_default(monkeypatch):
    monkeypatch.setenv("SMART_ROUTER_NAME", "  ")

    assert codex_routing.routing.configured_router_name() == codex_routing.routing.ROUTER_NAME


def test_router_model_is_not_substituted_when_exact_model_is_unavailable():
    model = codex_routing.resolve_routed_model(
        "gpt-5-6-luna",
        ["databricks-gpt-5-4-nano", "databricks-gpt-5", "databricks-gpt-5-5"],
    )

    assert model is None


def test_glm_maps_to_databricks_gateway_model():
    model = codex_routing.resolve_routed_model(
        "glm-5-2",
        ["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol", "system.ai.glm-5-2"],
    )

    assert model == "system.ai.glm-5-2"


def test_exact_router_model_is_preserved():
    model = codex_routing.resolve_routed_model(
        "gpt-5-6-sol",
        ["databricks-gpt-5-5", "databricks-gpt-5-6-sol"],
    )

    assert model == "databricks-gpt-5-6-sol"


def test_router_failure_fails_open(monkeypatch):
    monkeypatch.setattr(
        codex_routing.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    decision, error = codex_routing.request_routing_decision(
        WS,
        "token",
        "task",
        ["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
    )

    assert decision is None
    assert "offline" in str(error)


def test_spawn_rewrite_preserves_original_input(monkeypatch):
    encrypted_message = {"encrypted": "opaque-ciphertext"}
    payload = {
        "tool_name": "collaborationspawn_agent",
        "tool_input": {
            "task_name": "reviewer",
            "message": encrypted_message,
            "fork": False,
        },
    }
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="databricks-gpt-5-5",
                raw_model="gpt-5-6-sol",
                rationale="Review needs deeper reasoning.",
            ),
            None,
        ),
    )

    output = codex_routing.route_pre_tool_use(
        payload,
        workspace=WS,
        token="token",
        available_models=["databricks-gpt-5-5"],
    )

    hook = output["hookSpecificOutput"]
    # The rationale is surfaced in BOTH the systemMessage (shown to the user) and
    # permissionDecisionReason, so the "why" is visible, not just the "what".
    expected_message = codex_routing.routing.format_subagent_message(
        "gpt-5.5", "Review needs deeper reasoning."
    )
    assert output["systemMessage"] == expected_message
    assert "Using Unity Gateway Smart Router - Subagent" in expected_message
    assert codex_routing.routing.SUBAGENT_ROUTING_DISCLAIMER not in expected_message
    assert hook["permissionDecision"] == "allow"
    assert hook["updatedInput"] == {
        "task_name": "reviewer",
        "message": encrypted_message,
        "fork": False,
        "model": "gpt-5.5",
    }
    assert hook["permissionDecisionReason"] == expected_message


def test_spawn_rewrite_uses_codex_model_id_for_uc_endpoint(monkeypatch):
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.gpt-5-6-luna",
                raw_model="gpt-5-6-luna",
            ),
            None,
        ),
    )

    output = codex_routing.route_pre_tool_use(
        {
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "routing-smoke-test", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.gpt-5-6-luna"],
    )

    assert output["systemMessage"] == codex_routing.routing.format_subagent_message(
        "gpt-5.6-luna", ""
    )
    assert output["hookSpecificOutput"]["updatedInput"]["model"] == "gpt-5.6-luna"


def test_codex_model_id_maps_uc_gpt_models_to_codex_slugs():
    expected = {
        "system.ai.gpt-5-2": "gpt-5.2",
        "system.ai.gpt-5-4": "gpt-5.4",
        "system.ai.gpt-5-4-mini": "gpt-5.4-mini",
        "system.ai.gpt-5-5": "gpt-5.5",
        "system.ai.gpt-5-6-luna": "gpt-5.6-luna",
        "system.ai.gpt-5-6-sol": "gpt-5.6-sol",
        "system.ai.gpt-5-6-terra": "gpt-5.6-terra",
    }
    assert {model: codex_routing.codex_model_id(model) for model in expected} == expected
    assert codex_routing.codex_model_id("system.ai.gpt-5-6-experimental") == "gpt-5.6-experimental"
    assert codex_routing.codex_model_id("system.ai.glm-5-2") == "system.ai.glm-5-2"


def test_spawn_glm_decision_applies_glm_model(monkeypatch):
    # GLM is no longer skipped for Codex subagents: a GLM routing decision is
    # applied like any other arm.
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.glm-5-2",
                raw_model="glm-5-2",
            ),
            None,
        ),
    )

    output = codex_routing.route_pre_tool_use(
        {
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "glm-task", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=[
            "system.ai.gpt-5-6-luna",
            "system.ai.gpt-5-6-sol",
            "system.ai.glm-5-2",
        ],
    )

    assert output["hookSpecificOutput"]["updatedInput"]["model"] == "system.ai.glm-5-2"
    assert "Selected Model : system.ai.glm-5-2" in output["systemMessage"]


def test_non_spawn_tool_has_no_opinion():
    assert (
        codex_routing.route_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "true"}},
            workspace=WS,
            token="token",
            available_models=["databricks-gpt-5"],
        )
        is None
    )


def test_spawn_routes_on_plaintext_message_when_present(monkeypatch):
    # When the spawn's `message` is a plaintext string at PreToolUse (before
    # Codex encrypts it at send-time), routing uses it as the task — giving the
    # router real signal instead of the generic fallback.
    captured = {}

    def fake_decision(*args, **kwargs):
        captured["task"] = args[2] if len(args) > 2 else kwargs.get("task")
        return (
            codex_routing.RoutingDecision(model="databricks-gpt-5-5", raw_model="gpt-5-6-sol"),
            None,
        )

    monkeypatch.setattr(codex_routing, "request_routing_decision", fake_decision)
    codex_routing.route_pre_tool_use(
        {
            "tool_name": "collaborationspawn_agent",
            "tool_input": {
                "task_name": "task_3",
                "message": "Review the parser error handling and add missing null checks",
            },
        },
        workspace=WS,
        token="token",
        available_models=["databricks-gpt-5-5"],
    )
    assert captured["task"] == "Review the parser error handling and add missing null checks"


def test_spawn_falls_through_encrypted_message_to_task_name(monkeypatch):
    # When `message` is an encrypted dict (not a plaintext string), routing
    # falls through to `task_name` — no regression from the encrypted case.
    captured = {}

    def fake_decision(*args, **kwargs):
        captured["task"] = args[2] if len(args) > 2 else kwargs.get("task")
        return (
            codex_routing.RoutingDecision(model="databricks-gpt-5-5", raw_model="gpt-5-6-sol"),
            None,
        )

    monkeypatch.setattr(codex_routing, "request_routing_decision", fake_decision)
    codex_routing.route_pre_tool_use(
        {
            "tool_name": "collaborationspawn_agent",
            "tool_input": {
                "task_name": "reviewer",
                "message": {"encrypted": "opaque-ciphertext"},
            },
        },
        workspace=WS,
        token="token",
        available_models=["databricks-gpt-5-5"],
    )
    # Encrypted dict skipped (not a string), fell through to task_name.
    assert captured["task"] == "reviewer"


def test_canary_and_audit_are_written(tmp_path, monkeypatch):
    canary = tmp_path / "canary.json"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(codex_routing, "CANARY_PATH", canary)
    monkeypatch.setattr(codex_routing, "AUDIT_PATH", audit)

    codex_routing.record_session_start({"session_id": "s1", "model": "gpt-5.5"})
    codex_routing.record_subagent_start(
        {"session_id": "s1", "agent_id": "a1", "agent_type": "reviewer", "model": "gpt-5"}
    )

    assert json.loads(canary.read_text())["session_id"] == "s1"
    assert json.loads(audit.read_text().strip())["agent_id"] == "a1"


def test_decision_is_reconciled_with_actual_subagent_model(tmp_path, monkeypatch):
    decisions = tmp_path / "decisions.jsonl"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(codex_routing, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(codex_routing, "AUDIT_PATH", audit)
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.gpt-5-6-luna",
                raw_model="gpt-5-6-luna",
            ),
            None,
        ),
    )

    codex_routing.route_pre_tool_use(
        {
            "session_id": "s1",
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "glm-task", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.gpt-5-6-luna", "system.ai.gpt-5-6-sol"],
        audit_decision=True,
    )
    record = codex_routing.record_subagent_start(
        {"session_id": "s1", "agent_id": "a1", "model": "gpt-5.6-luna"}
    )

    assert record["router_model"] == "gpt-5-6-luna"
    assert record["requested_model"] == "gpt-5.6-luna"
    assert record["matches_router_decision"] is True


def test_decision_record_persists_rationale(tmp_path, monkeypatch):
    decisions = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(codex_routing, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(
        codex_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            codex_routing.RoutingDecision(
                model="system.ai.gpt-5-6-sol",
                raw_model="gpt-5-6-sol",
                rationale="Cross-cutting refactor needs the strongest model.",
            ),
            None,
        ),
    )

    codex_routing.route_pre_tool_use(
        {
            "session_id": "s1",
            "tool_name": "collaborationspawn_agent",
            "tool_input": {"task_name": "refactor", "message": "encrypted"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.gpt-5-6-sol", "system.ai.gpt-5-6-luna"],
        audit_decision=True,
    )

    # The rationale is persisted so an empty value distinguishes "gateway
    # returned none" from a display-placement bug.
    record = json.loads(decisions.read_text().strip())
    assert record["rationale"] == "Cross-cutting refactor needs the strongest model."
