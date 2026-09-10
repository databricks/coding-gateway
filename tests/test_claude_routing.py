"""Tests for Claude Code smart-routing helpers."""

from __future__ import annotations

import json
import urllib.error

from ucode.smart_routing import claude_routing

WS = "https://example.databricks.com"


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_routes_with_default_claude_menu(monkeypatch):
    monkeypatch.delenv("SMART_ROUTER_NAME", raising=False)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            {
                "route_selection": [
                    {"route_option": {"model": "claude-opus-4-8", "harness": "claude"}}
                ],
                "rationale": "Cross-cutting change needs the strongest model.",
            }
        )

    monkeypatch.setattr(claude_routing.urllib.request, "urlopen", fake_urlopen)

    decision, error = claude_routing.request_routing_decision(
        WS,
        "token",
        "Refactor the parser",
        ["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
    )

    assert error is None
    assert decision == claude_routing.RoutingDecision(
        model="system.ai.claude-opus-4-8",
        raw_model="claude-opus-4-8",
        rationale="Cross-cutting change needs the strongest model.",
    )
    assert captured["url"] == f"{WS}/ai-gateway/routing/v1/routes:select"
    assert captured["headers"]["Authorization"] == "Bearer token"
    # The cc-scenario menu requires BOTH Claude arms be offered.
    assert captured["body"] == {
        "route_options": [
            {"model": "claude-opus-4-8", "harness": "claude"},
            {"model": "claude-sonnet-5", "harness": "claude"},
        ],
        "task": {"prompt": "Refactor the parser"},
        "route_selector": {"router_name": claude_routing.ROUTER_NAME},
    }


def test_missing_arm_short_circuits_without_calling_router(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("router must not be called when an arm is missing")

    monkeypatch.setattr(claude_routing.urllib.request, "urlopen", fail)

    # Only sonnet available; the cc menu also requires opus, so we bail early
    # rather than send an under-offered request the router would reject.
    decision, error = claude_routing.request_routing_decision(
        WS, "token", "task", ["system.ai.claude-sonnet-5"]
    )

    assert decision is None
    assert "claude-opus-4-8" in error


def test_router_pick_resolves_to_workspace_id(monkeypatch):
    monkeypatch.setattr(
        claude_routing.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {"route_selection": [{"route_option": {"model": "claude-sonnet-5"}}]}
        ),
    )

    decision, error = claude_routing.request_routing_decision(
        WS,
        "token",
        "small fix",
        ["databricks-claude-opus-4-8", "databricks-claude-sonnet-5"],
    )

    assert error is None
    # The bare router arm maps back to the workspace's routable id.
    assert decision.model == "databricks-claude-sonnet-5"


def test_router_failure_fails_open(monkeypatch):
    monkeypatch.setattr(
        claude_routing.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    decision, error = claude_routing.request_routing_decision(
        WS,
        "token",
        "task",
        ["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
    )

    assert decision is None
    assert "offline" in str(error)


def test_spawn_rewrite_injects_routed_model(monkeypatch):
    payload = {
        "tool_name": "Agent",
        "tool_input": {
            "subagent_type": "Explore",
            "prompt": "map the codebase",
            "description": "explore",
        },
    }
    monkeypatch.setattr(
        claude_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            claude_routing.RoutingDecision(
                model="system.ai.claude-opus-4-8",
                raw_model="claude-opus-4-8",
                rationale="Deep exploration needs the strongest model.",
            ),
            None,
        ),
    )

    output = claude_routing.route_pre_tool_use(
        payload,
        workspace=WS,
        token="token",
        available_models=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
    )

    hook = output["hookSpecificOutput"]
    # The rationale is surfaced in the systemMessage (shown to the user), not
    # only in permissionDecisionReason. The model field is the short family
    # name ("opus") that Claude Code's Agent tool schema accepts.
    expected_message = claude_routing.routing.format_subagent_message(
        "opus", "Deep exploration needs the strongest model."
    )
    assert output["systemMessage"] == expected_message
    assert hook["permissionDecision"] == "allow"
    assert hook["updatedInput"] == {
        "subagent_type": "Explore",
        "prompt": "map the codebase",
        "description": "explore",
        "model": "opus",
    }
    assert hook["permissionDecisionReason"] == expected_message


def test_task_tool_alias_is_routed(monkeypatch):
    monkeypatch.setattr(
        claude_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            claude_routing.RoutingDecision(
                model="system.ai.claude-sonnet-5", raw_model="claude-sonnet-5"
            ),
            None,
        ),
    )

    output = claude_routing.route_pre_tool_use(
        {"tool_name": "Task", "tool_input": {"subagent_type": "general", "prompt": "x"}},
        workspace=WS,
        token="token",
        available_models=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
    )

    assert output["hookSpecificOutput"]["updatedInput"]["model"] == "sonnet"


def test_spawn_routes_on_prompt_not_generic_label(monkeypatch):
    # Claude Code's Agent tool puts the subagent task in `prompt`/`description`,
    # not `message`. The router should receive the real prompt, not the generic
    # "Claude Code subagent task" fallback.
    captured = {}

    def fake_decision(workspace, token, task, models, **kwargs):
        captured["task"] = task
        return (
            claude_routing.RoutingDecision(
                model="system.ai.claude-sonnet-5", raw_model="claude-sonnet-5"
            ),
            None,
        )

    monkeypatch.setattr(claude_routing, "request_routing_decision", fake_decision)

    claude_routing.route_pre_tool_use(
        {
            "tool_name": "Agent",
            "tool_input": {
                "subagent_type": "Explore",
                "prompt": "find all Python entry points",
                "description": "explore the repo",
            },
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
    )

    assert captured["task"] == "find all Python entry points"


def test_non_spawn_tool_has_no_opinion():
    assert (
        claude_routing.route_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "true"}},
            workspace=WS,
            token="token",
            available_models=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
        )
        is None
    )


def test_canary_and_audit_are_written(tmp_path, monkeypatch):
    canary = tmp_path / "canary.json"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(claude_routing, "CANARY_PATH", canary)
    monkeypatch.setattr(claude_routing, "AUDIT_PATH", audit)

    claude_routing.record_session_start({"session_id": "s1", "model": "system.ai.claude-opus-4-8"})
    claude_routing.record_subagent_start(
        {
            "session_id": "s1",
            "agent_id": "a1",
            "agent_type": "Explore",
            "model": "system.ai.claude-sonnet-5",
        }
    )

    assert json.loads(canary.read_text())["session_id"] == "s1"
    assert json.loads(audit.read_text().strip())["agent_id"] == "a1"


def test_decision_is_reconciled_with_actual_subagent_model(tmp_path, monkeypatch):
    decisions = tmp_path / "decisions.jsonl"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(claude_routing, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(claude_routing, "AUDIT_PATH", audit)
    monkeypatch.setattr(
        claude_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            claude_routing.RoutingDecision(
                model="system.ai.claude-opus-4-8", raw_model="claude-opus-4-8"
            ),
            None,
        ),
    )

    claude_routing.route_pre_tool_use(
        {
            "session_id": "s1",
            "tool_name": "Agent",
            "tool_input": {"subagent_type": "Explore", "prompt": "x"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
        audit_decision=True,
    )
    record = claude_routing.record_subagent_start(
        {"session_id": "s1", "agent_id": "a1", "model": "opus"}
    )

    assert record["router_model"] == "claude-opus-4-8"
    assert record["requested_model"] == "opus"
    assert record["matches_router_decision"] is True


def test_subagent_start_without_model_reports_unknown_not_mismatch(tmp_path, monkeypatch):
    # Claude Code's SubagentStart event does not include the subagent's model,
    # so reconciliation can't verify the match — record None (unknown) rather
    # than a false mismatch. The PreToolUse hook already injected the model.
    decisions = tmp_path / "decisions.jsonl"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(claude_routing, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(claude_routing, "AUDIT_PATH", audit)
    monkeypatch.setattr(
        claude_routing,
        "request_routing_decision",
        lambda *args, **kwargs: (
            claude_routing.RoutingDecision(
                model="system.ai.claude-sonnet-5", raw_model="claude-sonnet-5"
            ),
            None,
        ),
    )

    claude_routing.route_pre_tool_use(
        {
            "session_id": "s2",
            "tool_name": "Agent",
            "tool_input": {"subagent_type": "Explore", "prompt": "x"},
        },
        workspace=WS,
        token="token",
        available_models=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
        audit_decision=True,
    )
    record = claude_routing.record_subagent_start(
        {"session_id": "s2", "agent_id": "a2", "model": None}
    )

    assert record["requested_model"] == "sonnet"
    assert record["matches_router_decision"] is None
