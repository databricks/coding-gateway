"""Tests for Claude's experimental first-prompt PTY routing path."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from ucode.agents import claude
from ucode.databricks import AnthropicModelCatalog
from ucode.smart_routing import claude_hooks, claude_pty, routing, v2


class TestManagedModelPicker:
    def test_reads_model_ids_from_managed_picker(self, tmp_path, monkeypatch):
        path = tmp_path / "managed-settings.json"
        path.write_text(
            json.dumps(
                {
                    "modelPicker": {
                        "options": [
                            {"model": "system.ai.claude-opus-4-8", "label": "Opus"},
                            {"model": "system.ai.claude-sonnet-5", "label": "Sonnet"},
                        ]
                    }
                }
            )
        )
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: path)

        catalog = v2._model_picker_catalog()

        assert catalog is not None
        assert catalog.model_ids == ["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"]
        assert catalog.model_id_to_display_name == {}

    def test_ignores_empty_or_missing_picker(self, tmp_path, monkeypatch):
        path = tmp_path / "managed-settings.json"
        path.write_text(json.dumps({"env": {}}))
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", tmp_path / "ucode-settings.json")

        assert v2._model_picker_catalog() is None

    def test_falls_back_to_ucode_settings_picker(self, tmp_path, monkeypatch):
        managed = tmp_path / "managed-settings.json"
        managed.write_text(json.dumps({"env": {}}))
        ucode_settings = tmp_path / "ucode-settings.json"
        ucode_settings.write_text(
            json.dumps({"modelPicker": {"options": [{"model": "system.ai.claude-opus-5"}]}})
        )
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: managed)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", ucode_settings)

        catalog = v2._model_picker_catalog()

        assert catalog is not None
        assert catalog.model_ids == ["system.ai.claude-opus-5"]

    def test_falls_back_to_user_settings_picker(self, tmp_path, monkeypatch):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(
            json.dumps({"modelPicker": {"options": [{"model": "system.ai.claude-sonnet-5"}]}})
        )
        monkeypatch.setattr(claude, "_managed_settings_path", lambda: tmp_path / "missing-managed")
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", tmp_path / "missing-ucode")
        monkeypatch.setattr(claude, "CLAUDE_USER_SETTINGS_PATH", user_settings)

        catalog = v2._model_picker_catalog()

        assert catalog is not None
        assert catalog.model_ids == ["system.ai.claude-sonnet-5"]


class TestDirectModelCommand:
    @pytest.mark.parametrize(
        "name",
        ["system.ai.claude-opus-4-8[1m]", "databricks-claude-sonnet-5", "opus"],
    )
    def test_accepts_model_names(self, name):
        assert claude_pty.valid_model_name(name)

    @pytest.mark.parametrize("name", ["", "a b", "a\nb", "x" * 201, None])
    def test_rejects_unsafe_model_names(self, name):
        assert not claude_pty.valid_model_name(name)


class TestFirstPromptHook:
    def test_renders_boxed_router_notice(self):
        model = "system.ai.claude-sonnet-4-6[1m]"
        reason = "Routed to Sonnet because the task is narrowly scoped."
        result = claude_pty.first_prompt_hook_output(
            {"action": "block", "model": model, "rationale": reason}
        )

        assert result == {
            "decision": "block",
            "reason": v2.format_routing_notice(model, reason),
        }

    def test_omits_reason_when_router_returns_none(self):
        result = claude_pty.first_prompt_hook_output(
            {"action": "block", "model": "system.ai.claude-sonnet-5"}
        )

        assert "Reason" not in result["reason"]

    def test_displays_catalog_name_while_retaining_routable_model(self):
        result = claude_pty.first_prompt_hook_output(
            {
                "action": "block",
                "model": "anthropic-aigw-77df06ea-system.ai.glm-5-3-flash",
                "display_model": "GLM 5.3 Flash",
            }
        )

        assert "Selected Model : GLM 5.3 Flash" in result["reason"]
        assert "anthropic-aigw-77df06ea" not in result["reason"]

    def test_blocks_once_then_allows_replay(self, tmp_path):
        socket_path = tmp_path / "first.sock"
        blocked: list[tuple[str, str]] = []
        stop = threading.Event()
        claude_pty.serve_first_prompt_socket(
            socket_path,
            lambda _prompt: claude_pty.FirstPromptRoute(
                model="sonnet", display_model="sonnet", rationale="Selected for a narrow task."
            ),
            lambda prompt, model: blocked.append((prompt, model)),
            stop,
        )
        try:
            deadline = time.monotonic() + 5
            while not socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            first = claude_pty.request_first_prompt_route(
                socket_path, {"session_id": "s1", "prompt": "fix the parser"}
            )
            replay = claude_pty.request_first_prompt_route(
                socket_path, {"session_id": "s1", "prompt": "fix the parser"}
            )
            assert first == {
                "action": "block",
                "model": "sonnet",
                "display_model": "sonnet",
                "rationale": "Selected for a narrow task.",
            }
            assert replay == {"action": "allow"}
            assert blocked == [("fix the parser", "sonnet")]
        finally:
            stop.set()

    def test_first_prompt_hook_is_per_launch(self):
        settings = {"hooks": {"PreToolUse": [{"hooks": [{"command": "user-policy"}]}]}}
        claude_hooks.sync_first_prompt_hook(settings, "/bin/ucode")
        claude_hooks.sync_first_prompt_hook(settings, "/bin/ucode")
        command = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert command == "/bin/ucode claude-router-hook route-first-prompt"
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1
        assert "user-policy" in str(settings["hooks"]["PreToolUse"])


class TestV2Launch:
    def test_strips_gateway_prefix_for_interposer(self):
        model = "anthropic-aigw-73ea02b2-system.ai.glm-5-2"
        assert v2._unwrapped_claude_model_id(model) == "system.ai.glm-5-2"

    def test_restores_model_captured_immediately_before_switch(self, tmp_path, monkeypatch):
        ucode_settings = tmp_path / "ucode-settings.json"
        user_settings = tmp_path / "settings.json"
        ucode_settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw"}}))
        user_settings.write_text(json.dumps({"model": "opus", "theme": "dark"}))
        monkeypatch.setattr(claude, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", ucode_settings)
        monkeypatch.setattr(claude, "CLAUDE_USER_SETTINGS_PATH", user_settings)
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        monkeypatch.setattr(v2, "CLAUDE_PTY_LOG", tmp_path / "v2.log")
        monkeypatch.setattr(v2, "get_databricks_token", lambda *_args, **_kwargs: "token")
        monkeypatch.setattr(v2, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])
        monkeypatch.setattr(
            v2,
            "list_anthropic_model_catalog",
            lambda *_args: AnthropicModelCatalog(
                model_ids=["system.ai.claude-opus-4-8", "system.ai.claude-sonnet-5"],
                model_id_to_display_name={"system.ai.claude-sonnet-5": "Claude Sonnet 5"},
            ),
        )
        monkeypatch.setattr(
            v2,
            "_route_claude_prompt",
            lambda *_args: v2.routing.RoutingDecision(
                model="system.ai.claude-sonnet-5",
                raw_model="claude-sonnet-5",
                rationale="Selected for the parser task.",
            ),
        )
        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            agents_index = argv.index("--agents")
            captured["agents"] = json.loads(argv[agents_index + 1])
            captured["routed_model"] = kwargs["route_prompt"]("fix the parser")
            generated = Path(argv[argv.index("--settings") + 1])
            captured["settings"] = json.loads(generated.read_text())
            # A user-selected model before the first prompt becomes the value to restore.
            user_settings.write_text(json.dumps({"model": "haiku", "theme": "dark"}))
            kwargs["prepare_model_switch"]("system.ai.claude-sonnet-5")
            # Simulate `/model` changing the user file, plus an unrelated concurrent edit.
            user_settings.write_text(json.dumps({"model": "routed", "theme": "light", "new": True}))
            kwargs["restore_model_setting"]()
            captured["restored_during_run"] = json.loads(user_settings.read_text())
            # A later user choice must survive session exit.
            user_settings.write_text(
                json.dumps({"model": "user-selected", "theme": "light", "new": True})
            )
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit) as exc:
            v2.launch_claude(
                {"workspace": "https://example.com"},
                ["--debug"],
                binary="claude",
                user_settings_path=user_settings,
                launch_model="opus",
                compose_settings=claude._compose_v2_settings,
                launch_model_args=claude._launch_model_args,
                model_name=claude._maybe_add_1m_suffix,
            )

        assert exc.value.code == 0
        assert captured["argv"][3:5] == ["--model", "opus"]
        assert captured["argv"][-1] == "--debug"
        assert captured["routed_model"] == claude_pty.FirstPromptRoute(
            model="system.ai.claude-sonnet-5[1m]",
            display_model="Claude Sonnet 5",
            rationale="Selected for the parser task.",
        )
        assert {definition["model"] for definition in captured["agents"].values()} == {
            "system.ai.claude-opus-4-8",
            "system.ai.claude-sonnet-5",
        }
        assert captured["settings"]["modelOverrides"] == {
            "claude-opus-4-8": "system.ai.claude-opus-4-8",
            "claude-sonnet-5": "system.ai.claude-sonnet-5",
        }
        assert claude_hooks.FIRST_PROMPT_SOCKET_ENV in captured["settings"]["env"]
        first_prompt_command = captured["settings"]["hooks"]["UserPromptSubmit"][0]["hooks"][0][
            "command"
        ]
        assert first_prompt_command == "ucode claude-router-hook route-first-prompt"
        route_commands = [
            hook["command"]
            for group in captured["settings"]["hooks"]["PreToolUse"]
            for hook in group["hooks"]
            if "route-subagent" in hook["command"]
        ]
        assert len(route_commands) == 1
        assert "--model system.ai.claude-opus-4-8" in route_commands[0]
        assert "--model system.ai.claude-sonnet-5" in route_commands[0]
        assert "modelPicker" not in captured["settings"]
        assert captured["restored_during_run"] == {
            "model": "haiku",
            "theme": "light",
            "new": True,
        }
        assert json.loads(user_settings.read_text()) == {
            "model": "user-selected",
            "theme": "light",
            "new": True,
        }

    def test_does_not_restore_when_wrapper_never_switches(self, tmp_path, monkeypatch):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(json.dumps({"model": "opus"}))
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        monkeypatch.setattr(v2, "get_databricks_token", lambda *_args, **_kwargs: "token")
        monkeypatch.setattr(v2, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])
        monkeypatch.setattr(
            v2,
            "list_anthropic_model_catalog",
            lambda *_args: AnthropicModelCatalog(model_ids=["opus"], model_id_to_display_name={}),
        )

        def fake_run(_argv, **_kwargs):
            user_settings.write_text(json.dumps({"model": "user-selected"}))
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit):
            v2.launch_claude(
                {"workspace": "https://example.com"},
                [],
                binary="claude",
                user_settings_path=user_settings,
                launch_model="opus",
                compose_settings=lambda _args: ({}, []),
                launch_model_args=claude._launch_model_args,
                model_name=claude._maybe_add_1m_suffix,
            )

        assert json.loads(user_settings.read_text()) == {"model": "user-selected"}

    def test_restores_after_routed_model_persists_and_preserves_later_choice(
        self, tmp_path, monkeypatch
    ):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(json.dumps({"model": "haiku", "theme": "dark"}))
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        monkeypatch.setattr(v2, "get_databricks_token", lambda *_args, **_kwargs: "token")
        monkeypatch.setattr(v2, "build_auth_token_argv", lambda *_args, **_kwargs: ["ucode"])
        monkeypatch.setattr(
            v2,
            "list_anthropic_model_catalog",
            lambda *_args: AnthropicModelCatalog(model_ids=["opus"], model_id_to_display_name={}),
        )

        def fake_run(_argv, **kwargs):
            routed_model = "system.ai.claude-opus-4-8"
            kwargs["prepare_model_switch"](routed_model)
            user_settings.write_text(json.dumps({"model": routed_model, "theme": "dark"}))
            assert kwargs["model_switch_persisted"]() is True
            kwargs["restore_model_setting"]()
            assert json.loads(user_settings.read_text()) == {"model": "haiku", "theme": "dark"}
            # The same model chosen explicitly later in the session must survive.
            user_settings.write_text(json.dumps({"model": v2.CLAUDE_TARGET_MODEL, "theme": "dark"}))
            return 0

        monkeypatch.setattr(claude_pty, "run_claude_pty", fake_run)
        with pytest.raises(SystemExit):
            v2.launch_claude(
                {"workspace": "https://example.com"},
                [],
                binary="claude",
                user_settings_path=user_settings,
                launch_model=None,
                compose_settings=lambda _args: ({}, []),
                launch_model_args=claude._launch_model_args,
                model_name=claude._maybe_add_1m_suffix,
            )

        assert json.loads(user_settings.read_text()) == {
            "model": v2.CLAUDE_TARGET_MODEL,
            "theme": "dark",
        }


class TestSubagentRouting:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("anthropic-aigw-73ea02b2-system.ai.glm-5-2", "glm-5-2"),
            (
                "anthropic-aigw-73ea02b-system.ai.glm-5-2",
                "anthropic-aigw-73ea02b-system.ai.glm-5-2",
            ),
            ("system.ai.claude-opus-4-8", "claude-opus-4-8"),
        ],
    )
    def test_normalizes_router_model_id(self, model, expected):
        assert v2._claude_router_model_id(model) == expected

    def test_routes_anthropic_gateway_alias_by_embedded_model_id(self, monkeypatch):
        gateway_alias = "anthropic-aigw-73ea02b2-system.ai.glm-5-2"
        captured = {}
        monkeypatch.setenv("SMART_ROUTER_NAME", "task_v2")

        def fake_select(workspace, token, task, route_options, resolve, **kwargs):
            captured["route_options"] = list(route_options)
            captured["router_name"] = kwargs["router_name"]
            return (
                routing.RoutingDecision(
                    model=resolve("glm-5-2"),
                    raw_model="glm-5-2",
                ),
                None,
            )

        monkeypatch.setattr(routing, "select_route", fake_select)
        decision, error = v2._request_claude_routing_decision(
            "https://example.com",
            "secret-token",
            "inspect the parser",
            ["system.ai.claude-opus-4-8", gateway_alias],
        )

        assert error is None
        assert decision.model == gateway_alias
        assert captured == {
            "route_options": [
                ("claude-opus-4-8", "claude"),
                ("glm-5-2", "claude"),
            ],
            "router_name": "task_v2",
        }

    def test_routes_agent_prompt_with_initialized_model_menu(self, tmp_path, monkeypatch):
        captured = {}
        decisions_path = tmp_path / "decisions.jsonl"
        monkeypatch.setattr(v2.claude_routing, "DECISIONS_PATH", decisions_path)

        def fake_select(workspace, token, task, route_options, resolve, **kwargs):
            captured.update(
                workspace=workspace,
                token=token,
                task=task,
                route_options=list(route_options),
            )
            return (
                routing.RoutingDecision(
                    model=resolve("claude-opus-4-8"),
                    raw_model="claude-opus-4-8",
                ),
                None,
            )

        monkeypatch.setattr(routing, "select_route", fake_select)
        output = v2.route_claude_pre_tool_use(
            {
                "tool_name": "Agent",
                "tool_input": {"prompt": "inspect the parser", "model": "sonnet"},
            },
            workspace="https://example.com",
            token="token",
            available_models=[
                "system.ai.claude-opus-4-8",
                "databricks-claude-sonnet-5",
            ],
            audit_decision=True,
        )

        assert captured == {
            "workspace": "https://example.com",
            "token": "token",
            "task": "inspect the parser",
            "route_options": [
                ("claude-opus-4-8", "claude"),
                ("claude-sonnet-5", "claude"),
            ],
        }
        updated_input = output["hookSpecificOutput"]["updatedInput"]
        assert "model" not in updated_input
        assert updated_input["subagent_type"] == v2._routed_claude_agent_name(
            "system.ai.claude-opus-4-8"
        )
        expected_message = routing.format_subagent_message(
            "system.ai.claude-opus-4-8",
            "",
        )
        assert output["systemMessage"] == expected_message
        assert output["hookSpecificOutput"]["permissionDecisionReason"] == expected_message
        decision_record = json.loads(decisions_path.read_text())
        assert decision_record["requested_model"] == "system.ai.claude-opus-4-8"

    def test_merges_caller_agents_with_transient_routed_agents(self):
        args = v2._with_routed_claude_agents(
            [
                "--agents",
                json.dumps(
                    {
                        "reviewer": {
                            "description": "Reviews code",
                            "prompt": "Review the requested code.",
                        }
                    }
                ),
                "--debug",
            ],
            ["databricks-claude-opus-4-8"],
        )

        assert args[0] == "--agents"
        definitions = json.loads(args[1])
        assert definitions["reviewer"]["prompt"] == "Review the requested code."
        routed = definitions[v2._routed_claude_agent_name("system.ai.claude-opus-4-8")]
        assert routed["model"] == "system.ai.claude-opus-4-8"
        assert args[2:] == ["--debug"]

    def test_leaves_non_claude_custom_agent_model_unchanged(self):
        definitions = v2._routed_claude_agent_definitions(["catalog.schema.gpt-5"])

        assert next(iter(definitions.values()))["model"] == "catalog.schema.gpt-5"

    def test_maps_gateway_claude_ids_to_known_model_metadata(self):
        assert v2._claude_model_overrides(
            [
                "system.ai.claude-opus-4-8",
                "databricks-claude-sonnet-5",
                "catalog.schema.gpt-5",
            ]
        ) == {
            "claude-opus-4-8": "system.ai.claude-opus-4-8",
            "claude-sonnet-5": "system.ai.claude-sonnet-5",
        }

    def test_model_switch_lock_serializes_routed_sessions(self, tmp_path, monkeypatch):
        user_settings = tmp_path / "settings.json"
        user_settings.write_text(json.dumps({"model": "haiku"}))
        monkeypatch.setattr(v2, "APP_DIR", tmp_path)
        first = v2._ClaudeModelSettingGuard(user_settings)
        second = v2._ClaudeModelSettingGuard(user_settings)
        captured: list[str] = []

        first.begin("system.ai.claude-opus-4-8")
        user_settings.write_text(json.dumps({"model": "system.ai.claude-opus-4-8"}))

        def run_second() -> None:
            second.begin("system.ai.claude-sonnet-5")
            captured.append(json.loads(user_settings.read_text())["model"])
            second.restore()

        thread = threading.Thread(target=run_second)
        thread.start()
        first.restore()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert captured == ["haiku"]
        assert json.loads(user_settings.read_text()) == {"model": "haiku"}


class TestPtyFlow:
    def test_does_not_launch_when_socket_startup_fails(self, tmp_path, monkeypatch):
        class StoppedThread:
            @staticmethod
            def is_alive():
                return False

        monkeypatch.setattr(
            claude_pty,
            "serve_first_prompt_socket",
            lambda *_args, **_kwargs: StoppedThread(),
        )
        monkeypatch.setattr(
            claude_pty.pty,
            "fork",
            lambda: pytest.fail("Claude must not launch without the routing socket"),
        )

        with pytest.raises(RuntimeError, match="Claude was not launched"):
            claude_pty.run_claude_pty(
                ["claude"],
                route_prompt=lambda _prompt: claude_pty.FirstPromptRoute(
                    model="sonnet", display_model="sonnet", rationale=""
                ),
                socket_path=tmp_path / "missing.sock",
            )

    def test_direct_switch_restore_and_replay(self, tmp_path):
        fake_claude = tmp_path / "fake_claude.py"
        capture = tmp_path / "capture.json"
        restored = tmp_path / "restored"
        socket_path = tmp_path / "first.sock"
        fake_claude.write_text(
            """
import json
import os
import socket
import sys
import tty
from pathlib import Path

socket_path = sys.argv[1]
capture_path = Path(sys.argv[2])
restored_path = Path(sys.argv[3])
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(socket_path)
client.sendall((json.dumps({
    "method": "route_first_prompt",
    "prompt": "fix\\nthe parser",
    "session_id": "s1",
}) + "\\n").encode())
response = client.makefile("rb").readline()
client.close()
assert json.loads(response)["action"] == "block"
print("Smart Routing blocked the prompt", flush=True)
tty.setraw(0)

def read_until(suffix):
    data = b""
    while not data.endswith(suffix):
        data += os.read(0, 1)
    return data

model_command = read_until(b"\\r")
print("Set model to system.ai.claude-sonnet-5", flush=True)
replayed = read_until(b"\\x1b[201~\\r")
capture_path.write_text(json.dumps({
    "command": model_command.decode(),
    "replayed": replayed.decode(),
    "restored_before_replay": restored_path.exists(),
}))
""".lstrip()
        )
        result = claude_pty.run_claude_pty(
            [
                sys.executable,
                str(fake_claude),
                str(socket_path),
                str(capture),
                str(restored),
            ],
            route_prompt=lambda _prompt: claude_pty.FirstPromptRoute(
                model="system.ai.claude-sonnet-5",
                display_model="system.ai.claude-sonnet-5",
                rationale="",
            ),
            socket_path=socket_path,
            restore_model_setting=lambda: restored.write_text("restored"),
        )

        assert result == 0
        assert json.loads(capture.read_text()) == {
            "command": "/model system.ai.claude-sonnet-5\r",
            "replayed": "\x1b[200~fix\nthe parser\x1b[201~\r",
            "restored_before_replay": True,
        }
