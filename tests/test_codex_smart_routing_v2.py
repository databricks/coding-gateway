from __future__ import annotations

import json

import pytest

from ucode.agents import LaunchOptions, codex
from ucode.smart_routing import codex_interposer, codex_routing, v2

WS = "https://example.databricks.com"


def test_smart_routing_switch_message_is_boxed():
    message = v2.format_routing_notice("model-x", "Because X.")

    assert message == (
        "┌───────────────────────────────────────────────────────────────────────────┐\n"
        "│ Using Unity Gateway Smart Router.                                         │\n"
        "│ Selected Model : model-x                                                  │\n"
        "│ Reason : Because X.                                                       │\n"
        "│ Spawned subagents are routed independently based on their own complexity. │\n"
        "└───────────────────────────────────────────────────────────────────────────┘"
    )


class TestLaunchCodex:
    def test_rejects_unsupported_codex_version(self, monkeypatch):
        monkeypatch.setenv(v2.ENV_VAR, "1")
        monkeypatch.setattr(codex, "clear_model_preferences", lambda state: False)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.144.0")
        monkeypatch.setattr(v2, "launch_codex", lambda *args, **kwargs: pytest.fail("launched"))

        with pytest.raises(RuntimeError, match="requires Codex 0.145.0 or newer; found 0.144.0"):
            codex.launch({"workspace": WS}, [], options=LaunchOptions(launch_smart_routing=True))

    @pytest.mark.parametrize(
        ("tool_args", "options"),
        [
            ([], LaunchOptions(launch_smart_routing=True)),
            (["fix the parser"], LaunchOptions(launch_smart_routing=True)),
        ],
    )
    def test_codex_smart_routing_launch_dispatches_to_v2(self, monkeypatch, tool_args, options):
        calls = []
        monkeypatch.setenv(v2.ENV_VAR, "1")
        monkeypatch.setattr(codex, "default_model", lambda state: "gpt-start")
        monkeypatch.setattr(codex, "clear_model_preferences", lambda state: False)

        def launch_v2(state, tool_args, **kwargs):
            calls.append((state, tool_args, kwargs))
            raise SystemExit(0)

        monkeypatch.setattr(v2, "launch_codex", launch_v2)
        state = {"workspace": WS}

        with pytest.raises(SystemExit) as exc:
            codex.launch(state, tool_args, options=options)

        assert exc.value.code == 0
        assert calls == [
            (
                state,
                tool_args,
                {
                    "binary": "codex",
                    "start_model": "gpt-start",
                    "render_overlay": codex.render_overlay,
                },
            )
        ]

    @pytest.mark.parametrize(
        "tool_args",
        [
            ["exec", "fix this"],
            ["review"],
            ["app-server"],
            ["update"],
            ["--model", "gpt-5.6-sol"],
            ["--model", "gpt-5.6-sol", "--", "fix this"],
            ["fix this"],
        ],
    )
    def test_codex_launch_bypasses_routing_for_other_shapes(self, tmp_path, monkeypatch, tool_args):
        launches = []
        profile_path = tmp_path / "ucode.config.toml"
        profile_path.write_text('model_provider = "ucode-databricks"\n', encoding="utf-8")
        monkeypatch.setenv(v2.ENV_VAR, "1")
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", profile_path)
        monkeypatch.setattr(codex, "clear_model_preferences", lambda state: False)
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.144.0")
        monkeypatch.setattr(codex, "get_databricks_token", lambda *_args: "token")
        monkeypatch.setattr(v2, "launch_codex", lambda *args, **kwargs: pytest.fail("launched"))
        monkeypatch.setattr(codex, "exec_or_spawn", lambda argv: launches.append(argv))

        codex.launch(
            {"workspace": WS},
            tool_args,
            options=LaunchOptions(),
        )

        assert launches == [["codex", "--config", 'model_provider="ucode-databricks"', *tool_args]]

    def test_codex_launch_normalizes_cached_bootstrap_model(self, monkeypatch):
        calls = []
        monkeypatch.setenv(v2.ENV_VAR, "1")
        monkeypatch.setattr(codex, "clear_model_preferences", lambda state: False)
        monkeypatch.setattr(codex, "default_model", lambda state: None)

        def launch_v2(state, tool_args, **kwargs):
            calls.append(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(v2, "launch_codex", launch_v2)

        with pytest.raises(SystemExit):
            codex.launch(
                {"workspace": WS, "codex_models": ["system.ai.gpt-5-6-luna"]},
                [],
                options=LaunchOptions(launch_smart_routing=True),
            )

        assert calls[0]["start_model"] == "gpt-5.6-luna"

    def test_owns_app_server_interposer_and_tui_lifecycle(self, monkeypatch):
        processes = []
        interposer_args = {}
        stopped = []
        token_calls = []
        monkeypatch.setenv("CODEX_HOME", "/user/codex-home")
        monkeypatch.setattr(codex, "ucode_version", lambda: "0.1.0")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "0.148.0")

        class FakeProcess:
            def __init__(self, argv, **kwargs):
                self.argv = argv
                self.kwargs = kwargs
                self.terminated = False
                processes.append(self)

            def wait(self, timeout=None):
                return 0 if timeout is not None else 7

            def terminate(self):
                self.terminated = True

            def kill(self):
                raise AssertionError("clean shutdown should not need kill")

            def send_signal(self, _signal):
                raise AssertionError("test does not interrupt the TUI")

        monkeypatch.setattr(v2.subprocess, "Popen", FakeProcess)

        def get_token(workspace, profile):
            token_calls.append((workspace, profile))
            return f"token-{len(token_calls)}"

        monkeypatch.setattr(v2, "get_databricks_token", get_token)
        monkeypatch.setattr(v2, "_free_port", lambda: 41001)
        monkeypatch.setattr(v2, "_wait_for_app_server", lambda port, timeout: True)

        def start_interposer(*args, **kwargs):
            interposer_args["args"] = args
            interposer_args["kwargs"] = kwargs
            return 41002, lambda: stopped.append(True)

        monkeypatch.setattr(codex_interposer, "start_interposer_thread", start_interposer)

        with pytest.raises(SystemExit) as exc:
            v2.launch_codex(
                {
                    "workspace": WS,
                    "profile": "myprof",
                    "codex_models": ["system.ai.gpt-5-6-sol"],
                    "oss_models": ["system.ai.glm-5-2"],
                },
                ["--search"],
                binary="codex",
                start_model="gpt-start",
                render_overlay=codex.render_overlay,
            )

        assert exc.value.code == 7
        assert processes[0].argv[:7] == [
            "codex",
            "app-server",
            "--config",
            'model_provider="ucode-databricks"',
            "--config",
            'model="gpt-start"',
            "--config",
        ]
        assert processes[0].argv[7].startswith("model_providers.ucode-databricks={")
        assert processes[0].argv[8] == "--config"
        hook_override = processes[0].argv[9]
        assert hook_override.startswith("hooks.PreToolUse=[{")
        assert 'matcher = "Agent|.*spawn_agent$"' in hook_override
        assert "codex-router-hook route-subagent" in hook_override
        assert f"--host {WS}" in hook_override
        assert "--profile myprof" in hook_override
        assert "--model system.ai.gpt-5-6-sol" in hook_override
        assert "--model system.ai.glm-5-2" in hook_override
        assert processes[0].argv[10:] == [
            "--listen",
            "ws://127.0.0.1:41001",
        ]
        assert processes[0].kwargs["env"][v2.OAUTH_TOKEN_ENV_VAR] == "token-1"
        assert processes[0].kwargs["env"]["CODEX_HOME"] == "/user/codex-home"
        assert processes[1].argv == [
            "codex",
            "--remote",
            "ws://127.0.0.1:41002",
            "--model",
            "gpt-start",
            "--search",
        ]
        assert interposer_args["args"] == (v2.LOOPBACK_HOST, "ws://127.0.0.1:41001")
        assert interposer_args["kwargs"]["available_models"] == [
            "system.ai.gpt-5-6-sol",
            "system.ai.glm-5-2",
        ]
        assert interposer_args["kwargs"]["workspace"] == WS
        assert token_calls == [(WS, "myprof")]
        assert interposer_args["kwargs"]["token_provider"]() == "token-2"
        assert token_calls == [(WS, "myprof"), (WS, "myprof")]
        assert interposer_args["kwargs"]["switch_message_fn"] is v2.format_routing_notice
        assert stopped == [True]
        assert processes[0].terminated is True

    def test_v2_pre_tool_hook_preserves_user_hooks(self, tmp_path, monkeypatch):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(
            "[[hooks.PreToolUse]]\n"
            'matcher = "Bash"\n'
            "[[hooks.PreToolUse.hooks]]\n"
            'type = "command"\n'
            'command = "user-policy"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(codex_home))

        configured = v2._v2_pre_tool_use_hooks(
            {"workspace": WS, "profile": "myprof"},
            ["system.ai.gpt-5-6-sol"],
        )

        assert configured[0]["hooks"][0]["command"] == "user-policy"
        assert configured[1]["matcher"] == "Agent|.*spawn_agent$"
        assert "--model system.ai.gpt-5-6-sol" in configured[1]["hooks"][0]["command"]

    def test_v2_pre_tool_hook_replaces_existing_ucode_hook(self, tmp_path, monkeypatch):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(
            "[[hooks.PreToolUse]]\n"
            'matcher = "Agent|.*spawn_agent$"\n'
            "[[hooks.PreToolUse.hooks]]\n"
            'type = "command"\n'
            'command = "ucode codex-router-hook route-subagent --model old"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(codex_home))

        configured = v2._v2_pre_tool_use_hooks(
            {"workspace": WS, "profile": "myprof"},
            ["system.ai.gpt-5-6-sol"],
        )

        routing_commands = [
            hook["command"]
            for group in configured
            for hook in group["hooks"]
            if "codex-router-hook" in hook["command"]
        ]
        assert len(routing_commands) == 1
        assert "--model system.ai.gpt-5-6-sol" in routing_commands[0]
        assert "--model old" not in routing_commands[0]

    def test_missing_cached_models_starts_with_bootstrap_model(self, monkeypatch):
        monkeypatch.setattr(v2, "get_databricks_token", lambda workspace, profile: "token")
        monkeypatch.setattr(codex, "agent_version", lambda binary: "unknown")
        monkeypatch.setattr(v2, "_free_port", lambda: 41001)
        monkeypatch.setattr(v2, "_wait_for_app_server", lambda port, timeout: True)
        monkeypatch.setattr(
            v2.subprocess,
            "Popen",
            lambda *args, **kwargs: type(
                "Process",
                (),
                {
                    "wait": lambda self, timeout=None: 0,
                    "terminate": lambda self: None,
                    "kill": lambda self: None,
                },
            )(),
        )
        monkeypatch.setattr(
            codex_interposer,
            "start_interposer_thread",
            lambda *args, **kwargs: (41002, lambda: None),
        )

        with pytest.raises(SystemExit) as exc:
            v2.launch_codex(
                {"workspace": WS},
                [],
                binary="codex",
                start_model="gpt-5.6-luna",
                render_overlay=codex.render_overlay,
            )

        assert exc.value.code == 0


def test_interposer_startup_failure_is_propagated(monkeypatch):
    async def fail_to_serve(*args, **kwargs):
        raise OSError("bind failed")

    monkeypatch.setattr(codex_interposer, "_serve", fail_to_serve)

    with pytest.raises(RuntimeError, match="failed to start") as exc:
        codex_interposer.start_interposer_thread(
            v2.LOOPBACK_HOST,
            "ws://127.0.0.1:41001",
            "model-x",
        )

    assert isinstance(exc.value.__cause__, OSError)


class TestInterposerSession:
    def _turn_start(self, model: str, thread_id: str = "t1", prompt: str = "Fix the parser") -> str:
        return json.dumps(
            {
                "method": codex_interposer.TURN_START,
                "id": 1,
                "params": {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                    "model": model,
                },
            }
        )

    def test_switches_first_turn(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        result = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-luna"))
        assert json.loads(result.frame)["params"]["model"] == "gpt-5.5"
        assert result.needs_settings_update

    def test_does_not_schedule_notification_when_model_is_already_selected(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        frame = self._turn_start("gpt-5.5")
        assert sess.on_tui_frame(frame) == codex_interposer.TuiFrameResult(
            frame, needs_settings_update=False
        )
        assert sess.on_engine_frame(self._turn_started("turn-1")) == []
        later_selection = self._turn_start("gpt-5.6")
        assert sess.on_tui_frame(later_selection) == codex_interposer.TuiFrameResult(
            later_selection, needs_settings_update=False
        )

    def test_non_turn_frames_pass_through(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        frame = json.dumps({"method": "initialize", "id": 1, "params": {}})
        assert sess.on_tui_frame(frame) == codex_interposer.TuiFrameResult(
            frame, needs_settings_update=False
        )

    def _turn_started(self, turn_id: str, thread_id: str = "t1") -> str:
        return json.dumps(
            {
                "method": codex_interposer.TURN_STARTED,
                "params": {"threadId": thread_id, "turn": {"id": turn_id}},
            }
        )

    def test_injects_note_when_switched_turn_starts(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        settings = next(m for m in injected if m["method"] == codex_interposer.SETTINGS_UPDATED)
        assert settings["params"]["threadId"] == "t1"
        assert settings["params"]["threadSettings"]["model"] == "gpt-5.5"

    def test_injects_switch_note_as_agent_message_when_message_set(self):
        sess = codex_interposer._Session(
            "gpt-5.5", log=lambda _m: None, switch_message="selected glm-5-2 because X"
        )
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        started = next(m for m in injected if m["method"] == codex_interposer.ITEM_STARTED)
        completed = next(m for m in injected if m["method"] == codex_interposer.ITEM_COMPLETED)
        assert started["params"]["turnId"] == "turn-1"
        assert completed["params"]["turnId"] == "turn-1"
        for frame in (started, completed):
            item = frame["params"]["item"]
            assert item["type"] == "agentMessage"
            assert item["text"] == "selected glm-5-2 because X"
        assert started["params"]["item"]["id"] == completed["params"]["item"]["id"]

    def test_no_note_without_message(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        injected = sess.on_engine_frame(self._turn_started("turn-1"))
        assert [m["method"] for m in injected] == [codex_interposer.SETTINGS_UPDATED]

    def test_routes_only_first_turn_and_preserves_later_model_selection(self):
        sess = codex_interposer._Session("gpt-5.5", log=lambda _m: None)
        sess.on_tui_frame(self._turn_start("luna"))
        assert sess.on_engine_frame(self._turn_started("turn-1"))
        second_turn = self._turn_start("luna")
        assert sess.on_tui_frame(second_turn) == codex_interposer.TuiFrameResult(
            second_turn, needs_settings_update=False
        )
        assert sess.on_engine_frame(self._turn_started("turn-2")) == []

    def test_routes_first_prompt_and_uses_returned_model_and_rationale(self):
        calls = []

        def select(prompt):
            calls.append(prompt)
            return (
                codex_interposer.routing.RoutingDecision(
                    model="claude-opus-4-8",
                    raw_model="claude-opus-4-8",
                    rationale="Task classified as bugfix.",
                ),
                None,
            )

        sess = codex_interposer._Session(
            None,
            log=lambda _m: None,
            available_models=["claude-opus-4-8", "gpt-5.5"],
            route_decision=select,
            switch_message_fn=v2.format_routing_notice,
        )

        result = sess.on_tui_frame(self._turn_start("gpt-5.5", prompt="Fix issue #42"))

        assert calls == ["Fix issue #42"]
        assert json.loads(result.frame)["params"]["model"] == "claude-opus-4-8"
        assert result.needs_settings_update
        assert "Task classified as bugfix." in sess.switch_message

    def test_maps_selected_uc_gpt_model_and_shows_routing_notice(self):
        def select(_prompt):
            return (
                codex_interposer.routing.RoutingDecision(
                    model="system.ai.gpt-5-6-luna",
                    raw_model="gpt-5-6-luna",
                    rationale="Trivial task.",
                ),
                None,
            )

        sess = codex_interposer._Session(
            None,
            log=lambda _m: None,
            route_decision=select,
            switch_message_fn=v2._switch_message,
        )
        frame = self._turn_start("system.ai.gpt-5-6-luna")

        result = sess.on_tui_frame(frame)
        assert json.loads(result.frame)["params"]["model"] == "gpt-5.6-luna"
        injected = sess.on_engine_frame(self._turn_started("turn-1"))

        assert [message["method"] for message in injected] == [
            codex_interposer.SETTINGS_UPDATED,
            codex_interposer.ITEM_STARTED,
            codex_interposer.ITEM_COMPLETED,
        ]
        assert "Selected Model : gpt-5.6-luna" in (injected[1]["params"]["item"]["text"])

    def test_routes_first_prompt_to_oss_model(self):
        def select(_prompt):
            return (
                codex_interposer.routing.RoutingDecision(
                    model="system.ai.glm-5-2",
                    raw_model="glm-5-2",
                    rationale="Short isolated task.",
                ),
                None,
            )

        sess = codex_interposer._Session(
            None,
            log=lambda _m: None,
            available_models=["system.ai.gpt-5-6-sol", "system.ai.glm-5-2"],
            route_decision=select,
            switch_message_fn=v2._switch_message,
        )

        result = sess.on_tui_frame(self._turn_start("system.ai.gpt-5-6-sol"))

        assert json.loads(result.frame)["params"]["model"] == "system.ai.glm-5-2"
        assert result.needs_settings_update
        assert "Selected Model : system.ai.glm-5-2" in sess.switch_message

    def test_router_failure_keeps_original_model(self):
        sess = codex_interposer._Session(
            None,
            log=lambda _m: None,
            route_decision=lambda prompt: (None, "router unavailable"),
        )
        frame = self._turn_start("gpt-start")

        assert sess.on_tui_frame(frame) == codex_interposer.TuiFrameResult(
            frame, needs_settings_update=False
        )

    def test_rewrites_nested_collaboration_mode_model(self):
        """The app-server re-derives the thread model from
        collaborationMode.settings.model on every turn/start, so the
        interposer must rewrite that nested field too — not just the
        top-level ``model`` field."""
        sess = codex_interposer._Session(
            None,
            log=lambda _m: None,
            route_decision=lambda _p: (
                codex_interposer.routing.RoutingDecision(
                    model="gpt-5.6-luna",
                    raw_model="gpt-5-6-luna",
                    rationale="trivial",
                ),
                None,
            ),
            switch_message_fn=v2._switch_message,
        )
        frame = json.dumps(
            {
                "method": codex_interposer.TURN_START,
                "id": 1,
                "params": {
                    "threadId": "t1",
                    "input": [{"type": "text", "text": "hello"}],
                    "model": "gpt-6-astra",
                    "collaborationMode": {
                        "mode": "default",
                        "settings": {
                            "model": "gpt-6-astra",
                            "reasoning_effort": "high",
                        },
                    },
                },
            }
        )
        result = sess.on_tui_frame(frame)
        parsed = json.loads(result.frame)
        assert parsed["params"]["model"] == "gpt-5.6-luna"
        assert parsed["params"]["collaborationMode"]["settings"]["model"] == "gpt-5.6-luna"
        assert result.needs_settings_update


def test_routing_request_uses_models_prompt_and_same_token(monkeypatch):
    monkeypatch.delenv("SMART_ROUTER_NAME", raising=False)
    captured = {}
    logged = []

    def select_route(workspace, token, task, route_options, resolve, *, router_name, timeout):
        captured.update(
            workspace=workspace,
            token=token,
            task=task,
            route_options=list(route_options),
            router_name=router_name,
            timeout=timeout,
        )
        return (
            codex_interposer.routing.RoutingDecision(
                model=resolve("gpt-5-6-sol"),
                raw_model="gpt-5-6-sol",
                rationale="Bugfix needs deeper reasoning.",
            ),
            None,
        )

    monkeypatch.setattr(codex_routing.routing, "select_route", select_route)

    decision, reason = codex_routing.request_routing_decision(
        WS,
        "same-oauth-token",
        "Fix the parser",
        [
            "system.ai.kimi-k3-neo",
            "system.ai.gpt-5-6-sol",
            "system.ai.gpt-5-6-luna",
            "system.ai.glm-5-2",
        ],
        log=logged.append,
    )

    assert reason is None
    assert decision.model == "system.ai.gpt-5-6-sol"
    assert captured == {
        "workspace": WS,
        "token": "same-oauth-token",
        "task": "Fix the parser",
        "router_name": codex_routing.routing.ROUTER_NAME,
        "timeout": codex_routing.REQUEST_TIMEOUT_S,
        "route_options": [
            ("kimi-k3-neo", "codex"),
            ("gpt-5-6-sol", "codex"),
            ("gpt-5-6-luna", "codex"),
            ("glm-5-2", "codex"),
        ],
    }
    assert len(logged) == 1
    assert logged[0].startswith(f"[ROUTE] request POST {WS}/ai-gateway/routing/v1/routes:select: ")
    request_payload = json.loads(logged[0].split(": ", 1)[1])
    assert request_payload == {
        "route_options": [
            {"model": "kimi-k3-neo", "harness": "codex"},
            {"model": "gpt-5-6-sol", "harness": "codex"},
            {"model": "gpt-5-6-luna", "harness": "codex"},
            {"model": "glm-5-2", "harness": "codex"},
        ],
        "task": {"prompt": "Fix the parser"},
        "route_selector": {"router_name": codex_routing.routing.ROUTER_NAME},
    }
    assert "same-oauth-token" not in logged[0]
