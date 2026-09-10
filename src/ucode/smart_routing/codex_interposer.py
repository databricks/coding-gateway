from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ucode.smart_routing import codex_routing, routing

SETTINGS_UPDATED = "thread/settings/updated"
SETTINGS_UPDATE = "thread/settings/update"
ITEM_STARTED = "item/started"
ITEM_COMPLETED = "item/completed"
TURN_START = "turn/start"
TURN_STARTED = "turn/started"

RouteDecisionFn = Callable[[str], tuple[routing.RoutingDecision | None, str | None]]
SwitchMessageFn = Callable[[str, str], str]
TokenProvider = Callable[[], str]


def _prompt_from_turn(params: dict) -> str | None:
    """Extract the plaintext portions of a Codex ``turn/start`` input."""
    raw_input = params.get("input")
    if isinstance(raw_input, str):
        return raw_input if raw_input.strip() else None
    if not isinstance(raw_input, list):
        return None

    parts: list[str] = []
    for item in raw_input:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    prompt = "\n".join(part for part in parts if part.strip())
    return prompt or None


@dataclass
class TuiFrameResult:
    frame: str
    needs_settings_update: bool


class _Session:
    def __init__(
        self,
        target_model: str | None,
        log: Callable[[str], None],
        switch_message: str | None = None,
        available_models: list[str] | None = None,
        route_decision: RouteDecisionFn | None = None,
        switch_message_fn: SwitchMessageFn | None = None,
    ) -> None:
        self.target = target_model
        self.available_models = list(available_models or [])
        self.log = log
        self.switch_message = switch_message
        self.route_decision = route_decision
        self.switch_message_fn = switch_message_fn
        self.thread_id: str | None = None
        self.settings: dict | None = None
        self.first_turn_seen = False
        self.switch_pending = False
        self.notice_pending = False
        self.injected = False
        self.settings_update_id: str | None = None
        self.settings_update_event: asyncio.Event = asyncio.Event()

    def on_tui_frame(self, raw: str) -> TuiFrameResult:
        try:
            msg = json.loads(raw)
        except ValueError:
            return TuiFrameResult(raw, needs_settings_update=False)
        if not isinstance(msg, dict):
            return TuiFrameResult(raw, needs_settings_update=False)
        params = msg.get("params")
        if msg.get("method") == TURN_START and isinstance(params, dict):
            if isinstance(params.get("threadId"), str):
                self.thread_id = params["threadId"]
            if self.first_turn_seen:
                return TuiFrameResult(raw, needs_settings_update=False)
            self.first_turn_seen = True
            if self.route_decision is not None:
                prompt = _prompt_from_turn(params)
                if prompt is None:
                    self.log("[ROUTE] first turn had no plaintext prompt; keeping current model")
                    return TuiFrameResult(raw, needs_settings_update=False)
                decision, reason = self.route_decision(prompt)
                if decision is None:
                    self.log(f"[ROUTE] selection failed; keeping current model: {reason}")
                    return TuiFrameResult(raw, needs_settings_update=False)
                decision = replace(decision, model=codex_routing.codex_model_id(decision.model))
                self.target = decision.model
                if self.switch_message_fn is not None:
                    self.switch_message = self.switch_message_fn(decision.model, decision.rationale)
                self.notice_pending = self.switch_message is not None
                self.log(f"[ROUTE] selected {decision.model!r}; rationale={decision.rationale!r}")
            old = params.get("model")
            if self.target is not None and old != self.target:
                params["model"] = self.target
                collab = params.get("collaborationMode")
                if isinstance(collab, dict):
                    settings = collab.get("settings")
                    if isinstance(settings, dict) and isinstance(settings.get("model"), str):
                        settings["model"] = self.target
                self.switch_pending = True
                self.log(f"[REWRITE] model {old!r} -> {self.target!r}")
                return TuiFrameResult(json.dumps(msg), needs_settings_update=True)
        return TuiFrameResult(raw, needs_settings_update=False)

    def on_engine_frame(self, raw: str) -> list[dict]:
        try:
            msg = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(msg, dict):
            return []
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        result = msg.get("result") if isinstance(msg.get("result"), dict) else {}
        for src in (params, result):
            thread = src.get("thread")
            tid = src.get("threadId") or (thread.get("id") if isinstance(thread, dict) else None)
            if isinstance(tid, str):
                self.thread_id = tid
            ts = src.get("threadSettings")
            if isinstance(ts, dict):
                self.settings = ts
        if (
            self.settings_update_id is not None
            and isinstance(msg.get("id"), str)
            and msg["id"] == self.settings_update_id
        ):
            self.settings_update_id = None
            self.settings_update_event.set()
        if (
            msg.get("method") == TURN_STARTED
            and not self.injected
            and (self.switch_pending or self.notice_pending)
            and self.thread_id
        ):
            self.injected = True
            switch_pending = self.switch_pending
            self.switch_pending = False
            self.notice_pending = False
            injected: list[dict] = []
            if switch_pending:
                settings = dict(self.settings) if isinstance(self.settings, dict) else {}
                settings["model"] = self.target
                self.log(f"[INJECT] {SETTINGS_UPDATED}: model -> {self.target!r} (flip TUI chip)")
                injected.append(
                    {
                        "method": SETTINGS_UPDATED,
                        "params": {"threadId": self.thread_id, "threadSettings": settings},
                    }
                )
            if self.switch_message:
                turn = params.get("turn")
                turn_id = turn.get("id") if isinstance(turn, dict) else None
                now_ms = int(time.time() * 1000)
                item = {
                    "type": "agentMessage",
                    "id": f"ucode-smart-router-{uuid.uuid4().hex}",
                    "text": self.switch_message,
                    "phase": None,
                    "memoryCitation": None,
                }
                self.log(f"[INJECT] agentMessage note (started+completed): {self.switch_message!r}")
                injected.append(
                    {
                        "method": ITEM_STARTED,
                        "params": {
                            "item": item,
                            "threadId": self.thread_id,
                            "turnId": turn_id,
                            "startedAtMs": now_ms,
                        },
                    }
                )
                injected.append(
                    {
                        "method": ITEM_COMPLETED,
                        "params": {
                            "item": item,
                            "threadId": self.thread_id,
                            "turnId": turn_id,
                            "completedAtMs": now_ms,
                        },
                    }
                )
            return injected
        return []


async def _handle_tui(
    tui,
    upstream_uri: str,
    target_model: str | None,
    log,
    switch_message: str | None = None,
    available_models: list[str] | None = None,
    workspace: str | None = None,
    token_provider: TokenProvider | None = None,
    switch_message_fn: SwitchMessageFn | None = None,
) -> None:
    path = getattr(getattr(tui, "request", None), "path", "/") or "/"
    uri = upstream_uri.rstrip("/") + path
    log(f"[CONN] TUI connected (path={path}); dialing app-server {uri}")
    route_decision: RouteDecisionFn | None = None
    if workspace is not None and token_provider is not None:

        def route_decision(prompt: str):
            try:
                token = token_provider()
            except RuntimeError as exc:
                return None, f"could not refresh workspace auth: {exc}"
            return codex_routing.request_routing_decision(
                workspace,
                token,
                prompt,
                list(available_models or []),
                log=log,
            )

    sess = _Session(
        target_model,
        log,
        switch_message,
        available_models,
        route_decision,
        switch_message_fn,
    )
    async with connect(uri, max_size=None) as upstream:

        async def tui_to_app():
            async for frame in tui:
                if isinstance(frame, str):
                    result = await asyncio.to_thread(sess.on_tui_frame, frame)
                    if result.needs_settings_update and sess.thread_id:
                        update_id = f"ucode-route-{uuid.uuid4().hex}"
                        update_req = json.dumps(
                            {
                                "id": update_id,
                                "method": SETTINGS_UPDATE,
                                "params": {
                                    "threadId": sess.thread_id,
                                    "model": sess.target,
                                },
                            }
                        )
                        sess.settings_update_id = update_id
                        log(
                            f"[ROUTE] sending {SETTINGS_UPDATE} model={sess.target!r} to app-server"
                        )
                        await upstream.send(update_req)
                        try:
                            await asyncio.wait_for(sess.settings_update_event.wait(), timeout=10)
                            log("[ROUTE] settings/update confirmed by app-server")
                        except TimeoutError:
                            log("[ROUTE] settings/update timed out; forwarding turn/start anyway")
                    await upstream.send(result.frame)
                else:
                    await upstream.send(frame)

        async def app_to_tui():
            async for frame in upstream:
                await tui.send(frame)
                if isinstance(frame, str):
                    for inj in sess.on_engine_frame(frame):
                        await tui.send(json.dumps(inj))

        a = asyncio.create_task(tui_to_app())
        b = asyncio.create_task(app_to_tui())
        done, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    log("[CONN] TUI session closed")


async def _serve(
    host: str,
    port: int,
    upstream_uri: str,
    model: str | None,
    log,
    switch_message: str | None = None,
    available_models: list[str] | None = None,
    workspace: str | None = None,
    token_provider: TokenProvider | None = None,
    switch_message_fn: SwitchMessageFn | None = None,
):
    async def handler(tui):
        try:
            await _handle_tui(
                tui,
                upstream_uri,
                model,
                log,
                switch_message,
                available_models,
                workspace,
                token_provider,
                switch_message_fn,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[ERR] session: {exc!r}")

    server = await serve(handler, host, port, max_size=None)
    bound_port = server.sockets[0].getsockname()[1]
    log(f"[READY] ws://{host}:{bound_port} -> {upstream_uri} (switch -> {model!r})")
    return server


def start_interposer_thread(
    host: str,
    upstream_uri: str,
    model: str | None = None,
    *,
    available_models: list[str] | None = None,
    workspace: str | None = None,
    token_provider: TokenProvider | None = None,
    switch_message_fn: SwitchMessageFn | None = None,
    switch_message: str | None = None,
    log_path: Path | None = None,
    ready_timeout: float = 10.0,
) -> tuple[int, Callable[[], None]]:
    def log(message: str) -> None:
        if log_path is None:
            return
        line = f"{time.strftime('%H:%M:%S')} {message}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass

    loop = asyncio.new_event_loop()
    holder: dict = {}
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        try:
            holder["server"] = loop.run_until_complete(
                _serve(
                    host,
                    0,
                    upstream_uri,
                    model,
                    log,
                    switch_message,
                    available_models,
                    workspace,
                    token_provider,
                    switch_message_fn,
                )
            )
            holder["port"] = holder["server"].sockets[0].getsockname()[1]
        except Exception as exc:  # noqa: BLE001
            holder["error"] = exc
            log(f"[ERR] failed to start interposer: {exc!r}")
            ready.set()
            loop.close()
            return
        ready.set()
        loop.run_forever()
        server = holder.get("server")
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                loop.run_until_complete(server.wait_closed())
        loop.close()

    thread = threading.Thread(target=run, name="codex-interposer", daemon=True)
    thread.start()

    if not ready.wait(timeout=ready_timeout):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=ready_timeout)
        raise RuntimeError("Codex interposer did not become ready in time.")
    if error := holder.get("error"):
        raise RuntimeError("Codex interposer failed to start.") from error

    def stop() -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=ready_timeout)

    return holder["port"], stop
