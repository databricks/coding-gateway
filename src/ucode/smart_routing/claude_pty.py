"""PTY wrapper that routes Claude Code's first prompt with ``/model <name>``.

Claude Code has no runtime control protocol, so a per-launch UserPromptSubmit
hook captures and blocks the first prompt. While the TUI is idle, this wrapper
types a direct model command, restores the user's persisted default model, and
replays the prompt. Terminal traffic is otherwise forwarded unchanged.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pty
import re
import select
import signal
import socket
import struct
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MAX_MODEL_NAME_LEN = 200
CONFIRM_TIMEOUT_S = 3.0
SWITCH_TIMEOUT_S = 6.0
MODEL_PERSIST_TIMEOUT_S = 2.0
READY_QUIET_S = 0.75
SELECT_TIMEOUT_S = 0.2
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:/\-\[\]]+$")
_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_ansi(data: bytes) -> str:
    return _ANSI_RE.sub(b"", data).decode("utf-8", "replace")


def _squash(text: str) -> str:
    return "".join(text.split())


def _match_text(data: bytes) -> str:
    return _squash(strip_ansi(data))


def valid_model_name(name: object) -> bool:
    """Return whether *name* is safe to type as one slash-command argument."""
    return (
        isinstance(name, str)
        and 0 < len(name) <= MAX_MODEL_NAME_LEN
        and bool(_MODEL_NAME_RE.fullmatch(name))
    )


@dataclass(frozen=True)
class FirstPromptRoute:
    """The model switch and user-facing label for a first-prompt route."""

    model: str
    display_model: str
    rationale: str


class ConfirmationState:
    """Detect and accept Claude's optional cache-cost confirmation dialog."""

    PROMPT_MARKERS = ("Switch model?", "Yes, switch to", "No, go back")

    def __init__(self, window: int = 4096) -> None:
        self._buf = ""
        self._armed_until = 0.0
        self._window = window

    def arm(self, deadline: float) -> None:
        self._armed_until = deadline
        self._buf = ""

    def clear(self) -> None:
        self._armed_until = 0.0
        self._buf = ""

    def observe(self, chunk: bytes, now: float) -> bytes | None:
        if self._armed_until == 0.0:
            return None
        if now > self._armed_until:
            self.clear()
            return None
        self._buf = (self._buf + _match_text(chunk))[-self._window :]
        if all(_squash(marker) in self._buf for marker in self.PROMPT_MARKERS):
            self.clear()
            return b"\r"
        return None


class OutputMarkerDetector:
    """Latching ANSI-insensitive substring detector."""

    def __init__(self, markers: tuple[str, ...], window: int = 4096) -> None:
        self._markers = markers
        self._buf = ""
        self._window = window
        self.triggered = False

    def observe(self, chunk: bytes) -> bool:
        if self.triggered:
            return True
        self._buf = (self._buf + _match_text(chunk))[-self._window :]
        if any(_squash(marker) in self._buf for marker in self._markers):
            self.triggered = True
        return self.triggered


def inject_model_switch(master_fd: int, model: str) -> None:
    """Type Claude Code's direct, persistent model command."""
    if not valid_model_name(model):
        raise ValueError(f"Unsafe Claude model name: {model!r}")
    os.write(master_fd, f"/model {model}\r".encode())


def inject_prompt(master_fd: int, prompt: str, *, submit: bool = True) -> None:
    """Replay a captured prompt as one bracketed paste."""
    clean = prompt.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.replace("\x00", "").replace("\x1b", "")
    suffix = b"\r" if submit else b""
    os.write(master_fd, b"\x1b[200~" + clean.encode() + b"\x1b[201~" + suffix)


def inject_note(out_fd: int, message: str) -> None:
    os.write(out_fd, ("\r\n\x1b[36m" + message + "\x1b[0m\r\n").encode())


def request_first_prompt_route(path: Path, payload: dict, *, timeout: float = 5.0) -> dict | None:
    """Send a UserPromptSubmit payload to the owning PTY wrapper."""
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    request = {
        "method": "route_first_prompt",
        "prompt": prompt,
        "session_id": payload.get("session_id"),
    }
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        with client:
            client.connect(str(path))
            client.sendall((json.dumps(request) + "\n").encode())
            with client.makefile("rb") as stream:
                raw = stream.readline()
        response = json.loads(raw) if raw else None
    except (OSError, ValueError):
        return None
    return response if isinstance(response, dict) else None


def first_prompt_hook_output(response: dict | None) -> dict | None:
    """Translate the wrapper response into Claude hook output."""
    from ucode.smart_routing.v2 import format_routing_notice

    if not isinstance(response, dict) or response.get("action") != "block":
        return None
    model = response.get("model")
    if not valid_model_name(model):
        return None
    assert isinstance(model, str)
    display_model = response.get("display_model", model)
    if not isinstance(display_model, str) or not display_model:
        display_model = model
    rationale = response.get("rationale")
    return {
        "decision": "block",
        "reason": format_routing_notice(
            display_model, rationale if isinstance(rationale, str) else None
        ),
    }


def serve_first_prompt_socket(
    path: Path,
    route_prompt: Callable[[str], FirstPromptRoute],
    on_blocked_prompt: Callable[[str, str], None],
    stop: threading.Event,
    *,
    log: Callable[[str], None] = lambda _message: None,
) -> threading.Thread:
    """Serve the hook protocol, blocking exactly one non-command prompt."""

    def serve() -> None:
        claimed = False
        try:
            path.unlink(missing_ok=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(4)
            server.settimeout(0.5)
        except OSError as exc:
            log(f"[ERR] first-prompt socket bind failed: {exc!r}")
            return
        log(f"[READY] first-prompt socket {path}")
        try:
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                with conn, conn.makefile("rwb") as stream:
                    response: dict = {"action": "allow"}
                    blocked: tuple[str, str] | None = None
                    try:
                        request = json.loads(stream.readline())
                        prompt = request.get("prompt") if isinstance(request, dict) else None
                        is_route = (
                            isinstance(request, dict)
                            and request.get("method") == "route_first_prompt"
                        )
                        is_command = isinstance(prompt, str) and prompt.lstrip().startswith("/")
                        if (
                            is_route
                            and not claimed
                            and isinstance(prompt, str)
                            and prompt.strip()
                            and not is_command
                        ):
                            routed = route_prompt(prompt)
                            model = routed.model
                            display_model = routed.display_model
                            rationale = routed.rationale
                            if valid_model_name(model):
                                claimed = True
                                response = {
                                    "action": "block",
                                    "model": model,
                                    "rationale": rationale,
                                }
                                response["display_model"] = display_model
                                blocked = (prompt, model)
                    except Exception as exc:  # noqa: BLE001 - hooks must fail open
                        log(f"[ERR] first-prompt request: {exc!r}")
                    stream.write((json.dumps(response) + "\n").encode())
                    stream.flush()
                    if blocked is not None:
                        on_blocked_prompt(*blocked)
        finally:
            server.close()

    thread = threading.Thread(target=serve, name="claude-first-prompt", daemon=True)
    thread.start()
    return thread


class TerminalModeGuard:
    """Put stdin in raw mode for the PTY session and restore it on exit."""

    def __init__(self, fd: int = 0) -> None:
        self.fd = fd
        self._saved: list | None = None

    def __enter__(self) -> TerminalModeGuard:
        if os.isatty(self.fd):
            self._saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None


def sync_winsize(master_fd: int, stdin_fd: int = 0) -> None:
    if not os.isatty(stdin_fd):
        return
    try:
        packed = fcntl.ioctl(stdin_fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


def run_claude_pty(
    argv: list[str],
    *,
    route_prompt: Callable[[str], FirstPromptRoute],
    socket_path: Path,
    prepare_model_switch: Callable[[str], None] = lambda _model: None,
    model_switch_persisted: Callable[[], bool] = lambda: True,
    restore_model_setting: Callable[[], None] = lambda: None,
    log_path: Path | None = None,
) -> int:
    """Run Claude in a PTY, switch its model, and replay the first prompt."""

    def log(message: str) -> None:
        if log_path is None:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        except OSError:
            pass

    debug = os.environ.get("UCODE_CLAUDE_PTY_DEBUG") == "1"
    confirm = ConfirmationState()
    stop = threading.Event()
    pending_lock = threading.Lock()
    pending: dict[str, tuple[str, str] | None] = {"value": None}

    def on_blocked_prompt(prompt: str, model: str) -> None:
        with pending_lock:
            pending["value"] = (prompt, model)
        log(f"[ROUTE] first prompt -> {model!r}")

    server_thread = serve_first_prompt_socket(
        socket_path, route_prompt, on_blocked_prompt, stop, log=log
    )
    socket_deadline = time.monotonic() + 2.0
    while (
        not socket_path.exists() and server_thread.is_alive() and time.monotonic() < socket_deadline
    ):
        time.sleep(0.01)
    if not socket_path.exists():
        log("[ERR] first-prompt socket was not ready before Claude launch")
        stop.set()
        raise RuntimeError(
            "Smart routing could not start its local prompt-routing socket; Claude was not launched."
        )

    pid, master_fd = pty.fork()
    if pid == 0:
        os.execvp(argv[0], argv)
        os._exit(127)

    previous_winch = signal.getsignal(signal.SIGWINCH)

    def on_winch(_signum: int, _frame: object) -> None:
        sync_winsize(master_fd)

    try:
        with TerminalModeGuard(0):
            signal.signal(signal.SIGWINCH, on_winch)
            sync_winsize(master_fd)
            stdin_open = True
            last_output = 0.0
            phase = "waiting_prompt"
            routed_prompt = ""
            routed_model = ""
            switch_started = 0.0
            switch_complete: OutputMarkerDetector | None = None
            while True:
                readable = [master_fd, 0] if stdin_open else [master_fd]
                try:
                    ready_fds, _, _ = select.select(readable, [], [], SELECT_TIMEOUT_S)
                except InterruptedError:
                    continue

                if 0 in ready_fds:
                    try:
                        data = os.read(0, 4096)
                    except OSError:
                        data = b""
                    if not data:
                        stdin_open = False
                    else:
                        os.write(master_fd, data)

                if master_fd in ready_fds:
                    try:
                        chunk = os.read(master_fd, 8192)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        break
                    os.write(1, chunk)
                    last_output = time.monotonic()
                    if debug:
                        log(f"[OUT] {strip_ansi(chunk)[:400]!r}")
                    keystroke = confirm.observe(chunk, last_output)
                    if keystroke is not None:
                        os.write(master_fd, keystroke)
                    if phase == "switching" and switch_complete is not None:
                        switch_complete.observe(chunk)

                if phase == "waiting_prompt":
                    with pending_lock:
                        captured = pending["value"]
                    if captured is not None:
                        routed_prompt, routed_model = captured
                        phase = "waiting_to_switch"

                now = time.monotonic()
                idle = last_output > 0.0 and now - last_output >= READY_QUIET_S
                if phase == "waiting_to_switch" and idle:
                    prepare_model_switch(routed_model)
                    inject_model_switch(master_fd, routed_model)
                    confirm.arm(now + CONFIRM_TIMEOUT_S)
                    switch_complete = OutputMarkerDetector(("Set model to", "Model set to"))
                    switch_started = now
                    phase = "switching"
                    log(f"[SWITCH] /model {routed_model}")
                elif (
                    phase == "switching"
                    and switch_complete is not None
                    and switch_complete.triggered
                ):
                    phase = "waiting_persist"
                    switch_started = now
                elif phase == "waiting_persist" and model_switch_persisted():
                    restore_model_setting()
                    inject_prompt(master_fd, routed_prompt)
                    phase = "done"
                    log("[REPLAY] first prompt submitted")
                elif phase == "waiting_persist" and now - switch_started >= MODEL_PERSIST_TIMEOUT_S:
                    inject_note(
                        1,
                        "Smart Routing could not safely preserve your default model. "
                        "Claude is exiting without submitting your prompt.",
                    )
                    os.kill(pid, signal.SIGTERM)
                    phase = "failed"
                    log("[ERR] routed model was not persisted before timeout")
                elif phase == "switching" and now - switch_started >= SWITCH_TIMEOUT_S:
                    os.write(master_fd, b"\x1b")
                    restore_model_setting()
                    inject_note(
                        1,
                        "Smart Routing could not confirm the model switch. "
                        "Your prompt was restored but not submitted.",
                    )
                    inject_prompt(master_fd, routed_prompt, submit=False)
                    phase = "failed"
                    log(f"[ERR] direct model switch timed out for {routed_model!r}")
    finally:
        signal.signal(signal.SIGWINCH, previous_winch)
        stop.set()
        with contextlib.suppress(OSError):
            os.close(master_fd)
        socket_path.unlink(missing_ok=True)

    _waited_pid, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1
