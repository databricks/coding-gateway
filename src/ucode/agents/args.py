"""Parsing helpers shared by coding-agent launchers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchOptions:
    """Invocation-scoped options shared by agent launchers."""

    launch_smart_routing: bool = False
    # Claude's --model is consumed by ucode, so it must be passed separately for this launch.
    # Codex keeps --model in the forwarded tool arguments instead.
    claude_launch_model: str | None = None


def explicit_model_arg_value(tool_args: list[str]) -> str | None:
    """Return the last model selected before the harness's ``--`` separator."""
    model: str | None = None
    index = 0
    while index < len(tool_args):
        arg = tool_args[index]
        if arg == "--":
            break
        if arg in {"--model", "-m"}:
            if index + 1 < len(tool_args) and not tool_args[index + 1].startswith("-"):
                model = tool_args[index + 1]
                index += 1
        elif arg.startswith("--model="):
            value = arg.partition("=")[2]
            if value:
                model = value
        index += 1
    return model


def has_explicit_model_arg(tool_args: list[str]) -> bool:
    """Return whether the harness receives a ``--model`` option before ``--``."""
    for arg in tool_args:
        if arg == "--":
            return False
        if arg in {"--model", "-m"} or arg.startswith("--model="):
            return True
    return False
