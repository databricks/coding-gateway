"""Shared helpers for passing Codex configuration on the command line."""

from __future__ import annotations

from collections.abc import Mapping

import tomlkit
from tomlkit.items import Item


def _toml_item(value: object) -> Item:
    if isinstance(value, Mapping):
        inline = tomlkit.inline_table()
        for key, child in value.items():
            inline[str(key)] = _toml_item(child)
        return inline
    if isinstance(value, list):
        array = tomlkit.array()
        for child in value:
            array.append(_toml_item(child))
        return array
    if isinstance(value, Item):
        return value
    return tomlkit.item(value)


def _toml_value(value: object) -> str:
    return _toml_item(value).as_string()


def codex_config_args(config: dict) -> list[str]:
    """Render a Codex config layer as repeatable ``--config`` overrides."""
    args: list[str] = []
    for key, value in config.items():
        # These maps contain named entries. Override each entry individually so
        # the rest of the user's base map remains intact.
        if key in {"hooks", "model_providers"} and isinstance(value, dict):
            for entry_name, entry_config in value.items():
                args.extend(
                    [
                        "--config",
                        f"{key}.{entry_name}={_toml_value(entry_config)}",
                    ]
                )
        else:
            args.extend(["--config", f"{key}={_toml_value(value)}"])
    return args
