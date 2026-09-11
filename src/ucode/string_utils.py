"""Shared string validation helpers."""

from __future__ import annotations


def is_valid_catalog_schema(value: str) -> bool:
    """Return whether value is a safe ``<catalog>.<schema>`` reference."""
    parts = value.split(".")
    return len(parts) == 2 and all(
        part
        and part.isprintable()
        and not any(character.isspace() or character == "/" for character in part)
        for part in parts
    )
