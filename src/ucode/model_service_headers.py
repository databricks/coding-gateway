"""Headers used to route model service requests."""

from __future__ import annotations

from ucode.constants import MODEL_PROVIDER_SERVICE_HEADER, MODEL_SERVICE_PARENT_SCHEMA_HEADER


def model_service_routing_headers(
    provider: str | None,
    parent_schema: str | None,
) -> dict[str, str]:
    # A provider selects one MPS; a parent selects Model Services, so the modes are exclusive.
    if provider:
        return {MODEL_PROVIDER_SERVICE_HEADER: provider}
    if parent_schema:
        return {MODEL_SERVICE_PARENT_SCHEMA_HEADER: parent_schema}
    return {}
