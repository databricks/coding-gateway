"""Tests for model service routing headers."""

from ucode.constants import MODEL_PROVIDER_SERVICE_HEADER, MODEL_SERVICE_PARENT_SCHEMA_HEADER
from ucode.model_service_headers import model_service_routing_headers


def test_provider_header():
    assert model_service_routing_headers("main.default.provider", None) == {
        MODEL_PROVIDER_SERVICE_HEADER: "main.default.provider"
    }


def test_parent_schema_header():
    assert model_service_routing_headers(None, "main.default") == {
        MODEL_SERVICE_PARENT_SCHEMA_HEADER: "main.default"
    }


def test_provider_takes_precedence():
    assert model_service_routing_headers("main.default.provider", "main.default") == {
        MODEL_PROVIDER_SERVICE_HEADER: "main.default.provider"
    }
