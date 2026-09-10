"""Tests for databricks.py — pure helpers and URL builders that don't hit the network."""

from __future__ import annotations

import json
import os
import subprocess
from decimal import Decimal
from urllib.parse import parse_qs

import pytest

import ucode.databricks as db_mod
from ucode.databricks import (
    CODING_AGENT_RECOMMEND_MODEL_PATH,
    _format_subprocess_result,
    _parse_databricks_cli_version,
    _run_databricks_cli_installer,
    _scrub_databrickscfg,
    _scrub_json,
    all_users_can_use_schema,
    build_auth_shell_command,
    build_auth_token_argv,
    build_databricks_cli_env,
    build_opencode_base_urls,
    build_shared_base_urls,
    build_skills_mcp_url,
    build_tool_base_url,
    classify_model_family,
    databricks_cli_version,
    discover_sql_warehouses,
    ensure_databricks_cli_version,
    ensure_pat_bearer,
    get_databricks_profiles,
    get_databricks_token,
    install_ai_tools,
    list_databricks_apps,
    list_databricks_connections,
    list_genie_spaces,
    list_workspace_budgets,
    resolve_current_budget_spend,
    upgrade_databricks_cli,
    workspace_hostname,
)

WS = "https://example.databricks.com"
WS_HOST = "example.databricks.com"


class _FakeResponse:
    """Minimal urlopen context manager returning a JSON body."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class TestWorkspaceHostname:
    def test_extracts_hostname(self):
        assert workspace_hostname(WS) == "example.databricks.com"

    def test_handles_path(self):
        assert (
            workspace_hostname("https://foo.azuredatabricks.net/some/path")
            == "foo.azuredatabricks.net"
        )

    def test_invalid_url_raises(self):
        with pytest.raises((RuntimeError, ValueError)):
            workspace_hostname("")


class TestBuildDatabricksCliEnv:
    def test_sets_databricks_host(self):
        env = build_databricks_cli_env(WS)
        assert env["DATABRICKS_HOST"] == WS

    def test_strips_ambient_profile_without_explicit_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "other-workspace")

        env = build_databricks_cli_env(WS)

        assert env["DATABRICKS_HOST"] == WS
        assert "DATABRICKS_CONFIG_PROFILE" not in env

    def test_preserves_ambient_profile_with_explicit_profile(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "other-workspace")

        env = build_databricks_cli_env(WS, profile="stablebox")

        assert env["DATABRICKS_HOST"] == WS
        assert env["DATABRICKS_CONFIG_PROFILE"] == "other-workspace"


class TestBuildToolBaseUrl:
    def test_codex(self):
        url = build_tool_base_url("codex", WS)
        assert url == f"{WS}/ai-gateway/codex/v1"

    def test_claude(self):
        url = build_tool_base_url("claude", WS)
        assert url == f"{WS}/ai-gateway/anthropic"

    def test_gemini(self):
        url = build_tool_base_url("gemini", WS)
        assert url == f"{WS}/ai-gateway/gemini"

    def test_opencode_raises(self):
        with pytest.raises(RuntimeError, match="multiple base URLs"):
            build_tool_base_url("opencode", WS)

    def test_unsupported_tool_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported"):
            build_tool_base_url("unknown", WS)


class TestBuildOpencodeBaseUrls:
    def test_returns_anthropic_gemini_and_oss(self):
        urls = build_opencode_base_urls(WS)
        assert urls["anthropic"] == f"{WS}/ai-gateway/anthropic/v1"
        assert urls["gemini"] == f"{WS}/ai-gateway/gemini/v1beta"
        assert urls["oss"] == f"{WS}/ai-gateway/mlflow/v1"


class TestBuildSharedBaseUrls:
    def test_contains_all_tools(self):
        urls = build_shared_base_urls(WS)
        assert "codex" in urls
        assert "claude" in urls
        assert "gemini" in urls
        assert "opencode" in urls

    def test_opencode_is_dict(self):
        urls = build_shared_base_urls(WS)
        assert isinstance(urls["opencode"], dict)

    def test_codex_url_format(self):
        urls = build_shared_base_urls(WS)
        assert urls["codex"] == f"{WS}/ai-gateway/codex/v1"


class TestBuildSkillsMcpUrl:
    def test_empty_locations_returns_bare_route(self):
        assert build_skills_mcp_url(WS, []) == f"{WS}/ai-gateway/skills/"

    def test_single_location_appends_schema_query(self):
        assert build_skills_mcp_url(WS, ["main.default"]) == (
            f"{WS}/ai-gateway/skills/?schema=main.default"
        )

    def test_multiple_locations_preserve_order(self):
        assert build_skills_mcp_url(WS, ["a.b", "c.d"]) == (
            f"{WS}/ai-gateway/skills/?schema=a.b&schema=c.d"
        )


class TestDiscoverClaudeModels:
    def test_lists_all_anthropic_model_ids_without_legacy_validation(self, monkeypatch):
        captured = {}
        payload = {
            "data": [
                {"id": "system.ai.claude-opus-5"},
                {"id": "databricks-claude-sonnet-5"},
                {"id": "opaque-model-id"},
                {"id": "opaque-model-id"},
                {"id": ""},
            ]
        }

        def fake_get(url, token, **kwargs):
            captured["request"] = (url, token, kwargs)
            return payload, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        models, reason = db_mod.list_anthropic_models(WS, "token")

        assert reason is None
        assert models == [
            "system.ai.claude-opus-5",
            "databricks-claude-sonnet-5",
            "opaque-model-id",
        ]
        assert captured["request"] == (
            f"{WS}/ai-gateway/anthropic/v1/models",
            "token",
            {"max_retries": 2},
        )

    def test_lists_anthropic_display_names_with_model_ids(self, monkeypatch):
        payload = {
            "data": [
                {"id": "system.ai.glm-5-3-flash", "display_name": "GLM 5.3 Flash"},
                {"id": "opaque-model-id"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda *_args, **_kwargs: (payload, None))

        catalog = db_mod.list_anthropic_model_catalog(WS, "token")

        assert catalog.model_ids == ["system.ai.glm-5-3-flash", "opaque-model-id"]
        assert catalog.model_id_to_display_name == {"system.ai.glm-5-3-flash": "GLM 5.3 Flash"}
        assert catalog.error_msg is None

    def test_selects_opus_4_8_when_advertised(self, monkeypatch):
        payload = {
            "data": [
                {"id": "databricks-claude-opus-4-7"},
                {"id": "databricks-claude-opus-4-8"},
                {"id": "databricks-claude-sonnet-4-6"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token, **kwargs: (payload, None))

        models, reason = db_mod.discover_claude_models(WS, "token")

        assert reason is None
        assert models["opus"] == "databricks-claude-opus-4-8"

    def test_buckets_system_ai_claude_models(self, monkeypatch):
        payload = {
            "data": [
                {"id": "system.ai.claude-opus-4-8"},
                {"id": "system.ai.claude-sonnet-4-6"},
                {"id": "system.ai.glm-5-3-flash"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda *_args, **_kwargs: (payload, None))

        models, reason = db_mod.discover_claude_models(WS, "token")

        assert reason is None
        assert models == {
            "opus": "system.ai.claude-opus-4-8",
            "sonnet": "system.ai.claude-sonnet-4-6",
        }

    def test_buckets_fable_family(self, monkeypatch):
        payload = {
            "data": [
                {"id": "databricks-claude-fable-5"},
                {"id": "databricks-claude-opus-4-8"},
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token, **kwargs: (payload, None))

        models, reason = db_mod.discover_claude_models(WS, "token")

        assert reason is None
        assert models["fable"] == "databricks-claude-fable-5"


def _model_service(model_id: str) -> dict:
    """A model-services entry whose `name` strips to `model_id`."""
    return {"name": f"model-services/{model_id}"}


class TestModelTokenLimits:
    def test_glm_is_capped(self):
        assert db_mod.model_token_limits("system.ai.glm-5-2") == {
            "context": 200_000,
            "output": 25_000,
        }

    def test_glm_matches_any_version(self):
        assert db_mod.model_token_limits("system.ai.glm-4-6-flash") == {
            "context": 200_000,
            "output": 25_000,
        }

    def test_uncapped_model_returns_none(self):
        assert db_mod.model_token_limits("system.ai.kimi-k2-7-code") is None


class TestDiscoverModelServices:
    def test_buckets_families_by_name(self, monkeypatch):
        payload = {
            "model_services": [
                _model_service("system.ai.claude-fable-5"),
                _model_service("system.ai.claude-opus-4-7"),
                _model_service("system.ai.claude-opus-4-8"),
                _model_service("system.ai.claude-sonnet-4-6"),
                _model_service("system.ai.gpt-5"),
                _model_service("system.ai.gemini-2-5-flash"),
                _model_service("system.ai.gemini-3-5-flash"),
                _model_service("system.ai.kimi-k2-7-code"),
                _model_service("system.ai.glm-5-2"),
                _model_service("system.ai.deepseek-v4-pro"),
                _model_service("system.ai.llama-4-maverick"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, oss, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        # Fable bucketed; newest opus wins; sonnet bucketed; haiku absent.
        assert claude == {
            "fable": "system.ai.claude-fable-5",
            "opus": "system.ai.claude-opus-4-8",
            "sonnet": "system.ai.claude-sonnet-4-6",
        }
        assert codex == ["system.ai.gpt-5"]
        # Gemini ordered newest-first via the shared sort key.
        assert gemini[0] == "system.ai.gemini-3-5-flash"
        # DeepSeek, GLM, and Kimi are allowlisted OSS families; Llama is not.
        assert oss == [
            "system.ai.deepseek-v4-pro",
            "system.ai.glm-5-2",
            "system.ai.kimi-k2-7-code",
        ]

    def test_oss_allowlist_drops_unsupported_families(self, monkeypatch):
        # Only explicitly supported chat families are retained.
        payload = {
            "model_services": [
                _model_service("system.ai.glm-5-2"),
                _model_service("system.ai.kimi-k2-7-code"),
                _model_service("system.ai.qwen-3-coder"),
                _model_service("system.ai.deepseek-v4-pro"),
                _model_service("system.ai.gte-large-embed"),
                _model_service("system.ai.bge-reranker-v2"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, oss, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        assert (claude, codex, gemini) == ({}, [], [])
        assert oss == [
            "system.ai.deepseek-v4-pro",
            "system.ai.glm-5-2",
            "system.ai.kimi-k2-7-code",
        ]

    def test_paginates_via_next_page_token(self, monkeypatch):
        pages = {
            None: {
                "model_services": [_model_service("system.ai.gpt-5")],
                "next_page_token": "tok2",
            },
            "tok2": {
                "model_services": [_model_service("system.ai.claude-opus-4-8")],
            },
        }

        def fake_get(url, token, timeout=10):
            token_param = None
            if "page_token=" in url:
                token_param = url.split("page_token=")[1].split("&")[0]
            return pages[token_param], None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        claude, codex, _, _, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        assert codex == ["system.ai.gpt-5"]
        assert claude == {"opus": "system.ai.claude-opus-4-8"}

    def test_http_failure_returns_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (None, "HTTP 500 Server Error")
        )

        claude, codex, gemini, oss, reason = db_mod.discover_model_services(WS, "token")

        assert (claude, codex, gemini, oss) == ({}, [], [], [])
        assert reason == "HTTP 500 Server Error"

    def test_no_matching_families_reports_sample(self, monkeypatch):
        payload = {"model_services": [_model_service("system.ai.llama-4-maverick")]}
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, oss, reason = db_mod.discover_model_services(WS, "token")

        assert (claude, codex, gemini, oss) == ({}, [], [], [])
        assert reason is not None and "llama-4-maverick" in reason

    def test_ignores_non_system_ai_schemas(self, monkeypatch):
        # The metastore listing returns services from every schema; only
        # system.ai.* foundation models should be picked up.
        payload = {
            "model_services": [
                _model_service("system.ai.gpt-5"),
                _model_service("main.schema3.gpt-5-5"),
                _model_service("temp.erni.kimi-k2-7-code"),
                _model_service("temp.erni.claude-opus-4-8"),
                _model_service("dnasi_agent_cuj.default.dnasi-gpt55-test"),
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=10: (payload, None)
        )

        claude, codex, gemini, oss, reason = db_mod.discover_model_services(WS, "token")

        assert reason is None
        assert codex == ["system.ai.gpt-5"]
        assert claude == {}  # temp.erni.claude-* must not be bucketed
        assert gemini == []
        assert oss == []

    def test_requests_bounded_page_size(self, monkeypatch):
        # The endpoint 499s without a bounded page_size, so every request must
        # carry one.
        urls: list[str] = []

        def fake_get(url, token, timeout=10):
            urls.append(url)
            return {"model_services": [_model_service("system.ai.gpt-5")]}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        ids, reason = db_mod.list_model_services(WS, "token")

        assert ids == ["system.ai.gpt-5"]
        assert reason is None
        assert all("page_size=" in u for u in urls)
        # Scope to the `system.ai` schema so the endpoint returns just the
        # foundation models rather than walking the whole metastore.
        assert all("parent=schemas%2Fsystem.ai" in u for u in urls)

    def test_retries_page_before_giving_up(self, monkeypatch):
        payload = {"model_services": [_model_service("system.ai.gpt-5")]}
        calls = {"n": 0}

        def flaky_get(url, token, timeout=10):
            calls["n"] += 1
            if calls["n"] < 3:
                return None, "HTTP 499 Unknown"
            return payload, None

        monkeypatch.setattr(db_mod, "_http_get_json", flaky_get)

        ids, reason = db_mod.list_model_services(WS, "token")

        assert reason is None
        assert ids == ["system.ai.gpt-5"]
        assert calls["n"] == 3  # two failures, third succeeds


class TestModelServiceExists:
    def test_true_when_listed_in_its_schema(self, monkeypatch):
        urls: list[str] = []

        def fake_get(url, token, timeout=30):
            urls.append(url)
            return {"model_services": [_model_service("main.aarushi.claude-opus-4-5")]}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        exists, reason = db_mod.model_service_exists(WS, "token", "main.aarushi.claude-opus-4-5")
        assert (exists, reason) == (True, None)
        # Scoped to the typed name's own schema, not system.ai.
        assert all("parent=schemas%2Fmain.aarushi" in u for u in urls)

    def test_false_when_schema_has_no_such_service(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (
                {"model_services": [_model_service("main.aarushi.some-other-model")]},
                None,
            ),
        )
        exists, reason = db_mod.model_service_exists(WS, "token", "main.aarushi.claude-opus-4-5")
        assert (exists, reason) == (False, None)

    def test_bad_name_is_inconclusive(self, monkeypatch):
        def fail(*a, **k):
            raise AssertionError("should not hit the API for a malformed name")

        monkeypatch.setattr(db_mod, "_http_get_json", fail)
        for bad in ("just-a-name", "main.aarushi", "main..model", "main.aarushi.model.extra"):
            exists, reason = db_mod.model_service_exists(WS, "token", bad)
            assert exists is None and reason

    def test_http_error_is_inconclusive_not_absent(self, monkeypatch):
        # A transient failure must read as "couldn't verify", never "doesn't exist" — the caller
        # would otherwise reject a valid model on a blip.
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (None, "HTTP 500 Server Error"),
        )
        exists, reason = db_mod.model_service_exists(WS, "token", "main.aarushi.claude-opus-4-5")
        assert exists is None
        assert "500" in reason

    def test_not_found_means_absent(self, monkeypatch):
        # A 404 is the catalog/schema not existing, so the model can't either — a definitive "no"
        # the caller re-prompts on, not an inconclusive "couldn't verify".
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (
                None,
                'HTTP 404 Not Found: {"error_code":"NOT_FOUND","message":"Resource not found"}',
            ),
        )
        exists, _ = db_mod.model_service_exists(WS, "token", "maikjn.default.aar")
        assert exists is False

    def test_paginates_until_found(self, monkeypatch):
        pages = {
            None: {
                "model_services": [_model_service("main.aarushi.other")],
                "next_page_token": "n",
            },
            "n": {"model_services": [_model_service("main.aarushi.claude-opus-4-5")]},
        }

        def fake_get(url, token, timeout=30):
            tok = url.split("page_token=")[1].split("&")[0] if "page_token=" in url else None
            return pages[tok], None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        exists, _ = db_mod.model_service_exists(WS, "token", "main.aarushi.claude-opus-4-5")
        assert exists is True


class TestListModelProviderServices:
    _PAYLOAD = {
        "model_provider_services": [
            {
                "name": "model-provider-services/main.schema1.anthropic-svc",
                "config": {"provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_ANTHROPIC"},
            },
            {
                "name": "model-provider-services/main.schema1.claude-max-svc",
                "config": {
                    "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_ANTHROPIC",
                    "anthropic": {"relayed": {}},
                },
            },
            {
                "name": "model-provider-services/main.schema1.openai-svc",
                "config": {"provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_OPENAI"},
            },
            {
                "name": "model-provider-services/main.schema2.bedrock-svc",
                "config": {
                    "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_AMAZON_BEDROCK",
                    "allow_all_targets": False,
                    "targets": [
                        {
                            "model": "us.anthropic.claude-sonnet-4-6",
                            "native_api_types": ["anthropic/v1/messages"],
                        },
                        {"model": "global.anthropic.claude-opus-4-8"},
                    ],
                },
            },
            {
                "name": "model-provider-services/main.schema2.bedrock-titan-svc",
                "config": {
                    "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_AMAZON_BEDROCK",
                    "targets": [{"model": "amazon.titan-text-express-v1"}],
                },
            },
        ]
    }

    def test_strips_prefix_and_tags_provider_type(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        services, reason = db_mod.list_model_provider_services(WS, "token")
        assert reason is None
        assert services[0] == {
            "name": "main.schema1.anthropic-svc",
            "provider_type": "anthropic",
            "targets": [],
            "allow_all_targets": False,
            "relayed": False,
        }
        assert {s["provider_type"] for s in services} == {
            "anthropic",
            "openai",
            "amazon_bedrock",
        }

    def test_flags_relayed_anthropic(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        services, _ = db_mod.list_model_provider_services(WS, "token")
        by_name = {s["name"]: s for s in services}
        assert by_name["main.schema1.claude-max-svc"]["relayed"] is True
        assert by_name["main.schema1.anthropic-svc"]["relayed"] is False

    def test_extracts_targets(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        services, _ = db_mod.list_model_provider_services(WS, "token")
        bedrock = next(s for s in services if s["name"] == "main.schema2.bedrock-svc")
        assert bedrock["targets"] == [
            "us.anthropic.claude-sonnet-4-6",
            "global.anthropic.claude-opus-4-8",
        ]

    def test_returns_reason_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (None, "HTTP 500 Server Error")
        )
        services, reason = db_mod.list_model_provider_services(WS, "token")
        assert services == []
        assert reason == "HTTP 500 Server Error"

    def test_claude_includes_anthropic_and_usable_bedrock(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        names, reason = db_mod.list_tool_provider_services("claude", WS, "token")
        assert reason is None
        # Anthropic (stored-key + relayed) + the Bedrock service with Claude
        # targets; the Bedrock service exposing only Titan is hidden (no Claude
        # models to pin).
        assert names == [
            "main.schema1.anthropic-svc",
            "main.schema1.claude-max-svc",
            "main.schema2.bedrock-svc",
        ]

    def test_codex_filters_to_openai(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )
        names, _ = db_mod.list_tool_provider_services("codex", WS, "token")
        assert names == ["main.schema1.openai-svc"]


class TestMapClaudeFamilyModels:
    def test_maps_families(self):
        models = db_mod.map_claude_family_models(
            [
                "us.anthropic.claude-sonnet-4-6",
                "global.anthropic.claude-opus-4-8",
                "anthropic.claude-haiku-4-5",
                "amazon.titan-text-express-v1",
            ]
        )
        assert models == {
            "sonnet": "us.anthropic.claude-sonnet-4-6",
            "opus": "global.anthropic.claude-opus-4-8",
            "haiku": "anthropic.claude-haiku-4-5",
        }

    def test_prefers_highest_version(self):
        models = db_mod.map_claude_family_models(
            ["us.anthropic.claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-6"]
        )
        assert models["sonnet"] == "us.anthropic.claude-sonnet-4-6"

    def test_region_tie_break_prefers_global(self):
        models = db_mod.map_claude_family_models(
            [
                "us.anthropic.claude-opus-4-8",
                "global.anthropic.claude-opus-4-8",
                "eu.anthropic.claude-opus-4-8",
            ]
        )
        assert models["opus"] == "global.anthropic.claude-opus-4-8"

    def test_maps_canonical_anthropic_ids(self):
        # An Anthropic service publishes canonical ids (no region prefix); the same mapper groups
        # them by family and picks the highest version, so per-family pinning works there too.
        models = db_mod.map_claude_family_models(
            ["claude-sonnet-5", "claude-haiku-4-5", "claude-opus-4-8"]
        )
        assert models == {
            "sonnet": "claude-sonnet-5",
            "haiku": "claude-haiku-4-5",
            "opus": "claude-opus-4-8",
        }

    def test_empty_when_no_claude(self):
        assert db_mod.map_claude_family_models(["amazon.titan-text-express-v1"]) == {}


class TestResolveProviderLaunchModel:
    def test_none_when_service_offers_opus(self):
        # Claude Code's own opus default already works, so we pin nothing (and avoid the duplicate
        # /model picker row that setting ANTHROPIC_MODEL causes).
        models = {
            "opus": "claude-opus-4-8",
            "sonnet": "claude-sonnet-5",
            "haiku": "claude-haiku-4-5",
        }
        assert db_mod.resolve_provider_launch_model(None, models) is None

    def test_falls_back_to_best_tier_when_no_opus(self):
        # No opus target: launch on the most capable tier the service does offer (sonnet > haiku).
        models = {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"}
        assert db_mod.resolve_provider_launch_model(None, models) == "claude-sonnet-5"

    def test_falls_back_to_haiku_when_only_haiku(self):
        assert db_mod.resolve_provider_launch_model(None, {"haiku": "claude-haiku-4-5"}) == (
            "claude-haiku-4-5"
        )

    def test_family_alias_resolves_to_declared_target(self):
        models = {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"}
        assert db_mod.resolve_provider_launch_model("haiku", models) == "claude-haiku-4-5"

    def test_family_alias_not_offered_raises_with_available_list(self):
        models = {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"}
        with pytest.raises(RuntimeError, match="does not offer a 'opus' model.*haiku, sonnet"):
            db_mod.resolve_provider_launch_model("opus", models)

    def test_raw_target_id_is_trusted(self):
        # A non-family value is a raw target the user knows the service allows; pass it through.
        models = {"sonnet": "claude-sonnet-5"}
        assert db_mod.resolve_provider_launch_model("claude-3-7-sonnet", models) == (
            "claude-3-7-sonnet"
        )

    def test_no_models_and_no_override_is_none(self):
        assert db_mod.resolve_provider_launch_model(None, {}) is None


class TestProviderServicePagination:
    """The listing is paginated; ignoring next_page_token hid services on later pages entirely."""

    @staticmethod
    def _page(names, next_token=None):
        payload = {
            "model_provider_services": [
                {
                    "name": f"model-provider-services/{n}",
                    "config": {"provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_OPENAI"},
                }
                for n in names
            ]
        }
        if next_token:
            payload["next_page_token"] = next_token
        return payload

    def test_follows_next_page_token(self, monkeypatch):
        pages = [
            self._page(["main.s.one"], next_token="tok2"),
            self._page(["main.s.two"]),
        ]
        seen: list[str] = []

        def fake_get(url, token, **kwargs):
            seen.append(url)
            return pages[len(seen) - 1], None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        services, reason = db_mod.list_model_provider_services("https://ws", "tok")

        assert reason is None
        assert [s["name"] for s in services] == ["main.s.one", "main.s.two"]
        assert "page_token=tok2" in seen[1]

    def test_stops_on_a_repeated_token(self, monkeypatch):
        # A server that echoes the same token would otherwise spin forever.
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, **kw: (self._page(["main.s.one"], next_token="same"), None),
        )
        services, reason = db_mod.list_model_provider_services("https://ws", "tok")
        assert reason is None
        assert len(services) >= 1

    def test_keeps_earlier_pages_when_a_later_one_fails(self, monkeypatch):
        calls = {"n": 0}

        def fake_get(url, token, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return self._page(["main.s.one"], next_token="tok2"), None
            return None, "HTTP 500 Server Error"

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        services, reason = db_mod.list_model_provider_services("https://ws", "tok")

        # A mid-pagination blip should degrade to partial results, not to an error.
        assert reason is None
        assert [s["name"] for s in services] == ["main.s.one"]

    def test_reports_the_failure_when_nothing_was_collected(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, **kw: (None, "HTTP 403 Forbidden")
        )
        services, reason = db_mod.list_model_provider_services("https://ws", "tok")
        assert services == []
        assert reason == "HTTP 403 Forbidden"

    def test_parent_scopes_the_listing(self, monkeypatch):
        seen: dict = {}

        def fake_get(url, token, **kwargs):
            seen["url"] = url
            return self._page(["main.tien_le.openai"]), None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        db_mod.list_model_provider_services("https://ws", "tok", parent="main.tien_le")

        assert "parent=schemas%2Fmain.tien_le" in seen["url"]

    def test_page_size_is_always_sent(self, monkeypatch):
        seen: dict = {}

        def fake_get(url, token, **kwargs):
            seen["url"] = url
            return self._page(["main.s.one"]), None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        db_mod.list_model_provider_services("https://ws", "tok")

        assert "page_size=" in seen["url"]


class TestGetModelProviderService:
    def test_addresses_the_service_directly(self, monkeypatch):
        seen: dict = {}

        def fake_get(url, token, **kwargs):
            seen["url"] = url
            return {
                "name": "model-provider-services/main.tien_le.openai_all",
                "config": {
                    "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_OPENAI",
                    "allow_all_targets": True,
                },
            }, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)
        service, reason = db_mod.get_model_provider_service(
            "main.tien_le.openai_all", "https://ws", "tok"
        )

        assert reason is None
        assert service is not None
        assert service["name"] == "main.tien_le.openai_all"
        assert service["allow_all_targets"] is True
        assert seen["url"].endswith("/model-provider-services/main.tien_le.openai_all")

    def test_missing_service_returns_the_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, **kw: (None, "HTTP 404 Not Found")
        )
        service, reason = db_mod.get_model_provider_service("main.a.b", "https://ws", "tok")
        assert service is None
        assert "404" in (reason or "")


class TestResolveProviderService:
    _PAYLOAD = TestListModelProviderServices._PAYLOAD

    def _patch(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (self._PAYLOAD, None)
        )

    def test_anthropic_ok(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.schema1.anthropic-svc", WS, "token"
        )
        assert error is None
        assert service["provider_type"] == "anthropic"

    def test_bedrock_with_claude_ok(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.schema2.bedrock-svc", WS, "token"
        )
        assert error is None
        assert service["provider_type"] == "amazon_bedrock"

    def test_wrong_type_rejected(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.schema1.openai-svc", WS, "token"
        )
        assert service is None
        assert "can't route to" in error

    def test_bedrock_without_claude_rejected(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service(
            "claude", "main.schema2.bedrock-titan-svc", WS, "token"
        )
        assert service is None
        assert "no Claude models" in error

    def test_not_found_lists_usable(self, monkeypatch):
        self._patch(monkeypatch)
        service, error = db_mod.resolve_provider_service("claude", "main.x.missing", WS, "token")
        assert service is None
        assert "was not found" in error
        assert "main.schema1.anthropic-svc" in error

    def test_feature_unavailable(self, monkeypatch):
        reason = "HTTP 400 Bad Request: ModelProviderService feature is not available"
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token, timeout=30: (None, reason))
        service, error = db_mod.resolve_provider_service("claude", "main.x.y", WS, "token")
        assert service is None
        assert "not available" in error

    def test_gemini_enterprise_ok_for_gemini(self, monkeypatch):
        payload = {
            "model_provider_services": [
                {
                    "name": "model-provider-services/main.schema1.gemini-svc",
                    "config": {
                        "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_GEMINI_ENTERPRISE",
                        "targets": [{"model": "gemini-3.5-flash"}],
                    },
                }
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )
        service, error = db_mod.resolve_provider_service(
            "gemini", "main.schema1.gemini-svc", WS, "token"
        )
        assert error is None
        assert service["provider_type"] == "gemini_enterprise"
        assert service["targets"] == ["gemini-3.5-flash"]


class TestModelProviderFeatureUnavailable:
    def test_detects_feature_not_available(self):
        reason = (
            'HTTP 400 Bad Request: {"error_code":"BAD_REQUEST",'
            '"message":"ModelProviderService feature is not available"}'
        )
        assert db_mod.is_model_provider_feature_unavailable(reason) is True

    def test_false_for_other_errors(self):
        assert db_mod.is_model_provider_feature_unavailable("HTTP 500 Server Error") is False
        assert db_mod.is_model_provider_feature_unavailable(None) is False


class TestListMcpServices:
    def test_accepts_entries_without_connection_status(self, monkeypatch):
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"usage_tracking": {"enabled": True}, "tracing": {"enabled": True}},
                },
                {
                    "name": "mcp-services/system.ai.atlassian",
                    "config": {},
                },
                {
                    "name": "mcp-services/system.ai.slack",
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert reason is None
        assert names == ["system.ai.atlassian", "system.ai.github", "system.ai.slack"]

    def test_accepts_legacy_active_status(self, monkeypatch):
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"connection": {"status": "ACTIVE"}},
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert reason is None
        assert names == ["system.ai.github"]

    def test_rejects_explicit_non_active_status(self, monkeypatch):
        # If the field is present and non-ACTIVE, drop the entry — the
        # backing connection is broken and the proxy will fail.
        payload = {
            "mcp_services": [
                {
                    "name": "mcp-services/system.ai.github",
                    "config": {"connection": {"status": "ACTIVE"}},
                },
                {
                    "name": "mcp-services/system.ai.broken",
                    "config": {"connection": {"status": "FAILED"}},
                },
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, _reason = db_mod.list_mcp_services(WS, "token")

        assert names == ["system.ai.github"]

    def test_ignores_non_system_ai_entries(self, monkeypatch):
        payload = {
            "mcp_services": [
                {"name": "mcp-services/system.ai.github"},
                {"name": "mcp-services/main.schema3.github_mcp"},
                {"name": "mcp-services/temp.erni.github_mcp"},
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, _reason = db_mod.list_mcp_services(WS, "token")

        assert names == ["system.ai.github"]

    def test_http_failure_propagates_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (None, "HTTP 500 Server Error"),
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert names == []
        assert reason == "HTTP 500 Server Error"

    def test_empty_payload_is_successful_with_no_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: ({"mcp_services": []}, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token")

        assert names == []
        assert reason is None

    def test_custom_parent_passes_through_to_url(self, monkeypatch):
        captured: dict[str, str] = {}

        def fake_get(url, token, timeout=30):
            captured["url"] = url
            return {"mcp_services": []}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        db_mod.list_mcp_services(WS, "token", parent="main.schema3")

        assert "parent=schemas%2Fmain.schema3" in captured["url"]

    def test_custom_parent_filters_to_namespace(self, monkeypatch):
        payload = {
            "mcp_services": [
                {"name": "mcp-services/main.schema3.github"},
                {"name": "mcp-services/main.schema3.slack"},
                {"name": "mcp-services/system.ai.github"},
            ]
        }
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

        names, reason = db_mod.list_mcp_services(WS, "token", parent="main.schema3")

        assert reason is None
        assert names == ["main.schema3.github", "main.schema3.slack"]

    def test_http_404_reason_surfaces_for_invalid_parent(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (None, "HTTP 404 Not Found: NOT_FOUND"),
        )

        names, reason = db_mod.list_mcp_services(WS, "token", parent="nope.nope")

        assert names == []
        assert reason and reason.startswith("HTTP 404")


class TestListAllMcpServices:
    """Workspace-wide walk: catalogs -> schemas -> per-schema mcp-services."""

    def _fake_http(self, catalogs, schemas_by_catalog, services_by_schema):
        """Route `_http_get_json` by URL to the right stubbed payload."""

        def fake_get(url, token, timeout=30):
            if "unity-catalog/catalogs" in url:
                return {"catalogs": [{"name": c} for c in catalogs]}, None
            if "unity-catalog/schemas" in url:
                cat = url.split("catalog_name=")[1].split("&")[0]
                return {"schemas": [{"name": s} for s in schemas_by_catalog.get(cat, [])]}, None
            if "unity-catalog/mcp-services" in url:
                # parent is url-encoded as `schemas%2F<cat>.<schema>`
                parent = url.split("parent=")[1].split("&")[0]
                schema_ref = parent.replace("schemas%2F", "").replace("schemas/", "")
                return {
                    "mcp_services": [
                        {"name": f"mcp-services/{full}"}
                        for full in services_by_schema.get(schema_ref, [])
                    ]
                }, None
            return None, "unexpected url"

        return fake_get

    def test_aggregates_services_across_catalogs_and_schemas(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            self._fake_http(
                catalogs=["mycat", "other"],
                schemas_by_catalog={"mycat": ["myschema", "information_schema"], "other": ["ops"]},
                services_by_schema={
                    "mycat.myschema": ["mycat.myschema.weather", "mycat.myschema.news"],
                    "other.ops": ["other.ops.pager"],
                },
            ),
        )

        names, reason = db_mod.list_all_mcp_services(WS, "token")

        assert reason is None
        # information_schema is skipped; results are sorted and de-duplicated.
        assert names == [
            "mycat.myschema.news",
            "mycat.myschema.weather",
            "other.ops.pager",
        ]

    def test_reports_progress_per_schema(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            self._fake_http(
                catalogs=["mycat"],
                schemas_by_catalog={"mycat": ["a", "b"]},
                services_by_schema={"mycat.a": ["mycat.a.one"], "mycat.b": ["mycat.b.two"]},
            ),
        )
        progress: list[tuple[int, int, int]] = []

        names, reason = db_mod.list_all_mcp_services(
            WS,
            "token",
            on_progress=lambda done, total, found: progress.append((done, total, found)),
        )

        assert reason is None
        assert names == ["mycat.a.one", "mycat.b.two"]
        # One callback per schema; the total is fixed and done/found climb.
        assert len(progress) == 2
        assert [p[1] for p in progress] == [2, 2]
        assert progress[-1][0] == 2
        assert progress[-1][2] == 2

    def test_skips_internal_catalogs(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            self._fake_http(
                catalogs=["system", "hive_metastore", "samples", "__databricks_internal"],
                schemas_by_catalog={},
                services_by_schema={},
            ),
        )

        names, reason = db_mod.list_all_mcp_services(WS, "token")

        assert names == []
        assert reason == "no user UC catalogs found"

    def test_returns_reason_when_no_catalogs(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: ({"catalogs": []}, None)
        )

        names, reason = db_mod.list_all_mcp_services(WS, "token")

        assert names == []
        assert reason == "no UC catalogs found"


def _foundation_models_payload(names):
    return {
        "endpoints": [
            {
                "name": name,
                "config": {
                    "served_entities": [
                        {
                            "foundation_model": {
                                "ai_gateway_v2_supported": True,
                                "api_types": ["gemini/v1/generateContent"],
                            }
                        }
                    ]
                },
            }
            for name in names
        ]
    }


class TestModelVersionSortKey:
    def test_orders_newest_version_first(self):
        names = [
            "databricks-gemini-2-5-flash",
            "databricks-gemini-2-5-pro",
            "databricks-gemini-3-1-flash-lite",
            "databricks-gemini-3-1-pro",
            "databricks-gemini-3-5-flash",
            "databricks-gemini-3-flash",
            "databricks-gemini-3-pro",
        ]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered[0] == "databricks-gemini-3-5-flash"

    def test_treats_bare_major_as_dot_zero(self):
        # 3-flash is 3.0, so 3-5-flash (3.5) must sort ahead of it.
        names = ["databricks-gemini-3-flash", "databricks-gemini-3-5-flash"]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered == [
            "databricks-gemini-3-5-flash",
            "databricks-gemini-3-flash",
        ]

    def test_unversioned_names_sort_last_alphabetically(self):
        names = ["databricks-gemini-2-5-flash", "custom-endpoint", "another-endpoint"]
        ordered = sorted(names, key=db_mod.model_version_sort_key)
        assert ordered[0] == "databricks-gemini-2-5-flash"
        assert ordered[1:] == ["another-endpoint", "custom-endpoint"]


class TestDiscoverGeminiModels:
    def test_returns_newest_flash_first(self, monkeypatch):
        payload = _foundation_models_payload(
            [
                "databricks-gemini-2-5-flash",
                "databricks-gemini-3-5-flash",
                "databricks-gemini-3-flash",
            ]
        )
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_gemini_models(WS, "token")

        assert reason is None
        assert models[0] == "databricks-gemini-3-5-flash"

    def test_codex_discovery_orders_newest_version_first(self, monkeypatch):
        # Codex orders newest model version first (like gemini), so the picker's top choice and
        # default is the newest gpt, not the alphabetically-first one.
        payload = {
            "endpoints": [
                {
                    "name": name,
                    "config": {
                        "served_entities": [
                            {
                                "foundation_model": {
                                    "ai_gateway_v2_supported": True,
                                    "api_types": ["openai/v1/responses"],
                                }
                            }
                        ]
                    },
                }
                for name in ["databricks-gpt-4-1", "databricks-gpt-5-2-codex"]
            ]
        }
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: (payload, None))

        models, reason = db_mod.discover_codex_models(WS, "token")

        assert reason is None
        assert models == ["databricks-gpt-5-2-codex", "databricks-gpt-4-1"]


class TestResolvePatToken:
    def test_reads_pat_profile_token_from_cfg(self, monkeypatch, tmp_path):
        cfg = tmp_path / "databrickscfg"
        cfg.write_text(f"[lakebox]\nhost = {WS}\ntoken = dapi-from-cfg\n")
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [{"name": "lakebox", "host": WS, "auth_type": "pat"}],
        )
        assert db_mod.resolve_pat_token("lakebox") == "dapi-from-cfg"

    def test_default_section_token_does_not_leak_into_named_profiles(self, monkeypatch, tmp_path):
        cfg = tmp_path / "databrickscfg"
        cfg.write_text(
            f"[DEFAULT]\nhost = {WS}\ntoken = dapi-default\n"
            "[other]\nhost = https://other.databricks.com\n"
        )
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [
                {"name": "DEFAULT", "host": WS, "auth_type": "pat"},
                {"name": "other", "host": "https://other.databricks.com", "auth_type": "pat"},
            ],
        )
        assert db_mod.resolve_pat_token("DEFAULT") == "dapi-default"
        assert db_mod.resolve_pat_token("other") is None

    def test_returns_none_for_oauth_profile(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "list_profile_entries",
            lambda: [{"name": "oauth", "host": WS, "auth_type": "databricks-cli"}],
        )
        assert db_mod.resolve_pat_token("oauth") is None

    def test_returns_none_without_profile(self):
        assert db_mod.resolve_pat_token(None) is None


class TestApplyPatEnvironment:
    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # apply_pat_environment writes os.environ directly; restore it even
        # though monkeypatch can't track writes made by code under test.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_exports_bearer_for_use_pat_state(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"use_pat": True, "profile": "DEFAULT"})

        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_noop_without_use_pat(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"profile": "DEFAULT"})

        assert "DATABRICKS_BEARER" not in os.environ

    def test_existing_bearer_wins(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "explicit-bearer")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")

        db_mod.apply_pat_environment({"use_pat": True, "profile": "DEFAULT"})

        assert os.environ["DATABRICKS_BEARER"] == "explicit-bearer"


class TestBuildAuthTokenArgv:
    def test_basic_argv(self):
        argv = build_auth_token_argv(WS)
        # First element resolves to the ucode executable; the rest is the
        # cross-platform helper invocation — no `sh`, no `jq`, no shell syntax.
        assert argv[0].endswith("ucode") or argv[0] == "ucode"
        assert argv[1:] == ["auth-token", "--host", WS]

    def test_strips_trailing_slash_from_host(self):
        argv = build_auth_token_argv(WS + "/")
        assert "--host" in argv
        assert argv[argv.index("--host") + 1] == WS

    def test_embeds_profile_when_provided(self):
        argv = build_auth_token_argv(WS, profile="stablebox")
        assert argv[argv.index("--profile") + 1] == "stablebox"

    def test_profile_passed_as_separate_argv_element(self):
        # Metacharacters need no shell quoting — argv is never parsed by a shell.
        argv = build_auth_token_argv(WS, profile="weird name; rm -rf /")
        assert "weird name; rm -rf /" in argv

    def test_use_pat_flag(self):
        argv = build_auth_token_argv(WS, profile="DEFAULT", use_pat=True)
        assert "--use-pat" in argv
        assert argv[argv.index("--profile") + 1] == "DEFAULT"

    def test_no_use_pat_flag_by_default(self):
        assert "--use-pat" not in build_auth_token_argv(WS)


class TestBuildAuthShellCommand:
    def test_contains_workspace(self):
        cmd = build_auth_shell_command(WS)
        assert WS in cmd

    def test_is_ucode_auth_token_invocation(self):
        # The persisted helper now points at the `ucode auth-token` executable
        # on every platform — not a POSIX `databricks ... | jq` pipeline.
        cmd = build_auth_shell_command(WS)
        assert "auth-token" in cmd
        assert "--host" in cmd
        # POSIX-only constructs that broke Windows (#116) must be gone.
        assert "jq" not in cmd
        assert "if [ -n" not in cmd

    def test_embeds_profile_when_provided(self):
        cmd = build_auth_shell_command(WS, profile="stablebox")
        assert "--profile stablebox" in cmd

    def test_quotes_profile_shell_metacharacters(self):
        cmd = build_auth_shell_command(WS, profile="weird name; rm -rf /")
        # On POSIX shlex.join quotes the value so the string form cannot be
        # interpreted as a shell injection if a tool runs it via a shell.
        if os.name != "nt":
            assert "'weird name; rm -rf /'" in cmd

    def test_use_pat_emits_flag(self):
        cmd = build_auth_shell_command(WS, profile="DEFAULT", use_pat=True)
        assert "--use-pat" in cmd
        assert "--profile DEFAULT" in cmd


class TestEnsurePatBearer:
    """ensure_pat_bearer is the empty-aware DATABRICKS_BEARER export used by the
    --use-pat path on configure, launch, and the auth-token helper."""

    @pytest.fixture(autouse=True)
    def _isolated_bearer(self):
        # ensure_pat_bearer writes os.environ directly; restore it even though
        # monkeypatch can't track writes made by code under test.
        original = os.environ.pop("DATABRICKS_BEARER", None)
        yield
        if original is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = original

    def test_exports_pat_when_env_absent(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")
        assert ensure_pat_bearer("p") is True
        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_overwrites_empty_env(self, monkeypatch):
        # The regression: an empty DATABRICKS_BEARER must be treated as absent
        # so the PAT is still exported (old `if [ -n ... ]` parity).
        monkeypatch.setenv("DATABRICKS_BEARER", "")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")
        assert ensure_pat_bearer("p") is True
        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_non_empty_env_wins_without_resolving(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "ci-bearer")
        called = []
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: called.append(p) or "dapi-pat")
        assert ensure_pat_bearer("p") is True
        # Pre-set bearer is honored; we don't even read the PAT.
        assert os.environ["DATABRICKS_BEARER"] == "ci-bearer"
        assert called == []

    def test_returns_false_when_no_pat(self, monkeypatch):
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: None)
        assert ensure_pat_bearer("p") is False
        assert "DATABRICKS_BEARER" not in os.environ

    def test_whitespace_only_env_treated_as_empty(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_BEARER", "   ")
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: "dapi-pat")
        assert ensure_pat_bearer("p") is True
        assert os.environ["DATABRICKS_BEARER"] == "dapi-pat"

    def test_explicit_pat_arg_skips_cfg_read(self, monkeypatch):
        # Callers that already resolved the PAT (configure_shared_state) pass it
        # in; ensure_pat_bearer must use it without re-reading ~/.databrickscfg.
        called = []
        monkeypatch.setattr(db_mod, "resolve_pat_token", lambda p: called.append(p) or "from-cfg")
        assert ensure_pat_bearer("p", "explicit-pat") is True
        assert os.environ["DATABRICKS_BEARER"] == "explicit-pat"
        assert called == []


class TestFormatSubprocessResult:
    def test_suppresses_stdout_on_success(self):
        result = subprocess.CompletedProcess(
            args=["databricks", "auth", "token"],
            returncode=0,
            stdout='{"access_token": "dapi-secret-do-not-leak", "token_type": "Bearer"}',
            stderr="",
        )
        formatted = _format_subprocess_result(result)
        assert "dapi-secret-do-not-leak" not in formatted
        assert "rc=0" in formatted

    def test_includes_stdout_on_failure(self):
        result = subprocess.CompletedProcess(
            args=["databricks", "auth", "token"],
            returncode=1,
            stdout="useful diagnostic output",
            stderr="error: no matching profile",
        )
        formatted = _format_subprocess_result(result)
        assert "rc=1" in formatted
        assert "useful diagnostic output" in formatted
        assert "no matching profile" in formatted


class TestScrubDatabrickscfg:
    def test_redacts_token_value(self):
        text = "[DEFAULT]\nhost = https://example.databricks.com\ntoken = dapi-secret\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "dapi-secret" not in scrubbed
        assert "token = <redacted>" in scrubbed
        assert "host = https://example.databricks.com" in scrubbed

    def test_redacts_various_secret_keys(self):
        text = (
            "[p]\n"
            "client_secret = secret-val-1\n"
            "bearer_token = secret-val-2\n"
            "api_key = secret-val-3\n"
            "password = secret-val-4\n"
            "auth_type = oauth-u2m\n"
        )
        scrubbed = _scrub_databrickscfg(text)
        for secret in ("secret-val-1", "secret-val-2", "secret-val-3", "secret-val-4"):
            assert secret not in scrubbed
        assert "auth_type = oauth-u2m" in scrubbed

    def test_preserves_comments_and_sections(self):
        text = "# comment\n[DEFAULT]\nhost = https://x\n; another comment with token = leak\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "# comment" in scrubbed
        assert "[DEFAULT]" in scrubbed
        assert "; another comment with token = leak" in scrubbed

    def test_key_matching_is_case_insensitive(self):
        text = "[p]\nTOKEN = upper\nAccess_Token = mixed\n"
        scrubbed = _scrub_databrickscfg(text)
        assert "upper" not in scrubbed
        assert "mixed" not in scrubbed


class TestScrubJson:
    def test_redacts_secret_keys(self):
        payload = {
            "access_token": "dapi-secret",
            "host": "https://example.databricks.com",
        }
        scrubbed = _scrub_json(payload)
        assert isinstance(scrubbed, dict)
        assert scrubbed["access_token"] == "<redacted>"
        assert scrubbed["host"] == "https://example.databricks.com"

    def test_recurses_into_nested_structures(self):
        payload = {
            "profiles": [
                {"name": "DEFAULT", "client_secret": "abc"},
                {"name": "other", "password": "pw"},
            ]
        }
        scrubbed = _scrub_json(payload)
        assert scrubbed == {
            "profiles": [
                {"name": "DEFAULT", "client_secret": "<redacted>"},
                {"name": "other", "password": "<redacted>"},
            ]
        }

    def test_passes_through_scalars_and_non_secret_keys(self):
        assert _scrub_json("plain") == "plain"
        assert _scrub_json(42) == 42
        assert _scrub_json({"host": "x", "auth_type": "pat"}) == {
            "host": "x",
            "auth_type": "pat",
        }


class TestGetDatabricksToken:
    def _fake_databricks(self, tmp_path, script: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\n{script}\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_returns_token_on_success(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "good-token"

    def test_strips_ambient_profile_when_profile_not_provided(self, tmp_path, monkeypatch):
        profile_log = tmp_path / "profile"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s" "${{DATABRICKS_CONFIG_PROFILE:-}}" > {profile_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        env["DATABRICKS_CONFIG_PROFILE"] = "other-workspace"
        monkeypatch.setattr("os.environ", env)

        token = get_databricks_token(WS)

        assert token == "good-token"
        assert profile_log.read_text() == ""

    def test_has_valid_auth_strips_ambient_profile_without_explicit_profile(
        self, tmp_path, monkeypatch
    ):
        profile_log = tmp_path / "profile"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s" "${{DATABRICKS_CONFIG_PROFILE:-}}" > {profile_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        env["DATABRICKS_CONFIG_PROFILE"] = "other-workspace"
        monkeypatch.setattr("os.environ", env)

        assert db_mod.has_valid_databricks_auth(WS)
        assert profile_log.read_text() == ""

    def test_reauths_and_retries_when_token_empty(self, tmp_path, monkeypatch):
        call_count = tmp_path / "calls"
        call_count.write_text("0")
        env = self._fake_databricks(
            tmp_path,
            f"count=$(cat {call_count})\n"
            f"echo $((count + 1)) > {call_count}\n"
            'case "$*" in\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'if [ "$count" -eq 0 ]; then\n'
            '  echo \'{"access_token": "", "token_type": "Bearer"}\'\n'
            "else\n"
            '  echo \'{"access_token": "refreshed-token", "token_type": "Bearer"}\'\n'
            "fi",
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "refreshed-token"

    def test_retries_on_cache_lock_contention(self, tmp_path, monkeypatch):
        # Concurrent `databricks auth token` calls racing on the shared token
        # cache fail with "cache update: exit status 45". That's transient (the
        # credential is fine), so we must retry — not treat it as a dead session.
        call_count = tmp_path / "calls"
        call_count.write_text("0")
        env = self._fake_databricks(
            tmp_path,
            f"count=$(cat {call_count})\n"
            f"echo $((count + 1)) > {call_count}\n"
            'if [ "$count" -lt 2 ]; then\n'
            '  echo "Error: forced token refresh: cache update: exit status 45" >&2\n'
            "  exit 1\n"
            "else\n"
            '  echo \'{"access_token": "won-the-lock", "token_type": "Bearer"}\'\n'
            "fi",
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS)
        assert token == "won-the-lock"

    def test_raises_when_reauth_also_fails(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="no access token"):
            get_databricks_token(WS)

    def test_passes_profile_flag_when_provided(self, tmp_path, monkeypatch):
        # Fake CLI that records its argv to a file so we can assert the
        # --profile flag is forwarded to `databricks auth token`.
        argv_log = tmp_path / "argv"
        env = self._fake_databricks(
            tmp_path,
            f'printf "%s\\n" "$@" >> {argv_log}\n'
            'echo \'{"access_token": "good-token", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)
        token = get_databricks_token(WS, profile="stablebox")
        assert token == "good-token"
        argv = argv_log.read_text().splitlines()
        assert "--profile" in argv
        assert argv[argv.index("--profile") + 1] == "stablebox"

    def test_error_suggests_logout_when_matching_profile_exists(self, tmp_path, monkeypatch):
        env = self._fake_databricks(
            tmp_path,
            'case "$*" in\n'
            '  *"auth profiles"*) echo \'{"profiles": [{"host": "'
            + WS
            + '", "name": "example-profile", "auth_type": "databricks-cli"}]}\'; exit 0 ;;\n'
            '  *"auth login"*) exit 0 ;;\n'
            "esac\n"
            'echo \'{"access_token": "", "token_type": "Bearer"}\'',
        )
        monkeypatch.setattr("os.environ", env)

        with pytest.raises(RuntimeError) as exc_info:
            get_databricks_token(WS)

        message = str(exc_info.value)
        assert "stale or invalid" in message
        assert "databricks auth logout --profile example-profile" in message
        assert f"databricks auth login --host {WS} --profile example-profile" in message


class TestGetDatabricksProfiles:
    def _patched_run(self, monkeypatch, payload: dict, returncode: int = 0) -> None:
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, returncode, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

    def test_keeps_duplicate_hosts_as_separate_entries(self, monkeypatch):
        self._patched_run(
            monkeypatch,
            {
                "profiles": [
                    {"host": WS, "name": "first", "auth_type": "databricks-cli"},
                    {"host": WS, "name": "second", "auth_type": "databricks-cli"},
                    {
                        "host": "https://other.databricks.com",
                        "name": "third",
                        "auth_type": "databricks-cli",
                    },
                ]
            },
        )
        profiles = get_databricks_profiles()
        assert profiles == [
            (WS, "first"),
            (WS, "second"),
            ("https://other.databricks.com", "third"),
        ]

    def test_skips_pat_profiles(self, monkeypatch):
        self._patched_run(
            monkeypatch,
            {
                "profiles": [
                    {"host": WS, "name": "oauth", "auth_type": "databricks-cli"},
                    {"host": WS, "name": "tokenized", "auth_type": "pat"},
                ]
            },
        )
        assert get_databricks_profiles() == [(WS, "oauth")]

    def test_strips_trailing_slash_on_host(self, monkeypatch):
        self._patched_run(
            monkeypatch,
            {
                "profiles": [
                    {"host": f"{WS}/", "name": "p", "auth_type": "databricks-cli"},
                ]
            },
        )
        assert get_databricks_profiles() == [(WS, "p")]

    def test_returns_empty_on_non_zero_exit(self, monkeypatch):
        self._patched_run(monkeypatch, {"profiles": []}, returncode=1)
        assert get_databricks_profiles() == []


class TestListDatabricksConnections:
    def test_lists_paginated_connections_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"connections": [{"name": "jira-mcp", "connection_type": "HTTP"}]}
            else:
                payload = {
                    "connections": [{"name": "confluence-mcp", "connection_type": "HTTP"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_connections(WS) == [
            {"name": "confluence-mcp", "connection_type": "HTTP"},
            {"name": "jira-mcp", "connection_type": "HTTP"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "connections",
            "list",
            "--max-results",
            "0",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"connections": []}))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_databricks_connections(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_connections(WS)


class TestListGenieSpaces:
    def test_lists_paginated_spaces_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if "--page-token" in args:
                payload = {"spaces": [{"space_id": "space-2", "title": "Second"}]}
            else:
                payload = {
                    "spaces": [{"space_id": "space-1", "title": "First"}],
                    "next_page_token": "next-page",
                }
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_genie_spaces(WS) == [
            {"space_id": "space-1", "title": "First"},
            {"space_id": "space-2", "title": "Second"},
        ]
        assert calls[0]["args"] == [
            "databricks",
            "genie",
            "list-spaces",
            "--page-size",
            "100",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS
        assert calls[1]["args"][-2:] == ["--page-token", "next-page"]

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"spaces": []}))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_genie_spaces(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_genie_spaces(WS)


class TestListDatabricksApps:
    def test_lists_apps_with_workspace_env(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            payload = [
                {
                    "name": "my-app",
                    "url": "https://my-app.example.databricksapps.com",
                }
            ]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_apps(WS) == [
            {
                "name": "my-app",
                "url": "https://my-app.example.databricksapps.com",
            }
        ]
        assert calls[0]["args"] == [
            "databricks",
            "apps",
            "list",
            "--limit",
            "1000",
            "--output",
            "json",
        ]
        assert calls[0]["kwargs"]["env"]["DATABRICKS_HOST"] == WS

    def test_passes_profile_when_provided(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps([]))

        monkeypatch.setattr(db_mod, "run", fake_run)

        list_databricks_apps(WS, "my-profile")

        assert "--profile" in calls[0]
        assert calls[0][calls[0].index("--profile") + 1] == "my-profile"

    def test_accepts_object_wrapped_apps(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps({"apps": [{"name": "my-app", "url": "https://example.com"}]}),
            )

        monkeypatch.setattr(db_mod, "run", fake_run)

        assert list_databricks_apps(WS) == [{"name": "my-app", "url": "https://example.com"}]

    def test_raises_on_invalid_json(self, monkeypatch):
        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="not-json")

        monkeypatch.setattr(db_mod, "run", fake_run)

        with pytest.raises(RuntimeError, match="invalid JSON"):
            list_databricks_apps(WS)


class TestProbeUnityGatewayCapabilities:
    def test_model_service_resource_returns_success(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, token):
            calls.append(url)
            return {"model_services": [{"name": "model-services/system.ai.gpt-5"}]}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        model_service_probe = db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        assert model_service_probe == db_mod.GatewayProbe(
            True, "reachable, accessible model service returned", True
        )
        assert calls == [
            f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50",
        ]

    def test_empty_first_page_follows_cursor_to_accessible_model_service(self, monkeypatch):
        calls: list[str] = []
        responses = iter(
            [
                ({"next_page_token": "cursor-1"}, None),
                ({"model_services": [{"name": "model-services/system.ai.gpt-5"}]}, None),
            ]
        )

        def fake_get(url, token):
            calls.append(url)
            return next(responses)

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        model_service_probe = db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        assert model_service_probe == db_mod.GatewayProbe(
            True, "reachable, accessible model service returned", True
        )
        assert calls == [
            f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50",
            f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50"
            "&page_token=cursor-1",
        ]

    def test_empty_pages_exhaust_cursor_before_reporting_no_model_service(self, monkeypatch):
        calls: list[str] = []
        responses = iter(
            [
                ({"next_page_token": "cursor-1"}, None),
                ({"model_services": []}, None),
            ]
        )

        def fake_get(url, token):
            calls.append(url)
            return next(responses)

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        model_service_probe = db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        assert model_service_probe == db_mod.GatewayProbe(
            True,
            "reachable, no accessible model services returned; check USE CATALOG on system, and "
            "USE SCHEMA and EXECUTE on system.ai",
        )
        assert calls == [
            f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50",
            f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50"
            "&page_token=cursor-1",
        ]

    def test_empty_model_service_response_includes_permission_hint(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token: ({}, None),
        )

        model_service_probe = db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        assert model_service_probe == db_mod.GatewayProbe(
            True,
            "reachable, no accessible model services returned; check USE CATALOG on system, and "
            "USE SCHEMA and EXECUTE on system.ai",
        )

    def test_model_service_unavailable_raises(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, token):
            calls.append(url)
            return None, "HTTP 404: model services unavailable"

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        with pytest.raises(RuntimeError, match="not enabled") as excinfo:
            db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        message = str(excinfo.value)
        assert "HTTP 404: model services unavailable" in message
        # surfaced errors must stay free of internal version vocabulary
        assert "v2" not in message.lower()
        assert "v3" not in message.lower()
        assert "legacy" not in message.lower()
        assert calls == [f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50"]

    def test_model_service_auth_failure_raises(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, token):
            calls.append(url)
            return None, "HTTP 401: Unauthorized"

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        with pytest.raises(RuntimeError, match="rejected"):
            db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        assert calls == [f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50"]

    def test_missing_scope_403_routes_to_reauth_not_grants(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, token):
            calls.append(url)
            return None, (
                "HTTP 403 Forbidden: Provided OAuth token does not have required "
                "scopes: unity-catalog"
            )

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        with pytest.raises(RuntimeError, match="missing an OAuth scope") as excinfo:
            db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        message = str(excinfo.value)
        assert "databricks auth login" in message
        assert "USE CATALOG" not in message
        assert "USE SCHEMA" not in message
        assert calls == [f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50"]

    def test_probe_model_services_later_page_error_is_reachable_not_empty(self, monkeypatch):
        responses = iter(
            [
                ({"next_page_token": "cursor-1"}, None),
                (None, "HTTP 500: Internal Server Error"),
            ]
        )
        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: next(responses))

        assert db_mod._probe_model_services(WS, "fake-token") == db_mod.GatewayProbe(
            True, "reachable", conclusive=False
        )

    def test_probe_model_services_page_cap_with_pending_cursor_is_reachable_not_empty(
        self, monkeypatch
    ):
        calls: list[str] = []

        def fake_get(url, token):
            calls.append(url)
            return {"next_page_token": "more"}, None

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        assert db_mod._probe_model_services(WS, "fake-token") == db_mod.GatewayProbe(
            True, "reachable", conclusive=False
        )
        assert len(calls) == db_mod._MODEL_SERVICE_PROBE_MAX_PAGES

    def test_inconclusive_model_service_probe_returns_reachable(self, monkeypatch):
        responses = iter(
            [
                ({"next_page_token": "cursor-1"}, None),
                (None, "HTTP 500: Internal Server Error"),
            ]
        )

        monkeypatch.setattr(db_mod, "_http_get_json", lambda url, token: next(responses))

        assert db_mod.probe_unity_gateway_capabilities(WS, "fake-token") == db_mod.GatewayProbe(
            True, "reachable", conclusive=False
        )

    def test_scope_failure_matches_oauth_token_but_not_pat(self):
        assert db_mod._looks_like_scope_failure(
            "HTTP 403 Forbidden: Provided OAuth token does not have required scopes: unity-catalog"
        )
        assert not db_mod._looks_like_scope_failure(
            "HTTP 403 Forbidden: Provided access token does not have required scopes"
        )
        assert not db_mod._looks_like_scope_failure("HTTP 403: Missing Unity Catalog grants")

    def test_model_service_forbidden_reports_permission_error(self, monkeypatch):
        calls: list[str] = []

        def fake_get(url, token):
            calls.append(url)
            return None, "HTTP 403: Missing Unity Catalog grants"

        monkeypatch.setattr(db_mod, "_http_get_json", fake_get)

        with pytest.raises(RuntimeError, match="model service access could not be verified") as e:
            db_mod.probe_unity_gateway_capabilities(WS, "fake-token")

        message = str(e.value)
        assert "USE SCHEMA" in message
        assert "EXECUTE" in message
        assert "v2" not in message.lower()
        assert "v3" not in message.lower()
        assert "legacy" not in message.lower()
        assert "rejected the access token" not in message
        assert "not enabled" not in message
        assert calls == [f"https://{WS_HOST}/api/2.1/unity-catalog/model-services?page_size=50"]


class TestHttpGetJsonReason:
    """The `reason` string returned by `_http_get_json` must include the response body
    so callers (e.g. the Unity Gateway capability probe) can route on it. Before issue #84's fix
    the body was logged only when UCODE_DEBUG=1 and dropped from the bubbled error."""

    @staticmethod
    def _http_error(code: int, msg: str, body: str = ""):
        import io
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        fp = io.BytesIO(body.encode("utf-8")) if body else None
        return HTTPError(url="", code=code, msg=msg, hdrs=MagicMock(), fp=fp)

    def test_reason_includes_body_on_http_error(self):
        from unittest.mock import patch

        from ucode.databricks import _http_get_json

        exc = self._http_error(400, "Bad Request", body="Invalid Token")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            payload, reason = _http_get_json("https://x/y", "tok")
        assert payload is None
        assert "HTTP 400" in reason
        assert "Invalid Token" in reason

    def test_reason_without_body_is_status_only(self):
        from unittest.mock import patch

        from ucode.databricks import _http_get_json

        exc = self._http_error(404, "Not Found")
        with patch("ucode.databricks.urllib_request.urlopen", side_effect=exc):
            payload, reason = _http_get_json("https://x/y", "tok")
        assert payload is None
        assert reason == "HTTP 404 Not Found"


class TestHttpGetJsonRetries:
    @staticmethod
    def _http_error(code: int, message: str, headers: dict[str, str] | None = None):
        from urllib.error import HTTPError

        return HTTPError(url="", code=code, msg=message, hdrs=headers or {}, fp=None)

    def test_retries_429_after_retry_after_delay(self, monkeypatch):
        outcomes = iter(
            [
                self._http_error(429, "Too Many Requests", {"Retry-After": "1"}),
                _FakeResponse({"data": []}),
            ]
        )
        calls = []
        sleeps = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", fake_urlopen)
        monkeypatch.setattr(db_mod.random, "uniform", lambda low, high: high)
        monkeypatch.setattr(db_mod.time, "sleep", sleeps.append)

        payload, reason = db_mod._http_get_json("https://x/y", "tok", max_retries=2)

        assert payload == {"data": []}
        assert reason is None
        assert len(calls) == 2
        assert sleeps == [1.25]

    def test_retries_network_error_with_exponential_backoff(self, monkeypatch):
        from urllib.error import URLError

        outcomes = iter([URLError("connection reset"), _FakeResponse({"ok": True})])
        sleeps = []

        def fake_urlopen(request, timeout=None):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", fake_urlopen)
        monkeypatch.setattr(db_mod.random, "uniform", lambda low, high: high)
        monkeypatch.setattr(db_mod.time, "sleep", sleeps.append)

        payload, reason = db_mod._http_get_json("https://x/y", "tok", max_retries=2)

        assert payload == {"ok": True}
        assert reason is None
        assert sleeps == [1.25]

    def test_stops_after_configured_retries(self, monkeypatch):
        calls = []
        sleeps = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            raise self._http_error(429, "Too Many Requests")

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", fake_urlopen)
        monkeypatch.setattr(db_mod.random, "uniform", lambda low, high: high)
        monkeypatch.setattr(db_mod.time, "sleep", sleeps.append)

        payload, reason = db_mod._http_get_json("https://x/y", "tok", max_retries=2)

        assert payload is None
        assert reason == "HTTP 429 Too Many Requests"
        assert len(calls) == 3
        assert sleeps == [1.25, 2.5]

    def test_does_not_retry_other_http_error(self, monkeypatch):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            raise self._http_error(503, "Service Unavailable")

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            db_mod.time,
            "sleep",
            lambda delay: pytest.fail(f"unexpected retry delay: {delay}"),
        )

        payload, reason = db_mod._http_get_json("https://x/y", "tok", max_retries=2)

        assert payload is None
        assert reason == "HTTP 503 Service Unavailable"
        assert len(calls) == 1


class TestParseDatabricksCliVersion:
    def test_parses_standard_format(self):
        assert _parse_databricks_cli_version("Databricks CLI v0.299.2") == (0, 299, 2)

    def test_parses_without_v_prefix(self):
        assert _parse_databricks_cli_version("Databricks CLI 0.298.0") == (0, 298, 0)

    def test_returns_none_on_garbage(self):
        assert _parse_databricks_cli_version("not a version") is None


class TestEnsureDatabricksCliVersion:
    def _fake_databricks(self, tmp_path, version_output: str) -> dict:
        fake = tmp_path / "databricks"
        fake.write_text(f"#!/bin/sh\necho '{version_output}'\n")
        fake.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    def test_passes_when_version_meets_minimum(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "Databricks CLI v1.0.0")
        monkeypatch.setattr("os.environ", env)
        ensure_databricks_cli_version()  # should not raise

    def test_passes_when_version_exceeds_minimum(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "Databricks CLI v1.8.0")
        monkeypatch.setattr("os.environ", env)
        ensure_databricks_cli_version()

    def test_auto_upgrades_when_version_too_old(self, tmp_path, monkeypatch):
        import ucode.databricks as db_mod

        env = self._fake_databricks(tmp_path, "Databricks CLI v0.299.2")
        monkeypatch.setattr("os.environ", env)
        upgraded = []
        monkeypatch.setattr(
            db_mod,
            "_run_databricks_cli_installer",
            lambda brew_subcommand="install": upgraded.append(brew_subcommand),
        )
        # Stop the recursive re-check after upgrade
        call_count = [0]
        original = db_mod.ensure_databricks_cli_version

        def once(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                original()

        monkeypatch.setattr(db_mod, "ensure_databricks_cli_version", once)
        once()
        assert upgraded == ["upgrade"]

    def test_raises_when_version_unparseable(self, tmp_path, monkeypatch):
        env = self._fake_databricks(tmp_path, "completely broken output")
        monkeypatch.setattr("os.environ", env)
        with pytest.raises(RuntimeError, match="Could not parse"):
            ensure_databricks_cli_version()


class TestDatabricksCliVersion:
    def test_none_when_absent(self, monkeypatch):
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: None)
        assert databricks_cli_version() is None

    def test_parses_installed_version(self, monkeypatch):
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: "/usr/bin/databricks")
        monkeypatch.setattr(
            db_mod,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 0, "Databricks CLI v0.299.2", ""),
        )
        assert databricks_cli_version() == (0, 299, 2)

    def test_none_on_unparseable_output(self, monkeypatch):
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: "/usr/bin/databricks")
        monkeypatch.setattr(
            db_mod, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "garbage", "")
        )
        assert databricks_cli_version() is None

    def test_never_raises_on_subprocess_error(self, monkeypatch):
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: "/usr/bin/databricks")

        def boom(*a, **kw):
            raise OSError("nope")

        monkeypatch.setattr(db_mod, "run", boom)
        assert databricks_cli_version() is None


class TestUpgradeDatabricksCli:
    def test_true_on_success(self, monkeypatch):
        monkeypatch.setattr(db_mod, "_run_databricks_cli_installer", lambda **kw: None)
        assert upgrade_databricks_cli() is True

    def test_false_when_installer_fails(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("install failed")

        monkeypatch.setattr(db_mod, "_run_databricks_cli_installer", boom)
        assert upgrade_databricks_cli() is False


class TestRunDatabricksCliInstaller:
    @pytest.mark.parametrize("brew_subcommand", ["install", "upgrade"])
    def test_macos_uses_fully_qualified_tap_formula(self, monkeypatch, brew_subcommand):
        calls = []
        monkeypatch.setattr(db_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(db_mod.shutil, "which", lambda cmd: "/opt/homebrew/bin/brew")
        monkeypatch.setattr(db_mod, "run", lambda cmd, **kw: calls.append(cmd))

        _run_databricks_cli_installer(brew_subcommand=brew_subcommand)

        # The fully-qualified formula forces Homebrew to the Databricks CLI in
        # databricks/tap and fails if absent, rather than falling back to the
        # unrelated `databricks` cask.
        assert calls == [["brew", brew_subcommand, "databricks/tap/databricks"]]


class TestIsUsageTableAccessError:
    """Pin which `ServerOperationError` strings trigger the friendly
    `system.ai_gateway.usage` permissions hint vs. fall through to the
    generic `Usage query failed: ...` arm."""

    @staticmethod
    def _err(msg: str):
        from databricks.sql.exc import ServerOperationError

        return ServerOperationError(msg)

    def test_table_level_select_denial_matches(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have SELECT on Table 'system.ai_gateway.usage'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True

    def test_schema_level_use_schema_denial_matches(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE SCHEMA on Schema 'system.ai_gateway'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True

    def test_unrelated_catalog_denial_falls_through(self):
        msg = (
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE CATALOG on Catalog 'schema1'. "
            "SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is False

    def test_other_error_code_on_same_table_falls_through(self):
        """Different code on the right table must not trip the gate — the
        helper requires INSUFFICIENT_PERMISSIONS specifically so we don't
        mask e.g. missing-table failures with a permissions-shaped hint."""
        msg = (
            "[TABLE_OR_VIEW_NOT_FOUND] The table or view "
            "`system`.`ai_gateway`.`usage` cannot be found. SQLSTATE: 42P01"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is False

    @pytest.mark.parametrize(
        "quoted",
        [
            "`system`.`ai_gateway`.`usage`",
            "[system].[ai_gateway].[usage]",
        ],
    )
    def test_identifier_quoting_variants_all_match(self, quoted):
        msg = (
            f"[INSUFFICIENT_PERMISSIONS] User does not have SELECT on Table "
            f"{quoted}. SQLSTATE: 42501"
        )
        assert db_mod._is_usage_table_access_error(self._err(msg)) is True


class TestRunUsageQuery:
    """Cover the two control-flow arms `_is_usage_table_access_error` gates:
    friendly RuntimeError for matching errors, raw-text fallback for the rest.
    `from exc` chaining is also pinned so `--debug` still surfaces the
    underlying connector error."""

    @staticmethod
    def _patch_connect_to_raise(monkeypatch, exc):
        import databricks.sql as sql_mod

        def fake_connect(*args, **kwargs):
            raise exc

        monkeypatch.setattr(sql_mod, "connect", fake_connect)

    def test_raises_actionable_message_for_table_access_error(self, monkeypatch):
        from databricks.sql.exc import ServerOperationError

        original = ServerOperationError(
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have SELECT on Table 'system.ai_gateway.usage'. "
            "SQLSTATE: 42501"
        )
        self._patch_connect_to_raise(monkeypatch, original)

        with pytest.raises(RuntimeError, match="Ask your workspace admin") as exc_info:
            db_mod.run_usage_query(WS, "/sql/1.0/warehouses/abc", "tok", "SELECT 1")
        assert "system.ai_gateway.usage" in str(exc_info.value)
        # The original ServerOperationError must survive on __cause__ so
        # `--debug` / stack traces still show the underlying connector error.
        assert exc_info.value.__cause__ is original

    def test_falls_through_for_unrelated_permission_error(self, monkeypatch):
        from databricks.sql.exc import ServerOperationError

        original = ServerOperationError(
            "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: "
            "User does not have USE CATALOG on Catalog 'schema1'. SQLSTATE: 42501"
        )
        self._patch_connect_to_raise(monkeypatch, original)

        with pytest.raises(RuntimeError, match="schema1") as exc_info:
            db_mod.run_usage_query(WS, "/sql/1.0/warehouses/abc", "tok", "SELECT 1")
        assert "Ask your workspace admin" not in str(exc_info.value)
        assert str(exc_info.value).startswith("Usage query failed:")


class TestHttpGetJsonTimeout:
    """A socket read timeout raises a bare TimeoutError (an OSError), not a
    URLError. It must be returned as a reason, not propagated — otherwise it
    escapes the best-effort MCP discovery flow and crashes the command."""

    def test_read_timeout_returns_reason_instead_of_raising(self, monkeypatch):
        def raise_timeout(request, timeout=None):
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", raise_timeout)

        payload, reason = db_mod._http_get_json(f"{WS}/api/2.0/anything", "tok")

        assert payload is None
        assert reason is not None
        assert "timed out" in reason

    def test_post_read_timeout_returns_reason_instead_of_raising(self, monkeypatch):
        def raise_timeout(request, timeout=None):
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", raise_timeout)

        payload, reason = db_mod._http_post_json(f"{WS}/api/2.0/anything", "tok", {"k": "v"})

        assert payload is None
        assert reason is not None
        assert "timed out" in reason


class TestInstallAiTools:
    def _capture_run(self, monkeypatch, *, raises=None):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if raises is not None:
                raise raises
            return subprocess.CompletedProcess(args, 0, "Installed 1 skill.", "")

        monkeypatch.setattr(db_mod, "run", fake_run)
        return calls

    def test_no_tokens_skips_entirely(self, monkeypatch):
        calls = self._capture_run(monkeypatch)
        install_ai_tools([])
        assert calls == []

    def test_invokes_aitools_install(self, monkeypatch):
        calls = self._capture_run(monkeypatch)
        install_ai_tools(["claude-code", "codex"])
        assert len(calls) == 1
        cmd = calls[0]
        assert cmd[:3] == ["databricks", "aitools", "install"]
        assert "--agents" in cmd and cmd[cmd.index("--agents") + 1] == "claude-code,codex"
        assert "--scope" in cmd and cmd[cmd.index("--scope") + 1] == "global"
        assert "--profile" not in cmd

    def test_passes_profile_when_set(self, monkeypatch):
        calls = self._capture_run(monkeypatch)
        install_ai_tools(["claude-code"], profile="myprofile")
        cmd = calls[0]
        assert "--profile" in cmd and cmd[cmd.index("--profile") + 1] == "myprofile"

    def test_install_failure_is_non_fatal(self, monkeypatch):
        self._capture_run(monkeypatch, raises=subprocess.CalledProcessError(1, "databricks"))
        # Must not raise — AI Tools are best-effort.
        install_ai_tools(["claude-code"])

    def test_timeout_is_non_fatal(self, monkeypatch):
        self._capture_run(monkeypatch, raises=subprocess.TimeoutExpired("databricks", 300))
        install_ai_tools(["claude-code"])

    def test_timeout_stderr_bytes_decoded_in_warning(self, monkeypatch):
        # TimeoutExpired.stderr is bytes even with text=True; the warning must
        # decode it, not render a `b'...'` repr.
        err = subprocess.TimeoutExpired("databricks", 300)
        err.stderr = b"resolving agents...\ninstall timed out"
        self._capture_run(monkeypatch, raises=err)
        warnings = []
        monkeypatch.setattr(db_mod, "print_warning", warnings.append)
        install_ai_tools(["claude-code"])
        assert len(warnings) == 1
        assert "install timed out" in warnings[0]
        assert "b'" not in warnings[0]

    def test_failure_surfaces_cli_stderr(self, monkeypatch):
        # A modern CLI can still fail (e.g. an agent binary missing from PATH);
        # the warning must show the CLI's real error, not blame the version.
        err = subprocess.CalledProcessError(1, "databricks")
        err.stderr = "resolving agents...\ncopilot: cli-not-on-path: could not resolve copilot"
        self._capture_run(monkeypatch, raises=err)
        warnings = []
        monkeypatch.setattr(db_mod, "print_warning", warnings.append)
        install_ai_tools(["copilot"])
        assert len(warnings) == 1
        assert "copilot: cli-not-on-path: could not resolve copilot" in warnings[0]


class TestClassifyModelFamily:
    """Recovers the bucket a model would land in from discovery, so a managed config's flat list
    can be translated into the per-family state each agent reads."""

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("system.ai.claude-opus-4-8", "opus"),
            ("system.ai.claude-sonnet-5", "sonnet"),
            ("databricks-claude-haiku-4-5", "haiku"),
            ("system.ai.claude-fable-5", "fable"),
            ("system.ai.gpt-5-3-codex", "codex"),
            ("system.ai.gemini-3-flash", "gemini"),
            ("system.ai.kimi-k2-7-code", "oss"),
            ("system.ai.glm-4-6", "oss"),
            ("system.ai.deepseek-v4-pro", "oss"),
            ("something-unrecognized", None),
        ],
    )
    def test_buckets_by_family(self, model_id, expected):
        assert classify_model_family(model_id) == expected


class TestModelServicesCache:
    """A successful listing is memoized per workspace: several callers want different views of the
    same paginated walk (bucketed families vs the raw Claude ids), so one `ucode setup` run would
    otherwise page the whole catalog twice."""

    @staticmethod
    def _counting_page(calls: dict):
        def page(url, token):
            calls["n"] = calls.get("n", 0) + 1
            return {
                "model_services": [
                    {"name": "model-services/system.ai.claude-opus-5"},
                    {"name": "model-services/system.ai.claude-opus-4-8"},
                ]
            }, None

        return page

    def test_repeat_listings_hit_the_api_once(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_get_model_services_page", self._counting_page(calls))
        first, _ = db_mod.list_model_services(WS, "tok")
        second, _ = db_mod.list_model_services(WS, "tok")
        assert first == second
        assert calls["n"] == 1

    def test_the_two_discovery_helpers_share_one_walk(self, monkeypatch):
        # The duplicate spinner in `ucode setup`: `discover_model_services` and
        # `discover_claude_models_unbucketed` both page the same endpoint.
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_get_model_services_page", self._counting_page(calls))
        claude, _codex, _gemini, _oss, _reason = db_mod.discover_model_services(WS, "tok")
        unbucketed, _ = db_mod.discover_claude_models_unbucketed(WS, "tok")
        assert calls["n"] == 1
        # Both views still come back intact: newest-per-family (pinned to opus-4-8
        # for smart-routing compatibility by _prefer_opus_4_8), and the full list.
        assert claude["opus"] == "system.ai.claude-opus-4-8"
        assert unbucketed == ["system.ai.claude-opus-4-8", "system.ai.claude-opus-5"]

    def test_use_cache_false_forces_a_fresh_walk(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_get_model_services_page", self._counting_page(calls))
        db_mod.list_model_services(WS, "tok")
        db_mod.list_model_services(WS, "tok", use_cache=False)
        assert calls["n"] == 2

    def test_each_workspace_is_cached_separately(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_get_model_services_page", self._counting_page(calls))
        db_mod.list_model_services(WS, "tok")
        db_mod.list_model_services("https://other.databricks.com", "tok")
        assert calls["n"] == 2

    def test_failures_are_not_cached(self, monkeypatch):
        # A transient error must not poison the rest of the process into believing there are no
        # models on the workspace.
        calls: dict = {}

        def failing(url, token):
            calls["n"] = calls.get("n", 0) + 1
            return None, "HTTP 500 Server Error"

        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_get_model_services_page", failing)
        ids, reason = db_mod.list_model_services(WS, "tok")
        assert ids == [] and reason is not None

        monkeypatch.setattr(db_mod, "_get_model_services_page", self._counting_page(calls))
        ids, reason = db_mod.list_model_services(WS, "tok")
        assert reason is None
        assert ids


class TestModelProviderServicesCache:
    """The MPS listing is workspace-wide and filtered per agent afterwards, so one call serves every
    agent — `ucode setup` used to re-list it once per MPS-capable agent."""

    @staticmethod
    def _counting_listing(calls: dict):
        def get_json(url, token, timeout=10):
            calls["n"] = calls.get("n", 0) + 1
            return {
                "model_provider_services": [
                    {
                        "name": "model-provider-services/main.j.ant",
                        "config": {
                            "provider_type": "ANTHROPIC",
                            "targets": [{"model": "claude-opus-5"}],
                        },
                    },
                    {
                        "name": "model-provider-services/main.j.oai",
                        "config": {"provider_type": "OPENAI", "targets": [{"model": "gpt-5"}]},
                    },
                ]
            }, None

        return get_json

    def test_one_call_serves_every_agent(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_http_get_json", self._counting_listing(calls))
        claude, _ = db_mod.list_tool_provider_services("claude", WS, "tok")
        codex, _ = db_mod.list_tool_provider_services("codex", WS, "tok")
        assert calls["n"] == 1
        # Each agent still gets only the services matching its API dialect.
        assert claude == ["main.j.ant"]
        assert codex == ["main.j.oai"]

    def test_use_cache_false_forces_a_fresh_call(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_http_get_json", self._counting_listing(calls))
        db_mod.list_model_provider_services(WS, "tok")
        db_mod.list_model_provider_services(WS, "tok", use_cache=False)
        assert calls["n"] == 2

    def test_each_workspace_is_cached_separately(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_http_get_json", self._counting_listing(calls))
        db_mod.list_model_provider_services(WS, "tok")
        db_mod.list_model_provider_services("https://other.databricks.com", "tok")
        assert calls["n"] == 2

    def test_failures_are_not_cached(self, monkeypatch):
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_http_get_json", lambda *a, **k: (None, "HTTP 500"))
        services, reason = db_mod.list_model_provider_services(WS, "tok")
        assert services == [] and reason is not None
        monkeypatch.setattr(db_mod, "_http_get_json", self._counting_listing(calls))
        services, reason = db_mod.list_model_provider_services(WS, "tok")
        assert reason is None and services

    def test_the_first_caller_cannot_corrupt_the_cache(self, monkeypatch):
        # The caller that populates the cache gets the same list that was stored, so mutating it
        # would poison every later reader.
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_http_get_json", self._counting_listing(calls))
        first, _ = db_mod.list_model_provider_services(WS, "tok")
        first[0]["name"] = "clobbered"
        first.pop()
        second, _ = db_mod.list_model_provider_services(WS, "tok")
        assert [s["name"] for s in second] == ["main.j.ant", "main.j.oai"]

    def test_a_later_caller_cannot_corrupt_the_cache(self, monkeypatch):
        # And so does every cache *hit* — the wizard filters this list per agent, so the second
        # agent's read must not see what the first one did to it.
        calls: dict = {}
        db_mod.clear_model_services_cache()
        monkeypatch.setattr(db_mod, "_http_get_json", self._counting_listing(calls))
        db_mod.list_model_provider_services(WS, "tok")  # populate
        hit, _ = db_mod.list_model_provider_services(WS, "tok")
        hit[0]["name"] = "clobbered"
        hit.pop()
        again, _ = db_mod.list_model_provider_services(WS, "tok")
        assert [s["name"] for s in again] == ["main.j.ant", "main.j.oai"]


class TestIsWorkspaceAdmin:
    """Admin detection reuses the SCIM `Me` payload, which carries group membership."""

    @staticmethod
    def _stub(monkeypatch, payload):
        monkeypatch.setattr(db_mod, "_scim_me", lambda ws, tok: payload)

    def test_true_when_in_the_admins_group(self, monkeypatch):
        self._stub(monkeypatch, {"groups": [{"display": "users"}, {"display": "admins"}]})
        assert db_mod.is_workspace_admin("https://w", "tok") is True

    def test_false_without_the_admins_group(self, monkeypatch):
        self._stub(monkeypatch, {"groups": [{"display": "users"}]})
        assert db_mod.is_workspace_admin("https://w", "tok") is False

    def test_none_when_the_check_could_not_be_made(self, monkeypatch):
        # An unreachable SCIM is "unknown", not "not an admin" — the caller must not send a real
        # admin down the non-admin dead end.
        self._stub(monkeypatch, None)
        assert db_mod.is_workspace_admin("https://w", "tok") is None

    @pytest.mark.parametrize("payload", [{}, {"groups": "not-a-list"}])
    def test_false_when_the_payload_names_no_groups(self, monkeypatch, payload):
        # A well-formed `Me` for a user in no groups omits `groups` entirely.
        self._stub(monkeypatch, payload)
        assert db_mod.is_workspace_admin("https://w", "tok") is False


class TestCodingAgentConfigUrls:
    def test_collection_url(self):
        assert db_mod._coding_agent_config_url(WS) == f"{WS}/api/ai-gateway/v2/coding-agent-configs"

    def test_resource_url_appends_the_server_assigned_name(self):
        # The API templates Get/Update/Delete on `{name=coding-agent-configs/*}`, so the resource
        # name already carries the collection segment and must not be duplicated.
        url = db_mod._coding_agent_config_url(WS, "coding-agent-configs/abc123")
        assert url == f"{WS}/api/ai-gateway/v2/coding-agent-configs/abc123"

    def test_stray_slashes_are_tolerated(self):
        url = db_mod._coding_agent_config_url(WS, "/coding-agent-configs/abc123/")
        assert url == f"{WS}/api/ai-gateway/v2/coding-agent-configs/abc123"


class TestHttpDelete:
    """A successful delete returns `google.protobuf.Empty`, so an empty body is success."""

    @staticmethod
    def _empty_response(body: str = ""):
        from unittest.mock import MagicMock

        response = MagicMock()
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)
        response.read.return_value = body.encode("utf-8")
        response.status = 200
        return response

    def test_empty_body_is_success_not_a_decode_error(self, monkeypatch):
        # Without `allow_empty_body` this would fail with "response was not valid JSON".
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda request, timeout=None: self._empty_response()
        )
        payload, reason = db_mod._http_delete(f"{WS}/api/anything", "tok")
        assert reason is None
        assert payload is None

    def test_empty_json_object_is_also_success(self, monkeypatch):
        monkeypatch.setattr(
            db_mod.urllib_request,
            "urlopen",
            lambda request, timeout=None: self._empty_response("{}"),
        )
        payload, reason = db_mod._http_delete(f"{WS}/api/anything", "tok")
        assert reason is None
        assert payload == {}

    def test_uses_the_delete_verb_and_sends_no_body(self, monkeypatch):
        seen = {}

        def capture(request, timeout=None):
            seen["method"] = request.get_method()
            seen["data"] = request.data
            return self._empty_response()

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", capture)
        db_mod._http_delete(f"{WS}/api/anything", "tok")
        assert seen["method"] == "DELETE"
        assert seen["data"] is None

    def test_http_error_surfaces_the_body(self, monkeypatch):
        import io
        from unittest.mock import MagicMock
        from urllib.error import HTTPError

        body = '{"error_code":"PERMISSION_DENIED","message":"admin required"}'

        def raise_http_error(request, timeout=None):
            raise HTTPError(
                url="", code=403, msg="Forbidden", hdrs=MagicMock(), fp=io.BytesIO(body.encode())
            )

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", raise_http_error)
        _, reason = db_mod._http_delete(f"{WS}/api/anything", "tok")
        assert reason is not None
        assert "403" in reason
        assert "PERMISSION_DENIED" in reason


class TestHttpPatchJson:
    def test_uses_the_patch_verb_and_sends_the_body(self, monkeypatch):
        from unittest.mock import MagicMock

        seen = {}

        def capture(request, timeout=None):
            seen["method"] = request.get_method()
            seen["data"] = request.data
            seen["content_type"] = request.get_header("Content-type")
            response = MagicMock()
            response.__enter__ = lambda s: s
            response.__exit__ = MagicMock(return_value=False)
            response.read.return_value = b'{"name":"coding-agent-configs/x"}'
            response.status = 200
            return response

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", capture)
        payload, reason = db_mod._http_patch_json(f"{WS}/api/anything", "tok", {"k": "v"})
        assert reason is None
        assert payload == {"name": "coding-agent-configs/x"}
        assert seen["method"] == "PATCH"
        assert json.loads(seen["data"]) == {"k": "v"}
        assert seen["content_type"] == "application/json"


class TestCodingAgentConfigCrudClients:
    CONFIG = {"default_agent": "CODING_AGENT_CLAUDE_CODE"}

    def test_create_posts_the_config_to_the_collection(self, monkeypatch):
        seen = {}

        def fake_post(url, token, payload, *, timeout=10):
            seen.update(url=url, payload=payload)
            return {"name": "coding-agent-configs/new"}, None

        monkeypatch.setattr(db_mod, "_http_post_json", fake_post)
        config, reason = db_mod.create_coding_agent_config(WS, "tok", self.CONFIG)
        assert reason is None
        assert config == {"name": "coding-agent-configs/new"}
        assert seen["url"] == f"{WS}/api/ai-gateway/v2/coding-agent-configs"
        assert seen["payload"] == self.CONFIG

    def test_create_surfaces_the_failure_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda *a, **k: (None, 'HTTP 400: {"error_code":"ALREADY_EXISTS"}'),
        )
        config, reason = db_mod.create_coding_agent_config(WS, "tok", self.CONFIG)
        assert config is None
        assert "ALREADY_EXISTS" in reason

    def test_update_patches_the_resource_with_a_mask(self, monkeypatch):
        seen = {}

        def fake_patch(url, token, payload, *, timeout=10):
            seen.update(url=url, payload=payload)
            return {"name": "coding-agent-configs/abc"}, None

        monkeypatch.setattr(db_mod, "_http_patch_json", fake_patch)
        config, reason = db_mod.update_coding_agent_config(
            WS, "tok", "coding-agent-configs/abc", self.CONFIG
        )
        assert reason is None
        assert config == {"name": "coding-agent-configs/abc"}
        # The mask rides in the query string: the RPC binds `body: "coding_agent_config"`, so the
        # config is the whole body and a mask nested inside it is read as an unknown config field —
        # the server then reports the mask as missing. A FieldMask's JSON form is one
        # comma-separated string, not a `{"paths": [...]}` object.
        url, _, query = seen["url"].partition("?")
        assert url == f"{WS}/api/ai-gateway/v2/coding-agent-configs/abc"
        mask = parse_qs(query)["update_mask"][0].split(",")
        assert mask == list(db_mod.MANAGED_CONFIG_UPDATE_MASK_PATHS)
        assert "update_mask" not in seen["payload"]
        # `name` still goes in the body: the API's path template reads it from the config.
        assert seen["payload"]["name"] == "coding-agent-configs/abc"
        assert seen["payload"]["default_agent"] == "CODING_AGENT_CLAUDE_CODE"

    def test_update_mask_never_names_a_field_the_server_rejects(self):
        # The server's mutable set is the upper bound; `budget_id` is in it but deprecated and
        # rejected on write, so ucode must not name it. `default_options`/`tiers` are the legacy
        # model-only shape ucode never authors.
        assert "budget_id" not in db_mod.MANAGED_CONFIG_UPDATE_MASK_PATHS
        assert "default_options" not in db_mod.MANAGED_CONFIG_UPDATE_MASK_PATHS
        assert "tiers" not in db_mod.MANAGED_CONFIG_UPDATE_MASK_PATHS
        assert "spec_version" not in db_mod.MANAGED_CONFIG_UPDATE_MASK_PATHS

    def test_update_mask_covers_every_field_the_manifest_can_set(self):
        # A path ucode omits is a field a re-run silently cannot clear, since the server merges per
        # path. Derive the expectation from the serializer rather than restating it, so adding a
        # manifest field fails here instead of shipping a mask that can't clear it.
        from ucode.managed_setup import serialize_managed_config

        emitted = set(
            serialize_managed_config(
                {
                    "display_name": "org config",
                    "default_agent": "claude",
                    "enabled_agents": {
                        "claude": {"model_config": {"default_model": "system.ai.claude-opus-5"}}
                    },
                    "mcp_servers": [{"name": "databricks-sql", "type": "sql"}],
                    "skills": {"names": ["main.default"]},
                    "tracing_table": "main.default.traces",
                    "budget_policy": {
                        "budget_id": "11111111-1111-1111-1111-111111111111",
                        "tiers": [],
                    },
                }
            )
        )
        assert set(db_mod.MANAGED_CONFIG_UPDATE_MASK_PATHS) == emitted

    def test_delete_returns_only_a_reason(self, monkeypatch):
        seen = {}

        def fake_delete(url, token, *, timeout=10):
            seen["url"] = url
            return None, None

        monkeypatch.setattr(db_mod, "_http_delete", fake_delete)
        assert db_mod.delete_coding_agent_config(WS, "tok", "coding-agent-configs/abc") is None
        assert seen["url"] == f"{WS}/api/ai-gateway/v2/coding-agent-configs/abc"

    def test_delete_surfaces_the_failure_reason(self, monkeypatch):
        monkeypatch.setattr(db_mod, "_http_delete", lambda *a, **k: (None, "HTTP 404 Not Found"))
        reason = db_mod.delete_coding_agent_config(WS, "tok", "coding-agent-configs/abc")
        assert reason == "HTTP 404 Not Found"


class TestResolveCurrentBudgetSpend:
    def test_parses_spend_and_threshold(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: (
                {"current_spend": "12.34", "effective_threshold": "100"},
                None,
            ),
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend == (Decimal("12.34"), Decimal("100"))
        assert reason is None

    def test_posts_to_recommend_model_with_no_available_models(self, monkeypatch):
        captured = {}

        def fake_post(url, token, payload, timeout=10):
            captured["url"] = url
            captured["payload"] = payload
            return {"current_spend": "1", "effective_threshold": "2"}, None

        monkeypatch.setattr(db_mod, "_http_post_json", fake_post)
        resolve_current_budget_spend("https://ws.example.com", "token")
        assert captured["url"] == (f"https://ws.example.com{CODING_AGENT_RECOMMEND_MODEL_PATH}")
        # Empty list applies no availability filter; we want the spend only.
        assert captured["payload"] == {"available_models": []}

    def test_ignores_recommended_models(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: (
                {
                    "recommended_models": ["system.ai.claude-sonnet-4-5"],
                    "current_spend": "12.34",
                    "effective_threshold": "100",
                },
                None,
            ),
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend == (Decimal("12.34"), Decimal("100"))
        assert reason is None

    def test_recommendation_without_spend_is_no_spend(self, monkeypatch):
        # A config with no matching budget still recommends models, but both
        # spend fields come back unset.
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: (
                {"recommended_models": ["system.ai.claude-sonnet-4-5"]},
                None,
            ),
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend is None
        assert "no coding-agent budget spend" in reason

    def test_feature_disabled_returns_reason(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: (
                None,
                "HTTP 400 Bad Request: FEATURE_DISABLED",
            ),
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend is None
        assert "FEATURE_DISABLED" in reason

    def test_unset_fields_treated_as_no_spend(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_post_json", lambda url, token, payload, timeout=10: ({}, None)
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend is None
        assert "no coding-agent budget spend" in reason

    def test_spend_without_threshold_is_no_spend(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: ({"current_spend": "12.34"}, None),
        )
        spend, _ = resolve_current_budget_spend("https://ws", "token")
        assert spend is None

    def test_threshold_without_spend_is_zero_spend(self, monkeypatch):
        # A per-user threshold with no spend yet this period is $0 spent, not "no budget" — the
        # developer should still see their budget rather than a blank.
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: ({"effective_threshold": "100"}, None),
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend == (Decimal(0), Decimal("100"))
        assert reason is None

    def test_malformed_decimal_is_no_spend(self, monkeypatch):
        monkeypatch.setattr(
            db_mod,
            "_http_post_json",
            lambda url, token, payload, timeout=10: (
                {"current_spend": "not-a-number", "effective_threshold": "100"},
                None,
            ),
        )
        spend, _ = resolve_current_budget_spend("https://ws", "token")
        assert spend is None

    def test_non_object_payload_is_no_spend(self, monkeypatch):
        monkeypatch.setattr(
            db_mod, "_http_post_json", lambda url, token, payload, timeout=10: ([], None)
        )
        spend, reason = resolve_current_budget_spend("https://ws", "token")
        assert spend is None
        assert "not a JSON object" in reason


class TestListWorkspaceBudgets:
    PER_USER = "ALERT_CONFIGURATION_SCOPE_TYPE_PER_USER"
    SHARED = "ALERT_CONFIGURATION_SCOPE_TYPE_SHARED"

    def _stub(self, monkeypatch, payload):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, None)
        )

    BLOCK = "BLOCK_USAGE"
    EMAIL = "EMAIL_NOTIFICATION"

    def test_flags_per_user_block_presence(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "workspace_ai_gateway_budgets": [
                    {
                        "budget_configuration_id": "blocks",
                        "display_name": "per-user block",
                        "alert_configurations": [
                            {"scope_type": self.SHARED},
                            {
                                "scope_type": self.PER_USER,
                                "action_configurations": [{"action_type": self.BLOCK}],
                            },
                        ],
                    },
                    {
                        "budget_configuration_id": "email_only",
                        "display_name": "per-user email only",
                        "alert_configurations": [
                            {
                                "scope_type": self.PER_USER,
                                "action_configurations": [
                                    {"action_type": self.EMAIL, "target": "a@b.com"}
                                ],
                            }
                        ],
                    },
                    {
                        "budget_configuration_id": "shared_block",
                        "display_name": "shared block only",
                        "alert_configurations": [
                            {
                                "scope_type": self.SHARED,
                                "action_configurations": [{"action_type": self.BLOCK}],
                            }
                        ],
                    },
                ]
            },
        )
        budgets, reason = list_workspace_budgets("https://ws", "token")
        assert reason is None
        by_id = {b["id"]: b for b in budgets}
        # Only a per-user threshold that also carries a BLOCK_USAGE action enforces spend routing.
        assert by_id["blocks"]["has_per_user_block"] is True
        assert by_id["email_only"]["has_per_user_block"] is False
        assert by_id["shared_block"]["has_per_user_block"] is False

    def test_missing_alert_configs_is_not_per_user_block(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "workspace_ai_gateway_budgets": [
                    {"budget_configuration_id": "b", "display_name": "x"}
                ]
            },
        )
        budgets, _ = list_workspace_budgets("https://ws", "token")
        assert budgets[0]["has_per_user_block"] is False

    def test_extracts_per_user_block_threshold(self, monkeypatch):
        # The per-user hard block's `quantity_threshold` is the monthly dollar cap; it's read off the
        # same alert `has_per_user_block` gates on so the tier prompt can show it. A per-user alert
        # without a block action (email only) contributes no threshold.
        self._stub(
            monkeypatch,
            {
                "workspace_ai_gateway_budgets": [
                    {
                        "budget_configuration_id": "capped",
                        "display_name": "capped",
                        "alert_configurations": [
                            {
                                "scope_type": self.PER_USER,
                                "quantity_threshold": "500.00",
                                "action_configurations": [{"action_type": self.BLOCK}],
                            }
                        ],
                    },
                    {
                        "budget_configuration_id": "email_only",
                        "display_name": "email only",
                        "alert_configurations": [
                            {
                                "scope_type": self.PER_USER,
                                "quantity_threshold": "500.00",
                                "action_configurations": [{"action_type": self.EMAIL}],
                            }
                        ],
                    },
                ]
            },
        )
        budgets, _ = list_workspace_budgets("https://ws", "token")
        by_id = {b["id"]: b for b in budgets}
        assert by_id["capped"]["per_user_threshold"] == Decimal("500.00")
        assert by_id["email_only"]["per_user_threshold"] is None

    def test_missing_threshold_is_none_but_block_still_flagged(self, monkeypatch):
        # A hard block with no (or unparseable) `quantity_threshold` still enforces routing, so the
        # budget stays offerable — the tier prompt just omits the dollar hint.
        self._stub(
            monkeypatch,
            {
                "workspace_ai_gateway_budgets": [
                    {
                        "budget_configuration_id": "b",
                        "display_name": "x",
                        "alert_configurations": [
                            {
                                "scope_type": self.PER_USER,
                                "action_configurations": [{"action_type": self.BLOCK}],
                            }
                        ],
                    }
                ]
            },
        )
        budgets, _ = list_workspace_budgets("https://ws", "token")
        assert budgets[0]["has_per_user_block"] is True
        assert budgets[0]["per_user_threshold"] is None


class TestDiscoverSqlWarehouses:
    def _payload(self, *entries: dict) -> dict:
        return {"warehouses": list(entries)}

    def test_explicit_id_skips_discovery(self, monkeypatch):
        def fail(*a, **k):
            raise AssertionError("discovery should not be called")

        monkeypatch.setattr(db_mod.urllib_request, "urlopen", fail)
        assert discover_sql_warehouses(WS, "token", warehouse_id="abc") == [
            db_mod.SqlWarehouse("/sql/1.0/warehouses/abc", "abc", "REQUESTED")
        ]

    def test_running_sorted_before_stopped(self, monkeypatch):
        payload = self._payload(
            {"id": "s1", "name": "stopped", "state": "STOPPED"},
            {"id": "r1", "name": "running", "state": "RUNNING"},
        )
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        result = discover_sql_warehouses(WS, "token")
        assert [w.label for w in result] == ["running", "stopped"]

    def test_returns_all_candidates(self, monkeypatch):
        payload = self._payload(
            {"id": "a", "name": "A", "state": "RUNNING"},
            {"id": "b", "name": "B", "state": "RUNNING"},
        )
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        assert len(discover_sql_warehouses(WS, "token")) == 2

    def test_skips_entries_without_id(self, monkeypatch):
        payload = self._payload(
            {"name": "no id", "state": "RUNNING"},
            {"id": "b", "name": "B", "state": "RUNNING"},
        )
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        assert [w.label for w in discover_sql_warehouses(WS, "token")] == ["B"]

    def test_falls_back_to_id_as_label(self, monkeypatch):
        payload = self._payload({"id": "abc", "state": "RUNNING"})
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        assert discover_sql_warehouses(WS, "token")[0].label == "abc"

    def test_empty_list_raises_with_flag_hint(self, monkeypatch):
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda *a, **k: _FakeResponse({"warehouses": []})
        )
        with pytest.raises(RuntimeError, match="--warehouse-id"):
            discover_sql_warehouses(WS, "token")

    def test_only_unusable_entries_raises(self, monkeypatch):
        payload = self._payload({"name": "no id", "state": "RUNNING"})
        monkeypatch.setattr(
            db_mod.urllib_request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        with pytest.raises(RuntimeError, match="No usable SQL warehouse"):
            discover_sql_warehouses(WS, "token")


class TestAllUsersCanUseSchema:
    def _stub(self, monkeypatch, payload, reason=None):
        monkeypatch.setattr(
            db_mod, "_http_get_json", lambda url, token, timeout=30: (payload, reason)
        )

    def test_true_when_account_users_have_use_schema(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "privilege_assignments": [
                    {"principal": "account users", "privileges": [{"privilege": "USE_SCHEMA"}]}
                ]
            },
        )
        assert all_users_can_use_schema("https://ws", "tok", "main.default") is True

    def test_true_when_account_users_have_all_privileges(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "privilege_assignments": [
                    {"principal": "account users", "privileges": [{"privilege": "ALL_PRIVILEGES"}]}
                ]
            },
        )
        assert all_users_can_use_schema("https://ws", "tok", "main.default") is True

    def test_false_when_no_use_schema(self, monkeypatch):
        # An empty assignment list is what the API returns when the group has no grant on the schema.
        self._stub(monkeypatch, {})
        assert all_users_can_use_schema("https://ws", "tok", "main.tien_le") is False

    def test_false_when_only_unrelated_privileges(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                "privilege_assignments": [
                    {"principal": "account users", "privileges": [{"privilege": "MANAGE"}]}
                ]
            },
        )
        assert all_users_can_use_schema("https://ws", "tok", "main.default") is False

    def test_none_when_the_call_fails(self, monkeypatch):
        self._stub(monkeypatch, None, reason="HTTP 403 Forbidden")
        assert all_users_can_use_schema("https://ws", "tok", "main.default") is None

    def test_none_on_unexpected_shape(self, monkeypatch):
        self._stub(monkeypatch, [])
        assert all_users_can_use_schema("https://ws", "tok", "main.default") is None

    def test_requests_the_account_users_principal(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            db_mod,
            "_http_get_json",
            lambda url, token, timeout=30: (seen.setdefault("url", url), ({}, None))[1],
        )
        all_users_can_use_schema("https://ws", "tok", "main.tien_le")
        assert "effective-permissions/schema/main.tien_le" in seen["url"]
        assert "principal=account%20users" in seen["url"]
