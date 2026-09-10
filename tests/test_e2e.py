"""End-to-end integration tests that require a live Databricks workspace.

Run with:
    UCODE_TEST_WORKSPACE=https://your-workspace.databricks.com uv run pytest tests/test_e2e.py -v

All tests in this file are skipped automatically when the env var is not set.
The agent-launch tests are also skipped per-agent/model when the binary is not
installed or no models are available.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest

from ucode.databricks import (
    build_shared_base_urls,
    build_tool_base_url,
    discover_model_services,
    discover_sql_warehouses,
    fetch_ai_gateway_claude_models,
    fetch_codex_models,
    fetch_gemini_models,
    has_valid_databricks_auth,
    is_model_provider_feature_unavailable,
    list_model_provider_services,
    list_tool_provider_services,
    probe_unity_gateway_capabilities,
    service_usable_for_tool,
    workspace_hostname,
)
from ucode.ui import normalize_workspace_url

# ---------------------------------------------------------------------------
# CI provider-launch pinning
# ---------------------------------------------------------------------------
# The claude and codex provider-launch tests each route through one fixed,
# non-relayed MPS that exposes only its cheapest model, so the inference is
# deterministic and ~a cent per run rather than depending on whichever service
# happened to list first. Hardcoded on purpose.
CI_ANTHROPIC_MPS = "main.ucode.ci_e2e_anthropic_nonrelay_mps"  # api-key Anthropic (for claude)
CI_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
CI_OPENAI_MPS = "main.ucode.ci_openai_mps"  # api-key OpenAI (for codex)
CI_OPENAI_MODEL = "gpt-5-nano"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws() -> str:
    raw = os.environ.get("UCODE_TEST_WORKSPACE", "").strip().rstrip("/")
    return normalize_workspace_url(raw) if raw else ""


def _skip_if_no_workspace():
    if not _ws():
        pytest.skip("Set UCODE_TEST_WORKSPACE=https://... to run E2E tests")


def _run_agent(
    cmd: list[str], env: dict | None = None, timeout: int = 60
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def _codex_home_outside_tmp() -> Path:
    """Create a fresh CODEX_HOME under the user's home dir, registered for cleanup at exit.

    pytest's ``tmp_path`` lives under ``/tmp``; codex (>=0.134) refuses to create its helper
    binaries when ``CODEX_HOME`` is under a temporary dir, so launching codex from ``tmp_path``
    fails before doing anything. Rooting CODEX_HOME under ``$HOME`` sidesteps that guard."""
    home = Path(tempfile.mkdtemp(prefix=".ucode-e2e-codex-", dir=Path.home()))
    atexit.register(shutil.rmtree, home, ignore_errors=True)
    return home


def _run_gemini_gateway_smoke(workspace: str, model: str, token: str) -> str:
    """Call the Gemini gateway directly with a text-only prompt.

    This keeps auth recovery coverage focused on the recovered Databricks token
    instead of Gemini CLI's separate tool-calling request shape.
    """
    url = f"{build_tool_base_url('gemini', workspace)}/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "say hi in 5 words or less"}]},
        ],
    }
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise AssertionError(f"Gemini gateway smoke failed: HTTP {exc.code}: {body[:500]}") from exc

    data = json.loads(body)
    return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")


def _launchable_model_items(models: dict) -> list[tuple[str, str]]:
    return [(family, model_id) for family, model_id in models.items() if model_id]


# ---------------------------------------------------------------------------
# Databricks auth / token
# ---------------------------------------------------------------------------


class TestDatabricksAuth:
    def test_has_valid_auth(self, e2e_workspace):
        assert has_valid_databricks_auth(e2e_workspace), (
            "No valid Databricks auth found. Run `databricks auth login` first."
        )

    def test_get_token_returns_non_empty_string(self, e2e_token):
        assert isinstance(e2e_token, str) and len(e2e_token) > 10


# ---------------------------------------------------------------------------
# AI Gateway probe
# ---------------------------------------------------------------------------


class TestAiGateway:
    def test_probe_unity_gateway_capabilities_does_not_raise(self, e2e_workspace, e2e_token):
        probe_unity_gateway_capabilities(e2e_workspace, e2e_token)

    def test_workspace_hostname_resolves(self, e2e_workspace):
        hostname = workspace_hostname(e2e_workspace)
        assert "." in hostname


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


class TestModelDiscovery:
    def test_fetch_claude_models_returns_dict(self, e2e_workspace, e2e_token):
        models = fetch_ai_gateway_claude_models(e2e_workspace, e2e_token)
        assert isinstance(models, dict)
        assert models, "No Claude models found — is the Anthropic route enabled on this workspace?"

    def test_fetch_gemini_models_returns_list(self, e2e_workspace, e2e_token):
        models = fetch_gemini_models(e2e_workspace, e2e_token)
        assert isinstance(models, list)

    def test_fetch_codex_models_returns_list(self, e2e_workspace, e2e_token):
        models = fetch_codex_models(e2e_workspace, e2e_token)
        assert isinstance(models, list)


# ---------------------------------------------------------------------------
# Model Provider Services discovery
# ---------------------------------------------------------------------------


class TestModelProviderServicesDiscovery:
    def test_list_returns_services_or_skips_when_feature_off(self, e2e_workspace, e2e_token):
        services, reason = list_model_provider_services(e2e_workspace, e2e_token)
        if is_model_provider_feature_unavailable(reason):
            pytest.skip("Model Provider Service feature not enabled on this workspace")
        assert reason is None, f"listing failed: {reason}"
        assert isinstance(services, list)
        for svc in services:
            assert set(svc) >= {"name", "provider_type", "targets", "allow_all_targets"}
            # Names are stripped of the `model-provider-services/` API prefix.
            assert svc["name"] and "/" not in svc["name"]

    def test_tool_filter_matches_provider_type(self, e2e_workspace, e2e_token):
        services, reason = list_model_provider_services(e2e_workspace, e2e_token)
        if is_model_provider_feature_unavailable(reason):
            pytest.skip("Model Provider Service feature not enabled on this workspace")
        assert reason is None
        claude_names, _ = list_tool_provider_services("claude", e2e_workspace, e2e_token)
        codex_names, _ = list_tool_provider_services("codex", e2e_workspace, e2e_token)
        # claude routes through Anthropic and Bedrock services (Bedrock only when
        # it exposes Claude models); codex through OpenAI.
        assert set(claude_names) == {
            s["name"] for s in services if service_usable_for_tool("claude", s)
        }
        assert set(codex_names) == {
            s["name"] for s in services if service_usable_for_tool("codex", s)
        }


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


class TestUrlBuilders:
    def test_codex_url_contains_workspace(self, e2e_workspace):
        assert e2e_workspace in build_tool_base_url("codex", e2e_workspace)

    def test_claude_url_contains_workspace(self, e2e_workspace):
        assert e2e_workspace in build_tool_base_url("claude", e2e_workspace)

    def test_shared_base_urls_all_tools(self, e2e_workspace):
        urls = build_shared_base_urls(e2e_workspace)
        for tool in ("codex", "claude", "gemini", "opencode", "copilot", "pi"):
            assert tool in urls


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    def test_configure_shared_state_and_reload(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace
    ):
        import ucode.config_io as config_io_mod
        import ucode.state as state_mod
        from ucode.state import load_state, save_state

        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)

        save_state(e2e_state)
        loaded = load_state()
        assert loaded["workspace"] == e2e_workspace
        assert loaded["claude_models"] == e2e_state["claude_models"]
        assert loaded["base_urls"]["codex"] == f"{e2e_workspace}/ai-gateway/codex/v1"


# ---------------------------------------------------------------------------
# SQL warehouse discovery
# ---------------------------------------------------------------------------


class TestSqlWarehouseDiscovery:
    def test_discovers_http_path(self, e2e_workspace, e2e_token):
        try:
            candidates = discover_sql_warehouses(e2e_workspace, e2e_token)
        except RuntimeError as exc:
            pytest.skip(f"No SQL warehouse available: {exc}")
        assert candidates
        assert all(w.http_path.startswith("/sql/1.0/warehouses/") for w in candidates)


# ---------------------------------------------------------------------------
# Configure flow with user-selected subset
# ---------------------------------------------------------------------------
#
# Verifies that when the user picks a subset in the multi-select prompt,
# only those tools get configured and previously-configured tools are
# preserved in state["available_tools"].
# ---------------------------------------------------------------------------


class TestConfigureSubset:
    def _redirect_config_paths(self, monkeypatch, tmp_path):
        """Redirect every agent's config path into tmp_path so the test
        doesn't touch the developer's real ~/.codex, ~/.claude, etc."""
        import ucode.config_io as config_io_mod
        from ucode.agents import claude, codex, copilot, gemini, opencode, pi

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)

        codex_dir = tmp_path / "codex_home" / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", codex_dir / "ucode.config.toml")
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "codex.backup.toml")

        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", tmp_path / "claude-settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "claude.backup.json")

        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / "gemini-ucode.env")
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "gemini.backup")

        monkeypatch.setattr(opencode, "OPENCODE_CONFIG_PATH", tmp_path / "opencode.json")
        monkeypatch.setattr(opencode, "OPENCODE_BACKUP_PATH", tmp_path / "opencode.backup.json")

        monkeypatch.setattr(copilot, "COPILOT_ENV_PATH", tmp_path / ".copilot-env")
        monkeypatch.setattr(copilot, "COPILOT_BACKUP_PATH", tmp_path / "copilot.backup")

        monkeypatch.setattr(pi, "PI_CONFIG_PATH", tmp_path / "pi-models.json")
        monkeypatch.setattr(pi, "PI_BACKUP_PATH", tmp_path / "pi-models.backup.json")

        return codex_dir / "ucode.config.toml"

    def test_only_picks_codex_writes_only_codex_config(self, tmp_path, monkeypatch, e2e_workspace):
        """User selects only codex → only codex's config file is written and
        state['available_tools'] contains exactly ['codex']."""
        import ucode.cli as cli_mod
        import ucode.state as state_mod
        from ucode.state import load_state

        codex_path = self._redirect_config_paths(monkeypatch, tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        # Don't actually run `databricks auth login`; the developer running
        # this suite is already authenticated.
        monkeypatch.setattr("ucode.cli.run_databricks_login", lambda ws, profile=None: None)
        # Skip the workspace prompt and the multi-select picker.
        monkeypatch.setattr(
            cli_mod, "_prompt_for_configuration", lambda tool=None: (e2e_workspace, None)
        )
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["codex"])
        # Skip binary install + post-config validation; we're testing the
        # selection plumbing, not the agent binaries themselves.
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda tool, **kwargs: True)
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)
        # Answer the provider picker; "databricks" keeps the Databricks path.
        monkeypatch.setattr(cli_mod, "prompt_for_selection", lambda prompt, options: "databricks")

        rc = cli_mod.configure_workspace_command()
        assert rc == 0
        assert codex_path.exists(), "codex config should have been written"
        assert not (tmp_path / "claude-settings.json").exists(), "claude config should NOT exist"
        assert not (tmp_path / "gemini-ucode.env").exists(), "gemini env should NOT exist"
        assert not (tmp_path / "opencode.json").exists(), "opencode config should NOT exist"
        assert not (tmp_path / ".copilot-env").exists(), "copilot env should NOT exist"
        assert not (tmp_path / "pi-models.json").exists(), "pi config should NOT exist"

        state = load_state()
        assert state["available_tools"] == ["codex"]

    def test_rerun_with_different_pick_preserves_previous(
        self, tmp_path, monkeypatch, e2e_workspace
    ):
        """First run picks codex; second run picks claude. State should end
        up with both tools in available_tools (the un-picked codex is not
        dropped on the second run)."""
        import ucode.cli as cli_mod
        import ucode.state as state_mod
        from ucode.state import load_state

        self._redirect_config_paths(monkeypatch, tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr("ucode.cli.run_databricks_login", lambda ws, profile=None: None)
        monkeypatch.setattr(
            cli_mod, "_prompt_for_configuration", lambda tool=None: (e2e_workspace, None)
        )
        monkeypatch.setattr(cli_mod, "install_tool_binary", lambda tool, **kwargs: True)
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)
        # Answer the provider picker; "databricks" keeps the Databricks path.
        monkeypatch.setattr(cli_mod, "prompt_for_selection", lambda prompt, options: "databricks")

        # First run: pick codex.
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["codex"])
        assert cli_mod.configure_workspace_command() == 0
        assert load_state()["available_tools"] == ["codex"]

        # Claude needs to be available on this workspace for the second run
        # to be a meaningful test.
        from ucode.databricks import fetch_ai_gateway_claude_models, get_databricks_token

        token = get_databricks_token(e2e_workspace)
        if not fetch_ai_gateway_claude_models(e2e_workspace, token):
            pytest.skip("No Claude models on this workspace; can't test multi-tool merge.")

        # Second run: pick claude only. Codex should remain in available_tools.
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: ["claude"])
        assert cli_mod.configure_workspace_command() == 0
        assert set(load_state()["available_tools"]) == {"codex", "claude"}

    def test_empty_pick_returns_zero_and_writes_nothing(self, tmp_path, monkeypatch, e2e_workspace):
        """User unchecks everything in the picker → no config files are
        written and the command exits 0."""
        import ucode.cli as cli_mod
        import ucode.state as state_mod

        codex_path = self._redirect_config_paths(monkeypatch, tmp_path)
        monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "state.json")
        monkeypatch.setattr("ucode.cli.run_databricks_login", lambda ws, profile=None: None)
        monkeypatch.setattr(
            cli_mod, "_prompt_for_configuration", lambda tool=None: (e2e_workspace, None)
        )
        monkeypatch.setattr(cli_mod, "prompt_for_tools", lambda available: [])
        install_calls: list[str] = []
        monkeypatch.setattr(
            cli_mod,
            "install_tool_binary",
            lambda tool, **kwargs: install_calls.append(tool) or True,
        )
        monkeypatch.setattr(cli_mod, "validate_all_tools", lambda state: None)

        rc = cli_mod.configure_workspace_command()
        assert rc == 0
        assert not codex_path.exists()
        assert install_calls == [], "no tool binaries should be installed when nothing is picked"


# ---------------------------------------------------------------------------
# Agent launch tests — one test per (agent, model)
# ---------------------------------------------------------------------------
#
# Each test:
#   1. Writes the agent config for the specific model
#   2. Runs the binary with the validate_cmd prompt
#   3. Asserts exit code 0 and non-empty stdout
#
# Tests are skipped when the binary is not installed or no models are available.
# ---------------------------------------------------------------------------

# Model exclusions live here so a temporary outage can be scoped consistently
# across every harness that exercises the affected model.
E2E_MODEL_SKIP_HARNESSES: dict[str, frozenset[str]] = {
    # Codex nano is unreliably slow and exceeds the E2E timeout.
    "gpt-5-4-nano": frozenset({"codex"}),
    # The CI OpenAI project cannot serve this snapshot in its geography.
    "gpt-5-3-codex": frozenset({"codex", "pi"}),
    # Bedrock Grok rejects request/tool shapes from these harnesses.
    "grok": frozenset({"codex", "copilot", "pi"}),
    # These Gemini endpoints hang OpenCode well past its E2E timeout.
    "databricks-gemini-3-1-flash-lite": frozenset({"opencode"}),
    # Codex-tuned and newer GPT endpoints do not support Copilot's MLflow chat route.
    "-codex": frozenset({"copilot"}),
    "gpt-5-5": frozenset({"copilot"}),
    "gpt-5-6": frozenset({"copilot"}),
    # Astra currently fails through these paths in prod-aws-us-east-1.
    "astra": frozenset({"copilot", "pi", "web_search"}),
}


def _model_is_skipped(model: str, harness: str) -> bool:
    normalized = model.lower()
    return any(
        fragment in normalized and harness in harnesses
        for fragment, harnesses in E2E_MODEL_SKIP_HARNESSES.items()
    )


def _require_binary(binary: str):
    if not shutil.which(binary):
        pytest.skip(f"`{binary}` is not installed")


class TestCodexLaunch:
    """Run codex against every available codex model."""

    def _codex_models(self, e2e_state: dict) -> list[str]:
        models = [
            model
            for model in (e2e_state.get("codex_models") or [])
            if not _model_is_skipped(model, "codex")
        ]
        if not models:
            pytest.skip("No Codex models available on this workspace")
        return models

    def test_astra_is_not_skipped(self):
        assert self._codex_models({"codex_models": ["databricks-gpt-6-astra"]}) == [
            "databricks-gpt-6-astra"
        ]

    def test_launch_codex_per_model(self, tmp_path, monkeypatch, e2e_state, e2e_workspace):
        """Parametrized inline — iterates over all codex models and asserts each works."""
        import ucode.config_io as config_io_mod
        from ucode.agents import codex

        _require_binary("codex")
        models = self._codex_models(e2e_state)

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_dir = _codex_home_outside_tmp() / ".codex"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "ucode.config.toml"
        backup_path = tmp_path / "codex-config.backup.toml"
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_path)
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", backup_path)

        failures = []
        timeout_seconds = int(os.environ.get("UCODE_E2E_AGENT_TIMEOUT", "60"))
        for model in models:
            state = {**e2e_state, "workspace": e2e_workspace}
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("ucode.state.save_state", lambda s: None)
                codex.write_tool_config(state, model)

            cmd = codex.validate_cmd("codex")
            try:
                result = _run_agent(
                    cmd,
                    env={**os.environ, "CODEX_HOME": str(config_dir)},
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"model={model} timed out after {timeout_seconds}s")
                continue

            if result.returncode != 0 or not (result.stdout or result.stderr).strip():
                # Keep a generous tail of stderr. codex-cli logs a non-fatal model-listing error
                # first and the actual cause last, so a short prefix reports the wrong problem —
                # at 200 chars the geography failure above read as a `/v1/models` routing error.
                failures.append(
                    f"model={model} rc={result.returncode} "
                    f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-1500:]!r}"
                )

        assert not failures, "Codex launch failures:\n" + "\n".join(failures)


class TestClaudeLaunch:
    """Run claude against every available claude model (sonnet, opus, haiku)."""

    def test_launch_claude_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import claude

        _require_binary("claude")
        claude_models: dict = e2e_state.get("claude_models") or {}
        if not claude_models:
            pytest.skip("No Claude models available on this workspace")
        launchable_models = _launchable_model_items(claude_models)
        if not launchable_models:
            pytest.skip("No launchable Claude models available on this workspace")

        # Use an isolated config dir so the claude subprocess never reads or
        # writes ~/.claude/settings.json during this test.
        config_dir = tmp_path / "claude_config"
        config_dir.mkdir()
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", config_dir / "settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "claude-settings.backup.json")

        base_url = build_tool_base_url("claude", e2e_workspace)

        failures = []
        for family, model_id in launchable_models:
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("ucode.state.save_state", lambda s: None)
                claude.write_tool_config({**e2e_state, "workspace": e2e_workspace}, model_id)

            env = {
                **os.environ,
                "CLAUDE_CONFIG_DIR": str(config_dir),
                "ANTHROPIC_MODEL": model_id,
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": e2e_token,
            }
            cmd = claude.validate_cmd("claude")
            result = _run_agent(cmd, env=env, timeout=90)
            combined = (result.stdout + result.stderr).strip()
            if result.returncode != 0 or not combined:
                failures.append(
                    f"family={family} model={model_id} rc={result.returncode} "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "Claude launch failures:\n" + "\n".join(failures)


class TestModelProviderLaunch:
    """Launch claude/codex routed through a real Model Provider Service.

    Both are pinned to fixed CI MPSes (CI_ANTHROPIC_MPS / CI_OPENAI_MPS), each
    exposing only its cheapest model, so a real request flows through the MPS
    gateway deterministically. Skips when the MPS is absent or the caller lacks
    permission on the backing connection.
    """

    @staticmethod
    def _skip_if_provider_unusable(combined: str, provider: str) -> None:
        # Environmental provider-account conditions, not ucode bugs: the test only proves routing
        # reaches the provider, so skip (rather than fail) when the account lacks a grant on the
        # connection or has run out of credits — state outside the code under test.
        if "USE CONNECTION" in combined or "EXECUTE" in combined:
            pytest.skip(f"no permission on provider {provider}: {combined[:200]}")
        if "Credit balance is too low" in combined:
            pytest.skip(f"provider {provider} account is out of credits: {combined[:200]}")

    def test_launch_claude_through_provider(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import claude, resolve_provider_models

        _require_binary("claude")
        # Pinned to a fixed CI MPS so the inference is deterministic and cheap. Skip (don't fail)
        # when it's absent, so e2e runs against other workspaces still work.
        provider = CI_ANTHROPIC_MPS
        state = {**e2e_state, "workspace": e2e_workspace}
        # Resolve the provider's models as the launch path does. This Anthropic MPS declares a
        # single Haiku target, so provider_models pins that family and route_root_model (below)
        # makes Claude Code launch on exactly it — its built-in default tier isn't in the allowlist.
        provider_models, error, _relayed = resolve_provider_models("claude", state, provider)
        if error is not None:
            pytest.skip(f"CI Anthropic MPS {provider} unavailable on this workspace: {error}")

        config_dir = tmp_path / "claude_config"
        config_dir.mkdir()
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(claude, "CLAUDE_SETTINGS_PATH", config_dir / "settings.json")
        monkeypatch.setattr(claude, "CLAUDE_BACKUP_PATH", tmp_path / "claude-settings.backup.json")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            claude.write_tool_config(
                state,
                None,
                provider=provider,
                provider_models=provider_models,
                route_root_model=CI_ANTHROPIC_MODEL,
            )

        env = {
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "ANTHROPIC_BASE_URL": build_tool_base_url("claude", e2e_workspace),
            "ANTHROPIC_API_KEY": e2e_token,
        }
        result = _run_agent(claude.validate_cmd("claude"), env=env, timeout=90)
        combined = (result.stdout + result.stderr).strip()
        self._skip_if_provider_unusable(combined, provider)
        assert result.returncode == 0 and combined, (
            f"provider={provider} rc={result.returncode} "
            f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
        )

    def test_launch_codex_through_provider(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import codex, resolve_provider_models

        _require_binary("codex")
        # Pinned to the fixed CI OpenAI MPS (Nano-only) — the codex counterpart to the claude pin
        # (codex speaks the OpenAI API, so it can't use the Anthropic MPS). Skip when it's absent.
        provider = CI_OPENAI_MPS
        state = {**e2e_state, "workspace": e2e_workspace, "codex_default_model": CI_OPENAI_MODEL}
        _, error, _ = resolve_provider_models("codex", state, provider)
        if error is not None:
            pytest.skip(f"CI OpenAI MPS {provider} unavailable on this workspace: {error}")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        config_dir = _codex_home_outside_tmp() / ".codex"
        config_dir.mkdir(parents=True)
        monkeypatch.setattr(codex, "CODEX_CONFIG_PATH", config_dir / "ucode.config.toml")
        monkeypatch.setattr(codex, "CODEX_BACKUP_PATH", tmp_path / "codex-config.backup.toml")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            # codex.write_tool_config pins the model from state["codex_default_model"] (set above),
            # so it lands as gpt-5-nano — the only model this MPS allows.
            codex.write_tool_config(state, None, provider=provider)

        timeout_seconds = int(os.environ.get("UCODE_E2E_AGENT_TIMEOUT", "60"))
        try:
            result = _run_agent(
                codex.validate_cmd("codex"),
                env={**os.environ, "CODEX_HOME": str(config_dir)},
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"provider={provider} timed out after {timeout_seconds}s")
        combined = (result.stdout + result.stderr).strip()
        self._skip_if_provider_unusable(combined, provider)
        assert result.returncode == 0 and combined, (
            f"provider={provider} rc={result.returncode} "
            f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
        )


class TestGeminiLaunch:
    """Run gemini against every available gemini model."""

    def test_launch_gemini_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import gemini, validate_tool

        _require_binary("gemini")
        # Gemini CLI >= 0.45 rewrites forced flash model ids (e.g.
        # 'databricks-gemini-3-5-flash') to 'gemini-3.5-flash', which Unity
        # Catalog rejects. ucode caps the supported version below 0.45 and
        # offers a downgrade; skip here if the runner still has a too-new build
        # rather than asserting against a version we deliberately don't support.
        too_new = gemini.too_new_version()
        if too_new is not None:
            pytest.skip(
                f"Installed Gemini CLI {too_new} is past the supported ceiling "
                f"({gemini.MAX_GEMINI_VERSION_TEXT}); run `ucode gemini` to downgrade."
            )
        gemini_models: list = e2e_state.get("gemini_models") or []
        if not gemini_models:
            pytest.skip("No Gemini models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / "ucode.env")
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "gemini-ucode-env.backup")
        monkeypatch.setattr(gemini, "GEMINI_HOME_DIR", tmp_path / ".gemini-home")
        monkeypatch.setattr(
            gemini, "GEMINI_SETTINGS_PATH", tmp_path / ".gemini-home" / ".gemini" / "settings.json"
        )
        # Run from tmp_path so Gemini sees an untrusted folder — that mirrors
        # what users hit on a fresh checkout and exercises the trust + .env
        # discovery code paths that previously broke validation.
        monkeypatch.chdir(tmp_path)

        failures = []
        for model in gemini_models:
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("ucode.state.save_state", lambda s: None)
                mp.setattr(
                    "ucode.agents.gemini.get_databricks_token",
                    lambda ws, profile=None, **kwargs: e2e_token,
                )
                state = {**e2e_state, "workspace": e2e_workspace}
                gemini.write_tool_config(state, model, token=e2e_token)
                # Exercise the real production validate flow — same code path
                # that `ucode configure` invokes after writing the config.
                captured_state = state
                mp.setattr("ucode.agents.load_state", lambda s=captured_state: s)
                ok, err = validate_tool("gemini")
            if not ok:
                failures.append(f"model={model} err={err}")

        assert not failures, "Gemini launch failures:\n" + "\n".join(failures)


class TestGeminiFreshInstall:
    """Verify Gemini works from ucode env without writing user settings.json."""

    def test_does_not_write_settings_json_for_auth(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import gemini

        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / "ucode.env")
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "gemini-ucode-env.backup")

        gemini_models: list = e2e_state.get("gemini_models") or []
        model = gemini_models[0] if gemini_models else "some-model"

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            gemini.write_tool_config(
                {**e2e_state, "workspace": e2e_workspace}, model, token=e2e_token
            )

        assert not settings_path.exists(), "settings.json should not be created"
        assert (tmp_path / "ucode.env").exists(), "ucode Gemini env file was not created"


class TestOpencodeLaunch:
    """Run OpenCode against the available native and OSS model providers."""

    def _all_models(self, e2e_state: dict) -> list[tuple[str, str]]:
        """Return [(provider, model_id), ...] for all opencode models."""
        opencode_models: dict = e2e_state.get("opencode_models") or {}
        out: list[tuple[str, str]] = []
        for provider, models in opencode_models.items():
            for model in models or []:
                if _model_is_skipped(model, "opencode"):
                    continue
                out.append((provider, model))
        return out

    def test_launch_opencode_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import opencode

        _require_binary("opencode")
        models = self._all_models(e2e_state)
        if not models:
            pytest.skip("No OpenCode models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        xdg = tmp_path / "opencode-xdg"
        config_path = xdg / "opencode" / "opencode.json"
        backup_path = tmp_path / "opencode-config.backup.json"
        monkeypatch.setattr(opencode, "OPENCODE_XDG_CONFIG_HOME", xdg)
        monkeypatch.setattr(opencode, "OPENCODE_CONFIG_PATH", config_path)
        monkeypatch.setattr(opencode, "OPENCODE_BACKUP_PATH", backup_path)

        import sys
        import time

        print(f"\n[opencode-per-model] {len(models)} models to test", flush=True)
        failures = []
        for provider, model in models:
            # Reset config file before each model so configs don't bleed together
            if config_path.exists():
                config_path.unlink()

            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("ucode.state.save_state", lambda s: None)
                mp.setattr(
                    "ucode.agents.opencode.get_databricks_token",
                    lambda ws, profile=None, **kwargs: e2e_token,
                )
                opencode.write_tool_config(
                    {**e2e_state, "workspace": e2e_workspace},
                    model,
                    token=e2e_token,
                )

            cmd = opencode.validate_cmd("opencode")
            print(f"[opencode-per-model] -> {provider}/{model}", flush=True)
            t0 = time.monotonic()
            try:
                result = _run_agent(cmd, env=opencode.build_runtime_env(e2e_token), timeout=180)
            except subprocess.TimeoutExpired as exc:
                elapsed = time.monotonic() - t0
                partial_stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
                partial_stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                print(
                    f"[opencode-per-model] {provider}/{model} TIMEOUT after {elapsed:.1f}s\n"
                    f"  partial stdout: {partial_stdout[:500]!r}\n"
                    f"  partial stderr: {partial_stderr[:500]!r}",
                    flush=True,
                    file=sys.stderr,
                )
                failures.append(
                    f"provider={provider} model={model} TIMEOUT after {elapsed:.1f}s "
                    f"stderr={partial_stderr[:300]!r}"
                )
                continue
            elapsed = time.monotonic() - t0
            combined = (result.stdout + result.stderr).strip()
            status = "OK" if result.returncode == 0 and combined else f"FAIL rc={result.returncode}"
            print(f"[opencode-per-model] {provider}/{model} {status} ({elapsed:.1f}s)", flush=True)
            if result.returncode != 0 or not combined:
                failures.append(
                    f"provider={provider} model={model} rc={result.returncode} "
                    f"elapsed={elapsed:.1f}s "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "OpenCode launch failures:\n" + "\n".join(failures)

    def test_launch_deepseek_v4_pro(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        """Discover and invoke the live versioned DeepSeek V4 Pro model."""
        import ucode.config_io as config_io_mod
        from ucode.agents import opencode

        _require_binary("opencode")
        _, _, _, oss_models, reason = discover_model_services(e2e_workspace, e2e_token)
        assert reason is None, reason
        model = next((m for m in oss_models if "deepseek-v4-pro" in m), None)
        if model is None:
            pytest.skip("DeepSeek V4 Pro is not available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        xdg = tmp_path / "opencode-xdg"
        monkeypatch.setattr(opencode, "OPENCODE_XDG_CONFIG_HOME", xdg)
        monkeypatch.setattr(opencode, "OPENCODE_CONFIG_PATH", xdg / "opencode" / "opencode.json")
        monkeypatch.setattr(
            opencode, "OPENCODE_BACKUP_PATH", tmp_path / "opencode-config.backup.json"
        )
        monkeypatch.setattr("ucode.state.save_state", lambda state: None)
        monkeypatch.setattr(
            "ucode.agents.opencode.get_databricks_token",
            lambda workspace, profile=None, **kwargs: e2e_token,
        )

        state = {
            **e2e_state,
            "workspace": e2e_workspace,
            "oss_models": oss_models,
            "opencode_models": {
                **(e2e_state.get("opencode_models") or {}),
                "oss": oss_models,
            },
        }
        opencode.write_tool_config(state, model, token=e2e_token)

        result = _run_agent(
            opencode.validate_cmd("opencode"),
            env=opencode.build_runtime_env(e2e_token),
            timeout=180,
        )
        combined = (result.stdout + result.stderr).strip()
        assert result.returncode == 0 and combined, (
            f"DeepSeek model={model} rc={result.returncode} "
            f"stdout={result.stdout[:300]!r} stderr={result.stderr[:500]!r}"
        )

    def test_recovers_from_rejected_initial_token(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        """Exercise the real OpenCode plugin's 401 refresh-and-retry path."""
        import ucode.config_io as config_io_mod
        from ucode.agents import opencode

        _require_binary("opencode")
        models = self._all_models(e2e_state)
        if not models:
            pytest.skip("No OpenCode models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        xdg = tmp_path / "opencode-xdg"
        monkeypatch.setattr(opencode, "OPENCODE_XDG_CONFIG_HOME", xdg)
        monkeypatch.setattr(opencode, "OPENCODE_CONFIG_PATH", xdg / "opencode" / "opencode.json")
        monkeypatch.setattr(
            opencode, "OPENCODE_BACKUP_PATH", tmp_path / "opencode-config.backup.json"
        )
        monkeypatch.setattr("ucode.state.save_state", lambda s: None)

        _, model = models[0]
        rejected_token = "ucode-intentionally-rejected-token"
        opencode.write_tool_config(
            {**e2e_state, "workspace": e2e_workspace},
            model,
            token=rejected_token,
        )

        env = opencode.build_runtime_env(rejected_token)
        # CI authenticates this way; the helper subprocess inherits it and
        # returns the known-good token after OpenCode's first request gets 401.
        env["DATABRICKS_BEARER"] = e2e_token
        result = _run_agent(
            opencode.validate_cmd("opencode"),
            env=env,
            timeout=180,
        )
        combined = (result.stdout + result.stderr).strip()
        assert result.returncode == 0 and combined, (
            "OpenCode did not recover after its initial token was rejected: "
            f"rc={result.returncode} stdout={result.stdout[:300]!r} "
            f"stderr={result.stderr[:500]!r}"
        )


class TestCopilotLaunch:
    """Run copilot against every Claude/codex model via the MLflow chat-completions gateway.

    Gemini is excluded by design — Databricks' Gemini translator rejects the
    `stream_options` field Copilot CLI sends. Other incompatible models are
    scoped to Copilot in E2E_MODEL_SKIP_HARNESSES.
    """

    def _all_models(self, e2e_state: dict) -> list[tuple[str, str]]:
        """Return [(family, model_id), ...] for every model copilot can talk to."""
        out: list[tuple[str, str]] = []
        claude_models: dict = e2e_state.get("claude_models") or {}
        for family, model_id in _launchable_model_items(claude_models):
            out.append((f"claude-{family}", model_id))
        for model in e2e_state.get("codex_models") or []:
            if _model_is_skipped(model, "copilot"):
                continue
            out.append(("codex", model))
        return out

    def test_astra_is_skipped(self):
        state = {"codex_models": ["databricks-gpt-6-astra", "databricks-gpt-5-4"]}
        assert self._all_models(state) == [("codex", "databricks-gpt-5-4")]

    def test_launch_copilot_per_model(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import copilot

        _require_binary("copilot")
        models = self._all_models(e2e_state)
        if not models:
            pytest.skip("No Copilot-compatible models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        env_path = tmp_path / ".copilot-env"
        backup_path = tmp_path / "copilot-env.backup"
        monkeypatch.setattr(copilot, "COPILOT_ENV_PATH", env_path)
        monkeypatch.setattr(copilot, "COPILOT_BACKUP_PATH", backup_path)

        failures = []
        for family, model in models:
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("ucode.state.save_state", lambda s: None)
                mp.setattr(
                    "ucode.agents.copilot.get_databricks_token",
                    lambda ws, profile=None, **kwargs: e2e_token,
                )
                copilot.write_tool_config(
                    {**e2e_state, "workspace": e2e_workspace}, model, token=e2e_token
                )

            env = copilot.build_runtime_env(e2e_workspace, model, e2e_token)
            cmd = copilot.validate_cmd("copilot")
            result = _run_agent(cmd, env=env, timeout=120)
            combined = (result.stdout + result.stderr).strip()
            if result.returncode != 0 or not combined:
                failures.append(
                    f"family={family} model={model} rc={result.returncode} "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "Copilot launch failures:\n" + "\n".join(failures)


class TestPiLaunch:
    """Run pi against every available model across all four providers.

    Pi has dedicated providers per family (claude, codex, gemini, oss); this
    test exercises each one end-to-end through the validation path.
    """

    def _all_models(self, e2e_state: dict) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        claude_models: dict = e2e_state.get("claude_models") or {}
        for family, model_id in _launchable_model_items(claude_models):
            out.append((f"claude-{family}", model_id))
        for model in e2e_state.get("codex_models") or []:
            if not _model_is_skipped(model, "pi"):
                out.append(("codex", model))
        for model in e2e_state.get("gemini_models") or []:
            out.append(("gemini", model))
        return out

    def test_astra_is_skipped(self):
        state = {"codex_models": ["databricks-gpt-6-astra", "databricks-gpt-5-4"]}
        assert self._all_models(state) == [("codex", "databricks-gpt-5-4")]

    def test_launch_pi_per_model(self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token):
        import ucode.config_io as config_io_mod
        from ucode.agents import pi

        _require_binary("pi")
        models = self._all_models(e2e_state)
        if not models:
            pytest.skip("No Pi-compatible models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        # Point PI_CODING_AGENT_DIR and ucode's config writer at the same
        # isolated directory without changing the process HOME.
        pi_home = tmp_path / "pi-home"
        pi_dir = pi_home / ".pi" / "agent"
        config_path = pi_dir / "models.json"
        backup_path = tmp_path / "pi-models.backup.json"
        monkeypatch.setattr(pi, "PI_UCODE_HOME", pi_home)
        monkeypatch.setattr(pi, "PI_CONFIG_DIR", pi_dir)
        monkeypatch.setattr(pi, "PI_CONFIG_PATH", config_path)
        monkeypatch.setattr(pi, "PI_SETTINGS_PATH", pi_dir / "settings.json")
        monkeypatch.setattr(pi, "PI_SETTINGS_BACKUP_PATH", tmp_path / "pi-settings.backup.json")
        monkeypatch.setattr(pi, "PI_BACKUP_PATH", backup_path)

        failures = []
        for family, model in models:
            if config_path.exists():
                config_path.unlink()

            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("ucode.state.save_state", lambda s: None)
                mp.setattr(
                    "ucode.agents.pi.get_databricks_token",
                    lambda ws, profile=None, **kwargs: e2e_token,
                )
                pi.write_tool_config(
                    {**e2e_state, "workspace": e2e_workspace},
                    model,
                    token=e2e_token,
                )

            env = pi.build_runtime_env(e2e_token)
            cmd = pi.validate_cmd("pi")
            result = _run_agent(cmd, env=env, timeout=120)
            combined = (result.stdout + result.stderr).strip()
            if result.returncode != 0 or not combined:
                failures.append(
                    f"family={family} model={model} rc={result.returncode} "
                    f"stdout={result.stdout[:300]!r} stderr={result.stderr[:300]!r}"
                )

        assert not failures, "Pi launch failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Web search MCP — Databricks-backed Responses API
# ---------------------------------------------------------------------------
#
# Verifies the web_search MCP path against a real workspace:
#   1. The Responses API call with tools=[{type: web_search}] returns text.
#   2. The MCP server subprocess answers initialize/tools/list and tools/call
#      correctly when DATABRICKS_HOST and UCODE_WEB_SEARCH_MODEL are set.
#
# Skipped when the workspace has no Responses-API endpoint (codex_models
# empty), since web_search is unavailable in that case by design.
# ---------------------------------------------------------------------------


def _first_codex_model(e2e_state: dict) -> str:
    models = [
        model
        for model in (e2e_state.get("codex_models") or [])
        if not _model_is_skipped(model, "web_search")
    ]
    if not models:
        pytest.skip("No Responses-API (codex) models available on this workspace")
    return models[0]


def test_web_search_skips_astra():
    state = {"codex_models": ["databricks-gpt-6-astra", "databricks-gpt-5-4"]}
    assert _first_codex_model(state) == "databricks-gpt-5-4"


class TestWebSearchResponsesApi:
    """Hit the real Databricks Codex (Responses API) endpoint with native
    web_search and assert the model returns non-empty text."""

    def test_call_responses_api_returns_text(self, monkeypatch, e2e_state, e2e_workspace):
        from ucode import mcp_web_search

        model = _first_codex_model(e2e_state)
        monkeypatch.setenv("DATABRICKS_HOST", e2e_workspace)
        monkeypatch.setenv("UCODE_WEB_SEARCH_MODEL", model)

        payload = mcp_web_search._call_responses_api(
            "What is today's date? Use web search to confirm."
        )
        assert isinstance(payload, dict)
        text = mcp_web_search._extract_response_text(payload)
        assert text, (
            f"Responses API returned no text output. Full payload (truncated): {str(payload)[:500]}"
        )


class TestWebSearchMcpSubprocess:
    """Drive the `ucode mcp web-search` subprocess over stdio and assert the
    full MCP protocol works end-to-end with a real workspace."""

    def test_subprocess_initialize_list_and_call(self, e2e_state, e2e_workspace):
        if not shutil.which("ucode"):
            pytest.skip("`ucode` binary is not on PATH")
        model = _first_codex_model(e2e_state)

        env = {
            **os.environ,
            "DATABRICKS_HOST": e2e_workspace,
            "UCODE_WEB_SEARCH_MODEL": model,
        }
        # Three MCP requests, one per line.
        requests = [
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
            (
                '{"jsonrpc":"2.0","id":3,"method":"tools/call",'
                '"params":{"name":"web_search","arguments":'
                '{"query":"latest anthropic announcement"}}}'
            ),
        ]
        proc = subprocess.run(
            ["ucode", "mcp", "web-search"],
            input="\n".join(requests) + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"ucode mcp web-search exited {proc.returncode}; stderr={proc.stderr[:500]!r}"
        )

        import json as _json

        responses = [_json.loads(line) for line in proc.stdout.strip().splitlines()]
        assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}: {responses}"

        init = responses[0]["result"]
        assert init["serverInfo"]["name"] == "ucode-web-search"

        tools = responses[1]["result"]["tools"]
        assert any(t["name"] == "web_search" for t in tools)

        call_result = responses[2]["result"]
        assert "isError" not in call_result, (
            f"web_search tool call returned an error: {call_result['content'][0]['text'][:300]}"
        )
        text = call_result["content"][0]["text"]
        assert isinstance(text, str) and text.strip(), "tool call returned empty text"


# ---------------------------------------------------------------------------
# Auth recovery tests
# ---------------------------------------------------------------------------
#
# These tests verify that when Databricks auth fails (empty token), the agents
# recover by re-authenticating rather than hanging or crashing.
#
# Claude uses apiKeyHelper (shell command called by Claude Code on each refresh).
# Gemini/OpenCode/Copilot use get_databricks_token() at launch and on refresh.
# ---------------------------------------------------------------------------


def _make_reauth_fake_databricks(tmp_path, real_token: str) -> str:
    """Write a fake `databricks` binary that returns empty on the first `auth token`
    call, then returns a real token on subsequent calls (simulating session expiry
    followed by successful re-auth). Returns the directory containing the binary."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    call_count = tmp_path / "db_calls"
    call_count.write_text("0")
    fake = tmp_path / "databricks"
    fake.write_text(
        "#!/bin/sh\n"
        f"count=$(cat {call_count})\n"
        f"echo $((count + 1)) > {call_count}\n"
        # auth login is a silent no-op (re-auth succeeds immediately)
        'case "$*" in\n'
        '  *"auth login"*) exit 0 ;;\n'
        "esac\n"
        # first auth token call returns empty (simulates expired session)
        'if [ "$count" -eq 0 ]; then\n'
        '  echo \'{"access_token": "", "token_type": "Bearer"}\'\n'
        "else\n"
        f'  echo \'{{"access_token": "{real_token}", "token_type": "Bearer"}}\'\n'
        "fi\n"
    )
    fake.chmod(0o755)
    return str(tmp_path)


class TestGeminiAuthRecovery:
    """Gemini uses get_databricks_token() at launch — verify it reauths and
    recovers when the first token fetch returns empty."""

    def test_recovers_when_initial_token_empty(
        self, tmp_path, monkeypatch, e2e_state, e2e_workspace, e2e_token
    ):
        import ucode.config_io as config_io_mod
        from ucode.agents import gemini

        _require_binary("gemini")
        gemini_models: list = e2e_state.get("gemini_models") or []
        if not gemini_models:
            pytest.skip("No Gemini models available on this workspace")

        monkeypatch.setattr(config_io_mod, "APP_DIR", tmp_path)
        monkeypatch.setattr(gemini, "GEMINI_ENV_PATH", tmp_path / "ucode.env")
        monkeypatch.setattr(gemini, "GEMINI_BACKUP_PATH", tmp_path / "gemini-ucode-env.backup")
        monkeypatch.setattr(gemini, "GEMINI_HOME_DIR", tmp_path / ".gemini-home")
        monkeypatch.setattr(
            gemini, "GEMINI_SETTINGS_PATH", tmp_path / ".gemini-home" / ".gemini" / "settings.json"
        )

        model = gemini_models[0]
        fake_db_dir = _make_reauth_fake_databricks(tmp_path / "fake_db", e2e_token)

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("ucode.state.save_state", lambda s: None)
            mp.setenv("PATH", f"{fake_db_dir}:{os.environ['PATH']}")
            # get_databricks_token will fail first, reauth, then return e2e_token
            _, recovered_token = gemini.write_tool_config(
                {**e2e_state, "workspace": e2e_workspace}, model
            )

        assert recovered_token == e2e_token, (
            "Expected recovered token after reauth, got empty. "
            "get_databricks_token may not be retrying after auth login."
        )

        assert _run_gemini_gateway_smoke(e2e_workspace, model, recovered_token).strip()
