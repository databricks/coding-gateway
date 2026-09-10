"""Oh-my-pi (omp) coding agent: writes a ucode-private models.yml with Databricks-backed providers.

omp (https://omp.sh) is a multi-provider coding agentIAM-compatible with Pi's
provider model but configured in YAML. We register three providers in its
`models.yml`, each speaking the API dialect best suited to that family's
gateway path:

- `databricks-claude`  (api: anthropic-messages)       → /ai-gateway/anthropic
- `databricks-openai`  (api: openai-responses)         → /ai-gateway/codex/v1
- `databricks-gemini`  (api: google-generative-ai)     → /ai-gateway/gemini/v1beta

Per-provider `compat` flags work around fields the gateway translators reject:

- claude: `supportsEagerToolInputStreaming: false` — the Anthropic translator
  rejects `tools[].eager_input_streaming` on the streaming + tools path that
  omp uses for every request. With this flag omp omits the per-tool field and
  sends the legacy `anthropic-beta: fine-grained-tool-streaming-...` header
  instead, which the gateway accepts.

The default model is pinned via `modelRoles.default` in omp's `config.yml`
(the role value omp's startup resolution consults before falling back to the
first available model), never in `models.yml` — whose root carries only
`providers` (unknown root keys fail omp's schema validation).

OSS / Databricks-foundation models (Llama, Qwen, etc.) are not exposed via
omp today — they live behind /ai-gateway/mlflow/v1 with per-model
`max_tokens` caps that omp has no global way to honor without per-model
config we don't currently maintain.

The bearer token is baked into the file and refreshed by a background thread
while the session runs (same pattern as OpenCode/Copilot).
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from ucode.config_io import (
    APP_DIR,
    ToolSpec,
    backup_existing_file,
    deep_merge_dict,
    read_json_safe,
    read_yaml_safe,
    write_json_file,
    write_yaml_file,
)
from ucode.databricks import (
    TOKEN_REFRESH_INTERVAL_SECONDS,
    build_pi_base_urls,
    get_databricks_token,
)
from ucode.state import mark_tool_managed, save_state
from ucode.telemetry import agent_version, ucode_version

from .args import LaunchOptions

OMP_UCODE_HOME = APP_DIR / "omp-home"
OMP_AGENT_DIR = OMP_UCODE_HOME / ".omp" / "agent"
OMP_MODELS_PATH = OMP_AGENT_DIR / "models.yml"
OMP_CONFIG_PATH = OMP_AGENT_DIR / "config.yml"
OMP_MCP_PATH = OMP_AGENT_DIR / "mcp.json"
OMP_MODELS_BACKUP_PATH = APP_DIR / "omp-models.backup.yml"
OMP_CONFIG_BACKUP_PATH = APP_DIR / "omp-config.backup.yml"
OMP_MCP_BACKUP_PATH = APP_DIR / "omp-mcp.backup.json"

SPEC: ToolSpec = {
    "binary": "omp",
    "package": "@oh-my-pi/pi-coding-agent",
    "display": "Oh My Pi",
    "config_path": OMP_MODELS_PATH,
    "backup_path": OMP_MODELS_BACKUP_PATH,
}

PROVIDER_NAMES = (
    "databricks-claude",
    "databricks-openai",
    "databricks-gemini",
)

PROVIDER_KEYS: list[list[str]] = [["providers", name] for name in PROVIDER_NAMES]

# Old provider names earlier ucode versions wrote; cleaned up on each write so
# users don't end up with stale entries pointing at routes that 400.
LEGACY_PROVIDER_NAMES = ("databricks-anthropic", "databricks-codex", "databricks-oss")


def _resolve_model_selector(
    model: str,
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
) -> str:
    """Return an omp model selector in `<provider>/<model>` form when possible."""
    for name in PROVIDER_NAMES:
        if model.startswith(f"{name}/"):
            return model
    if model in claude_models.values():
        return f"databricks-claude/{model}"
    if model in codex_models:
        return f"databricks-openai/{model}"
    if model in gemini_models:
        return f"databricks-gemini/{model}"
    return model


def render_overlay(
    model: str,
    token: str,
    omp_base_urls: dict[str, str],
    claude_models: dict[str, str],
    codex_models: list[str],
    gemini_models: list[str],
) -> tuple[dict, list[list[str]]]:
    """Return (overlay, managed_key_paths) for omp's private agent config.

    The overlay carries only ``providers``: omp's ``models.yml`` schema
    rejects unknown root keys, so the default-model selector is pinned in
    ``config.yml`` by ``_write_default_model`` instead.
    """
    providers: dict = {}
    keys: list[list[str]] = []
    # omp expands header values that match an env var name. Our UA contains
    # `/` and a space so it can never collide — safe to pass as a literal.
    ua_headers = {"User-Agent": f"ucode/{ucode_version()} omp/{agent_version('omp')}"}

    claude_ids = sorted(set(claude_models.values()))
    if claude_ids:
        providers["databricks-claude"] = {
            "baseUrl": omp_base_urls["claude"],
            "api": "anthropic-messages",
            "apiKey": token,
            "authHeader": True,
            # Gateway's Anthropic translator rejects per-tool
            # `eager_input_streaming` on the streaming + tools path. omp sends
            # the legacy beta header instead when this is false.
            "compat": {"supportsEagerToolInputStreaming": False},
            "headers": ua_headers,
            "models": [{"id": m} for m in claude_ids],
        }
        keys.append(["providers", "databricks-claude"])
    if codex_models:
        providers["databricks-openai"] = {
            "baseUrl": omp_base_urls["openai"],
            "api": "openai-responses",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in codex_models],
        }
        keys.append(["providers", "databricks-openai"])
    if gemini_models:
        providers["databricks-gemini"] = {
            "baseUrl": omp_base_urls["gemini"],
            "api": "google-generative-ai",
            "apiKey": token,
            "authHeader": True,
            "headers": ua_headers,
            "models": [{"id": m} for m in gemini_models],
        }
        keys.append(["providers", "databricks-gemini"])
    overlay: dict = {}
    if providers:
        overlay["providers"] = providers
    return overlay, keys


def write_tool_config(
    state: dict,
    model: str,
    token: str | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[dict, str]:
    backup_existing_file(OMP_MODELS_PATH, OMP_MODELS_BACKUP_PATH)
    if token is None:
        token = get_databricks_token(
            state["workspace"], state.get("profile"), force_refresh=force_refresh
        )
    omp_base_urls = state.get("base_urls", {}).get("omp") or build_pi_base_urls(state["workspace"])
    claude_models = state.get("claude_models") or {}
    codex_models = state.get("codex_models") or []
    gemini_models = state.get("gemini_models") or []
    overlay, managed_keys = render_overlay(
        model,
        token,
        omp_base_urls,
        claude_models,
        codex_models,
        gemini_models,
    )
    existing = read_yaml_safe(OMP_MODELS_PATH)
    providers = existing.get("providers")
    if isinstance(providers, dict):
        for stale in (*PROVIDER_NAMES, *LEGACY_PROVIDER_NAMES):
            providers.pop(stale, None)
    merged = deep_merge_dict(existing, overlay)
    write_yaml_file(OMP_MODELS_PATH, merged)
    _write_default_model(_resolve_model_selector(model, claude_models, codex_models, gemini_models))
    state = mark_tool_managed(state, "omp", managed_keys)
    save_state(state)
    return state, token


def _write_default_model(model_selector: str) -> None:
    # Pin modelRoles.default in config.yml so omp starts on the Databricks
    # model rather than falling through to the first available model (e.g. an
    # env-key-backed provider) in its startup resolution order.
    if "/" not in model_selector:
        return
    backup_existing_file(OMP_CONFIG_PATH, OMP_CONFIG_BACKUP_PATH)
    existing = read_yaml_safe(OMP_CONFIG_PATH)
    merged = deep_merge_dict(existing, {"modelRoles": {"default": model_selector}})
    write_yaml_file(OMP_CONFIG_PATH, merged)


def default_model(state: dict) -> str | None:
    """Prefer Claude opus → sonnet → haiku; fall back to codex, gemini."""
    claude_models = state.get("claude_models") or {}
    for family in ("opus", "sonnet", "haiku"):
        if claude_models.get(family):
            return claude_models[family]
    codex_models = state.get("codex_models") or []
    if codex_models:
        return codex_models[0]
    gemini_models = state.get("gemini_models") or []
    return gemini_models[0] if gemini_models else None


def _refresh_token_once(state: dict, *, force_refresh: bool = False) -> str:
    model = default_model(state)
    if not model:
        raise RuntimeError("No Oh My Pi model is available on this workspace.")
    _, token = write_tool_config(state, model, force_refresh=force_refresh)
    return token


def _refresh_forever(state: dict, stop_event: threading.Event) -> None:
    while not stop_event.wait(TOKEN_REFRESH_INTERVAL_SECONDS):
        try:
            _refresh_token_once(state, force_refresh=True)
        except RuntimeError:
            continue


def build_runtime_env() -> dict[str, str]:
    # omp reads no token from the environment (auth is the baked models.yml
    # apiKey); only redirect its agent dir into the ucode-private home.
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(OMP_AGENT_DIR)
    return env


def launch(state: dict, tool_args: list[str], *, options: LaunchOptions) -> None:
    _refresh_token_once(state)
    env = build_runtime_env()

    stop_event = threading.Event()
    refresher = threading.Thread(
        target=_refresh_forever,
        args=(state, stop_event),
        daemon=True,
    )
    refresher.start()

    proc = subprocess.Popen([SPEC["binary"], *tool_args], env=env)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        stop_event.set()
        refresher.join(timeout=1)

    raise SystemExit(returncode)


def validate_cmd(binary: str) -> list[str]:
    return [binary, "--print", "say hi in 5 words or less"]


def validate_env(state: dict) -> dict[str, str]:
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("No workspace configured.")
    # Fetch a token to fail fast on bad auth; omp itself reads the baked file.
    get_databricks_token(workspace, state.get("profile"))
    return build_runtime_env()


def build_mcp_server_entry(argv: list[str]) -> dict:
    # omp's stdioServer schema allows only command/args/env/cwd (stdio is the
    # default transport when `command` is present without `url`), so the entry
    # is exactly this — no `type`/`tools` keys like other clients use.
    return {
        "command": argv[0],
        "args": list(argv[1:]),
    }


def write_mcp_server_config(name: str, argv: list[str]) -> bool:
    backup_existing_file(OMP_MCP_PATH, OMP_MCP_BACKUP_PATH)
    existing = read_json_safe(OMP_MCP_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    removed = name in mcp_servers
    mcp_servers[name] = build_mcp_server_entry(argv)
    existing["mcpServers"] = mcp_servers
    write_json_file(OMP_MCP_PATH, existing)
    return removed


def remove_mcp_server_config(name: str) -> bool:
    existing = read_json_safe(OMP_MCP_PATH)
    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict) or name not in mcp_servers:
        return False
    mcp_servers.pop(name)
    existing["mcpServers"] = mcp_servers
    write_json_file(OMP_MCP_PATH, existing)
    return True
