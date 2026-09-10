"""Usage report querying & rendering.

Reads from `system.ai_gateway.usage` via a Databricks SQL warehouse.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import NamedTuple, cast

from ucode.databricks import (
    SqlWarehouse,
    apply_pat_environment,
    discover_sql_warehouses,
    ensure_databricks_auth,
    fetch_external_model_prices,
    get_databricks_token,
    resolve_current_budget_spend,
    run_usage_query,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    format_cost_usd,
    format_meter,
    format_token_count,
    format_usd,
    heading,
    label,
    muted,
    print_heading,
    print_note,
    print_warning,
    prompt_for_selection,
    prompt_yes_no_default,
    render_box_table,
    spinner,
    value,
)

USAGE_BREAKDOWN_DAYS = 7
USAGE_SUMMARY_DAYS = 30

QUERY_MESSAGE = "Querying system.ai_gateway.usage..."
STARTUP_MESSAGE = "Starting up warehouse..."
PRICES_MESSAGE = "Fetching model prices..."
# `REQUESTED` is an explicit --warehouse-id, whose state we never looked up.
WARM_WAREHOUSE_STATES = ("RUNNING", "REQUESTED")

MILLION = Decimal(1_000_000)

# Region/provider prefixes the price catalog puts on some model ids (e.g. `eu/gpt-5.6-sol`,
# `us.anthropic.claude-opus-4-8`). They're stripped before matching so a regional id and its base id
# collide onto the same price key.
_PRICE_STRIP_PREFIXES = (
    "eu/",
    "us/",
    "global/",
    "apac.",
    "au.",
    "us.",
    "eu.",
    "anthropic.",
    "databricks-",
    "ap-northeast-1/",
    "ap-northeast-2/",
    "ap-southeast-1/",
    "ap-southeast-2/",
    "ap-south-1/",
    "ca-central-1/",
    "sa-east-1/",
)


class ModelPrice(NamedTuple):
    """USD rates per million tokens. Any field may be None when the catalog omits it."""

    input: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None


def _price_decimal(raw: object) -> Decimal | None:
    if not isinstance(raw, (int, float, str)) or not str(raw).strip():
        return None
    try:
        return Decimal(str(raw).strip())
    except InvalidOperation:
        return None


def normalize_price_key(model_name: str) -> str:
    """Separator-insensitive key that collapses a model's spellings for price matching.

    Usage `gpt-5-6-sol` / catalog `gpt-5.6-sol` / `us.anthropic.claude-opus-4-8` all reduce to one
    key by stripping region/provider prefixes and version/date suffixes, then all non-alphanumerics.
    """
    name = (model_name or "").strip().lower()
    # Loop so stacked prefixes collapse too, e.g. `us.` then `anthropic.`.
    changed = True
    while changed:
        changed = False
        for prefix in _PRICE_STRIP_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    name = name.split("@", 1)[0]
    name = re.sub(r"-v\d+(:\d+)?$", "", name)  # Bedrock version suffix, e.g. `-v1:0`
    name = re.sub(r"-20\d{6}$", "", name)  # date stamp, e.g. `-20251001`
    return re.sub(r"[^a-z0-9]+", "", name)


def _is_bare_model_name(model_name: str) -> bool:
    """True when a catalog id carries no region/provider prefix (its base price)."""
    lowered = model_name.strip().lower()
    return not any(lowered.startswith(prefix) for prefix in _PRICE_STRIP_PREFIXES)


def build_price_lookup(raw_models: list[dict]) -> dict[str, ModelPrice]:
    """Map normalized price keys to `ModelPrice` from the raw catalog listing.

    When several catalog ids collapse to one key (a regional id and its base), the priced, un-prefixed
    entry wins so we bill at the base rate rather than a regional markup.
    """
    lookup: dict[str, ModelPrice] = {}
    scores: dict[str, tuple[bool, bool]] = {}
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("model_name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = normalize_price_key(name)
        if not key:
            continue
        pricing = entry.get("base_pricing")
        pricing = pricing if isinstance(pricing, Mapping) else {}
        price = ModelPrice(
            input=_price_decimal(pricing.get("input_per_million_tokens")),
            output=_price_decimal(pricing.get("output_per_million_tokens")),
            cache_read=_price_decimal(pricing.get("cache_read_per_million_tokens")),
        )
        score = (price.input is not None, _is_bare_model_name(name))
        if key not in scores or score > scores[key]:
            scores[key] = score
            lookup[key] = price
    return lookup


def estimate_model_cost(
    price: ModelPrice | None,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> Decimal | None:
    """Estimate USD cost for one model's token usage, or None when it can't be priced.

    Cached tokens are a subset of input tokens billed at the (cheaper) cache-read rate, so the
    uncached remainder is priced at the input rate. Returns None when the catalog has no input rate
    or when no token breakdown is available (input/output/cached all zero), so the caller shows tokens
    without a dollar figure rather than a misleading $0.
    """
    if price is None or price.input is None:
        return None
    if input_tokens <= 0 and output_tokens <= 0 and cached_tokens <= 0:
        return None
    cached = min(max(cached_tokens, 0), max(input_tokens, 0))
    uncached = max(input_tokens, 0) - cached
    cache_rate = price.cache_read if price.cache_read is not None else price.input
    output_rate = price.output if price.output is not None else Decimal(0)
    total = (
        Decimal(uncached) * price.input
        + Decimal(cached) * cache_rate
        + Decimal(max(output_tokens, 0)) * output_rate
    )
    return total / MILLION


def build_usage_report_query() -> str:
    return f"""
WITH usage_events AS (
SELECT
  current_user() AS requester_name,
  CASE
    WHEN lower(user_agent) LIKE '%codex%' THEN 'codex'
    WHEN lower(user_agent) LIKE '%claude%' THEN 'claude'
    WHEN lower(user_agent) LIKE '%gemini%' THEN 'gemini'
    WHEN lower(user_agent) LIKE '%opencode%' THEN 'opencode'
    ELSE 'other'
  END AS tool,
  date(event_time) AS usage_day,
  request_id,
  event_time,
  destination_model,
  COALESCE(total_tokens, 0) AS total_tokens_used,
  COALESCE(input_tokens, 0) AS input_tokens_used,
  COALESCE(token_details.cache_read_input_tokens, 0) AS cached_tokens_used,
  COALESCE(output_tokens, 0) AS output_tokens_used
FROM system.ai_gateway.usage
WHERE event_time >= current_timestamp() - interval {USAGE_SUMMARY_DAYS} days
  AND requester = current_user()
  AND (
    lower(user_agent) LIKE '%codex%'
    OR lower(user_agent) LIKE '%claude%'
    OR lower(user_agent) LIKE '%gemini%'
    OR lower(user_agent) LIKE '%opencode%'
  )
),
daily_usage AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    SUM(total_tokens_used) AS total_tokens_used,
    COUNT(DISTINCT request_id) AS sessions
  FROM usage_events
  GROUP BY 1, 2, 3
),
model_usage AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    destination_model,
    COUNT(DISTINCT request_id) AS model_requests,
    SUM(total_tokens_used) AS model_tokens_used,
    SUM(input_tokens_used) AS model_input_used,
    SUM(cached_tokens_used) AS model_cached_used,
    SUM(output_tokens_used) AS model_output_used
  FROM usage_events
  GROUP BY 1, 2, 3, 4
),
model_rollup AS (
  SELECT
    requester_name,
    tool,
    usage_day,
    TO_JSON(
      SORT_ARRAY(
        COLLECT_LIST(
          NAMED_STRUCT(
            'model', destination_model,
            'requests', model_requests,
            'tokens', model_tokens_used,
            'input', model_input_used,
            'cached', model_cached_used,
            'output', model_output_used
          )
        )
      )
    ) AS model_tokens
  FROM model_usage
  GROUP BY 1, 2, 3
)
SELECT
  daily_usage.requester_name,
  daily_usage.tool,
  daily_usage.usage_day,
  daily_usage.total_tokens_used,
  daily_usage.sessions,
  COALESCE(model_rollup.model_tokens, '[]') AS model_tokens
FROM daily_usage
LEFT JOIN model_rollup
  ON daily_usage.requester_name = model_rollup.requester_name
  AND daily_usage.tool = model_rollup.tool
  AND daily_usage.usage_day = model_rollup.usage_day
ORDER BY daily_usage.usage_day DESC, daily_usage.tool ASC
""".strip()


def build_current_user_query() -> str:
    return "SELECT current_user() AS requester_name"


def parse_usage_rows(columns: list[str], rows: list[tuple]) -> list[dict[str, object]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def configured_usage_tools(state: dict, tool_displays: dict[str, str]) -> list[str]:
    configured = state.get("available_tools") or state.get("managed_configs", {}).keys()
    if not isinstance(configured, list):
        configured = list(configured)
    return [tool for tool in tool_displays if tool in configured]


def filter_records_for_tools(
    records: list[dict[str, object]],
    tools: list[str],
) -> list[dict[str, object]]:
    configured = set(tools)
    return [record for record in records if record.get("tool") in configured]


def coerce_date(value_obj: object) -> date | None:
    if isinstance(value_obj, date) and not isinstance(value_obj, datetime):
        return value_obj
    if isinstance(value_obj, datetime):
        return value_obj.date()
    if isinstance(value_obj, str):
        try:
            return datetime.fromisoformat(value_obj).date()
        except ValueError:
            return None
    return None


def simplify_model_name(tool: str, model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return "-"

    prefix = "databricks-"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]

    tool_prefixes = {
        "claude": "claude-",
        "gemini": "gemini-",
        "codex": "gpt-",
    }
    tool_prefix = tool_prefixes.get(tool)
    if tool_prefix and normalized.startswith(tool_prefix):
        normalized = normalized[len(tool_prefix) :]
    return normalized


def extract_model_names(tool: str, raw_models: object) -> list[str]:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return []

    unique_models: list[str] = []
    for item in raw_models.split(","):
        simplified = simplify_model_name(tool, item.strip())
        if simplified != "-" and simplified not in unique_models:
            unique_models.append(simplified)
    return unique_models


def summarize_models(tool: str, raw_models: object) -> str:
    if not isinstance(raw_models, str) or not raw_models.strip():
        return "-"
    parts = extract_model_names(tool, raw_models)
    return ", ".join(parts) if parts else "-"


class ModelUsage(NamedTuple):
    """One model's usage for a day/window.

    `key` is a separator-insensitive identity used to merge spellings of the same model (id form vs.
    display form vs. regional variant); `name` is the display name derived from `raw_names` (the raw
    `destination_model` ids). `input` includes cached tokens (`cached` is its cache-read subset).
    """

    name: str
    key: str
    requests: int
    total: int
    input: int
    cached: int
    output: int
    raw_names: tuple[str, ...]


UNKNOWN_MODEL = "<unknown>"
_MODEL_DATABRICKS_PREFIX = "databricks-"


def model_identity_key(name: str) -> str:
    """Identity used to merge spellings of the same model, falling back to the lowercased name."""
    return normalize_price_key(name) or name.strip().lower() or UNKNOWN_MODEL


def canonical_model_name(raw_names: tuple[str, ...]) -> str:
    """Display name for a merged model: prefer the lowercase, space-free id spelling over a display
    label (e.g. `claude-opus-4-8` over `Claude Opus 4.8`), keeping the family prefix and dropping
    only `databricks-`."""
    cleaned: list[str] = []
    for raw in raw_names:
        name = raw.strip()
        if name.lower().startswith(_MODEL_DATABRICKS_PREFIX):
            name = name[len(_MODEL_DATABRICKS_PREFIX) :]
        if name:
            cleaned.append(name)
    if not cleaned:
        return UNKNOWN_MODEL
    id_forms = [name for name in cleaned if " " not in name and name == name.lower()]
    return sorted(id_forms or cleaned, key=lambda name: (len(name), name))[0]


def _coerce_int(raw: object) -> int:
    try:
        return int(cast(int | float | str, raw or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_model_usage_item(item: object) -> ModelUsage | None:
    if not isinstance(item, Mapping):
        return None
    item_mapping = cast(Mapping[str, object], item)

    raw_model = item_mapping.get("model")
    if isinstance(raw_model, str) and raw_model.strip():
        raw_names: tuple[str, ...] = (raw_model.strip(),)
        key = model_identity_key(raw_model)
        name = canonical_model_name(raw_names)
    else:
        # Requests the gateway logged without a destination model still count toward the total.
        raw_names = ()
        key = UNKNOWN_MODEL
        name = UNKNOWN_MODEL
    return ModelUsage(
        name=name,
        key=key,
        requests=_coerce_int(item_mapping.get("requests")),
        total=_coerce_int(item_mapping.get("tokens")),
        input=_coerce_int(item_mapping.get("input")),
        cached=_coerce_int(item_mapping.get("cached")),
        output=_coerce_int(item_mapping.get("output")),
        raw_names=raw_names,
    )


def extract_model_usage(raw_model_tokens: object) -> list[ModelUsage]:
    """Per-model usage from a row's `model_tokens` JSON, merged by identity, highest tokens first."""
    try:
        items = (
            json.loads(raw_model_tokens) if isinstance(raw_model_tokens, str) else raw_model_tokens
        )
    except json.JSONDecodeError:
        items = []

    merged: dict[str, ModelUsage] = {}
    if isinstance(items, list):
        for item in items:
            coerced = _coerce_model_usage_item(item)
            if coerced:
                _merge_model_usage(merged, coerced)
    return sorted(merged.values(), key=lambda u: (-u.total, u.name.lower()))


def _merge_model_usage(merged: dict[str, ModelUsage], usage: ModelUsage) -> None:
    """Accumulate `usage` into `merged` by identity key; the display name is recomputed from the
    unioned raw names so a model's id and display spellings fold into one row across days."""
    existing = merged.get(usage.key)
    if existing is None:
        merged[usage.key] = usage
        return
    raw_names = tuple(dict.fromkeys(existing.raw_names + usage.raw_names))
    merged[usage.key] = existing._replace(
        name=canonical_model_name(raw_names) if raw_names else existing.name,
        requests=existing.requests + usage.requests,
        total=existing.total + usage.total,
        input=existing.input + usage.input,
        cached=existing.cached + usage.cached,
        output=existing.output + usage.output,
        raw_names=raw_names,
    )


def model_usage_cost(
    usage: ModelUsage,
    price_lookup: dict[str, ModelPrice] | None,
) -> Decimal | None:
    """Estimated USD cost for one model's usage, or None when it can't be priced."""
    if not price_lookup:
        return None
    for raw_name in usage.raw_names:
        price = price_lookup.get(normalize_price_key(raw_name))
        if price is None:
            continue
        cost = estimate_model_cost(price, usage.input, usage.cached, usage.output)
        if cost is not None:
            return cost
    return None


def has_tool_usage_last_week(records: list[dict[str, object]], tool: str) -> bool:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    for record in records:
        if record.get("tool") != tool:
            continue
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day or usage_day < week_start:
            continue
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        session_total = int(cast(int, record.get("sessions") or 0))
        if token_total or session_total:
            return True
    return False


class ToolUsageTotals(NamedTuple):
    """Tool-level rollup for the week: total requests, tokens, and cost (None when unpriceable)."""

    requests: int
    tokens: int
    cost: Decimal | None


TOOL_MODEL_TABLE_HEADERS = [
    "Model",
    "Requests",
    "Input (incl. cache)",
    "Output",
    "Est. Cost (USD)",
]


def aggregate_tool_model_usage(records: list[dict[str, object]], tool: str) -> list[ModelUsage]:
    """Per-model usage for `tool` over the last 7 days, merged across days, highest tokens first."""
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    merged: dict[str, ModelUsage] = {}
    for record in records:
        if record.get("tool") != tool:
            continue
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day or usage_day < week_start:
            continue
        for model_usage in extract_model_usage(record.get("model_tokens")):
            _merge_model_usage(merged, model_usage)
    return sorted(merged.values(), key=lambda u: (-u.total, u.name.lower()))


def build_tool_model_rows(
    records: list[dict[str, object]],
    tool: str,
    price_lookup: dict[str, ModelPrice] | None = None,
) -> tuple[list[list[str]], ToolUsageTotals]:
    """Per-model table rows + tool totals for the week.

    Columns match `TOOL_MODEL_TABLE_HEADERS`: model, requests, input (incl. cache), output, cost.
    Cost is a dash for models the price catalog doesn't cover; the totals' cost is None when nothing
    in the tool could be priced.
    """
    rows: list[list[str]] = []
    total_requests = 0
    total_tokens = 0
    total_cost = Decimal(0)
    any_priced = False
    for usage in aggregate_tool_model_usage(records, tool):
        cost = model_usage_cost(usage, price_lookup)
        rows.append(
            [
                usage.name,
                f"{usage.requests:,}",
                format_token_count(usage.input),
                format_token_count(usage.output),
                format_cost_usd(cost) if cost is not None else "-",
            ]
        )
        total_requests += usage.requests
        total_tokens += usage.input + usage.output
        if cost is not None:
            total_cost += cost
            any_priced = True
    totals = ToolUsageTotals(
        requests=total_requests,
        tokens=total_tokens,
        cost=total_cost if any_priced else None,
    )
    return rows, totals


def find_requester_name(
    workspace: str,
    http_path: str,
    token: str,
    records: list[dict[str, object]],
) -> str:
    for record in records:
        requester_name = record.get("requester_name")
        if isinstance(requester_name, str) and requester_name.strip():
            return requester_name.strip()

    columns, rows = run_usage_query(workspace, http_path, token, build_current_user_query())
    parsed_rows = parse_usage_rows(columns, rows)
    if parsed_rows:
        requester_name = parsed_rows[0].get("requester_name")
        if isinstance(requester_name, str) and requester_name.strip():
            return requester_name.strip()
    return "current user"


def render_budget_lines(budget_spend: tuple[Decimal, Decimal] | None) -> list[str]:
    """Spend-against-threshold lines, or nothing when unavailable."""
    if budget_spend is None:
        return []
    spend, threshold = budget_spend
    # No whole to be a fraction of; dividing would raise.
    if threshold <= 0:
        return [f"{label('Budget spend:')} {value(format_usd(spend))}"]
    fraction = float(spend / threshold)
    summary = f"{format_usd(spend)} of {format_usd(threshold)} ({fraction:.0%})"
    return [
        f"{label('Budget spend:')} {value(summary)}",
        muted(format_meter(fraction)),
    ]


def render_usage_summary(
    records: list[dict[str, object]],
    requester_name: str,
    tool_displays: dict[str, str],
    budget_spend: tuple[Decimal, Decimal] | None = None,
    price_lookup: dict[str, ModelPrice] | None = None,
) -> str:
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    month_start = today - timedelta(days=USAGE_SUMMARY_DAYS - 1)

    daily_total = 0
    weekly_total = 0
    monthly_total = 0
    active_tools_last_week: list[str] = []
    weekly_model_usage: dict[str, ModelUsage] = {}
    for record in records:
        usage_day = coerce_date(record.get("usage_day"))
        if not usage_day:
            continue
        token_total = int(cast(int, record.get("total_tokens_used") or 0))
        tool = record.get("tool")
        if usage_day >= month_start:
            monthly_total += token_total
        if usage_day >= week_start:
            weekly_total += token_total
            if (
                isinstance(tool, str)
                and tool in tool_displays
                and tool not in active_tools_last_week
            ):
                active_tools_last_week.append(tool)
            if isinstance(tool, str):
                for model_usage in extract_model_usage(record.get("model_tokens")):
                    _merge_model_usage(weekly_model_usage, model_usage)
        if usage_day == today:
            daily_total += token_total

    today_text = format_token_count(daily_total)
    weekly_text = format_token_count(weekly_total)
    monthly_text = format_token_count(monthly_total)
    token_table = render_box_table(
        ["Today", "Last 7 days", "Last 30 days"],
        [[f"{today_text} tokens", f"{weekly_text} tokens", f"{monthly_text} tokens"]],
    )
    for token_text in {today_text, weekly_text, monthly_text}:
        plain_value = f"{token_text} tokens"
        token_table = token_table.replace(plain_value, value(plain_value))
    lines = [
        heading(f"Usage Summary for {requester_name}"),
        "",
        "[bold green]✓[/bold green]  [bold]Databricks AI Gateway Token Usage[/bold]",
        token_table,
    ]
    if active_tools_last_week:
        tool_text = " · ".join(tool_displays[tool] for tool in active_tools_last_week)
        lines.extend(["", f"   🤖  {label('Top Harnesses:')} {value(tool_text)}"])
    if weekly_model_usage:
        top_models = sorted(
            weekly_model_usage.values(),
            key=lambda u: (-u.total, u.name.lower()),
        )[:3]
        models_text = " · ".join(usage.name for usage in top_models)
        lines.append(f"   ⭐  {label('Top Models:')} {value(models_text)}")
        weekly_cost = sum(
            (
                model_usage_cost(usage, price_lookup) or Decimal(0)
                for usage in weekly_model_usage.values()
            ),
            Decimal(0),
        )
        if weekly_cost > 0:
            lines.append(
                f"   💰  {label('Estimated Cost for Last 7 Days:')} "
                f"{value(format_cost_usd(weekly_cost))}"
            )
    lines.extend(render_budget_lines(budget_spend))
    return "\n".join(lines)


def run_query_on_first_working_warehouse(
    workspace: str,
    token: str,
    candidates: list[SqlWarehouse],
    query: str,
) -> tuple[str, list[str], list[tuple]]:
    """Run `query` on the first candidate that accepts the connection.

    Returns the warehouse's http path alongside the result so later queries
    reuse it. Raises the last error when every candidate fails.
    """
    last_error: RuntimeError | None = None
    for warehouse in candidates:
        print_note(f"Using SQL warehouse `{warehouse.label}` ({warehouse.state}).")
        try:
            # Inside the loop so the spinner stops before any warning prints.
            columns, rows = _query_with_progress(workspace, token, warehouse, query)
        except RuntimeError as exc:
            last_error = exc
            print_warning(f"SQL warehouse `{warehouse.label}` is unusable: {exc}")
            continue
        return warehouse.http_path, columns, rows
    raise last_error or RuntimeError("No SQL warehouse could run the usage query.")


def select_sql_warehouse(candidates: list[SqlWarehouse]) -> SqlWarehouse | None:
    """Ask which discovered warehouse should run the detailed usage query."""
    selected_path = prompt_for_selection(
        "Select a SQL warehouse for the usage query:",
        [
            (warehouse.http_path, f"{warehouse.label} ({warehouse.state})")
            for warehouse in candidates
        ],
    )
    return next(
        (warehouse for warehouse in candidates if warehouse.http_path == selected_path),
        None,
    )


def select_and_run_usage_query(
    workspace: str,
    token: str,
    candidates: list[SqlWarehouse],
    query: str,
) -> tuple[str, list[str], list[tuple]] | None:
    """Let the user choose warehouses until one runs the query or they cancel.

    A failed warehouse is removed before the picker is shown again so the user cannot
    accidentally retry the same unusable option forever. Returns ``None`` when the picker
    is cancelled and raises the final warehouse error when no candidates remain.
    """
    remaining = list(candidates)
    last_error: RuntimeError | None = None
    while remaining:
        selected = select_sql_warehouse(remaining)
        if selected is None:
            return None
        try:
            return run_query_on_first_working_warehouse(workspace, token, [selected], query)
        except RuntimeError as exc:
            last_error = exc
            remaining = [
                warehouse for warehouse in remaining if warehouse.http_path != selected.http_path
            ]
    raise last_error or RuntimeError("No SQL warehouse could run the usage query.")


def _query_with_progress(
    workspace: str,
    token: str,
    warehouse: SqlWarehouse,
    query: str,
) -> tuple[list[str], list[tuple]]:
    """Run the query, reporting a cold start until the connection opens.

    A warehouse that isn't already up costs minutes to start, so the spinner
    says that until `run_usage_query` reports it connected.
    """
    connected = warehouse.state in WARM_WAREHOUSE_STATES

    def mark_connected() -> None:
        nonlocal connected
        connected = True

    with spinner(lambda: QUERY_MESSAGE if connected else STARTUP_MESSAGE):
        return run_usage_query(workspace, warehouse.http_path, token, query, mark_connected)


def usage(warehouse_id: str | None = None) -> int:
    # Late import to avoid circular import (agents → state, but usage uses TOOL_SPECS for displays).
    from ucode.agents import TOOL_SPECS

    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    profile = state.get("profile")
    apply_pat_environment(state)
    ensure_databricks_auth(workspace, profile)
    with spinner("Retrieving Databricks access token..."):
        token = get_databricks_token(workspace, profile)

    # Budget spend comes from AI Gateway directly, so show the useful at-a-glance result before
    # asking whether to start (and potentially wait for) a SQL warehouse for the detailed report.
    with spinner("Checking budget spend..."):
        budget_spend, _ = resolve_current_budget_spend(workspace, token)
    budget_lines = render_budget_lines(budget_spend)
    if budget_lines:
        console.print("\n".join([heading("Usage Budget"), "", *budget_lines]))
    else:
        print_note(
            "Budget spend and threshold are unavailable. "
            "Please configure a coding agent budget policy first."
        )

    if not prompt_yes_no_default(
        "Show token usage and estimated cost details? This queries a SQL warehouse.",
        default=False,
    ):
        return 0

    with spinner("Discovering SQL warehouse..."):
        candidates = discover_sql_warehouses(workspace, token, warehouse_id=warehouse_id)
    if warehouse_id is None:
        query_result = select_and_run_usage_query(
            workspace, token, candidates, build_usage_report_query()
        )
        if query_result is None:
            print_note("Usage details cancelled.")
            return 0
        resolved_http_path, columns, rows = query_result
    else:
        resolved_http_path, columns, rows = run_query_on_first_working_warehouse(
            workspace, token, candidates, build_usage_report_query()
        )
    records = parse_usage_rows(columns, rows)
    requester_name = find_requester_name(workspace, resolved_http_path, token, records)

    # Per-model dollar cost is estimated from tokens × catalog prices; omit cost rather than fail
    # when the price catalog is unreachable.
    with spinner(PRICES_MESSAGE):
        raw_prices, _ = fetch_external_model_prices(workspace, token)
    price_lookup = build_price_lookup(raw_prices)

    tool_displays = {tool: spec["display"] for tool, spec in TOOL_SPECS.items()}
    configured_tools = configured_usage_tools(state, tool_displays)
    configured_tool_displays = {tool: tool_displays[tool] for tool in configured_tools}
    records = filter_records_for_tools(records, configured_tools)

    console.print(
        render_usage_summary(
            records,
            requester_name,
            configured_tool_displays,
            price_lookup=price_lookup,
        )
    )

    table_widths = [24, 10, 20, 10, 12]
    today = date.today()
    week_start = today - timedelta(days=USAGE_BREAKDOWN_DAYS - 1)
    date_range = f"{week_start:%b %d}–{today:%b %d, %Y}"

    if not configured_tools:
        print_note("No coding agents configured. Run `ucode configure` to set up agents.")
        return 0

    for tool in configured_tools:
        display = tool_displays[tool]
        print_heading(f"{display} · Last {USAGE_BREAKDOWN_DAYS} Days")
        if not has_tool_usage_last_week(records, tool):
            print_note(f"No usage for {display} in the last {USAGE_BREAKDOWN_DAYS} days.")
            continue
        rows, totals = build_tool_model_rows(records, tool, price_lookup)
        console.print(muted(date_range))
        console.print(f"{label('Requests:')} {value(f'{totals.requests:,}')}")
        console.print(f"{label('Total tokens:')} {value(f'{totals.tokens:,}')}")
        if totals.cost is not None:
            console.print(f"{label('Est. Cost (USD):')} {value(format_cost_usd(totals.cost))}")
        console.print(render_box_table(TOOL_MODEL_TABLE_HEADERS, rows, max_widths=table_widths))
    return 0
