"""Tests for usage.py — query builders, parsing/formatting, rendering."""

from __future__ import annotations

import contextlib
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

import ucode.usage as usage_mod
from ucode.databricks import SqlWarehouse
from ucode.ui import label, value
from ucode.usage import (
    TOOL_MODEL_TABLE_HEADERS,
    USAGE_BREAKDOWN_DAYS,
    USAGE_SUMMARY_DAYS,
    ModelPrice,
    aggregate_tool_model_usage,
    build_current_user_query,
    build_price_lookup,
    build_tool_model_rows,
    build_usage_report_query,
    coerce_date,
    configured_usage_tools,
    estimate_model_cost,
    extract_model_names,
    extract_model_usage,
    filter_records_for_tools,
    has_tool_usage_last_week,
    model_usage_cost,
    normalize_price_key,
    parse_usage_rows,
    render_budget_lines,
    render_usage_summary,
    run_query_on_first_working_warehouse,
    select_and_run_usage_query,
    select_sql_warehouse,
    simplify_model_name,
    summarize_models,
    usage,
)


class TestBuildUsageReportQuery:
    def test_contains_system_table(self):
        q = build_usage_report_query()
        assert "system.ai_gateway.usage" in q

    def test_contains_interval(self):
        q = build_usage_report_query()
        assert str(USAGE_SUMMARY_DAYS) in q

    def test_filters_known_tools(self):
        q = build_usage_report_query()
        for tool in ("codex", "claude", "gemini", "opencode"):
            assert tool in q

    def test_includes_per_model_token_rollup(self):
        q = build_usage_report_query()
        assert "model_tokens" in q
        assert "SUM(total_tokens_used) AS model_tokens_used" in q
        assert "'model', destination_model" in q
        assert "'tokens', model_tokens_used" in q

    def test_includes_per_model_cost_token_breakdown(self):
        q = build_usage_report_query()
        # input/cached/output tokens are needed to price a model's usage at distinct rates.
        assert "COALESCE(input_tokens, 0) AS input_tokens_used" in q
        assert "COALESCE(token_details.cache_read_input_tokens, 0) AS cached_tokens_used" in q
        assert "COALESCE(output_tokens, 0) AS output_tokens_used" in q
        assert "'input', model_input_used" in q
        assert "'cached', model_cached_used" in q
        assert "'output', model_output_used" in q

    def test_includes_per_model_request_count(self):
        q = build_usage_report_query()
        assert "COUNT(DISTINCT request_id) AS model_requests" in q
        assert "'requests', model_requests" in q


class TestBuildCurrentUserQuery:
    def test_uses_current_user(self):
        q = build_current_user_query()
        assert "current_user()" in q


class TestParseUsageRows:
    def test_zips_columns_and_rows(self):
        columns = ["a", "b", "c"]
        rows = [(1, 2, 3), (4, 5, 6)]
        result = parse_usage_rows(columns, rows)
        assert result == [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

    def test_empty_rows(self):
        assert parse_usage_rows(["a"], []) == []


class TestConfiguredUsageTools:
    def test_uses_available_tools_in_display_order(self):
        tool_displays = {"claude": "Claude Code", "codex": "Codex", "gemini": "Gemini"}
        state = {"available_tools": ["codex", "claude"]}
        assert configured_usage_tools(state, tool_displays) == ["claude", "codex"]

    def test_falls_back_to_managed_configs(self):
        tool_displays = {"claude": "Claude Code", "codex": "Codex"}
        state = {"managed_configs": {"codex": {"keys": []}}}
        assert configured_usage_tools(state, tool_displays) == ["codex"]

    def test_ignores_unknown_tools(self):
        tool_displays = {"claude": "Claude Code"}
        state = {"available_tools": ["claude", "unknown"]}
        assert configured_usage_tools(state, tool_displays) == ["claude"]


class TestFilterRecordsForTools:
    def test_keeps_only_configured_tools(self):
        records = [
            {"tool": "claude", "total_tokens_used": 100},
            {"tool": "gemini", "total_tokens_used": 200},
            {"tool": "codex", "total_tokens_used": 300},
        ]
        assert filter_records_for_tools(records, ["claude", "codex"]) == [
            {"tool": "claude", "total_tokens_used": 100},
            {"tool": "codex", "total_tokens_used": 300},
        ]


class TestHasToolUsageLastWeek:
    def test_true_for_recent_tokens(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is True

    def test_true_for_recent_session_even_without_tokens(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 0,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is True

    def test_false_for_only_old_usage(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today() - timedelta(days=USAGE_BREAKDOWN_DAYS),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is False

    def test_false_for_other_tool_usage(self):
        records = [
            {
                "tool": "codex",
                "usage_day": date.today(),
                "total_tokens_used": 100,
                "sessions": 1,
            }
        ]
        assert has_tool_usage_last_week(records, "claude") is False


class TestCoerceDate:
    def test_date_passthrough(self):
        d = date(2024, 6, 1)
        assert coerce_date(d) == d

    def test_datetime_to_date(self):
        dt = datetime(2024, 6, 1, 12, 0, 0)
        assert coerce_date(dt) == date(2024, 6, 1)

    def test_iso_string(self):
        assert coerce_date("2024-06-01") == date(2024, 6, 1)

    def test_invalid_string_returns_none(self):
        assert coerce_date("not-a-date") is None

    def test_none_returns_none(self):
        assert coerce_date(None) is None


class TestSimplifyModelName:
    def test_strips_databricks_and_tool_prefix(self):
        # databricks- stripped first, then claude- stripped → "sonnet-4"
        assert simplify_model_name("claude", "databricks-claude-sonnet-4") == "sonnet-4"

    def test_gemini_prefix(self):
        result = simplify_model_name("gemini", "databricks-gemini-2.0-flash")
        assert result == "2.0-flash"

    def test_codex_strips_gpt_prefix(self):
        result = simplify_model_name("codex", "databricks-gpt-4o")
        assert result == "4o"

    def test_empty_returns_dash(self):
        assert simplify_model_name("claude", "") == "-"

    def test_no_known_prefix_returns_as_is(self):
        result = simplify_model_name("claude", "some-other-model")
        assert result == "some-other-model"

    def test_only_databricks_prefix_stripped_for_unknown_tool(self):
        result = simplify_model_name("opencode", "databricks-claude-sonnet-4")
        assert result == "claude-sonnet-4"


class TestExtractModelNames:
    def test_single_model(self):
        result = extract_model_names("claude", "databricks-claude-sonnet-4")
        assert result == ["sonnet-4"]

    def test_multiple_models(self):
        result = extract_model_names(
            "claude", "databricks-claude-sonnet-4, databricks-claude-opus-4"
        )
        assert "sonnet-4" in result
        assert "opus-4" in result

    def test_deduplicates(self):
        result = extract_model_names(
            "claude", "databricks-claude-sonnet-4, databricks-claude-sonnet-4"
        )
        assert result.count("sonnet-4") == 1

    def test_empty_returns_empty_list(self):
        assert extract_model_names("claude", "") == []

    def test_non_string_returns_empty_list(self):
        assert extract_model_names("claude", None) == []


class TestSummarizeModels:
    def test_single_model(self):
        result = summarize_models("claude", "databricks-claude-sonnet-4")
        assert result == "sonnet-4"

    def test_multiple_models_joined(self):
        result = summarize_models("claude", "databricks-claude-sonnet-4, databricks-claude-opus-4")
        assert "sonnet-4" in result
        assert "," in result

    def test_empty_returns_dash(self):
        assert summarize_models("claude", "") == "-"

    def test_none_returns_dash(self):
        assert summarize_models("claude", None) == "-"


class TestExtractModelUsage:
    def test_extracts_json_model_tokens_full_names_ordered(self):
        raw = (
            '[{"model":"databricks-claude-opus-4", "tokens":236000}, '
            '{"model":"databricks-claude-haiku-4.5", "tokens":920}]'
        )
        # Full model names (family prefix kept), highest tokens first.
        usages = extract_model_usage(raw)
        assert [(u.name, u.total) for u in usages] == [
            ("claude-opus-4", 236000),
            ("claude-haiku-4.5", 920),
        ]

    def test_merges_duplicate_model_spellings(self):
        raw = [
            {"model": "databricks-claude-opus-4", "tokens": 100},
            {"model": "claude-opus-4", "tokens": 50},
        ]
        usages = extract_model_usage(raw)
        assert [(u.name, u.total) for u in usages] == [("claude-opus-4", 150)]

    def test_empty_or_unparseable_yields_nothing(self):
        assert extract_model_usage("[]") == []
        assert extract_model_usage("not json") == []
        assert extract_model_usage(None) == []


class TestNormalizePriceKey:
    def test_dash_and_dot_versions_collapse(self):
        # Usage table uses dashes (`gpt-5-6-sol`); the catalog uses dots (`gpt-5.6-sol`).
        assert normalize_price_key("gpt-5-6-sol") == normalize_price_key("gpt-5.6-sol")

    def test_display_name_matches_id(self):
        assert normalize_price_key("Claude Opus 4.8") == normalize_price_key(
            "anthropic.claude-opus-4-8"
        )

    def test_region_prefix_stripped(self):
        assert normalize_price_key("eu/gpt-5.6-sol") == normalize_price_key("gpt-5.6-sol")

    def test_bedrock_version_and_date_suffix_stripped(self):
        assert normalize_price_key(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        ) == normalize_price_key("claude-haiku-4-5")

    def test_distinct_models_stay_distinct(self):
        assert normalize_price_key("gpt-5-nano") != normalize_price_key("gpt-5")


class TestBuildPriceLookup:
    def _catalog(self):
        return [
            {
                "model_name": "gpt-5.6-sol",
                "base_pricing": {
                    "input_per_million_tokens": 5.0,
                    "output_per_million_tokens": 30.0,
                    "cache_read_per_million_tokens": 0.5,
                },
            },
            {
                "model_name": "eu/gpt-5.6-sol",
                "base_pricing": {
                    "input_per_million_tokens": 6.0,
                    "output_per_million_tokens": 36.0,
                },
            },
        ]

    def test_prefers_bare_name_over_region_prefixed(self):
        lookup = build_price_lookup(self._catalog())
        price = lookup[normalize_price_key("gpt-5.6-sol")]
        assert price.input == Decimal("5.0")
        assert price.cache_read == Decimal("0.5")

    def test_ignores_entries_without_model_name(self):
        assert build_price_lookup([{"base_pricing": {"input_per_million_tokens": 1.0}}]) == {}


class TestEstimateModelCost:
    def test_prices_uncached_cached_and_output_separately(self):
        price = ModelPrice(input=Decimal("5"), output=Decimal("30"), cache_read=Decimal("0.5"))
        # 1M input of which 200k cached, 100k output.
        cost = estimate_model_cost(price, 1_000_000, 200_000, 100_000)
        # 800k*5 + 200k*0.5 + 100k*30, all /1e6 = 4.0 + 0.1 + 3.0
        assert cost == Decimal("7.1")

    def test_cached_falls_back_to_input_rate_when_no_cache_price(self):
        price = ModelPrice(input=Decimal("5"), output=Decimal("30"), cache_read=None)
        cost = estimate_model_cost(price, 1_000_000, 200_000, 0)
        assert cost == Decimal("5")

    def test_none_when_no_input_rate(self):
        price = ModelPrice(input=None, output=Decimal("30"), cache_read=None)
        assert estimate_model_cost(price, 1_000_000, 0, 0) is None

    def test_none_when_no_token_breakdown(self):
        price = ModelPrice(input=Decimal("5"), output=Decimal("30"), cache_read=None)
        assert estimate_model_cost(price, 0, 0, 0) is None


class TestModelUsageCostAndRendering:
    def _lookup(self):
        return build_price_lookup(
            [
                {
                    "model_name": "anthropic.claude-opus-4-8",
                    "base_pricing": {
                        "input_per_million_tokens": 5.0,
                        "output_per_million_tokens": 25.0,
                        "cache_read_per_million_tokens": 0.5,
                    },
                }
            ]
        )

    def test_extract_model_usage_carries_token_breakdown(self):
        raw = (
            '[{"model":"claude-opus-4-8","requests":3,'
            '"tokens":1000,"input":800,"cached":200,"output":100}]'
        )
        usages = extract_model_usage(raw)
        assert len(usages) == 1
        u = usages[0]
        assert (u.name, u.requests, u.total, u.input, u.cached, u.output) == (
            "claude-opus-4-8",
            3,
            1000,
            800,
            200,
            100,
        )
        assert u.raw_names == ("claude-opus-4-8",)

    def test_null_model_bucketed_as_unknown(self):
        raw = '[{"model":null,"requests":1,"tokens":500,"input":400,"cached":0,"output":100}]'
        (u,) = extract_model_usage(raw)
        assert u.name == "<unknown>"
        assert u.raw_names == ()

    def test_cost_matched_via_raw_name(self):
        raw = (
            '[{"model":"claude-opus-4-8","requests":1,"tokens":1100000,'
            '"input":1000000,"cached":200000,"output":100000}]'
        )
        (u,) = extract_model_usage(raw)
        cost = model_usage_cost(u, self._lookup())
        # 800k*5 + 200k*0.5 + 100k*25 = 4 + 0.1 + 2.5
        assert cost == Decimal("6.6")


class TestBuildToolModelRows:
    def _lookup(self):
        return build_price_lookup(
            [
                {
                    "model_name": "anthropic.claude-opus-4-8",
                    "base_pricing": {
                        "input_per_million_tokens": 5.0,
                        "output_per_million_tokens": 25.0,
                        "cache_read_per_million_tokens": 0.5,
                    },
                }
            ]
        )

    def _records(self):
        # Two days of the same tool; opus-4-8 appears on both and must aggregate.
        return [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 1_100_000,
                "model_tokens": (
                    '[{"model":"claude-opus-4-8","requests":2,"tokens":1100000,'
                    '"input":1000000,"cached":200000,"output":100000}]'
                ),
            },
            {
                "tool": "claude",
                "usage_day": date.today() - timedelta(days=1),
                "total_tokens_used": 500,
                "model_tokens": (
                    '[{"model":"mystery-model","requests":1,"tokens":500,'
                    '"input":400,"cached":0,"output":100}]'
                ),
            },
        ]

    def test_aggregates_per_model_over_week(self):
        usages = aggregate_tool_model_usage(self._records(), "claude")
        # opus-4-8 (1.1M) sorts ahead of mystery-model (500).
        assert [u.name for u in usages] == ["claude-opus-4-8", "mystery-model"]
        assert usages[0].requests == 2

    def test_rows_and_totals(self):
        rows, totals = build_tool_model_rows(self._records(), "claude", self._lookup())
        # columns: model, requests, input (incl cache), output, cost
        assert rows[0] == ["claude-opus-4-8", "2", "1.0M", "100.0K", "$6.60"]
        assert rows[1] == ["mystery-model", "1", "400", "100", "-"]
        assert totals.requests == 3
        assert totals.tokens == 1_100_000 + 500
        assert totals.cost == Decimal("6.6")

    def test_totals_cost_none_when_nothing_priced(self):
        records = [self._records()[1]]  # only the unpriced mystery-model
        _, totals = build_tool_model_rows(records, "claude", self._lookup())
        assert totals.cost is None

    def test_cost_header_is_explicitly_estimated(self):
        assert TOOL_MODEL_TABLE_HEADERS[-1] == "Est. Cost (USD)"


class TestRenderBudgetLines:
    def test_no_lines_when_unavailable(self):
        assert render_budget_lines(None) == []

    def test_shows_spend_threshold_and_percent(self):
        lines = render_budget_lines((Decimal("12.34"), Decimal("100")))
        assert "$12.34" in lines[0]
        assert "$100.00" in lines[0]
        assert "12%" in lines[0]

    def test_renders_meter(self):
        lines = render_budget_lines((Decimal("50"), Decimal("100")))
        assert len(lines) == 2
        assert "█" in lines[1]
        assert "░" in lines[1]

    def test_zero_threshold_omits_percent_and_meter(self):
        lines = render_budget_lines((Decimal("5"), Decimal("0")))
        assert lines == [f"{label('Budget spend:')} {value('$5.00')}"]

    def test_spend_over_threshold_clamps_meter(self):
        lines = render_budget_lines((Decimal("250"), Decimal("100")))
        assert "250%" in lines[0]
        assert "░" not in lines[1]

    def test_thousands_separator(self):
        lines = render_budget_lines((Decimal("1234.5"), Decimal("10000")))
        assert "$1,234.50" in lines[0]
        assert "$10,000.00" in lines[0]


class TestRenderUsageSummary:
    def _make_record(self, days_ago: int, tool: str, tokens: int) -> dict:
        return {
            "tool": tool,
            "usage_day": date.today() - timedelta(days=days_ago),
            "total_tokens_used": tokens,
        }

    def test_contains_requester_name(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "alice@example.com", {"claude": "Claude Code"})
        assert "alice@example.com" in result

    def test_today_total(self):
        records = [self._make_record(0, "claude", 5000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "5.0K" in result
        assert "Databricks AI Gateway Token Usage" in result
        assert "┃ Today" in result
        assert "Last 7 days" in result
        assert "Last 30 days" in result
        assert "│ [cyan]5.0K tokens[/cyan]" in result

    def test_weekly_total_includes_past_week(self):
        records = [
            self._make_record(0, "claude", 1000),
            self._make_record(3, "claude", 2000),
            self._make_record(USAGE_BREAKDOWN_DAYS, "claude", 9999),  # outside window
        ]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        # only 3K from the last 7 days; 9999 from day 7 (boundary) may vary
        assert "3.0K" in result or "3" in result

    def test_active_tools_listed(self):
        records = [
            self._make_record(0, "claude", 1000),
            self._make_record(0, "codex", 1000),
        ]
        result = render_usage_summary(records, "user", {"claude": "Claude Code", "codex": "Codex"})
        assert "🤖  [bold]Top Harnesses:[/bold] [cyan]Claude Code · Codex[/cyan]" in result

    def test_top_models_listed(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 5000,
                "model_tokens": '[{"model":"databricks-claude-sonnet-4","tokens":5000}]',
            }
        ]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "claude-sonnet-4" in result

    def test_estimated_cost_label(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 1_000_000,
                "model_tokens": (
                    '[{"model":"databricks-claude-sonnet-4","tokens":1000000,'
                    '"input":1000000,"output":0}]'
                ),
            }
        ]
        result = render_usage_summary(
            records,
            "user",
            {"claude": "Claude Code"},
            price_lookup={"claudesonnet4": ModelPrice(Decimal("4.12"), None, None)},
        )
        assert "Estimated Cost for Last 7 Days:" in result
        assert "$4.12" in result

    def test_includes_budget_spend_when_available(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(
            records,
            "user",
            {"claude": "Claude Code"},
            budget_spend=(Decimal("12.34"), Decimal("100")),
        )
        assert "$12.34 of $100.00" in result

    def test_omits_budget_spend_by_default(self):
        records = [self._make_record(0, "claude", 1000)]
        result = render_usage_summary(records, "user", {"claude": "Claude Code"})
        assert "Budget spend" not in result

    def test_top_models_uses_per_model_token_totals(self):
        records = [
            {
                "tool": "claude",
                "usage_day": date.today(),
                "total_tokens_used": 237000,
                "models": "databricks-claude-haiku-4.5, databricks-claude-opus-4",
                "model_tokens": (
                    '[{"model":"databricks-claude-haiku-4.5", "tokens":920}, '
                    '{"model":"databricks-claude-opus-4", "tokens":236080}]'
                ),
            },
            {
                "tool": "codex",
                "usage_day": date.today(),
                "total_tokens_used": 13300,
                "models": "databricks-gpt-5",
                "model_tokens": '[{"model":"databricks-gpt-5", "tokens":13300}]',
            },
        ]
        result = render_usage_summary(
            records,
            "user",
            {"claude": "Claude Code", "codex": "Codex"},
        )
        # Top-models line is full names only, ranked by per-model token totals
        # (claude-opus-4 236.1K > gpt-5 13.3K > claude-haiku-4.5 920).
        assert "⭐  [bold]Top Models:[/bold]" in result
        assert "claude-opus-4 · gpt-5 · claude-haiku-4.5" in result
        # No token counts in this line — those live in the per-model table.
        assert "236.1K" not in result

    def test_empty_records(self):
        result = render_usage_summary([], "user", {"claude": "Claude Code"})
        assert "user" in result


class TestUsageCommand:
    def test_filters_to_configured_agents_and_skips_inactive_tables(self, monkeypatch):
        today = date.today()
        old_day = today - timedelta(days=USAGE_BREAKDOWN_DAYS)
        columns = [
            "requester_name",
            "tool",
            "usage_day",
            "total_tokens_used",
            "sessions",
            "model_tokens",
        ]
        rows = [
            (
                "user@example.com",
                "codex",
                today,
                100,
                1,
                '[{"model":"databricks-gpt-5","requests":1,"tokens":100,"input":80,"output":20}]',
            ),
            (
                "user@example.com",
                "claude",
                old_day,
                200,
                1,
                '[{"model":"databricks-claude-opus-4", "tokens":200}]',
            ),
            (
                "user@example.com",
                "gemini",
                today,
                900,
                1,
                '[{"model":"databricks-gemini-2.0-flash", "tokens":900}]',
            ),
        ]

        printed: list[str] = []
        headings: list[str] = []
        notes: list[str] = []
        rendered_tables: list[list[list[str]]] = []

        class DummyConsole:
            def print(self, value):
                printed.append(str(value))

        def fake_render_box_table(headers, table_rows, max_widths=None):
            rendered_tables.append(table_rows)
            return "TABLE"

        monkeypatch.setattr(
            usage_mod,
            "load_state",
            lambda: {"workspace": "https://workspace", "available_tools": ["claude", "codex"]},
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *args, **kwargs: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *args, **kwargs: "token")
        monkeypatch.setattr(
            usage_mod,
            "discover_sql_warehouses",
            lambda *args, **kwargs: [SqlWarehouse("/sql/1.0/warehouses/abc", "wh", "RUNNING")],
        )
        monkeypatch.setattr(usage_mod, "run_usage_query", lambda *args, **kwargs: (columns, rows))
        monkeypatch.setattr(
            usage_mod, "resolve_current_budget_spend", lambda *args, **kwargs: (None, "disabled")
        )
        monkeypatch.setattr(usage_mod, "prompt_yes_no_default", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            usage_mod, "prompt_for_selection", lambda prompt, options: options[0][0]
        )
        monkeypatch.setattr(
            usage_mod, "fetch_external_model_prices", lambda *args, **kwargs: ([], "disabled")
        )
        monkeypatch.setattr(usage_mod, "console", DummyConsole())
        monkeypatch.setattr(usage_mod, "print_heading", headings.append)
        monkeypatch.setattr(usage_mod, "print_note", notes.append)
        monkeypatch.setattr(usage_mod, "render_box_table", fake_render_box_table)

        assert usage() == 0

        assert "Codex · Last 7 Days" in headings
        assert "Claude Code · Last 7 Days" in headings
        assert all("Gemini" not in heading for heading in headings)
        assert notes == [
            "Budget spend and threshold are unavailable. "
            "Please configure a coding agent budget policy first.",
            "Using SQL warehouse `wh` (RUNNING).",
            f"No usage for Claude Code in the last {USAGE_BREAKDOWN_DAYS} days.",
        ]
        assert len(rendered_tables) == 2
        assert rendered_tables[0] == [["100 tokens", "100 tokens", "300 tokens"]]
        # One per-model row for codex: full model name "gpt-5", 1 request, 80 input / 20 output.
        assert rendered_tables[1][0][0] == "gpt-5"
        assert rendered_tables[1][0][1] == "1"
        assert rendered_tables[1][0][2] == "80"
        assert "gemini" not in "\n".join(printed).lower()
        assert "900" not in "\n".join(printed)

    def test_queries_only_the_selected_warehouse(self, monkeypatch):
        queried_paths: list[str] = []
        candidates = [
            SqlWarehouse("/sql/1.0/warehouses/first", "First", "RUNNING"),
            SqlWarehouse("/sql/1.0/warehouses/second", "Second", "STOPPED"),
        ]

        monkeypatch.setattr(
            usage_mod,
            "load_state",
            lambda: {"workspace": "https://workspace", "available_tools": []},
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *args, **kwargs: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *args, **kwargs: "token")
        monkeypatch.setattr(
            usage_mod,
            "resolve_current_budget_spend",
            lambda *args, **kwargs: ((Decimal("1"), Decimal("10")), None),
        )
        monkeypatch.setattr(usage_mod, "prompt_yes_no_default", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            usage_mod, "discover_sql_warehouses", lambda *args, **kwargs: candidates
        )

        def choose_second(prompt, options):
            assert options == [
                (candidates[0].http_path, "First (RUNNING)"),
                (candidates[1].http_path, "Second (STOPPED)"),
            ]
            return candidates[1].http_path

        monkeypatch.setattr(usage_mod, "prompt_for_selection", choose_second)

        def fake_query(workspace, http_path, token, query, on_connected=None):
            queried_paths.append(http_path)
            return ["requester_name"], [("user@example.com",)]

        monkeypatch.setattr(usage_mod, "run_usage_query", fake_query)
        monkeypatch.setattr(
            usage_mod, "fetch_external_model_prices", lambda *args, **kwargs: ([], None)
        )
        monkeypatch.setattr(usage_mod, "print_note", lambda *args: None)
        monkeypatch.setattr(usage_mod, "console", type("C", (), {"print": lambda *args: None})())

        assert usage() == 0
        assert queried_paths == [candidates[1].http_path]

    def test_shows_budget_before_prompt_and_skips_sql_when_declined(self, monkeypatch):
        events: list[str] = []

        class DummyConsole:
            def print(self, output):
                events.append(str(output))

        monkeypatch.setattr(
            usage_mod,
            "load_state",
            lambda: {"workspace": "https://workspace", "available_tools": ["codex"]},
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *args, **kwargs: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *args, **kwargs: "token")
        monkeypatch.setattr(
            usage_mod,
            "resolve_current_budget_spend",
            lambda *args, **kwargs: ((Decimal("12.34"), Decimal("100")), None),
        )
        monkeypatch.setattr(usage_mod, "console", DummyConsole())

        def decline(prompt, *, default):
            assert "$12.34 of $100.00" in "\n".join(events)
            assert "SQL warehouse" in prompt
            assert default is False
            return False

        monkeypatch.setattr(usage_mod, "prompt_yes_no_default", decline)
        monkeypatch.setattr(
            usage_mod,
            "discover_sql_warehouses",
            lambda *args, **kwargs: pytest.fail("SQL discovery should not run"),
        )
        monkeypatch.setattr(
            usage_mod,
            "fetch_external_model_prices",
            lambda *args, **kwargs: pytest.fail("price lookup should not run"),
        )

        assert usage() == 0


class TestSelectSqlWarehouse:
    def test_returns_selected_candidate(self, monkeypatch):
        candidates = [
            SqlWarehouse("/sql/1.0/warehouses/a", "Alpha", "RUNNING"),
            SqlWarehouse("/sql/1.0/warehouses/b", "Beta", "STOPPED"),
        ]
        monkeypatch.setattr(
            usage_mod, "prompt_for_selection", lambda prompt, options: candidates[1].http_path
        )
        assert select_sql_warehouse(candidates) == candidates[1]

    def test_returns_none_when_cancelled(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "prompt_for_selection", lambda prompt, options: None)
        assert select_sql_warehouse([SqlWarehouse("/path", "Warehouse", "RUNNING")]) is None


class TestRunQueryOnFirstWorkingWarehouse:
    _COLUMNS = ["requester_name"]
    _ROWS = [("user@example.com",)]

    def _warehouses(self, *labels: str) -> list[SqlWarehouse]:
        return [SqlWarehouse(f"/sql/1.0/warehouses/{label}", label, "RUNNING") for label in labels]

    def test_returns_first_working_warehouse(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(
            usage_mod, "run_usage_query", lambda *a, **k: (self._COLUMNS, self._ROWS)
        )
        http_path, columns, rows = run_query_on_first_working_warehouse(
            "https://ws", "token", self._warehouses("a", "b"), "SELECT 1"
        )
        assert http_path == "/sql/1.0/warehouses/a"
        assert (columns, rows) == (self._COLUMNS, self._ROWS)

    def test_falls_through_to_next_warehouse(self, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "print_warning", warnings.append)
        attempted: list[str] = []

        def flaky(workspace, http_path, token, query, on_connected=None):
            attempted.append(http_path)
            if http_path.endswith("dead"):
                raise RuntimeError("ENDPOINT_NOT_FOUND")
            return self._COLUMNS, self._ROWS

        monkeypatch.setattr(usage_mod, "run_usage_query", flaky)
        http_path, _, _ = run_query_on_first_working_warehouse(
            "https://ws", "token", self._warehouses("dead", "alive"), "SELECT 1"
        )
        assert http_path == "/sql/1.0/warehouses/alive"
        assert attempted == ["/sql/1.0/warehouses/dead", "/sql/1.0/warehouses/alive"]
        assert len(warnings) == 1
        assert "dead" in warnings[0]

    def test_raises_last_error_when_all_fail(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "print_warning", lambda *a: None)

        def always_fail(workspace, http_path, token, query, on_connected=None):
            raise RuntimeError(f"boom {http_path[-1]}")

        monkeypatch.setattr(usage_mod, "run_usage_query", always_fail)
        with pytest.raises(RuntimeError, match="boom b"):
            run_query_on_first_working_warehouse(
                "https://ws", "token", self._warehouses("a", "b"), "SELECT 1"
            )

    def test_raises_when_no_candidates(self, monkeypatch):
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        with pytest.raises(RuntimeError, match="No SQL warehouse could run"):
            run_query_on_first_working_warehouse("https://ws", "token", [], "SELECT 1")


class TestSelectAndRunUsageQuery:
    _COLUMNS = ["requester_name"]
    _ROWS = [("user@example.com",)]

    def _warehouses(self) -> list[SqlWarehouse]:
        return [
            SqlWarehouse("/sql/1.0/warehouses/dead", "Dead", "RUNNING"),
            SqlWarehouse("/sql/1.0/warehouses/alive", "Alive", "RUNNING"),
        ]

    def test_prompts_again_without_failed_warehouse(self, monkeypatch):
        candidates = self._warehouses()
        picker_options: list[list[str]] = []

        def choose(remaining):
            picker_options.append([warehouse.label for warehouse in remaining])
            return remaining[0]

        def query(workspace, http_path, token, query, on_connected=None):
            if http_path.endswith("dead"):
                raise RuntimeError("warehouse unavailable")
            return self._COLUMNS, self._ROWS

        monkeypatch.setattr(usage_mod, "select_sql_warehouse", choose)
        monkeypatch.setattr(usage_mod, "run_usage_query", query)
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "print_warning", lambda *a: None)

        result = select_and_run_usage_query("https://ws", "token", candidates, "SELECT 1")

        assert result == (candidates[1].http_path, self._COLUMNS, self._ROWS)
        assert picker_options == [["Dead", "Alive"], ["Alive"]]

    def test_can_cancel_after_a_warehouse_fails(self, monkeypatch):
        candidates = self._warehouses()
        choices = iter([candidates[0], None])
        monkeypatch.setattr(usage_mod, "select_sql_warehouse", lambda remaining: next(choices))
        monkeypatch.setattr(
            usage_mod,
            "run_usage_query",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("warehouse unavailable")),
        )
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "print_warning", lambda *a: None)

        assert select_and_run_usage_query("https://ws", "token", candidates, "SELECT 1") is None


class TestUsageWarehouseIdPassthrough:
    def test_forwards_warehouse_id_to_discovery(self, monkeypatch):
        captured = {}

        def fake_discover(workspace, token, *, warehouse_id=None):
            captured["warehouse_id"] = warehouse_id
            return [SqlWarehouse("/sql/1.0/warehouses/xyz", "xyz", "REQUESTED")]

        monkeypatch.setattr(
            usage_mod, "load_state", lambda: {"workspace": "https://ws", "available_tools": []}
        )
        monkeypatch.setattr(usage_mod, "ensure_databricks_auth", lambda *a, **k: None)
        monkeypatch.setattr(usage_mod, "get_databricks_token", lambda *a, **k: "token")
        monkeypatch.setattr(usage_mod, "discover_sql_warehouses", fake_discover)
        monkeypatch.setattr(usage_mod, "run_usage_query", lambda *a, **k: (["c"], []))
        monkeypatch.setattr(
            usage_mod, "resolve_current_budget_spend", lambda *a, **k: (None, "disabled")
        )
        monkeypatch.setattr(usage_mod, "prompt_yes_no_default", lambda *a, **k: True)
        monkeypatch.setattr(
            usage_mod, "fetch_external_model_prices", lambda *a, **k: ([], "disabled")
        )
        monkeypatch.setattr(usage_mod, "print_note", lambda *a: None)
        monkeypatch.setattr(usage_mod, "console", type("C", (), {"print": lambda *a: None})())

        assert usage(warehouse_id="xyz") == 0
        assert captured["warehouse_id"] == "xyz"


class TestQueryProgressMessage:
    def _messages(self, monkeypatch, state: str, connect: bool) -> list[str]:
        """Spinner messages rendered for a warehouse in `state`."""
        seen: list[str] = []

        @contextlib.contextmanager
        def fake_spinner(message):
            seen.append(message() if callable(message) else message)
            yield
            seen.append(message() if callable(message) else message)

        def fake_query(workspace, http_path, token, query, on_connected=None):
            if connect and on_connected is not None:
                on_connected()
            return ["c"], []

        monkeypatch.setattr(usage_mod, "spinner", fake_spinner)
        monkeypatch.setattr(usage_mod, "run_usage_query", fake_query)
        usage_mod._query_with_progress(
            "https://ws", "token", SqlWarehouse("/p", "wh", state), "SELECT 1"
        )
        return seen

    def test_running_shows_query_message(self, monkeypatch):
        assert self._messages(monkeypatch, "RUNNING", connect=True) == [
            usage_mod.QUERY_MESSAGE,
            usage_mod.QUERY_MESSAGE,
        ]

    def test_requested_shows_query_message(self, monkeypatch):
        # An explicit --warehouse-id; its real state was never looked up.
        assert self._messages(monkeypatch, "REQUESTED", connect=True)[0] == usage_mod.QUERY_MESSAGE

    def test_stopped_starts_with_startup_message(self, monkeypatch):
        assert self._messages(monkeypatch, "STOPPED", connect=False)[0] == usage_mod.STARTUP_MESSAGE

    def test_stopped_switches_to_query_once_connected(self, monkeypatch):
        seen = self._messages(monkeypatch, "STOPPED", connect=True)
        assert seen == [usage_mod.STARTUP_MESSAGE, usage_mod.QUERY_MESSAGE]
