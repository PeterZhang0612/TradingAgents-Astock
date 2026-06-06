"""Static prompt-contract checks for A-stock agent integration.

These tests avoid importing LangChain so they can run in minimal environments.
They guard the integration contracts that are easy to break when prompts and
ToolNodes are edited separately.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _call_names(source: str) -> set[str]:
    module = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_smart_search_response_preserves_source_appendix():
    source = _source("tradingagents/agents/utils/news_data_tools.py")

    assert "def _format_smart_search_response" in source
    assert "primary_sources" in source
    assert "extra_sources" in source
    assert "Sources" in source
    assert "_format_smart_search_response" in _call_names(source)


def test_trading_framework_reads_upstream_analyst_reports():
    source = _source("tradingagents/agents/analysts/trading_framework_analyst.py")

    for report_key in (
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "policy_report",
        "hot_money_report",
        "lockup_report",
    ):
        assert report_key in source

    assert "prior_reports_context" in source


def test_decision_agents_consume_trading_framework_report():
    trader_source = _source("tradingagents/agents/trader/trader.py")
    pm_source = _source("tradingagents/agents/managers/portfolio_manager.py")

    assert "trading_framework_report" in trader_source
    assert "Trading Framework / Discipline Report" in trader_source
    assert "trading_framework_report" in pm_source
    assert "Trading Framework / Discipline Report" in pm_source


def test_decision_schemas_expose_trading_discipline_fields():
    source = _source("tradingagents/agents/schemas.py")

    for field_name in (
        "probability_assessment",
        "risk_first_summary",
        "invalidation_condition",
        "validity_horizon",
        "reverse_case",
        "stop_loss_plan",
    ):
        assert field_name in source


def test_social_media_analyst_observes_nga_bigtime():
    source = _source("tradingagents/agents/analysts/social_media_analyst.py")

    assert "NGA" in source
    assert "大时代" in source
    assert "https://bbs.nga.cn/thread.php?fid=706" in source
    assert "fid=706" in source
