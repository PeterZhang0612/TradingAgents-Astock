from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news,
    get_guba_sentiment,
    get_xueqiu_discussions,
    smart_search_cli,
)
from tradingagents.agents.utils.signal_data_tools import (
    get_profit_forecast,
    get_hot_stocks,
    get_northbound_flow,
    get_concept_blocks,
    get_fund_flow,
    get_dragon_tiger_board,
    get_lockup_expiry,
    get_industry_comparison,
    get_margin_balance,
    get_shibor,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_smart_search_evidence_instruction() -> str:
    """Return shared evidence rules for agents that use smart_search_cli."""
    return (
        "\n\n📌 smart_search_cli 证据规范："
        "\n- 同一重要方向尽量用 2-3 组关键词交叉验证；突发事件优先搜索当日/近7日，基本面研报可放宽到近1-3个月。"
        "\n- 报告中引用 smart_search_cli 结果时必须写明来源 URL、发布时间/日期；缺少来源时标注 [数据缺失: 来源]。"
        "\n- 明确区分 新闻事实 / 公司公告 / 分析师观点 / 市场传闻，不要把观点或传闻当成事实。"
        "\n- 与结构化数据工具冲突时，说明冲突点，并降低该结论置信度。"
    )


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content="Continue")
        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
