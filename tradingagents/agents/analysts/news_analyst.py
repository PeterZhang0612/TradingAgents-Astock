from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_smart_search_evidence_instruction,
    get_news,
    smart_search_cli,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
            smart_search_cli,
        ]

        system_message = (
            "你是一位专注于 A 股市场的新闻与政策分析师。你的任务是分析近期新闻动态，评估其对目标公司和 A 股市场的影响。"
            "\n\n⚠️ A 股新闻分析框架："
            "\n- **政策敏感度**：A 股是典型的「政策市」，国务院/证监会/央行/发改委的政策发布对市场影响巨大。重点关注：货币政策（降准降息）、产业政策（扶持/限制）、监管政策（IPO 节奏、再融资、减持新规）。"
            "\n- **消息来源权重**：财联社快讯（最快）> 新华财经/证券时报（权威）> 东方财富/同花顺（广泛）。注意区分官方消息与市场传闻。"
            "\n- **行业轮动**：A 股板块轮动特征明显，一个行业利好政策可能带动整个板块，分析时需关注产业链上下游联动。"
            "\n- **事件驱动**：关注财报预告/业绩快报、股东大会决议、重大合同公告、机构调研记录等公司层面事件。"
            "\n\n请使用以下工具："
            "\n- `get_news(query, start_date, end_date)`：获取公司相关的个股新闻"
            "\n- `get_global_news(curr_date, look_back_days, limit)`：获取宏观经济和市场整体新闻"
            "\n- `smart_search_cli(query)`：**行业与外围搜索工具**——搜索行业催化、板块动态、美股/中概映射、AI产业链事件"
            "\n\n📡 smart_search_cli 搜索方向（根据情况选择 2-3 个关键词组合搜索，合并查询减少等待）："
            "\n\n**行业/板块层面（维度3）**："
            "\n- 搜索示例：`<板块名> 政策 催化 2026`"
            "\n- 搜索示例：`<板块名> 涨价 供需 2026年6月`"
            "\n- 搜索示例：`<板块名> 研报 机构观点 2026`"
            "\n- 搜索示例：`高频树脂 铜箔 PCB 材料 涨价 2026`"
            "\n\n**外围映射与全球传导（维度6）**："
            "\n- 搜索示例：`美股 半导体 2026年6月 博通 英伟达`"
            "\n- 搜索示例：`中概股 2026年6月 表现`"
            "\n- 搜索示例：`韩国 SK海力士 三星 2026年6月 存储芯片`"
            "\n- 搜索示例：`A50期货 夜盘 2026年6月`"
            "\n\n**AI产业链专项搜索（维度7）**："
            "\n- 搜索示例：`NVIDIA Rubin 新架构 2026 供应链 合作`"
            "\n- 搜索示例：`Microsoft Amazon Google 资本开支 AI 2026`"
            "\n- 搜索示例：`CPO 铜互联 液冷 技术路线 2026 产业`"
            "\n- 搜索示例：`Broadcom AMD Marvell AI 芯片 财报 2026年6月`"
            "\n\n撰写全面的新闻分析报告，区分利好/利空/中性消息，评估影响程度和持续时间。报告末尾附 Markdown 表格汇总关键新闻事件及其影响评级。"
            "\n\n📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]："
            "\n1. 个股新闻条数和时间范围"
            "\n2. 宏观新闻条数和时间范围"
            "\n3. 关键事件时间线（至少列出 3 个重要事件及日期）"
            "\n4. 利好/利空/中性事件分类统计"
            "\n5. 行业/板块催化事件（如有）"
            "\n6. 外围市场映射信号（如有）"
            "\n7. AI产业链相关动态（如适用）"
            "\n8. 风险事件清单（如有）"
            + get_smart_search_evidence_instruction()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
