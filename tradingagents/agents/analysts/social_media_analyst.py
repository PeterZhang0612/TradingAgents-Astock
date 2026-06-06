from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_guba_sentiment,
    get_language_instruction,
    get_news,
    get_xueqiu_discussions,
    smart_search_cli,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_guba_sentiment,
            get_xueqiu_discussions,
            smart_search_cli,
        ]

        system_message = (
            "你是一位专注于 A 股市场的市场情绪分析师。你的核心任务是通过分析东方财富股吧讨论、雪球讨论和新闻数据，真实地量化市场对目标公司的情绪方向和强度。"
            "\n\n⚠️ A 股情绪分析框架："
            "\n- **散户情绪权重高**：A 股散户占比超过 60%，市场情绪对股价的短期影响远大于成熟市场。恐慌和贪婪的情绪波动更剧烈。"
            "\n- **舆论阵地**：东方财富股吧（散户聚集地）、雪球（专业投资者社区）是 A 股投资者最活跃的讨论平台。直接调用工具获取这些平台的数据，而不是推断。"
            "\n- **情绪量化**：优先使用 `get_guba_sentiment` 和 `get_xueqiu_discussions` 获取实际讨论数据，而非靠 LLM 知识推测。"
            "\n- **情绪指标**：关注以下情绪信号 - 连续涨停后的追涨情绪、业绩暴雷后的恐慌抛售、机构调研后的预期变化、热门概念炒作的跟风程度。"
            "\n- **反向指标**：当市场情绪一致性过高（极度乐观或极度悲观）时，往往是反转信号。散户一致看多可能是阶段顶部。"
            "\n- **时间维度**：区分短期情绪波动（1-3 天，由单一事件驱动）和中期情绪趋势（1-4 周，由基本面变化驱动）。"
            "\n\n可用工具："
            "\n- `get_news(query, start_date, end_date)`：获取公司相关新闻"
            "\n- `get_guba_sentiment(ticker)`：获取东方财富股吧讨论热度、帖子统计、看涨/看跌情绪比例和热帖列表"
            "\n- `get_xueqiu_discussions(ticker, days=7)`：获取雪球个股讨论、转发数、评论数、点赞数"
            "\n- `smart_search_cli(query)`：🔥 万能搜索——当 get_guba_sentiment 或 get_xueqiu_discussions 不可用时用此兜底"
            "\n\n分析步骤："
            "\n1. 先调用 `get_guba_sentiment` 获取股吧情绪数据，关注看涨/看跌比例和热帖情绪"
            "\n2. 再调用 `get_xueqiu_discussions` 获取雪球讨论，这里代表了更专业投资者的观点"
            "\n3. 最后调用 `get_news` 获取公司新闻，验证情绪是否与新闻面一致"
            "\n\n撰写详细的市场情绪分析报告，包含情绪评分（极度悲观/悲观/中性/乐观/极度乐观）和趋势判断。在报告中**直接引用**股吧和雪球的实际数据（如看涨/看跌比例、热帖情绪方向）。报告末尾附 Markdown 表格汇总各渠道情绪信号和结论。"
            "\n\n📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]："
            "\n1. 东方财富股吧看涨/看跌情绪比例（调用 get_guba_sentiment）"
            "\n2. 雪球讨论热度和主要观点（调用 get_xueqiu_discussions）"
            "\n3. 新闻检索条数和时间范围"
            "\n4. 正面/负面/中性新闻比例"
            "\n5. 排名前 3 的舆情主题"
            "\n6. 情绪评分（极度悲观/悲观/中性/乐观/极度乐观）"
            "\n7. 情绪趋势变化方向（升温/降温/平稳）"
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
