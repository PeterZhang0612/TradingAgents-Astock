from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_smart_search_evidence_instruction,
    get_news,
    smart_search_cli,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            smart_search_cli,
        ]

        system_message = (
            "你是一位专注于 A 股市场的市场情绪分析师。你的核心任务是通过分析东方财富股吧讨论、雪球讨论、NGA 大时代帖子和新闻数据，真实地量化市场对目标公司的情绪方向和强度。"
            "\n\n⚠️ A 股情绪分析框架："
            "\n- **散户情绪权重高**：A 股散户占比超过 60%，市场情绪对股价的短期影响远大于成熟市场。恐慌和贪婪的情绪波动更剧烈。"
            "\n- **舆论阵地**：东方财富股吧（散户聚集地）、雪球（专业投资者社区）、NGA 大时代（高活跃散户/主题讨论社区，https://bbs.nga.cn/thread.php?fid=706）是 A 股投资者重要讨论平台。直接调用工具获取这些平台的数据，而不是推断。"
            "\n- **情绪量化**：使用 `smart_search_cli` 搜索股吧、雪球和 NGA 大时代的真实讨论数据，引用搜索结果而非靠 LLM 知识推测。"
            "\n- **情绪指标**：关注以下情绪信号 - 连续涨停后的追涨情绪、业绩暴雷后的恐慌抛售、机构调研后的预期变化、热门概念炒作的跟风程度。"
            "\n- **反向指标**：当市场情绪一致性过高（极度乐观或极度悲观）时，往往是反转信号。散户一致看多可能是阶段顶部。"
            "\n- **时间维度**：区分短期情绪波动（1-3 天，由单一事件驱动）和中期情绪趋势（1-4 周，由基本面变化驱动）。"
            "\n\n可用工具："
            "\n- `get_news(query, start_date, end_date)`：获取公司相关新闻"
            "\n- `smart_search_cli(query)`：**主力情绪工具**——搜索股吧散户、雪球专业投资者、NGA 大时代（fid=706）、X/Twitter 全球、微博热搜"
            "\n\n📡 smart_search_cli 搜索方向（合并查询 2-3 个为一组减少等待）："
            "\n\n**情绪与舆情层面（维度5）**："
            "\n- 搜索示例：`<股票名> X Twitter 讨论 2026`"
            "\n- 搜索示例：`<股票名/板块> 股吧 雪球 情绪`"
            "\n- 搜索示例：`site:bbs.nga.cn/thread.php?fid=706 <股票名/板块> 大时代 A股 情绪`"
            "\n- 搜索示例：`https://bbs.nga.cn/thread.php?fid=706 <股票名/板块> NGA 大时代 讨论`"
            "\n- 搜索示例：`A股 市场情绪 恐慌 2026年6月`"
            "\n- 搜索示例：`<股票名> 股吧 散户 雪球 NGA 大时代 投资者 X Twitter 讨论 情绪 新闻 舆论 2026年6月`"
            "\n\n分析步骤（smart_search 每次 60-120s，合并查询减少等待）："
            "\n1. 用 smart_search_cli 搜索 '<股票名> 股吧 散户 雪球 NGA 大时代 投资者 X Twitter 讨论 情绪 新闻 舆论 2026年6月'"
            "\n2. 用 smart_search_cli 搜索 'site:bbs.nga.cn/thread.php?fid=706 <股票名/板块> 大时代 A股 情绪'，观察 NGA 大时代相关帖子；如无法获取，标注 [数据缺失: NGA 大时代]。"
            "\n3. 调用 get_news 获取公司新闻，验证情绪一致性"
            "\n\n撰写详细的市场情绪分析报告，包含情绪评分（极度悲观/悲观/中性/乐观/极度乐观）和趋势判断。在报告中**直接引用**股吧、雪球和 NGA 大时代的实际数据（如看涨/看跌比例、热帖情绪方向、主题讨论方向）。报告末尾附 Markdown 表格汇总各渠道情绪信号和结论。"
            "\n\n📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]："
            "\n1. 股吧散户 + 雪球投资者情绪（用 smart_search_cli 合并搜索）"
            "\n2. NGA 大时代帖子观察（https://bbs.nga.cn/thread.php?fid=706）：相关帖子/主题方向/主流分歧，无法获取时标注 [数据缺失: NGA 大时代]"
            "\n3. X/Twitter 全球投资者观点 + 新闻舆论方向（用 smart_search_cli 合并搜索）"
            "\n4. 新闻检索条数和时间范围"
            "\n5. 正面/负面/中性新闻比例"
            "\n6. 排名前 3 的舆情主题"
            "\n7. 情绪评分（极度悲观/悲观/中性/乐观/极度乐观）"
            "\n8. 情绪趋势变化方向（升温/降温/平稳）"
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
