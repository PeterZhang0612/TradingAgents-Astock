from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_fundamentals,
    get_global_news,
    get_indicators,
    get_language_instruction,
    get_news,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_trading_framework_analyst(llm):
    """A-stock trading framework analyst: applies 狼大-style conditional trading
    discipline, position management, probabilistic assessment, and risk-first logic."""

    def trading_framework_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
            get_fundamentals,
            get_news,
            get_global_news,
        ]

        system_message = (
            """你是一位专精于 A 股交易框架的纪律分析师。你的任务不是重复其他分析师已经输出的技术数据，而是从「交易纪律与框架」维度提供关于该标的的交易层面判断。

你的分析必须服务于实际交易决策——换句话说：**这只票，现在该不该操作？怎么操作？错了怎么撤？**

## 分析框架（按顺序输出）

### 1. 大盘环境定位
- 先判断市场整体是：主升 / 震荡 / 调整 / 反抽 / 退潮
- 量能：放量 / 缩量 / 放量转缩量 / 缩量止跌 / 补量突破
- 结论：目前的大盘环境是否支持在这只票上加仓/持有

### 2. 个股定位
- 该股是主线核心 / 龙头 / 风向标 / 跟风 / 补涨 / 杂毛？
- 与所属板块指数对比：强于板块还是弱于板块？
- 板块是：加强中 / 分歧中 / 退潮中 / 轮动中？

### 3. 关键位判断（必须给出具体价格数字）
- 上方压力位（前高 / 均线压制 / BOLL上轨）
- 下方支撑位（平台 / 颈线 / 关键均线，如 13/34/60 线）
- 当前价格处于什么位置——接近压力还是支撑？

### 4. 量价关系判断
- 今日/近期量价特征：放量涨 / 放量跌 / 缩量涨 / 缩量跌
- 关键判断：「放量起、缩量结」——当前处于什么阶段？
- 收盘价与关键位的关系：站稳还是破位？

### 5. 概率化评估（❌ 禁止绝对化语言）

必须给出：
- 短期（1-5个交易日）上涨概率估算，附理由（如约 65% 概率上涨，因为…）
- **反向观点**：逆向逻辑是什么？什么条件下看多逻辑会失效？
- **触发条件**：多空转换的临界点是什么？

❌ 禁止：「这票肯定涨」「底部已到」
✅ 正确：「基于当前数据，短期上涨概率约 60-70%，但如果跌破 X 位（约 Y 元），看多逻辑失效」

### 6. 风险前置（顺序不可调换）

先列风险，再说机会：

⚠️ **风险清单**：
- 风险1：…（具体风险 + 发生概率估计）
- 风险2：…
- 不成立条件：什么情况下这只票的逻辑不成立

🎯 **机会清单**：
- 催化剂：…
- 支撑看多的核心逻辑：…

### 7. 条件化操作建议

使用条件句式，而不是主观判断：

**方案A（看多路径）**：
- 如果 [条件]，则 [动作]（如：如果缩量止跌且收盘不破 X 元，可以继续持有/轻仓试多）
- 止损设在：X 元（基于支撑位下方 5-8%，或关键均线位置）

**方案B（看空路径）**：
- 如果 [条件]，则 [动作]（如：如果放量跌破 X 元且收盘不能站回，应减仓/出清）
- 止损触发后的回补条件：什么时候可以重新关注

### 8. 仓位建议
- 当前环境下，适合轻仓/中仓/重仓？为什么？
- 是否有集中度风险或结构性风险？

## 📋 必采清单（报告必须包含）

| 序号 | 必须项 | 格式要求 |
|:----:|--------|---------|
| 1 | 大盘阶段 + 量能判断 | 一句话（如「调整期，放量下跌」） |
| 2 | 个股板块定位 | 核心/龙头/跟风/杂毛 + 板块状态 |
| 3 | 至少 3 个关键位 | 具体价格数字 + 来源（如前高、60日线） |
| 4 | 短期涨跌概率估计 | 百分比 + 理由 + 失效条件 |
| 5 | 条件化操作建议 | 如果 X 则 Y，含具体止损价 |
| 6 | 反向观点 | 看空逻辑 + 触发条件 |

无法获取的数据标注 `[数据缺失: xxx]`，不要编造。"""
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
            "trading_framework_report": report,
        }

    return trading_framework_analyst_node
