# TradingAgents-Astock

## 项目概述
基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)（65K Stars）的 A 股深度特化 fork。多 Agent 投研框架，7 个 Analyst 角色通过 Bull/Bear 辩论 + 三方风险辩论生成投资报告。

- **仓库**: https://github.com/simonlin1212/TradingAgents-astock
- **协议**: Apache 2.0
- **Python**: >=3.10
- **当前版本**: 0.2.11

## 架构

### 数据层（v0.2.5 全部直连 HTTP，零第三方数据库依赖）
| 来源 | 协议 | 数据 |
|------|------|------|
| mootdx | TCP 7709 | OHLCV K线、财务快照、F10 文本 |
| 腾讯财经 | HTTP (qt.gtimg.cn) | PE/PB/市值/换手率 |
| 东方财富 datacenter | HTTP (datacenter-web) | 龙虎榜、限售解禁、板块行情 |
| 东方财富 push2/push2his | HTTP (push2.eastmoney) | 实时行情、个股信息、板块列表、资金流(分钟+日级) |
| 东方财富 np-weblist | HTTP | 滚动新闻 |
| 新浪财经 | HTTP (money.finance.sina) | K线历史、财报三表 |
| 同花顺 10jqka | HTTP | EPS 一致预期、热股题材 |
| 财联社 cls.cn | HTTP | 全球财经快讯 |
| 百度股市通 | HTTP (gushitong.baidu) | 概念板块归属（资金流已迁移至东财push2） |

### Agent 角色（7 个）
原版 4 个（市场/情绪/新闻/基本面）+ A 股特化 3 个（政策分析师/游资追踪/解禁监控）

### 关键路径
- `tradingagents/dataflows/a_stock.py` — A 股数据 vendor，所有数据获取入口
- `tradingagents/dataflows/utils.py` — `safe_ticker_component` 路径安全校验 + 中文 ticker 自动解析
- `tradingagents/agents/` — 7 个 Analyst + Bull/Bear 辩论逻辑
- `web/` — Streamlit Web UI
- `cli/` — CLI 入口

### 中文股票名解析链路
用户/LLM 输入 → `safe_ticker_component` 检测中文 → `resolve_ticker()` → `_build_name_code_map()`（mootdx 全市场映射，缓存）→ 返回 6 位代码

## 已知问题与注意事项

### 依赖冲突（v0.2.6 已缓解）
mootdx 锁死 httpx==0.25.2，与 langchain-google-genai 的 httpx>=0.28.1 冲突。v0.2.6 将 google-genai 移至可选依赖 `[google]`，`pip install -e .` 不再冲突。需要 Google 模型时 `pip install -e ".[google]"`。

### akshare 已移除（v0.2.5）
v0.2.5 起完全移除 akshare 依赖，所有数据通过直连 HTTP API 获取。

### 百度 PAE 资金流接口已下线（v0.2.7 已修复）
`fundsortlist` 和 `fundflow` 两个接口返回空（2026-05-19 确认）。v0.2.7 已替换为东财 push2 资金流 API。同时修复了 `RPT_ORGANIZATION_BUSSINESS`（改用席位筛选机构）和东财全球资讯 `req_trace` 参数。

### 东财接口防封限流（v0.2.11 新增，移植自 a-stock-data v3.2）
`a_stock.py` 里所有指向 `eastmoney.com` 的请求（push2 / push2his / datacenter-web / search-api / np-weblist 共 7 个调用点）统一走节流入口 `_em_get()`：模块级时间戳串行限流（默认间隔 `EM_MIN_INTERVAL=1.0s`，可用同名环境变量覆盖）+ 0.1~0.5s 随机抖动 + 复用 `requests.Session`（Keep-Alive）+ 默认 UA。多 Agent 跑批量分析不再触发东财临时封 IP。**仅东财限流**——mootdx(TCP) / 腾讯 / 新浪 / 同花顺 / 财联社 / 百度 等非东财源不受影响。批量场景可设 `EM_MIN_INTERVAL=1.5~2` 进一步降速。新增东财端点时务必走 `_em_get` 而非裸 `requests.get`。

### 模型兼容性
deepseek-v4-flash 等模型在 tool call 时可能返回中文股票名而非 6 位代码。`safe_ticker_component` 已加兜底自动转码，但不同模型表现仍有差异。

### 待处理 PR
- PR #18（hejingchi）：start_date 功能 + 主题切换 + Windows 字体。不建议直接 merge（与 v0.2.6 冲突），start_date 功能值得后续自行实现。

## Issue 归档
所有 GitHub Issue 的详细记录在 `issues/` 文件夹中，包含问题描述、根因分析、修复方案和当前状态。

## 开发规范
- 改动前先跑 `python -m pytest tests/ -v` 确保不破坏现有测试
- `safe_ticker_component` 是安全边界，任何绕过路径校验的改动必须慎重评估
- 数据层新增接口遵循 `tradingagents/dataflows/interface.py` 的 vendor 路由模式
- Web UI 改动在 `web/` 目录，用 `streamlit run web/launch.py` 本地测试

## StockAgent 集成（2026-06-05 ~ 2026-06-06）

本项目的第 8 个 Analyst —— **TradingFramework Analyst**，来自 StockAgents 的交易纪律框架。

### 新增/变更文件（10 个）
| 文件 | 变更 |
|------|------|
| `tradingagents/agents/analysts/trading_framework_analyst.py` | 新增：8 维度 A 股交易纪律分析 |
| `tradingagents/agents/analysts/social_media_analyst.py` | 用 smart_search_cli 替代 get_guba_sentiment / get_xueqiu_discussions |
| `tradingagents/agents/analysts/lockup_watcher.py` | 用 smart_search_cli 替代 get_insider_transactions |
| `tradingagents/agents/analysts/hot_money_tracker.py` | 用 smart_search_cli 替代 get_insider_transactions |
| `tradingagents/agents/utils/news_data_tools.py` | 新增 smart_search_cli @tool（subprocess 调 npm smart-search CLI）+ 所有工具调用日志 |
| `tradingagents/agents/utils/tool_log.py` | 新增：LangChain 回调写 tool_calls.jsonl |
| `tradingagents/graph/trading_graph.py` | 同步 ToolNode 配置 + max_tokens 支持 |
| `web/runner.py` | 集成 ToolLogCallbackHandler |
| `web/app.py` | API Key / max_tokens 从 .env 读取 |
| 其他 4 个 tool 文件 | _logged_vendor_call 包装所有 route_to_vendor |

### 关键教训
- ⚠️ **改工具必须同步两处**：analyst 的 `tools = [...]` 和 `trading_graph.py` 的 `ToolNode`。不同步 → LLM 调用工具但 ToolNode 无此工具 → 400 错误
- smart_search_cli 是 subprocess 调 npm 全局 `smart-search` CLI，PATH 查找顺序：`SMART_SEARCH_PATH` 环境变量 → `shutil.which` → `%APPDATA%/npm/smart-search.cmd`
- 每次 smart_search 调用 60-120s，合并多维度关键词到 1 次调用是关键优化
- 工具调用日志在 `~/.tradingagents/tool_calls.jsonl`

## 相关项目
- [a-stock-data](https://github.com/simonlin1212/a-stock-data) — A 股 MCP 数据服务（Claude Code 用的 skill）
- 上游 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) — 原版框架
- [StockAgents](https://github.com/...) — 本项目的 StockAgent 来源（狼大/图哥/科技主线坚守指南）
