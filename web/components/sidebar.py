"""Sidebar: stock input, LLM config, and history list."""

from __future__ import annotations

from datetime import date

import streamlit as st

from tradingagents.dataflows.a_stock import resolve_ticker
from web.history import get_history
from web.watchlist_store import (
    add_to_watchlist,
    is_in_watchlist,
    load_watchlist,
    remove_from_watchlist,
)


def _resolve_user_input(raw: str) -> tuple[str, str | None]:
    """Resolve raw user input to (ticker_code, error_msg).

    Accepts 6-digit codes or Chinese stock names (e.g. '宝光股份').
    Returns (code, None) on success or ("", error_msg) on failure.
    """
    from tradingagents.dataflows.a_stock import resolve_ticker

    try:
        code = resolve_ticker(raw)
        return code, None
    except ValueError as e:
        return "", str(e)


def _render_llm_config() -> None:
    """Render simplified LLM config with defaults from .env."""
    import os

    # Initialize session_state from .env on first load (only if not already set)
    for key, env_var, default in [
        ("deep_think_llm", "DEEP_THINK_LLM", "deepseek-v4-pro"),
        ("quick_think_llm", "QUICK_THINK_LLM", "deepseek-v4-flash"),
        ("llm_base_url", "BACKEND_URL", "https://api.deepseek.com/v1"),
        ("max_tokens", "MAX_TOKENS", "262144"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = os.getenv(env_var, default)

    st.caption("🔧 模型配置（可从 .env 读取默认值）")

    st.text_input(
        "深度思考模型",
        key="deep_think_llm",
        placeholder="例: deepseek-v4-pro",
        help=".env: DEEP_THINK_LLM",
    )

    st.text_input(
        "快速思考模型",
        key="quick_think_llm",
        placeholder="例: deepseek-v4-flash",
        help=".env: QUICK_THINK_LLM",
    )

    st.text_input(
        "API Base URL",
        key="llm_base_url",
        placeholder="https://api.deepseek.com/v1",
        help=".env: BACKEND_URL",
    )

    st.text_input(
        "Max Output Tokens",
        key="max_tokens",
        placeholder="262144",
        help="推理模型建议 256K+",
    )

    st.session_state["llm_provider"] = "openai_compatible"


def render_sidebar() -> None:
    """Render the sidebar with input controls and history."""

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1.5rem;">
            <span style="font-size:2rem; font-weight:800; color:#ff5a1f;">Trading</span><span style="font-size:2rem; font-weight:800; color:#f5f1eb;">Agents</span><span style="font-size:2rem; font-weight:800; color:#f5f1eb;">-</span><span style="font-size:2rem; font-weight:800; color:#ff5a1f;">Astock</span>
            <div style="font-size:0.85rem; color:#888; margin-top:0.2rem;">
                A股多Agent投研系统
            </div>
            <div style="font-size:0.7rem; color:#555; margin-top:0.3rem;">
                by <a href="https://github.com/simonlin1212" style="color:#ff5a1f; text-decoration:none;">simonlin1212</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 新建分析")

    # Ticker input + optional "加入自选" button
    ticker_col, add_col = st.columns([3, 1])
    with ticker_col:
        ticker = st.text_input(
            "股票代码",
            placeholder="例: 300750 或 宁德时代",
            key="input_ticker",
            help="输入6位A股代码或中文股票全称",
        )
    with add_col:
        # Show "加入自选" button if a valid ticker is entered and not yet in watchlist
        _raw_ticker = ticker.strip()
        _resolved_code: str | None = None
        if _raw_ticker:
            try:
                _resolved_code = resolve_ticker(_raw_ticker)
            except ValueError:
                pass
        if _resolved_code and not is_in_watchlist(_resolved_code):
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            if st.button("加入自选", key="add_watchlist_btn"):
                add_to_watchlist(_resolved_code)
                st.rerun()

    trade_date = st.date_input(
        "分析日期",
        value=date.today(),
        key="input_date",
    )

    with st.expander("⚙️ 模型配置", expanded=False):
        _render_llm_config()

    tracker = st.session_state.get("tracker")
    is_busy = tracker is not None and tracker.is_running

    if st.button(
        "开始分析" if not is_busy else "分析进行中...",
        use_container_width=True,
        disabled=is_busy or not ticker,
        type="primary",
    ):
        resolved_code, err = _resolve_user_input(ticker)
        if err:
            st.error(f"❌ {err}")
        else:
            if resolved_code != ticker.strip():
                st.success(f"✅ {ticker.strip()} → {resolved_code}")
            st.session_state["start_analysis"] = {
                "ticker": resolved_code,
                "trade_date": trade_date.strftime("%Y-%m-%d"),
            }
            st.session_state["viewing_history"] = None

    st.markdown("---")
    st.markdown("#### 自选股")

    watchlist = load_watchlist()
    if watchlist:
        for entry in watchlist:
            cols = st.columns([5, 1])
            with cols[0]:
                label = entry["ticker"]
                if entry.get("name"):
                    label += f" · {entry['name']}"
                if st.button(label, key=f"wl_{entry['ticker']}", use_container_width=True):
                    st.session_state["input_ticker"] = entry["ticker"]
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"wl_del_{entry['ticker']}"):
                    remove_from_watchlist(entry["ticker"])
                    st.rerun()
    else:
        st.caption("暂无自选股")

    st.markdown("---")
    st.markdown("#### 历史记录")

    history = get_history()
    if not history:
        st.caption("暂无历史记录")
        return

    for entry in history[:20]:
        t, d = entry["ticker"], entry["date"]
        label = f"{t}  ·  {d}"
        if st.button(label, key=f"hist_{t}_{d}", use_container_width=True):
            st.session_state["viewing_history"] = entry["path"]
            st.session_state["start_analysis"] = None

    st.markdown("---")
    st.caption("⚠️ 仅供学习研究，不构成投资建议")
