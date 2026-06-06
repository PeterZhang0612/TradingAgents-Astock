"""自选股存储 — 基于 JSON 文件持久化。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

WATCHLIST_PATH = Path(__file__).resolve().parent / "watchlist.json"


def _load_raw() -> list[dict]:
    """Load raw JSON data from the watchlist file."""
    if not WATCHLIST_PATH.exists():
        return []
    try:
        with open(WATCHLIST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save(watchlist: list[dict]) -> None:
    """Persist watchlist to JSON file."""
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)


def load_watchlist() -> list[dict]:
    """加载自选股列表。

    每个条目：{ticker, name, added_date}
    """
    return _load_raw()


def _resolve_name_from_code(ticker: str) -> str:
    """Look up stock name from ticker code using the built-in name-code map."""
    from tradingagents.dataflows.a_stock import _build_name_code_map

    try:
        _, c2n = _build_name_code_map()
        return c2n.get(ticker, "")
    except Exception:
        return ""


def add_to_watchlist(ticker: str, name: str = "") -> list[dict]:
    """添加股票到自选。name 可选，为空时从股票名称映射自动获取。"""
    watchlist = _load_raw()

    # Avoid duplicates
    if any(item["ticker"] == ticker for item in watchlist):
        return watchlist

    # Resolve name if not provided
    resolved_name = name if name else _resolve_name_from_code(ticker)

    watchlist.append({
        "ticker": ticker,
        "name": resolved_name,
        "added_date": date.today().isoformat(),
    })

    _save(watchlist)
    return watchlist


def remove_from_watchlist(ticker: str) -> list[dict]:
    """从自选删除指定股票。"""
    watchlist = _load_raw()
    watchlist = [item for item in watchlist if item["ticker"] != ticker]
    _save(watchlist)
    return watchlist


def is_in_watchlist(ticker: str) -> bool:
    """检查指定股票是否已在自选。"""
    watchlist = _load_raw()
    return any(item["ticker"] == ticker for item in watchlist)
