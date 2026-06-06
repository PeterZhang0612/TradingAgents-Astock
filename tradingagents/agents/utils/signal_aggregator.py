"""Signal aggregator -- cross-run analysis of TradingAgents signals.

This module reads the append-only memory log and combines historical decisions
with market-price data to answer three questions:

1. **Aggregation** -- what signals did the system produce across runs?
2. **Win rate** -- how did those signals perform against actual price moves?
3. **LLM drift** -- how consistent is the LLM's analysis for the same stock
   across different sessions?

Usage
-----
    from tradingagents.agents.utils.signal_aggregator import (
        aggregate_signals, calculate_win_rate, llm_drift_report,
    )

    # Basic aggregation
    stats = aggregate_signals("000001")
    print(stats)

    # Win-rate validation against price data
    win = calculate_win_rate("000001", price_df, forward_days=5)
    print(win)

    # Drift analysis
    drift = llm_drift_report("000001")
    print(drift)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pattern to match the tag line: [YYYY-MM-DD | ticker | rating | ...]
_TAG_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2})\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|(.+?)\]$",
    re.MULTILINE,
)

# Pattern to extract individual analyst scores from the decision body
_ANALYST_SCORE_RE = re.compile(
    r"(Market|Social|News|Fundamentals|Policy|Hot[_\s]?Money|Lockup).*?"
    r"(?:score|rating|sentiment)[:\s]*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _load_memory_log(memory_log_path: str) -> List[Dict[str, Any]]:
    """Parse the memory log file into structured entries.

    Parameters
    ----------
    memory_log_path :
        Absolute path to ``trading_memory.md``.

    Returns
    -------
    List of entry dicts with keys: date, ticker, rating, raw_return,
    alpha_return, holding_days, decision, reflection, pending.
    """
    from pathlib import Path

    path = Path(memory_log_path)
    if not path.exists():
        logger.warning("Memory log not found at %s", memory_log_path)
        return []

    text = path.read_text(encoding="utf-8")
    entries: List[Dict[str, Any]] = []

    # Split on the separator comment that TradingMemoryLog uses
    for block in text.split("<!-- ENTRY_END -->"):
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        if not lines:
            continue

        tag_match = _TAG_RE.match(lines[0].strip())
        if not tag_match:
            continue

        date_str, ticker, rating, rest_field = tag_match.groups()

        # Parse the rest of the tag fields
        fields = [f.strip() for f in rest_field.split("|")]
        pending = fields[0] == "pending"
        raw = fields[0] if not pending and len(fields) > 0 else None
        alpha = fields[1] if not pending and len(fields) > 1 else None
        holding = fields[2] if not pending and len(fields) > 2 else None

        body = "\n".join(lines[1:]).strip()

        decision: str = ""
        reflection: str = ""
        decision_match = re.search(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", body, re.DOTALL)
        reflection_match = re.search(r"REFLECTION:\n(.*?)$", body, re.DOTALL)
        if decision_match:
            decision = decision_match.group(1).strip()
        if reflection_match:
            reflection = reflection_match.group(1).strip()

        entries.append({
            "date": date_str,
            "ticker": ticker,
            "rating": rating,
            "pending": pending,
            "raw_return": float(raw.replace("%", "")) / 100 if raw and "%" in raw else (float(raw) if raw and raw != "n/a" else None),
            "alpha_return": float(alpha.replace("%", "")) / 100 if alpha and "%" in alpha else (float(alpha) if alpha and alpha != "n/a" else None),
            "holding_days": int(holding.rstrip("d")) if holding and "d" in holding else None,
            "decision": decision,
            "reflection": reflection,
        })

    return entries


def _extract_analyst_scores(decision_text: str) -> Dict[str, float]:
    """Extract individual analyst scores from decision prose.

    Looks for patterns like "Market Analyst score: 85", "Social rating: 60",
    "Fundamentals sentiment: 0.75" in the decision body.

    Returns a dict of {analyst_name: score}.
    """
    scores: Dict[str, float] = {}
    for match in _ANALYST_SCORE_RE.finditer(decision_text):
        name = match.group(1).lower().replace(" ", "_").replace("__", "_")
        try:
            scores[name] = float(match.group(2))
        except ValueError:
            pass
    return scores


def _extract_model_info(decision_text: str) -> Optional[str]:
    """Extract the LLM model identifier from decision text if present."""
    m = re.search(r"(?:model|llm)[:\s]+(\S+)", decision_text, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_data_quality(decision_text: str) -> Optional[str]:
    """Extract a data-quality rating if present (e.g. ``quality: high``)."""
    m = re.search(
        r"(?:data[-\s]?quality|quality[-\s]?rating)[:\s]+(\w+)",
        decision_text,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_signals(ticker: str, memory_log_path: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate all historical decisions for *ticker* from the memory log.

    Parameters
    ----------
    ticker :
        Stock ticker to filter by.
    memory_log_path :
        Path to ``trading_memory.md``.  Falls back to the default location
        under ``~/.tradingagents/memory/`` when None.

    Returns
    -------
    dict with keys:
        - ticker
        - total_signals
        - rating_distribution: {rating: count}
        - rating_percentages: {rating: percentage}
        - signal_switches: number of rating changes between consecutive entries
        - first_date, last_date
        - pending_count
    """
    if memory_log_path is None:
        from pathlib import Path
        memory_log_path = str(Path.home() / ".tradingagents" / "memory" / "trading_memory.md")

    all_entries = _load_memory_log(memory_log_path)
    ticker_entries = [e for e in all_entries if e["ticker"] == ticker]

    if not ticker_entries:
        return {
            "ticker": ticker,
            "total_signals": 0,
            "rating_distribution": {},
            "rating_percentages": {},
            "signal_switches": 0,
            "first_date": None,
            "last_date": None,
            "pending_count": 0,
        }

    # Sort by date
    ticker_entries.sort(key=lambda e: e["date"])

    # Rating distribution
    resolved = [e for e in ticker_entries if not e["pending"]]
    ratings = [e["rating"] for e in resolved]
    rating_dist = dict(Counter(ratings))
    total = len(ratings)
    rating_pct = {
        k: round(v / total * 100, 1) if total > 0 else 0.0
        for k, v in rating_dist.items()
    }

    # Signal switches (consecutive rating changes)
    switches = 0
    for i in range(1, len(resolved)):
        if resolved[i]["rating"] != resolved[i - 1]["rating"]:
            switches += 1

    pending_count = sum(1 for e in ticker_entries if e["pending"])

    return {
        "ticker": ticker,
        "total_signals": len(ticker_entries),
        "rating_distribution": rating_dist,
        "rating_percentages": rating_pct,
        "signal_switches": switches,
        "first_date": ticker_entries[0]["date"] if ticker_entries else None,
        "last_date": ticker_entries[-1]["date"] if ticker_entries else None,
        "pending_count": pending_count,
    }


def calculate_win_rate(
    ticker: str,
    price_data: pd.DataFrame,
    forward_days: int = 5,
    memory_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate historical signals against subsequent price moves.

    For each Buy/Overweight signal, check whether the stock price increased
    over the following *forward_days*.  For Sell/Underweight, check whether
    it decreased.  Hold is omitted from win/loss accounting.

    Parameters
    ----------
    ticker :
        Stock ticker to analyse.
    price_data :
        DataFrame with a datetime index and a ``close`` column.  Must cover
        the entire signal period plus *forward_days* after the last signal.
    forward_days :
        Holding period in calendar days to evaluate price change.
    memory_log_path :
        Override path to the memory log.

    Returns
    -------
    dict with keys:
        - ticker, forward_days
        - total_valid_signals, wins, losses, win_rate
        - avg_win_return, avg_loss_return, profit_loss_ratio
        - per_rating: {rating: {wins, losses, total, avg_return}}
    """
    if memory_log_path is None:
        from pathlib import Path
        memory_log_path = str(Path.home() / ".tradingagents" / "memory" / "trading_memory.md")

    all_entries = _load_memory_log(memory_log_path)
    ticker_entries = [
        e for e in all_entries
        if e["ticker"] == ticker and not e["pending"]
    ]

    if not ticker_entries:
        return {
            "ticker": ticker,
            "forward_days": forward_days,
            "total_valid_signals": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win_return": 0.0,
            "avg_loss_return": 0.0,
            "profit_loss_ratio": 0.0,
            "per_rating": {},
        }

    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data = price_data.copy()
        price_data.index = pd.to_datetime(price_data.index)

    price_data = price_data.sort_index()
    close_series = price_data["close"]

    wins = 0
    losses = 0
    total_win_return = 0.0
    total_loss_return = 0.0
    per_rating: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total": 0, "total_return": 0.0}
    )

    # Map rating to expected direction: True = bullish (expect price up),
    # False = bearish (expect price down), None = neutral (skip)
    def _direction(rating: str) -> Optional[bool]:
        r = rating.lower()
        if r in ("buy", "overweight"):
            return True
        if r in ("sell", "underweight"):
            return False
        return None

    for entry in ticker_entries:
        direction = _direction(entry["rating"])
        if direction is None:
            continue

        entry_date = pd.Timestamp(entry["date"])
        if entry_date not in close_series.index:
            # Find the next available trading day
            idx = close_series.index.searchsorted(entry_date)
            if idx >= len(close_series):
                continue
            entry_date = close_series.index[idx]

        # Forward price
        end_idx = close_series.index.searchsorted(
            entry_date + timedelta(days=forward_days)
        )
        end_idx = min(end_idx, len(close_series) - 1)
        if end_idx <= close_series.index.get_loc(entry_date):
            continue  # not enough data

        entry_price = close_series.loc[entry_date]
        exit_price = close_series.iloc[end_idx]

        ret = (exit_price - entry_price) / entry_price

        rating_key = entry["rating"]
        per_rating[rating_key]["total"] += 1
        per_rating[rating_key]["total_return"] += ret

        if (direction and ret > 0) or (not direction and ret < 0):
            wins += 1
            total_win_return += abs(ret)
            per_rating[rating_key]["wins"] += 1
        else:
            losses += 1
            total_loss_return += abs(ret)
            per_rating[rating_key]["losses"] += 1

    total_valid = wins + losses
    win_rate = wins / total_valid if total_valid > 0 else 0.0
    avg_win = total_win_return / wins if wins > 0 else 0.0
    avg_loss = total_loss_return / losses if losses > 0 else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return {
        "ticker": ticker,
        "forward_days": forward_days,
        "total_valid_signals": total_valid,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win_return": avg_win,
        "avg_loss_return": avg_loss,
        "profit_loss_ratio": pl_ratio,
        "per_rating": dict(per_rating),
    }


def llm_drift_report(ticker: str, memory_log_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyse LLM analysis consistency for the same stock across dates.

    Detects:
    - Signal flip-flopping (Buy -> Sell -> Buy without fundamental change)
    - Analyst score volatility across sessions
    - Sudden rating changes that may indicate model instability

    Parameters
    ----------
    ticker :
        Stock ticker.
    memory_log_path :
        Override path to the memory log.

    Returns
    -------
    dict with keys:
        - ticker
        - total_entries
        - signal_flip_count: number of Buy<->Sell or Overweight<->Underweight flips
        - flip_rate: fraction of transitions that are flips
        - consecutive_same: max consecutive identical ratings
        - analyst_score_volatility: {analyst_name: std_dev_of_scores}
        - date_range: [first_date, last_date]
        - timeline: [{date, rating, analyst_scores}]
    """
    if memory_log_path is None:
        from pathlib import Path
        memory_log_path = str(Path.home() / ".tradingagents" / "memory" / "trading_memory.md")

    all_entries = _load_memory_log(memory_log_path)
    ticker_entries = [
        e for e in all_entries
        if e["ticker"] == ticker and not e["pending"]
    ]

    if not ticker_entries:
        return {
            "ticker": ticker,
            "total_entries": 0,
            "signal_flip_count": 0,
            "flip_rate": 0.0,
            "consecutive_same": 0,
            "analyst_score_volatility": {},
            "date_range": None,
            "timeline": [],
        }

    ticker_entries.sort(key=lambda e: e["date"])

    # Build polarity map for flip detection
    def _polarity(rating: str) -> int:
        """1 = bullish, -1 = bearish, 0 = neutral."""
        r = rating.lower()
        if r in ("buy", "overweight"):
            return 1
        if r in ("sell", "underweight"):
            return -1
        return 0

    # Flip detection
    flip_count = 0
    total_transitions = 0
    max_consecutive = 1
    current_streak = 1

    timeline: List[Dict[str, Any]] = []

    # Collect analyst scores per analyst across entries for volatility calc
    score_history: Dict[str, List[float]] = defaultdict(list)

    for i, entry in enumerate(ticker_entries):
        scores = _extract_analyst_scores(entry["decision"])
        model = _extract_model_info(entry["decision"])
        quality = _extract_data_quality(entry["decision"])

        timeline.append({
            "date": entry["date"],
            "rating": entry["rating"],
            "polarity": _polarity(entry["rating"]),
            "analyst_scores": scores,
            "model": model,
            "data_quality": quality,
        })

        for analyst_name, score in scores.items():
            score_history[analyst_name].append(score)

        if i > 0:
            total_transitions += 1
            prev_pol = _polarity(ticker_entries[i - 1]["rating"])
            curr_pol = _polarity(entry["rating"])

            # A flip is a polarity reversal (bull -> bear or bear -> bull)
            if prev_pol * curr_pol < 0:
                flip_count += 1

            # Consecutive same rating
            if entry["rating"] == ticker_entries[i - 1]["rating"]:
                current_streak += 1
                if current_streak > max_consecutive:
                    max_consecutive = current_streak
            else:
                current_streak = 1

    # Analyst score volatility (standard deviation)
    analyst_volatility: Dict[str, float] = {}
    for name, scores in score_history.items():
        if len(scores) > 1:
            import statistics
            analyst_volatility[name] = round(statistics.stdev(scores), 2)
        else:
            analyst_volatility[name] = 0.0

    flip_rate = flip_count / total_transitions if total_transitions > 0 else 0.0

    return {
        "ticker": ticker,
        "total_entries": len(ticker_entries),
        "signal_flip_count": flip_count,
        "flip_rate": round(flip_rate, 4),
        "consecutive_same": max_consecutive,
        "analyst_score_volatility": analyst_volatility,
        "date_range": [ticker_entries[0]["date"], ticker_entries[-1]["date"]],
        "timeline": timeline,
    }
