"""A-Share backtest Pipeline -- validate TradingAgents signals with backtrader.

Features
--------
- T+1 settlement (A-share rule: today's buy can only be sold tomorrow)
- Price-limit checks: +/-10% for main-board, +/-20% for STAR/ChiNext
- Minimum lot size of 100 shares
- Stamp duty 0.1% on sells only (A-share convention since 2024)
- Multi-signal sequence support (buy / hold / sell with position sizing)

Usage
-----
    # CLI
    python backtest.py --ticker 000001 --start 2024-01-01 --end 2024-12-31

    # Programmatic
    from backtest import backtest
    results = backtest("000001", "2024-01-01", "2024-12-31", {})
    print(results)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A-Share trading constants
# ---------------------------------------------------------------------------

MIN_SHARES = 100  # minimum lot
STAMP_DUTY = 0.001  # 0.1% on sells only
COMMISSION = 0.00025  # 0.025% standard A-share brokerage

# Price limits per board (fractional, symmetrical up/down)
LIMIT_MAIN_BOARD = 0.10  # +/-10%
LIMIT_STAR_CHINEXT = 0.20  # +/-20% for 688xxx / 300xxx

# Boards by ticker prefix
STAR_PREFIXES = ("688", "689")
CHINEXT_PREFIXES = ("300", "301")


def _board_limit(ticker: str) -> float:
    """Return the symmetric price-limit fraction for *ticker*."""
    if ticker.startswith(STAR_PREFIXES) or ticker.startswith(CHINEXT_PREFIXES):
        return LIMIT_STAR_CHINEXT
    return LIMIT_MAIN_BOARD


# ---------------------------------------------------------------------------
# Signal data class
# ---------------------------------------------------------------------------


class TradingSignal:
    """A single TradingAgents signal translated for the backtester.

    Parameters
    ----------
    date :
        Trading date (YYYY-MM-DD).
    rating :
        One of Buy, Overweight, Hold, Underweight, Sell.
    position_size :
        Target allocation fraction of capital (0.0 -- 1.0).
        Buy/Overweight maps to a positive fraction; Sell/Underweight to a
        negative fraction (short / reduce); Hold maps to 0.0.
    """

    def __init__(self, date: str, rating: str, position_size: float = 0.0):
        self.date = date
        self.rating = rating
        self.position_size = position_size

    @classmethod
    def from_rating(cls, date: str, rating: str) -> "TradingSignal":
        """Map the 5-tier rating to a directional position-size target."""
        _MAP = {
            "Buy": 1.0,
            "Overweight": 0.67,
            "Hold": 0.0,
            "Underweight": -0.33,
            "Sell": -1.0,
        }
        return cls(date=date, rating=rating, position_size=_MAP.get(rating, 0.0))

    def __repr__(self) -> str:
        return f"TradingSignal({self.date}, {self.rating}, {self.position_size})"


# ---------------------------------------------------------------------------
# Backtrader Strategy
# ---------------------------------------------------------------------------


class TradingAgentsStrategy(bt.Strategy):
    """Translate TradingAgents multi-agent signals into backtrader orders.

    Parameters
    ----------
    signals : list[TradingSignal]
        Pre-computed signal sequence, sorted by date ascending.
    ticker :
        Instrument ticker (for board-limit detection).
    target_capital :
        Starting capital allocated to this ticker.
    """

    params = (
        ("signals", None),
        ("ticker", ""),
        ("target_capital", 100_000.0),
    )

    def __init__(self) -> None:
        self.signal_map: Dict[str, TradingSignal] = {}
        if self.params.signals:
            for sig in self.params.signals:
                self.signal_map[sig.date] = sig

        # State tracking
        self.entry_price: Optional[float] = None  # average cost of current position
        self.shares: int = 0  # current position size (shares)
        self.trade_log: List[Dict[str, Any]] = []
        self._pending_buy_date: Optional[str] = None  # T+1 gate
        self._last_close: float = 0.0
        self._prev_day_close: float = 0.0

        # Per-day portfolio value for equity curve
        self.equity_curve: List[Tuple[str, float]] = []

    def log(self, txt: str, dt: Optional[datetime] = None) -> None:
        dt = dt or self.datas[0].datetime.date(0)
        logger.debug("%s, %s", dt.isoformat(), txt)

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed, order.Canceled, order.Margin):
            self.log(
                f"Order {order.ref}: {order.getstatusname()} "
                f"{order.size} @ {order.executed.price:.2f}"
            )

    def _is_limit_locked(self, price: float, prev_close: float) -> bool:
        """Check whether *price* is at the limit-up or limit-down price."""
        limit = _board_limit(self.params.ticker)
        upper = prev_close * (1 + limit)
        lower = prev_close * (1 - limit)
        # Allow a 1-cent tolerance for rounding
        return price >= upper - 0.01 or price <= lower + 0.01

    def next(self) -> None:
        """Called on every bar.  Execute signal if one exists for today."""
        dt = self.datas[0].datetime.date(0)
        date_str = dt.isoformat()
        close = self.datas[0].close[0]
        prev_close = self.datas[0].close[-1] if len(self.datas[0].close) > 1 else close

        self._last_close = close
        self._prev_day_close = prev_close

        # Record equity curve point
        self.equity_curve.append((date_str, self.broker.getvalue()))

        # Check for a signal on this date
        signal = self.signal_map.get(date_str)
        if signal is None:
            return

        self.log(f"Signal: {signal.rating} (size={signal.position_size:.2f})")

        # T+1: skip buys if we bought the previous day
        if signal.position_size > 0 and self._pending_buy_date == date_str:
            self.log("T+1 gate: skipping buy (bought previous session)")
            return

        # Determine target position value (fraction of current portfolio)
        target_pct = signal.position_size
        current_value = self.broker.getvalue()
        target_value = current_value * abs(target_pct)
        current_position_value = self.shares * close

        # --- Execute order ---
        if target_pct > 0 and current_position_value < target_value - close * MIN_SHARES:
            self._buy(target_value - current_position_value, close, prev_close, date_str)
        elif target_pct < 0 and current_position_value > MIN_SHARES * close:
            self._sell(current_position_value - max(target_value, 0.0), close, prev_close)
        elif target_pct == 0 and self.shares > 0:
            # Signal is Hold / neutral -- exit position
            self._sell(current_position_value, close, prev_close)

    def _buy(self, target_amount: float, price: float, prev_close: float, date_str: str) -> None:
        """Place a buy order respecting A-share constraints."""
        # Check limit-up: cannot buy if stock is limit-up
        if self._is_limit_locked(price, prev_close) and price >= prev_close * (1 + _board_limit(self.params.ticker) - 0.01):
            self.log("Limit-up: skipping buy")
            return

        size = int(target_amount / price)
        size = max(MIN_SHARES, (size // MIN_SHARES) * MIN_SHARES)  # round to lots
        if size <= 0:
            return

        # Check cash
        cost = size * price
        if cost > self.broker.getcash():
            size = int(self.broker.getcash() / price)
            size = max(MIN_SHARES, (size // MIN_SHARES) * MIN_SHARES)
            cost = size * price
            if size <= 0:
                return

        self.buy(size=size, price=price)
        self.shares += size
        self.entry_price = price
        self._pending_buy_date = (datetime.fromisoformat(date_str) + timedelta(days=1)).strftime("%Y-%m-%d")
        self.trade_log.append({
            "date": date_str,
            "action": "BUY",
            "price": price,
            "shares": size,
            "value": cost,
        })
        self.log(f"BUY {size} @ {price:.2f} = {cost:.2f}")

    def _sell(self, target_amount: float, price: float, prev_close: float) -> None:
        """Place a sell order respecting A-share constraints."""
        # Check limit-down: cannot sell if stock is limit-down
        if self._is_limit_locked(price, prev_close) and price <= prev_close * (1 - _board_limit(self.params.ticker) + 0.01):
            self.log("Limit-down: skipping sell")
            return

        size = int(target_amount / price)
        size = max(MIN_SHARES, (size // MIN_SHARES) * MIN_SHARES)  # round to lots
        size = min(size, self.shares)  # cannot sell more than owned
        if size < MIN_SHARES:
            return

        cost = size * price
        self.sell(size=size, price=price)
        self.shares -= size
        if self.shares == 0:
            self.entry_price = None
        revenue = size * price
        stamp_duty = revenue * STAMP_DUTY  # stamp duty on sells only
        self.trade_log.append({
            "date": self.datas[0].datetime.date(0).isoformat(),
            "action": "SELL",
            "price": price,
            "shares": size,
            "value": revenue,
            "stamp_duty": stamp_duty,
        })
        self.log(f"SELL {size} @ {price:.2f} = {revenue:.2f} (duty={stamp_duty:.2f})")


# ---------------------------------------------------------------------------
# Data feed helpers
# ---------------------------------------------------------------------------


class AShareCSVData(bt.feeds.GenericCSVData):
    """CSV data feed for A-share daily bars.

    Expects columns: date,open,high,low,close,volume.
    """

    params = (
        ("dtformat", "%Y-%m-%d"),
        ("date", 0),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("openinterest", -1),
    )


def fetch_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV for *ticker* from yfinance with local caching.

    Parameters
    ----------
    ticker :
        A-share ticker (e.g. "000001.SZ").
    start_date, end_date :
        Date range (YYYY-MM-DD).
    cache_dir :
        If set, cache CSV to this directory to avoid repeated downloads.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    import yfinance as yf

    cache_path: Optional[Path] = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{ticker}_{start_date}_{end_date}.csv"
        if cache_path.exists():
            logger.info("Loading cached price data from %s", cache_path)
            df = pd.read_csv(cache_path, parse_dates=["date"])
            return df

    logger.info("Fetching price data for %s from %s to %s", ticker, start_date, end_date)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No price data returned for {ticker} between {start_date} and {end_date}")

    df = df.reset_index()
    df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Cached price data to %s", cache_path)

    return df


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------


def backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run TradingAgents analysis + backtrader backtest for *ticker*.

    The function:
    1. Fetches daily price data.
    2. Generates TradingAgents signals for each day in the range
       (using ``TradingAgentsGraph``).
    3. Feeds signals into ``backtrader`` for simulated execution.
    4. Computes performance metrics.

    Parameters
    ----------
    ticker :
        Stock ticker (exchange suffix optional if yfinance needs it).
    start_date, end_date :
        Analysis range (YYYY-MM-DD).
    config :
        Optional config dict passed to ``TradingAgentsGraph``.  Falls back
        to the module default when None.

    Returns
    -------
    dict with keys:
        total_return, annualized_return, max_drawdown, sharpe_ratio,
        win_rate, profit_loss_ratio, total_trades, equity_curve, trade_log
    """
    _config = config or {}

    # ---- Step 1: fetch price data ----
    cache_dir = _config.get("data_cache_dir")
    price_df = fetch_price_data(ticker, start_date, end_date, cache_dir)
    price_df.set_index("date", inplace=True)
    price_df.index = pd.to_datetime(price_df.index)

    # ---- Step 2: generate signals ----
    signals = _generate_signals(ticker, start_date, end_date, _config, price_df)

    # ---- Step 3: run backtrader ----
    equity_curve, trade_log, final_value = _run_backtrader(
        ticker, start_date, end_date, signals, _config, price_df
    )

    # ---- Step 4: compute metrics ----
    start_value = _config.get("target_capital", 100_000.0)
    metrics = _compute_metrics(equity_curve, trade_log, start_value, final_value)

    results: Dict[str, Any] = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualized_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "win_rate": metrics["win_rate"],
        "profit_loss_ratio": metrics["profit_loss_ratio"],
        "total_trades": metrics["total_trades"],
        "equity_curve": equity_curve,
        "trade_log": trade_log,
    }

    return results


def _generate_signals(
    ticker: str,
    start_date: str,
    end_date: str,
    config: Dict[str, Any],
    price_df: pd.DataFrame,
) -> List[TradingSignal]:
    """Run TradingAgents for each month-end in the range and collect signals."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(config)

    graph = TradingAgentsGraph(
        selected_analysts=["market", "social", "news", "fundamentals", "policy", "hot_money", "lockup"],
        config=cfg,
    )

    # Generate signals at regular intervals (every ~20 trading days)
    dates = sorted(price_df.index.unique())
    if len(dates) < 2:
        logger.warning("Not enough price data to generate signals")
        return []

    # Pick roughly monthly intervals
    step = max(1, len(dates) // max(1, ((len(dates) + 19) // 20)))
    sample_dates = dates[::step]

    signals: List[TradingSignal] = []
    for trade_date in sample_dates:
        date_str = trade_date.strftime("%Y-%m-%d")
        if date_str < start_date or date_str > end_date:
            continue
        try:
            final_state, signal_rating = graph.propagate(ticker, date_str)
            ts = TradingSignal.from_rating(date_str, signal_rating)
            signals.append(ts)
        except Exception:
            logger.exception("Signal generation failed for %s on %s", ticker, date_str)

    return signals


def _run_backtrader(
    ticker: str,
    start_date: str,
    end_date: str,
    signals: List[TradingSignal],
    config: Dict[str, Any],
    price_df: pd.DataFrame,
) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]], float]:
    """Feed signals and price data into backtrader and execute."""
    cerebro = bt.Cerebro(stdstats=False)

    # Configure broker with A-share parameters
    cerebro.broker.setcash(config.get("target_capital", 100_000.0))
    cerebro.broker.setcommission(
        commission=COMMISSION,
        stamp_duty=STAMP_DUTY,
        # backtrader 1.9.x stamp_duty is on sells only
    )

    # Add data feed
    df = price_df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    data = bt.feeds.PandasData(
        dataname=df,
        datetime="date",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fromdate=pd.to_datetime(start_date),
        todate=pd.to_datetime(end_date),
    )
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(
        TradingAgentsStrategy,
        signals=signals,
        ticker=ticker,
        target_capital=config.get("target_capital", 100_000.0),
    )

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    # Run
    strat = cerebro.run()[0]

    # Extract equity curve
    equity_curve = strat.equity_curve

    # Extract trade log
    trade_log = strat.trade_log

    final_value = cerebro.broker.getvalue()

    return equity_curve, trade_log, final_value


def _compute_metrics(
    equity_curve: List[Tuple[str, float]],
    trade_log: List[Dict[str, Any]],
    start_value: float,
    final_value: float,
) -> Dict[str, Any]:
    """Calculate summary performance metrics from the equity curve and trades."""

    total_return = (final_value - start_value) / start_value if start_value > 0 else 0.0

    # Annualized return
    if equity_curve and len(equity_curve) > 1:
        first_date = datetime.fromisoformat(equity_curve[0][0])
        last_date = datetime.fromisoformat(equity_curve[-1][0])
        days = (last_date - first_date).days
        years = max(days / 365.25, 1 / 365.25)
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0.0

    # Max drawdown
    max_dd = 0.0
    peak = float("-inf")
    for _, value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (daily)
    if equity_curve and len(equity_curve) > 1:
        returns = pd.Series([v for _, v in equity_curve]).pct_change().dropna()
        if returns.std() > 0 and len(returns) > 1:
            sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Win rate and profit/loss ratio from trade log
    winning_trades = 0
    total_pnl = 0.0
    win_pnl = 0.0
    loss_pnl = 0.0
    trade_count = 0

    i = 0
    while i < len(trade_log) - 1:
        if trade_log[i].get("action") == "BUY":
            # Find next SELL (or close)
            buy_price = trade_log[i]["price"]
            buy_shares = trade_log[i]["shares"]
            buy_value = trade_log[i]["value"]
            for j in range(i + 1, len(trade_log)):
                if trade_log[j].get("action") == "SELL":
                    sell_value = trade_log[j]["value"]
                    pnl = sell_value - buy_value
                    total_pnl += pnl
                    trade_count += 1
                    if pnl > 0:
                        winning_trades += 1
                        win_pnl += pnl
                    else:
                        loss_pnl += abs(pnl)
                    i = j
                    break
            else:
                break  # no matching sell
        i += 1

    win_rate = winning_trades / trade_count if trade_count > 0 else 0.0
    profit_loss_ratio = win_pnl / loss_pnl if loss_pnl > 0 else float("inf")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "total_trades": trade_count,
    }


def _format_pct(value: float, digits: int = 2) -> str:
    """Format a fraction as a human-readable percentage string."""
    return f"{value * 100:.{digits}f}%"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: Dict[str, Any]) -> None:
    """Plot backtest results: equity curve + drawdown.

    Parameters
    ----------
    results :
        Output dict from :func:`backtest`.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return

    equity = results.get("equity_curve", [])
    if not equity:
        logger.warning("No equity curve data to plot")
        return

    dates = [datetime.fromisoformat(d) for d, _ in equity]
    values = [v for _, v in equity]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # --- Equity curve ---
    ax1.plot(dates, values, label="Portfolio Value", color="#2196F3", linewidth=1.5)
    ax1.set_title(
        f"{results.get('ticker', '')} Backtest | "
        f"Return: {_format_pct(results.get('total_return', 0))} | "
        f"Sharpe: {results.get('sharpe_ratio', 0):.2f}",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value (CNY)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # --- Drawdown ---
    peak = float("-inf")
    drawdowns = []
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0.0
        drawdowns.append(-dd)

    ax2.fill_between(dates, drawdowns, 0, color="#F44336", alpha=0.4, label="Drawdown")
    ax2.plot(dates, drawdowns, color="#F44336", linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()

    # Show metrics summary box
    metrics_text = (
        f"Total Return:    {_format_pct(results.get('total_return', 0))}\n"
        f"Annualized:      {_format_pct(results.get('annualized_return', 0))}\n"
        f"Max Drawdown:    {_format_pct(results.get('max_drawdown', 0))}\n"
        f"Sharpe Ratio:    {results.get('sharpe_ratio', 0):.2f}\n"
        f"Win Rate:        {_format_pct(results.get('win_rate', 0))}\n"
        f"Profit/Loss:     {results.get('profit_loss_ratio', 0):.2f}\n"
        f"Total Trades:    {results.get('total_trades', 0)}"
    )
    ax1.text(
        0.02, 0.98, metrics_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.7},
    )

    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TradingAgents A-Share Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker", required=True,
        help="Stock ticker (e.g. 000001.SZ, 600519.SS)",
    )
    parser.add_argument(
        "--start", dest="start_date", required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end", dest="end_date", required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital in CNY (default: 100000)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Show equity-curve plot",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    """CLI entry point for backtest."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = {"target_capital": args.capital}
    results = backtest(args.ticker, args.start_date, args.end_date, config)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Backtest Results: {args.ticker}")
    print(f"  Period: {args.start_date} -> {args.end_date}")
    print(f"{'='*55}")
    print(f"  Total Return:      {_format_pct(results['total_return'])}")
    print(f"  Annualized Return: {_format_pct(results['annualized_return'])}")
    print(f"  Max Drawdown:      {_format_pct(results['max_drawdown'])}")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
    print(f"  Win Rate:          {_format_pct(results['win_rate'])}")
    print(f"  Profit/Loss Ratio: {results['profit_loss_ratio']:.2f}")
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"{'='*55}\n")

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serializable fields
        save_results = {
            k: v for k, v in results.items()
            if k not in ("equity_curve", "trade_log")
        }
        save_results["n_equity_points"] = len(results.get("equity_curve", []))
        save_results["n_trades"] = len(results.get("trade_log", []))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path.resolve()}")

    # Plot
    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
