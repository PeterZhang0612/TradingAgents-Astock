from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
import subprocess
import json
import os
import logging

logger = logging.getLogger(__name__)

# Resolve smart-search path once at import time
_SMART_SEARCH_CMD = None

def _find_smart_search():
    """Resolve the smart-search CLI command.

    Priority:
    1. SMART_SEARCH_PATH env var (explicit path for server deployment)
    2. shutil.which("smart-search") (system PATH)
    3. Windows npm global path (%APPDATA%/npm/smart-search.cmd)
    """
    import shutil
    # Explicit path from env (for server deployment)
    env_path = os.environ.get("SMART_SEARCH_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path
    # System PATH
    which = shutil.which("smart-search")
    if which:
        return which
    # Windows: npm global packages
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        candidate = os.path.join(appdata, "npm", "smart-search.cmd")
        if os.path.exists(candidate):
            return candidate
    return None

_SMART_SEARCH_CMD = _find_smart_search()

@tool
def smart_search_cli(
    query: Annotated[str, "Search query string. Use natural language or keywords. For best results, include relevant dates and keywords in Chinese or English"],
) -> str:
    """
    Search the web for real-time news, social sentiment, policy updates, and market information using the smart-search CLI.
    This is the PRIMARY tool for: social media sentiment (X/Twitter, 雪球, 股吧), breaking news,
    policy announcements, industry catalysts, lockup expiry news, insider trading rumors,
    and any information not available through structured data APIs.

    Use this when:
    - Checking social media sentiment and discussion trends
    - Searching for recent news about a company, sector, or policy
    - Finding lockup/restricted share expiry announcements
    - Looking for industry catalysts or supply chain updates
    - Any query that requires broad web search beyond structured data

    Args:
        query (str): Search query string. Include relevant dates, company names, or keywords.
    Returns:
        str: Search results with content, sources, and URLs
    """
    if not _SMART_SEARCH_CMD:
        return "smart-search CLI not found. Install with: npm install -g smart-search-cli"

    try:
        result = subprocess.run(
            [_SMART_SEARCH_CMD, "search", query, "--extra-sources", "2", "--timeout", "90", "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            content = data.get("content", result.stdout)
            if len(content) > 20000:
                content = content[:20000] + "\n...[truncated]..."
            return content
        else:
            try:
                result2 = subprocess.run(
                    [_SMART_SEARCH_CMD, "search", query, "--timeout", "60", "--format", "content"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
                output = result2.stdout.strip()
                if len(output) > 20000:
                    output = output[:20000] + "\n...[truncated]..."
                return output
            except Exception:
                return f"smart-search error: {result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return "smart-search timed out after 120s. Try a shorter or more specific query."
    except Exception as e:
        return f"smart-search error: {str(e)[:500]}"

@tool
def get_news(
    ticker: Annotated[str, "6-digit A-stock code (e.g. 600379). Must be numeric, NOT company name or Chinese text"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given stock code.
    Uses the configured news_data vendor.
    Args:
        ticker (str): 6-digit A-stock code, e.g. 600379, 300750. Must be the numeric code, not the company name.
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "6-digit A-stock code (e.g. 600379). Must be numeric, NOT company name"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): 6-digit A-stock code, e.g. 600379
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)


@tool
def get_guba_sentiment(
    ticker: Annotated[str, "6-digit A-stock code (e.g. 600519). Must be numeric, NOT company name"],
) -> str:
    """
    Retrieve guba.eastmoney.com discussion sentiment data for a stock.
    Shows total posts, new posts today, hot post titles, and bullish/bearish ratio.
    Uses the configured news_data vendor.
    Args:
        ticker (str): 6-digit A-stock code, e.g. 600519
    Returns:
        str: Guba discussion sentiment report
    """
    return route_to_vendor("get_guba_sentiment", ticker)


@tool
def get_xueqiu_discussions(
    ticker: Annotated[str, "6-digit A-stock code (e.g. 600519). Must be numeric, NOT company name"],
    days: Annotated[int, "Days to look back for discussions (default 7)"] = 7,
) -> str:
    """
    Retrieve xueqiu.com (Snowball) discussion posts for a stock.
    Shows recent user posts, likes, replies, and content summaries.
    Uses the configured news_data vendor.
    Args:
        ticker (str): 6-digit A-stock code, e.g. 600519
        days (int): Days to look back (default 7)
    Returns:
        str: Xueqiu discussion report
    """
    return route_to_vendor("get_xueqiu_discussions", ticker, days)
