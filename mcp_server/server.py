from fastmcp import FastMCP
from mcp_server.tools import mcp_tools
from config.settings import settings
from loguru import logger

mcp = FastMCP("Crypto Trading System")


def _direct_trade_blocked(reason: str) -> dict:
    logger.warning(f"MCP direct trade blocked: {reason}")
    return {"success": False, "error": reason}


@mcp.tool()
def analyze_market(symbol: str) -> dict:
    """Run multi-timeframe technical analysis on a crypto symbol (e.g. BTCUSDT).
    Uses current trading mode (swing/position) for mode-specific indicators."""
    logger.info(f"MCP tool: analyze_market {symbol} (mode={settings.trading_mode.value})")
    return mcp_tools.get_market_analysis(symbol)


@mcp.tool()
def get_news_summary(hours: int = 4) -> dict:
    """Get aggregated news from Telegram channels for the last N hours."""
    logger.info(f"MCP tool: get_news_summary {hours}h")
    return mcp_tools.get_telegram_summary(hours)


@mcp.tool()
def get_funding_info(symbol: str) -> dict:
    """Get Binance funding rate, OI, long/short ratio with context analysis."""
    logger.info(f"MCP tool: get_funding_info {symbol}")
    return mcp_tools.get_funding_data(symbol)


@mcp.tool()
def get_global_oi(symbol: str) -> dict:
    """Get Global Open Interest breakdown from 3 exchanges (Binance+Bybit+OKX).
    Returns total OI in USD and per-exchange breakdown."""
    logger.info(f"MCP tool: get_global_oi {symbol}")
    return mcp_tools.get_global_oi(symbol)


@mcp.tool()
def get_cvd(symbol: str, minutes: int = 240) -> dict:
    """Get CVD (Cumulative Volume Delta) showing net buying/selling pressure.
    Taker Buy vs Taker Sell volume from Binance Futures klines.
    Default 240 minutes = 4 hours."""
    logger.info(f"MCP tool: get_cvd {symbol} {minutes}min")
    return mcp_tools.get_cvd_data(symbol, limit=minutes)


@mcp.tool()
def search_narrative(symbol: str) -> dict:
    """Search market narrative via Perplexity API.
    Returns WHY the price is moving: sentiment, bullish/bearish factors, macro context."""
    logger.warning(f"MCP tool: search_narrative blocked for {symbol}")
    return {
        "success": False,
        "error": "Public MCP search_narrative is disabled. Use the Telegram natural-language tool path instead.",
    }


@mcp.tool()
def query_knowledge_graph(query: str, mode: str = "hybrid") -> dict:
    """Query the LightRAG knowledge graph for relationship context.
    Modes: 'local' (entity facts), 'global' (market themes), 'hybrid' (both).
    Example: query_knowledge_graph('Bitcoin BTC', 'hybrid')"""
    logger.info(f"MCP tool: query_knowledge_graph '{query}' mode={mode}")
    return mcp_tools.query_rag(query, mode=mode)


@mcp.tool()
def get_latest_trading_report() -> dict:
    """Get the most recent AI trading decision report."""
    logger.info("MCP tool: get_latest_trading_report")
    return mcp_tools.get_latest_report()


@mcp.tool()
def get_current_position(symbol: str) -> dict:
    """Get the current trade position status for a symbol."""
    logger.info(f"MCP tool: get_current_position {symbol}")
    return mcp_tools.get_position_status(symbol)


@mcp.tool()
def execute_trade(symbol: str, side: str, amount: float, leverage: int = 1) -> dict:
    """Execute a trade on Binance futures. Disabled by default for safety."""
    if not settings.ENABLE_DIRECT_MCP_TRADING:
        return _direct_trade_blocked(
            "Direct MCP trading is disabled. Use the orchestrator or Telegram-controlled workflow."
        )
    if not settings.PAPER_TRADING_MODE:
        return _direct_trade_blocked("Direct MCP trading is restricted to paper mode.")
    logger.info(f"MCP tool: execute_trade {side} {amount} {symbol} {leverage}x")
    from executors.trade_executor import trade_executor
    return trade_executor.execute(symbol=symbol, side=side, amount=amount, leverage=leverage)


@mcp.tool()
def get_chart_image(symbol: str, lane: str = "swing") -> dict:
    """Generate chart image by fixed lane only: 'swing' or 'position'."""
    logger.info(f"MCP tool: get_chart_image {symbol} (lane={lane})")
    return mcp_tools.get_chart_image(symbol, lane=lane)


@mcp.tool()
def get_chart_images(symbol: str, lane: str = "swing") -> dict:
    """Generate split human-readable chart images for a lane."""
    logger.info(f"MCP tool: get_chart_images {symbol} (lane={lane})")
    return mcp_tools.get_chart_images(symbol, lane=lane)


@mcp.tool()
def get_indicator_summary(symbol: str) -> dict:
    """Get multi-timeframe technical indicator summary (compact format)."""
    logger.info(f"MCP tool: get_indicator_summary {symbol}")
    return mcp_tools.get_indicator_summary(symbol)


@mcp.tool()
def get_trading_mode() -> dict:
    """Get fixed dual-mode policy configuration."""
    return {
        "mode": "dual",
        "swing": {"venue": "binance_futures", "direction": "long_short"},
        "position": {"venue": "binance_spot_upbit", "direction": "long_only"},
        "chart_enabled": settings.should_use_chart,
        "primary_scheduler_utc": {
            "job_daily_precision": f"{int(getattr(settings, 'DAILY_PRECISION_HOUR_UTC', 0)):02d}:{int(getattr(settings, 'DAILY_PRECISION_MINUTE_UTC', 30)):02d}",
            "job_hourly_monitor": "hh:15",
            "job_routine_market_status": "hh:20",
        },
    }


@mcp.tool()
def get_feedback_history(limit: int = 5) -> dict:
    """Get past trading mistakes and lessons learned from self-correction loop."""
    logger.info(f"MCP tool: get_feedback_history limit={limit}")
    return mcp_tools.get_feedback_history(limit=limit)


if __name__ == "__main__":
    # SSE transport (2026 SOTA): Explicit host/port for VM network access
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
    #mcp.run(transport="sse", host="127.0.0.1", port=8001)
