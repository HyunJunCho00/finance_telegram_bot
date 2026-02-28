from fastmcp import FastMCP
from mcp_server.tools import mcp_tools
from config.settings import settings, TradingMode
from loguru import logger

mcp = FastMCP("Crypto Trading System")


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
    logger.info(f"MCP tool: search_narrative {symbol}")
    return mcp_tools.search_market_narrative(symbol)


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
    """Execute a trade on Binance futures. Use with caution."""
    logger.info(f"MCP tool: execute_trade {side} {amount} {symbol} {leverage}x")
    from executors.trade_executor import trade_executor
    return trade_executor.execute(symbol=symbol, side=side, amount=amount, leverage=leverage)


@mcp.tool()
def get_chart_image(symbol: str) -> dict:
    """Generate and return a base64-encoded chart image for a symbol.
    Chart type depends on trading mode (4h swing / 1d position)."""
    logger.info(f"MCP tool: get_chart_image {symbol}")
    return mcp_tools.get_chart_image(symbol)


@mcp.tool()
def get_indicator_summary(symbol: str) -> dict:
    """Get multi-timeframe technical indicator summary (compact format)."""
    logger.info(f"MCP tool: get_indicator_summary {symbol}")
    return mcp_tools.get_indicator_summary(symbol)


@mcp.tool()
def switch_trading_mode(mode: str) -> dict:
    """Switch between 'swing', and 'position' trading modes.
    This changes which timeframes and indicators are used for analysis."""
    logger.info(f"MCP tool: switch_trading_mode to {mode}")
    return mcp_tools.switch_mode(mode)


@mcp.tool()
def get_trading_mode() -> dict:
    """Get the current trading mode and its configuration."""
    return {
        "mode": settings.trading_mode.value,
        "candle_limit": settings.candle_limit,
        "chart_enabled": settings.should_use_chart,
        "analysis_interval_hours": settings.analysis_interval_hours,
    }


@mcp.tool()
def get_feedback_history(limit: int = 5) -> dict:
    """Get past trading mistakes and lessons learned from self-correction loop."""
    logger.info(f"MCP tool: get_feedback_history limit={limit}")
    return mcp_tools.get_feedback_history(limit=limit)


if __name__ == "__main__":
    # SSE transport (2026 SOTA): Explicit host/port for VM network access
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
