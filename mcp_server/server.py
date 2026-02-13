from fastmcp import FastMCP
from mcp_server.tools import mcp_tools
from config.settings import settings, TradingMode
from loguru import logger

mcp = FastMCP("Crypto Trading System")


@mcp.tool()
def analyze_market(symbol: str) -> dict:
    """Run multi-timeframe technical analysis on a crypto symbol (e.g. BTCUSDT).
    Uses current trading mode (swing/scalp) for mode-specific indicators."""
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
    Chart type depends on trading mode (4H swing / 5M scalp)."""
    logger.info(f"MCP tool: get_chart_image {symbol}")
    return mcp_tools.get_chart_image(symbol)


@mcp.tool()
def get_indicator_summary(symbol: str) -> dict:
    """Get multi-timeframe technical indicator summary (compact format)."""
    logger.info(f"MCP tool: get_indicator_summary {symbol}")
    return mcp_tools.get_indicator_summary(symbol)


@mcp.tool()
def switch_trading_mode(mode: str) -> dict:
    """Switch between 'swing' (long-term) and 'scalp' (short-term) trading mode.
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
        "analysis_interval_hours": settings.ANALYSIS_INTERVAL_HOURS,
    }


if __name__ == "__main__":
    # SSE transport for MCP communication
    mcp.run(transport="sse")
