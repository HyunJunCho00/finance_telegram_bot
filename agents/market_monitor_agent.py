import json
from agents.ai_router import ai_client
from loguru import logger
from datetime import datetime
from config.database import db

class MarketMonitorAgent:
    """Routine Market Status Agent (2026 Free-Tier Optimized).
    
    Provides periodic summaries of market indicators (Funding, CVD, OI) 
    using Groq/Workers AI to minimize costs.
    """

    def __init__(self):
        self.role = "market_monitor"

    def summarize_current_status(self, indicators: dict) -> str:
        """Summarizes market status using free-tier LLMs."""
        
        system_prompt = (
            "You are a Market Status Monitor. Your goal is to provide a concise, "
            "data-driven summary of current market indicators for a professional trader. "
            "Focus on spotting anomalies in Funding Rates, CVD, and Open Interest. "
            "Format the output in clean Markdown."
        )
        
        user_message = f"""
        Current Market Indicators (UTC: {datetime.now().isoformat()}):
        {json.dumps(indicators, indent=2)}
        
        Please provide:
        1. Quick Sentiment (Bullish/Bearish/Neutral)
        2. Notable anomalies in CVD or Funding
        3. Strategic outlook for the next 4 hours
        """

        try:
            # ai_client will automatically prioritize Groq/Workers AI for this role
            summary = ai_client.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                role="chat", # Use 'chat' role for lower-tier model targeting
                temperature=0.4
            )
            return summary
        except Exception as e:
            logger.error(f"MarketMonitorAgent failed: {e}")
            return "Failed to generate market summary."

market_monitor_agent = MarketMonitorAgent()
