import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class DivergenceEngine:
    """
    Narrative-Price Divergence Engine
    Calculates the mismatch between News Sentiment and actual Market Action (Price + Funding).
    """

    def __init__(self):
        self.DIVERGENCE_THRESHOLD = 50.0  # Min score to trigger contrarian signal

    def calculate_divergence(
        self, 
        news_sentiment: float, 
        price_strength: float, 
        funding_rate_skew: float
    ) -> Tuple[float, str]:
        """
        Calculate the divergence score.
        Args:
            news_sentiment: -10 (Extreme Bearish) to +10 (Extreme Bullish)
            price_strength: -10 (Breaking Support) to +10 (Breaking Resistance)
            funding_rate_skew: -10 (Heavy Short Bias) to +10 (Heavy Long Bias)

        Returns:
            Tuple[float, str]: (divergence_score, signal_type)
            Score range is 0 to 100.
            signal_type is 'NONE', 'CONTRARIAN_LONG', or 'CONTRARIAN_SHORT'.
        """
        score = 0.0
        signal_type = "NONE"

        # Case 1: Bad News, but Price is Strong and Retail is Shorting (CONTRARIAN LONG)
        if news_sentiment <= -3.0:
            if price_strength >= 0.0 and funding_rate_skew <= 0.0:
                # Severity of bad news * (Price resilience + Short crowding)
                raw_score = abs(news_sentiment) * (price_strength + abs(funding_rate_skew))
                # Normalize roughly to 0-100 (Max news=10, Max price=10, Max funding=10 -> 10*20=200)
                score = min(100.0, raw_score * 0.5)
                
                if score >= self.DIVERGENCE_THRESHOLD:
                    signal_type = "CONTRARIAN_LONG"
                    logger.info(f"DIVERGENCE ENGINE: Contrarian LONG triggered. Score: {score:.1f} (News:{news_sentiment}, Price:{price_strength}, Funding:{funding_rate_skew})")

        # Case 2: Good News, but Price is Weak and Retail is Longing (CONTRARIAN SHORT)
        elif news_sentiment >= 3.0:
            if price_strength <= 0.0 and funding_rate_skew >= 0.0:
                raw_score = news_sentiment * (abs(price_strength) + funding_rate_skew)
                score = min(100.0, raw_score * 0.5)
                
                if score >= self.DIVERGENCE_THRESHOLD:
                    signal_type = "CONTRARIAN_SHORT"
                    logger.info(f"DIVERGENCE ENGINE: Contrarian SHORT triggered. Score: {score:.1f} (News:{news_sentiment}, Price:{price_strength}, Funding:{funding_rate_skew})")

        return score, signal_type

    def normalize_inputs(self, snapshot: Dict) -> Tuple[float, float, float]:
        """
        Helper method to extract and normalize raw snapshot data into -10 to 10 ranges.
        """
        # 1. News Sentiment (-10 to 10)
        # Assuming the RAG / News summary outputs some narrative score. If not available, we estimate.
        # For now, we look for a 'narrative_sentiment' key. Default is 0.
        news_sentiment = snapshot.get("narrative_sentiment", 0.0)

        # 2. Price Strength (-10 to 10)
        # Using 4h/1d price change or distance to EMA
        p_change = snapshot.get("change_4h", 0.0)
        # 5% move = 10 score
        price_strength = max(-10.0, min(10.0, p_change * 2.0))

        # 3. Funding Skew (-10 to 10)
        # Normal funding is ~0.01%. Extreme is >0.05% or <-0.05%
        raw_funding = snapshot.get("funding_rate", 0.0)
        funding_skew = max(-10.0, min(10.0, raw_funding * 200.0))

        return news_sentiment, price_strength, funding_skew
