"""Chart generator — structure-only overlays for VLM analysis.

Design principle:
  Charts show STRUCTURE (patterns, trendlines, levels) only.
  All indicator VALUES (RSI, MACD, OBV, etc.) go via text — VLM can't read numbers accurately.

What's drawn on the chart:
  - Candlesticks (price)
  - Pivot point markers (triangle up/down)
  - Diagonal trendlines (support/resistance)
  - Fibonacci retracement horizontal lines
  - Swing High/Low horizontal lines (liquidity levels)
  - Liquidation markers (optional, if data provided)

What's NOT drawn (sent as text instead):
  - RSI, MACD, ADX, BB values → text
  - OI, Funding, CVD values → text
  - All numeric indicators → text

Three modes:
  - DAY_TRADING: 15m candles, last ~8 hours
  - SWING: 4h candles, last ~10 days
  - POSITION: 1d candles, last ~90 days
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
import base64
from typing import Dict, List, Optional
from config.settings import settings, TradingMode
from loguru import logger
from scipy.signal import argrelextrema


class ChartGenerator:
    """Generate structure-focused candlestick charts for VLM analysis."""

    def __init__(self):
        self.width = settings.CHART_IMAGE_WIDTH
        self.height = settings.CHART_IMAGE_HEIGHT
        self.dpi = settings.CHART_IMAGE_DPI

    def generate_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str,
                       mode: TradingMode = TradingMode.SWING,
                       liquidation_df: Optional[pd.DataFrame] = None) -> Optional[bytes]:
        """Generate structure chart for any mode."""
        config = self._get_mode_config(mode)
        return self._generate_structure_chart(df, analysis, symbol, config, liquidation_df)

    def _get_mode_config(self, mode: TradingMode) -> Dict:
        """Per-mode chart configuration."""
        if mode == TradingMode.DAY_TRADING:
            return {
                'resample_rule': '15min',
                'tail_candles': 32,   # ~8 hours of 15m
                'min_candles': 10,
                'title_suffix': '15M DAY_TRADING',
                'fib_tf': '15m',
                'structure_tfs': ['15m', '1h'],
                'swing_tf': '1h',
            }
        elif mode == TradingMode.POSITION:
            return {
                'resample_rule': '1D',
                'tail_candles': 90,   # ~3 months
                'min_candles': 15,
                'title_suffix': '1D POSITION',
                'fib_tf': '1d',
                'structure_tfs': ['1d'],
                'swing_tf': '1d',
            }
        else:  # SWING (default)
            return {
                'resample_rule': '4h',
                'tail_candles': 60,   # ~10 days
                'min_candles': 10,
                'title_suffix': '4H SWING',
                'fib_tf': '4h',
                'structure_tfs': ['1h', '4h'],
                'swing_tf': '4h',
            }

    def _generate_structure_chart(self, df: pd.DataFrame, analysis: Dict,
                                   symbol: str, config: Dict,
                                   liquidation_df: Optional[pd.DataFrame] = None) -> Optional[bytes]:
        """Core chart generation — candlesticks + structure overlays only."""
        try:
            # Prepare OHLCV
            tmp = df.copy()
            tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], utc=True)
            tmp = tmp.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                tmp[col] = tmp[col].astype(float)

            chart_df = tmp.resample(config['resample_rule']).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().tail(config['tail_candles'])

            if len(chart_df) < config['min_candles']:
                return None

            chart_df.index.name = 'Date'

            # Style — clean, minimal
            mc = mpf.make_marketcolors(
                up='#4CAF50', down='#F44336',
                edge='inherit', wick='inherit',
                volume={'up': '#81C784', 'down': '#E57373'}
            )
            style = mpf.make_mpf_style(
                marketcolors=mc, gridstyle='-', gridcolor='#E0E0E0',
                facecolor='white', figcolor='white'
            )

            # Plot — candlesticks + volume only (NO indicator subplots)
            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                title=f'\n{symbol} {config["title_suffix"]}',
                figsize=(self.width / self.dpi, self.height / self.dpi),
                returnfig=True,
                panel_ratios=(5, 1),  # Large price panel, small volume
            )

            price_ax = axes[0]
            current_price = analysis.get('current_price', '?')
            price_ax.set_title(f'{symbol} | {current_price} | {config["title_suffix"]}',
                              fontsize=10, loc='left')

            # ── Overlay 1: Pivot Points (turning points) ──
            self._draw_pivot_points(price_ax, chart_df)

            # ── Overlay 2: Diagonal Trendlines (support/resistance) ──
            structure = analysis.get('structure', {}) or {}
            for tf in config['structure_tfs']:
                self._draw_diagonal_line(price_ax, chart_df,
                                         structure.get(f'support_{tf}'),
                                         'support_price', '#2E7D32')
                self._draw_diagonal_line(price_ax, chart_df,
                                         structure.get(f'resistance_{tf}'),
                                         'resistance_price', '#C62828')

            # ── Overlay 3: Fibonacci Levels ──
            fib = (analysis.get('fibonacci', {}) or {}).get(config['fib_tf'])
            if not fib:
                # Fallback to any available fib
                fib_data = analysis.get('fibonacci', {}) or {}
                for tf_key in ['4h', '1d', '1w', '15m']:
                    if tf_key in fib_data and fib_data[tf_key]:
                        fib = fib_data[tf_key]
                        break
            self._draw_fibonacci(price_ax, chart_df, fib)

            # ── Overlay 4: Swing High/Low (Liquidity Levels) ──
            swing = (analysis.get('swing_levels', {}) or {}).get(config['swing_tf'])
            self._draw_swing_levels(price_ax, chart_df, swing)

            # ── Overlay 5: Liquidation Markers ──
            if liquidation_df is not None and not liquidation_df.empty:
                self._draw_liquidation_markers(price_ax, chart_df, liquidation_df)

            # Save
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Chart error for {symbol} ({config['title_suffix']}): {e}")
            return None

    # ─────────────── Overlay Drawing Methods ───────────────

    def _draw_pivot_points(self, ax, chart_df: pd.DataFrame):
        """Draw turning point markers on pivot highs/lows."""
        try:
            close = chart_df['close'].astype(float).values
            if len(close) < 15:
                return

            order = max(3, len(close) // 20)
            mins = argrelextrema(close, np.less, order=order)[0]
            maxs = argrelextrema(close, np.greater, order=order)[0]

            if len(mins):
                ax.scatter(chart_df.index[mins], close[mins],
                          marker='^', color='#2E7D32', s=25, alpha=0.85, zorder=5)
            if len(maxs):
                ax.scatter(chart_df.index[maxs], close[maxs],
                          marker='v', color='#C62828', s=25, alpha=0.85, zorder=5)
        except Exception as e:
            logger.debug(f"Pivot points draw error: {e}")

    def _draw_diagonal_line(self, ax, chart_df: pd.DataFrame,
                             line_info: Optional[Dict], value_key: str, color: str):
        """Draw diagonal support/resistance trendline."""
        if not line_info:
            return
        try:
            slope = line_info.get('slope')
            current_value = line_info.get(value_key)
            if slope is None or current_value is None:
                return

            n = len(chart_df)
            x = np.arange(n)
            intercept = float(current_value) - float(slope) * (n - 1)
            y = float(slope) * x + intercept
            ax.plot(chart_df.index, y, color=color, linewidth=1.2,
                    linestyle='-.', alpha=0.9, zorder=3)
        except Exception:
            return

    def _draw_fibonacci(self, ax, chart_df: pd.DataFrame, fib: Optional[Dict]):
        """Draw Fibonacci retracement horizontal lines."""
        if not fib:
            return

        fib_styles = {
            'fib_236': ('#90CAF9', 0.5, '23.6%'),
            'fib_382': ('#64B5F6', 0.7, '38.2%'),
            'fib_500': ('#42A5F5', 0.8, '50%'),
            'fib_618': ('#1E88E5', 0.9, '61.8%'),
            'fib_786': ('#1565C0', 0.7, '78.6%'),
        }

        for key, (color, alpha, label) in fib_styles.items():
            val = fib.get(key)
            if isinstance(val, (int, float)):
                ax.axhline(val, color=color, linestyle='--', linewidth=0.9,
                           alpha=alpha, zorder=2)
                ax.text(chart_df.index[-1], val, f' {label}',
                        color=color, fontsize=7, va='center', ha='left')

    def _draw_swing_levels(self, ax, chart_df: pd.DataFrame, swing: Optional[Dict]):
        """Draw horizontal lines at swing highs/lows (liquidity pools)."""
        if not swing:
            return

        for high in swing.get('swing_highs', []):
            if isinstance(high, (int, float)):
                ax.axhline(high, color='#FF5722', linestyle=':', linewidth=0.7,
                           alpha=0.6, zorder=2)

        for low in swing.get('swing_lows', []):
            if isinstance(low, (int, float)):
                ax.axhline(low, color='#00BCD4', linestyle=':', linewidth=0.7,
                           alpha=0.6, zorder=2)

    def _draw_liquidation_markers(self, ax, chart_df: pd.DataFrame,
                                   liq_df: pd.DataFrame):
        """Draw liquidation event markers on the chart."""
        try:
            liq_df = liq_df.copy()
            liq_df['timestamp'] = pd.to_datetime(liq_df['timestamp'], utc=True)

            chart_start = chart_df.index[0]
            chart_end = chart_df.index[-1]
            mask = (liq_df['timestamp'] >= chart_start) & (liq_df['timestamp'] <= chart_end)
            visible = liq_df[mask]

            if visible.empty:
                return

            for _, row in visible.iterrows():
                total_liq = row.get('long_liq_usd', 0) + row.get('short_liq_usd', 0)
                if total_liq < 50000:  # Skip tiny liquidations
                    continue

                # Size proportional to liquidation amount
                size = min(100, max(15, total_liq / 100000 * 30))

                # Red = long liquidated (bearish), Blue = short liquidated (bullish)
                if row.get('long_liq_usd', 0) > row.get('short_liq_usd', 0):
                    color = '#F44336'  # Red (longs got rekt)
                else:
                    color = '#2196F3'  # Blue (shorts got rekt)

                # Find nearest chart timestamp
                nearest_idx = chart_df.index.get_indexer([row['timestamp']], method='nearest')[0]
                if 0 <= nearest_idx < len(chart_df):
                    price = chart_df.iloc[nearest_idx]['close']
                    ax.scatter(chart_df.index[nearest_idx], price,
                              marker='o', color=color, s=size, alpha=0.7,
                              edgecolors='black', linewidth=0.5, zorder=6)

        except Exception as e:
            logger.debug(f"Liquidation markers draw error: {e}")

    # ─────────────── Utility Methods ───────────────

    def chart_to_base64(self, chart_bytes: bytes) -> str:
        return base64.b64encode(chart_bytes).decode('utf-8')

    def resize_for_low_res(self, chart_bytes: bytes, max_dim: int = 512) -> bytes:
        """Resize to 512x512 for VLM (~1024 tokens)."""
        try:
            from PIL import Image
            img = Image.open(BytesIO(chart_bytes))
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format='PNG', optimize=True)
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Image resize error: {e}")
            return chart_bytes


chart_generator = ChartGenerator()
