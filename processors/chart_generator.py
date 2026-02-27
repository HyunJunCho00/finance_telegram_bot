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
        if mode == TradingMode.POSITION:
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

            # 1. Prepare FULL resampled DF for calculations
            full_resampled = tmp.resample(config['resample_rule']).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

            if len(full_resampled) < config['min_candles']:
                return None

            # 2. Calculate Indicators on FULL history
            # Add EMA200 to full_resampled before slicing
            import pandas_ta as ta
            full_resampled['ema200'] = ta.ema(full_resampled['close'], length=min(200, len(full_resampled)-1))

            # 3. Slice for VISUAL window
            chart_df = full_resampled.tail(config['tail_candles']).copy()
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

            # Plot — candlesticks + volume only
            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                title=f'\n{symbol} {config["title_suffix"]}',
                figsize=(self.width / self.dpi, self.height / self.dpi),
                returnfig=True,
                panel_ratios=(5, 1),
            )

            price_ax = axes[0]
            current_price = analysis.get('current_price', '?')
            price_ax.set_title(f'{symbol} | {current_price} | {config["title_suffix"]}',
                              fontsize=10, loc='left')

            # ── Overlay 1: Pivot Points (turning points) ──
            self._draw_pivot_points(price_ax, chart_df)

            # ── Overlay 2: Market Structure Labels (HH/HL/LH/LL) ──
            self._draw_market_structure_labels(price_ax, chart_df)

            # ── Overlay 3: EMA200 line ──
            self._draw_ema200(price_ax, chart_df)

            # ── Overlay 4: Diagonal Trendlines (support/resistance) ──
            structure = analysis.get('structure', {}) or {}
            for tf in config['structure_tfs']:
                self._draw_diagonal_line(price_ax, chart_df,
                                         structure.get(f'support_{tf}'),
                                         'support_price', '#2E7D32')
                self._draw_diagonal_line(price_ax, chart_df,
                                         structure.get(f'resistance_{tf}'),
                                         'resistance_price', '#C62828')

            # ── Overlay 5: Fibonacci Levels ──
            fib = (analysis.get('fibonacci', {}) or {}).get(config['fib_tf'])
            if not fib:
                # Fallback to any available fib
                fib_data = analysis.get('fibonacci', {}) or {}
                for tf_key in ['4h', '1d', '1w', '15m']:
                    if tf_key in fib_data and fib_data[tf_key]:
                        fib = fib_data[tf_key]
                        break
            self._draw_fibonacci(price_ax, chart_df, fib)

            # ── Overlay 6: Swing High/Low (Liquidity Levels) ──
            swing = (analysis.get('swing_levels', {}) or {}).get(config['swing_tf'])
            self._draw_swing_levels(price_ax, chart_df, swing)

            # ── Overlay 7: Volume Profile Histogram (right side) ──
            self._draw_volume_profile_histogram(price_ax, chart_df)

            # ── Overlay 8: Liquidation Markers ──
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

    def _draw_ema200(self, ax, chart_df: pd.DataFrame):
        """Draw EMA200 line. Uses pre-calculated high-precision EMA from full history."""
        try:
            if 'ema200' not in chart_df.columns:
                return
            
            ema = chart_df['ema200']
            if ema is None or ema.isna().all():
                return
                
            ax.plot(chart_df.index, ema.values,
                    color='#FF9800', linewidth=1.5,
                    linestyle='-', alpha=0.85, zorder=4)
            
            last_val = ema.dropna().iloc[-1] if not ema.dropna().empty else None
            if last_val:
                ax.text(chart_df.index[-1], float(last_val),
                        ' EMA200 (6Y Context)', color='#FF9800',
                        fontsize=7, va='center', ha='left', fontweight='bold')
        except Exception as e:
            logger.debug(f"EMA200 draw error: {e}")

    def _draw_market_structure_labels(self, ax, chart_df: pd.DataFrame):
        """Label pivot highs/lows as HH, LH, HL, LL for market structure context."""
        try:
            high = chart_df['high'].astype(float).values
            low = chart_df['low'].astype(float).values
            if len(high) < 15:
                return

            order = max(3, len(high) // 20)
            min_idx = argrelextrema(low, np.less, order=order)[0]
            max_idx = argrelextrema(high, np.greater, order=order)[0]

            # Label swing highs: HH or LH
            for i, idx in enumerate(max_idx):
                if i == 0:
                    label = 'H'
                else:
                    prev = high[max_idx[i - 1]]
                    label = 'HH' if high[idx] > prev else 'LH'
                ax.annotate(label,
                            xy=(chart_df.index[idx], high[idx]),
                            xytext=(0, 7), textcoords='offset points',
                            fontsize=6, color='#C62828', ha='center',
                            va='bottom', fontweight='bold',
                            annotation_clip=True)

            # Label swing lows: HL or LL
            for i, idx in enumerate(min_idx):
                if i == 0:
                    label = 'L'
                else:
                    prev = low[min_idx[i - 1]]
                    label = 'HL' if low[idx] > prev else 'LL'
                ax.annotate(label,
                            xy=(chart_df.index[idx], low[idx]),
                            xytext=(0, -7), textcoords='offset points',
                            fontsize=6, color='#1565C0', ha='center',
                            va='top', fontweight='bold',
                            annotation_clip=True)
        except Exception as e:
            logger.debug(f"Market structure labels error: {e}")

    def _draw_volume_profile_histogram(self, ax, chart_df: pd.DataFrame, bins: int = 24):
        """Draw horizontal volume profile histogram on right side of price panel."""
        try:
            prices = chart_df['close'].astype(float).values
            volumes = chart_df['volume'].astype(float).values
            if len(prices) < 10:
                return

            price_min = float(prices.min())
            price_max = float(prices.max())
            if price_max <= price_min:
                return

            price_bins = np.linspace(price_min, price_max, bins + 1)
            vol_at_price = []
            for i in range(bins):
                mask = (prices >= price_bins[i]) & (prices < price_bins[i + 1])
                vol_at_price.append(float(volumes[mask].sum()))

            max_vol = max(vol_at_price) if max(vol_at_price) > 0 else 1
            vol_norm = [v / max_vol for v in vol_at_price]
            poc_idx = int(np.argmax(vol_at_price))
            threshold = float(np.percentile(vol_at_price, 70))

            # Inset axes on right side: [x0, y0, width, height] in axes fraction
            ax_vp = ax.inset_axes([0.86, 0.0, 0.14, 1.0])
            ax_vp.set_xlim(0, 1)
            ax_vp.set_ylim(price_min, price_max)
            ax_vp.set_facecolor('none')
            for spine in ax_vp.spines.values():
                spine.set_visible(False)
            ax_vp.tick_params(left=False, bottom=False,
                              labelleft=False, labelbottom=False)

            bar_h = (price_bins[1] - price_bins[0]) * 0.85
            for i in range(bins):
                center = (price_bins[i] + price_bins[i + 1]) / 2.0
                if i == poc_idx:
                    color, alpha = '#FF5722', 0.85   # POC: red-orange
                elif vol_at_price[i] >= threshold:
                    color, alpha = '#FFA726', 0.65   # HVN: orange
                else:
                    color, alpha = '#90A4AE', 0.30   # Normal: grey
                ax_vp.barh(center, vol_norm[i], height=bar_h,
                           color=color, alpha=alpha)

            # POC label
            poc_center = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2.0
            ax_vp.text(min(vol_norm[poc_idx] + 0.05, 0.95), poc_center,
                       'POC', fontsize=6, color='#FF5722',
                       va='center', ha='left', fontweight='bold')
        except Exception as e:
            logger.debug(f"Volume profile histogram error: {e}")

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
