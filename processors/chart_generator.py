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

Two modes:
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
from processors.math_engine import MathEngine

math_engine = MathEngine()


class ChartGenerator:
    """Generate structure-focused candlestick charts for VLM analysis."""

    def __init__(self):
        self.width = settings.CHART_IMAGE_WIDTH
        self.height = settings.CHART_IMAGE_HEIGHT
        self.dpi = settings.CHART_IMAGE_DPI

    def generate_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str,
                       mode: TradingMode = TradingMode.SWING,
                       liquidation_df: Optional[pd.DataFrame] = None,
                       cvd_df: Optional[pd.DataFrame] = None,
                       funding_df: Optional[pd.DataFrame] = None,
                       df_1d: Optional[pd.DataFrame] = None,
                       df_1w: Optional[pd.DataFrame] = None) -> Optional[bytes]:
        """Generate structure chart for any mode."""
        config = self._get_mode_config(mode)
        return self._generate_structure_chart(df, analysis, symbol, config, 
                                              liquidation_df, cvd_df, funding_df,
                                              df_1d=df_1d, df_1w=df_1w)

    def _get_mode_config(self, mode: TradingMode) -> Dict:
        """Per-mode chart configuration."""
        if mode == TradingMode.POSITION:
            return {
                'resample_rule': '1W',
                'tail_candles': 260,   # ~5 years (Includes previous ATH for macro resistance levels)
                'min_candles': 20,
                'title_suffix': '1W POSITION (Macro Cycle)',
                'fib_tf': '1w',
                'structure_tfs': ['1w'],
                'swing_tf': '1w',
            }
        else:  # SWING (default)
            return {
                'resample_rule': '1D',
                'tail_candles': 180,   # ~6 months (Maximized visual clarity for VLM to spot recent HH/HL structure and FVGs)
                'min_candles': 30,
                'title_suffix': '1D SWING',
                'fib_tf': '1d',
                'structure_tfs': ['1d'],   # 1D chart → use 1D slope (4h slope unit mismatch 방지)
                'swing_tf': '1d',
            }

    def _generate_structure_chart(self, df: pd.DataFrame, analysis: Dict,
                                   symbol: str, config: Dict,
                                   liquidation_df: Optional[pd.DataFrame] = None,
                                   cvd_df: Optional[pd.DataFrame] = None,
                                   funding_df: Optional[pd.DataFrame] = None,
                                   df_1d: Optional[pd.DataFrame] = None,
                                   df_1w: Optional[pd.DataFrame] = None) -> Optional[bytes]:
        """Core chart generation — candlesticks + structure overlays only."""
        try:
            # Prepare OHLCV (Realtime)
            tmp = df.copy()
            tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], utc=True)
            tmp = tmp.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                tmp[col] = tmp[col].astype(float)

            # 1. Prepare FULL resampled DF for calculations
            # [FIX] Merge with GCS historical data if available to ensure long-term indicators (EMA200, MACD) 
            # have enough history, even right after VM boot.
            full_resampled = tmp.resample(config['resample_rule']).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

            if config['resample_rule'] == '1D' and df_1d is not None and not df_1d.empty:
                hist_df = df_1d.copy()
                # Ensure we use the 'timestamp' column for indexing, not the RangeIndex
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
                hist_df = hist_df.set_index('timestamp')
                # Combine: historical + realtime, dropping duplicates favoring realtime
                full_resampled = pd.concat([hist_df, full_resampled])
                full_resampled = full_resampled[~full_resampled.index.duplicated(keep='last')].sort_index()
            elif config['resample_rule'] == '1W' and df_1w is not None and not df_1w.empty:
                hist_df = df_1w.copy()
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
                hist_df = hist_df.set_index('timestamp')
                full_resampled = pd.concat([hist_df, full_resampled])
                full_resampled = full_resampled[~full_resampled.index.duplicated(keep='last')].sort_index()

            # Trim to the exact required tail length to avoid overloading matplotlib, plus a buffer for EMAs
            full_resampled = full_resampled.tail(config['tail_candles'] + 220)

            if len(full_resampled) < min(config['min_candles'], 5):
                logger.warning(f"Not enough candles: {len(full_resampled)} < {config['min_candles']}")
                return None

            # 2. Calculate Indicators on FULL history
            import pandas_ta_classic as ta
            full_resampled['ema200'] = ta.ema(full_resampled['close'], length=min(200, len(full_resampled)-1))

            # 3. Slice for VISUAL window
            chart_df = full_resampled.tail(config['tail_candles']).copy()
            chart_df.index.name = 'Date'

            # 4. Integrate CVD if provided
            apds = []
            if cvd_df is not None and not cvd_df.empty:
                cvd = cvd_df.copy()
                if pd.api.types.is_numeric_dtype(cvd['timestamp']):
                    cvd['timestamp'] = pd.to_datetime(cvd['timestamp'], unit='ms', utc=True).dt.floor('min')
                else:
                    cvd['timestamp'] = pd.to_datetime(cvd['timestamp'], utc=True).dt.floor('min')
                cvd = cvd.set_index('timestamp')
                # Resample CVD to match OHLCV rule
                cvd_resampled = cvd.resample(config['resample_rule']).sum().fillna(0)
                # Calculate Cumulative Delta for visual trend
                cvd_resampled['cvd_acc'] = (cvd_resampled.get('whale_buy_vol', 0) - cvd_resampled.get('whale_sell_vol', 0)).cumsum()
                # Calculate individual bar delta
                cvd_resampled['delta'] = cvd_resampled.get('whale_buy_vol', 0) - cvd_resampled.get('whale_sell_vol', 0)
                
                # Align with chart_df index
                cvd_aligned = cvd_resampled.reindex(chart_df.index).ffill().fillna(0)
                
                # Panel 2: (Price 0, Vol 1, CVD Panel 2)
                # 1. Cumulative CVD line
                apds.append(mpf.make_addplot(cvd_aligned['cvd_acc'], panel=2, 
                                           color='#8E44AD', width=1.5, 
                                           ylabel='CVD', secondary_y=False))
                # 2. Individual Delta bars (Green for Buy > Sell, Red for Sell > Buy)
                colors = ['#27AE60' if d >= 0 else '#C0392B' for d in cvd_aligned['delta']]
                apds.append(mpf.make_addplot(cvd_aligned['delta'], panel=2, 
                                           type='bar', color=colors, alpha=0.3, 
                                           secondary_y=True))

            # 5. Integrate OI and Funding if provided
            if funding_df is not None and not funding_df.empty:
                fnd = funding_df.copy()
                if pd.api.types.is_numeric_dtype(fnd['timestamp']):
                    fnd['timestamp'] = pd.to_datetime(fnd['timestamp'], unit='ms', utc=True).dt.floor('min')
                else:
                    fnd['timestamp'] = pd.to_datetime(fnd['timestamp'], utc=True).dt.floor('min')
                fnd = fnd.set_index('timestamp')
                # Resample (OI is mean or last, Funding is mean)
                fnd_resampled = fnd.resample(config['resample_rule']).agg({
                    'open_interest': 'last',
                    'funding_rate': 'mean'
                }).ffill().fillna(0)
                
                fnd_aligned = fnd_resampled.reindex(chart_df.index).ffill().fillna(0)
                
                # Panel 3: Open Interest
                apds.append(mpf.make_addplot(fnd_aligned['open_interest'], panel=3,
                                           color='#2980B9', width=1.2,
                                           ylabel='OI', secondary_y=False))
                
                # Funding Heat-tape (Calculated as a bar chart at the bottom of Panel 0)
                # We'll use a specific color mapping for funding
                # Green = Positive (Longs pay), Red = Negative (Shorts pay)
                f_colors = ['#27AE60' if r > 0.0001 else '#C0392B' if r < -0.0001 else '#BDC3C7' 
                           for r in fnd_aligned['funding_rate']]
                # We plot this as a tiny bar at the bottom of the main panel or a new panel
                # Let's put it in Panel 4 as a "Heat Tape"
                apds.append(mpf.make_addplot(fnd_aligned['funding_rate'], panel=4,
                                           type='bar', color=f_colors, alpha=0.8,
                                           ylabel='Fnd', secondary_y=False))

            # Upbit-style Professional Light Theme
            mc = mpf.make_marketcolors(
                up='#E74C3C', down='#3498DB',  # Red Up, Blue Down
                edge='inherit', wick='inherit',
                volume={'up': '#E74C3C44', 'down': '#3498DB44'}
            )
            style = mpf.make_mpf_style(
                marketcolors=mc, 
                gridstyle='-', gridcolor='#F2F2F2', 
                facecolor='white', figcolor='white',
                edgecolor='#EEEEEE'
            )

            # Professional Font Config — Legibility Boost
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['text.color'] = '#333333'
            plt.rcParams['axes.labelcolor'] = '#333333'
            plt.rcParams['xtick.color'] = '#666666'
            plt.rcParams['ytick.color'] = '#666666'

            # Plot — Adaptive panel layout (Price 0, Vol 1, CVD 2, OI 3, Funding 4)
            num_panels = 2 # Basic: Price + Vol
            if cvd_df is not None: num_panels += 1
            if funding_df is not None: num_panels += 2 # OI Panel + Funding Tape Panel
            
            p_ratios = [5, 1.2] # Price, Vol
            if cvd_df is not None: p_ratios.append(1.8)
            if funding_df is not None: 
                p_ratios.append(1.5) # OI
                p_ratios.append(0.6) # Funding Tape (Thin)

            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                addplot=apds,
                title='',  # We will draw a custom title
                ylabel='Price',
                figsize=(15, 12 if num_panels > 3 else 10),
                panel_ratios=tuple(p_ratios),
                tight_layout=False,  # We'll use subplots_adjust for margins
                returnfig=True
            )
            
            # ── Layout Optimization ──
            fig.subplots_adjust(top=0.92, right=0.92, left=0.08, bottom=0.08)
            price_ax = axes[0]
            
            # ── Enhanced Title & Metadata ──
            price_ax.text(0.01, 1.05, f"{symbol} | {df['close'].iloc[-1]:.2f}",
                         transform=price_ax.transAxes, fontsize=14, fontweight='bold', 
                         color='#2C3E50', ha='left')

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

            # ── Overlay 9: Fair Value Gaps (FVG) ──
            self._draw_fair_value_gaps(price_ax, chart_df)

            # ── [NEW] Overlay 10: Macro Order Blocks (OB) ──
            macro_obs = math_engine.calculate_macro_order_blocks(full_resampled, count=2)
            self._draw_macro_order_blocks(price_ax, chart_df, macro_obs)

            # ── [NEW] Overlay 11: Anchored VWAP ──
            self._draw_anchored_vwap(price_ax, chart_df, full_resampled)

            # ── [NEW] Overlay 12: Alpha Divergence Markers (CVD) ──
            if cvd_df is not None:
                div = math_engine.detect_macro_divergences(full_resampled, cvd_df)
                self._draw_macro_alpha_markers(price_ax, chart_df, div)

            # ── Lock Y-axis to candlestick range ──
            # Must happen AFTER all overlays so nothing autoscales the axis
            _y_lo = float(chart_df['low'].min())
            _y_hi = float(chart_df['high'].max())
            _margin = (_y_hi - _y_lo) * 0.04
            price_ax.set_ylim(_y_lo - _margin, _y_hi + _margin)

            # Save with high-quality settings
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='#CCCCCC')
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
                
            # Use integer indices for x-axis to align with mpf candles
            x_range = np.arange(len(chart_df))
            ax.plot(x_range, ema.values,
                    color='#7F8C8D', linewidth=0.8,
                    linestyle='--', alpha=0.6, zorder=4)
            
            last_val = ema.dropna().iloc[-1] if not ema.dropna().empty else None
            if last_val:
                ax.text(x_range[-1], float(last_val),
                        ' EMA200', color='#7F8C8D',
                        fontsize=6, va='center', ha='left', alpha=0.7)
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
                            xy=(idx, high[idx]),
                            xytext=(0, 6), textcoords='offset points',
                            fontsize=7, color='#E74C3C', ha='center',
                            va='bottom', fontweight='bold', alpha=0.8,
                            annotation_clip=True)

            # Label swing lows: HL or LL
            for i, idx in enumerate(min_idx):
                if i == 0:
                    label = 'L'
                else:
                    prev = low[min_idx[i - 1]]
                    label = 'HL' if low[idx] > prev else 'LL'
                ax.annotate(label,
                            xy=(idx, low[idx]),
                            xytext=(0, -6), textcoords='offset points',
                            fontsize=7, color='#3498DB', ha='center',
                            va='top', fontweight='bold', alpha=0.8,
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
            ax_vp = ax.inset_axes([0.88, 0.0, 0.12, 1.0])
            ax_vp.set_xlim(0, 1)
            ax_vp.set_ylim(price_min, price_max)
            ax_vp.set_facecolor('none')
            for spine in ax_vp.spines.values():
                spine.set_visible(False)
            ax_vp.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            bar_h = (price_bins[1] - price_bins[0]) * 0.9
            for i in range(bins):
                center = (price_bins[i] + price_bins[i + 1]) / 2.0
                if i == poc_idx:
                    color, alpha = '#E67E22', 0.6   # POC: Stronger Orange
                elif vol_at_price[i] >= threshold:
                    color, alpha = '#BDC3C7', 0.3   # HVN: Stronger Grey
                else:
                    color, alpha = '#BDC3C7', 0.15  # Normal: Subtle Grey
                ax_vp.barh(center, vol_norm[i], height=bar_h,
                           color=color, alpha=alpha, edgecolor='none')

            # POC label
            poc_center = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2.0
            ax_vp.text(min(vol_norm[poc_idx] + 0.05, 0.95), poc_center,
                       'POC', fontsize=7, color='#E67E22',
                       va='center', ha='left', alpha=0.8, fontweight='bold')
        except Exception as e:
            logger.debug(f"Volume profile histogram error: {e}")

    def _draw_pivot_points(self, ax, chart_df: pd.DataFrame):
        """Draw turning point markers on pivot highs/lows."""
        try:
            high = chart_df['high'].astype(float).values
            low = chart_df['low'].astype(float).values
            close = chart_df['close'].astype(float).values
            if len(close) < 15:
                return

            order = max(3, len(close) // 20)
            mins = argrelextrema(close, np.less, order=order)[0]
            maxs = argrelextrema(close, np.greater, order=order)[0]

            if len(mins):
                ax.scatter(mins, low[mins] * 0.998,
                          marker='x', color='#3498DB', s=15, alpha=0.5, zorder=5)
            if len(maxs):
                ax.scatter(maxs, high[maxs] * 1.002,
                          marker='x', color='#E74C3C', s=15, alpha=0.5, zorder=5)
        except Exception as e:
            logger.debug(f"Pivot points draw error: {e}")

    def _draw_diagonal_line(self, ax, chart_df: pd.DataFrame,
                             line_info: Optional[Dict], value_key: str, color: str):
        """Draw diagonal support/resistance trendline.

        Slope from math_engine is always in the same timeframe units as the chart
        (1d slope for 1D chart, 1w slope for 1W chart) — unit mismatch is prevented
        upstream by matching structure_tfs to the chart's resample_rule.

        Y-clamping: prevents off-chart line endpoints from distorting matplotlib's
        auto-scale. We draw only the visible portion and lock the y-axis afterward.
        """
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

            # Chart's actual price range (candlestick body)
            y_lo = float(chart_df['low'].min())
            y_hi = float(chart_df['high'].max())
            margin = (y_hi - y_lo) * 0.05  # 5% padding

            # Skip entirely if line is completely outside the visible range
            if np.all(y > y_hi + margin) or np.all(y < y_lo - margin):
                return

            # Clamp y to [y_lo - margin, y_hi + margin] so matplotlib won't autoscale
            y_plot = np.clip(y, y_lo - margin, y_hi + margin)

            kind = "SUP" if "support" in value_key else "RES"
            label_text = f"TREND {kind}"
            line_color = '#27AE60' if kind == "SUP" else '#C0392B'

            ax.plot(x, y_plot, color=line_color, linewidth=2.0,
                    linestyle='-', alpha=0.9, zorder=10)

            # Labels — only where the true (un-clamped) value is in range
            if y_lo <= y[0] <= y_hi:
                ax.text(0.5, y[0], f' {label_text}', color=line_color,
                        fontsize=7, fontweight='bold', va='bottom', ha='left',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

            if y_lo <= y[-1] <= y_hi:
                ax.text(n - 1.5, y[-1], f' {label_text}', color=line_color,
                        fontsize=7, fontweight='bold', va='bottom', ha='right',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        except Exception:
            return

    def _draw_fibonacci(self, ax, chart_df: pd.DataFrame, fib: Optional[Dict]):
        """Draw Fibonacci retracement horizontal lines."""
        if not fib:
            return

        fib_styles = {
            'fib_236': ('#95A5A6', 0.3, 'FIB 23.6%'),
            'fib_382': ('#95A5A6', 0.4, 'FIB 38.2%'),
            'fib_500': ('#34495E', 0.5, 'FIB 50%'),
            'fib_618': ('#95A5A6', 0.4, 'FIB 61.8%'),
            'fib_786': ('#95A5A6', 0.3, 'FIB 78.6%'),
        }

        n = len(chart_df)
        for key, (color, alpha, label) in fib_styles.items():
            val = fib.get(key)
            if isinstance(val, (int, float)):
                # Dotted -> Dashed for better VLM visibility
                ax.axhline(val, color=color, linestyle='--', linewidth=0.8,
                           alpha=alpha + 0.3, zorder=2)
                # Move text slightly left of the edge to ensure it's on-canvas
                ax.text(n - 0.5, val, f' {label}',
                        color=color, fontsize=7, va='center', ha='left', 
                        alpha=0.9, fontweight='bold')

    def _draw_swing_levels(self, ax, chart_df: pd.DataFrame, swing: Optional[Dict]):
        """Draw horizontal lines and shaded zones at swing highs/lows (liquidity pools)."""
        if not swing:
            return

        n = len(chart_df)
        for high in swing.get('swing_highs', []):
            if isinstance(high, (int, float)):
                # Draw main line
                ax.axhline(high, color='#C0392B', linestyle='-', linewidth=0.8,
                           alpha=0.4, zorder=2)
                # Draw subtle shaded zone for "Liquidity Pool" (+/- 0.3% range)
                ax.axhspan(high * 0.997, high * 1.003, color='#C0392B', 
                           alpha=0.08, zorder=1)
                # Add label
                ax.text(0, high, " SWING RES", color='#C0392B', 
                        fontsize=6, fontweight='bold', va='bottom', alpha=0.9)

        for low in swing.get('swing_lows', []):
            if isinstance(low, (int, float)):
                # Draw main line
                ax.axhline(low, color='#27AE60', linestyle='-', linewidth=0.8,
                           alpha=0.4, zorder=2)
                # Draw subtle shaded zone for "Liquidity Pool" (+/- 0.3% range)
                ax.axhspan(low * 0.997, low * 1.003, color='#27AE60', 
                           alpha=0.08, zorder=1)
                # Add label
                ax.text(0, low, " SWING SUP", color='#27AE60', 
                        fontsize=6, fontweight='bold', va='top', alpha=0.9)

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

                # Find nearest chart index (integer offset 0 to N-1)
                nearest_idx = chart_df.index.get_indexer([row['timestamp']], method='nearest')[0]
                if 0 <= nearest_idx < len(chart_df):
                    price = chart_df.iloc[nearest_idx]['close']
                    ax.scatter(nearest_idx, price,
                              marker='o', color=color, s=size, alpha=0.7,
                              edgecolors='black', linewidth=0.5, zorder=6)

        except Exception as e:
            logger.debug(f"Liquidation markers draw error: {e}")

    def _draw_fair_value_gaps(self, ax, chart_df: pd.DataFrame):
        """Identify and draw Fair Value Gaps (FVG) as shaded boxes.
        Bullish FVG: Low[n+1] > High[n-1]
        Bearish FVG: High[n+1] < Low[n-1]
        """
        try:
            high = chart_df['high'].astype(float).values
            low = chart_df['low'].astype(float).values
            n = len(chart_df)
            if n < 3:
                return

            bull_gaps = []
            bear_gaps = []
            
            for i in range(1, n - 1):
                # Bullish FVG
                if low[i + 1] > high[i - 1]:
                    bull_gaps.append({'idx': i, 'top': low[i + 1], 'bottom': high[i - 1]})
                # Bearish FVG
                elif high[i + 1] < low[i - 1]:
                    bear_gaps.append({'idx': i, 'top': low[i - 1], 'bottom': high[i + 1]})

            # ── Merge and Draw Bullish Gaps ──
            if bull_gaps:
                merged = []
                if bull_gaps:
                    curr = bull_gaps[0].copy()
                    curr['start'] = curr['idx']
                    curr['end'] = curr['idx']
                    for g in bull_gaps[1:]:
                        if g['idx'] <= curr['end'] + 1: # Consecutive
                            curr['end'] = g['idx']
                            curr['top'] = max(curr['top'], g['top'])
                            curr['bottom'] = min(curr['bottom'], g['bottom'])
                        else:
                            merged.append(curr)
                            curr = g.copy()
                            curr['start'] = curr['idx']
                            curr['end'] = curr['idx']
                    merged.append(curr)

                vap_safe_idx = n * 0.82 # More aggressive 18% safety buffer for VAP/POC
                
                for m in merged:
                    # Clip x_range to stay out of VAP zone
                    x_start = m['start'] - 0.5
                    x_end = min(m['end'] + 1.5, vap_safe_idx)
                    
                    if x_end <= x_start:
                        continue # Box completely obscured by VAP
                        
                    x_range = [x_start, x_end]
                    ax.fill_between(x_range, m['bottom'], m['top'], color='#27AE60', 
                                    alpha=0.4, zorder=1, edgecolor='none')
                    
                    # Smart Label Pivoting: If close to right edge, push text to the left
                    is_near_edge = x_end > (n * 0.75)
                    if is_near_edge:
                        txt_x = x_start + 0.5
                        ha = 'left'
                    else:
                        txt_x = (x_start + x_end) / 2
                        ha = 'center'

                    ax.text(txt_x, (m['top'] + m['bottom']) / 2, 
                            "FVG", color='#27AE60', fontsize=9, alpha=0.9, 
                            ha=ha, va='center', fontweight='bold')

            # ── Merge and Draw Bearish Gaps ──
            if bear_gaps:
                merged = []
                curr = bear_gaps[0].copy()
                curr['start'] = curr['idx']
                curr['end'] = curr['idx']
                for g in bear_gaps[1:]:
                    if g['idx'] <= curr['end'] + 1:
                        curr['end'] = g['idx']
                        curr['top'] = max(curr['top'], g['top'])
                        curr['bottom'] = min(curr['bottom'], g['bottom'])
                    else:
                        merged.append(curr)
                        curr = g.copy()
                        curr['start'] = curr['idx']
                        curr['end'] = curr['idx']
                merged.append(curr)

                vap_safe_idx = n * 0.82
                for m in merged:
                    x_start = m['start'] - 0.5
                    x_end = min(m['end'] + 1.5, vap_safe_idx)
                    
                    if x_end <= x_start:
                        continue
                        
                    x_range = [x_start, x_end]
                    ax.fill_between(x_range, m['bottom'], m['top'], color='#C0392B', 
                                    alpha=0.4, zorder=1, edgecolor='none')
                    
                    is_near_edge = x_end > (n * 0.75)
                    if is_near_edge:
                        txt_x = x_start + 0.5
                        ha = 'left'
                    else:
                        txt_x = (x_start + x_end) / 2
                        ha = 'center'

                    ax.text(txt_x, (m['top'] + m['bottom']) / 2, 
                            "FVG", color='#C0392B', fontsize=9, alpha=0.9, 
                            ha=ha, va='center', fontweight='bold')

        except Exception as e:
            logger.debug(f"FVG draw error: {e}")

    def _draw_macro_order_blocks(self, ax, chart_df: pd.DataFrame, obs: List[Dict]):
        """Draw Macro Order Blocks as shaded zones with labels."""
        try:
            if not obs: return
            
            n = len(chart_df)
            for ob in obs:
                # Find if OB is within or near visible range
                color = '#27AE60' if ob['type'] == 'BULLISH' else '#C0392B'
                label = "MACRO BULL OB" if ob['type'] == 'BULLISH' else "MACRO BEAR OB"
                
                # Check if OB top/bottom is in price range approximately
                y_lo, y_hi = chart_df['low'].min(), chart_df['high'].max()
                if ob['bottom'] > y_hi * 1.2 or ob['top'] < y_lo * 0.8:
                    continue
                
                # Draw horizontal span across the whole chart for macro levels
                ax.axhspan(ob['bottom'], ob['top'], color=color, alpha=0.1, zorder=1)
                ax.axhline(ob['top'], color=color, linestyle=':', linewidth=0.5, alpha=0.5, zorder=2)
                ax.axhline(ob['bottom'], color=color, linestyle=':', linewidth=0.5, alpha=0.5, zorder=2)
                
                # Label on the left
                ax.text(5, (ob['top'] + ob['bottom']) / 2, label, 
                        color=color, fontsize=7, fontweight='bold', va='center', alpha=0.8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        except Exception as e:
            logger.debug(f"Macro OB draw error: {e}")

    def _draw_anchored_vwap(self, ax, chart_df: pd.DataFrame, full_df: pd.DataFrame):
        """Find significant cycle low/high and anchor a VWAP there."""
        try:
            # Find the most significant low/high in the full_df history (~6-12 months)
            prices = full_df['close'].values
            low_idx = np.argmin(prices)
            high_idx = np.argmax(prices)
            
            # Use the most recent significant extreme as the anchor
            anchor_idx = low_idx if low_idx > high_idx else high_idx
            
            avwap = math_engine.calculate_anchored_vwap(full_df, anchor_idx)
            if avwap is None: return
            
            # Align with visual chart_df
            avwap_aligned = avwap.reindex(chart_df.index).values
            x_range = np.arange(len(chart_df))
            
            ax.plot(x_range, avwap_aligned, color='#D4AC0D', linewidth=1.5, 
                    linestyle='-.', alpha=0.8, zorder=4)
            
            # Label
            label = "AVWAP (Cycle Low)" if anchor_idx == low_idx else "AVWAP (Cycle High)"
            last_valid_idx = np.where(~np.isnan(avwap_aligned))[0]
            if len(last_valid_idx) > 0:
                ax.text(last_valid_idx[-1], avwap_aligned[last_valid_idx[-1]], f" {label}", 
                        color='#D4AC0D', fontsize=7, fontweight='bold', va='bottom', ha='right')
        except Exception as e:
            logger.debug(f"AVWAP draw error: {e}")

    def _draw_macro_alpha_markers(self, ax, chart_df: pd.DataFrame, divergence: Dict):
        """Draw explicit markers for Macro Alpha confluences."""
        try:
            n = len(chart_df)
            if divergence.get('macro_bear_div'):
                ax.text(n-5, chart_df['high'].max(), "[!] MACRO DISTRIBUTION DETECTED (Price HH vs CVD LH)", 
                        color='#C0392B', fontsize=9, fontweight='bold', ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#C0392B', pad=2))
            
            if divergence.get('macro_bull_div'):
                ax.text(n-5, chart_df['low'].min(), "[!] MACRO ACCUMULATION DETECTED (Price LL vs CVD HL)", 
                        color='#27AE60', fontsize=9, fontweight='bold', ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#27AE60', pad=2))
        except Exception as e:
            logger.debug(f"Alpha markers draw error: {e}")

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
