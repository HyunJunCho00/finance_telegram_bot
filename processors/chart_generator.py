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
                       timeframe: Optional[str] = None,
                       liquidation_df: Optional[pd.DataFrame] = None,
                       cvd_df: Optional[pd.DataFrame] = None,
                       funding_df: Optional[pd.DataFrame] = None,
                       df_1d: Optional[pd.DataFrame] = None,
                       df_1w: Optional[pd.DataFrame] = None) -> Optional[bytes]:
        """Generate structure chart for any mode or specific timeframe."""
        config = self._get_mode_config(mode, timeframe)
        # Filter analysis data for visual clarity (active elements only)
        current_price = float(df['close'].iloc[-1])
        analysis = self._filter_active_elements(current_price, analysis)
        
        return self._generate_structure_chart(df, analysis, symbol, config, 
                                              liquidation_df, cvd_df, funding_df,
                                              df_1d=df_1d, df_1w=df_1w)

    def _get_mode_config(self, mode: TradingMode, timeframe: Optional[str] = None) -> Dict:
        """Per-mode or per-timeframe chart configuration."""
        # 1. Base config determined by Mode (for Overlays / MTFA)
        if mode == TradingMode.POSITION:
            config = {
                'resample_rule': '1W',
                'tail_candles': 260,   # ~5 years 
                'min_candles': 20,
                'title_suffix': '1W POSITION (Macro Cycle)',
                'fib_tf': '1w',
                'structure_tfs': ['1w'],
                'swing_tf': '1w',
                'is_custom': False
            }
        else:  # SWING (default)
            config = {
                'resample_rule': '1D',
                'tail_candles': 365,   # ~12 months
                'min_candles': 30,
                'title_suffix': '1D SWING',
                'fib_tf': '1d',
                'structure_tfs': ['1d'],
                'swing_tf': '1d',
                'is_custom': False
            }

        # 2. Override ONLY visual display aspects if custom timeframe requested (MTFA)
        if timeframe and timeframe.lower() not in ('auto', 'none'):
            tf = timeframe.lower().strip()
            tf_map = {
                '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
                '1d': '1D', '3d': '3D', '1w': '1W-MON', '1M': '1MS',
            }
            config['resample_rule'] = tf_map.get(tf, '1h')
            # Dynamic tail calculation based on TF to show enough history
            tail = 200
            if tf in ('1m', '5m', '15m', '30m'): tail = 150 # Intraday needs more zoom
            elif tf in ('1d', '1w'): tail = 200      # Macro needs context
            config['tail_candles'] = tail
            
            # For VLM display, label the title correctly
            config['title_suffix'] = f"{tf.upper()} (MTFA: {config['title_suffix']})"
            config['is_custom'] = True
            
            # NOTE: We intentionally DO NOT OVERRIDE 'fib_tf', 'structure_tfs', and 'swing_tf'.
            # They remain locked to the HTF (e.g. '1d' or '1w') defined by the Trade Mode.

        return config

    def _filter_active_elements(self, current_price: float, analysis: Dict) -> Dict:
        """Filter analysis data to keep only elements relevant to current price action.
        Reduces visual noise and prevents VLM 'encyclopedia' overexposure.
        """
        if not analysis:
            return {}
        
        filtered = analysis.copy()
        
        # 1. Filter Swing Levels (Liquidity Pools)
        # Keep only levels near current price (e.g., within 5%) or the closest 2 above/below
        if 'swing_levels' in filtered:
            new_swing = {}
            for tf, levels in filtered['swing_levels'].items():
                if not levels: continue
                sh = sorted(levels.get('swing_highs', []), reverse=False)
                sl = sorted(levels.get('swing_lows', []), reverse=True)
                
                # Keep 2 nearest resistance (above) and 2 nearest support (below)
                active_sh = [h for h in sh if h > current_price][:2]
                active_sl = [l for l in sl if l < current_price][:2]
                
                new_swing[tf] = {'swing_highs': active_sh, 'swing_lows': active_sl}
            filtered['swing_levels'] = new_swing

        # 2. Structure Labels (HH/HL/LH/LL)
        # Usually handled by the drafting logic itself, but we can prune history if needed
        # (Already handled in _draw_market_structure_labels via 'order' and visual slicing)
        
        return filtered

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

            rule_upper = str(config['resample_rule']).upper()
            if rule_upper.startswith('1D') and df_1d is not None and not df_1d.empty:
                hist_df = df_1d.copy()
                # Ensure we use the 'timestamp' column for indexing, not the RangeIndex
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
                hist_df = hist_df.set_index('timestamp')
                # Combine: historical + realtime, dropping duplicates favoring realtime
                full_resampled = pd.concat([hist_df, full_resampled])
                full_resampled = full_resampled[~full_resampled.index.duplicated(keep='last')].sort_index()
            elif rule_upper.startswith('1W') and df_1w is not None and not df_1w.empty:
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

            # 3. Get Trading Mode for Header context
            trading_mode = getattr(settings, 'trading_mode', TradingMode.SWING)
            
            # 4. Slice for VISUAL window
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
                # Resample CVD to match OHLCV rule, but use min_count=1 to keep structural NaNs
                cvd_resampled = cvd.resample(config['resample_rule']).sum(min_count=1)
                # Calculate Cumulative Delta for visual trend
                if 'whale_buy_vol' in cvd_resampled and 'whale_sell_vol' in cvd_resampled:
                    buy_vol = cvd_resampled['whale_buy_vol']
                    sell_vol = cvd_resampled['whale_sell_vol']
                    cvd_resampled['delta'] = buy_vol - sell_vol
                    cvd_resampled['cvd_acc'] = cvd_resampled['delta'].cumsum()
                else:
                    cvd_resampled['delta'] = pd.Series(np.nan, index=cvd_resampled.index)
                    cvd_resampled['cvd_acc'] = pd.Series(np.nan, index=cvd_resampled.index)
                
                # Align with chart_df index
                cvd_aligned = cvd_resampled.reindex(chart_df.index)
                
                # Forward fill to prevent empty visual gaps for historical dates not in DB
                # We remove bfill() to prevent artificial flat lines when data doesn't exist yet
                cvd_acc_plot = cvd_aligned['cvd_acc'].ffill()
                # Individual delta bars can be 0 when missing
                cvd_delta_plot = cvd_aligned['delta'].fillna(0)
                
                # Panel 2: (Price 0, Vol 1, CVD Panel 2)
                apds.append(mpf.make_addplot(cvd_acc_plot, panel=2, 
                                           color='#8E44AD', width=1.5, 
                                           ylabel='CVD', secondary_y=False))
                colors = ['#27AE60' if d >= 0 else '#C0392B' for d in cvd_delta_plot]
                apds.append(mpf.make_addplot(cvd_delta_plot, panel=2, 
                                           type='bar', color=colors, alpha=0.3, 
                                           secondary_y=True))

            # 5. Integrate OI and Funding if provided
            _has_oi = True  # track for panel count correction below
            if funding_df is not None and not funding_df.empty:
                fnd = funding_df.copy()
                if pd.api.types.is_numeric_dtype(fnd['timestamp']):
                    fnd['timestamp'] = pd.to_datetime(fnd['timestamp'], unit='ms', utc=True).dt.floor('min')
                else:
                    fnd['timestamp'] = pd.to_datetime(fnd['timestamp'], utc=True).dt.floor('min')
                fnd = fnd.set_index('timestamp')
                
                # Safely determine what columns are available
                has_oi = 'open_interest' in fnd.columns
                has_funding = 'funding_rate' in fnd.columns
                
                agg_map = {}
                if has_oi: agg_map['open_interest'] = 'last'
                if has_funding: agg_map['funding_rate'] = 'mean'
                
                if not agg_map:
                    # No usable columns in funding_df at all
                    funding_df = None
                else:
                    fnd_resampled = fnd.resample(config['resample_rule']).agg(agg_map)
                    fnd_aligned = fnd_resampled.reindex(chart_df.index)
                    
                    # OI Panel — only if column exists
                    if has_oi:
                        fnd_oi_plot = fnd_aligned['open_interest'].ffill()
                        apds.append(mpf.make_addplot(fnd_oi_plot, panel=3,
                                                   color='#2980B9', width=1.2,
                                                   ylabel='OI', secondary_y=False))
                    else:
                        # No OI — flag for panel count correction below
                        _has_oi = False
                    
                    # Funding Rate Tape — only if column exists
                    if has_funding:
                        fnd_rate_plot = fnd_aligned['funding_rate'].fillna(0)
                        f_colors = ['#27AE60' if r > 0.0001 else '#C0392B' if r < -0.0001 else '#BDC3C7' 
                                   for r in fnd_rate_plot]
                        tape_panel = 4 if has_oi else 3  # shift panel if OI missing
                        apds.append(mpf.make_addplot(fnd_rate_plot, panel=tape_panel,
                                                   type='bar', color=f_colors, alpha=0.8,
                                                   ylabel='Fnd', secondary_y=False))

            # Dynamic Theme Selection
            theme = getattr(settings, 'CHART_THEME', 'dark_premium').lower()
            
            if theme == 'light_premium':
                # Professional Light Theme (TradingView / Upbit Style)
                mc = mpf.make_marketcolors(
                    up='#089981', down='#F23645',  # Emerald Green / Crimson Red
                    edge='inherit', wick='inherit',
                    volume={'up': '#08998144', 'down': '#F2364544'}
                )
                style = mpf.make_mpf_style(
                    marketcolors=mc, 
                    gridstyle='-', gridcolor='#F0F3FA', 
                    facecolor='white', figcolor='white',
                    edgecolor='#E0E3EB'
                )
                text_color = '#131722'
                grid_color = '#F0F3FA'
            else:
                # Default: Professional Dark Premium (Exchange Style)
                mc = mpf.make_marketcolors(
                    up='#089981', down='#F23645',
                    edge='inherit', wick='inherit',
                    volume={'up': '#08998166', 'down': '#F2364566'}
                )
                style = mpf.make_mpf_style(
                    marketcolors=mc, 
                    gridstyle='-', gridcolor='#1E222D', 
                    facecolor='#131722', figcolor='#131722',
                    edgecolor='#2A2E39'
                )
                text_color = '#D1D4DC'
                grid_color = '#1E222D'

            # Professional Font Config — Legibility Boost
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['text.color'] = text_color
            plt.rcParams['axes.labelcolor'] = text_color
            plt.rcParams['xtick.color'] = '#787B86'
            plt.rcParams['ytick.color'] = '#787B86'

            # Plot — Adaptive panel layout (Price 0, Vol 1, CVD 2, OI 3, Funding 4)
            num_panels = 2 # Basic: Price + Vol
            if cvd_df is not None: num_panels += 1
            if funding_df is not None:
                num_panels += 2 if _has_oi else 1  # OI + Funding, or just Funding

            p_ratios = [5, 1.2] # Price, Vol
            if cvd_df is not None: p_ratios.append(1.8)
            if funding_df is not None:
                if _has_oi: p_ratios.append(1.5) # OI
                p_ratios.append(0.6) # Funding Tape (Thin)
                
            logger.debug(f"Calling mpf.plot with {len(apds)} addplots, {num_panels} panels, ratios {p_ratios}")

            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                addplot=apds,
                title='',  # We will draw a custom title
                ylabel='Price',
                figsize=(15, 12 if num_panels > 3 else 10),
                panel_ratios=tuple(p_ratios),
                datetime_format='%Y-%m-%d',
                xrotation=0,
                tight_layout=False,  # We'll use subplots_adjust for margins
                returnfig=True
            )
            
            # ── Layout Optimization ──
            fig.subplots_adjust(top=0.92, right=0.92, left=0.08, bottom=0.08)
            price_ax = axes[0]
            
            # ── Enhanced Title & Metadata ──
            price_ax.text(0.01, 1.05, f"{symbol} | {df['close'].iloc[-1]:.2f}",
                         transform=price_ax.transAxes, fontsize=14, fontweight='bold', 
                         color=text_color, ha='left')

            # ── Add Background Watermark (Big Symbol) ──
            price_ax.text(0.5, 0.5, symbol.split('USDT')[0],
                         transform=price_ax.transAxes, fontsize=80, fontweight='bold',
                         color=text_color, alpha=0.03, ha='center', va='center', zorder=0)

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
                                         'support_price', '#089981', theme, text_color)
                self._draw_diagonal_line(price_ax, chart_df,
                                         structure.get(f'resistance_{tf}'),
                                         'resistance_price', '#F23645', theme, text_color)

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
            self._draw_fair_value_gaps(price_ax, chart_df, mode)

            # ── [NEW] Overlay 10: Macro Order Blocks (OB) ──
            # count=4 → returns up to 8 candidates (obs[:count*2])
            # Ensures enough pool to guarantee 1 above + 1 below current price
            # even in strongly trending markets where recent OBs cluster on one side
            macro_obs = math_engine.calculate_macro_order_blocks(full_resampled, count=4)
            self._draw_macro_order_blocks(price_ax, chart_df, macro_obs)

            # ── [NEW] Overlay 11: Anchored VWAP ──
            self._draw_anchored_vwap(price_ax, chart_df, full_resampled)

            # ── [NEW] Overlay 12: Alpha Divergence Markers (CVD) ──
            if cvd_df is not None:
                div = math_engine.detect_macro_divergences(full_resampled, cvd_df)
                self._draw_macro_alpha_markers(price_ax, chart_df, div)

            # ── [PRO] Overlay 13: Header Legend ──
            self._draw_header_legend(price_ax, chart_df, symbol, config, text_color)

            # ── [PRO] Overlay 14: Session Breaks (Day-dividers) ──
            self._draw_session_breaks(price_ax, chart_df)
            
            # ── [PRO] Overlay 15: Current Price Label (On Y-axis) ──
            self._draw_current_price_label(price_ax, chart_df, text_color)

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
            
            ema_clean = ema.dropna()
            last_val = float(ema_clean.iloc[-1]) if not ema_clean.empty else None
            if last_val is not None:
                ax.text(x_range[-1], last_val,
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

            # Build full label history first (full history needed for correct HH/LH classification)
            # Then render only the LAST 3 swing highs and lows — enough to read current structure
            # (e.g. HH→HH→LH signals potential trend break; HL→HL→LL signals continuation of downtrend)
            all_high_labels = []
            for i, idx in enumerate(max_idx):
                if i == 0:
                    label = 'S_H'
                else:
                    prev = high[max_idx[i - 1]]
                    label = 'S_HH' if high[idx] > prev else 'S_LH'
                all_high_labels.append((idx, label))

            all_low_labels = []
            for i, idx in enumerate(min_idx):
                if i == 0:
                    label = 'S_L'
                else:
                    prev = low[min_idx[i - 1]]
                    label = 'S_HL' if low[idx] > prev else 'S_LL'
                all_low_labels.append((idx, label))

            # Draw only the most recent 3 swing highs
            for idx, label in all_high_labels[-3:]:
                ax.annotate(label,
                            xy=(idx, high[idx]),
                            xytext=(0, 7), textcoords='offset points',
                            fontsize=8, color='#E74C3C', ha='center',
                            va='bottom', fontweight='bold', alpha=0.9,
                            annotation_clip=True,
                            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=0.5))

            # Draw only the most recent 3 swing lows
            for idx, label in all_low_labels[-3:]:
                ax.annotate(label,
                            xy=(idx, low[idx]),
                            xytext=(0, -7), textcoords='offset points',
                            fontsize=8, color='#3498DB', ha='center',
                            va='top', fontweight='bold', alpha=0.9,
                            annotation_clip=True,
                            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=0.5))
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
                             line_info: Optional[Dict], value_key: str, color: str,
                             theme: str = 'dark_premium', text_color: str = '#D1D4DC'):
        """Draw diagonal support/resistance trendline using exact timestamps.
        
        Using timestamps instead of row indices ensures that if there is an offline
        data gap in the chart_df, the line will be drawn correctly across the gap
        rather than being distorted by the missing rows.
        """
        if not line_info or 'point1' not in line_info or 'point2' not in line_info:
            return
        
        try:
            pt1 = line_info['point1']
            pt2 = line_info['point2']
            
            ts1, y1 = pt1
            ts2, y2 = pt2
            
            # chart_df index is datetime
            chart_start = chart_df.index[0]
            chart_end = chart_df.index[-1]
            
            # Ensure both are timezone aware for subtraction
            if ts1.tzinfo is None: ts1 = ts1.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            if ts2.tzinfo is None: ts2 = ts2.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            
            # Calculate slope in units of price per second
            dt = (ts2 - ts1).total_seconds()
            if dt == 0:
                return
            slope_sec = (y2 - y1) / dt
            
            # Project line to the edges of the visible chart window
            # Start point (left edge)
            if chart_start.tzinfo is None: chart_start = chart_start.tz_localize('UTC')
            dt_start = (chart_start - ts1).total_seconds()
            y_start = y1 + slope_sec * dt_start
            
            # End point (right edge)
            if chart_end.tzinfo is None: chart_end = chart_end.tz_localize('UTC')
            dt_end = (chart_end - ts1).total_seconds()
            y_end = y1 + slope_sec * dt_end
            
            # We want to draw a line from (chart_start, y_start) to (chart_end, y_end)
            # using matplotlib's standard plot, but mapped to the chart_df's integer x-axis.
            # Since mplfinance removes gaps visually, a straight line in time might 
            # look slightly kinked if plotted over a gap. But usually drawing straight
            # across the visual space is preferred.
            
            y_lo = float(chart_df['low'].min())
            y_hi = float(chart_df['high'].max())
            margin = (y_hi - y_lo) * 0.05
            
            if max(y_start, y_end) < y_lo - margin or min(y_start, y_end) > y_hi + margin:
                return # Out of bounds
            
            # To draw on the mpf axes natively, we calculate the Y value for every X position
            # using the actual timestamp of that X position.
            x_vals = np.arange(len(chart_df))
            timestamps = chart_df.index
            
            # Ensure both are timezone aware for subtraction
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize('UTC')
            if ts1.tzinfo is None:
                ts1 = ts1.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            
            # Calculate Y for every point based on exact timestamp
            seconds_diff = (timestamps - ts1).total_seconds()
            y_vals = y1 + slope_sec * seconds_diff
            
            # Clamp to prevent autoscaling issues
            y_plot = np.clip(y_vals, y_lo - margin, y_hi + margin)
            
            kind = "SUP" if "support" in line_info.get('type', value_key) else "RES"
            label_text = f"TREND LIQ" # Rebranded as Trendline Liquidity Pools
            
            # Draw as dashed lines to indicate liquidity pools rather than solid hard boundaries
            ax.plot(x_vals, y_plot, color=color, linewidth=1.5,
                    linestyle='--', alpha=0.6, zorder=10)
            
            ax.text(x_vals[-1] - 1.5, y_vals[-1], f' {label_text}', color=color,
                    fontsize=7, fontweight='bold', va='bottom', ha='right',
                    bbox=dict(facecolor=text_color if theme == 'light_premium' else '#131722', 
                              alpha=0.5, edgecolor='none', pad=1))
                    
        except Exception as e:
            logger.debug(f"Trendline draw error: {e}")
            return

    def _draw_fibonacci(self, ax, chart_df: pd.DataFrame, fib: Optional[Dict]):
        """Draw Fibonacci retracement horizontal lines with anchor markers.
        Anchor markers are critical: VLM schema now requires explicit anchor_high/anchor_low,
        so the anchors must be visually identifiable in the chart image.
        """
        if not fib:
            return

        fib_styles = {
            'fib_500': ('#34495E', 0.5, 'FIB 50%'),
            'fib_618': ('#95A5A6', 0.4, 'FIB 61.8%'),
            'fib_705': ('#95A5A6', 0.4, 'FIB 70.5%'),
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

        # Draw anchor markers so VLM can identify exact swing_high / swing_low used for Fib
        # Without these markers, VLM must guess anchors → fibonacci_context.anchor_high/low unreliable
        anchor_high = fib.get('swing_high')
        anchor_low = fib.get('swing_low')
        if isinstance(anchor_high, (int, float)):
            ax.axhline(anchor_high, color='#F39C12', linestyle='-', linewidth=1.0,
                       alpha=0.5, zorder=3)
            ax.text(1, anchor_high, ' FIB HIGH ▲',
                    color='#F39C12', fontsize=7, va='bottom', ha='left',
                    fontweight='bold', alpha=0.9)
        if isinstance(anchor_low, (int, float)):
            ax.axhline(anchor_low, color='#F39C12', linestyle='-', linewidth=1.0,
                       alpha=0.5, zorder=3)
            ax.text(1, anchor_low, ' FIB LOW ▼',
                    color='#F39C12', fontsize=7, va='top', ha='left',
                    fontweight='bold', alpha=0.9)

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

    def _draw_fair_value_gaps(self, ax, chart_df: pd.DataFrame,
                               mode: TradingMode = TradingMode.SWING):
        """Identify and draw Fair Value Gaps (FVG) as shaded boxes.
        Bullish FVG: Low[n+1] > High[n-1]
        Bearish FVG: High[n+1] < Low[n-1]
        POSITION mode: filter gaps < 1% (same as math_engine.calculate_fvg)
        """
        try:
            high = chart_df['high'].astype(float).values
            low = chart_df['low'].astype(float).values
            close = chart_df['close'].astype(float).values
            current_price = close[-1]
            n = len(chart_df)
            if n < 3:
                return
            min_gap_pct = 1.0 if mode == TradingMode.POSITION else 0.0

            bull_gaps = []
            bear_gaps = []
            
            for i in range(1, n - 1):
                # Bullish FVG: Low[i+1] > High[i-1]
                if low[i + 1] > high[i - 1]:
                    top = low[i + 1]
                    bottom = high[i - 1]
                    # Skip tiny gaps in POSITION mode (< 1%)
                    if bottom > 0 and (top - bottom) / bottom * 100 < min_gap_pct:
                        continue
                    # [VLM OPT] Only show UNMITIGATED gaps (Price has not returned to fill them)
                    post_gap_lows = low[i + 2:]
                    is_mitigated = (post_gap_lows < top).any() if len(post_gap_lows) > 0 else False
                    if not is_mitigated:
                        bull_gaps.append({'idx': i, 'top': top, 'bottom': bottom})

                # Bearish FVG: High[i+1] < Low[i-1]
                elif high[i + 1] < low[i - 1]:
                    top = low[i - 1]
                    bottom = high[i + 1]
                    # Skip tiny gaps in POSITION mode (< 1%)
                    if bottom > 0 and (top - bottom) / bottom * 100 < min_gap_pct:
                        continue
                    post_gap_highs = high[i + 2:]
                    is_mitigated = (post_gap_highs > bottom).any() if len(post_gap_highs) > 0 else False
                    if not is_mitigated:
                        bear_gaps.append({'idx': i, 'top': top, 'bottom': bottom})

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
            
            current_price = float(chart_df['close'].iloc[-1])
            n = len(chart_df)
            
            # [VLM OPT] Keep only the nearest order blocks to current price
            # 1 nearest above (Resistance), 1 nearest below (Support)
            obs_above = sorted([ob for ob in obs if ob['bottom'] > current_price], key=lambda x: x['bottom'])
            obs_below = sorted([ob for ob in obs if ob['top'] < current_price], key=lambda x: x['top'], reverse=True)
            
            active_obs = []
            if obs_above: active_obs.append(obs_above[0])
            if obs_below: active_obs.append(obs_below[0])

            for ob in active_obs:
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

    def _draw_header_legend(self, ax, chart_df: pd.DataFrame, symbol: str, config: Dict, text_color: str):
        """Draw a professional TradingView-style header legend in the top-left."""
        try:
            last_candle = chart_df.iloc[-1]
            tf_str = config.get('title_suffix', '4H').split(' ')[0] # Get '15M' from '15M' or '1D' from '1D SWING'
            
            # 1. Main Header: SYMBOL TF EXCHANGE (Data up to: TIMESTAMP)
            last_ts_str = chart_df.index[-1].strftime('%Y-%m-%d %H:%M')
            header_text = f"{symbol} · {tf_str} · CRYPTO (Data up to: {last_ts_str})"
            ax.text(0.01, 0.98, header_text, transform=ax.transAxes, 
                    fontsize=10, fontweight='bold', color=text_color, alpha=0.9,
                    ha='left', va='top')
            
            # 2. Price Info: O H L C
            price_text = (f"O{last_candle['open']:.2f}  H{last_candle['high']:.2f}  "
                         f"L{last_candle['low']:.2f}  C{last_candle['close']:.2f}")
            ax.text(0.01, 0.94, price_text, transform=ax.transAxes,
                    fontsize=8, color=text_color, alpha=0.7,
                    ha='left', va='top')
            
            # 3. Indicators Legend
            indicators = []
            if 'ema200' in chart_df.columns:
                ema_val = last_candle['ema200']
                if not pd.isna(ema_val):
                    indicators.append(f"EMA 200: {ema_val:.2f}")
            
            if indicators:
                ind_text = "  /  ".join(indicators)
                ax.text(0.01, 0.90, ind_text, transform=ax.transAxes,
                        fontsize=8, color='#787B86', alpha=0.8,
                        ha='left', va='top')
            
            # 4. [VLM OPT] Analytical Overlay Guide (Rosetta Stone)
            # This helps VLM map visual colors to semantic concepts
            overlay_guide = ("LEGEND: [Green Box=Bull FVG/OB] [Red Box=Bear FVG/OB] "
                            "[Yellow Line=AVWAP] [Blue X=Structure Low] [Red X=Structure High]")
            ax.text(0.01, 0.02, overlay_guide, transform=ax.transAxes,
                    fontsize=7, color=text_color, alpha=0.5,
                    ha='left', va='bottom', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1))
        except Exception as e:
            logger.debug(f"Header legend draw error: {e}")

    def _draw_session_breaks(self, ax, chart_df: pd.DataFrame):
        """Draw vertical dashed lines at day changes (00:00 UTC)."""
        try:
            # Find indices where the date changes
            dates = chart_df.index.date
            breaks = np.where(dates[1:] != dates[:-1])[0] + 1
            
            for idx in breaks:
                ax.axvline(idx - 0.5, color='#787B86', linestyle=':', 
                           linewidth=0.5, alpha=0.3, zorder=0)
        except Exception as e:
            logger.debug(f"Session breaks draw error: {e}")

    def _draw_current_price_label(self, ax, chart_df: pd.DataFrame, text_color: str):
        """Draw a highlighted price label on the right Y-axis for the current price."""
        try:
            last_close = float(chart_df['close'].iloc[-1])
            prev_close = float(chart_df['close'].iloc[-2]) if len(chart_df) > 1 else last_close
            
            # Color based on whether latest change is positive or negative
            bg_color = '#089981' if last_close >= prev_close else '#F23645'
            
            # Draw label on the right side of the axis
            # We use ax.annotate to place it exactly at the price level on the right edge
            ax.annotate(f"{last_close:.2f}",
                        xy=(1, last_close), xycoords=('axes fraction', 'data'),
                        xytext=(3, 0), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, 
                                  edgecolor=bg_color, alpha=1.0),
                        color='white', fontsize=9, fontweight='bold',
                        va='center', ha='left', zorder=100)
            
            # Draw a horizontal line matching the price label
            ax.axhline(last_close, color=bg_color, linestyle='-', 
                       linewidth=0.8, alpha=0.8, zorder=99)
        except Exception as e:
            logger.debug(f"Current price label error: {e}")

    # ─────────────── Utility Methods ───────────────

    def chart_to_base64(self, chart_bytes: bytes) -> str:
        return base64.b64encode(chart_bytes).decode('utf-8')

    def resize_for_low_res(self, chart_bytes: bytes, max_dim: int = 768) -> bytes:
        """Resize for VLM analysis. 
        2026 SOTA models like Gemini 3.1 Pro handle 768px-1024px with perfect precision.
        Using 768px as default balance between cost and visual fidelity.
        """
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
