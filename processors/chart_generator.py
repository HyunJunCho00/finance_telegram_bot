"""Chart generator  structure-only overlays for VLM analysis.

Design principle:
  Charts show STRUCTURE (patterns, trendlines, levels) only.
  All indicator VALUES (RSI, MACD, OBV, etc.) go via text  VLM can't read numbers accurately.

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
  - SWING: 4h candles, last ~12 months
  - POSITION: 1d candles, last ~60 months
"""

import pandas as pd
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
import base64
import gc
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
from config.settings import get_settings, TradingMode
from loguru import logger
from scipy.signal import argrelextrema
from processors.math_engine import MathEngine

math_engine = MathEngine()


class ChartGenerator:
    """Generate structure-focused candlestick charts for VLM analysis."""

    def __init__(self):
        settings = get_settings()
        self.width = settings.CHART_IMAGE_WIDTH
        self.height = settings.CHART_IMAGE_HEIGHT
        self.dpi = settings.CHART_IMAGE_DPI

    @staticmethod
    def _lane_lookback_months(mode: TradingMode) -> int:
        return int(get_settings().SWING_HISTORY_MONTHS)

    @staticmethod
    def _tail_candles_for_rule(months: int, resample_rule: str) -> int:
        rule = str(resample_rule).upper()
        if rule.startswith("4H"):
            return max(30, int(round(months * 365.25 / 12.0 * 6.0)))
        if rule.startswith("1D"):
            return max(20, int(round(months * 365.25 / 12.0)))
        if rule.startswith("1W"):
            return max(12, int(round(months * 52.1775 / 12.0)))
        return 200

    @staticmethod
    def _lane_visual_window(df: pd.DataFrame, months: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Return a shared calendar window so lane panels start from the same date."""
        if df is None or df.empty or 'timestamp' not in df.columns:
            return None, None

        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dropna()
        if ts.empty:
            return None, None

        window_end = ts.max()
        window_start = (window_end - pd.DateOffset(months=months)).normalize()
        return window_start, window_end

    def generate_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str,
                       mode: TradingMode = TradingMode.SWING,
                       timeframe: Optional[str] = None,
                       liquidation_df: Optional[pd.DataFrame] = None,
                       cvd_df: Optional[pd.DataFrame] = None,
                       funding_df: Optional[pd.DataFrame] = None,
                       df_4h: Optional[pd.DataFrame] = None,
                       df_1d: Optional[pd.DataFrame] = None,
                       df_1w: Optional[pd.DataFrame] = None,
                       prefer_lane: bool = True) -> Optional[bytes]:
        """Generate structure chart for any mode or specific timeframe."""
        config = self._get_mode_config(mode, timeframe)
        # Filter analysis data for visual clarity (active elements only)
        current_price = float(df['close'].iloc[-1])
        analysis = self._filter_active_elements(current_price, analysis)

        timeframe_norm = (timeframe or "").lower().strip()
        is_lane_chart = prefer_lane and timeframe_norm in ("", "4h")
        if is_lane_chart:
            return self._generate_lane_chart(
                df, analysis, symbol, mode,
                liquidation_df=liquidation_df,
                cvd_df=cvd_df,
                funding_df=funding_df,
                df_4h=df_4h,
                df_1d=df_1d,
                df_1w=df_1w,
            )

        return self._generate_structure_chart(df, analysis, symbol, config,
                                              liquidation_df, cvd_df, funding_df,
                                              df_4h=df_4h, df_1d=df_1d, df_1w=df_1w, mode=mode)

    def _get_mode_config(self, mode: TradingMode, timeframe: Optional[str] = None) -> Dict:
        """Per-mode or per-timeframe chart configuration."""
        # 1. Base config determined by Mode (for Overlays / MTFA)
        # Swing config (4h execution + 1w context)
        if True:
            config = {
                'resample_rule': '4h',
                'tail_candles': 600,    # ~3.5 months (Q4 2025 ~ 2026-03)
                'min_candles': 30,
                'title_suffix': '4H SWING',
                'fib_tf': '1d',          # Fib based on Daily trend for reliable pullbacks
                'structure_tfs': ['1d', '4h'], # Daily main boundaries + 4H local channels
                'swing_tf': '4h',        # Liquidity levels where other swing traders put stops
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
            elif tf in ('1h', '2h', '4h', '6h', '8h', '12h'): tail = 200
            elif tf in ('1d', '3d'): tail = 365      # Swing needs 12 months context
            elif tf in ('1w', '1M'): tail = 260      # Macro needs 5 years context
            config['tail_candles'] = tail
            
            # For VLM display, label the title correctly
            config['title_suffix'] = f"{tf.upper()} (MTFA: {config['title_suffix']})"
            config['is_custom'] = True
            
            # NOTE: We intentionally DO NOT OVERRIDE 'fib_tf', 'structure_tfs', and 'swing_tf'.
            # They remain locked to the HTF (e.g. '1d' or '1w') defined by the Trade Mode.

        return config

    def _get_lane_panel_configs(self, mode: TradingMode) -> List[Dict]:
        lookback_months = self._lane_lookback_months(mode)
        execution_profile = bool(getattr(get_settings(), 'use_execution_philosophy', False))
        return [
            {
                'resample_rule': '1D',
                'tail_candles': self._tail_candles_for_rule(lookback_months, '1D'),
                'min_candles': 40,
                'title_suffix': f'TOP 1D STRUCTURE ({lookback_months}M)',
                'panel_label': f'Top: 1D Structure ({lookback_months}M)',
                'fib_tf': '1d',
                'structure_tfs': ['1d'],
                'swing_tf': None,
                'is_custom': False,
                'draw_pivots': False,
                'draw_market_structure_labels': False,
                'draw_structure_events': False,
                'draw_ema200': True,
                'draw_diagonal_lines': True,
                'draw_fibonacci': True,
                'draw_swing_levels': False,
                'draw_confluence': True,
                'draw_execution_plan': False,
                'draw_volume_profile': execution_profile,
                'draw_liquidations': False,
                'draw_fvg': execution_profile,
                'draw_macro_order_blocks': execution_profile,
                'draw_avwap': execution_profile,
                'draw_macro_alpha': False,
                'draw_session_breaks': False,
            },
            {
                'resample_rule': '4h',
                'tail_candles': self._tail_candles_for_rule(lookback_months, '4h'),
                'min_candles': 40,
                'title_suffix': f'BOTTOM 4H EXECUTION ({lookback_months}M)',
                'panel_label': f'Bottom: 4H Execution ({lookback_months}M)',
                'fib_tf': '4h',
                'structure_tfs': ['4h'],
                'swing_tf': '4h',
                'is_custom': False,
                'draw_pivots': False,
                'draw_market_structure_labels': False,
                'draw_structure_events': True,
                'draw_ema200': True,
                'draw_diagonal_lines': True,
                'draw_fibonacci': True,
                'draw_swing_levels': True,
                'draw_confluence': True,
                'draw_execution_plan': True,
                'draw_volume_profile': execution_profile,
                'draw_liquidations': False,
                'draw_fvg': execution_profile,
                'draw_macro_order_blocks': execution_profile,
                'draw_avwap': execution_profile,
                'draw_macro_alpha': False,
                'draw_session_breaks': True,
            },
        ]

    def _generate_lane_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str,
                             mode: TradingMode,
                             liquidation_df: Optional[pd.DataFrame] = None,
                             cvd_df: Optional[pd.DataFrame] = None,
                             funding_df: Optional[pd.DataFrame] = None,
                             df_4h: Optional[pd.DataFrame] = None,
                             df_1d: Optional[pd.DataFrame] = None,
                             df_1w: Optional[pd.DataFrame] = None) -> Optional[bytes]:
        panel_configs = self._get_lane_panel_configs(mode)
        panel_images: List[bytes] = []
        lookback_months = self._lane_lookback_months(mode)
        window_start, window_end = self._lane_visual_window(df, lookback_months)

        for panel_cfg in panel_configs:
            panel_cfg = dict(panel_cfg)
            if window_start is not None:
                panel_cfg['visual_window_start'] = window_start
            if window_end is not None:
                panel_cfg['visual_window_end'] = window_end
            img = self._generate_structure_chart(
                df, analysis, symbol, panel_cfg,
                liquidation_df=liquidation_df,
                cvd_df=cvd_df,
                funding_df=funding_df,
                df_4h=df_4h,
                df_1d=df_1d,
                df_1w=df_1w,
                mode=mode,
            )
            if img:
                panel_images.append(img)

        if not panel_images:
            return None
        if len(panel_images) == 1:
            return panel_images[0]
        return self._stack_images_vertical(
            panel_images,
            title=(
                f"{symbol} {mode.value.upper()} | "
                f"Top: {'1D Structure' if mode == TradingMode.SWING else '1W Structure'} / "
                f"Bottom: {'4H Execution' if mode == TradingMode.SWING else '1D Execution'} | "
                f"Lookback: {lookback_months}M"
            ),
        )

    def _stack_images_vertical(self, images: List[bytes], title: str) -> Optional[bytes]:
        try:
            pil_images = [Image.open(BytesIO(img)).convert('RGBA') for img in images]
            max_width = max(img.width for img in pil_images)
            title_height = 44
            separator = 8
            total_height = title_height + sum(img.height for img in pil_images) + separator * max(len(pil_images) - 1, 0)

            canvas = Image.new('RGBA', (max_width, total_height), 'white')
            draw = ImageDraw.Draw(canvas)
            try:
                title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            except Exception:
                title_font = ImageFont.load_default()

            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text(((max_width - title_width) / 2, 12), title, fill='#111111', font=title_font)

            y = title_height
            for img in pil_images:
                x = (max_width - img.width) // 2
                canvas.alpha_composite(img, (x, y))
                y += img.height + separator

            buf = BytesIO()
            canvas.convert('RGB').save(buf, format='PNG')
            buf.seek(0)
            result = buf.getvalue()
            # [OOM FIX] PIL 이미지 명시적 해제
            for img in pil_images:
                img.close()
            canvas.close()
            del pil_images, canvas
            gc.collect()
            return result
        except Exception as e:
            logger.error(f"Lane chart stack error: {e}")
            return None

    def _filter_active_elements(self, current_price: float, analysis: Dict) -> Dict:
        """Filter analysis data to keep only elements relevant to current price action.
        Reduces visual noise and prevents VLM 'encyclopedia' overexposure.
        """
        if not analysis:
            return {}
        
        filtered = copy.deepcopy(analysis)
        
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

        # 2. Keep only the strongest/nearest confluence zones
        zones = filtered.get('confluence_zones', [])
        if isinstance(zones, list) and zones:
            zones = [
                z for z in zones
                if isinstance(z, dict)
                and isinstance(z.get('price_low'), (int, float))
                and isinstance(z.get('price_high'), (int, float))
            ]
            zones.sort(
                key=lambda z: (
                    abs((((float(z['price_low']) + float(z['price_high'])) / 2.0) - current_price)),
                    -float(z.get('strength', 0) or 0),
                )
            )
            filtered['confluence_zones'] = zones[:2]

        # 3. Structure Labels (HH/HL/LH/LL)
        # Usually handled by the drafting logic itself, but we can prune history if needed
        # (Already handled in _draw_market_structure_labels via 'order' and visual slicing)
        
        return filtered

    def _generate_structure_chart(self, df: pd.DataFrame, analysis: Dict,
                                   symbol: str, config: Dict,
                                   liquidation_df: Optional[pd.DataFrame] = None,
                                   cvd_df: Optional[pd.DataFrame] = None,
                                   funding_df: Optional[pd.DataFrame] = None,
                                   df_4h: Optional[pd.DataFrame] = None,
                                   df_1d: Optional[pd.DataFrame] = None,
                                   df_1w: Optional[pd.DataFrame] = None,
                                   mode: TradingMode = TradingMode.SWING) -> Optional[bytes]:
        """Core chart generation  candlesticks + structure overlays only."""
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
            if rule_upper.startswith('4H') and df_4h is not None and not df_4h.empty:
                hist_df = df_4h.copy()
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
                hist_df = hist_df.set_index('timestamp')
                full_resampled = pd.concat([hist_df, full_resampled])
                full_resampled = full_resampled[~full_resampled.index.duplicated(keep='last')].sort_index()
            elif rule_upper.startswith('1D') and df_1d is not None and not df_1d.empty:
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

            visual_window_start = config.get('visual_window_start')
            visual_window_end = config.get('visual_window_end')
            if visual_window_start is not None:
                visual_window_start = pd.Timestamp(visual_window_start)
                if visual_window_start.tzinfo is None:
                    visual_window_start = visual_window_start.tz_localize('UTC')
            if visual_window_end is not None:
                visual_window_end = pd.Timestamp(visual_window_end)
                if visual_window_end.tzinfo is None:
                    visual_window_end = visual_window_end.tz_localize('UTC')

            visual_window_df = None
            if visual_window_end is not None:
                full_resampled = full_resampled.loc[full_resampled.index <= visual_window_end]

            if visual_window_start is not None or visual_window_end is not None:
                visual_window_df = full_resampled.copy()
                if visual_window_start is not None:
                    visual_window_df = visual_window_df.loc[visual_window_df.index >= visual_window_start]
                if visual_window_end is not None:
                    visual_window_df = visual_window_df.loc[visual_window_df.index <= visual_window_end]

                calc_tail = max(len(visual_window_df) + 220, config['min_candles'] + 220)
                full_resampled = full_resampled.tail(calc_tail)
            else:
                # Trim to the exact required tail length to avoid overloading matplotlib, plus a buffer for EMAs
                full_resampled = full_resampled.tail(config['tail_candles'] + 220)

            if len(full_resampled) < min(config['min_candles'], 5):
                logger.warning(f"Not enough candles: {len(full_resampled)} < {config['min_candles']}")
                return None

            # 2. Calculate Indicators on FULL history
            import pandas_ta_classic as ta
            full_resampled['ema200'] = ta.ema(full_resampled['close'], length=min(200, len(full_resampled)-1))

            # 3. Get Trading Mode for Header context
            settings = get_settings()
            trading_mode = getattr(settings, 'trading_mode', TradingMode.SWING)
            show_cvd_panel = bool(getattr(settings, 'CHART_SHOW_CVD_PANEL', False))
            show_cvd_overlay = bool(getattr(settings, 'CHART_SHOW_CVD_OVERLAY', False))
            show_oi_panel = bool(getattr(settings, 'CHART_SHOW_OI_PANEL', False))
            show_funding_panel = bool(getattr(settings, 'CHART_SHOW_FUNDING_PANEL', False))
            
            # 4. Slice for VISUAL window
            if visual_window_df is not None:
                chart_df = full_resampled.loc[full_resampled.index.intersection(visual_window_df.index)].copy()
            else:
                chart_df = full_resampled.tail(config['tail_candles']).copy()
            chart_df.index.name = 'Date'

            if len(chart_df) < min(config['min_candles'], 5):
                logger.warning(f"Not enough candles in visual window: {len(chart_df)} < {config['min_candles']}")
                return None

            # 4. Integrate CVD if provided
            apds = []
            _cvd_added = False  # tracks whether a CVD panel was actually appended to mpf
            if show_cvd_panel and cvd_df is not None and not cvd_df.empty:
                cvd = cvd_df.copy()
                if pd.api.types.is_numeric_dtype(cvd['timestamp']):
                    cvd['timestamp'] = pd.to_datetime(cvd['timestamp'], unit='ms', utc=True).dt.floor('min')
                else:
                    cvd['timestamp'] = pd.to_datetime(cvd['timestamp'], utc=True).dt.floor('min')
                cvd = cvd.set_index('timestamp')
                
                # [FIX V14.7] Robust column handling: Binance uses 'taker_', tools use 'whale_'
                # Standardize to ensure delta calculation works regardless of source
                col_map = {
                    'taker_buy_volume': 'whale_buy_vol',
                    'taker_sell_volume': 'whale_sell_vol',
                    'buy_vol': 'whale_buy_vol',
                    'sell_vol': 'whale_sell_vol'
                }
                cvd = cvd.rename(columns={k: v for k, v in col_map.items() if k in cvd.columns})
                cvd = cvd.loc[:, ~cvd.columns.duplicated()] # [FIX] Ensure no duplicate columns after renaming
                
                # Resample CVD to match OHLCV rule
                cvd_resampled = cvd.resample(config['resample_rule']).sum(min_count=1)
                
                # [FIX V14.7] Calculate Cumulative Delta on FULL available history 
                # before alignment to preserve long-term trend visibility
                if 'whale_buy_vol' in cvd_resampled.columns and 'whale_sell_vol' in cvd_resampled.columns:
                    buy_vol = cvd_resampled['whale_buy_vol'].fillna(0)
                    sell_vol = cvd_resampled['whale_sell_vol'].fillna(0)
                    
                    delta = buy_vol - sell_vol
                    cvd_resampled['delta'] = delta
                    cvd_resampled['cvd_acc'] = delta.cumsum()
                else:
                    # Fallback to 'volume_delta' or 'delta' if direct volumes missing
                    delta_col = next((c for c in ['volume_delta', 'delta'] if c in cvd_resampled.columns), None)
                    if delta_col:
                        cvd_resampled['delta'] = cvd_resampled[delta_col].fillna(0)
                        cvd_resampled['cvd_acc'] = cvd_resampled['delta'].cumsum()
                    else:
                        cvd_resampled['delta'] = pd.Series(0, index=cvd_resampled.index)
                        cvd_resampled['cvd_acc'] = pd.Series(0, index=cvd_resampled.index)
                
                # Align with chart_df index (Visual Window)
                cvd_aligned = cvd_resampled.reindex(chart_df.index)
                cvd_acc_plot = cvd_aligned['cvd_acc'].ffill().bfill()
                cvd_delta_plot = cvd_aligned['delta'].fillna(0)
                
                # Panel 2: (Price 0, Vol 1, CVD Panel 2)
                apds.append(mpf.make_addplot(cvd_acc_plot, panel=2, 
                                           color='#8E44AD', width=1.5, 
                                           ylabel='CVD', secondary_y=False))
                colors = ['#27AE60' if d >= 0 else '#C0392B' for d in cvd_delta_plot]
                apds.append(mpf.make_addplot(cvd_delta_plot, panel=2,
                                           type='bar', color=colors, alpha=0.3,
                                           secondary_y=True))
                _cvd_added = True  # CVD panel successfully registered

            # 5. Integrate OI and Funding if provided
            _has_oi = False
            _has_funding = False
            if (show_oi_panel or show_funding_panel) and funding_df is not None and not funding_df.empty:
                fnd = funding_df.copy()
                if pd.api.types.is_numeric_dtype(fnd['timestamp']):
                    fnd['timestamp'] = pd.to_datetime(fnd['timestamp'], unit='ms', utc=True).dt.floor('min')
                else:
                    fnd['timestamp'] = pd.to_datetime(fnd['timestamp'], utc=True).dt.floor('min')
                fnd = fnd.set_index('timestamp')
                
                # Safely determine what columns are available
                has_oi = show_oi_panel and ('open_interest' in fnd.columns)
                has_funding = show_funding_panel and ('funding_rate' in fnd.columns)
                
                agg_map = {}
                if has_oi: agg_map['open_interest'] = 'last'
                if has_funding: agg_map['funding_rate'] = 'mean'
                
                if not agg_map:
                    # No usable columns in funding_df at all
                    funding_df = None
                else:
                    fnd_resampled = fnd.resample(config['resample_rule']).agg(agg_map)
                    fnd_aligned = fnd_resampled.reindex(chart_df.index)
                    
                    # Compute base panel: if CVD was added it occupies panel=2,
                    # so OI starts at panel=3; otherwise OI starts at panel=2.
                    _oi_panel = 3 if _cvd_added else 2

                    # ------------ OI Panel only if column exists ------------
                    if has_oi:
                        fnd_oi_plot = fnd_aligned['open_interest'].ffill()
                        apds.append(mpf.make_addplot(fnd_oi_plot, panel=_oi_panel,
                                                   color='#2980B9', width=1.2,
                                                   ylabel='OI', secondary_y=False))
                        _has_oi = True
                    else:
                        # ------ No OI flag for panel count correction below ------
                        _has_oi = False

                    # -------- Funding Rate Tape only if column exists --------
                    # Funding always goes one panel above OI; if OI is absent it takes OI's slot.
                    if has_funding:
                        fnd_rate_plot = fnd_aligned['funding_rate'].fillna(0)
                        f_colors = ['#27AE60' if r > 0.0001 else '#C0392B' if r < -0.0001 else '#BDC3C7'
                                   for r in fnd_rate_plot]
                        tape_panel = (_oi_panel + 1) if has_oi else _oi_panel
                        apds.append(mpf.make_addplot(fnd_rate_plot, panel=tape_panel,
                                                   type='bar', color=f_colors, alpha=0.8,
                                                   ylabel='Fnd', secondary_y=False))
                        _has_funding = True

            # Dynamic Theme Selection
            settings = get_settings()
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

            # ------- Professional Font Config Legibility Boost -------
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['text.color'] = text_color
            plt.rcParams['axes.labelcolor'] = text_color
            plt.rcParams['xtick.color'] = '#787B86'
            plt.rcParams['ytick.color'] = '#787B86'

            # ---- Plot Adaptive panel layout (Price 0 Vol 1 CVD 2 OI 2/3 Funding 3/4 ) ----
            # ---- Use _cvd_added (not cvd_df is not None) empty cvd_df must not reserve a panel ----
            num_panels = 2  # Basic: Price + Vol
            if _cvd_added: num_panels += 1
            if _has_oi:
                num_panels += 1
            if _has_funding:
                num_panels += 1

            p_ratios = [5, 1.2]  # Price, Vol
            if _cvd_added: p_ratios.append(1.8)   # CVD panel
            if _has_oi:
                p_ratios.append(1.5)  # OI
            if _has_funding:
                p_ratios.append(0.6)  # Funding Tape
                
            logger.debug(f"Calling mpf.plot with {len(apds)} addplots, {num_panels} panels, ratios {p_ratios}")
            warn_too_much_data = len(chart_df) + 1

            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                addplot=apds,
                title='',  # We will draw a custom title
                ylabel='Price',
                figsize=(self.width / self.dpi, (self.height / self.dpi) * (1.3 if num_panels > 3 else 1.0)),
                panel_ratios=tuple(p_ratios),
                datetime_format='%Y-%m-%d',
                xrotation=0,
                warn_too_much_data=warn_too_much_data,
                tight_layout=False,  # We'll use subplots_adjust for margins
                returnfig=True
            )
            
            # ------------------ Layout Optimization ------------------
            fig.subplots_adjust(top=0.92, right=0.92, left=0.08, bottom=0.08)
            price_ax = axes[0]
            
            # --------------- Enhanced Title & Metadata ---------------
            price_ax.text(0.01, 1.05, f"{symbol} | {df['close'].iloc[-1]:.2f}",
                         transform=price_ax.transAxes, fontsize=14, fontweight='bold', 
                         color=text_color, ha='left')

            # --------- Add Background Watermark (Big Symbol) ---------
            price_ax.text(0.5, 0.5, symbol.split('USDT')[0],
                         transform=price_ax.transAxes, fontsize=80, fontweight='bold',
                         color=text_color, alpha=0.03, ha='center', va='center', zorder=0)

            # ---- Overlay 1 Pivot Points (turning points) SWING only ----
            if config.get('draw_pivots', True):
                self._draw_pivot_points(price_ax, chart_df)

            # ---- Overlay 2 Market Structure Labels (HH/HL/LH/LL) ----
            if config.get('draw_market_structure_labels', True):
                self._draw_market_structure_labels(price_ax, chart_df)
            if config.get('draw_structure_events', False):
                tf_label = str(config.get('fib_tf') or config.get('structure_tfs', [''])[0])
                market_structure = (analysis.get('market_structure', {}) or {}).get(tf_label)
                self._draw_structure_events(price_ax, chart_df, market_structure, tf_label)

            # ----------------- Overlay 3 EMA200 line -----------------
            if config.get('draw_ema200', True):
                self._draw_ema200(price_ax, chart_df)

            # ---- Overlay 4 Diagonal Trendlines (support/resistance) ----
            structure = analysis.get('structure', {}) or {}
            if config.get('draw_diagonal_lines', True):
                for tf in config['structure_tfs']:
                    self._draw_diagonal_line(price_ax, chart_df,
                                             structure.get(f'support_{tf}'),
                                             'support_price', '#089981', theme, text_color)
                    self._draw_diagonal_line(price_ax, chart_df,
                                             structure.get(f'resistance_{tf}'),
                                             'resistance_price', '#F23645', theme, text_color)

            # -------------- Overlay 5 Fibonacci Levels --------------
            fib = None
            if config.get('draw_fibonacci', True):
                fib = (analysis.get('fibonacci', {}) or {}).get(config['fib_tf'])
                if not fib:
                    fib_data = analysis.get('fibonacci', {}) or {}
                    for tf_key in ['4h', '1d', '1w', '15m']:
                        if tf_key in fib_data and fib_data[tf_key]:
                            fib = fib_data[tf_key]
                            break
                self._draw_fibonacci(price_ax, chart_df, fib, config.get('fib_tf'))
            if config.get('draw_confluence', False):
                self._draw_confluence_zones(
                    price_ax,
                    chart_df,
                    analysis.get('confluence_zones', []) or [],
                    focus_tfs=config.get('structure_tfs', []),
                )

            # ------ Overlay 6 Swing High/Low (Liquidity Levels) ------
            if config.get('draw_swing_levels', False) and config.get('swing_tf'):
                swing = (analysis.get('swing_levels', {}) or {}).get(config['swing_tf'])
                self._draw_swing_levels(price_ax, chart_df, swing, config.get('swing_tf'))
            self._draw_scenario_liquidity_map(price_ax, chart_df, analysis)
            if config.get('draw_execution_plan', False):
                execution_plan = self._derive_execution_plan(chart_df, analysis, config)
                self._draw_execution_plan(price_ax, chart_df, execution_plan)

            # ---- Overlay 7 Volume Profile Histogram (right side) ----
            if config.get('draw_volume_profile', True):
                self._draw_volume_profile_histogram(price_ax, chart_df)

            # ------- Overlay 8 Liquidation Markers SWING only -------
            if config.get('draw_liquidations', True) and liquidation_df is not None and not liquidation_df.empty:
                self._draw_liquidation_markers(price_ax, chart_df, liquidation_df)

            # ------------ Overlay 9 Fair Value Gaps (FVG) ------------
            if config.get('draw_fvg', True):
                self._draw_fair_value_gaps(price_ax, chart_df, trading_mode)

            # -------- NEW Overlay 10 Macro Order Blocks (OB) --------
            # count=4 → returns up to 8 candidates (obs[:count*2])
            # Ensures enough pool to guarantee 1 above + 1 below current price
            # even in strongly trending markets where recent OBs cluster on one side
            if config.get('draw_macro_order_blocks', True):
                macro_obs = math_engine.calculate_macro_order_blocks(full_resampled, count=4)
                self._draw_macro_order_blocks(price_ax, chart_df, macro_obs)

            # ------------- NEW Overlay 11 Anchored VWAP -------------
            if config.get('draw_avwap', True):
                self._draw_anchored_vwap(price_ax, chart_df, full_resampled)

            # Overlay 12: optional CVD alpha divergence markers
            if config.get('draw_macro_alpha', show_cvd_overlay) and cvd_df is not None and not cvd_df.empty:
                div = math_engine.detect_macro_divergences(full_resampled, cvd_df)
                self._draw_macro_alpha_markers(price_ax, chart_df, div)

            # ------------- PRO Overlay 13 Header Legend -------------
            self._draw_header_legend(price_ax, chart_df, symbol, config, text_color)

            # ---- PRO Overlay 14 Session Breaks (Day dividers) SWING only ----
            if config.get('draw_session_breaks', True):
                self._draw_session_breaks(price_ax, chart_df)
            
            # ---- PRO Overlay 15 Current Price Label (On Y axis) ----
            self._draw_current_price_label(price_ax, chart_df, text_color)

            # ----------- Lock Y axis to candlestick range -----------
            # Must happen AFTER all overlays so nothing autoscales the axis
            _y_lo = float(chart_df['low'].min())
            _y_hi = float(chart_df['high'].max())
            _margin = (_y_hi - _y_lo) * 0.04
            price_ax.set_ylim(_y_lo - _margin, _y_hi + _margin)

            # Save with high-quality settings
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='#CCCCCC')
            buf.seek(0)
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Chart error for {symbol} ({config['title_suffix']}): {e}")
            return None

        finally:
            # [OOM FIX] 예외 여부와 관계없이 matplotlib figure 반드시 해제
            try:
                plt.close(fig)
            except Exception:
                pass
            try:
                plt.close('all')
            except Exception:
                pass
            # 로컬 대형 DataFrame 참조 명시적 해제
            try:
                del full_resampled, chart_df, tmp
            except Exception:
                pass
            gc.collect()

    # ---------------- Overlay Drawing Methods ----------------

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
            # ---- Then render only the LAST 3 swing highs and lows enough to read current structure ----
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
        """Draw diagonal support/resistance trendline from first TD point to right edge.

        Fix: The line starts at ts1's actual chart x-position (not always x=0).
        When TD points are recent (few weeks), starting from x=0 (12 months ago)
        caused massive backward extrapolation → wrong y values → clipped to chart
        corner → visually broken steep lines.
        """
        if not line_info or 'point1' not in line_info or 'point2' not in line_info:
            return

        try:
            pt1 = line_info['point1']
            pt2 = line_info['point2']

            ts1, y1 = pt1
            ts2, y2 = pt2

            # chart_df index is a UTC-aware DatetimeIndex
            chart_start = chart_df.index[0]
            chart_end   = chart_df.index[-1]

            # Ensure all timestamps are timezone-aware (UTC)
            if ts1.tzinfo is None: ts1 = ts1.tz_localize('UTC')
            if ts2.tzinfo is None: ts2 = ts2.tz_localize('UTC')
            if chart_start.tzinfo is None: chart_start = chart_start.tz_localize('UTC')
            if chart_end.tzinfo is None:   chart_end   = chart_end.tz_localize('UTC')

            # Calculate slope in price-per-second
            dt = (ts2 - ts1).total_seconds()
            if dt == 0:
                return
            slope_sec = (y2 - y1) / dt

            # --------------- Find line start position ---------------
            # If ts1 is WITHIN the chart window, start the line at ts1's x-position.
            # If ts1 is BEFORE chart_start, start at x=0 (left edge) with the
            # ---- correctly extrapolated y same math as before no change in that case. ----
            chart_ts = chart_df.index
            try:
                # Normalize timezone for searchsorted comparison
                tz = chart_ts.tz
                ts1_cmp = ts1.tz_convert(tz) if (tz is not None and ts1.tzinfo is not None) else ts1
                x_start = int(chart_ts.searchsorted(ts1_cmp, side='left'))
                x_start = max(0, min(x_start, len(chart_df) - 1))
            except Exception:
                x_start = 0  # Fallback: left edge

            # y value at x_start
            ts_at_x_start = chart_ts[x_start]
            if ts_at_x_start.tzinfo is None: ts_at_x_start = ts_at_x_start.tz_localize('UTC')
            y_x_start = y1 + slope_sec * (ts_at_x_start - ts1).total_seconds()

            # y value at right edge
            y_end = y1 + slope_sec * (chart_end - ts1).total_seconds()

            y_lo   = float(chart_df['low'].min())
            y_hi   = float(chart_df['high'].max())
            margin = (y_hi - y_lo) * 0.05

            # Discard if both endpoints are entirely outside the visible price range
            if max(y_x_start, y_end) < y_lo - margin or min(y_x_start, y_end) > y_hi + margin:
                return

            x_vals = np.array([x_start, len(chart_df) - 1], dtype=float)
            y_plot = np.clip(np.array([y_x_start, y_end], dtype=float),
                             y_lo - margin, y_hi + margin)

            kind       = "SUP" if "support" in str(line_info.get('type', value_key)).lower() else "RES"
            tf         = str(line_info.get('timeframe') or '').upper()
            label_text = f"{tf} {kind} TREND".strip()

            # Solid line for strong VLM visibility
            ax.plot(x_vals, y_plot, color=color, linewidth=2.0,
                    linestyle='-', alpha=0.8, zorder=10)

            # Draw TD anchor dots at pt1 and pt2 so VLM can see the trendline origin
            for ts_anchor, y_anchor in [(ts1, y1), (ts2, y2)]:
                try:
                    ts_cmp = ts_anchor.tz_convert(chart_ts.tz) if (chart_ts.tz is not None and ts_anchor.tzinfo is not None) else ts_anchor
                    x_anchor = int(chart_ts.searchsorted(ts_cmp, side='left'))
                    if 0 <= x_anchor < len(chart_df):
                        ax.scatter([x_anchor], [y_anchor], color=color, s=20,
                                   zorder=11, alpha=0.85, marker='o')
                except Exception:
                    pass

            ax.text(x_vals[-1] - 1.5, y_plot[-1], f' {label_text}', color=color,
                    fontsize=8, fontweight='bold', va='bottom', ha='right',
                    bbox=dict(facecolor=text_color if theme == 'light_premium' else '#131722',
                              alpha=0.7, edgecolor='none', pad=1))

        except Exception as e:
            logger.debug(f"Trendline draw error: {e}")
            return

    def _draw_fibonacci(self, ax, chart_df: pd.DataFrame, fib: Optional[Dict], fib_tf: Optional[str] = None):
        """Draw Fibonacci retracement horizontal lines with anchor markers.
        Anchor markers are critical: VLM schema now requires explicit anchor_high/anchor_low,
        so the anchors must be visually identifiable in the chart image.
        """
        if not fib:
            return

        tf_prefix = f"{str(fib_tf).upper()} " if fib_tf else ""
        fib_styles = {
            'fib_500': ('#F39C12', 0.75, f'{tf_prefix}FIB 50.0'),
            'fib_618': ('#F39C12', 0.80, f'{tf_prefix}FIB 61.8'),
            'fib_705': ('#F39C12', 0.72, f'{tf_prefix}FIB 70.5'),
            'fib_786': ('#F39C12', 0.65, f'{tf_prefix}FIB 78.6'),
        }

        n = len(chart_df)
        for key, (color, alpha, label) in fib_styles.items():
            val = fib.get(key)
            if isinstance(val, (int, float)):
                ax.axhline(val, color=color, linestyle='--', linewidth=1.2,
                           alpha=alpha, zorder=2)
                ax.text(n - 0.5, val, f' {label}',
                        color=color, fontsize=8, va='center', ha='left',
                        alpha=1.0, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))

        # Draw anchor markers so VLM can identify exact swing_high / swing_low used for Fib
        # Without these markers, VLM must guess anchors → fibonacci_context.anchor_high/low unreliable
        anchor_high = fib.get('swing_high')
        anchor_low = fib.get('swing_low')
        if isinstance(anchor_high, (int, float)):
            ax.axhline(anchor_high, color='#F39C12', linestyle='-', linewidth=1.0,
                       alpha=0.5, zorder=3)
            ax.text(1, anchor_high, f' {tf_prefix}FIB HIGH',
                    color='#F39C12', fontsize=7, va='bottom', ha='left',
                    fontweight='bold', alpha=0.9)
        if isinstance(anchor_low, (int, float)):
            ax.axhline(anchor_low, color='#F39C12', linestyle='-', linewidth=1.0,
                       alpha=0.5, zorder=3)
            ax.text(1, anchor_low, f' {tf_prefix}FIB LOW',
                    color='#F39C12', fontsize=7, va='top', ha='left',
                    fontweight='bold', alpha=0.9)

    def _draw_swing_levels(self, ax, chart_df: pd.DataFrame, swing: Optional[Dict], tf_label: Optional[str] = None):
        """Draw horizontal lines and shaded zones at swing highs/lows (liquidity pools)."""
        if not swing:
            return

        tf_prefix = f"{str(tf_label).upper()} " if tf_label else ""
        n = len(chart_df)
        current_price = float(chart_df['close'].iloc[-1])
        chart_low = float(chart_df['low'].min())
        chart_high = float(chart_df['high'].max())
        dedupe_gap = max(current_price * 0.004, (chart_high - chart_low) * 0.015)

        def _select_levels(levels: List[float]) -> List[float]:
            visible = []
            for level in levels or []:
                if not isinstance(level, (int, float)):
                    continue
                level = float(level)
                if level < chart_low or level > chart_high:
                    continue
                visible.append(level)

            visible = sorted(visible, key=lambda level: abs(level - current_price))
            selected: List[float] = []
            for level in visible:
                if any(abs(level - kept) <= dedupe_gap for kept in selected):
                    continue
                selected.append(level)
                if len(selected) >= 2:
                    break
            return selected

        for idx, high in enumerate(_select_levels(swing.get('swing_highs', []))):
            if isinstance(high, (int, float)):
                # Draw main line
                ax.axhline(high, color='#C0392B', linestyle='-', linewidth=0.8,
                           alpha=0.4, zorder=2)
                # Draw subtle shaded zone for "Liquidity Pool" (+/- 0.3% range)
                ax.axhspan(high * 0.997, high * 1.003, color='#C0392B', 
                           alpha=0.08, zorder=1)
                if idx == 0:
                    ax.text(2, high, f" {tf_prefix}SWING HIGH", color='#C0392B', 
                            fontsize=7, fontweight='bold', va='bottom', alpha=1.0,
                            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1))

        for idx, low in enumerate(_select_levels(swing.get('swing_lows', []))):
            if isinstance(low, (int, float)):
                # Draw main line
                ax.axhline(low, color='#27AE60', linestyle='-', linewidth=0.8,
                           alpha=0.4, zorder=2)
                # Draw subtle shaded zone for "Liquidity Pool" (+/- 0.3% range)
                ax.axhspan(low * 0.997, low * 1.003, color='#27AE60', 
                           alpha=0.08, zorder=1)
                if idx == 0:
                    ax.text(2, low, f" {tf_prefix}SWING LOW", color='#27AE60', 
                            fontsize=7, fontweight='bold', va='top', alpha=1.0,
                            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1))

    def _draw_structure_events(self, ax, chart_df: pd.DataFrame, market_structure: Optional[Dict], tf_label: str):
        """Draw execution-panel structure events such as CHoCH/MSB and last swing bounds."""
        if not market_structure:
            return

        tf_prefix = str(tf_label or '').upper()
        n = len(chart_df)

        last_high = market_structure.get('last_swing_high')
        if isinstance(last_high, (int, float)):
            ax.axhline(last_high, color='#C0392B', linestyle=':', linewidth=1.0, alpha=0.45, zorder=2)
            ax.text(n - 1, last_high, f' {tf_prefix} LAST HIGH', color='#C0392B',
                    fontsize=7, fontweight='bold', va='bottom', ha='right',
                    bbox=dict(facecolor='black', alpha=0.25, edgecolor='none', pad=1))

        last_low = market_structure.get('last_swing_low')
        if isinstance(last_low, (int, float)):
            ax.axhline(last_low, color='#27AE60', linestyle=':', linewidth=1.0, alpha=0.45, zorder=2)
            ax.text(n - 1, last_low, f' {tf_prefix} LAST LOW', color='#27AE60',
                    fontsize=7, fontweight='bold', va='top', ha='right',
                    bbox=dict(facecolor='black', alpha=0.25, edgecolor='none', pad=1))

        current_price = float(chart_df['close'].iloc[-1])
        choch = market_structure.get('choch')
        msb = market_structure.get('msb')
        msb_price = msb.get('price') if isinstance(msb, dict) else None
        if not isinstance(msb_price, (int, float)) and isinstance(msb, dict):
            msb_price = msb.get('broken_level')
        prices_too_close = (
            isinstance(choch, dict)
            and isinstance(choch.get('price'), (int, float))
            and isinstance(msb_price, (int, float))
            and abs(float(choch['price']) - float(msb_price)) <= (current_price * 0.004)
        )
        if isinstance(choch, dict) and isinstance(choch.get('price'), (int, float)):
            choch_type = str(choch.get('type', '')).lower()
            direction = 'BULLISH' if 'bullish' in choch_type else 'BEARISH'
            color = '#3498DB' if direction == 'BULLISH' else '#E74C3C'
            if not prices_too_close:
                ax.text(n - 2, float(choch['price']), f' {tf_prefix} {direction} CHOCH', color=color,
                        fontsize=8, fontweight='bold', va='center', ha='right',
                        bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1))

        if isinstance(msb, dict) and isinstance(msb_price, (int, float)):
            msb_type = str(msb.get('type', '')).lower()
            direction = 'BULLISH' if 'bullish' in msb_type else 'BEARISH'
            color = '#3498DB' if direction == 'BULLISH' else '#E74C3C'
            ax.text(n - 2, float(msb_price), f' {tf_prefix} {direction} MSB', color=color,
                    fontsize=8, fontweight='bold', va='center', ha='right',
                    bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1))

    def _draw_confluence_zones(self, ax, chart_df: pd.DataFrame, zones: List[Dict], focus_tfs: List[str]):
        """Draw up to two visible confluence zones for the active panel timeframes."""
        if not zones:
            return

        focus = {str(tf).lower() for tf in (focus_tfs or []) if tf}
        filtered: List[Dict] = []
        for zone in zones:
            zone_tfs = {str(tf).lower() for tf in zone.get('timeframes', [])}
            if not focus or focus.intersection(zone_tfs):
                filtered.append(zone)

        if not filtered:
            filtered = list(zones)

        chart_low = float(chart_df['low'].min())
        chart_high = float(chart_df['high'].max())
        visible: List[Dict] = []
        for zone in filtered:
            low = zone.get('price_low')
            high = zone.get('price_high')
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                continue
            if high < chart_low or low > chart_high:
                continue
            visible.append(zone)

        current_price = float(chart_df['close'].iloc[-1])
        visible.sort(key=lambda z: abs(((float(z['price_low']) + float(z['price_high'])) / 2.0) - current_price))

        for zone in visible[:2]:
            low = float(zone['price_low'])
            high = float(zone['price_high'])
            label_tfs = "/".join(str(tf).upper() for tf in zone.get('timeframes', [])[:2])
            label = f"{(label_tfs or 'HTF')} CONFLUENCE".strip()
            ax.axhspan(low, high, color='#F1C40F', alpha=0.14, zorder=1)
            ax.text(len(chart_df) - 1, high, f' {label}', color='#F1C40F',
                    fontsize=7, fontweight='bold', va='bottom', ha='right',
                    bbox=dict(facecolor='black', alpha=0.25, edgecolor='none', pad=1))

    def _draw_scenario_liquidity_map(self, ax, chart_df: pd.DataFrame, analysis: Dict):
        """Draw only the active scenario's liquidity hints with low visual noise."""
        scenario_engine = analysis.get('scenario_engine', {}) or {}
        if not isinstance(scenario_engine, dict) or not scenario_engine:
            return

        liquidity_map = scenario_engine.get('liquidity_map', {}) or {}
        active_setup = scenario_engine.get('active_setup', {}) or {}
        current_price = float(chart_df['close'].iloc[-1])

        equal_levels = liquidity_map.get('equal_levels', {}) or {}
        self._draw_equal_liquidity_levels(ax, chart_df, equal_levels, current_price)

        bpr_zone = liquidity_map.get('bpr_zone')
        self._draw_bpr_zone(ax, chart_df, bpr_zone, current_price)

        sr_flip = active_setup.get('sr_flip') or liquidity_map.get('sr_flip') or {}
        self._draw_sr_flip(ax, chart_df, sr_flip)

        trap_context = active_setup.get('trap_context') or liquidity_map.get('liquidity_sweep') or {}
        self._draw_liquidity_sweep_marker(ax, chart_df, trap_context)

        self._draw_scenario_revision_note(ax, scenario_engine)

    def _draw_equal_liquidity_levels(self, ax, chart_df: pd.DataFrame, equal_levels: Dict, current_price: float):
        """Draw at most one EQH and one EQL nearest to current price."""
        if not isinstance(equal_levels, dict):
            return

        chart_low = float(chart_df['low'].min())
        chart_high = float(chart_df['high'].max())
        clusters = []
        for label, color, items in (
            ('EQH', '#5DADE2', equal_levels.get('equal_highs', []) or []),
            ('EQL', '#5DADE2', equal_levels.get('equal_lows', []) or []),
        ):
            visible = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                price = item.get('price')
                if not isinstance(price, (int, float)):
                    continue
                price = float(price)
                if chart_low <= price <= chart_high:
                    visible.append((abs(price - current_price), price, int(item.get('touches', 0) or 0)))
            if visible:
                _, price, touches = sorted(visible, key=lambda row: row[0])[0]
                clusters.append((label, color, price, touches))

        n = len(chart_df)
        for label, color, price, touches in clusters:
            ax.axhline(price, color=color, linestyle=':', linewidth=0.9, alpha=0.22, zorder=2)
            ax.text(
                n - 1,
                price,
                f' {label} x{touches}'.strip(),
                color=color,
                fontsize=7,
                fontweight='bold',
                va='center',
                ha='right',
                bbox=dict(facecolor='black', alpha=0.16, edgecolor='none', pad=1),
            )

    def _draw_bpr_zone(self, ax, chart_df: pd.DataFrame, bpr_zone: Optional[Dict], current_price: float):
        """Draw only one visible BPR overlap zone near the active price."""
        if not isinstance(bpr_zone, dict):
            return
        low = bpr_zone.get('price_low')
        high = bpr_zone.get('price_high')
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            return

        low = float(low)
        high = float(high)
        chart_low = float(chart_df['low'].min())
        chart_high = float(chart_df['high'].max())
        if high < chart_low or low > chart_high:
            return
        if abs(((low + high) / 2.0) - current_price) / max(current_price, 1e-9) > 0.06:
            return

        ax.axhspan(low, high, color='#AF7AC5', alpha=0.08, zorder=1)
        ax.text(
            len(chart_df) - 1,
            high,
            ' BPR',
            color='#AF7AC5',
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right',
            bbox=dict(facecolor='black', alpha=0.18, edgecolor='none', pad=1),
        )

    def _draw_sr_flip(self, ax, chart_df: pd.DataFrame, sr_flip: Dict):
        """Draw only confirmed SR flips to avoid over-labeling."""
        if not isinstance(sr_flip, dict) or not sr_flip.get('confirmed'):
            return
        level = sr_flip.get('reference_level')
        if not isinstance(level, (int, float)):
            return

        status = str(sr_flip.get('status', 'none')).lower()
        color = '#48C9B0' if 'bullish' in status else '#F5B041'
        label = 'BULL SR FLIP' if 'bullish' in status else 'BEAR SR FLIP'
        ax.axhline(float(level), color=color, linestyle='--', linewidth=1.0, alpha=0.42, zorder=3)
        ax.text(
            2,
            float(level),
            f' {label}',
            color=color,
            fontsize=7,
            fontweight='bold',
            va='bottom' if 'bullish' in status else 'top',
            ha='left',
            bbox=dict(facecolor='black', alpha=0.20, edgecolor='none', pad=1),
        )

    def _draw_liquidity_sweep_marker(self, ax, chart_df: pd.DataFrame, trap_context: Dict):
        """Draw a single sweep marker near the latest candle when confirmed."""
        if not isinstance(trap_context, dict) or not trap_context.get('confirmed'):
            return

        status = str(trap_context.get('status', 'none')).lower()
        ref_level = trap_context.get('reference_level')
        if not isinstance(ref_level, (int, float)):
            return

        n = len(chart_df)
        x = max(0, n - 1)
        y = float(ref_level)
        if 'buy_side' in status:
            color = '#E74C3C'
            marker = 'v'
            label = 'BUY-SIDE SWEEP'
            va = 'top'
        else:
            color = '#27AE60'
            marker = '^'
            label = 'SELL-SIDE SWEEP'
            va = 'bottom'

        ax.scatter([x], [y], marker=marker, color=color, s=55, alpha=0.9, zorder=7, edgecolors='black', linewidth=0.5)
        ax.text(
            x - 1.0,
            y,
            f' {label}',
            color=color,
            fontsize=7,
            fontweight='bold',
            va=va,
            ha='right',
            bbox=dict(facecolor='black', alpha=0.24, edgecolor='none', pad=1),
        )

    def _draw_scenario_revision_note(self, ax, scenario_engine: Dict):
        """Show scenario revision only when the engine explicitly flagged one."""
        if not isinstance(scenario_engine, dict):
            return
        reason = str(scenario_engine.get('scenario_revision_reason', '') or '').strip()
        if not reason or reason == 'No forced scenario revision.':
            return

        active_setup = scenario_engine.get('active_setup', {}) or {}
        side = str(active_setup.get('side', 'N/A')).upper()
        note = f"SCENARIO REVIEW ({side}): {reason}"
        if len(note) > 120:
            note = note[:117] + '...'
        ax.text(
            0.99,
            0.96,
            note,
            transform=ax.transAxes,
            fontsize=7,
            color='#F7DC6F',
            ha='right',
            va='top',
            bbox=dict(facecolor='black', alpha=0.28, edgecolor='none', pad=2),
        )

    def _derive_execution_plan(self, chart_df: pd.DataFrame, analysis: Dict, config: Dict) -> Dict:
        """Infer practical execution levels from local structure for the lower panel."""
        scenario_engine = analysis.get('scenario_engine', {}) or {}
        active_setup = scenario_engine.get('active_setup', {}) or {}
        if active_setup:
            entry_low = active_setup.get('entry_zone_low')
            entry_high = active_setup.get('entry_zone_high')
            entry_zone = None
            if isinstance(entry_low, (int, float)) and isinstance(entry_high, (int, float)):
                entry_zone = (float(entry_low), float(entry_high))
            return {
                'bias': 'bullish' if str(active_setup.get('side', 'LONG')).upper() == 'LONG' else 'bearish',
                'timeframe': str(active_setup.get('timeframe', config.get('swing_tf') or config.get('fib_tf') or '')).upper(),
                'entry_zone': entry_zone,
                'invalidation': active_setup.get('invalidation'),
                'target': active_setup.get('tp1'),
                'target_2': active_setup.get('tp2'),
                'trigger': active_setup.get('trigger'),
                'split_entries': active_setup.get('split_entries', []),
                'breakeven_rule': active_setup.get('breakeven_rule'),
                'trap_context': active_setup.get('trap_context', {}),
                'sr_flip': active_setup.get('sr_flip', {}),
            }

        current_price = float(chart_df['close'].iloc[-1])
        tf = str(config.get('swing_tf') or config.get('fib_tf') or '').lower()

        structure = (analysis.get('structure', {}) or {})
        market_structure = (analysis.get('market_structure', {}) or {}).get(tf, {}) or {}
        fib = (analysis.get('fibonacci', {}) or {}).get(tf, {}) or {}
        swing = (analysis.get('swing_levels', {}) or {}).get(tf, {}) or {}
        zones = analysis.get('confluence_zones', []) or []

        supports: List[float] = []
        resistances: List[float] = []
        entry_zone = None

        diag_support = ((structure.get(f'support_{tf}') or {}).get('support_price'))
        if isinstance(diag_support, (int, float)):
            supports.append(float(diag_support))
        diag_resistance = ((structure.get(f'resistance_{tf}') or {}).get('resistance_price'))
        if isinstance(diag_resistance, (int, float)):
            resistances.append(float(diag_resistance))

        for key in ('fib_500', 'fib_618', 'fib_705', 'fib_786'):
            val = fib.get(key)
            if isinstance(val, (int, float)):
                if val <= current_price:
                    supports.append(float(val))
                if val >= current_price:
                    resistances.append(float(val))

        for val in swing.get('swing_lows', []) or []:
            if isinstance(val, (int, float)):
                supports.append(float(val))
        for val in swing.get('swing_highs', []) or []:
            if isinstance(val, (int, float)):
                resistances.append(float(val))

        last_low = market_structure.get('last_swing_low')
        if isinstance(last_low, (int, float)):
            supports.append(float(last_low))
        last_high = market_structure.get('last_swing_high')
        if isinstance(last_high, (int, float)):
            resistances.append(float(last_high))

        bias = 'neutral'
        msb_type = str((market_structure.get('msb') or {}).get('type', '')).lower()
        choch_type = str((market_structure.get('choch') or {}).get('type', '')).lower()
        structure_state = str(market_structure.get('structure', '')).lower()
        if 'bullish' in msb_type or 'bullish' in choch_type or structure_state == 'uptrend':
            bias = 'bullish'
        elif 'bearish' in msb_type or 'bearish' in choch_type or structure_state == 'downtrend':
            bias = 'bearish'

        relevant_zones = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            low = zone.get('price_low')
            high = zone.get('price_high')
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                continue
            zone_tfs = {str(x).lower() for x in zone.get('timeframes', [])}
            if tf and tf not in zone_tfs:
                continue
            relevant_zones.append(zone)

        if bias != 'bearish':
            support_candidates = sorted({round(v, 2) for v in supports if v < current_price}, reverse=True)
            nearest_support = support_candidates[0] if support_candidates else None
            invalidation = support_candidates[1] if len(support_candidates) > 1 else nearest_support
            target_candidates = sorted({round(v, 2) for v in resistances if v > current_price})
            target = target_candidates[0] if target_candidates else None

            for zone in relevant_zones:
                mid = (float(zone['price_low']) + float(zone['price_high'])) / 2.0
                if mid <= current_price * 1.01:
                    entry_zone = (float(zone['price_low']), float(zone['price_high']))
                    break
            if entry_zone is None and nearest_support is not None:
                band = max(current_price * 0.0025, nearest_support * 0.002)
                entry_zone = (nearest_support - band, nearest_support + band)

            if invalidation is None and nearest_support is not None:
                invalidation = nearest_support * 0.995
        else:
            resistance_candidates = sorted({round(v, 2) for v in resistances if v > current_price})
            nearest_resistance = resistance_candidates[0] if resistance_candidates else None
            invalidation = resistance_candidates[1] if len(resistance_candidates) > 1 else nearest_resistance
            target_candidates = sorted({round(v, 2) for v in supports if v < current_price}, reverse=True)
            target = target_candidates[0] if target_candidates else None

            for zone in relevant_zones:
                mid = (float(zone['price_low']) + float(zone['price_high'])) / 2.0
                if mid >= current_price * 0.99:
                    entry_zone = (float(zone['price_low']), float(zone['price_high']))
                    break
            if entry_zone is None and nearest_resistance is not None:
                band = max(current_price * 0.0025, nearest_resistance * 0.002)
                entry_zone = (nearest_resistance - band, nearest_resistance + band)

            if invalidation is None and nearest_resistance is not None:
                invalidation = nearest_resistance * 1.005

        return {
            'bias': bias,
            'timeframe': tf.upper(),
            'entry_zone': entry_zone,
            'invalidation': invalidation,
            'target': target,
        }

    def _draw_execution_plan(self, ax, chart_df: pd.DataFrame, execution_plan: Dict):
        """Draw trader-style execution zone, invalidation, and next target."""
        if not execution_plan:
            return

        bias = execution_plan.get('bias', 'neutral')
        tf = execution_plan.get('timeframe', '')
        entry_zone = execution_plan.get('entry_zone')
        invalidation = execution_plan.get('invalidation')
        target = execution_plan.get('target')
        target_2 = execution_plan.get('target_2')
        trigger = execution_plan.get('trigger')
        split_entries = execution_plan.get('split_entries', []) or []
        trap_context = execution_plan.get('trap_context', {}) or {}
        sr_flip = execution_plan.get('sr_flip', {}) or {}
        n = len(chart_df)

        if isinstance(entry_zone, tuple) and len(entry_zone) == 2:
            low, high = float(entry_zone[0]), float(entry_zone[1])
            color = '#2ECC71' if bias != 'bearish' else '#E67E22'
            label = 'LONG ENTRY ZONE' if bias != 'bearish' else 'SHORT ENTRY ZONE'
            ax.axhspan(low, high, color=color, alpha=0.12, zorder=1)
            ax.text(n - 1, high, f' {tf} {label}'.strip(), color=color,
                    fontsize=8, fontweight='bold', va='bottom', ha='right',
                    bbox=dict(facecolor='black', alpha=0.28, edgecolor='none', pad=1))

        if isinstance(invalidation, (int, float)):
            ax.axhline(float(invalidation), color='#F23645', linestyle='-.', linewidth=1.3, alpha=0.85, zorder=3)
            ax.text(n - 1, float(invalidation), f' {tf} INVALIDATION'.strip(), color='#F23645',
                    fontsize=8, fontweight='bold', va='center', ha='right',
                    bbox=dict(facecolor='black', alpha=0.30, edgecolor='none', pad=1))

        if isinstance(target, (int, float)):
            ax.axhline(float(target), color='#95A5A6', linestyle='--', linewidth=1.1, alpha=0.85, zorder=3)
            ax.text(n - 1, float(target), f' {tf} TP1'.strip(), color='#95A5A6',
                    fontsize=8, fontweight='bold', va='center', ha='right',
                    bbox=dict(facecolor='black', alpha=0.28, edgecolor='none', pad=1))

        if isinstance(target_2, (int, float)):
            ax.axhline(float(target_2), color='#BDC3C7', linestyle='--', linewidth=1.0, alpha=0.75, zorder=3)
            ax.text(n - 1, float(target_2), f' {tf} TP2'.strip(), color='#BDC3C7',
                    fontsize=8, fontweight='bold', va='center', ha='right',
                    bbox=dict(facecolor='black', alpha=0.24, edgecolor='none', pad=1))

        for idx, price in enumerate(split_entries[:3], start=1):
            if isinstance(price, (int, float)):
                ax.axhline(float(price), color='#3498DB', linestyle=':', linewidth=0.9, alpha=0.55, zorder=2)
                ax.text(n - 1, float(price), f' {tf} SCALE {idx}', color='#3498DB',
                        fontsize=7, va='center', ha='right',
                        bbox=dict(facecolor='black', alpha=0.18, edgecolor='none', pad=1))

        notes = []
        if trigger:
            notes.append(str(trigger).upper())
        if isinstance(trap_context, dict) and trap_context.get('confirmed'):
            notes.append(str(trap_context.get('status', 'trap')).upper())
        if isinstance(sr_flip, dict) and sr_flip.get('confirmed'):
            notes.append(str(sr_flip.get('status', 'sr_flip')).upper())
        if notes:
            ax.text(0.01, 0.02, " | ".join(notes), transform=ax.transAxes,
                    fontsize=8, color='#F1C40F', ha='left', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.20, edgecolor='none', pad=2))

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
            min_gap_pct = 0.0

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

            # -------------- Merge and Draw Bullish Gaps --------------
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

            # -------------- Merge and Draw Bearish Gaps --------------
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
            tf_str = config.get('title_suffix', '4H').split(' ')[0]
            panel_label = config.get('panel_label', '')
            
            # 1. Main Header: SYMBOL TF EXCHANGE (Data up to: TIMESTAMP)
            first_ts_str = chart_df.index[0].strftime('%Y-%m-%d %H:%M')
            last_ts_str = chart_df.index[-1].strftime('%Y-%m-%d %H:%M')
            header_text = f"{symbol} | {tf_str} | {panel_label} | Window {first_ts_str} -> {last_ts_str}"
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
            if config.get('draw_ema200', True) and 'ema200' in chart_df.columns:
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
            overlay_guide = (
                "LEGEND: Green=Support  Red=Resistance  Orange=Fib  Yellow=Confluence  "
                "Blue=Bullish Structure"
            )
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

    # -------------------- Utility Methods --------------------

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
