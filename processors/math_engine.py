import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, Tuple, Optional, List
import pandas_ta as ta
from loguru import logger
from config.settings import TradingMode


class MathEngine:
    """Pure data provider. Calculates technical indicators across multiple timeframes.
    No trading signals, no scoring, no recommendations.
    We just deliver market facts. AI interprets everything.

    Supports two modes:
    - SWING: Weekly/Daily/4H focus with Fibonacci, OI divergence, funding extremes
    - SCALP: 5m/15m/1H focus with VWAP, order flow proxies, Keltner channels
    """

    def __init__(self):
        self.order = 5

    # ─────────────── Resampling ───────────────

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if timeframe == '1m':
            return df.copy()

        tf_map = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
        rule = tf_map.get(timeframe)
        if not rule:
            return df.copy()

        tmp = df.copy()
        tmp = tmp.set_index('timestamp')
        resampled = tmp.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        return resampled

    # ─────────────── Structure Analysis ───────────────

    def find_pivot_points(self, df: pd.DataFrame, order: int = None) -> Tuple[np.ndarray, np.ndarray]:
        o = order or self.order
        close_prices = df['close'].values.astype(float)
        local_min_idx = argrelextrema(close_prices, np.less, order=o)[0]
        local_max_idx = argrelextrema(close_prices, np.greater, order=o)[0]
        return local_min_idx, local_max_idx

    def calculate_diagonal_support(self, df: pd.DataFrame) -> Optional[Dict]:
        try:
            local_min_idx, _ = self.find_pivot_points(df)
            if len(local_min_idx) < 2:
                return None

            recent_lows = local_min_idx[-3:] if len(local_min_idx) >= 3 else local_min_idx[-2:]
            x1, x2 = recent_lows[0], recent_lows[-1]
            y1, y2 = float(df.iloc[x1]['close']), float(df.iloc[x2]['close'])

            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            current_idx = len(df) - 1
            projected_support = slope * current_idx + intercept
            current_price = float(df.iloc[-1]['close'])
            distance_pct = ((current_price - projected_support) / projected_support) * 100 if projected_support else 0

            return {
                'support_price': round(projected_support, 2),
                'slope': round(slope, 6),
                'distance_pct': round(distance_pct, 2),
                'pivot_count': len(local_min_idx),
            }
        except Exception as e:
            logger.error(f"Diagonal support error: {e}")
            return None

    def calculate_diagonal_resistance(self, df: pd.DataFrame) -> Optional[Dict]:
        try:
            _, local_max_idx = self.find_pivot_points(df)
            if len(local_max_idx) < 2:
                return None

            recent_highs = local_max_idx[-3:] if len(local_max_idx) >= 3 else local_max_idx[-2:]
            x1, x2 = recent_highs[0], recent_highs[-1]
            y1, y2 = float(df.iloc[x1]['close']), float(df.iloc[x2]['close'])

            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            current_idx = len(df) - 1
            projected_resistance = slope * current_idx + intercept
            current_price = float(df.iloc[-1]['close'])
            distance_pct = ((projected_resistance - current_price) / current_price) * 100 if current_price else 0

            return {
                'resistance_price': round(projected_resistance, 2),
                'slope': round(slope, 6),
                'distance_pct': round(distance_pct, 2),
                'pivot_count': len(local_max_idx),
            }
        except Exception as e:
            logger.error(f"Diagonal resistance error: {e}")
            return None

    def detect_divergences(self, df: pd.DataFrame) -> Dict:
        close = df['close'].astype(float)
        rsi = ta.rsi(close, length=14)
        if rsi is None or rsi.empty:
            return {'bullish_divergence': False, 'bearish_divergence': False}

        local_min_idx, local_max_idx = self.find_pivot_points(df)
        bullish = False
        bearish = False

        if len(local_min_idx) >= 2:
            r = local_min_idx[-2:]
            if float(df.iloc[r[-1]]['close']) < float(df.iloc[r[-2]]['close']) and float(rsi.iloc[r[-1]]) > float(rsi.iloc[r[-2]]):
                bullish = True

        if len(local_max_idx) >= 2:
            r = local_max_idx[-2:]
            if float(df.iloc[r[-1]]['close']) > float(df.iloc[r[-2]]['close']) and float(rsi.iloc[r[-1]]) < float(rsi.iloc[r[-2]]):
                bearish = True

        return {'bullish_divergence': bullish, 'bearish_divergence': bearish}

    # ─────────────── Fibonacci Levels ───────────────

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Optional[Dict]:
        """Fibonacci retracement levels from recent swing high/low.
        Pro traders use 38.2%, 50%, 61.8% as key entry zones."""
        try:
            local_min_idx, local_max_idx = self.find_pivot_points(df, order=max(3, len(df) // 20))
            if len(local_min_idx) == 0 or len(local_max_idx) == 0:
                return None

            swing_low = float(df.iloc[local_min_idx[-1]]['close'])
            swing_high = float(df.iloc[local_max_idx[-1]]['close'])

            # Determine trend direction
            is_uptrend = local_min_idx[-1] < local_max_idx[-1]
            diff = swing_high - swing_low

            if is_uptrend:
                # Retracement from high
                levels = {
                    'swing_high': round(swing_high, 2),
                    'swing_low': round(swing_low, 2),
                    'fib_236': round(swing_high - diff * 0.236, 2),
                    'fib_382': round(swing_high - diff * 0.382, 2),
                    'fib_500': round(swing_high - diff * 0.500, 2),
                    'fib_618': round(swing_high - diff * 0.618, 2),
                    'fib_786': round(swing_high - diff * 0.786, 2),
                    'trend': 'up',
                }
            else:
                # Retracement from low
                levels = {
                    'swing_high': round(swing_high, 2),
                    'swing_low': round(swing_low, 2),
                    'fib_236': round(swing_low + diff * 0.236, 2),
                    'fib_382': round(swing_low + diff * 0.382, 2),
                    'fib_500': round(swing_low + diff * 0.500, 2),
                    'fib_618': round(swing_low + diff * 0.618, 2),
                    'fib_786': round(swing_low + diff * 0.786, 2),
                    'trend': 'down',
                }

            current_price = float(df.iloc[-1]['close'])
            levels['current_price'] = round(current_price, 2)
            levels['nearest_fib'] = self._nearest_fib(current_price, levels)
            return levels
        except Exception as e:
            logger.error(f"Fibonacci error: {e}")
            return None

    def _nearest_fib(self, price: float, levels: Dict) -> str:
        fibs = {k: abs(price - v) for k, v in levels.items() if k.startswith('fib_')}
        return min(fibs, key=fibs.get) if fibs else 'none'

    # ─────────────── Volume Profile (simplified) ───────────────

    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Optional[Dict]:
        """Simplified volume profile: find price levels with highest traded volume."""
        try:
            if len(df) < 20:
                return None

            prices = df['close'].astype(float).values
            volumes = df['volume'].astype(float).values

            price_range = np.linspace(prices.min(), prices.max(), bins + 1)
            vol_at_price = []
            for i in range(bins):
                mask = (prices >= price_range[i]) & (prices < price_range[i + 1])
                vol_at_price.append(float(volumes[mask].sum()))

            max_idx = np.argmax(vol_at_price)
            poc_price = (price_range[max_idx] + price_range[max_idx + 1]) / 2  # Point of Control

            # High volume nodes (top 30%)
            threshold = np.percentile(vol_at_price, 70)
            hvn = [(round((price_range[i] + price_range[i + 1]) / 2, 2), round(vol_at_price[i], 2))
                   for i in range(bins) if vol_at_price[i] >= threshold]

            return {
                'poc_price': round(poc_price, 2),
                'high_volume_nodes': hvn[:5],
                'total_volume': round(sum(vol_at_price), 2),
            }
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return None

    # ─────────────── Helper ───────────────

    def _safe_val(self, series, idx=-1) -> Optional[float]:
        if series is None or (hasattr(series, 'empty') and series.empty):
            return None
        try:
            v = series.iloc[idx] if hasattr(series, 'iloc') else series
            if pd.isna(v):
                return None
            return round(float(v), 4)
        except Exception:
            return None

    # ─────────────── Core Indicators (shared) ───────────────

    def calculate_indicators_for_timeframe(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators. Pure numbers, zero interpretation."""
        if len(df) < 20:
            return {'error': 'insufficient_data', 'candle_count': len(df)}

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        r = {}

        try:
            r['price'] = round(float(close.iloc[-1]), 2)
            if len(close) > 1:
                r['price_change_pct'] = round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 4)

            # EMAs & SMAs
            r['ema_9'] = self._safe_val(ta.ema(close, length=9))
            r['ema_21'] = self._safe_val(ta.ema(close, length=21))
            r['ema_50'] = self._safe_val(ta.ema(close, length=50))
            r['ema_200'] = self._safe_val(ta.ema(close, length=200))
            r['sma_50'] = self._safe_val(ta.sma(close, length=50))
            r['sma_200'] = self._safe_val(ta.sma(close, length=200))

            # Golden/Death Cross detection data
            if r.get('sma_50') and r.get('sma_200'):
                r['sma50_above_sma200'] = r['sma_50'] > r['sma_200']

            # MACD
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                cols = macd_df.columns.tolist()
                r['macd_line'] = self._safe_val(macd_df[cols[0]])
                r['macd_signal'] = self._safe_val(macd_df[cols[1]])
                r['macd_histogram'] = self._safe_val(macd_df[cols[2]])
                # Histogram direction change
                if len(macd_df) > 1:
                    r['macd_hist_prev'] = self._safe_val(macd_df[cols[2]], -2)

            # ADX
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                cols = adx_df.columns.tolist()
                r['adx'] = self._safe_val(adx_df[cols[0]])
                r['di_plus'] = self._safe_val(adx_df[cols[1]])
                r['di_minus'] = self._safe_val(adx_df[cols[2]])

            # Supertrend
            st_df = ta.supertrend(high, low, close, length=10, multiplier=3.0)
            if st_df is not None and not st_df.empty:
                cols = st_df.columns.tolist()
                r['supertrend_value'] = self._safe_val(st_df[cols[0]])
                st_dir = st_df[cols[1]].iloc[-1]
                r['supertrend_direction'] = int(st_dir) if not pd.isna(st_dir) else None

            # Ichimoku
            try:
                ichi_result = ta.ichimoku(high, low, close)
                if ichi_result and isinstance(ichi_result, tuple) and len(ichi_result) >= 1:
                    ichi = ichi_result[0]
                    if ichi is not None and not ichi.empty:
                        cols = ichi.columns.tolist()
                        if len(cols) >= 4:
                            r['ichimoku_tenkan'] = self._safe_val(ichi[cols[0]])
                            r['ichimoku_kijun'] = self._safe_val(ichi[cols[1]])
                            r['ichimoku_senkou_a'] = self._safe_val(ichi[cols[2]])
                            r['ichimoku_senkou_b'] = self._safe_val(ichi[cols[3]])
            except Exception:
                pass

            # RSI
            r['rsi'] = self._safe_val(ta.rsi(close, length=14))

            # Stochastic RSI
            stoch = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
            if stoch is not None and not stoch.empty:
                cols = stoch.columns.tolist()
                r['stoch_rsi_k'] = self._safe_val(stoch[cols[0]])
                r['stoch_rsi_d'] = self._safe_val(stoch[cols[1]])

            r['williams_r'] = self._safe_val(ta.willr(high, low, close, length=14))
            r['cci'] = self._safe_val(ta.cci(high, low, close, length=20))
            r['mfi'] = self._safe_val(ta.mfi(high, low, close, volume, length=14))

            # Bollinger Bands
            bb = ta.bbands(close, length=20, std=2.0)
            if bb is not None and not bb.empty:
                cols = bb.columns.tolist()
                r['bb_lower'] = self._safe_val(bb[cols[0]])
                r['bb_mid'] = self._safe_val(bb[cols[1]])
                r['bb_upper'] = self._safe_val(bb[cols[2]])
                r['bb_bandwidth'] = self._safe_val(bb[cols[3]]) if len(cols) > 3 else None
                r['bb_percent_b'] = self._safe_val(bb[cols[4]]) if len(cols) > 4 else None

            # ATR
            r['atr'] = self._safe_val(ta.atr(high, low, close, length=14))

            # Volume indicators
            r['obv'] = self._safe_val(ta.obv(close, volume))
            r['vwap'] = self._safe_val(ta.vwap(high, low, close, volume))
            r['cmf'] = self._safe_val(ta.cmf(high, low, close, volume, length=20))

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            r['calculation_error'] = str(e)

        return r

    # ─────────────── SCALP-specific indicators ───────────────

    def calculate_scalp_indicators(self, df: pd.DataFrame) -> Dict:
        """Extra indicators for scalp mode: Keltner Channels, VWAP bands, volume delta proxy."""
        if len(df) < 20:
            return {}

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        r = {}

        try:
            # Keltner Channels
            kc = ta.kc(high, low, close, length=20, scalar=1.5)
            if kc is not None and not kc.empty:
                cols = kc.columns.tolist()
                r['kc_lower'] = self._safe_val(kc[cols[0]])
                r['kc_mid'] = self._safe_val(kc[cols[1]])
                r['kc_upper'] = self._safe_val(kc[cols[2]])

            # Volume delta proxy (close vs midpoint of candle)
            # Positive = buying pressure, Negative = selling pressure
            midpoints = (high + low) / 2
            delta_proxy = ((close - midpoints) / (high - low + 1e-10)) * volume
            r['volume_delta_proxy'] = round(float(delta_proxy.iloc[-1]), 2)
            r['volume_delta_5bar'] = round(float(delta_proxy.tail(5).sum()), 2)

            # ALMA (Arnaud Legoux Moving Average) - smoother than EMA for scalping
            alma = ta.alma(close, length=9)
            if alma is not None and not alma.empty:
                r['alma_9'] = self._safe_val(alma)

            # Stochastic (fast, for scalping)
            stoch = ta.stoch(high, low, close, k=5, d=3, smooth_k=3)
            if stoch is not None and not stoch.empty:
                cols = stoch.columns.tolist()
                r['stoch_k'] = self._safe_val(stoch[cols[0]])
                r['stoch_d'] = self._safe_val(stoch[cols[1]])

        except Exception as e:
            logger.error(f"Scalp indicator error: {e}")

        return r

    # ─────────────── Funding Rate Context ───────────────

    def analyze_funding_context(self, funding_data: Dict) -> Dict:
        """Deliver funding rate context as raw facts for AI.
        Pro traders treat extreme funding as contrarian signals."""
        if not funding_data:
            return {}

        result = {}
        try:
            fr = funding_data.get('funding_rate')
            if fr is not None:
                fr = float(fr)
                result['funding_rate'] = fr
                result['funding_annualized_pct'] = round(fr * 3 * 365 * 100, 2)  # 8h intervals
                # Raw facts about extremes - AI decides interpretation
                result['funding_is_extreme_positive'] = fr > 0.0005  # >0.05%
                result['funding_is_extreme_negative'] = fr < -0.0003  # <-0.03%
                result['funding_is_neutral'] = -0.0001 <= fr <= 0.0001

            oi = funding_data.get('open_interest_value')
            if oi is not None:
                result['oi_value_usdt'] = float(oi)

            ls = funding_data.get('long_short_ratio')
            if ls is not None:
                result['long_short_ratio'] = float(ls)
                result['long_account_pct'] = funding_data.get('long_account')
                result['short_account_pct'] = funding_data.get('short_account')

        except Exception as e:
            logger.error(f"Funding context error: {e}")

        return result

    # ─────────────── Main Entry Points ───────────────

    def analyze_market_swing(self, df_1m: pd.DataFrame) -> Dict:
        """SWING mode: 1h, 4h, 1d focus with Fibonacci + volume profile.
        Returns ONLY factual data. AI decides everything."""

        result = {
            'mode': 'swing',
            'current_price': round(float(df_1m.iloc[-1]['close']), 2),
            'current_volume': round(float(df_1m.iloc[-1]['volume']), 2),
            'timeframes': {},
            'structure': {},
            'fibonacci': {},
            'volume_profile': {},
        }

        # Swing focuses on higher timeframes
        timeframes = {
            '1h': self.resample_to_timeframe(df_1m, '1h'),
            '4h': self.resample_to_timeframe(df_1m, '4h'),
            '1d': self.resample_to_timeframe(df_1m, '1d'),
        }

        for tf_name, tf_df in timeframes.items():
            if len(tf_df) >= 5:
                result['timeframes'][tf_name] = self.calculate_indicators_for_timeframe(tf_df)
                result['timeframes'][tf_name]['candle_count'] = len(tf_df)
            else:
                result['timeframes'][tf_name] = {'error': 'insufficient_data', 'candle_count': len(tf_df)}

        # Structural analysis on multiple timeframes
        for tf_name in ['1h', '4h']:
            tf_df = timeframes.get(tf_name, pd.DataFrame())
            if len(tf_df) >= 20:
                result['structure'][f'support_{tf_name}'] = self.calculate_diagonal_support(tf_df)
                result['structure'][f'resistance_{tf_name}'] = self.calculate_diagonal_resistance(tf_df)
                result['structure'][f'divergence_{tf_name}'] = self.detect_divergences(tf_df)

        # Fibonacci on 4h (most meaningful for swing)
        tf_4h = timeframes.get('4h', pd.DataFrame())
        if len(tf_4h) >= 20:
            result['fibonacci']['4h'] = self.calculate_fibonacci_levels(tf_4h)

        # Fibonacci on 1d
        tf_1d = timeframes.get('1d', pd.DataFrame())
        if len(tf_1d) >= 10:
            result['fibonacci']['1d'] = self.calculate_fibonacci_levels(tf_1d)

        # Volume profile on 4h
        if len(tf_4h) >= 20:
            result['volume_profile']['4h'] = self.calculate_volume_profile(tf_4h)

        # Last few candle patterns on 4h
        if len(tf_4h) >= 3:
            result['recent_4h_candles'] = self._recent_candle_data(tf_4h, count=5)

        return result

    def analyze_market_scalp(self, df_1m: pd.DataFrame) -> Dict:
        """SCALP mode: 5m, 15m, 1h focus with VWAP/Keltner/volume delta.
        Returns ONLY factual data. AI decides everything."""

        result = {
            'mode': 'scalp',
            'current_price': round(float(df_1m.iloc[-1]['close']), 2),
            'current_volume': round(float(df_1m.iloc[-1]['volume']), 2),
            'timeframes': {},
            'scalp_indicators': {},
            'structure': {},
        }

        timeframes = {
            '1m': df_1m,
            '5m': self.resample_to_timeframe(df_1m, '5m'),
            '15m': self.resample_to_timeframe(df_1m, '15m'),
            '1h': self.resample_to_timeframe(df_1m, '1h'),
        }

        for tf_name, tf_df in timeframes.items():
            if len(tf_df) >= 5:
                result['timeframes'][tf_name] = self.calculate_indicators_for_timeframe(tf_df)
                result['timeframes'][tf_name]['candle_count'] = len(tf_df)
            else:
                result['timeframes'][tf_name] = {'error': 'insufficient_data', 'candle_count': len(tf_df)}

        # Scalp-specific on 5m and 15m
        for tf_name in ['5m', '15m']:
            tf_df = timeframes.get(tf_name, pd.DataFrame())
            if len(tf_df) >= 20:
                result['scalp_indicators'][tf_name] = self.calculate_scalp_indicators(tf_df)

        # Quick structure on 15m
        tf_15m = timeframes.get('15m', pd.DataFrame())
        if len(tf_15m) >= 20:
            result['structure']['support_15m'] = self.calculate_diagonal_support(tf_15m)
            result['structure']['resistance_15m'] = self.calculate_diagonal_resistance(tf_15m)

        # Recent 5m candles for pattern reading
        tf_5m = timeframes.get('5m', pd.DataFrame())
        if len(tf_5m) >= 3:
            result['recent_5m_candles'] = self._recent_candle_data(tf_5m, count=10)

        return result

    def analyze_market(self, df_1m: pd.DataFrame, mode: TradingMode) -> Dict:
        """Unified entry point. Dispatches to mode-specific analysis."""
        if mode == TradingMode.SCALP:
            return self.analyze_market_scalp(df_1m)
        return self.analyze_market_swing(df_1m)

    # ─────────────── Compact Data Formatting ───────────────

    def format_compact(self, analysis: Dict) -> str:
        """Format analysis as compact text to save tokens.
        ~40-60% fewer tokens than json.dumps(indent=2)."""
        lines = []
        lines.append(f"MODE={analysis.get('mode','?')} PRICE={analysis.get('current_price',0)} VOL={analysis.get('current_volume',0)}")

        for tf, data in analysis.get('timeframes', {}).items():
            if isinstance(data, dict) and 'error' not in data:
                parts = [f"[{tf}]"]
                for k, v in data.items():
                    if v is not None and k != 'candle_count':
                        parts.append(f"{k}={v}")
                lines.append(' '.join(parts))

        struct = analysis.get('structure', {})
        if struct:
            lines.append("[STRUCTURE]")
            for k, v in struct.items():
                if v:
                    lines.append(f"  {k}: {v}")

        fib = analysis.get('fibonacci', {})
        if fib:
            lines.append("[FIBONACCI]")
            for k, v in fib.items():
                if v:
                    lines.append(f"  {k}: {v}")

        vp = analysis.get('volume_profile', {})
        if vp:
            lines.append("[VOLUME_PROFILE]")
            for k, v in vp.items():
                if v:
                    lines.append(f"  {k}: {v}")

        scalp = analysis.get('scalp_indicators', {})
        if scalp:
            lines.append("[SCALP_INDICATORS]")
            for k, v in scalp.items():
                if v:
                    lines.append(f"  {k}: {v}")

        candles = analysis.get('recent_4h_candles') or analysis.get('recent_5m_candles')
        if candles:
            lines.append("[RECENT_CANDLES]")
            for c in candles:
                lines.append(f"  O={c['open']} H={c['high']} L={c['low']} C={c['close']} V={c['volume']} {'UP' if c.get('is_bullish') else 'DN'}")

        return '\n'.join(lines)

    # ─────────────── Internal Helpers ───────────────

    def _recent_candle_data(self, df: pd.DataFrame, count: int = 5) -> List[Dict]:
        candles = []
        for _, row in df.tail(count).iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            body = abs(c - o)
            candles.append({
                'open': round(o, 2),
                'high': round(h, 2),
                'low': round(l, 2),
                'close': round(c, 2),
                'volume': round(float(row['volume']), 2),
                'body_size': round(body, 2),
                'upper_wick': round(h - max(o, c), 2),
                'lower_wick': round(min(o, c) - l, 2),
                'is_bullish': c > o,
            })
        return candles


math_engine = MathEngine()
