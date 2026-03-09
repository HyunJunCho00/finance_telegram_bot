import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, Tuple, Optional, List
import pandas_ta_classic as ta
from loguru import logger
from config.settings import TradingMode

# ── 분석 파이프라인(VLM + AI 에이전트)에 전달되는 지표 화이트리스트 ────────
# 사용자 질의용 지표(PSAR, KC, Aroon, HMA 등)는 계산되지만 여기엔 포함하지 않음
# 중복 제거 원칙:
#   모멘텀 → RSI + StochRSI + MFI  (Williams%R, CCI, OBV 제외)
#   이동평균 → EMA(21/50/200)       (EMA9, SMA50, SMA200 제외)
_COMPACT_KEYS = frozenset({
    'price', 'price_change_pct',
    'ema_21', 'ema_50', 'ema_200',
    'sma50_above_sma200',
    'macd_line', 'macd_signal', 'macd_histogram', 'macd_hist_prev',
    'adx', 'di_plus', 'di_minus',
    'supertrend_value', 'supertrend_direction',
    'rsi',
    'stoch_rsi_k', 'stoch_rsi_d',
    'mfi',
    'bb_lower', 'bb_mid', 'bb_upper', 'bb_bandwidth', 'bb_percent_b',
    'atr',
    'vwap',
    'cmf',
    'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
})


class MathEngine:
    """Pure data provider. Calculates technical indicators across multiple timeframes.
    No trading signals, no scoring, no recommendations.
    We just deliver market facts. AI interprets everything.

    Supports two modes:
    - SWING: Weekly/Daily/4H focus with Fibonacci, OI divergence, funding extremes
    """

    def __init__(self):
        self.order = 5

    def merge_history(self, df_main: pd.DataFrame, df_history: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge recent Supabase data with historical GCS data seamlessly."""
        if df_history is None or df_history.empty:
            return df_main
        
        # Ensure UTC and datetime
        df_main = df_main.copy()
        if 'timestamp' not in df_main.columns and isinstance(df_main.index, pd.DatetimeIndex):
            df_main['timestamp'] = df_main.index
        df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], utc=True)
        
        df_history = df_history.copy()
        if df_history is not None and not df_history.empty:
            if 'timestamp' not in df_history.columns and isinstance(df_history.index, pd.DatetimeIndex):
                df_history['timestamp'] = df_history.index
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], utc=True)
        
        # Combine
        combined = pd.concat([df_history, df_main], ignore_index=True)
        # Sort and deduplicate (keep latest data if overlap exists)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
        return combined.reset_index(drop=True)

    # ─────────────── Resampling ───────────────

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if timeframe == '1m':
            return df.copy()

        tf_map = {
            '1min': '1min', '3min': '3min', '5min': '5min', '15min': '15min', '30min': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1D', '3d': '3D',
            '1w': '1W-MON',  # Week starts Monday (Upbit KST 09:00 = UTC 00:00)
            '1M': '1MS',     # Month start
        }
        # Also support short forms
        short_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min'
        }
        rule = tf_map.get(timeframe) or short_map.get(timeframe)
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
        })
        
        # [Fix] Retain the true timestamps of the highs and lows so MTFA trendlines don't float
        resampled['high_time'] = tmp.groupby(pd.Grouper(freq=rule))['high'].idxmax()
        resampled['low_time'] = tmp.groupby(pd.Grouper(freq=rule))['low'].idxmin()

        resampled = resampled.dropna(subset=['open']).reset_index()

        # [Fix] Replace any NaT high_time/low_time (from empty periods) with the candle's own timestamp
        # NaT occurs when a resampled period has no 1m data (gap). Fallback to period open time.
        if 'high_time' in resampled.columns:
            nat_mask = resampled['high_time'].isna()
            if nat_mask.any():
                resampled.loc[nat_mask, 'high_time'] = resampled.loc[nat_mask, 'timestamp']
        if 'low_time' in resampled.columns:
            nat_mask = resampled['low_time'].isna()
            if nat_mask.any():
                resampled.loc[nat_mask, 'low_time'] = resampled.loc[nat_mask, 'timestamp']

        return resampled

    # ─────────────── Structure Analysis ───────────────

    def find_pivot_points(self, df: pd.DataFrame, order: int = None) -> Tuple[np.ndarray, np.ndarray]:
        o = order or self.order
        close_prices = df['close'].values.astype(float)
        local_min_idx = argrelextrema(close_prices, np.less, order=o)[0]
        local_max_idx = argrelextrema(close_prices, np.greater, order=o)[0]
        return local_min_idx, local_max_idx

    def find_td_points(self, df: pd.DataFrame, n: int = 5) -> Tuple[List[int], List[int]]:
        """Identify DeMark TD Points (Level 1/2).
        A high is a TD Point High if it's higher than N candles to its left and right.
        A low is a TD Point Low if it's lower than N candles to its left and right.
        """
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        length = len(df)
        
        td_highs = []
        td_lows = []
        
        for i in range(n, length - n):
            # Check for TD High
            is_high = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            if is_high:
                td_highs.append(i)
                
            # Check for TD Low
            is_low = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            if is_low:
                td_lows.append(i)
                
        return td_lows, td_highs

    def calculate_diagonal_support(self, df: pd.DataFrame, mode: TradingMode = TradingMode.SWING) -> Optional[Dict]:
        """TD Point-based Support Trendline."""
        try:
            n = 12 if mode == TradingMode.POSITION else 5
            local_min_idx, _ = self.find_td_points(df, n=n)
            
            if len(local_min_idx) < 2:
                # Fallback to standard pivots if TD points are too sparse
                local_min_idx, _ = self.find_pivot_points(df, order=n)
                
            if len(local_min_idx) < 2:
                return None

            # Institutional Rule: Connect the two most recent significant points
            x1, x2 = local_min_idx[-2], local_min_idx[-1]
            y1, y2 = float(df.iloc[x1]['low']), float(df.iloc[x2]['low'])
            
            # Map index to actual timestamps to prevent gap distortion
            if 'low_time' in df.columns:
                ts1 = df.iloc[x1]['low_time']
                ts2 = df.iloc[x2]['low_time']
            else:
                ts1 = df.index[x1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[x1]['timestamp']
                ts2 = df.index[x2] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[x2]['timestamp']
            
            # In case timestamps are not datetime yet
            ts1 = pd.to_datetime(ts1, utc=True)
            ts2 = pd.to_datetime(ts2, utc=True)

            current_val = y2 + ((y2 - y1) / (x2 - x1)) * (len(df) - 1 - x2)

            return {
                'point1': (ts1, y1),
                'point2': (ts2, y2),
                'pivot_count': len(local_min_idx),
                'support_price': round(float(current_val), 2),
                'type': 'support',
                '_x1': int(x1),
                '_x2': int(x2),
            }
        except Exception as e:
            logger.error(f"TD Support error: {e}")
            return None

    def calculate_diagonal_resistance(self, df: pd.DataFrame, mode: TradingMode = TradingMode.SWING) -> Optional[Dict]:
        """TD Point-based Resistance Trendline."""
        try:
            n = 12 if mode == TradingMode.POSITION else 5
            _, local_max_idx = self.find_td_points(df, n=n)
            
            if len(local_max_idx) < 2:
                # Fallback
                _, local_max_idx = self.find_pivot_points(df, order=n)
                
            if len(local_max_idx) < 2:
                return None

            x1, x2 = local_max_idx[-2], local_max_idx[-1]
            y1, y2 = float(df.iloc[x1]['high']), float(df.iloc[x2]['high'])
            
            # Map index to actual timestamps to prevent gap distortion
            if 'high_time' in df.columns:
                ts1 = df.iloc[x1]['high_time']
                ts2 = df.iloc[x2]['high_time']
            else:
                ts1 = df.index[x1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[x1]['timestamp']
                ts2 = df.index[x2] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[x2]['timestamp']
            
            # In case timestamps are not datetime yet
            ts1 = pd.to_datetime(ts1, utc=True)
            ts2 = pd.to_datetime(ts2, utc=True)

            current_val = y2 + ((y2 - y1) / (x2 - x1)) * (len(df) - 1 - x2)

            return {
                'point1': (ts1, y1),
                'point2': (ts2, y2),
                'pivot_count': len(local_max_idx),
                'resistance_price': round(float(current_val), 2),
                'type': 'resistance',
                '_x1': int(x1),
                '_x2': int(x2),
            }
        except Exception as e:
            logger.error(f"TD Resistance error: {e}")
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

    def detect_macro_divergences(self, df: pd.DataFrame, cvd_df: Optional[pd.DataFrame] = None) -> Dict:
        """Detect long-term divergence between Price and CVD/OI (Smart Money Footprint)."""
        if cvd_df is None or cvd_df.empty or len(df) < 50:
            return {'macro_bull_div': False, 'macro_bear_div': False}
            
        try:
            # Prepare aligned data
            df = df.copy()
            cvd = cvd_df.copy()
            cvd['timestamp'] = pd.to_datetime(cvd['timestamp'], utc=True)
            cvd = cvd.set_index('timestamp')
            
            # Resample CVD to match DF timeframe
            # Handle both column and index for timestamp
            df_ts = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['timestamp'], utc=True)
            resample_rule = '1D' if (df_ts.to_series().diff().median() >= pd.Timedelta(days=1)) else '4h'
            cvd_resampled = cvd.resample(resample_rule).sum().fillna(0)
            cvd_acc = (cvd_resampled.get('whale_buy_vol', 0) - cvd_resampled.get('whale_sell_vol', 0)).cumsum()
            
            # Align
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
            combined = pd.concat([df['close'], cvd_acc.rename('cvd')], axis=1).ffill().dropna()
            
            if len(combined) < 30:
                return {'macro_bull_div': False, 'macro_bear_div': False}
                
            # Find macro pivots (order=10 for macro)
            close = combined['close'].values
            cvd_vals = combined['cvd'].values
            min_idx = argrelextrema(close, np.less, order=10)[0]
            max_idx = argrelextrema(close, np.greater, order=10)[0]
            
            bull_div = False
            bear_div = False
            
            if len(min_idx) >= 2:
                # Lower Low in Price, Higher Low in CVD
                if close[min_idx[-1]] < close[min_idx[-2]] and cvd_vals[min_idx[-1]] > cvd_vals[min_idx[-2]]:
                    bull_div = True
            
            if len(max_idx) >= 2:
                # Higher High in Price, Lower High in CVD
                if close[max_idx[-1]] > close[max_idx[-2]] and cvd_vals[max_idx[-1]] < cvd_vals[max_idx[-2]]:
                    bear_div = True
                    
            return {'macro_bull_div': bull_div, 'macro_bear_div': bear_div}
        except Exception as e:
            logger.error(f"Macro divergence error: {e}")
            return {'macro_bull_div': False, 'macro_bear_div': False}

    # ─────────────── Order Blocks ───────────────

    def calculate_macro_order_blocks(self, df: pd.DataFrame, count: int = 3) -> List[Dict]:
        """Identify Institutional Order Blocks (OB).
        Bullish OB: The last bearish candle before a strong bullish break of structure.
        Bearish OB: The last bullish candle before a strong bearish break of structure.
        """
        try:
            if len(df) < 20: return []
            
            obs = []
            # Simplified OB detection: Look for 'Engulfing' style breakouts from local pivots
            # Bullish Breakout: Close crosses above a recent pivot high
            _, max_pivots = self.find_pivot_points(df, order=5)
            
            for i in range(2, len(df) - 1):
                # Potential Bullish OB: Candle i-1 was Red, Candle i is Green and strong
                if df.iloc[i]['close'] > df.iloc[i]['open'] and df.iloc[i-1]['close'] < df.iloc[i-1]['open']:
                    # Strong breakout condition: current close > previous high AND move is > 1.5%
                    move_size = (df.iloc[i]['close'] - df.iloc[i]['open']) / df.iloc[i]['open']
                    if move_size > 0.015:
                        obs.append({
                            'type': 'BULLISH',
                            'top': float(df.iloc[i-1]['high']),
                            'bottom': float(df.iloc[i-1]['low']),
                            'timestamp': df.index[i-1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[i-1]['timestamp'],
                            'strength': move_size
                        })
                
                # Potential Bearish OB
                if df.iloc[i]['close'] < df.iloc[i]['open'] and df.iloc[i-1]['close'] > df.iloc[i-1]['open']:
                    move_size = (df.iloc[i]['open'] - df.iloc[i]['close']) / df.iloc[i]['open']
                    if move_size > 0.015:
                        obs.append({
                            'type': 'BEARISH',
                            'top': float(df.iloc[i-1]['high']),
                            'bottom': float(df.iloc[i-1]['low']),
                            'timestamp': df.index[i-1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[i-1]['timestamp'],
                            'strength': move_size
                        })
            
            # Filter for most recent and strongest
            obs = sorted(obs, key=lambda x: x['timestamp'], reverse=True)
            return obs[:count*2] # Return a few of each
        except Exception as e:
            logger.error(f"Order Block calculation error: {e}")
            return []

    # ─────────────── Anchored VWAP ───────────────

    def calculate_anchored_vwap(self, df: pd.DataFrame, anchor_idx: int) -> Optional[pd.Series]:
        """Calculate VWAP starting from a specific anchor index."""
        try:
            if anchor_idx < 0 or anchor_idx >= len(df):
                return None
                
            subset = df.iloc[anchor_idx:].copy()
            # typical price * volume
            tp = (subset['high'] + subset['low'] + subset['close']) / 3
            pv = tp * subset['volume']
            
            cum_pv = pv.cumsum()
            cum_vol = subset['volume'].cumsum()
            
            avwap = cum_pv / cum_vol
            
            # Reindex to full length with NaNs before anchor
            full_series = pd.Series(index=df.index, dtype=float)
            full_series.iloc[anchor_idx:] = avwap.values
            return full_series
        except Exception as e:
            logger.error(f"Anchored VWAP error: {e}")
            return None

    # ─────────────── Fibonacci Levels ───────────────

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Optional[Dict]:
        """Fibonacci retracement levels from recent swing high/low.
        Pro traders use OTE entry zones: 50%, 61.8%, 70.5%, 78.6%."""
        try:
            working_df = df.tail(min(len(df), 180)).reset_index(drop=True)
            local_min_idx, local_max_idx = self.find_pivot_points(working_df, order=max(3, len(working_df) // 18))
            if len(local_min_idx) == 0 or len(local_max_idx) == 0:
                return None

            swing_low = float(working_df.iloc[local_min_idx[-1]]['low'])
            swing_high = float(working_df.iloc[local_max_idx[-1]]['high'])

            # Determine trend direction
            is_uptrend = local_min_idx[-1] < local_max_idx[-1]
            diff = swing_high - swing_low

            if is_uptrend:
                # Retracement from high
                levels = {
                    'swing_high': round(swing_high, 2),
                    'swing_low': round(swing_low, 2),
                    'fib_500': round(swing_high - diff * 0.500, 2),
                    'fib_618': round(swing_high - diff * 0.618, 2),
                    'fib_705': round(swing_high - diff * 0.705, 2),
                    'fib_786': round(swing_high - diff * 0.786, 2),
                    'trend': 'up',
                }
            else:
                # Retracement from low
                levels = {
                    'swing_high': round(swing_high, 2),
                    'swing_low': round(swing_low, 2),
                    'fib_500': round(swing_low + diff * 0.500, 2),
                    'fib_618': round(swing_low + diff * 0.618, 2),
                    'fib_705': round(swing_low + diff * 0.705, 2),
                    'fib_786': round(swing_low + diff * 0.786, 2),
                    'trend': 'down',
                }

            current_price = float(working_df.iloc[-1]['close'])
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

        # [FIX] pandas_ta VWAP requires a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                
        # [FIX] pandas_ta has issues with timezone-aware datetime indexes in some indicators (like MFI/VWAP)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        close = df['close'].astype('float64')
        high = df['high'].astype('float64')
        low = df['low'].astype('float64')
        volume = df['volume'].astype('float64')
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

            # In processors/math_engine.py
            import warnings
            
            try:
                r['williams_r'] = self._safe_val(ta.willr(high, low, close, length=14))
                r['cci'] = self._safe_val(ta.cci(high, low, close, length=20))
                
                # MFI requires strict float64 series and sometimes breaks if values are too large
                vol_mfi = volume.astype('float64')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    r['mfi'] = self._safe_val(ta.mfi(high, low, close, vol_mfi, length=14))
            except Exception as e:
                logger.warning(f"Error calculating willr/cci/mfi: {e}")

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
            try:
                r['obv'] = self._safe_val(ta.obv(close, volume))
                r['vwap'] = self._safe_val(ta.vwap(high, low, close, volume))
                r['cmf'] = self._safe_val(ta.cmf(high, low, close, volume, length=20))
            except Exception as e:
                logger.warning(f"Error calculating volume indicators (obv/vwap/cmf): {e}")

            # HMA (Hull Moving Average) — 빠른 MA, EMA보다 노이즈 적음
            r['hma_9'] = self._safe_val(ta.hma(close, length=9))
            r['hma_21'] = self._safe_val(ta.hma(close, length=21))

            # Parabolic SAR — 트렌드 전환 신호
            try:
                psar_df = ta.psar(high, low, close)
                if psar_df is not None and not psar_df.empty:
                    cols = psar_df.columns.tolist()
                    # cols: PSARl (long side), PSARs (short side), PSARaf, PSARr
                    psar_long = psar_df[cols[0]].iloc[-1]
                    psar_short = psar_df[cols[1]].iloc[-1]
                    # 활성 PSAR 값 (한쪽만 값이 있음)
                    if not pd.isna(psar_long):
                        r['psar'] = round(float(psar_long), 2)
                        r['psar_trend'] = 'BULLISH'
                    elif not pd.isna(psar_short):
                        r['psar'] = round(float(psar_short), 2)
                        r['psar_trend'] = 'BEARISH'
            except Exception:
                pass

            # Keltner Channel — 변동성 채널 (BB 스퀴즈 감지에 활용)
            try:
                kc_df = ta.kc(high, low, close, length=20)
                if kc_df is not None and not kc_df.empty:
                    cols = kc_df.columns.tolist()
                    r['kc_lower'] = self._safe_val(kc_df[cols[0]])
                    r['kc_mid'] = self._safe_val(kc_df[cols[1]])
                    r['kc_upper'] = self._safe_val(kc_df[cols[2]])
                    # BB inside KC = Squeeze (변동성 수축, 폭발 임박)
                    if r.get('bb_lower') and r.get('bb_upper') and r.get('kc_lower') and r.get('kc_upper'):
                        r['bb_kc_squeeze'] = r['bb_lower'] > r['kc_lower'] and r['bb_upper'] < r['kc_upper']
            except Exception:
                pass

            # Aroon — 트렌드 전환 조기 감지 (ADX 보완)
            try:
                aroon_df = ta.aroon(high, low, length=25)
                if aroon_df is not None and not aroon_df.empty:
                    cols = aroon_df.columns.tolist()
                    # cols: AROOND, AROONU, AROONOSC
                    r['aroon_down'] = self._safe_val(aroon_df[cols[0]])
                    r['aroon_up'] = self._safe_val(aroon_df[cols[1]])
                    r['aroon_osc'] = self._safe_val(aroon_df[cols[2]])
            except Exception:
                pass

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

    # ─────────────── FVG (Fair Value Gap) ───────────────

    def calculate_fvg(self, df: pd.DataFrame, max_gaps: int = 5, mode: TradingMode = TradingMode.SWING) -> List[Dict]:
        """Detect Fair Value Gaps (3-candle imbalance zones).
        FVG = gap between candle 1's wick and candle 3's wick.
        Unfilled FVGs act as magnets for price return."""
        if len(df) < 3:
            return []

        fvgs = []
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        current_price = float(df.iloc[-1]['close'])
        
        # In POSITION mode, filter out small FVGs (<1%)
        min_gap_pct = 1.0 if mode == TradingMode.POSITION else 0.0

        for i in range(len(df) - 2):
            # Bullish FVG: candle3.low > candle1.high (gap up)
            if lows[i + 2] > highs[i]:
                gap_low = highs[i]
                gap_high = lows[i + 2]
                gap_pct = (gap_high - gap_low) / gap_low * 100
                if gap_pct >= min_gap_pct:
                    # Check if unfilled (current price hasn't fully entered the gap)
                    filled = current_price <= gap_high and current_price >= gap_low
                    fvgs.append({
                        'type': 'bullish',
                        'gap_low': round(gap_low, 2),
                        'gap_high': round(gap_high, 2),
                        'gap_size_pct': round(gap_pct, 4),
                        'filled': filled,
                        'candle_idx': i + 1,
                    })

            # Bearish FVG: candle3.high < candle1.low (gap down)
            if highs[i + 2] < lows[i]:
                gap_high = lows[i]
                gap_low = highs[i + 2]
                gap_pct = (gap_high - gap_low) / gap_low * 100
                if gap_pct >= min_gap_pct:
                    filled = current_price >= gap_low and current_price <= gap_high
                    fvgs.append({
                        'type': 'bearish',
                        'gap_low': round(gap_low, 2),
                        'gap_high': round(gap_high, 2),
                        'gap_size_pct': round(gap_pct, 4),
                        'filled': filled,
                        'candle_idx': i + 1,
                    })

        # Return most recent unfilled gaps first
        unfilled = [g for g in fvgs if not g['filled']]
        return unfilled[-max_gaps:] if unfilled else fvgs[-max_gaps:]

    # ─────────────── Swing High/Low (Liquidity Levels) ───────────────

    def calculate_swing_levels(self, df: pd.DataFrame, lookback: int = None) -> Dict:
        """Find major swing highs/lows that act as liquidity pools.
        Stop losses cluster around these levels."""
        if len(df) < 10:
            return {}

        working_df = df.tail(min(len(df), 180)).reset_index(drop=True)
        order = lookback or max(4, min(14, len(working_df) // 12))
        highs = working_df['high'].astype(float).values
        lows = working_df['low'].astype(float).values

        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]

        current_price = float(working_df.iloc[-1]['close'])

        result = {
            'swing_highs': [round(highs[i], 2) for i in high_idx[-5:]],
            'swing_lows': [round(lows[i], 2) for i in low_idx[-5:]],
        }

        # Nearest levels above/below current price
        above = [h for h in result['swing_highs'] if h > current_price]
        below = [l for l in result['swing_lows'] if l < current_price]

        result['nearest_resistance'] = min(above) if above else None
        result['nearest_support'] = max(below) if below else None

        return result

    # ─────────────── Market Structure (MSB / CHoCH) ───────────────

    def detect_market_structure(self, df: pd.DataFrame, mode: TradingMode = TradingMode.SWING) -> Dict:
        """Detect HH/HL/LH/LL sequence and Market Structure Breaks.

        MSB  (Market Structure Break): price breaks through last significant
             swing level — confirms trend change.
        CHoCH (Change of Character): first counter-trend swing — early warning.
        """
        if len(df) < 20:
            return {'structure': 'insufficient_data'}
        try:
            working_df = df.tail(min(len(df), 160)).reset_index(drop=True)
            high  = working_df['high'].astype(float).values
            low   = working_df['low'].astype(float).values
            close = working_df['close'].astype(float).values

            order = max(8, min(18, len(working_df) // 10)) if mode == TradingMode.POSITION else max(3, min(12, len(working_df) // 14))
            min_idx = argrelextrema(low,  np.less,    order=order)[0]
            max_idx = argrelextrema(high, np.greater, order=order)[0]

            if len(min_idx) < 2 or len(max_idx) < 2:
                return {'structure': 'insufficient_pivots'}

            recent_highs = [float(high[i]) for i in max_idx[-4:]]
            recent_lows  = [float(low[i])  for i in min_idx[-4:]]

            hh = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            lh = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
            hl = sum(1 for i in range(1, len(recent_lows))  if recent_lows[i]  > recent_lows[i-1])
            ll = sum(1 for i in range(1, len(recent_lows))  if recent_lows[i]  < recent_lows[i-1])

            if hh >= lh and hl >= ll:
                structure = 'uptrend'
            elif lh > hh and ll > hl:
                structure = 'downtrend'
            else:
                structure = 'ranging'

            current_price        = float(close[-1])
            last_swing_high      = round(float(high[max_idx[-1]]), 2)
            last_swing_low       = round(float(low[min_idx[-1]]),  2)
            prev_swing_high      = round(float(high[max_idx[-2]]), 2)
            prev_swing_low       = round(float(low[min_idx[-2]]),  2)

            # CHoCH: first counter-trend swing
            choch = None
            if structure == 'uptrend' and recent_lows[-1] < recent_lows[-2]:
                choch = {
                    'type': 'bearish_choch',
                    'price': round(recent_lows[-1], 2),
                    'note': 'First LL in uptrend — early reversal warning',
                }
            elif structure == 'downtrend' and recent_highs[-1] > recent_highs[-2]:
                choch = {
                    'type': 'bullish_choch',
                    'price': round(recent_highs[-1], 2),
                    'note': 'First HH in downtrend — early reversal warning',
                }

            # MSB: price already through last key level
            msb = None
            if structure == 'uptrend' and current_price < last_swing_low:
                msb = {
                    'type': 'bearish_msb',
                    'broken_level': last_swing_low,
                    'note': 'Price broke below last swing low — structure break',
                }
            elif structure == 'downtrend' and current_price > last_swing_high:
                msb = {
                    'type': 'bullish_msb',
                    'broken_level': last_swing_high,
                    'note': 'Price broke above last swing high — structure break',
                }

            return {
                'structure':       structure,
                'hh': hh, 'lh': lh, 'hl': hl, 'll': ll,
                'last_swing_high': last_swing_high,
                'last_swing_low':  last_swing_low,
                'prev_swing_high': prev_swing_high,
                'prev_swing_low':  prev_swing_low,
                'choch':           choch,
                'msb':             msb,
                'current_price':   round(current_price, 2),
            }
        except Exception as e:
            logger.error(f"Market structure detection error: {e}")
            return {}

    # ─────────────── Trendline Quality Score ───────────────

    def score_trendline_quality(self, df: pd.DataFrame,
                                 line_info: Optional[Dict],
                                 line_type: str = 'support') -> Optional[Dict]:
        """Score a diagonal trendline on 4 dimensions (0-100).

        Dimensions:
          Angle      (30pts) — 15-45° normalised slope is sustainable
          Touches    (40pts) — more confirmed bounces = stronger line
          Recency    (20pts) — how recently was it last tested?
          Distance   (10pts) — how close is current price to the line?
        """
        if not line_info or len(df) < 10:
            return None
        try:
            # ── Derive slope + current_val from stored pivot row indices ──
            # calculate_diagonal_support/resistance stores '_x1', '_x2', 'point1', 'point2'
            x1 = line_info.get('_x1')
            x2 = line_info.get('_x2')
            pt1 = line_info.get('point1')
            pt2 = line_info.get('point2')

            if x1 is None or x2 is None or pt1 is None or pt2 is None:
                return None
            if x2 == x1:
                return None

            y1_val = float(pt1[1])
            y2_val = float(pt2[1])
            slope  = (y2_val - y1_val) / (x2 - x1)   # price per candle

            n = len(df)
            # Extrapolate trendline to the most recent candle
            current_val = y2_val + slope * (n - 1 - x2)

            avg_price = float(df['close'].mean())
            if avg_price == 0:
                return None

            # ── 1. Angle score ──────────────────────────────────────
            norm_slope = abs(slope) / avg_price * 100   # % per candle
            if 0.005 <= norm_slope <= 0.35:
                angle_score, angle_tag = 30, 'optimal'
            elif norm_slope < 0.005:
                angle_score, angle_tag = 12, 'too_flat'
            else:
                angle_score, angle_tag = 6,  'too_steep'

            # ── 2. Touch count score ────────────────────────────────
            x          = np.arange(n)
            intercept  = current_val - slope * (n - 1)
            line_vals  = slope * x + intercept
            close_vals = df['close'].astype(float).values
            tolerance  = avg_price * 0.006   # 0.6 % band around the line

            touches = int(np.sum(np.abs(close_vals - line_vals) <= tolerance))

            if touches >= 5:
                touch_score, touch_tag = 40, 'very_strong'
            elif touches == 4:
                touch_score, touch_tag = 32, 'strong'
            elif touches == 3:
                touch_score, touch_tag = 22, 'moderate'
            elif touches == 2:
                touch_score, touch_tag = 12, 'weak'
            else:
                touch_score, touch_tag = 4,  'very_weak'

            # ── 3. Recency score (pivot_count as proxy for age) ─────
            pivot_count = line_info.get('pivot_count', 0)
            if pivot_count >= 4:
                recency_score = 20
            elif pivot_count == 3:
                recency_score = 14
            else:
                recency_score = 7

            # ── 4. Distance score ───────────────────────────────────
            current_price = float(df['close'].iloc[-1])
            dist_pct = abs(current_price - current_val) / avg_price * 100
            if dist_pct <= 0.5:
                dist_score, dist_tag = 10, 'testing_now'
            elif dist_pct <= 2.0:
                dist_score, dist_tag = 7,  'nearby'
            elif dist_pct <= 5.0:
                dist_score, dist_tag = 3,  'approaching'
            else:
                dist_score, dist_tag = 0,  'distant'

            total = angle_score + touch_score + recency_score + dist_score
            total = min(100, total)
            grade = 'A' if total >= 80 else ('B' if total >= 60 else ('C' if total >= 40 else 'D'))

            return {
                'score':       total,
                'grade':       grade,
                'touch_count': touches,
                'angle_tag':   angle_tag,
                'touch_tag':   touch_tag,
                'dist_tag':    dist_tag,
            }
        except Exception as e:
            logger.error(f"Trendline quality score error: {e}")
            return None

    # ─────────────── Multi-TF Confluence Detection ───────────────

    def detect_confluence_zones(self, multi_tf_analysis: Dict,
                                  tolerance_pct: float = 0.8) -> List[Dict]:
        """Find price zones where ≥2 TF levels converge.

        Scans fibonacci, diagonal support/resistance, and swing levels
        across all provided timeframe analyses and clusters nearby prices.

        Returns top-5 zones sorted by strength (descending).
        """
        levels: List[Dict] = []

        weight_map = {'1w': 4, '1d': 3, '4h': 2, '1h': 1, '15m': 1}

        fib_data = multi_tf_analysis.get('fibonacci', {})
        for fib_tf, fib in fib_data.items():
            if not isinstance(fib, dict):
                continue
            fib_w = weight_map.get(fib_tf, 1)
            for key, price in fib.items():
                if key.startswith('fib_') and isinstance(price, (int, float)):
                    levels.append({
                        'price': float(price),
                        'source': key,
                        'tf': fib_tf,
                        'weight': fib_w,
                    })

        struct = multi_tf_analysis.get('structure', {})
        for key, val in struct.items():
            if not isinstance(val, dict):
                continue
            tf = key.replace('support_', '').replace('resistance_', '')
            base_weight = weight_map.get(tf, 1) + 1
            if 'support' in key:
                p = val.get('support_price')
                if isinstance(p, (int, float)):
                    levels.append({
                        'price': float(p),
                        'source': 'diag_support',
                        'tf': tf,
                        'weight': base_weight,
                    })
            elif 'resistance' in key:
                p = val.get('resistance_price')
                if isinstance(p, (int, float)):
                    levels.append({
                        'price': float(p),
                        'source': 'diag_resistance',
                        'tf': tf,
                        'weight': base_weight,
                    })

        swing_data = multi_tf_analysis.get('swing_levels', {})
        for stf, swing in swing_data.items():
            if not isinstance(swing, dict):
                continue
            sw = weight_map.get(stf, 1)
            for h in swing.get('swing_highs', []):
                levels.append({'price': float(h), 'source': 'swing_high', 'tf': stf, 'weight': sw})
            for l in swing.get('swing_lows', []):
                levels.append({'price': float(l), 'source': 'swing_low', 'tf': stf, 'weight': sw})

        if not levels:
            return []

        prices    = [l['price'] for l in levels]
        avg_price = float(np.mean(prices))
        tolerance = avg_price * (tolerance_pct / 100.0)

        clusters: List[Dict] = []
        used: set = set()

        for i, lvl in enumerate(levels):
            if i in used:
                continue
            group = [lvl]
            used.add(i)
            for j, other in enumerate(levels):
                if j in used:
                    continue
                if abs(lvl['price'] - other['price']) <= tolerance:
                    group.append(other)
                    used.add(j)

            if len(group) < 2:
                continue

            cluster_price = float(np.mean([g['price'] for g in group]))
            total_weight  = sum(g['weight'] for g in group)
            tfs           = sorted(set(g['tf'] for g in group))
            sources       = sorted(set(g['source'] for g in group))

            clusters.append({
                'price':       round(cluster_price, 2),
                'price_low':   round(min(g['price'] for g in group), 2),
                'price_high':  round(max(g['price'] for g in group), 2),
                'strength':    total_weight,
                'level_count': len(group),
                'timeframes':  tfs,
                'sources':     sources,
            })

        clusters.sort(key=lambda x: x['strength'], reverse=True)
        return clusters[:5]

    def detect_equal_levels(self, df: pd.DataFrame, mode: TradingMode = TradingMode.SWING) -> Dict:
        """Cluster nearly-equal swing highs/lows that often become liquidity magnets."""
        if len(df) < 20:
            return {"equal_highs": [], "equal_lows": []}

        working_df = df.tail(min(len(df), 180)).reset_index(drop=True)
        order = max(4, min(12, len(working_df) // 16)) if mode == TradingMode.SWING else max(6, min(18, len(working_df) // 14))
        highs = working_df['high'].astype(float).values
        lows = working_df['low'].astype(float).values

        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]
        price_ref = float(working_df['close'].iloc[-1])
        tolerance_pct = 0.2 if mode == TradingMode.SWING else 0.35
        tolerance_abs = max(price_ref * (tolerance_pct / 100.0), 1e-9)

        def _cluster(indices: np.ndarray, values: np.ndarray) -> List[Dict]:
            if len(indices) < 2:
                return []
            pivots = sorted([(int(i), float(values[i])) for i in indices], key=lambda item: item[1])
            groups: List[List[tuple[int, float]]] = []
            for idx, price in pivots:
                if not groups or abs(price - groups[-1][-1][1]) > tolerance_abs:
                    groups.append([(idx, price)])
                else:
                    groups[-1].append((idx, price))
            clusters = []
            for group in groups:
                if len(group) < 2:
                    continue
                prices = [p for _, p in group]
                clusters.append({
                    "price": round(float(np.mean(prices)), 2),
                    "price_low": round(float(min(prices)), 2),
                    "price_high": round(float(max(prices)), 2),
                    "touches": len(group),
                })
            return clusters[-3:]

        return {
            "equal_highs": _cluster(high_idx, highs),
            "equal_lows": _cluster(low_idx, lows),
        }

    def detect_sr_flip(self, df: pd.DataFrame, swing_levels: Dict) -> Dict:
        """Check whether a recently-broken S/R level is acting as flipped support or resistance."""
        if len(df) < 8 or not isinstance(swing_levels, dict):
            return {"status": "none", "confirmed": False}

        closes = df['close'].astype(float)
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        current_price = float(closes.iloc[-1])
        tolerance = current_price * 0.0025

        nearest_resistance = swing_levels.get("nearest_resistance")
        if isinstance(nearest_resistance, (int, float)):
            level = float(nearest_resistance)
            if current_price > level and float(lows.tail(3).min()) <= (level + tolerance):
                return {
                    "status": "bullish_flip",
                    "confirmed": True,
                    "reference_level": round(level, 2),
                    "reaction_close": round(current_price, 2),
                }

        nearest_support = swing_levels.get("nearest_support")
        if isinstance(nearest_support, (int, float)):
            level = float(nearest_support)
            if current_price < level and float(highs.tail(3).max()) >= (level - tolerance):
                return {
                    "status": "bearish_flip",
                    "confirmed": True,
                    "reference_level": round(level, 2),
                    "reaction_close": round(current_price, 2),
                }

        return {"status": "none", "confirmed": False}

    def evaluate_retest_quality(self, df: pd.DataFrame, level: Optional[float], side: str) -> Dict:
        """Score whether recent candles are respecting a support/resistance retest cleanly."""
        if level is None or len(df) < 6:
            return {"score": 0, "label": "unknown", "touches": 0, "close_acceptance": 0}

        level = float(level)
        recent = df.tail(10).copy()
        closes = recent['close'].astype(float)
        lows = recent['low'].astype(float)
        highs = recent['high'].astype(float)
        tolerance = max(level * 0.0025, 1e-9)

        if side.upper() == "LONG":
            touches = int(((lows >= (level - tolerance)) & (lows <= (level + tolerance))).sum())
            close_acceptance = int((closes >= (level - tolerance)).sum())
            reaction = int((closes.tail(3).diff().fillna(0) >= 0).sum())
        else:
            touches = int(((highs >= (level - tolerance)) & (highs <= (level + tolerance))).sum())
            close_acceptance = int((closes <= (level + tolerance)).sum())
            reaction = int((closes.tail(3).diff().fillna(0) <= 0).sum())

        score = min(100, touches * 22 + close_acceptance * 5 + reaction * 8)
        if score >= 70:
            label = "clean"
        elif score >= 45:
            label = "acceptable"
        elif score > 0:
            label = "weak"
        else:
            label = "failed"

        return {
            "score": score,
            "label": label,
            "touches": touches,
            "close_acceptance": close_acceptance,
        }

    def detect_liquidity_sweep(self, df: pd.DataFrame, swing_levels: Dict, equal_levels: Dict) -> Dict:
        """Detect simple buy-side / sell-side liquidity sweeps around clustered highs/lows."""
        if len(df) < 6:
            return {"status": "none", "confirmed": False}

        recent = df.tail(5).reset_index(drop=True)
        last = recent.iloc[-1]
        prev = recent.iloc[-2]
        current_price = float(last['close'])
        tolerance = max(current_price * 0.0025, 1e-9)

        high_candidates: List[float] = []
        low_candidates: List[float] = []
        for item in (equal_levels.get("equal_highs", []) or []):
            if isinstance(item, dict) and isinstance(item.get("price"), (int, float)):
                high_candidates.append(float(item["price"]))
        for item in (equal_levels.get("equal_lows", []) or []):
            if isinstance(item, dict) and isinstance(item.get("price"), (int, float)):
                low_candidates.append(float(item["price"]))

        nearest_resistance = swing_levels.get("nearest_resistance")
        nearest_support = swing_levels.get("nearest_support")
        if isinstance(nearest_resistance, (int, float)):
            high_candidates.append(float(nearest_resistance))
        if isinstance(nearest_support, (int, float)):
            low_candidates.append(float(nearest_support))

        above = sorted([p for p in high_candidates if p >= current_price])
        below = sorted([p for p in low_candidates if p <= current_price], reverse=True)

        if above:
            level = above[0]
            if float(last['high']) > (level + tolerance) and float(last['close']) < level and float(prev['close']) <= level:
                return {
                    "status": "buy_side_sweep",
                    "confirmed": True,
                    "reference_level": round(level, 2),
                    "sweep_extreme": round(float(last['high']), 2),
                    "close_back_inside": round(float(last['close']), 2),
                    "bias_after_sweep": "SHORT",
                }

        if below:
            level = below[0]
            if float(last['low']) < (level - tolerance) and float(last['close']) > level and float(prev['close']) >= level:
                return {
                    "status": "sell_side_sweep",
                    "confirmed": True,
                    "reference_level": round(level, 2),
                    "sweep_extreme": round(float(last['low']), 2),
                    "close_back_inside": round(float(last['close']), 2),
                    "bias_after_sweep": "LONG",
                }

        return {"status": "none", "confirmed": False}

    def detect_balance_price_range(self, fvg_data: List[Dict]) -> Optional[Dict]:
        """Approximate BPR by overlapping recent bullish and bearish imbalance zones."""
        if not isinstance(fvg_data, list) or len(fvg_data) < 2:
            return None

        bulls = [g for g in fvg_data if str(g.get("type")) == "bullish"]
        bears = [g for g in fvg_data if str(g.get("type")) == "bearish"]
        for bull in reversed(bulls[-3:]):
            for bear in reversed(bears[-3:]):
                low = max(float(bull.get("gap_low", 0)), float(bear.get("gap_low", 0)))
                high = min(float(bull.get("gap_high", 0)), float(bear.get("gap_high", 0)))
                if high > low:
                    return {
                        "price_low": round(low, 2),
                        "price_high": round(high, 2),
                        "label": "bpr_overlap",
                    }
        return None

    def _bias_from_structure(self, info: Dict) -> str:
        if not isinstance(info, dict):
            return "neutral"
        msb_type = str((info.get("msb") or {}).get("type", "")).lower()
        choch_type = str((info.get("choch") or {}).get("type", "")).lower()
        structure = str(info.get("structure", "")).lower()
        if "bullish" in msb_type or "bullish" in choch_type or structure == "uptrend":
            return "bullish"
        if "bearish" in msb_type or "bearish" in choch_type or structure == "downtrend":
            return "bearish"
        return "neutral"

    def _choose_targets(self, levels: List[float], entry_price: float, side: str) -> tuple[Optional[float], Optional[float]]:
        cleaned = sorted({round(float(v), 2) for v in levels if isinstance(v, (int, float))})
        if side == "LONG":
            candidates = [v for v in cleaned if v > entry_price]
            return (candidates[0] if candidates else None, candidates[1] if len(candidates) > 1 else None)
        candidates = [v for v in cleaned if v < entry_price]
        candidates = list(reversed(candidates))
        return (candidates[0] if candidates else None, candidates[1] if len(candidates) > 1 else None)

    def _build_scenario_for_side(
        self,
        side: str,
        current_price: float,
        supports: List[float],
        resistances: List[float],
        confluence_zones: List[Dict],
        liquidity_sweep: Dict,
        sr_flip: Dict,
        retest_quality: Dict,
        bpr_zone: Optional[Dict],
        timeframe: str,
    ) -> Dict:
        side = side.upper()
        entry_zone = None
        invalidation = None
        trigger = "wait_for_retest"

        if side == "LONG":
            support_candidates = sorted({round(float(v), 2) for v in supports if isinstance(v, (int, float)) and v < current_price}, reverse=True)
            reference = support_candidates[0] if support_candidates else None
            invalidation = support_candidates[1] if len(support_candidates) > 1 else reference
            for zone in confluence_zones:
                low = zone.get("price_low")
                high = zone.get("price_high")
                if isinstance(low, (int, float)) and isinstance(high, (int, float)) and float(high) <= current_price * 1.01:
                    entry_zone = (round(float(low), 2), round(float(high), 2))
                    break
            if entry_zone is None and isinstance(reference, (int, float)):
                band = max(reference * 0.002, current_price * 0.0018)
                entry_zone = (round(reference - band, 2), round(reference + band, 2))
            if liquidity_sweep.get("status") == "sell_side_sweep":
                trigger = "reclaim_after_sell_side_sweep"
            elif sr_flip.get("status") == "bullish_flip":
                trigger = "bullish_sr_flip_retest"
            elif bpr_zone:
                trigger = "acceptance_from_bpr"
        else:
            resistance_candidates = sorted({round(float(v), 2) for v in resistances if isinstance(v, (int, float)) and v > current_price})
            reference = resistance_candidates[0] if resistance_candidates else None
            invalidation = resistance_candidates[1] if len(resistance_candidates) > 1 else reference
            for zone in confluence_zones:
                low = zone.get("price_low")
                high = zone.get("price_high")
                if isinstance(low, (int, float)) and isinstance(high, (int, float)) and float(low) >= current_price * 0.99:
                    entry_zone = (round(float(low), 2), round(float(high), 2))
                    break
            if entry_zone is None and isinstance(reference, (int, float)):
                band = max(reference * 0.002, current_price * 0.0018)
                entry_zone = (round(reference - band, 2), round(reference + band, 2))
            if liquidity_sweep.get("status") == "buy_side_sweep":
                trigger = "reject_after_buy_side_sweep"
            elif sr_flip.get("status") == "bearish_flip":
                trigger = "bearish_sr_flip_retest"
            elif bpr_zone:
                trigger = "acceptance_from_bpr"

        entry_mid = None
        if isinstance(entry_zone, tuple) and len(entry_zone) == 2:
            entry_mid = round((float(entry_zone[0]) + float(entry_zone[1])) / 2.0, 2)
        tp1, tp2 = self._choose_targets(resistances if side == "LONG" else supports, entry_mid or current_price, side)

        if invalidation is None and entry_mid is not None:
            invalidation = entry_mid * (0.995 if side == "LONG" else 1.005)

        invalidation = round(float(invalidation), 2) if isinstance(invalidation, (int, float)) else None
        risk_pct = None
        if isinstance(entry_mid, (int, float)) and isinstance(invalidation, (int, float)) and entry_mid:
            risk_pct = round(abs(float(entry_mid) - float(invalidation)) / float(entry_mid) * 100.0, 3)

        split_entries = []
        if isinstance(entry_zone, tuple) and len(entry_zone) == 2:
            low, high = float(entry_zone[0]), float(entry_zone[1])
            split_entries = [round(high, 2), round((low + high) / 2.0, 2), round(low, 2)]
            split_entries = sorted(set(split_entries), reverse=(side == "LONG"))

        if isinstance(entry_zone, tuple) and len(entry_zone) == 2 and float(entry_zone[0]) <= current_price <= float(entry_zone[1]):
            status = "active_zone"
        else:
            status = "wait_trigger"

        trigger_conditions = []
        if isinstance(entry_mid, (int, float)):
            operator = ">=" if side == "LONG" and "reclaim" in trigger else "<=" if side == "SHORT" and "reject" in trigger else "<=" if side == "LONG" else ">="
            trigger_conditions.append({"metric": "price", "operator": operator, "value": float(entry_mid)})
        if liquidity_sweep.get("confirmed"):
            trigger_conditions.append({"metric": "price_chg_pct_1h", "operator": ">=" if side == "LONG" else "<=", "value": 0.1 if side == "LONG" else -0.1})

        invalidation_conditions = []
        if isinstance(invalidation, (int, float)):
            invalidation_conditions.append({"metric": "price", "operator": "<=" if side == "LONG" else ">=", "value": float(invalidation)})

        return {
            "side": side,
            "timeframe": timeframe,
            "status": status,
            "trigger": trigger,
            "entry_zone_low": round(float(entry_zone[0]), 2) if isinstance(entry_zone, tuple) else None,
            "entry_zone_high": round(float(entry_zone[1]), 2) if isinstance(entry_zone, tuple) else None,
            "entry_reference": entry_mid,
            "invalidation": invalidation,
            "tp1": tp1,
            "tp2": tp2,
            "risk_box_pct": risk_pct,
            "split_entries": split_entries,
            "partial_tp_plan": {
                "tp1_price": tp1,
                "tp1_exit_pct": 40,
                "tp2_price": tp2,
                "tp2_exit_pct": 40 if tp2 is not None else 0,
                "runner_exit_pct": 20 if tp2 is not None else 60,
            },
            "breakeven_rule": f"Move stop to breakeven after TP1 or after {str(retest_quality.get('label', 'clean'))} retest acceptance.",
            "retest_quality": retest_quality,
            "trap_context": liquidity_sweep,
            "sr_flip": sr_flip,
            "trigger_conditions": trigger_conditions,
            "invalidation_conditions": invalidation_conditions,
        }

    def build_scenario_engine(self, analysis: Dict, mode: TradingMode, raw_timeframes: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """Create an execution-first scenario object shared by chart, VLM, judge, and monitor."""
        if not isinstance(analysis, dict):
            return {}

        current_price = float(analysis.get("current_price", 0) or 0)
        if current_price <= 0:
            return {}

        execution_tf = "1d" if mode == TradingMode.POSITION else "4h"
        higher_tf = "1w" if mode == TradingMode.POSITION else "1d"

        market_structure = analysis.get("market_structure", {}) or {}
        exec_struct = market_structure.get(execution_tf, {}) or {}
        htf_struct = market_structure.get(higher_tf, {}) or {}

        htf_bias = self._bias_from_structure(htf_struct)
        execution_bias = self._bias_from_structure(exec_struct)

        structure = analysis.get("structure", {}) or {}
        fibonacci = analysis.get("fibonacci", {}) or {}
        swing = analysis.get("swing_levels", {}) or {}
        confluence = analysis.get("confluence_zones", []) or []
        fvg_data = (analysis.get("fvg", {}) or {}).get(execution_tf, []) or []

        supports: List[float] = []
        resistances: List[float] = []
        for val in (swing.get(execution_tf, {}) or {}).get("swing_lows", []) or []:
            if isinstance(val, (int, float)):
                supports.append(float(val))
        for val in (swing.get(execution_tf, {}) or {}).get("swing_highs", []) or []:
            if isinstance(val, (int, float)):
                resistances.append(float(val))
        for val in (swing.get(higher_tf, {}) or {}).get("swing_lows", []) or []:
            if isinstance(val, (int, float)):
                supports.append(float(val))
        for val in (swing.get(higher_tf, {}) or {}).get("swing_highs", []) or []:
            if isinstance(val, (int, float)):
                resistances.append(float(val))
        for tf in (execution_tf, higher_tf):
            fib = fibonacci.get(tf, {}) or {}
            for key in ("fib_500", "fib_618", "fib_705", "fib_786"):
                val = fib.get(key)
                if isinstance(val, (int, float)):
                    if val <= current_price:
                        supports.append(float(val))
                    if val >= current_price:
                        resistances.append(float(val))
            sup = (structure.get(f"support_{tf}") or {}).get("support_price")
            res = (structure.get(f"resistance_{tf}") or {}).get("resistance_price")
            if isinstance(sup, (int, float)):
                supports.append(float(sup))
            if isinstance(res, (int, float)):
                resistances.append(float(res))

        tf_df = (raw_timeframes or {}).get(execution_tf)
        equal_levels = self.detect_equal_levels(tf_df, mode=mode) if isinstance(tf_df, pd.DataFrame) and not tf_df.empty else {"equal_highs": [], "equal_lows": []}
        sr_flip = self.detect_sr_flip(tf_df, swing.get(execution_tf, {}) or {}) if isinstance(tf_df, pd.DataFrame) and not tf_df.empty else {"status": "none", "confirmed": False}
        liquidity_sweep = self.detect_liquidity_sweep(tf_df, swing.get(execution_tf, {}) or {}, equal_levels) if isinstance(tf_df, pd.DataFrame) and not tf_df.empty else {"status": "none", "confirmed": False}

        long_anchor = (swing.get(execution_tf, {}) or {}).get("nearest_support")
        short_anchor = (swing.get(execution_tf, {}) or {}).get("nearest_resistance")
        retest_long = self.evaluate_retest_quality(tf_df, long_anchor, "LONG") if isinstance(tf_df, pd.DataFrame) and not tf_df.empty else {"score": 0, "label": "unknown"}
        retest_short = self.evaluate_retest_quality(tf_df, short_anchor, "SHORT") if isinstance(tf_df, pd.DataFrame) and not tf_df.empty else {"score": 0, "label": "unknown"}
        bpr_zone = self.detect_balance_price_range(fvg_data)

        relevant_confluence = []
        for zone in confluence:
            if not isinstance(zone, dict):
                continue
            zone_tfs = {str(tf).lower() for tf in zone.get("timeframes", [])}
            if execution_tf in zone_tfs or higher_tf in zone_tfs:
                relevant_confluence.append(zone)

        primary_side = "LONG"
        if htf_bias == "bearish":
            primary_side = "SHORT"
        elif htf_bias == "neutral" and execution_bias == "bearish":
            primary_side = "SHORT"
        alternate_side = "SHORT" if primary_side == "LONG" else "LONG"

        primary = self._build_scenario_for_side(
            primary_side,
            current_price,
            supports,
            resistances,
            relevant_confluence,
            liquidity_sweep,
            sr_flip,
            retest_long if primary_side == "LONG" else retest_short,
            bpr_zone,
            execution_tf,
        )
        alternate = self._build_scenario_for_side(
            alternate_side,
            current_price,
            supports,
            resistances,
            relevant_confluence,
            liquidity_sweep,
            sr_flip,
            retest_short if alternate_side == "SHORT" else retest_long,
            bpr_zone,
            execution_tf,
        )

        if htf_bias != "neutral" and execution_bias not in ("neutral", htf_bias):
            revision_reason = f"HTF {htf_bias} but execution TF shifted to {execution_bias}; scenario must stay flexible."
        elif liquidity_sweep.get("confirmed"):
            revision_reason = f"Recent {liquidity_sweep.get('status')} detected near {liquidity_sweep.get('reference_level')}."
        else:
            revision_reason = "No forced scenario revision."

        active_setup = primary.copy()
        active_setup["bias_alignment"] = {
            "higher_timeframe": htf_bias,
            "execution_timeframe": execution_bias,
        }

        return {
            "profile": "inbum_shipalnam",
            "higher_timeframe_bias": htf_bias,
            "execution_bias": execution_bias,
            "active_setup": active_setup,
            "primary_scenario": primary,
            "alternate_scenario": alternate,
            "liquidity_map": {
                "equal_levels": equal_levels,
                "liquidity_sweep": liquidity_sweep,
                "sr_flip": sr_flip,
                "bpr_zone": bpr_zone,
                "execution_timeframe": execution_tf,
            },
            "scenario_revision_reason": revision_reason,
        }



    def analyze_market_swing(self, df_1m: pd.DataFrame,
                              df_1d: Optional[pd.DataFrame] = None) -> Dict:
        """SWING mode: 1h, 4h, 1d focus with Fibonacci + volume profile + FVG.
        df_1d: pre-loaded from GCS if available (for EMA200 on 1d)."""

        result = {
            'mode': 'swing',
            'current_price': round(float(df_1m.iloc[-1]['close']), 2),
            'current_volume': round(float(df_1m.iloc[-1]['volume']), 2),
            'timeframes': {},
            'structure': {},
            'market_structure': {},
            'trendline_quality': {},
            'fibonacci': {},
            'volume_profile': {},
            'fvg': {},
            'swing_levels': {},
            'confluence_zones': [],
            'scenario_engine': {},
        }

        timeframes = {
            '1h': self.resample_to_timeframe(df_1m, '1h'),
            '4h': self.resample_to_timeframe(df_1m, '4h'),
        }

        # Merge GCS 1d data with resampled 1m data for a unified continuum
        df_resampled_1d = self.resample_to_timeframe(df_1m, '1d')
        timeframes['1d'] = self.merge_history(df_resampled_1d, df_1d)

        for tf_name, tf_df in timeframes.items():
            if len(tf_df) >= 5:
                result['timeframes'][tf_name] = self.calculate_indicators_for_timeframe(tf_df)
                result['timeframes'][tf_name]['candle_count'] = len(tf_df)
            else:
                result['timeframes'][tf_name] = {'error': 'insufficient_data', 'candle_count': len(tf_df)}

        # Structural analysis on 1h and 4h
        for tf_name in ['1h', '4h']:
            tf_df = timeframes.get(tf_name, pd.DataFrame())
            if len(tf_df) >= 20:
                result['structure'][f'support_{tf_name}']    = self.calculate_diagonal_support(tf_df)
                result['structure'][f'resistance_{tf_name}'] = self.calculate_diagonal_resistance(tf_df)
                result['structure'][f'divergence_{tf_name}'] = self.detect_divergences(tf_df)
                result['market_structure'][tf_name]          = self.detect_market_structure(tf_df)
                if result['structure'].get(f'support_{tf_name}'):
                    result['structure'][f'support_{tf_name}']['timeframe'] = tf_name
                if result['structure'].get(f'resistance_{tf_name}'):
                    result['structure'][f'resistance_{tf_name}']['timeframe'] = tf_name

                sup = result['structure'].get(f'support_{tf_name}')
                res = result['structure'].get(f'resistance_{tf_name}')
                if sup:
                    result['trendline_quality'][f'support_{tf_name}']    = self.score_trendline_quality(tf_df, sup, 'support')
                if res:
                    result['trendline_quality'][f'resistance_{tf_name}'] = self.score_trendline_quality(tf_df, res, 'resistance')

        # Structural analysis on 1d (chart overlay: SWING chart draws 1D candles)
        tf_1d = timeframes.get('1d', pd.DataFrame())
        if len(tf_1d) >= 20:
            result['structure']['support_1d']    = self.calculate_diagonal_support(tf_1d)
            result['structure']['resistance_1d'] = self.calculate_diagonal_resistance(tf_1d)
            result['structure']['divergence_1d'] = self.detect_divergences(tf_1d)
            result['market_structure']['1d']     = self.detect_market_structure(tf_1d)
            if result['structure'].get('support_1d'):
                result['structure']['support_1d']['timeframe'] = '1d'
            if result['structure'].get('resistance_1d'):
                result['structure']['resistance_1d']['timeframe'] = '1d'
            sup_1d = result['structure'].get('support_1d')
            res_1d = result['structure'].get('resistance_1d')
            if sup_1d:
                result['trendline_quality']['support_1d']    = self.score_trendline_quality(tf_1d, sup_1d, 'support')
            if res_1d:
                result['trendline_quality']['resistance_1d'] = self.score_trendline_quality(tf_1d, res_1d, 'resistance')

        # Fibonacci on 4h and 1d
        tf_4h = timeframes.get('4h', pd.DataFrame())
        if len(tf_4h) >= 20:
            result['fibonacci']['4h'] = self.calculate_fibonacci_levels(tf_4h)

        if len(tf_1d) >= 10:
            result['fibonacci']['1d'] = self.calculate_fibonacci_levels(tf_1d)

        # Volume profile on 4h
        if len(tf_4h) >= 20:
            result['volume_profile']['4h'] = self.calculate_volume_profile(tf_4h)

        # FVG on 4h
        if len(tf_4h) >= 10:
            result['fvg']['4h'] = self.calculate_fvg(tf_4h)

        # Swing levels on 4h and 1d
        if len(tf_4h) >= 20:
            result['swing_levels']['4h'] = self.calculate_swing_levels(tf_4h)
        if len(tf_1d) >= 20:
            result['swing_levels']['1d'] = self.calculate_swing_levels(tf_1d)

        # Multi-TF confluence zones
        result['confluence_zones'] = self.detect_confluence_zones(result)
        result['scenario_engine'] = self.build_scenario_engine(result, TradingMode.SWING, raw_timeframes=timeframes)

        # Recent 4h candles
        if len(tf_4h) >= 3:
            result['recent_4h_candles'] = self._recent_candle_data(tf_4h, count=5)

        return result

    def analyze_market_position(self, df_1m: pd.DataFrame,
                                 df_1d: Optional[pd.DataFrame] = None,
                                 df_1w: Optional[pd.DataFrame] = None) -> Dict:
        """POSITION mode: 4h, 1d, 1w focus for weeks~months holding.
        df_1d/df_1w: pre-loaded from GCS (essential for EMA200)."""

        result = {
            'mode': 'position',
            'current_price': round(float(df_1m.iloc[-1]['close']), 2),
            'current_volume': round(float(df_1m.iloc[-1]['volume']), 2),
            'timeframes': {},
            'structure': {},
            'market_structure': {},
            'trendline_quality': {},
            'fibonacci': {},
            'volume_profile': {},
            'fvg': {},
            'swing_levels': {},
            'confluence_zones': [],
            'scenario_engine': {},
        }

        timeframes = {
            '4h': self.resample_to_timeframe(df_1m, '4h'),
        }

        # Merge GCS data with resampled 1m data for a unified continuum
        df_resampled_1d = self.resample_to_timeframe(df_1m, '1d')
        timeframes['1d'] = self.merge_history(df_resampled_1d, df_1d)

        df_resampled_1w = self.resample_to_timeframe(df_1m, '1w')
        timeframes['1w'] = self.merge_history(df_resampled_1w, df_1w)

        for tf_name, tf_df in timeframes.items():
            if len(tf_df) >= 5:
                result['timeframes'][tf_name] = self.calculate_indicators_for_timeframe(tf_df)
                result['timeframes'][tf_name]['candle_count'] = len(tf_df)
            else:
                result['timeframes'][tf_name] = {'error': 'insufficient_data', 'candle_count': len(tf_df)}

        # Structure on 1d
        tf_1d = timeframes.get('1d', pd.DataFrame())
        if len(tf_1d) >= 20:
            result['structure']['support_1d']    = self.calculate_diagonal_support(tf_1d, mode=TradingMode.POSITION)
            result['structure']['resistance_1d'] = self.calculate_diagonal_resistance(tf_1d, mode=TradingMode.POSITION)
            result['structure']['divergence_1d'] = self.detect_divergences(tf_1d)
            result['market_structure']['1d']     = self.detect_market_structure(tf_1d, mode=TradingMode.POSITION)
            if result['structure'].get('support_1d'):
                result['structure']['support_1d']['timeframe'] = '1d'
            if result['structure'].get('resistance_1d'):
                result['structure']['resistance_1d']['timeframe'] = '1d'

            # Trendline quality scores
            sup  = result['structure'].get('support_1d')
            res  = result['structure'].get('resistance_1d')
            if sup:
                result['trendline_quality']['support_1d']    = self.score_trendline_quality(tf_1d, sup, 'support')
            if res:
                result['trendline_quality']['resistance_1d'] = self.score_trendline_quality(tf_1d, res, 'resistance')

        tf_1w = timeframes.get('1w', pd.DataFrame())
        if len(tf_1w) >= 10:
            result['market_structure']['1w'] = self.detect_market_structure(tf_1w, mode=TradingMode.POSITION)
            # Structure on 1w (chart overlay: POSITION chart draws 1W candles)
            result['structure']['support_1w']    = self.calculate_diagonal_support(tf_1w, mode=TradingMode.POSITION)
            result['structure']['resistance_1w'] = self.calculate_diagonal_resistance(tf_1w, mode=TradingMode.POSITION)
            if result['structure'].get('support_1w'):
                result['structure']['support_1w']['timeframe'] = '1w'
            if result['structure'].get('resistance_1w'):
                result['structure']['resistance_1w']['timeframe'] = '1w'
            sup_1w = result['structure'].get('support_1w')
            res_1w = result['structure'].get('resistance_1w')
            if sup_1w:
                result['trendline_quality']['support_1w']    = self.score_trendline_quality(tf_1w, sup_1w, 'support')
            if res_1w:
                result['trendline_quality']['resistance_1w'] = self.score_trendline_quality(tf_1w, res_1w, 'resistance')

        # Fibonacci on 1d and 1w
        if len(tf_1d) >= 10:
            result['fibonacci']['1d'] = self.calculate_fibonacci_levels(tf_1d)
        if len(tf_1w) >= 10:
            result['fibonacci']['1w'] = self.calculate_fibonacci_levels(tf_1w)

        # Volume profile on 1d
        if len(tf_1d) >= 20:
            result['volume_profile']['1d'] = self.calculate_volume_profile(tf_1d)

        # FVG on 1d
        if len(tf_1d) >= 10:
            result['fvg']['1d'] = self.calculate_fvg(tf_1d)

        # Swing levels on 1d
        if len(tf_1d) >= 20:
            result['swing_levels']['1d'] = self.calculate_swing_levels(tf_1d)

        # Multi-TF confluence zones
        result['confluence_zones'] = self.detect_confluence_zones(result)
        result['scenario_engine'] = self.build_scenario_engine(result, TradingMode.POSITION, raw_timeframes=timeframes)

        # Recent 1d candles
        if len(tf_1d) >= 3:
            result['recent_1d_candles'] = self._recent_candle_data(tf_1d, count=10)

        return result

    def analyze_market_custom(self, df_1m: pd.DataFrame, timeframe: str,
                              df_1d: Optional[pd.DataFrame] = None,
                              df_1w: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze a single custom timeframe (e.g. 15m, 1h) for one-off charts."""
        # 1. Select the base data for resampling
        # If it's a 1d timeframe, use df_1d if available to ensure long-term indicators are accurate
        tf_df = None
        if timeframe.lower() in ('1d', 'd'):
            if df_1d is not None and not df_1d.empty:
                # Merge logic (simplified for custom TF)
                hist_df = df_1d.copy()
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
                hist_df = hist_df.set_index('timestamp')
                
                recent_df = self.resample_to_timeframe(df_1m, '1d').set_index('timestamp')
                tf_df = pd.concat([hist_df, recent_df])
                tf_df = tf_df[~tf_df.index.duplicated(keep='last')]
                tf_df = tf_df.sort_index()
                
        elif timeframe.lower() in ('1w', 'w'):
            if df_1w is not None and not df_1w.empty:
                hist_df = df_1w.copy()
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
                hist_df = hist_df.set_index('timestamp')
                
                recent_df = self.resample_to_timeframe(df_1m, '1w').set_index('timestamp')
                tf_df = pd.concat([hist_df, recent_df])
                tf_df = tf_df[~tf_df.index.duplicated(keep='last')]
                tf_df = tf_df.sort_index()

        if tf_df is None:
            tf_df = self.resample_to_timeframe(df_1m, timeframe)

        if len(tf_df) < 5:
            return {"error": "insufficient_data", "candle_count": len(tf_df)}
            
        result = {
            'mode': 'custom',
            'timeframe': timeframe,
            'current_price': round(float(df_1m.iloc[-1]['close']), 2),
            'current_volume': round(float(df_1m.iloc[-1]['volume']), 2),
            'timeframes': {timeframe: self.calculate_indicators_for_timeframe(tf_df)},
            'structure': {
                f'support_{timeframe}': self.calculate_diagonal_support(tf_df),
                f'resistance_{timeframe}': self.calculate_diagonal_resistance(tf_df),
                f'divergence_{timeframe}': self.detect_divergences(tf_df),
            },
            'market_structure': {timeframe: self.detect_market_structure(tf_df)},
            'fibonacci': {timeframe: self.calculate_fibonacci_levels(tf_df)},
            'swing_levels': {timeframe: self.calculate_swing_levels(tf_df)},
            'confluence_zones': [],
            'scenario_engine': {},
        }
        if result['structure'].get(f'support_{timeframe}'):
            result['structure'][f'support_{timeframe}']['timeframe'] = timeframe
        if result['structure'].get(f'resistance_{timeframe}'):
            result['structure'][f'resistance_{timeframe}']['timeframe'] = timeframe
        
        # Scoring
        for t in ['support', 'resistance']:
            key = f'{t}_{timeframe}'
            line = result['structure'].get(key)
            if line:
                if 'trendline_quality' not in result: result['trendline_quality'] = {}
                result['trendline_quality'][key] = self.score_trendline_quality(tf_df, line, t)

        mode_hint = TradingMode.POSITION if timeframe.lower() in ('1d', '1w', '3d') else TradingMode.SWING
        result['scenario_engine'] = self.build_scenario_engine(result, mode_hint, raw_timeframes={timeframe: tf_df})
        return result

    def analyze_market(self, df_1m: pd.DataFrame, mode: TradingMode,
                       df_4h: Optional[pd.DataFrame] = None,
                       df_1d: Optional[pd.DataFrame] = None,
                       df_1w: Optional[pd.DataFrame] = None,
                       timeframe: Optional[str] = None) -> Dict:
        """Unified entry point. Calculates base HTF structure and merges LTF if requested."""
        # 1. Base analysis determined by Mode (HTF)
        if mode == TradingMode.POSITION:
            result = self.analyze_market_position(df_1m, df_1d=df_1d, df_1w=df_1w)
        else:
            result = self.analyze_market_swing(df_1m, df_1d=df_1d)
            
        # 2. Add custom LTF data if requested (MTFA overlay)
        if timeframe and timeframe not in (None, 'auto'):
            custom_result = self.analyze_market_custom(df_1m, timeframe, df_1d=df_1d, df_1w=df_1w)
            
            # Merge custom LTF into base HTF result
            if timeframe not in result['timeframes']:
                result['timeframes'][timeframe] = custom_result['timeframes'].get(timeframe)
            if 'structure' in custom_result:
                result['structure'].update(custom_result['structure'])
            if 'market_structure' in custom_result:
                result['market_structure'].update(custom_result['market_structure'])
            if 'fibonacci' in custom_result:
                result['fibonacci'].update(custom_result['fibonacci'])
            if 'swing_levels' in custom_result:
                result['swing_levels'].update(custom_result['swing_levels'])
            if 'trendline_quality' in custom_result:
                result['trendline_quality'].update(custom_result.get('trendline_quality', {}))
                
            result['timeframe'] = timeframe  # Mark as custom display TF
            
        return result

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
                    if v is not None and k in _COMPACT_KEYS:  # 화이트리스트 필터
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

        scenario_engine = analysis.get('scenario_engine', {}) or {}
        if scenario_engine:
            active = scenario_engine.get('active_setup', {}) or {}
            lines.append("[SCENARIO]")
            lines.append(
                f"  htf_bias={scenario_engine.get('higher_timeframe_bias', 'neutral')} "
                f"exec_bias={scenario_engine.get('execution_bias', 'neutral')} "
                f"side={active.get('side', 'N/A')} trigger={active.get('trigger', 'N/A')}"
            )
            if active.get('entry_zone_low') is not None or active.get('entry_zone_high') is not None:
                lines.append(
                    f"  entry_zone={active.get('entry_zone_low')}~{active.get('entry_zone_high')} "
                    f"invalidation={active.get('invalidation')} tp1={active.get('tp1')} tp2={active.get('tp2')}"
                )
            revision = scenario_engine.get('scenario_revision_reason')
            if revision:
                lines.append(f"  revision={revision}")

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

        fvg = analysis.get('fvg', {})
        if fvg:
            lines.append("[FVG]")
            for tf, gaps in fvg.items():
                if gaps:
                    for g in gaps:
                        status = "FILLED" if g.get('filled') else "UNFILLED"
                        lines.append(f"  {tf} {g['type']}: {g['gap_low']}~{g['gap_high']} ({status})")

        sl = analysis.get('swing_levels', {})
        if sl:
            lines.append("[SWING_LEVELS]")
            for tf, levels in sl.items():
                if levels:
                    lines.append(f"  {tf}: highs={levels.get('swing_highs',[])} lows={levels.get('swing_lows',[])}")
                    if levels.get('nearest_resistance'):
                        lines.append(f"  nearest_R={levels['nearest_resistance']} nearest_S={levels.get('nearest_support')}")

        ms = analysis.get('market_structure', {})
        if ms:
            lines.append("[MARKET_STRUCTURE]")
            for tf, s in ms.items():
                if not isinstance(s, dict):
                    continue
                struct   = s.get('structure', '?')
                hh, lh   = s.get('hh', 0), s.get('lh', 0)
                hl, ll   = s.get('hl', 0), s.get('ll', 0)
                choch    = s.get('choch')
                msb      = s.get('msb')
                row = (f"  {tf}: {struct} HH={hh} LH={lh} HL={hl} LL={ll}"
                       f" SwH={s.get('last_swing_high')} SwL={s.get('last_swing_low')}")
                lines.append(row)
                if choch:
                    lines.append(f"    CHoCH={choch['type']} @ {choch['price']} — {choch['note']}")
                if msb:
                    lines.append(f"    MSB={msb['type']} broken={msb['broken_level']} — {msb['note']}")

        tq = analysis.get('trendline_quality', {})
        if tq:
            lines.append("[TRENDLINE_QUALITY]")
            for key, q in tq.items():
                if q:
                    lines.append(f"  {key}: score={q['score']} grade={q['grade']}"
                                 f" touches={q['touch_count']} angle={q['angle_tag']}"
                                 f" dist={q['dist_tag']}")

        cz = analysis.get('confluence_zones', [])
        if cz:
            lines.append("[CONFLUENCE_ZONES]")
            for z in cz:
                lines.append(f"  price={z['price']} ({z['price_low']}~{z['price_high']})"
                             f" strength={z['strength']} levels={z['level_count']}"
                             f" tfs={z['timeframes']} src={z['sources']}")

        candles = (analysis.get('recent_4h_candles')
                   or analysis.get('recent_15m_candles')
                   or analysis.get('recent_5m_candles')
                   or analysis.get('recent_1d_candles'))
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


def calculate_z_score(current: float, mean: float, std: float, min_std: float = 1e-9) -> float:
    """Z-Score: (current − mean) / std.  근접-제로 std를 안전하게 처리.

    해석 기준:
      |z| >= 3.5 → EXTREME  (상위/하위 0.02%)
      |z| >= 2.5 → ANOMALY  (상위/하위 0.6%)
      |z| >= 2.0 → ELEVATED (상위/하위 2.3%)
      |z| >= 1.0 → NOTABLE  (상위/하위 15.9%)
    """
    return (current - mean) / max(abs(std), min_std)
