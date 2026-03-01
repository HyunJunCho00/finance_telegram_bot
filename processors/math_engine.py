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
    - SCALP: 5m/15m/1H focus with VWAP, order flow proxies, Keltner channels
    """

    def __init__(self):
        self.order = 5

    def merge_history(self, df_main: pd.DataFrame, df_history: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge recent Supabase data with historical GCS data seamlessly."""
        if df_history is None or df_history.empty:
            return df_main
        
        # Ensure UTC and datetime
        df_main = df_main.copy()
        df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], utc=True)
        
        df_history = df_history.copy()
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
            '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h',
            '1d': '1D',
            '1w': '1W-MON',  # Week starts Monday (Upbit KST 09:00 = UTC 00:00)
        }
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

        # [FIX] pandas_ta VWAP requires a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
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

    def calculate_fvg(self, df: pd.DataFrame, max_gaps: int = 5) -> List[Dict]:
        """Detect Fair Value Gaps (3-candle imbalance zones).
        FVG = gap between candle 1's wick and candle 3's wick.
        Unfilled FVGs act as magnets for price return."""
        if len(df) < 3:
            return []

        fvgs = []
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        current_price = float(df.iloc[-1]['close'])

        for i in range(len(df) - 2):
            # Bullish FVG: candle3.low > candle1.high (gap up)
            if lows[i + 2] > highs[i]:
                gap_low = highs[i]
                gap_high = lows[i + 2]
                # Check if unfilled (current price hasn't fully entered the gap)
                filled = current_price <= gap_high and current_price >= gap_low
                fvgs.append({
                    'type': 'bullish',
                    'gap_low': round(gap_low, 2),
                    'gap_high': round(gap_high, 2),
                    'gap_size_pct': round((gap_high - gap_low) / gap_low * 100, 4),
                    'filled': filled,
                    'candle_idx': i + 1,
                })

            # Bearish FVG: candle3.high < candle1.low (gap down)
            if highs[i + 2] < lows[i]:
                gap_high = lows[i]
                gap_low = highs[i + 2]
                filled = current_price >= gap_low and current_price <= gap_high
                fvgs.append({
                    'type': 'bearish',
                    'gap_low': round(gap_low, 2),
                    'gap_high': round(gap_high, 2),
                    'gap_size_pct': round((gap_high - gap_low) / gap_low * 100, 4),
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

        order = lookback or max(5, len(df) // 10)
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values

        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]

        current_price = float(df.iloc[-1]['close'])

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

    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """Detect HH/HL/LH/LL sequence and Market Structure Breaks.

        MSB  (Market Structure Break): price breaks through last significant
             swing level — confirms trend change.
        CHoCH (Change of Character): first counter-trend swing — early warning.
        """
        if len(df) < 20:
            return {'structure': 'insufficient_data'}
        try:
            high  = df['high'].astype(float).values
            low   = df['low'].astype(float).values
            close = df['close'].astype(float).values

            order = max(3, len(df) // 15)
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
            price_key   = 'support_price' if line_type == 'support' else 'resistance_price'
            current_val = line_info.get(price_key)
            slope       = line_info.get('slope', 0)
            if current_val is None:
                return None

            avg_price = float(df['close'].mean())
            if avg_price == 0:
                return None

            # ── 1. Angle score ──────────────────────────────────────
            norm_slope = abs(float(slope)) / avg_price * 100   # % per candle
            if 0.005 <= norm_slope <= 0.35:
                angle_score, angle_tag = 30, 'optimal'
            elif norm_slope < 0.005:
                angle_score, angle_tag = 12, 'too_flat'
            else:
                angle_score, angle_tag = 6,  'too_steep'

            # ── 2. Touch count score ────────────────────────────────
            n          = len(df)
            x          = np.arange(n)
            intercept  = float(current_val) - float(slope) * (n - 1)
            line_vals  = float(slope) * x + intercept
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
            dist_pct = abs(line_info.get('distance_pct', 100))
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

        for tf, analysis in multi_tf_analysis.items():
            if not isinstance(analysis, dict):
                continue
            w = weight_map.get(tf, 1)

            # Fibonacci levels
            fib_data = analysis.get('fibonacci', {})
            for fib_tf, fib in fib_data.items():
                if not isinstance(fib, dict):
                    continue
                fib_w = weight_map.get(fib_tf, 1)
                for key, price in fib.items():
                    if key.startswith('fib_') and isinstance(price, (int, float)):
                        levels.append({'price': float(price), 'source': key,
                                       'tf': fib_tf, 'weight': fib_w})

            # Diagonal support / resistance
            struct = analysis.get('structure', {})
            for key, val in struct.items():
                if not isinstance(val, dict):
                    continue
                if 'support' in key:
                    p = val.get('support_price')
                    if p:
                        levels.append({'price': float(p), 'source': 'diag_support',
                                       'tf': key.replace('support_', ''), 'weight': w + 1})
                elif 'resistance' in key:
                    p = val.get('resistance_price')
                    if p:
                        levels.append({'price': float(p), 'source': 'diag_resistance',
                                       'tf': key.replace('resistance_', ''), 'weight': w + 1})

            # Swing levels
            swing_data = analysis.get('swing_levels', {})
            for stf, swing in swing_data.items():
                if not isinstance(swing, dict):
                    continue
                sw = weight_map.get(stf, 1)
                for h in swing.get('swing_highs', []):
                    levels.append({'price': float(h), 'source': 'swing_high',
                                   'tf': stf, 'weight': sw})
                for l in swing.get('swing_lows', []):
                    levels.append({'price': float(l), 'source': 'swing_low',
                                   'tf': stf, 'weight': sw})

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
            result['structure']['support_1d']    = self.calculate_diagonal_support(tf_1d)
            result['structure']['resistance_1d'] = self.calculate_diagonal_resistance(tf_1d)
            result['structure']['divergence_1d'] = self.detect_divergences(tf_1d)
            result['market_structure']['1d']     = self.detect_market_structure(tf_1d)

            # Trendline quality scores
            sup  = result['structure'].get('support_1d')
            res  = result['structure'].get('resistance_1d')
            if sup:
                result['trendline_quality']['support_1d']    = self.score_trendline_quality(tf_1d, sup, 'support')
            if res:
                result['trendline_quality']['resistance_1d'] = self.score_trendline_quality(tf_1d, res, 'resistance')

        tf_1w = timeframes.get('1w', pd.DataFrame())
        if len(tf_1w) >= 10:
            result['market_structure']['1w'] = self.detect_market_structure(tf_1w)
            # Structure on 1w (chart overlay: POSITION chart draws 1W candles)
            result['structure']['support_1w']    = self.calculate_diagonal_support(tf_1w)
            result['structure']['resistance_1w'] = self.calculate_diagonal_resistance(tf_1w)
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

        # Recent 1d candles
        if len(tf_1d) >= 3:
            result['recent_1d_candles'] = self._recent_candle_data(tf_1d, count=10)

        return result

    def analyze_market(self, df_1m: pd.DataFrame, mode: TradingMode,
                       df_4h: Optional[pd.DataFrame] = None,
                       df_1d: Optional[pd.DataFrame] = None,
                       df_1w: Optional[pd.DataFrame] = None) -> Dict:
        """Unified entry point. Dispatches to mode-specific analysis.
        Higher timeframe DataFrames (df_4h, df_1d, df_1w) come from GCS
        when available, providing deeper history for EMA200 etc."""
        if mode == TradingMode.POSITION:
            return self.analyze_market_position(df_1m, df_1d=df_1d, df_1w=df_1w)
        return self.analyze_market_swing(df_1m, df_1d=df_1d)

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
    def _safe_val(self, val) -> Optional[float]:
        """Convert various TA outputs to a safe float or None."""
        try:
            if val is None: return None
            if isinstance(val, (pd.Series, np.ndarray)):
                if len(val) == 0: return None
                val = val[-1] if isinstance(val, np.ndarray) else val.iloc[-1]
            fval = float(val)
            if np.isnan(fval) or np.isinf(fval): return None
            return round(fval, 4)
        except Exception:
            return None

    def _safe_series(self, series: pd.Series) -> Optional[List[float]]:
        """Convert a pandas Series to a list of floats, handling NaNs."""
        try:
            if series is None or series.empty: return None
            return [round(float(x), 4) if not (np.isnan(x) or np.isinf(x)) else None for x in series.tolist()]
        except Exception:
            return None

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
