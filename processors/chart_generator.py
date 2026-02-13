import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
import base64
from typing import Dict, Optional
from config.settings import settings, TradingMode
from loguru import logger
import pandas_ta as ta


class ChartGenerator:
    """Generate candlestick chart images.
    - SWING mode: 4h candles with Fibonacci, BB, EMA50/200 (for Judge VLM + Telegram)
    - SCALP mode: 5m candles with VWAP, Keltner, EMA9/21 (Telegram only, no VLM)
    """

    def __init__(self):
        self.width = settings.CHART_IMAGE_WIDTH
        self.height = settings.CHART_IMAGE_HEIGHT
        self.dpi = settings.CHART_IMAGE_DPI

    def generate_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str,
                       mode: TradingMode = TradingMode.SWING) -> Optional[bytes]:
        if mode == TradingMode.SCALP:
            return self._generate_scalp_chart(df, analysis, symbol)
        return self._generate_swing_chart(df, analysis, symbol)

    def _generate_swing_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str) -> Optional[bytes]:
        """4H chart with BB, EMA50/200, RSI, MACD - clean for VLM reading."""
        try:
            # Resample to 4h for swing
            tmp = df.copy()
            tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], utc=True)
            tmp = tmp.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                tmp[col] = tmp[col].astype(float)

            chart_df = tmp.resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().tail(60)  # Last 60 4h candles = ~10 days

            if len(chart_df) < 10:
                return None

            chart_df.index.name = 'Date'
            close = chart_df['close']
            high = chart_df['high']
            low = chart_df['low']

            addplots = []

            # Bollinger Bands
            bb = ta.bbands(close, length=20, std=2.0)
            if bb is not None and not bb.empty:
                cols = bb.columns.tolist()
                addplots.append(mpf.make_addplot(bb[cols[0]], color='#2196F3', width=0.6, linestyle='--'))
                addplots.append(mpf.make_addplot(bb[cols[2]], color='#2196F3', width=0.6, linestyle='--'))

            # EMA 50 & 200 (swing important)
            e50 = ta.ema(close, length=50) if len(close) >= 50 else None
            e200 = ta.ema(close, length=200) if len(close) >= 200 else None
            if e50 is not None and not e50.empty:
                addplots.append(mpf.make_addplot(e50, color='#FF9800', width=1.0))
            if e200 is not None and not e200.empty:
                addplots.append(mpf.make_addplot(e200, color='#9C27B0', width=1.0))

            # RSI subplot
            rsi_series = ta.rsi(close, length=14)
            if rsi_series is not None and not rsi_series.empty:
                addplots.append(mpf.make_addplot(rsi_series, panel=2, color='#9C27B0', width=0.8, ylabel='RSI'))

            # MACD subplot
            macd_result = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                cols = macd_result.columns.tolist()
                addplots.append(mpf.make_addplot(macd_result[cols[0]], panel=3, color='#2196F3', width=0.8, ylabel='MACD'))
                addplots.append(mpf.make_addplot(macd_result[cols[1]], panel=3, color='#FF5722', width=0.8))
                hist_colors = ['#4CAF50' if v >= 0 else '#F44336' for v in macd_result[cols[2]].fillna(0)]
                addplots.append(mpf.make_addplot(macd_result[cols[2]], panel=3, type='bar', color=hist_colors, width=0.5))

            mc = mpf.make_marketcolors(up='#4CAF50', down='#F44336', edge='inherit', wick='inherit',
                                       volume={'up': '#81C784', 'down': '#E57373'})
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='#E0E0E0',
                                       facecolor='white', figcolor='white')

            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                addplot=addplots if addplots else None,
                title=f'\n{symbol} 4H SWING',
                figsize=(self.width / self.dpi, self.height / self.dpi),
                returnfig=True,
                panel_ratios=(4, 1, 1, 1) if addplots else (4, 1),
            )

            price = analysis.get('current_price', '?')
            axes[0].set_title(f'{symbol} 4H | {price} | SWING', fontsize=10, loc='left')

            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Swing chart error for {symbol}: {e}")
            return None

    def _generate_scalp_chart(self, df: pd.DataFrame, analysis: Dict, symbol: str) -> Optional[bytes]:
        """5m chart with VWAP, EMA9/21, RSI - for Telegram only (no VLM)."""
        try:
            tmp = df.copy()
            tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], utc=True)
            tmp = tmp.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                tmp[col] = tmp[col].astype(float)

            chart_df = tmp.resample('5min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().tail(100)  # Last 100 5m candles = ~8 hours

            if len(chart_df) < 10:
                return None

            chart_df.index.name = 'Date'
            close = chart_df['close']

            addplots = []

            # EMA 9 & 21 (scalp fast MAs)
            e9 = ta.ema(close, length=9)
            e21 = ta.ema(close, length=21)
            if e9 is not None and not e9.empty:
                addplots.append(mpf.make_addplot(e9, color='#4CAF50', width=0.8))
            if e21 is not None and not e21.empty:
                addplots.append(mpf.make_addplot(e21, color='#E91E63', width=0.8))

            # VWAP
            vwap = ta.vwap(chart_df['high'], chart_df['low'], close, chart_df['volume'])
            if vwap is not None and not vwap.empty:
                addplots.append(mpf.make_addplot(vwap, color='#FF9800', width=1.2, linestyle='--'))

            # RSI
            rsi_series = ta.rsi(close, length=14)
            if rsi_series is not None and not rsi_series.empty:
                addplots.append(mpf.make_addplot(rsi_series, panel=2, color='#9C27B0', width=0.8, ylabel='RSI'))

            mc = mpf.make_marketcolors(up='#4CAF50', down='#F44336', edge='inherit', wick='inherit',
                                       volume={'up': '#81C784', 'down': '#E57373'})
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='#E0E0E0',
                                       facecolor='white', figcolor='white')

            fig, axes = mpf.plot(
                chart_df, type='candle', style=style, volume=True,
                addplot=addplots if addplots else None,
                title=f'\n{symbol} 5M SCALP',
                figsize=(self.width / self.dpi, self.height / self.dpi),
                returnfig=True,
                panel_ratios=(4, 1, 1) if addplots else (4, 1),
            )

            price = analysis.get('current_price', '?')
            axes[0].set_title(f'{symbol} 5M | {price} | SCALP', fontsize=10, loc='left')

            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Scalp chart error for {symbol}: {e}")
            return None

    def chart_to_base64(self, chart_bytes: bytes) -> str:
        return base64.b64encode(chart_bytes).decode('utf-8')

    def resize_for_low_res(self, chart_bytes: bytes, max_dim: int = 512) -> bytes:
        """Resize to 512x512 for VLM (~1024 tokens). Always used for Judge."""
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
