"""
Trend indicator calculations for the Technical Analysis Engine.

Covers: SMA, EMA, MACD, ADX/DI, Parabolic SAR, Ichimoku Cloud,
        trend strength, and moving-average cross signals.
"""

import numpy as np
import pandas as pd
from typing import Dict

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level primitives shared with other sub-modules
# ---------------------------------------------------------------------------

def calculate_ema(values: np.ndarray, period: int) -> float:
    """Calculate Exponential Moving Average, returning the last value."""
    if len(values) < period:
        return float(np.mean(values)) if len(values) > 0 else 0.0

    alpha = 2.0 / (period + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1 - alpha) * ema
    return ema


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(high) < 2:
        return 0.0

    true_ranges = []
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        true_ranges.append(max(tr1, tr2, tr3))

    if len(true_ranges) < period:
        return float(np.mean(true_ranges)) if true_ranges else 0.0

    return float(np.mean(true_ranges[-period:]))


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Dict:
    """Calculate ADX and Directional Indicators (+DI/-DI)."""
    if len(high) < period + 1:
        return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}

    plus_dm = np.maximum(high[1:] - high[:-1], 0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0)

    tr = []
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr.append(max(tr1, tr2, tr3))

    if len(tr) < period:
        return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}

    avg_tr = float(np.mean(tr[-period:]))
    avg_plus_dm = float(np.mean(plus_dm[-period:]))
    avg_minus_dm = float(np.mean(minus_dm[-period:]))

    plus_di = (avg_plus_dm / avg_tr) * 100 if avg_tr > 0 else 0.0
    minus_di = (avg_minus_dm / avg_tr) * 100 if avg_tr > 0 else 0.0

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100 if (plus_di + minus_di) > 0 else 0.0
    adx = dx  # simplified – should be smoothed over 'period' bars

    return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}


# ---------------------------------------------------------------------------
# Composite trend indicators
# ---------------------------------------------------------------------------

def calculate_macd(close: np.ndarray) -> Dict:
    """Calculate MACD line, signal, and histogram (fallback implementation)."""
    macd = calculate_ema(close, 12) - calculate_ema(close, 26)
    signal = macd * 0.8  # simplified approximation
    histogram = macd - signal
    return {'macd': macd, 'signal': signal, 'histogram': histogram}


def calculate_sar(high: np.ndarray, low: np.ndarray) -> float:
    """Calculate Parabolic SAR (simplified – returns a value near recent lows)."""
    if len(high) < 2:
        return float(high[-1]) if len(high) > 0 else 0.0
    return float(np.min(low[-10:])) if len(low) >= 10 else float(low[-1])


def calculate_ichimoku(df: pd.DataFrame) -> Dict:
    """Calculate Ichimoku Cloud components."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    period9_high = pd.Series(high).rolling(window=9).max()
    period9_low = pd.Series(low).rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    period26_high = pd.Series(high).rolling(window=26).max()
    period26_low = pd.Series(low).rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    period52_high = pd.Series(high).rolling(window=52).max()
    period52_low = pd.Series(low).rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    chikou_span = pd.Series(close).shift(-26)

    return {
        'tenkan_sen': tenkan_sen.iloc[-1] if not tenkan_sen.empty else 0,
        'kijun_sen': kijun_sen.iloc[-1] if not kijun_sen.empty else 0,
        'senkou_span_a': senkou_span_a.iloc[-1] if not senkou_span_a.empty else 0,
        'senkou_span_b': senkou_span_b.iloc[-1] if not senkou_span_b.empty else 0,
        'chikou_span': chikou_span.iloc[-1] if not chikou_span.empty else 0,
    }


def calculate_trend_strength(df: pd.DataFrame) -> float:
    """Return trend strength on a 0-1 scale using ADX."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    adx_data = calculate_adx(high, low, close)
    return min(adx_data['adx'] / 50, 1.0)


# ---------------------------------------------------------------------------
# Public API: calculate all trend indicators for a standardised DataFrame
# ---------------------------------------------------------------------------

def calculate_trend_indicators(df: pd.DataFrame) -> Dict:
    """Calculate all trend-following indicators and return as a flat dict."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    indicators: Dict = {}

    if TALIB_AVAILABLE:
        close_d = close.astype(np.float64)
        high_d = high.astype(np.float64)
        low_d = low.astype(np.float64)

        indicators['sma_5'] = talib.SMA(close_d, timeperiod=5)[-1]
        indicators['sma_20'] = talib.SMA(close_d, timeperiod=20)[-1]
        indicators['sma_50'] = talib.SMA(close_d, timeperiod=50)[-1]
        indicators['sma_200'] = talib.SMA(close_d, timeperiod=200)[-1]

        indicators['ema_12'] = talib.EMA(close_d, timeperiod=12)[-1]
        indicators['ema_26'] = talib.EMA(close_d, timeperiod=26)[-1]
        indicators['ema_50'] = talib.EMA(close_d, timeperiod=50)[-1]

        macd_line, signal_line, histogram = talib.MACD(
            close_d, fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators['macd'] = macd_line[-1]
        indicators['macd_signal'] = signal_line[-1]
        indicators['macd_histogram'] = histogram[-1]
    else:
        indicators['sma_5'] = float(np.mean(close[-5:])) if len(close) >= 5 else float(close[-1])
        indicators['sma_20'] = float(np.mean(close[-20:])) if len(close) >= 20 else float(close[-1])
        indicators['sma_50'] = float(np.mean(close[-50:])) if len(close) >= 50 else float(close[-1])
        indicators['sma_200'] = float(np.mean(close[-200:])) if len(close) >= 200 else float(close[-1])

        indicators['ema_12'] = calculate_ema(close, 12)
        indicators['ema_26'] = calculate_ema(close, 26)
        indicators['ema_50'] = calculate_ema(close, 50)

        macd_data = calculate_macd(close)
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']

    adx_data = calculate_adx(high, low, close)
    indicators['adx'] = adx_data['adx']
    indicators['plus_di'] = adx_data['plus_di']
    indicators['minus_di'] = adx_data['minus_di']

    indicators['sar'] = calculate_sar(high, low)
    indicators.update(calculate_ichimoku(df))
    indicators['trend_strength'] = calculate_trend_strength(df)

    current_price = float(close[-1])
    indicators['price_vs_sma20'] = (
        (current_price - indicators['sma_20']) / indicators['sma_20']
    ) * 100
    indicators['price_vs_sma50'] = (
        (current_price - indicators['sma_50']) / indicators['sma_50']
    ) * 100
    indicators['price_vs_sma200'] = (
        (current_price - indicators['sma_200']) / indicators['sma_200']
    ) * 100

    return indicators
