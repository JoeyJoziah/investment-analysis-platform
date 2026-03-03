"""
Volatility indicator calculations for the Technical Analysis Engine.

Covers: Bollinger Bands, ATR, Keltner Channels, Historical Volatility,
        Chaikin Volatility, Standard Deviation, and Normalised ATR.
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
# Low-level primitives
# ---------------------------------------------------------------------------

def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict:
    """Calculate Bollinger Bands (upper, middle, lower, width, %B)."""
    if len(close) < period:
        return {
            'upper': float(close[-1]) * 1.02,
            'middle': float(close[-1]),
            'lower': float(close[-1]) * 0.98,
            'width': float(close[-1]) * 0.04,
            'percent': 0.5,
        }

    sma = float(np.mean(close[-period:]))
    std = float(np.std(close[-period:]))

    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = upper - lower
    percent = (float(close[-1]) - lower) / width if width > 0 else 0.5

    return {
        'upper': upper,
        'middle': sma,
        'lower': lower,
        'width': width,
        'percent': percent,
    }


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


def calculate_historical_volatility(prices: np.ndarray, period: int) -> float:
    """Calculate annualised historical volatility (as a percentage)."""
    if len(prices) < period:
        return 0.0

    returns = np.diff(np.log(prices))[-period:]
    return float(np.std(returns)) * np.sqrt(252) * 100


def calculate_keltner_channels(df: pd.DataFrame) -> Dict:
    """Calculate Keltner Channels (middle = 20-EMA, width = 2 * ATR(20))."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Import from trend_indicators to avoid duplicating the EMA logic
    from backend.analytics.trend_indicators import calculate_ema

    middle_val = calculate_ema(close, 20)
    atr = calculate_atr(high, low, close, 20)

    return {
        'keltner_upper': middle_val + (2 * atr),
        'keltner_middle': middle_val,
        'keltner_lower': middle_val - (2 * atr),
    }


def calculate_chaikin_volatility(high: np.ndarray, low: np.ndarray) -> float:
    """Calculate Chaikin Volatility."""
    if len(high) < 20:
        return 0.0

    from backend.analytics.trend_indicators import calculate_ema

    hl_diff = high - low
    ema10_val = calculate_ema(hl_diff, 10)

    if len(high) < 21:
        return 0.0

    ema10_prev = calculate_ema(hl_diff[:-10], 10)
    if ema10_prev == 0:
        return 0.0

    return ((ema10_val - ema10_prev) / ema10_prev) * 100.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_volatility_indicators(df: pd.DataFrame) -> Dict:
    """Calculate all volatility indicators and return as a flat dict."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    indicators: Dict = {}

    if TALIB_AVAILABLE:
        close_d = close.astype(np.float64)
        high_d = high.astype(np.float64)
        low_d = low.astype(np.float64)

        upper, middle, lower = talib.BBANDS(
            close_d, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        indicators['bb_upper'] = upper[-1]
        indicators['bb_middle'] = middle[-1]
        indicators['bb_lower'] = lower[-1]
        bb_width = upper[-1] - lower[-1]
        indicators['bb_width'] = bb_width
        indicators['bb_percent'] = (
            (close[-1] - lower[-1]) / bb_width if bb_width > 0 else 0.5
        )

        indicators['atr_14'] = talib.ATR(high_d, low_d, close_d, timeperiod=14)[-1]
        indicators['atr_20'] = talib.ATR(high_d, low_d, close_d, timeperiod=20)[-1]
    else:
        bb_data = calculate_bollinger_bands(close)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']
        indicators['bb_width'] = bb_data['width']
        indicators['bb_percent'] = bb_data['percent']

        indicators['atr_14'] = calculate_atr(high, low, close, 14)
        indicators['atr_20'] = calculate_atr(high, low, close, 20)

    indicators.update(calculate_keltner_channels(df))

    indicators['hv_20'] = calculate_historical_volatility(close, 20)
    indicators['hv_60'] = calculate_historical_volatility(close, 60)
    indicators['chaikin_volatility'] = calculate_chaikin_volatility(high, low)

    indicators['stddev_20'] = float(np.std(close[-20:])) if len(close) >= 20 else 0.0

    atr14 = calculate_atr(high, low, close, 14)
    indicators['natr'] = (atr14 / float(close[-1])) * 100.0 if float(close[-1]) > 0 else 0.0

    return indicators
