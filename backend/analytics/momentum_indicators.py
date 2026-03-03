"""
Momentum indicator calculations for the Technical Analysis Engine.

Covers: RSI (multiple periods), Stochastic, StochRSI, Williams %R,
        CCI, MFI, Ultimate Oscillator, ROC, and Momentum.
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

def calculate_rsi(close: np.ndarray, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return 50.0

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Dict:
    """Calculate Stochastic Oscillator (%K and %D)."""
    if len(high) < k_period:
        return {'k': 50.0, 'd': 50.0}

    lowest_low = float(np.min(low[-k_period:]))
    highest_high = float(np.max(high[-k_period:]))

    if highest_high == lowest_low:
        k = 50.0
    else:
        k = ((float(close[-1]) - lowest_low) / (highest_high - lowest_low)) * 100.0

    d = k * 0.8  # simplified
    return {'k': k, 'd': d}


def calculate_stoch_rsi(close: np.ndarray, period: int = 14) -> Dict:
    """Calculate Stochastic RSI (simplified)."""
    rsi = calculate_rsi(close, period)
    return {'k': rsi, 'd': rsi * 0.8}


def calculate_williams_r(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> float:
    """Calculate Williams %R."""
    if len(high) < period:
        return -50.0

    highest_high = float(np.max(high[-period:]))
    lowest_low = float(np.min(low[-period:]))

    if highest_high == lowest_low:
        return -50.0

    return ((highest_high - float(close[-1])) / (highest_high - lowest_low)) * -100.0


def calculate_cci(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20
) -> float:
    """Calculate Commodity Channel Index."""
    if len(high) < period:
        return 0.0

    typical_price = (high + low + close) / 3.0
    sma = float(np.mean(typical_price[-period:]))
    mean_deviation = float(np.mean(np.abs(typical_price[-period:] - sma)))

    if mean_deviation == 0:
        return 0.0

    return (float(typical_price[-1]) - sma) / (0.015 * mean_deviation)


def calculate_mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
) -> float:
    """Calculate Money Flow Index."""
    if len(high) < period:
        return 50.0

    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume

    positive_flow = 0.0
    negative_flow = 0.0

    for i in range(1, min(period + 1, len(typical_price))):
        if typical_price[-i] > typical_price[-i - 1]:
            positive_flow += money_flow[-i]
        elif typical_price[-i] < typical_price[-i - 1]:
            negative_flow += money_flow[-i]

    if negative_flow == 0:
        return 100.0

    money_ratio = positive_flow / negative_flow
    return 100.0 - (100.0 / (1.0 + money_ratio))


def calculate_ultimate_oscillator(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> float:
    """Calculate Ultimate Oscillator (simplified)."""
    if len(high) < 7:
        return 50.0

    # Import here to avoid circular dependency
    from backend.analytics.trend_indicators import calculate_atr

    tr = calculate_atr(high, low, close, 7)
    bp = float(close[-1]) - float(np.min(low[-7:]))

    if tr == 0:
        return 50.0

    return (bp / tr) * 100.0


def calculate_roc(values: np.ndarray, period: int = 10) -> float:
    """Calculate Rate of Change."""
    if len(values) < period + 1:
        return 0.0

    prev = float(values[-period - 1])
    if prev == 0:
        return 0.0
    return ((float(values[-1]) - prev) / prev) * 100.0


def calculate_momentum(close: np.ndarray, period: int = 10) -> float:
    """Calculate Momentum (price difference over period)."""
    if len(close) < period + 1:
        return 0.0

    return float(close[-1]) - float(close[-period - 1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_momentum_indicators(df: pd.DataFrame) -> Dict:
    """Calculate all momentum indicators and return as a flat dict."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    indicators: Dict = {}

    if TALIB_AVAILABLE:
        close_d = close.astype(np.float64)
        high_d = high.astype(np.float64)
        low_d = low.astype(np.float64)
        volume_d = volume.astype(np.float64)

        indicators['rsi_14'] = talib.RSI(close_d, timeperiod=14)[-1]
        indicators['rsi_9'] = talib.RSI(close_d, timeperiod=9)[-1]
        indicators['rsi_25'] = talib.RSI(close_d, timeperiod=25)[-1]

        slowk, slowd = talib.STOCH(
            high_d, low_d, close_d,
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0,
        )
        indicators['stoch_k'] = slowk[-1]
        indicators['stoch_d'] = slowd[-1]

        fastk, fastd = talib.STOCHRSI(
            close_d, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        indicators['stochrsi_k'] = fastk[-1]
        indicators['stochrsi_d'] = fastd[-1]

        indicators['williams_r'] = talib.WILLR(high_d, low_d, close_d, timeperiod=14)[-1]
        indicators['cci'] = talib.CCI(high_d, low_d, close_d, timeperiod=20)[-1]
        indicators['mfi'] = talib.MFI(high_d, low_d, close_d, volume_d, timeperiod=14)[-1]
        indicators['ultimate_oscillator'] = talib.ULTOSC(high_d, low_d, close_d)[-1]
        indicators['roc'] = talib.ROC(close_d, timeperiod=10)[-1]
        indicators['momentum'] = talib.MOM(close_d, timeperiod=10)[-1]
    else:
        indicators['rsi_14'] = calculate_rsi(close, 14)
        indicators['rsi_9'] = calculate_rsi(close, 9)
        indicators['rsi_25'] = calculate_rsi(close, 25)

        stoch_data = calculate_stochastic(high, low, close)
        indicators['stoch_k'] = stoch_data['k']
        indicators['stoch_d'] = stoch_data['d']

        stoch_rsi_data = calculate_stoch_rsi(close)
        indicators['stochrsi_k'] = stoch_rsi_data['k']
        indicators['stochrsi_d'] = stoch_rsi_data['d']

        indicators['williams_r'] = calculate_williams_r(high, low, close)
        indicators['cci'] = calculate_cci(high, low, close)
        indicators['mfi'] = calculate_mfi(high, low, close, volume)
        indicators['ultimate_oscillator'] = calculate_ultimate_oscillator(high, low, close)
        indicators['roc'] = calculate_roc(close, 10)
        indicators['momentum'] = calculate_momentum(close, 10)

    return indicators
