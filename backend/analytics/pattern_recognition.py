"""
Pattern recognition for the Technical Analysis Engine.

Covers: candlestick patterns (via TA-Lib or simplified fallback),
        chart patterns (head-and-shoulders, double top/bottom,
        triangles, flag/pennant, cup-and-handle).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from scipy import stats
from scipy.signal import argrelextrema

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------

def detect_simple_patterns(
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> Dict:
    """Detect simple candlestick patterns without TA-Lib."""
    patterns: Dict = {}

    if len(close) < 5:
        return patterns

    body = abs(float(close[-1]) - float(open_prices[-1]))
    total_range = float(high[-1]) - float(low[-1])

    if total_range > 0 and body / total_range < 0.1:
        patterns['doji'] = {'detected': True, 'strength': 1, 'position': 0}

    lower_shadow = min(float(open_prices[-1]), float(close[-1])) - float(low[-1])
    if total_range > 0 and lower_shadow / total_range > 0.6:
        patterns['hammer'] = {
            'detected': True,
            'strength': 1 if float(close[-1]) > float(open_prices[-1]) else -1,
            'position': 0,
        }

    return patterns


def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    """Return detected candlestick patterns, using TA-Lib when available."""
    open_prices = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    if TALIB_AVAILABLE:
        op = open_prices.astype(np.float64)
        hi = high.astype(np.float64)
        lo = low.astype(np.float64)
        cl = close.astype(np.float64)

        talib_patterns = {
            'doji': talib.CDLDOJI(op, hi, lo, cl),
            'hammer': talib.CDLHAMMER(op, hi, lo, cl),
            'engulfing': talib.CDLENGULFING(op, hi, lo, cl),
            'morning_star': talib.CDLMORNINGSTAR(op, hi, lo, cl),
            'evening_star': talib.CDLEVENINGSTAR(op, hi, lo, cl),
            'shooting_star': talib.CDLSHOOTINGSTAR(op, hi, lo, cl),
            'hanging_man': talib.CDLHANGINGMAN(op, hi, lo, cl),
            'three_white_soldiers': talib.CDL3WHITESOLDIERS(op, hi, lo, cl),
            'three_black_crows': talib.CDL3BLACKCROWS(op, hi, lo, cl),
            'spinning_top': talib.CDLSPINNINGTOP(op, hi, lo, cl),
            'marubozu': talib.CDLMARUBOZU(op, hi, lo, cl),
            'harami': talib.CDLHARAMI(op, hi, lo, cl),
        }

        detected: Dict = {}
        for name, result in talib_patterns.items():
            last_val = int(result[-1])
            if last_val != 0:
                detected[name] = {
                    'detected': True,
                    'strength': 1 if last_val > 0 else -1,
                    'position': 0,
                }
        return detected

    return detect_simple_patterns(open_prices, high, low, close)


# ---------------------------------------------------------------------------
# Chart patterns
# ---------------------------------------------------------------------------

def detect_head_and_shoulders(prices: np.ndarray) -> Optional[Dict]:
    """Detect head-and-shoulders pattern."""
    if len(prices) < 50:
        return None

    window = 5
    local_max_indices = argrelextrema(prices, np.greater, order=window)[0]

    if len(local_max_indices) < 3:
        return None

    recent_peaks = local_max_indices[-3:]
    left_shoulder = prices[recent_peaks[0]]
    head = prices[recent_peaks[1]]
    right_shoulder = prices[recent_peaks[2]]

    if (head > left_shoulder and head > right_shoulder and
            abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
        between_peaks = prices[recent_peaks[0]:recent_peaks[2]]
        neckline = float(np.min(between_peaks))
        return {
            'pattern': 'head_and_shoulders',
            'bearish': True,
            'neckline': neckline,
            'target': neckline - (float(head) - neckline),
            'confidence': 0.7,
        }

    return None


def detect_double_patterns(prices: np.ndarray) -> Dict:
    """Detect double-top and double-bottom patterns."""
    patterns: Dict = {}

    if len(prices) < 30:
        return patterns

    window = 5
    local_max_indices = argrelextrema(prices, np.greater, order=window)[0]
    local_min_indices = argrelextrema(prices, np.less, order=window)[0]

    if len(local_max_indices) >= 2:
        recent_tops = local_max_indices[-2:]
        top1 = prices[recent_tops[0]]
        top2 = prices[recent_tops[1]]
        if abs(top1 - top2) / top1 < 0.03:
            patterns['double_top'] = {
                'bearish': True,
                'resistance': (float(top1) + float(top2)) / 2,
                'confidence': 0.6,
            }

    if len(local_min_indices) >= 2:
        recent_bottoms = local_min_indices[-2:]
        bottom1 = prices[recent_bottoms[0]]
        bottom2 = prices[recent_bottoms[1]]
        if abs(bottom1 - bottom2) / bottom1 < 0.03:
            patterns['double_bottom'] = {
                'bullish': True,
                'support': (float(bottom1) + float(bottom2)) / 2,
                'confidence': 0.6,
            }

    return patterns


def _calculate_triangle_apex(
    high_slope: float, high_int: float, low_slope: float, low_int: float
) -> float:
    """Calculate the price level where triangle trend lines converge."""
    x = (low_int - high_int) / (high_slope - low_slope)
    return high_slope * x + high_int


def detect_triangle_patterns(df: pd.DataFrame) -> Dict:
    """Detect ascending, descending, and symmetrical triangle patterns."""
    patterns: Dict = {}

    if len(df) < 20:
        return patterns

    high = df['high'].values[-20:]
    low = df['low'].values[-20:]
    x = np.arange(len(high))

    high_slope, high_intercept, _, _, _ = stats.linregress(x, high)
    low_slope, low_intercept, _, _, _ = stats.linregress(x, low)

    if abs(high_slope) < 0.001 and low_slope > 0.001:
        patterns['ascending_triangle'] = {
            'bullish': True,
            'resistance': float(np.mean(high)),
            'confidence': 0.65,
        }
    elif high_slope < -0.001 and abs(low_slope) < 0.001:
        patterns['descending_triangle'] = {
            'bearish': True,
            'support': float(np.mean(low)),
            'confidence': 0.65,
        }
    elif high_slope < -0.001 and low_slope > 0.001:
        patterns['symmetrical_triangle'] = {
            'neutral': True,
            'apex': _calculate_triangle_apex(high_slope, high_intercept, low_slope, low_intercept),
            'confidence': 0.6,
        }

    return patterns


def detect_flag_pennant(df: pd.DataFrame) -> Dict:
    """Detect flag and pennant continuation patterns."""
    patterns: Dict = {}

    if len(df) < 30:
        return patterns

    close = df['close'].values

    momentum = close[-30:-20].mean() / close[-40:-30].mean() - 1

    if abs(momentum) > 0.1:
        recent_range = (close[-10:].max() - close[-10:].min()) / close[-10:].mean()
        if recent_range < 0.05:
            pattern_type = 'flag' if abs(momentum) > 0.15 else 'pennant'
            patterns[pattern_type] = {
                'bullish': bool(momentum > 0),
                'target': float(close[-1]) + (float(close[-1]) * abs(momentum)),
                'confidence': 0.6,
            }

    return patterns


def detect_cup_and_handle(prices: np.ndarray) -> Optional[Dict]:
    """Detect cup-and-handle continuation pattern."""
    if len(prices) < 60:
        return None

    window = 30
    left_peak = float(np.max(prices[-window:-window // 2]))
    bottom = float(np.min(prices[-window // 2:]))
    right_peak = float(prices[-1])

    if (left_peak > bottom * 1.1 and
            right_peak > bottom * 1.1 and
            abs(left_peak - right_peak) / left_peak < 0.05):
        return {
            'bullish': True,
            'resistance': (left_peak + right_peak) / 2,
            'support': bottom,
            'target': right_peak + (right_peak - bottom),
            'confidence': 0.65,
        }

    return None


def detect_chart_patterns(df: pd.DataFrame) -> Dict:
    """Run all chart-pattern detectors and aggregate results."""
    close = df['close'].values
    patterns: Dict = {}

    hs = detect_head_and_shoulders(close)
    if hs:
        patterns['head_and_shoulders'] = hs

    double_patterns = detect_double_patterns(close)
    if double_patterns:
        patterns.update(double_patterns)

    triangles = detect_triangle_patterns(df)
    if triangles:
        patterns.update(triangles)

    flag_pennant = detect_flag_pennant(df)
    if flag_pennant:
        patterns.update(flag_pennant)

    cup_handle = detect_cup_and_handle(close)
    if cup_handle:
        patterns['cup_and_handle'] = cup_handle

    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_patterns(df: pd.DataFrame) -> Dict:
    """Detect both candlestick and chart patterns, returning a combined dict."""
    return {
        'candlestick_patterns': detect_candlestick_patterns(df),
        'chart_patterns': detect_chart_patterns(df),
    }
