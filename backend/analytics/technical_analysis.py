"""
Advanced Technical Analysis Engine
Implements 200+ technical indicators and pattern recognition.

This module is the thin orchestrator.  All heavy logic lives in
dedicated sub-modules that are imported and re-exported here so
that every existing import path continues to work unchanged:

    from backend.analytics.technical_analysis import TechnicalAnalysisEngine
    from backend.analytics.technical_analysis import calculate_ema   # works too
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from scipy.signal import argrelextrema
import logging

# ---------------------------------------------------------------------------
# Sub-module imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from backend.analytics.trend_indicators import (
        calculate_ema,
        calculate_atr,
        calculate_adx,
        calculate_macd,
        calculate_sar,
        calculate_ichimoku,
        calculate_trend_strength,
        calculate_trend_indicators,
    )
except ImportError:
    from .trend_indicators import (  # type: ignore[no-redef]
        calculate_ema,
        calculate_atr,
        calculate_adx,
        calculate_macd,
        calculate_sar,
        calculate_ichimoku,
        calculate_trend_strength,
        calculate_trend_indicators,
    )

try:
    from backend.analytics.momentum_indicators import (
        calculate_rsi,
        calculate_stochastic,
        calculate_stoch_rsi,
        calculate_williams_r,
        calculate_cci,
        calculate_mfi,
        calculate_ultimate_oscillator,
        calculate_roc,
        calculate_momentum,
        calculate_momentum_indicators,
    )
except ImportError:
    from .momentum_indicators import (  # type: ignore[no-redef]
        calculate_rsi,
        calculate_stochastic,
        calculate_stoch_rsi,
        calculate_williams_r,
        calculate_cci,
        calculate_mfi,
        calculate_ultimate_oscillator,
        calculate_roc,
        calculate_momentum,
        calculate_momentum_indicators,
    )

try:
    from backend.analytics.volatility_indicators import (
        calculate_bollinger_bands,
        calculate_historical_volatility,
        calculate_keltner_channels,
        calculate_chaikin_volatility,
        calculate_volatility_indicators,
    )
except ImportError:
    from .volatility_indicators import (  # type: ignore[no-redef]
        calculate_bollinger_bands,
        calculate_historical_volatility,
        calculate_keltner_channels,
        calculate_chaikin_volatility,
        calculate_volatility_indicators,
    )

try:
    from backend.analytics.pattern_recognition import (
        detect_simple_patterns,
        detect_candlestick_patterns,
        detect_head_and_shoulders,
        detect_double_patterns,
        detect_triangle_patterns,
        detect_flag_pennant,
        detect_cup_and_handle,
        detect_chart_patterns,
        detect_patterns,
    )
except ImportError:
    from .pattern_recognition import (  # type: ignore[no-redef]
        detect_simple_patterns,
        detect_candlestick_patterns,
        detect_head_and_shoulders,
        detect_double_patterns,
        detect_triangle_patterns,
        detect_flag_pennant,
        detect_cup_and_handle,
        detect_chart_patterns,
        detect_patterns,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-exports (everything that was previously defined in this file)
# ---------------------------------------------------------------------------
__all__ = [
    # Class
    'TechnicalAnalysisEngine',
    # Trend
    'calculate_ema',
    'calculate_atr',
    'calculate_adx',
    'calculate_macd',
    'calculate_sar',
    'calculate_ichimoku',
    'calculate_trend_strength',
    'calculate_trend_indicators',
    # Momentum
    'calculate_rsi',
    'calculate_stochastic',
    'calculate_stoch_rsi',
    'calculate_williams_r',
    'calculate_cci',
    'calculate_mfi',
    'calculate_ultimate_oscillator',
    'calculate_roc',
    'calculate_momentum',
    'calculate_momentum_indicators',
    # Volatility
    'calculate_bollinger_bands',
    'calculate_historical_volatility',
    'calculate_keltner_channels',
    'calculate_chaikin_volatility',
    'calculate_volatility_indicators',
    # Patterns
    'detect_simple_patterns',
    'detect_candlestick_patterns',
    'detect_head_and_shoulders',
    'detect_double_patterns',
    'detect_triangle_patterns',
    'detect_flag_pennant',
    'detect_cup_and_handle',
    'detect_chart_patterns',
    'detect_patterns',
]


# ---------------------------------------------------------------------------
# TechnicalAnalysisEngine – thin orchestrator
# ---------------------------------------------------------------------------

class TechnicalAnalysisEngine:
    """
    Comprehensive technical analysis with pattern recognition.

    The computation is delegated to focused sub-modules:
    - trend_indicators.py
    - momentum_indicators.py
    - volatility_indicators.py
    - pattern_recognition.py

    This class preserves the original public interface exactly.
    """

    def __init__(self):
        self.indicators = {}
        self.patterns = {}
        self.signals = {}

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def analyze_stock(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform complete technical analysis on stock OHLCV data."""
        if len(price_data) < 200:
            logger.warning("Insufficient data for complete technical analysis")
            return {}

        price_data = self._standardize_columns(price_data)

        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_points': len(price_data),
            'trend_indicators': self._calculate_trend_indicators(price_data),
            'momentum_indicators': self._calculate_momentum_indicators(price_data),
            'volatility_indicators': self._calculate_volatility_indicators(price_data),
            'volume_indicators': self._calculate_volume_indicators(price_data),
            'pattern_recognition': self._detect_patterns(price_data),
            'support_resistance': self._find_support_resistance(price_data),
            'market_structure': self._analyze_market_structure(price_data),
            'composite_score': 0.0,
            'signals': [],
        }

        analysis['composite_score'] = self._calculate_composite_score(analysis)
        analysis['signals'] = self._generate_signals(analysis, price_data)

        return analysis

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise dataframe column names to lowercase."""
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adjusted_close',
        }
        df = df.rename(columns=column_map)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
        return df

    # ------------------------------------------------------------------
    # Delegating wrappers (preserve original method names)
    # ------------------------------------------------------------------

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        return calculate_trend_indicators(df)

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        return calculate_momentum_indicators(df)

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        return calculate_volatility_indicators(df)

    def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        return detect_patterns(df)

    # ------------------------------------------------------------------
    # Volume indicators (kept here – no separate sub-module needed)
    # ------------------------------------------------------------------

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        indicators: Dict = {}
        indicators['obv'] = self._calculate_obv(close, volume)
        indicators['ad_line'] = self._calculate_ad_line(high, low, close, volume)
        indicators['cmf'] = self._calculate_cmf(df)
        indicators['vroc'] = calculate_roc(volume.astype(float), 10)
        indicators['vwap'] = self._calculate_vwap(df)
        indicators['pvt'] = self._calculate_pvt(close, volume)
        indicators['volume_sma_20'] = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(volume[-1])
        indicators['volume_ratio'] = (
            float(volume[-1]) / indicators['volume_sma_20']
            if indicators['volume_sma_20'] > 0 else 1.0
        )
        indicators['force_index'] = self._calculate_force_index(close, volume)

        return indicators

    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> float:
        if len(close) < 2:
            return float(volume[-1]) if len(volume) > 0 else 0.0

        obv = 0.0
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv += volume[i]
            elif close[i] < close[i - 1]:
                obv -= volume[i]
        return obv

    def _calculate_ad_line(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        if len(high) == 0:
            return 0.0

        ad = 0.0
        for i in range(len(high)):
            if high[i] != low[i]:
                multiplier = (
                    (close[i] - low[i]) - (high[i] - close[i])
                ) / (high[i] - low[i])
                ad += multiplier * volume[i]
        return ad

    def _calculate_cmf(self, df: pd.DataFrame) -> float:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        if len(close) < 20:
            return 0.0

        mfm = ((close - low) - (high - close)) / (high - low)
        mfm[np.isnan(mfm)] = 0
        mfv = mfm * volume
        total_volume = float(np.sum(volume[-20:]))
        if total_volume == 0:
            return 0.0
        return float(np.sum(mfv[-20:])) / total_volume

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        total_volume = float(np.sum(df['volume']))
        if total_volume == 0:
            return float(typical_price.iloc[-1])
        return float(np.sum(typical_price * df['volume'])) / total_volume

    def _calculate_pvt(self, close: np.ndarray, volume: np.ndarray) -> float:
        if len(close) < 2:
            return 0.0

        pvt = np.zeros_like(close)
        pvt[0] = volume[0]
        for i in range(1, len(close)):
            pvt[i] = pvt[i - 1] + volume[i] * ((close[i] - close[i - 1]) / close[i - 1])
        return float(pvt[-1])

    def _calculate_force_index(self, close: np.ndarray, volume: np.ndarray) -> float:
        if len(close) < 13:
            return 0.0

        force = (close[1:] - close[:-1]) * volume[1:]
        return calculate_ema(force, 13)

    # ------------------------------------------------------------------
    # Support / Resistance
    # ------------------------------------------------------------------

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        window = 10
        local_max_indices = argrelextrema(high, np.greater, order=window)[0]
        local_min_indices = argrelextrema(low, np.less, order=window)[0]

        resistance_levels = high[local_max_indices]
        support_levels = low[local_min_indices]

        volume_profile_sr = self._calculate_volume_profile_sr(df)
        fib_levels = self._calculate_fibonacci_levels(high, low)

        all_resistance = np.concatenate([
            resistance_levels,
            volume_profile_sr['resistance'],
            fib_levels['resistance'],
        ])
        all_support = np.concatenate([
            support_levels,
            volume_profile_sr['support'],
            fib_levels['support'],
        ])

        resistance_clusters = self._cluster_levels(all_resistance, float(close[-1]))
        support_clusters = self._cluster_levels(all_support, float(close[-1]))

        return {
            'primary_resistance': resistance_clusters[0] if resistance_clusters else float(close[-1]) * 1.05,
            'secondary_resistance': resistance_clusters[1] if len(resistance_clusters) > 1 else float(close[-1]) * 1.10,
            'primary_support': support_clusters[0] if support_clusters else float(close[-1]) * 0.95,
            'secondary_support': support_clusters[1] if len(support_clusters) > 1 else float(close[-1]) * 0.90,
            'resistance_levels': resistance_clusters[:5],
            'support_levels': support_clusters[:5],
            'fibonacci_levels': fib_levels,
            'current_price': float(close[-1]),
        }

    def _calculate_volume_profile_sr(self, df: pd.DataFrame) -> Dict:
        close = df['close'].values
        volume = df['volume'].values

        num_bins = 50
        bins = np.linspace(close.min(), close.max(), num_bins)
        volume_profile = np.zeros(num_bins - 1)

        for i in range(len(close)):
            bin_idx = np.digitize(close[i], bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume[i]

        threshold = np.percentile(volume_profile, 70)
        high_volume_indices = np.where(volume_profile > threshold)[0]
        high_volume_prices = bins[high_volume_indices]

        current_price = float(close[-1])
        resistance = high_volume_prices[high_volume_prices > current_price]
        support = high_volume_prices[high_volume_prices < current_price]

        return {
            'resistance': resistance[:3] if len(resistance) > 0 else np.array([]),
            'support': support[-3:] if len(support) > 0 else np.array([]),
        }

    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray) -> Dict:
        recent_high = float(np.max(high[-50:]))
        recent_low = float(np.min(low[-50:]))
        diff = recent_high - recent_low

        fib_levels = {
            0.236: recent_high - (diff * 0.236),
            0.382: recent_high - (diff * 0.382),
            0.5: recent_high - (diff * 0.5),
            0.618: recent_high - (diff * 0.618),
            0.786: recent_high - (diff * 0.786),
        }

        current_price = float(high[-1])
        resistance = [level for level in fib_levels.values() if level > current_price]
        support = [level for level in fib_levels.values() if level < current_price]

        return {
            'levels': fib_levels,
            'resistance': np.array(resistance),
            'support': np.array(support),
        }

    def _cluster_levels(self, levels: np.ndarray, current_price: float) -> List[float]:
        if len(levels) == 0:
            return []

        unique_levels = np.unique(levels)
        clusters: List[float] = []
        used: set = set()

        for i, level in enumerate(unique_levels):
            if i in used:
                continue

            cluster = [float(level)]
            used.add(i)

            for j in range(i + 1, len(unique_levels)):
                if j not in used:
                    if abs(unique_levels[j] - level) / level < 0.01:
                        cluster.append(float(unique_levels[j]))
                        used.add(j)

            clusters.append(float(np.mean(cluster)))

        clusters.sort(key=lambda x: abs(x - current_price))
        return clusters

    # ------------------------------------------------------------------
    # Market structure analysis
    # ------------------------------------------------------------------

    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        structure: Dict = {}

        sma_20 = [float(np.mean(close[max(0, i - 19):i + 1])) for i in range(len(close))]
        sma_50 = [float(np.mean(close[max(0, i - 49):i + 1])) for i in range(len(close))]
        sma_200 = [float(np.mean(close[max(0, i - 199):i + 1])) for i in range(len(close))]

        if close[-1] > sma_20[-1] > sma_50[-1] > sma_200[-1]:
            structure['trend'] = 'strong_uptrend'
            structure['trend_strength'] = 1.0
        elif close[-1] > sma_50[-1] > sma_200[-1]:
            structure['trend'] = 'uptrend'
            structure['trend_strength'] = 0.7
        elif close[-1] < sma_20[-1] < sma_50[-1] < sma_200[-1]:
            structure['trend'] = 'strong_downtrend'
            structure['trend_strength'] = -1.0
        elif close[-1] < sma_50[-1] < sma_200[-1]:
            structure['trend'] = 'downtrend'
            structure['trend_strength'] = -0.7
        else:
            structure['trend'] = 'sideways'
            structure['trend_strength'] = 0.0

        structure['market_phase'] = self._identify_market_phase(df)
        structure['price_structure'] = self._analyze_price_structure(high, low)

        current_volatility = calculate_historical_volatility(close, 20)
        avg_volatility = calculate_historical_volatility(close, 60)

        if current_volatility > avg_volatility * 1.5:
            structure['volatility_regime'] = 'high'
        elif current_volatility < avg_volatility * 0.7:
            structure['volatility_regime'] = 'low'
        else:
            structure['volatility_regime'] = 'normal'

        structure['is_ranging'] = self._detect_ranging_market(df)

        return structure

    def _identify_market_phase(self, df: pd.DataFrame) -> str:
        close = df['close'].values
        volume = df['volume'].values

        if len(close) < 50:
            return 'unknown'

        price_trend = (float(close[-1]) - float(close[-20])) / float(close[-20])
        vol_base = float(volume[-30:-20].mean())
        volume_trend = (
            (float(volume[-10:].mean()) - vol_base) / vol_base
            if vol_base != 0 else 0.0
        )
        mean_close = float(np.mean(close[-20:]))
        volatility = float(np.std(close[-20:])) / mean_close if mean_close != 0 else 0.0

        if abs(price_trend) < 0.05 and volatility < 0.02:
            return 'accumulation' if volume_trend > 0.2 else 'consolidation'
        elif price_trend > 0.1:
            return 'markup' if volume_trend > 0 else 'distribution'
        elif price_trend < -0.1:
            return 'markdown'
        return 'transition'

    def _analyze_price_structure(self, high: np.ndarray, low: np.ndarray) -> Dict:
        if len(high) < 20:
            return {'structure': 'insufficient_data'}

        window = 5
        peaks = argrelextrema(high, np.greater, order=window)[0]
        troughs = argrelextrema(low, np.less, order=window)[0]

        if len(peaks) < 2 or len(troughs) < 2:
            return {'structure': 'no_clear_structure'}

        recent_peaks = peaks[-2:]
        recent_troughs = troughs[-2:]

        higher_high = high[recent_peaks[-1]] > high[recent_peaks[-2]]
        higher_low = low[recent_troughs[-1]] > low[recent_troughs[-2]]
        lower_high = high[recent_peaks[-1]] < high[recent_peaks[-2]]
        lower_low = low[recent_troughs[-1]] < low[recent_troughs[-2]]

        if higher_high and higher_low:
            return {
                'structure': 'uptrend',
                'strength': 'strong',
                'last_high': float(high[recent_peaks[-1]]),
                'last_low': float(low[recent_troughs[-1]]),
            }
        elif lower_high and lower_low:
            return {
                'structure': 'downtrend',
                'strength': 'strong',
                'last_high': float(high[recent_peaks[-1]]),
                'last_low': float(low[recent_troughs[-1]]),
            }
        return {
            'structure': 'mixed',
            'strength': 'weak',
            'last_high': float(high[recent_peaks[-1]]),
            'last_low': float(low[recent_troughs[-1]]),
        }

    def _detect_ranging_market(self, df: pd.DataFrame) -> bool:
        close = df['close'].values
        if len(close) < 20:
            return False

        recent_range = (close[-20:].max() - close[-20:].min()) / close[-20:].mean()
        adx_data = calculate_adx(df['high'].values, df['low'].values, close)
        return bool(recent_range < 0.1 and adx_data['adx'] < 25)

    # ------------------------------------------------------------------
    # Scoring and signal generation
    # ------------------------------------------------------------------

    def _calculate_composite_score(self, analysis: Dict) -> float:
        weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volume': 0.2,
            'patterns': 0.15,
            'structure': 0.1,
        }

        trend_score = 0.0
        trend_indicators = analysis.get('trend_indicators', {})
        if trend_indicators.get('price_vs_sma20', 0) > 0:
            trend_score += 0.25
        if trend_indicators.get('price_vs_sma50', 0) > 0:
            trend_score += 0.25
        if trend_indicators.get('price_vs_sma200', 0) > 0:
            trend_score += 0.25
        if trend_indicators.get('macd_histogram', 0) > 0:
            trend_score += 0.25

        momentum_score = 0.0
        momentum = analysis.get('momentum_indicators', {})
        rsi = momentum.get('rsi_14', 50)
        if 30 < rsi < 70:
            momentum_score += 0.5
        elif rsi <= 30:
            momentum_score += 1.0
        elif rsi >= 70:
            momentum_score -= 0.5

        volume_score = 0.0
        volume = analysis.get('volume_indicators', {})
        if volume.get('volume_ratio', 1) > 1.5:
            volume_score += 0.5
        if volume.get('cmf', 0) > 0:
            volume_score += 0.5

        pattern_score = 0.0
        patterns = analysis.get('pattern_recognition', {})
        bullish_patterns = ['hammer', 'morning_star', 'bullish_engulfing']
        bearish_patterns = ['shooting_star', 'evening_star', 'bearish_engulfing']
        for pattern in patterns.get('candlestick_patterns', {}):
            if pattern in bullish_patterns:
                pattern_score += 0.3
            elif pattern in bearish_patterns:
                pattern_score -= 0.3

        structure_score = 0.0
        market_structure = analysis.get('market_structure', {})
        if 'uptrend' in market_structure.get('trend', ''):
            structure_score += 0.5
        elif 'downtrend' in market_structure.get('trend', ''):
            structure_score -= 0.5

        score = (
            weights['trend'] * trend_score
            + weights['momentum'] * momentum_score
            + weights['volume'] * volume_score
            + weights['patterns'] * pattern_score
            + weights['structure'] * structure_score
        )

        return max(-1.0, min(1.0, score))

    def _generate_signals(self, analysis: Dict, df: pd.DataFrame) -> List[Dict]:
        signals: List[Dict] = []

        trend = analysis.get('trend_indicators', {})
        if trend.get('macd', 0) > trend.get('macd_signal', 0):
            signals.append({
                'type': 'trend',
                'name': 'MACD Bullish Cross',
                'strength': 'medium',
                'action': 'buy',
            })

        momentum = analysis.get('momentum_indicators', {})
        rsi = momentum.get('rsi_14', 50)
        if rsi < 30:
            signals.append({'type': 'momentum', 'name': 'RSI Oversold', 'strength': 'strong', 'action': 'buy'})
        elif rsi > 70:
            signals.append({'type': 'momentum', 'name': 'RSI Overbought', 'strength': 'strong', 'action': 'sell'})

        patterns = analysis.get('pattern_recognition', {}).get('candlestick_patterns', {})
        for pattern_name, pattern_data in patterns.items():
            if pattern_data['detected']:
                signals.append({
                    'type': 'pattern',
                    'name': f'{pattern_name.replace("_", " ").title()} Pattern',
                    'strength': 'medium',
                    'action': 'buy' if pattern_data['strength'] > 0 else 'sell',
                })

        sr = analysis.get('support_resistance', {})
        current_price = sr.get('current_price', 0)
        if current_price and sr.get('primary_support'):
            if abs(current_price - sr['primary_support']) / current_price < 0.02:
                signals.append({
                    'type': 'support_resistance',
                    'name': 'Near Support Level',
                    'strength': 'medium',
                    'action': 'buy',
                })

        return signals

    # ------------------------------------------------------------------
    # Backward-compatibility aliases for old private helper names
    # (callers that directly used _calculate_ema, etc. will still work)
    # ------------------------------------------------------------------

    def _calculate_ema(self, values, period):
        return calculate_ema(values, period)

    def _calculate_atr(self, high, low, close, period=14):
        return calculate_atr(high, low, close, period)

    def _calculate_adx(self, high, low, close, period=14):
        return calculate_adx(high, low, close, period)

    def _calculate_macd(self, close):
        return calculate_macd(close)

    def _calculate_sar(self, high, low):
        return calculate_sar(high, low)

    def _calculate_ichimoku(self, df):
        return calculate_ichimoku(df)

    def _calculate_trend_strength(self, df):
        return calculate_trend_strength(df)

    def _calculate_historical_volatility(self, prices, period):
        return calculate_historical_volatility(prices, period)

    def _calculate_keltner_channels(self, df):
        return calculate_keltner_channels(df)

    def _calculate_chaikin_volatility(self, high, low):
        return calculate_chaikin_volatility(high, low)

    def _calculate_bollinger_bands(self, close, period=20, std_dev=2):
        return calculate_bollinger_bands(close, period, std_dev)

    def _calculate_rsi(self, close, period=14):
        return calculate_rsi(close, period)

    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        return calculate_stochastic(high, low, close, k_period, d_period)

    def _calculate_stoch_rsi(self, close, period=14):
        return calculate_stoch_rsi(close, period)

    def _calculate_williams_r(self, high, low, close, period=14):
        return calculate_williams_r(high, low, close, period)

    def _calculate_cci(self, high, low, close, period=20):
        return calculate_cci(high, low, close, period)

    def _calculate_mfi(self, high, low, close, volume, period=14):
        return calculate_mfi(high, low, close, volume, period)

    def _calculate_ultimate_oscillator(self, high, low, close):
        return calculate_ultimate_oscillator(high, low, close)

    def _calculate_roc(self, values, period=10):
        return calculate_roc(values, period)

    def _calculate_momentum(self, close, period=10):
        return calculate_momentum(close, period)

    def _detect_simple_patterns(self, open_prices, high, low, close):
        return detect_simple_patterns(open_prices, high, low, close)

    def _detect_chart_patterns(self, df):
        return detect_chart_patterns(df)

    def _detect_head_and_shoulders(self, prices):
        return detect_head_and_shoulders(prices)

    def _detect_double_patterns(self, prices):
        return detect_double_patterns(prices)

    def _detect_triangle_patterns(self, df):
        return detect_triangle_patterns(df)

    def _calculate_triangle_apex(self, high_slope, high_int, low_slope, low_int):
        from backend.analytics.pattern_recognition import _calculate_triangle_apex
        return _calculate_triangle_apex(high_slope, high_int, low_slope, low_int)

    def _detect_flag_pennant(self, df):
        return detect_flag_pennant(df)

    def _detect_cup_and_handle(self, prices):
        return detect_cup_and_handle(prices)
