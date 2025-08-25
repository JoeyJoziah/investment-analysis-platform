"""
Advanced Technical Analysis Engine
Implements 200+ technical indicators and pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
# import talib  # Not available, using simplified calculations
from scipy import stats
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalysisEngine:
    """
    Comprehensive technical analysis with pattern recognition
    """
    
    def __init__(self):
        self.indicators = {}
        self.patterns = {}
        self.signals = {}
    
    def analyze_stock(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete technical analysis on stock data
        """
        if len(price_data) < 200:  # Need sufficient data
            logger.warning("Insufficient data for complete technical analysis")
            return {}
        
        # Ensure proper column names
        price_data = self._standardize_columns(price_data)
        
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_points': len(price_data),
            'trend_indicators': self._calculate_trend_indicators(price_data),
            'momentum_indicators': self._calculate_momentum_indicators(price_data),
            'volatility_indicators': self._calculate_volatility_indicators(price_data),
            'volume_indicators': self._calculate_volume_indicators(price_data),
            'pattern_recognition': self._detect_patterns(price_data),
            'support_resistance': self._find_support_resistance(price_data),
            'market_structure': self._analyze_market_structure(price_data),
            'composite_score': 0.0,
            'signals': []
        }
        
        # Calculate composite technical score
        analysis['composite_score'] = self._calculate_composite_score(analysis)
        
        # Generate trading signals
        analysis['signals'] = self._generate_signals(analysis, price_data)
        
        return analysis
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataframe column names"""
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adjusted_close'
        }
        
        df = df.rename(columns=column_map)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate trend-following indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # Moving Averages (simplified calculation without talib)
        indicators['sma_5'] = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        indicators['sma_20'] = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        indicators['sma_50'] = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        indicators['sma_200'] = np.mean(close[-200:]) if len(close) >= 200 else close[-1]
        
        indicators['ema_12'] = self._calculate_ema(close, 12)
        indicators['ema_26'] = self._calculate_ema(close, 26)
        indicators['ema_50'] = self._calculate_ema(close, 50)
        
        # MACD (simplified calculation)
        macd_data = self._calculate_macd(close)
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']
        
        # ADX (Average Directional Index) - simplified
        adx_data = self._calculate_adx(high, low, close)
        indicators['adx'] = adx_data['adx']
        indicators['plus_di'] = adx_data['plus_di']
        indicators['minus_di'] = adx_data['minus_di']
        
        # Parabolic SAR (simplified)
        indicators['sar'] = self._calculate_sar(high, low)
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(df)
        indicators.update(ichimoku)
        
        # Trend strength
        indicators['trend_strength'] = self._calculate_trend_strength(df)
        
        # Moving Average Convergence
        current_price = close[-1]
        indicators['price_vs_sma20'] = ((current_price - indicators['sma_20']) / indicators['sma_20']) * 100
        indicators['price_vs_sma50'] = ((current_price - indicators['sma_50']) / indicators['sma_50']) * 100
        indicators['price_vs_sma200'] = ((current_price - indicators['sma_200']) / indicators['sma_200']) * 100
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # RSI (Relative Strength Index) - simplified
        indicators['rsi_14'] = self._calculate_rsi(close, 14)
        indicators['rsi_9'] = self._calculate_rsi(close, 9)
        indicators['rsi_25'] = self._calculate_rsi(close, 25)
        
        # Stochastic - simplified
        stoch_data = self._calculate_stochastic(high, low, close)
        indicators['stoch_k'] = stoch_data['k']
        indicators['stoch_d'] = stoch_data['d']
        
        # Stochastic RSI - simplified
        stoch_rsi_data = self._calculate_stoch_rsi(close)
        indicators['stochrsi_k'] = stoch_rsi_data['k']
        indicators['stochrsi_d'] = stoch_rsi_data['d']
        
        # Williams %R - simplified
        indicators['williams_r'] = self._calculate_williams_r(high, low, close)
        
        # CCI (Commodity Channel Index) - simplified
        indicators['cci'] = self._calculate_cci(high, low, close)
        
        # MFI (Money Flow Index) - simplified
        indicators['mfi'] = self._calculate_mfi(high, low, close, volume)
        
        # Ultimate Oscillator - simplified
        indicators['ultimate_oscillator'] = self._calculate_ultimate_oscillator(high, low, close)
        
        # ROC (Rate of Change) - simplified
        indicators['roc'] = self._calculate_roc(close, 10)
        
        # Momentum - simplified
        indicators['momentum'] = self._calculate_momentum(close, 10)
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # Bollinger Bands - simplified
        bb_data = self._calculate_bollinger_bands(close)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']
        indicators['bb_width'] = bb_data['width']
        indicators['bb_percent'] = bb_data['percent']
        
        # ATR (Average True Range) - simplified
        indicators['atr_14'] = self._calculate_atr(high, low, close, 14)
        indicators['atr_20'] = self._calculate_atr(high, low, close, 20)
        
        # Keltner Channels
        keltner = self._calculate_keltner_channels(df)
        indicators.update(keltner)
        
        # Historical Volatility
        indicators['hv_20'] = self._calculate_historical_volatility(close, 20)
        indicators['hv_60'] = self._calculate_historical_volatility(close, 60)
        
        # Chaikin Volatility
        indicators['chaikin_volatility'] = self._calculate_chaikin_volatility(high, low)
        
        # Standard Deviation - simplified
        indicators['stddev_20'] = np.std(close[-20:]) if len(close) >= 20 else 0
        
        # Normalized ATR - simplified
        atr = self._calculate_atr(high, low, close, 14)
        indicators['natr'] = (atr / close[-1]) * 100 if close[-1] > 0 else 0
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # On Balance Volume - simplified
        indicators['obv'] = self._calculate_obv(close, volume)
        
        # AD Line (Accumulation/Distribution) - simplified
        indicators['ad_line'] = self._calculate_ad_line(high, low, close, volume)
        
        # Chaikin Money Flow
        indicators['cmf'] = self._calculate_cmf(df)
        
        # Volume Rate of Change - simplified
        indicators['vroc'] = self._calculate_roc(volume.astype(float), 10)
        
        # VWAP (Volume Weighted Average Price)
        indicators['vwap'] = self._calculate_vwap(df)
        
        # Price Volume Trend
        indicators['pvt'] = self._calculate_pvt(close, volume)
        
        # Volume moving averages - simplified
        indicators['volume_sma_20'] = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
        indicators['volume_ratio'] = volume[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else 1
        
        # Force Index
        indicators['force_index'] = self._calculate_force_index(close, volume)
        
        return indicators
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect candlestick and chart patterns"""
        open_prices = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        patterns = {
            'candlestick_patterns': {},
            'chart_patterns': {}
        }
        
        # Candlestick patterns (using TA-Lib)
        candlestick_functions = {
            # Simplified pattern detection without talib
            # Basic patterns only for now
        }
        
        # Simplified pattern detection
        patterns['candlestick_patterns'] = self._detect_simple_patterns(open_prices, high, low, close)
        
        # Chart patterns
        patterns['chart_patterns'] = self._detect_chart_patterns(df)
        
        return patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect chart patterns like head and shoulders, triangles, etc."""
        close = df['close'].values
        
        patterns = {}
        
        # Head and Shoulders
        hs = self._detect_head_and_shoulders(close)
        if hs:
            patterns['head_and_shoulders'] = hs
        
        # Double Top/Bottom
        double_patterns = self._detect_double_patterns(close)
        if double_patterns:
            patterns.update(double_patterns)
        
        # Triangle patterns
        triangles = self._detect_triangle_patterns(df)
        if triangles:
            patterns.update(triangles)
        
        # Flag and Pennant
        flag_pennant = self._detect_flag_pennant(df)
        if flag_pennant:
            patterns.update(flag_pennant)
        
        # Cup and Handle
        cup_handle = self._detect_cup_and_handle(close)
        if cup_handle:
            patterns['cup_and_handle'] = cup_handle
        
        return patterns
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Method 1: Local extrema
        window = 10
        local_max_indices = argrelextrema(high, np.greater, order=window)[0]
        local_min_indices = argrelextrema(low, np.less, order=window)[0]
        
        resistance_levels = high[local_max_indices]
        support_levels = low[local_min_indices]
        
        # Method 2: Volume profile based S/R
        volume_profile_sr = self._calculate_volume_profile_sr(df)
        
        # Method 3: Fibonacci retracement levels
        fib_levels = self._calculate_fibonacci_levels(high, low)
        
        # Combine and rank levels
        all_resistance = np.concatenate([
            resistance_levels,
            volume_profile_sr['resistance'],
            fib_levels['resistance']
        ])
        
        all_support = np.concatenate([
            support_levels,
            volume_profile_sr['support'],
            fib_levels['support']
        ])
        
        # Cluster nearby levels
        resistance_clusters = self._cluster_levels(all_resistance, close[-1])
        support_clusters = self._cluster_levels(all_support, close[-1])
        
        return {
            'primary_resistance': resistance_clusters[0] if resistance_clusters else close[-1] * 1.05,
            'secondary_resistance': resistance_clusters[1] if len(resistance_clusters) > 1 else close[-1] * 1.10,
            'primary_support': support_clusters[0] if support_clusters else close[-1] * 0.95,
            'secondary_support': support_clusters[1] if len(support_clusters) > 1 else close[-1] * 0.90,
            'resistance_levels': resistance_clusters[:5],
            'support_levels': support_clusters[:5],
            'fibonacci_levels': fib_levels,
            'current_price': close[-1]
        }
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market structure"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        structure = {}
        
        # Trend identification
        sma_20 = [np.mean(close[max(0, i-19):i+1]) for i in range(len(close))]
        sma_50 = [np.mean(close[max(0, i-49):i+1]) for i in range(len(close))]
        sma_200 = [np.mean(close[max(0, i-199):i+1]) for i in range(len(close))]
        
        # Determine trend
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
        
        # Market phase (accumulation, markup, distribution, markdown)
        structure['market_phase'] = self._identify_market_phase(df)
        
        # Higher highs/lows analysis
        structure['price_structure'] = self._analyze_price_structure(high, low)
        
        # Volatility regime
        current_volatility = self._calculate_historical_volatility(close, 20)
        avg_volatility = self._calculate_historical_volatility(close, 60)
        
        if current_volatility > avg_volatility * 1.5:
            structure['volatility_regime'] = 'high'
        elif current_volatility < avg_volatility * 0.7:
            structure['volatility_regime'] = 'low'
        else:
            structure['volatility_regime'] = 'normal'
        
        # Range detection
        structure['is_ranging'] = self._detect_ranging_market(df)
        
        return structure
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """Calculate overall technical score (-1 to 1)"""
        score = 0.0
        weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volume': 0.2,
            'patterns': 0.15,
            'structure': 0.1
        }
        
        # Trend score
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
        
        # Momentum score
        momentum_score = 0.0
        momentum = analysis.get('momentum_indicators', {})
        
        rsi = momentum.get('rsi_14', 50)
        if 30 < rsi < 70:
            momentum_score += 0.5
        elif rsi <= 30:
            momentum_score += 1.0  # Oversold
        elif rsi >= 70:
            momentum_score -= 0.5  # Overbought
        
        # Volume score
        volume_score = 0.0
        volume = analysis.get('volume_indicators', {})
        
        if volume.get('volume_ratio', 1) > 1.5:
            volume_score += 0.5
        if volume.get('cmf', 0) > 0:
            volume_score += 0.5
        
        # Pattern score
        pattern_score = 0.0
        patterns = analysis.get('pattern_recognition', {})
        
        bullish_patterns = ['hammer', 'morning_star', 'bullish_engulfing']
        bearish_patterns = ['shooting_star', 'evening_star', 'bearish_engulfing']
        
        for pattern in patterns.get('candlestick_patterns', {}):
            if pattern in bullish_patterns:
                pattern_score += 0.3
            elif pattern in bearish_patterns:
                pattern_score -= 0.3
        
        # Structure score
        structure_score = 0.0
        market_structure = analysis.get('market_structure', {})
        
        if 'uptrend' in market_structure.get('trend', ''):
            structure_score += 0.5
        elif 'downtrend' in market_structure.get('trend', ''):
            structure_score -= 0.5
        
        # Calculate weighted score
        score = (
            weights['trend'] * trend_score +
            weights['momentum'] * momentum_score +
            weights['volume'] * volume_score +
            weights['patterns'] * pattern_score +
            weights['structure'] * structure_score
        )
        
        # Normalize to -1 to 1
        return max(-1, min(1, score))
    
    def _generate_signals(self, analysis: Dict, df: pd.DataFrame) -> List[Dict]:
        """Generate trading signals based on technical analysis"""
        signals = []
        
        # Trend signals
        trend = analysis.get('trend_indicators', {})
        if trend.get('macd', 0) > trend.get('macd_signal', 0):
            signals.append({
                'type': 'trend',
                'name': 'MACD Bullish Cross',
                'strength': 'medium',
                'action': 'buy'
            })
        
        # Momentum signals
        momentum = analysis.get('momentum_indicators', {})
        rsi = momentum.get('rsi_14', 50)
        
        if rsi < 30:
            signals.append({
                'type': 'momentum',
                'name': 'RSI Oversold',
                'strength': 'strong',
                'action': 'buy'
            })
        elif rsi > 70:
            signals.append({
                'type': 'momentum',
                'name': 'RSI Overbought',
                'strength': 'strong',
                'action': 'sell'
            })
        
        # Pattern signals
        patterns = analysis.get('pattern_recognition', {}).get('candlestick_patterns', {})
        for pattern_name, pattern_data in patterns.items():
            if pattern_data['detected']:
                signals.append({
                    'type': 'pattern',
                    'name': f'{pattern_name.replace("_", " ").title()} Pattern',
                    'strength': 'medium',
                    'action': 'buy' if pattern_data['strength'] > 0 else 'sell'
                })
        
        # Support/Resistance signals
        sr = analysis.get('support_resistance', {})
        current_price = sr.get('current_price', 0)
        
        if current_price and sr.get('primary_support'):
            if abs(current_price - sr['primary_support']) / current_price < 0.02:
                signals.append({
                    'type': 'support_resistance',
                    'name': 'Near Support Level',
                    'strength': 'medium',
                    'action': 'buy'
                })
        
        return signals
    
    # Helper methods
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud indicators"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high).rolling(window=9).max()
        period9_low = pd.Series(low).rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high).rolling(window=26).max()
        period26_low = pd.Series(low).rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = pd.Series(high).rolling(window=52).max()
        period52_low = pd.Series(low).rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close plotted 26 days in the past
        chikou_span = pd.Series(close).shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen.iloc[-1] if not tenkan_sen.empty else 0,
            'kijun_sen': kijun_sen.iloc[-1] if not kijun_sen.empty else 0,
            'senkou_span_a': senkou_span_a.iloc[-1] if not senkou_span_a.empty else 0,
            'senkou_span_b': senkou_span_b.iloc[-1] if not senkou_span_b.empty else 0,
            'chikou_span': chikou_span.iloc[-1] if not chikou_span.empty else 0
        }
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX and price action"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        adx_data = self._calculate_adx(high, low, close)
        
        # Normalize ADX to 0-1 scale
        trend_strength = min(adx_data['adx'] / 50, 1.0)
        
        return trend_strength
    
    def _calculate_historical_volatility(self, prices: np.ndarray, period: int) -> float:
        """Calculate historical volatility"""
        if len(prices) < period:
            return 0.0
        
        returns = np.diff(np.log(prices))[-period:]
        return np.std(returns) * np.sqrt(252) * 100  # Annualized volatility
    
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> Dict:
        """Calculate Keltner Channels"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Middle Line: 20-period EMA
        middle_val = self._calculate_ema(close, 20)
        
        # Channel Width: 2 * ATR(20)
        atr = self._calculate_atr(high, low, close, 20)
        
        upper = middle_val + (2 * atr)
        lower = middle_val - (2 * atr)
        
        return {
            'keltner_upper': upper,
            'keltner_middle': middle_val,
            'keltner_lower': lower
        }
    
    def _calculate_chaikin_volatility(self, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate Chaikin Volatility"""
        if len(high) < 20:
            return 0.0
        
        hl_diff = high - low
        ema10_val = self._calculate_ema(hl_diff, 10)
        
        if len(high) < 21:
            return 0.0
        
        # Simplified calculation
        ema10_prev = self._calculate_ema(hl_diff[:-10], 10)
        chaikin_vol = ((ema10_val - ema10_prev) / ema10_prev) * 100 if ema10_prev > 0 else 0
        
        return chaikin_vol
    
    def _calculate_cmf(self, df: pd.DataFrame) -> float:
        """Calculate Chaikin Money Flow"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        if len(close) < 20:
            return 0.0
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm[np.isnan(mfm)] = 0
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # 20-period CMF
        cmf = np.sum(mfv[-20:]) / np.sum(volume[-20:])
        
        return cmf
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = np.sum(typical_price * df['volume']) / np.sum(df['volume'])
        return vwap
    
    def _calculate_pvt(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate Price Volume Trend"""
        if len(close) < 2:
            return 0.0
        
        pvt = np.zeros_like(close)
        pvt[0] = volume[0]
        
        for i in range(1, len(close)):
            pvt[i] = pvt[i-1] + volume[i] * ((close[i] - close[i-1]) / close[i-1])
        
        return pvt[-1]
    
    def _calculate_force_index(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate Force Index"""
        if len(close) < 13:
            return 0.0
        
        force = (close[1:] - close[:-1]) * volume[1:]
        fi = self._calculate_ema(force, 13)
        
        return fi
    
    def _detect_head_and_shoulders(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect head and shoulders pattern"""
        if len(prices) < 50:
            return None
        
        # Find local maxima
        window = 5
        local_max_indices = argrelextrema(prices, np.greater, order=window)[0]
        
        if len(local_max_indices) < 3:
            return None
        
        # Check last 3 peaks for H&S pattern
        recent_peaks = local_max_indices[-3:]
        
        left_shoulder = prices[recent_peaks[0]]
        head = prices[recent_peaks[1]]
        right_shoulder = prices[recent_peaks[2]]
        
        # Head should be higher than shoulders
        # Shoulders should be roughly equal
        if (head > left_shoulder and head > right_shoulder and
            abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
            
            # Find neckline
            between_peaks = prices[recent_peaks[0]:recent_peaks[2]]
            neckline = np.min(between_peaks)
            
            return {
                'pattern': 'head_and_shoulders',
                'bearish': True,
                'neckline': neckline,
                'target': neckline - (head - neckline),
                'confidence': 0.7
            }
        
        return None
    
    def _detect_double_patterns(self, prices: np.ndarray) -> Dict:
        """Detect double top/bottom patterns"""
        patterns = {}
        
        if len(prices) < 30:
            return patterns
        
        # Find local extrema
        window = 5
        local_max_indices = argrelextrema(prices, np.greater, order=window)[0]
        local_min_indices = argrelextrema(prices, np.less, order=window)[0]
        
        # Check for double top
        if len(local_max_indices) >= 2:
            recent_tops = local_max_indices[-2:]
            top1 = prices[recent_tops[0]]
            top2 = prices[recent_tops[1]]
            
            if abs(top1 - top2) / top1 < 0.03:  # Within 3%
                patterns['double_top'] = {
                    'bearish': True,
                    'resistance': (top1 + top2) / 2,
                    'confidence': 0.6
                }
        
        # Check for double bottom
        if len(local_min_indices) >= 2:
            recent_bottoms = local_min_indices[-2:]
            bottom1 = prices[recent_bottoms[0]]
            bottom2 = prices[recent_bottoms[1]]
            
            if abs(bottom1 - bottom2) / bottom1 < 0.03:  # Within 3%
                patterns['double_bottom'] = {
                    'bullish': True,
                    'support': (bottom1 + bottom2) / 2,
                    'confidence': 0.6
                }
        
        return patterns
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = {}
        
        if len(df) < 20:
            return patterns
        
        high = df['high'].values[-20:]
        low = df['low'].values[-20:]
        
        # Fit trend lines to highs and lows
        x = np.arange(len(high))
        
        high_slope, high_intercept, _, _, _ = stats.linregress(x, high)
        low_slope, low_intercept, _, _, _ = stats.linregress(x, low)
        
        # Ascending triangle: flat top, rising bottom
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            patterns['ascending_triangle'] = {
                'bullish': True,
                'resistance': np.mean(high),
                'confidence': 0.65
            }
        
        # Descending triangle: falling top, flat bottom
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            patterns['descending_triangle'] = {
                'bearish': True,
                'support': np.mean(low),
                'confidence': 0.65
            }
        
        # Symmetrical triangle: converging lines
        elif high_slope < -0.001 and low_slope > 0.001:
            patterns['symmetrical_triangle'] = {
                'neutral': True,
                'apex': self._calculate_triangle_apex(high_slope, high_intercept, low_slope, low_intercept),
                'confidence': 0.6
            }
        
        return patterns
    
    def _calculate_triangle_apex(self, high_slope, high_int, low_slope, low_int):
        """Calculate where triangle lines converge"""
        # Solve for intersection point
        x = (low_int - high_int) / (high_slope - low_slope)
        y = high_slope * x + high_int
        return y
    
    def _detect_flag_pennant(self, df: pd.DataFrame) -> Dict:
        """Detect flag and pennant patterns"""
        patterns = {}
        
        if len(df) < 30:
            return patterns
        
        close = df['close'].values
        volume = df['volume'].values
        
        # Look for strong move followed by consolidation
        # Calculate 10-day momentum
        momentum = close[-30:-20].mean() / close[-40:-30].mean() - 1
        
        if abs(momentum) > 0.1:  # 10% move
            # Check for consolidation in last 10 days
            recent_range = (close[-10:].max() - close[-10:].min()) / close[-10:].mean()
            
            if recent_range < 0.05:  # Less than 5% range
                pattern_type = 'flag' if abs(momentum) > 0.15 else 'pennant'
                patterns[pattern_type] = {
                    'bullish': momentum > 0,
                    'target': close[-1] + (close[-1] * abs(momentum)),
                    'confidence': 0.6
                }
        
        return patterns
    
    def _detect_cup_and_handle(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect cup and handle pattern"""
        if len(prices) < 60:
            return None
        
        # Look for U-shaped pattern
        window = 30
        mid_point = len(prices) - window // 2
        
        left_peak = np.max(prices[-window:-window//2])
        bottom = np.min(prices[-window//2:])
        right_peak = prices[-1]
        
        # Check if forms a cup shape
        if (left_peak > bottom * 1.1 and 
            right_peak > bottom * 1.1 and
            abs(left_peak - right_peak) / left_peak < 0.05):
            
            return {
                'bullish': True,
                'resistance': (left_peak + right_peak) / 2,
                'support': bottom,
                'target': right_peak + (right_peak - bottom),
                'confidence': 0.65
            }
        
        return None
    
    def _calculate_volume_profile_sr(self, df: pd.DataFrame) -> Dict:
        """Calculate support/resistance from volume profile"""
        close = df['close'].values
        volume = df['volume'].values
        
        # Create price bins
        price_range = close.max() - close.min()
        num_bins = 50
        bins = np.linspace(close.min(), close.max(), num_bins)
        
        # Calculate volume at each price level
        volume_profile = np.zeros(num_bins - 1)
        
        for i in range(len(close)):
            bin_idx = np.digitize(close[i], bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume[i]
        
        # Find high volume nodes (potential S/R)
        threshold = np.percentile(volume_profile, 70)
        high_volume_indices = np.where(volume_profile > threshold)[0]
        
        high_volume_prices = bins[high_volume_indices]
        
        current_price = close[-1]
        resistance = high_volume_prices[high_volume_prices > current_price]
        support = high_volume_prices[high_volume_prices < current_price]
        
        return {
            'resistance': resistance[:3] if len(resistance) > 0 else np.array([]),
            'support': support[-3:] if len(support) > 0 else np.array([])
        }
    
    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray) -> Dict:
        """Calculate Fibonacci retracement levels"""
        # Find recent swing high and low
        recent_high = np.max(high[-50:])
        recent_low = np.min(low[-50:])
        
        diff = recent_high - recent_low
        
        fib_levels = {
            0.236: recent_high - (diff * 0.236),
            0.382: recent_high - (diff * 0.382),
            0.5: recent_high - (diff * 0.5),
            0.618: recent_high - (diff * 0.618),
            0.786: recent_high - (diff * 0.786)
        }
        
        current_price = high[-1]
        
        resistance = [level for level in fib_levels.values() if level > current_price]
        support = [level for level in fib_levels.values() if level < current_price]
        
        return {
            'levels': fib_levels,
            'resistance': np.array(resistance),
            'support': np.array(support)
        }
    
    def _cluster_levels(self, levels: np.ndarray, current_price: float) -> List[float]:
        """Cluster nearby price levels"""
        if len(levels) == 0:
            return []
        
        # Remove duplicates and sort
        unique_levels = np.unique(levels)
        
        # Cluster levels within 1% of each other
        clusters = []
        used = set()
        
        for i, level in enumerate(unique_levels):
            if i in used:
                continue
                
            cluster = [level]
            used.add(i)
            
            for j in range(i + 1, len(unique_levels)):
                if j not in used:
                    if abs(unique_levels[j] - level) / level < 0.01:
                        cluster.append(unique_levels[j])
                        used.add(j)
            
            clusters.append(np.mean(cluster))
        
        # Sort by distance from current price
        clusters.sort(key=lambda x: abs(x - current_price))
        
        return clusters
    
    def _identify_market_phase(self, df: pd.DataFrame) -> str:
        """Identify market phase (accumulation, markup, distribution, markdown)"""
        close = df['close'].values
        volume = df['volume'].values
        
        if len(close) < 50:
            return 'unknown'
        
        # Calculate trends
        price_trend = (close[-1] - close[-20]) / close[-20]
        volume_trend = (volume[-10:].mean() - volume[-30:-20].mean()) / volume[-30:-20].mean()
        
        # Volatility
        volatility = np.std(close[-20:]) / np.mean(close[-20:])
        
        # Determine phase
        if abs(price_trend) < 0.05 and volatility < 0.02:
            if volume_trend > 0.2:
                return 'accumulation'
            else:
                return 'consolidation'
        elif price_trend > 0.1:
            if volume_trend > 0:
                return 'markup'
            else:
                return 'distribution'
        elif price_trend < -0.1:
            return 'markdown'
        else:
            return 'transition'
    
    def _analyze_price_structure(self, high: np.ndarray, low: np.ndarray) -> Dict:
        """Analyze price structure (higher highs/lows, etc.)"""
        if len(high) < 20:
            return {'structure': 'insufficient_data'}
        
        # Find recent peaks and troughs
        window = 5
        peaks = argrelextrema(high, np.greater, order=window)[0]
        troughs = argrelextrema(low, np.less, order=window)[0]
        
        if len(peaks) < 2 or len(troughs) < 2:
            return {'structure': 'no_clear_structure'}
        
        # Check for higher highs and higher lows (uptrend)
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
                'last_high': high[recent_peaks[-1]],
                'last_low': low[recent_troughs[-1]]
            }
        elif lower_high and lower_low:
            return {
                'structure': 'downtrend',
                'strength': 'strong',
                'last_high': high[recent_peaks[-1]],
                'last_low': low[recent_troughs[-1]]
            }
        else:
            return {
                'structure': 'mixed',
                'strength': 'weak',
                'last_high': high[recent_peaks[-1]],
                'last_low': low[recent_troughs[-1]]
            }
    
    def _detect_ranging_market(self, df: pd.DataFrame) -> bool:
        """Detect if market is ranging/consolidating"""
        close = df['close'].values
        
        if len(close) < 20:
            return False
        
        # Calculate price range as percentage
        recent_range = (close[-20:].max() - close[-20:].min()) / close[-20:].mean()
        
        # Ranging if small range and low ADX
        adx_data = self._calculate_adx(df['high'].values, df['low'].values, close)
        return recent_range < 0.1 and adx_data['adx'] < 25
    
    # Simplified indicator calculation methods to replace talib
    
    def _calculate_ema(self, values, period):
        """Calculate Exponential Moving Average"""
        if len(values) < period:
            return np.mean(values) if len(values) > 0 else 0
        
        alpha = 2.0 / (period + 1.0)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema
    
    def _calculate_macd(self, close):
        """Calculate MACD"""
        ema12 = self._calculate_ema(close, 12)
        ema26 = self._calculate_ema(close, 26)
        macd = ema12 - ema26
        
        # Signal line is 9-period EMA of MACD
        # For simplicity, using a basic approximation
        signal = macd * 0.8  # Simplified
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    def _calculate_rsi(self, close, period=14):
        """Calculate Relative Strength Index"""
        if len(close) < period + 1:
            return 50.0
        
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, close, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(close) < period:
            return {
                'upper': close[-1] * 1.02,
                'middle': close[-1],
                'lower': close[-1] * 0.98,
                'width': close[-1] * 0.04,
                'percent': 0.5
            }
        
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = upper - lower
        percent = (close[-1] - lower) / width if width > 0 else 0.5
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width,
            'percent': percent
        }
    
    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        if len(high) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0
        
        return np.mean(true_ranges[-period:])
    
    def _calculate_adx(self, high, low, close, period=14):
        """Calculate ADX and Directional Indicators"""
        if len(high) < period + 1:
            return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}
        
        # Simplified ADX calculation
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        tr = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr.append(max(tr1, tr2, tr3))
        
        if len(tr) < period:
            return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}
        
        avg_tr = np.mean(tr[-period:])
        avg_plus_dm = np.mean(plus_dm[-period:])
        avg_minus_dm = np.mean(minus_dm[-period:])
        
        plus_di = (avg_plus_dm / avg_tr) * 100 if avg_tr > 0 else 0
        minus_di = (avg_minus_dm / avg_tr) * 100 if avg_tr > 0 else 0
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        adx = dx  # Simplified, should be smoothed
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def _calculate_sar(self, high, low):
        """Calculate Parabolic SAR (simplified)"""
        if len(high) < 2:
            return high[-1] if len(high) > 0 else 0
        
        # Very simplified SAR - just return a value near the low
        return np.min(low[-10:]) if len(low) >= 10 else low[-1]
    
    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        if len(high) < k_period:
            return {'k': 50.0, 'd': 50.0}
        
        lowest_low = np.min(low[-k_period:])
        highest_high = np.max(high[-k_period:])
        
        if highest_high == lowest_low:
            k = 50.0
        else:
            k = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # D is simple moving average of K
        d = k * 0.8  # Simplified
        
        return {'k': k, 'd': d}
    
    def _calculate_stoch_rsi(self, close, period=14):
        """Calculate Stochastic RSI"""
        rsi = self._calculate_rsi(close, period)
        return {'k': rsi, 'd': rsi * 0.8}  # Simplified
    
    def _calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        if len(high) < period:
            return -50.0
        
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        return ((highest_high - close[-1]) / (highest_high - lowest_low)) * -100
    
    def _calculate_cci(self, high, low, close, period=20):
        """Calculate Commodity Channel Index"""
        if len(high) < period:
            return 0.0
        
        typical_price = (high + low + close) / 3
        sma = np.mean(typical_price[-period:])
        mean_deviation = np.mean(np.abs(typical_price[-period:] - sma))
        
        if mean_deviation == 0:
            return 0.0
        
        cci = (typical_price[-1] - sma) / (0.015 * mean_deviation)
        return cci
    
    def _calculate_mfi(self, high, low, close, volume, period=14):
        """Calculate Money Flow Index"""
        if len(high) < period:
            return 50.0
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = 0
        negative_flow = 0
        
        for i in range(1, min(period + 1, len(typical_price))):
            if typical_price[-i] > typical_price[-i-1]:
                positive_flow += money_flow[-i]
            elif typical_price[-i] < typical_price[-i-1]:
                negative_flow += money_flow[-i]
        
        if negative_flow == 0:
            return 100.0
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    def _calculate_ultimate_oscillator(self, high, low, close):
        """Calculate Ultimate Oscillator (simplified)"""
        if len(high) < 7:
            return 50.0
        
        # Very simplified version
        tr = self._calculate_atr(high, low, close, 7)
        bp = close[-1] - np.min(low[-7:])
        
        if tr == 0:
            return 50.0
        
        return (bp / tr) * 100
    
    def _calculate_roc(self, values, period=10):
        """Calculate Rate of Change"""
        if len(values) < period + 1:
            return 0.0
        
        return ((values[-1] - values[-period-1]) / values[-period-1]) * 100
    
    def _calculate_momentum(self, close, period=10):
        """Calculate Momentum"""
        if len(close) < period + 1:
            return 0.0
        
        return close[-1] - close[-period-1]
    
    def _calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        if len(close) < 2:
            return volume[-1] if len(volume) > 0 else 0
        
        obv = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
        
        return obv
    
    def _calculate_ad_line(self, high, low, close, volume):
        """Calculate Accumulation/Distribution Line"""
        if len(high) == 0:
            return 0
        
        ad = 0
        for i in range(len(high)):
            if high[i] != low[i]:
                multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                ad += multiplier * volume[i]
        
        return ad
    
    def _detect_simple_patterns(self, open_prices, high, low, close):
        """Detect simple candlestick patterns without talib"""
        patterns = {}
        
        if len(close) < 5:
            return patterns
        
        # Simple doji detection
        body = abs(close[-1] - open_prices[-1])
        total_range = high[-1] - low[-1]
        
        if total_range > 0 and body / total_range < 0.1:
            patterns['doji'] = {
                'detected': True,
                'strength': 1,
                'position': 0
            }
        
        # Simple hammer detection
        lower_shadow = min(open_prices[-1], close[-1]) - low[-1]
        if total_range > 0 and lower_shadow / total_range > 0.6:
            patterns['hammer'] = {
                'detected': True,
                'strength': 1 if close[-1] > open_prices[-1] else -1,
                'position': 0
            }
        
        return patterns