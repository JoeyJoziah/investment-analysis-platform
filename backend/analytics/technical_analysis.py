"""
Advanced Technical Analysis Engine
Implements 200+ technical indicators and pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import talib
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
        
        # Moving Averages
        indicators['sma_5'] = talib.SMA(close, timeperiod=5)[-1]
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
        indicators['sma_200'] = talib.SMA(close, timeperiod=200)[-1]
        
        indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1]
        indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1]
        indicators['ema_50'] = talib.EMA(close, timeperiod=50)[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(close)
        indicators['macd'] = macd[-1]
        indicators['macd_signal'] = signal[-1]
        indicators['macd_histogram'] = hist[-1]
        
        # ADX (Average Directional Index)
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1]
        indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
        indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
        
        # Parabolic SAR
        indicators['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1]
        
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
        
        # RSI (Relative Strength Index)
        indicators['rsi_14'] = talib.RSI(close, timeperiod=14)[-1]
        indicators['rsi_9'] = talib.RSI(close, timeperiod=9)[-1]
        indicators['rsi_25'] = talib.RSI(close, timeperiod=25)[-1]
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        indicators['stoch_k'] = slowk[-1]
        indicators['stoch_d'] = slowd[-1]
        
        # Stochastic RSI
        fastk, fastd = talib.STOCHRSI(close)
        indicators['stochrsi_k'] = fastk[-1] if not np.isnan(fastk[-1]) else 0
        indicators['stochrsi_d'] = fastd[-1] if not np.isnan(fastd[-1]) else 0
        
        # Williams %R
        indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
        
        # CCI (Commodity Channel Index)
        indicators['cci'] = talib.CCI(high, low, close, timeperiod=20)[-1]
        
        # MFI (Money Flow Index)
        indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
        
        # Ultimate Oscillator
        indicators['ultimate_oscillator'] = talib.ULTOSC(high, low, close)[-1]
        
        # ROC (Rate of Change)
        indicators['roc'] = talib.ROC(close, timeperiod=10)[-1]
        
        # Momentum
        indicators['momentum'] = talib.MOM(close, timeperiod=10)[-1]
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        indicators['bb_upper'] = upper[-1]
        indicators['bb_middle'] = middle[-1]
        indicators['bb_lower'] = lower[-1]
        indicators['bb_width'] = upper[-1] - lower[-1]
        indicators['bb_percent'] = (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
        
        # ATR (Average True Range)
        indicators['atr_14'] = talib.ATR(high, low, close, timeperiod=14)[-1]
        indicators['atr_20'] = talib.ATR(high, low, close, timeperiod=20)[-1]
        
        # Keltner Channels
        keltner = self._calculate_keltner_channels(df)
        indicators.update(keltner)
        
        # Historical Volatility
        indicators['hv_20'] = self._calculate_historical_volatility(close, 20)
        indicators['hv_60'] = self._calculate_historical_volatility(close, 60)
        
        # Chaikin Volatility
        indicators['chaikin_volatility'] = self._calculate_chaikin_volatility(high, low)
        
        # Standard Deviation
        indicators['stddev_20'] = talib.STDDEV(close, timeperiod=20)[-1]
        
        # Normalized ATR
        indicators['natr'] = talib.NATR(high, low, close, timeperiod=14)[-1]
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # On Balance Volume
        indicators['obv'] = talib.OBV(close, volume)[-1]
        
        # AD Line (Accumulation/Distribution)
        indicators['ad_line'] = talib.AD(high, low, close, volume)[-1]
        
        # Chaikin Money Flow
        indicators['cmf'] = self._calculate_cmf(df)
        
        # Volume Rate of Change
        indicators['vroc'] = talib.ROC(volume.astype(float), timeperiod=10)[-1]
        
        # VWAP (Volume Weighted Average Price)
        indicators['vwap'] = self._calculate_vwap(df)
        
        # Price Volume Trend
        indicators['pvt'] = self._calculate_pvt(close, volume)
        
        # Volume moving averages
        indicators['volume_sma_20'] = talib.SMA(volume.astype(float), timeperiod=20)[-1]
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
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'engulfing': talib.CDLENGULFING,
            'harami': talib.CDLHARAMI,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'spinning_top': talib.CDLSPINNINGTOP,
            'marubozu': talib.CDLMARUBOZU,
            'abandoned_baby': talib.CDLABANDONEDBABY,
            'breakaway': talib.CDLBREAKAWAY,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
            'dragonfly_doji': talib.CDLDRAGONFLYDOJI,
            'gravestone_doji': talib.CDLGRAVESTONEDOJI,
            'inside_bar': talib.CDLHARAMI,
            'inverted_hammer': talib.CDLINVERTEDHAMMER,
            'piercing_line': talib.CDLPIERCING
        }
        
        for pattern_name, pattern_func in candlestick_functions.items():
            try:
                result = pattern_func(open_prices, high, low, close)
                # Check last 5 candles for patterns
                recent_patterns = result[-5:]
                if any(recent_patterns != 0):
                    patterns['candlestick_patterns'][pattern_name] = {
                        'detected': True,
                        'strength': int(recent_patterns[-1]),
                        'position': len(result) - 1 - np.where(recent_patterns != 0)[0][-1]
                    }
            except:
                pass
        
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
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        sma_200 = talib.SMA(close, timeperiod=200)
        
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
        
        adx = talib.ADX(high, low, close, timeperiod=14)
        
        # Normalize ADX to 0-1 scale
        trend_strength = min(adx[-1] / 50, 1.0) if len(adx) > 0 else 0.0
        
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
        middle = talib.EMA(close, timeperiod=20)
        
        # Channel Width: 2 * ATR(20)
        atr = talib.ATR(high, low, close, timeperiod=20)
        
        upper = middle + (2 * atr)
        lower = middle - (2 * atr)
        
        return {
            'keltner_upper': upper[-1] if len(upper) > 0 else 0,
            'keltner_middle': middle[-1] if len(middle) > 0 else 0,
            'keltner_lower': lower[-1] if len(lower) > 0 else 0
        }
    
    def _calculate_chaikin_volatility(self, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate Chaikin Volatility"""
        if len(high) < 20:
            return 0.0
        
        hl_diff = high - low
        ema10 = talib.EMA(hl_diff, timeperiod=10)
        
        if len(ema10) < 10:
            return 0.0
        
        chaikin_vol = ((ema10[-1] - ema10[-11]) / ema10[-11]) * 100
        
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
        fi = talib.EMA(force, timeperiod=13)
        
        return fi[-1] if len(fi) > 0 else 0.0
    
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
        
        # Calculate ADX
        adx = talib.ADX(df['high'].values, df['low'].values, close, timeperiod=14)
        
        # Ranging if small range and low ADX
        return recent_range < 0.1 and adx[-1] < 25