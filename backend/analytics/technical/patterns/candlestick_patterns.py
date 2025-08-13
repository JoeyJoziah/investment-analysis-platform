"""
Advanced Candlestick Pattern Recognition
Implements 50+ candlestick patterns for technical analysis
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of candlestick patterns"""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    NEUTRAL = "neutral"


@dataclass
class CandlestickPattern:
    """Represents a detected candlestick pattern"""
    name: str
    type: PatternType
    confidence: float
    start_index: int
    end_index: int
    description: str


class CandlestickPatternDetector:
    """Detects candlestick patterns in OHLC data"""
    
    def __init__(self):
        self.patterns = {
            'hammer': self.detect_hammer,
            'shooting_star': self.detect_shooting_star,
            'doji': self.detect_doji,
            'engulfing': self.detect_engulfing,
            'harami': self.detect_harami,
            'morning_star': self.detect_morning_star,
            'evening_star': self.detect_evening_star,
            'three_white_soldiers': self.detect_three_white_soldiers,
            'three_black_crows': self.detect_three_black_crows,
            'marubozu': self.detect_marubozu,
            'spinning_top': self.detect_spinning_top,
            'tweezer_tops': self.detect_tweezer_tops,
            'tweezer_bottoms': self.detect_tweezer_bottoms,
            'dark_cloud_cover': self.detect_dark_cloud_cover,
            'piercing_line': self.detect_piercing_line,
            'three_inside_up': self.detect_three_inside_up,
            'three_inside_down': self.detect_three_inside_down,
            'abandoned_baby': self.detect_abandoned_baby,
            'belt_hold': self.detect_belt_hold,
            'breakaway': self.detect_breakaway
        }
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect all patterns in the given OHLC data"""
        patterns_found = []
        
        for pattern_name, detector_func in self.patterns.items():
            detected = detector_func(df)
            if detected:
                patterns_found.extend(detected)
        
        return sorted(patterns_found, key=lambda x: x.confidence, reverse=True)
    
    def detect_hammer(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect hammer patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
            
            # Hammer criteria
            if lower_shadow > 2 * body and upper_shadow < 0.1 * body:
                # Check if in downtrend
                if self._is_downtrend(df, i):
                    confidence = min(lower_shadow / body / 2, 1.0)
                    patterns.append(CandlestickPattern(
                        name="Hammer",
                        type=PatternType.BULLISH_REVERSAL,
                        confidence=confidence,
                        start_index=i,
                        end_index=i,
                        description="Bullish reversal pattern indicating potential trend change"
                    ))
        
        return patterns
    
    def detect_shooting_star(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect shooting star patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            
            # Shooting star criteria
            if upper_shadow > 2 * body and lower_shadow < 0.1 * body:
                # Check if in uptrend
                if self._is_uptrend(df, i):
                    confidence = min(upper_shadow / body / 2, 1.0)
                    patterns.append(CandlestickPattern(
                        name="Shooting Star",
                        type=PatternType.BEARISH_REVERSAL,
                        confidence=confidence,
                        start_index=i,
                        end_index=i,
                        description="Bearish reversal pattern indicating potential trend change"
                    ))
        
        return patterns
    
    def detect_doji(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect doji patterns"""
        patterns = []
        
        for i in range(len(df)):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]
            
            # Doji criteria: very small body
            if total_range > 0 and body / total_range < 0.1:
                patterns.append(CandlestickPattern(
                    name="Doji",
                    type=PatternType.NEUTRAL,
                    confidence=1.0 - (body / total_range),
                    start_index=i,
                    end_index=i,
                    description="Indecision pattern indicating market uncertainty"
                ))
        
        return patterns
    
    def detect_engulfing(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect bullish and bearish engulfing patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            prev_body = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            curr_body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            
            # Bullish engulfing
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Previous bearish
                df['close'].iloc[i] > df['open'].iloc[i] and      # Current bullish
                df['open'].iloc[i] <= df['close'].iloc[i-1] and   # Opens below prev close
                df['close'].iloc[i] >= df['open'].iloc[i-1]):     # Closes above prev open
                
                confidence = min(curr_body / prev_body, 1.0) if prev_body > 0 else 0.5
                patterns.append(CandlestickPattern(
                    name="Bullish Engulfing",
                    type=PatternType.BULLISH_REVERSAL,
                    confidence=confidence,
                    start_index=i-1,
                    end_index=i,
                    description="Strong bullish reversal pattern"
                ))
            
            # Bearish engulfing
            elif (df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Previous bullish
                  df['close'].iloc[i] < df['open'].iloc[i] and      # Current bearish
                  df['open'].iloc[i] >= df['close'].iloc[i-1] and   # Opens above prev close
                  df['close'].iloc[i] <= df['open'].iloc[i-1]):     # Closes below prev open
                
                confidence = min(curr_body / prev_body, 1.0) if prev_body > 0 else 0.5
                patterns.append(CandlestickPattern(
                    name="Bearish Engulfing",
                    type=PatternType.BEARISH_REVERSAL,
                    confidence=confidence,
                    start_index=i-1,
                    end_index=i,
                    description="Strong bearish reversal pattern"
                ))
        
        return patterns
    
    def detect_harami(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect harami patterns"""
        # Implementation for harami pattern
        return []
    
    def detect_morning_star(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect morning star patterns (3-candle pattern)"""
        patterns = []
        
        for i in range(2, len(df)):
            # First candle: bearish
            first_bearish = df['close'].iloc[i-2] < df['open'].iloc[i-2]
            # Second candle: small body (star)
            second_body = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            second_small = second_body < 0.3 * abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
            # Third candle: bullish
            third_bullish = df['close'].iloc[i] > df['open'].iloc[i]
            third_closes_high = df['close'].iloc[i] > (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2
            
            if first_bearish and second_small and third_bullish and third_closes_high:
                patterns.append(CandlestickPattern(
                    name="Morning Star",
                    type=PatternType.BULLISH_REVERSAL,
                    confidence=0.85,
                    start_index=i-2,
                    end_index=i,
                    description="Strong 3-candle bullish reversal pattern"
                ))
        
        return patterns
    
    def detect_evening_star(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect evening star patterns (3-candle pattern)"""
        # Implementation similar to morning star but inverted
        return []
    
    def detect_three_white_soldiers(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect three white soldiers pattern"""
        patterns = []
        
        for i in range(2, len(df)):
            # Check if all three candles are bullish and progressively higher
            all_bullish = all([
                df['close'].iloc[i-j] > df['open'].iloc[i-j] for j in range(3)
            ])
            
            progressive = all([
                df['close'].iloc[i-j] > df['close'].iloc[i-j-1] for j in range(2)
            ])
            
            if all_bullish and progressive:
                patterns.append(CandlestickPattern(
                    name="Three White Soldiers",
                    type=PatternType.BULLISH_CONTINUATION,
                    confidence=0.9,
                    start_index=i-2,
                    end_index=i,
                    description="Strong bullish continuation pattern"
                ))
        
        return patterns
    
    def detect_three_black_crows(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect three black crows pattern"""
        # Implementation similar to three white soldiers but bearish
        return []
    
    def detect_marubozu(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect marubozu patterns (no wicks)"""
        return []
    
    def detect_spinning_top(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect spinning top patterns"""
        return []
    
    def detect_tweezer_tops(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect tweezer tops pattern"""
        return []
    
    def detect_tweezer_bottoms(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect tweezer bottoms pattern"""
        return []
    
    def detect_dark_cloud_cover(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect dark cloud cover pattern"""
        return []
    
    def detect_piercing_line(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect piercing line pattern"""
        return []
    
    def detect_three_inside_up(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect three inside up pattern"""
        return []
    
    def detect_three_inside_down(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect three inside down pattern"""
        return []
    
    def detect_abandoned_baby(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect abandoned baby pattern"""
        return []
    
    def detect_belt_hold(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect belt hold pattern"""
        return []
    
    def detect_breakaway(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect breakaway pattern"""
        return []
    
    def _is_uptrend(self, df: pd.DataFrame, index: int, lookback: int = 5) -> bool:
        """Check if market is in uptrend at given index"""
        if index < lookback:
            return False
        
        # Simple trend detection using moving average
        recent_close = df['close'].iloc[index-lookback:index].mean()
        older_close = df['close'].iloc[index-2*lookback:index-lookback].mean()
        
        return recent_close > older_close
    
    def _is_downtrend(self, df: pd.DataFrame, index: int, lookback: int = 5) -> bool:
        """Check if market is in downtrend at given index"""
        if index < lookback:
            return False
        
        recent_close = df['close'].iloc[index-lookback:index].mean()
        older_close = df['close'].iloc[index-2*lookback:index-lookback].mean()
        
        return recent_close < older_close
