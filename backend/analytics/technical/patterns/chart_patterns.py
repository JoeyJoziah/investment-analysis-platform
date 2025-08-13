"""
Chart Pattern Recognition
Identifies classical chart patterns like Head & Shoulders, Triangles, etc.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.linear_model import LinearRegression


class ChartPatternType(Enum):
    """Types of chart patterns"""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG = "flag"
    PENNANT = "pennant"
    CUP_AND_HANDLE = "cup_and_handle"
    ROUNDING_BOTTOM = "rounding_bottom"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    RECTANGLE = "rectangle"


@dataclass
class ChartPattern:
    """Represents a detected chart pattern"""
    pattern_type: ChartPatternType
    start_index: int
    end_index: int
    confidence: float
    breakout_level: float
    target_price: float
    stop_loss: float
    pattern_points: List[Tuple[int, float]]  # Key points that form the pattern
    description: str


class ChartPatternDetector:
    """Detects classical chart patterns in price data"""
    
    def __init__(self, min_pattern_bars: int = 10):
        self.min_pattern_bars = min_pattern_bars
        self.pattern_detectors = {
            ChartPatternType.HEAD_AND_SHOULDERS: self._detect_head_and_shoulders,
            ChartPatternType.DOUBLE_TOP: self._detect_double_top,
            ChartPatternType.DOUBLE_BOTTOM: self._detect_double_bottom,
            ChartPatternType.ASCENDING_TRIANGLE: self._detect_ascending_triangle,
            ChartPatternType.DESCENDING_TRIANGLE: self._detect_descending_triangle,
            ChartPatternType.CUP_AND_HANDLE: self._detect_cup_and_handle,
            ChartPatternType.FLAG: self._detect_flag,
            ChartPatternType.CHANNEL_UP: self._detect_channel_up,
            ChartPatternType.CHANNEL_DOWN: self._detect_channel_down
        }
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect all chart patterns in the given price data"""
        patterns = []
        
        for pattern_type, detector_func in self.pattern_detectors.items():
            detected = detector_func(df)
            if detected:
                patterns.extend(detected)
        
        return sorted(patterns, key=lambda x: x.confidence, reverse=True)
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Head and Shoulders pattern"""
        patterns = []
        window = 30  # Look for patterns in 30-bar windows
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            highs = segment['high'].values
            
            # Find peaks
            peaks = self._find_peaks(highs)
            
            if len(peaks) >= 3:
                # Check for H&S pattern: 3 peaks with middle one highest
                for j in range(len(peaks) - 2):
                    left_shoulder = peaks[j]
                    head = peaks[j+1]
                    right_shoulder = peaks[j+2]
                    
                    # Head should be higher than shoulders
                    if (highs[head] > highs[left_shoulder] and 
                        highs[head] > highs[right_shoulder]):
                        
                        # Shoulders should be roughly equal
                        shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder])
                        avg_shoulder = (highs[left_shoulder] + highs[right_shoulder]) / 2
                        
                        if shoulder_diff / avg_shoulder < 0.05:  # Within 5%
                            # Find neckline
                            neckline = self._find_neckline(segment, left_shoulder, head, right_shoulder)
                            
                            # Calculate pattern metrics
                            pattern_height = highs[head] - neckline
                            target = neckline - pattern_height
                            
                            pattern = ChartPattern(
                                pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
                                start_index=i + left_shoulder,
                                end_index=i + right_shoulder,
                                confidence=0.75,
                                breakout_level=neckline,
                                target_price=target,
                                stop_loss=highs[head],
                                pattern_points=[
                                    (i + left_shoulder, highs[left_shoulder]),
                                    (i + head, highs[head]),
                                    (i + right_shoulder, highs[right_shoulder])
                                ],
                                description="Bearish reversal pattern"
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_double_top(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Double Top pattern"""
        patterns = []
        window = 20
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            highs = segment['high'].values
            
            # Find peaks
            peaks = self._find_peaks(highs)
            
            if len(peaks) >= 2:
                for j in range(len(peaks) - 1):
                    first_top = peaks[j]
                    second_top = peaks[j+1]
                    
                    # Tops should be roughly equal
                    top_diff = abs(highs[first_top] - highs[second_top])
                    avg_top = (highs[first_top] + highs[second_top]) / 2
                    
                    if top_diff / avg_top < 0.03:  # Within 3%
                        # Find valley between tops
                        valley_idx = first_top + np.argmin(highs[first_top:second_top])
                        valley_price = highs[valley_idx]
                        
                        # Calculate target
                        pattern_height = avg_top - valley_price
                        target = valley_price - pattern_height
                        
                        pattern = ChartPattern(
                            pattern_type=ChartPatternType.DOUBLE_TOP,
                            start_index=i + first_top,
                            end_index=i + second_top,
                            confidence=0.7,
                            breakout_level=valley_price,
                            target_price=target,
                            stop_loss=avg_top * 1.02,
                            pattern_points=[
                                (i + first_top, highs[first_top]),
                                (i + valley_idx, valley_price),
                                (i + second_top, highs[second_top])
                            ],
                            description="Bearish reversal pattern"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Double Bottom pattern"""
        patterns = []
        window = 20
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            lows = segment['low'].values
            
            # Find troughs
            troughs = self._find_troughs(lows)
            
            if len(troughs) >= 2:
                for j in range(len(troughs) - 1):
                    first_bottom = troughs[j]
                    second_bottom = troughs[j+1]
                    
                    # Bottoms should be roughly equal
                    bottom_diff = abs(lows[first_bottom] - lows[second_bottom])
                    avg_bottom = (lows[first_bottom] + lows[second_bottom]) / 2
                    
                    if bottom_diff / avg_bottom < 0.03:  # Within 3%
                        # Find peak between bottoms
                        peak_idx = first_bottom + np.argmax(lows[first_bottom:second_bottom])
                        peak_price = lows[peak_idx]
                        
                        # Calculate target
                        pattern_height = peak_price - avg_bottom
                        target = peak_price + pattern_height
                        
                        pattern = ChartPattern(
                            pattern_type=ChartPatternType.DOUBLE_BOTTOM,
                            start_index=i + first_bottom,
                            end_index=i + second_bottom,
                            confidence=0.7,
                            breakout_level=peak_price,
                            target_price=target,
                            stop_loss=avg_bottom * 0.98,
                            pattern_points=[
                                (i + first_bottom, lows[first_bottom]),
                                (i + peak_idx, peak_price),
                                (i + second_bottom, lows[second_bottom])
                            ],
                            description="Bullish reversal pattern"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Ascending Triangle pattern"""
        patterns = []
        window = 20
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            highs = segment['high'].values
            lows = segment['low'].values
            
            # Check for flat resistance and rising support
            high_slope = self._calculate_trendline_slope(highs)
            low_slope = self._calculate_trendline_slope(lows)
            
            # Ascending triangle: flat top, rising bottom
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                resistance = np.mean(highs[-5:])  # Average of recent highs
                
                pattern = ChartPattern(
                    pattern_type=ChartPatternType.ASCENDING_TRIANGLE,
                    start_index=i,
                    end_index=i + window - 1,
                    confidence=0.65,
                    breakout_level=resistance,
                    target_price=resistance + (resistance - lows[0]),
                    stop_loss=lows[-1],
                    pattern_points=[
                        (i, highs[0]),
                        (i + window - 1, highs[-1]),
                        (i, lows[0]),
                        (i + window - 1, lows[-1])
                    ],
                    description="Bullish continuation pattern"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_descending_triangle(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Descending Triangle pattern"""
        patterns = []
        window = 20
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            highs = segment['high'].values
            lows = segment['low'].values
            
            # Check for falling resistance and flat support
            high_slope = self._calculate_trendline_slope(highs)
            low_slope = self._calculate_trendline_slope(lows)
            
            # Descending triangle: falling top, flat bottom
            if high_slope < -0.001 and abs(low_slope) < 0.001:
                support = np.mean(lows[-5:])  # Average of recent lows
                
                pattern = ChartPattern(
                    pattern_type=ChartPatternType.DESCENDING_TRIANGLE,
                    start_index=i,
                    end_index=i + window - 1,
                    confidence=0.65,
                    breakout_level=support,
                    target_price=support - (highs[0] - support),
                    stop_loss=highs[-1],
                    pattern_points=[
                        (i, highs[0]),
                        (i + window - 1, highs[-1]),
                        (i, lows[0]),
                        (i + window - 1, lows[-1])
                    ],
                    description="Bearish continuation pattern"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cup_and_handle(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Cup and Handle pattern"""
        patterns = []
        window = 40  # Cup and handle needs more bars
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            lows = segment['low'].values
            
            # Look for U-shaped bottom (cup)
            cup_bottom = np.argmin(lows[:30])
            
            # Check if it forms a cup shape
            left_rim = lows[0]
            right_rim = lows[30]
            bottom = lows[cup_bottom]
            
            # Cup criteria
            if (abs(left_rim - right_rim) / left_rim < 0.1 and  # Rims are similar
                bottom < left_rim * 0.85):  # Bottom is at least 15% below rim
                
                # Look for handle (small dip after cup)
                handle_low = np.min(lows[30:])
                
                if handle_low > bottom * 1.05:  # Handle doesn't go below cup bottom
                    pattern = ChartPattern(
                        pattern_type=ChartPatternType.CUP_AND_HANDLE,
                        start_index=i,
                        end_index=i + window - 1,
                        confidence=0.6,
                        breakout_level=right_rim,
                        target_price=right_rim + (right_rim - bottom),
                        stop_loss=handle_low,
                        pattern_points=[
                            (i, left_rim),
                            (i + cup_bottom, bottom),
                            (i + 30, right_rim),
                            (i + 35, handle_low)
                        ],
                        description="Bullish continuation pattern"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_flag(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Flag pattern"""
        patterns = []
        window = 15
        
        if len(df) < window + 10:
            return patterns
        
        for i in range(10, len(df) - window):
            # Check for strong move before flag (pole)
            pole_start = i - 10
            pole_end = i
            pole_move = df['close'].iloc[pole_end] - df['close'].iloc[pole_start]
            
            if abs(pole_move) / df['close'].iloc[pole_start] > 0.05:  # At least 5% move
                # Check for consolidation (flag)
                segment = df.iloc[i:i+window]
                highs = segment['high'].values
                lows = segment['low'].values
                
                high_slope = self._calculate_trendline_slope(highs)
                low_slope = self._calculate_trendline_slope(lows)
                
                # Flag should have parallel lines
                if abs(high_slope - low_slope) < 0.001:
                    # Flag should be counter to pole direction
                    if (pole_move > 0 and high_slope < 0) or (pole_move < 0 and high_slope > 0):
                        pattern = ChartPattern(
                            pattern_type=ChartPatternType.FLAG,
                            start_index=pole_start,
                            end_index=i + window - 1,
                            confidence=0.7,
                            breakout_level=segment['high'].iloc[-1] if pole_move > 0 else segment['low'].iloc[-1],
                            target_price=segment['close'].iloc[-1] + pole_move,
                            stop_loss=segment['low'].iloc[-1] if pole_move > 0 else segment['high'].iloc[-1],
                            pattern_points=[
                                (pole_start, df['close'].iloc[pole_start]),
                                (pole_end, df['close'].iloc[pole_end]),
                                (i + window - 1, segment['close'].iloc[-1])
                            ],
                            description="Continuation pattern"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_channel_up(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Upward Channel pattern"""
        patterns = []
        window = 20
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            highs = segment['high'].values
            lows = segment['low'].values
            
            # Calculate trendline slopes
            high_slope = self._calculate_trendline_slope(highs)
            low_slope = self._calculate_trendline_slope(lows)
            
            # Both should be positive and roughly parallel
            if high_slope > 0.001 and low_slope > 0.001:
                slope_diff = abs(high_slope - low_slope)
                avg_slope = (high_slope + low_slope) / 2
                
                if slope_diff / avg_slope < 0.2:  # Within 20% of each other
                    pattern = ChartPattern(
                        pattern_type=ChartPatternType.CHANNEL_UP,
                        start_index=i,
                        end_index=i + window - 1,
                        confidence=0.6,
                        breakout_level=self._project_trendline(highs, len(highs)),
                        target_price=self._project_trendline(highs, len(highs) + 5),
                        stop_loss=self._project_trendline(lows, len(lows)),
                        pattern_points=[
                            (i, highs[0]),
                            (i + window - 1, highs[-1]),
                            (i, lows[0]),
                            (i + window - 1, lows[-1])
                        ],
                        description="Bullish channel pattern"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_channel_down(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Downward Channel pattern"""
        patterns = []
        window = 20
        
        if len(df) < window:
            return patterns
        
        for i in range(len(df) - window):
            segment = df.iloc[i:i+window]
            highs = segment['high'].values
            lows = segment['low'].values
            
            # Calculate trendline slopes
            high_slope = self._calculate_trendline_slope(highs)
            low_slope = self._calculate_trendline_slope(lows)
            
            # Both should be negative and roughly parallel
            if high_slope < -0.001 and low_slope < -0.001:
                slope_diff = abs(high_slope - low_slope)
                avg_slope = abs((high_slope + low_slope) / 2)
                
                if slope_diff / avg_slope < 0.2:  # Within 20% of each other
                    pattern = ChartPattern(
                        pattern_type=ChartPatternType.CHANNEL_DOWN,
                        start_index=i,
                        end_index=i + window - 1,
                        confidence=0.6,
                        breakout_level=self._project_trendline(lows, len(lows)),
                        target_price=self._project_trendline(lows, len(lows) + 5),
                        stop_loss=self._project_trendline(highs, len(highs)),
                        pattern_points=[
                            (i, highs[0]),
                            (i + window - 1, highs[-1]),
                            (i, lows[0]),
                            (i + window - 1, lows[-1])
                        ],
                        description="Bearish channel pattern"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_peaks(self, data: np.array, min_distance: int = 3) -> List[int]:
        """Find peaks in data"""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] > data[i-j] for j in range(1, min_distance+1)) and \
               all(data[i] > data[i+j] for j in range(1, min_distance+1)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.array, min_distance: int = 3) -> List[int]:
        """Find troughs in data"""
        troughs = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] < data[i-j] for j in range(1, min_distance+1)) and \
               all(data[i] < data[i+j] for j in range(1, min_distance+1)):
                troughs.append(i)
        return troughs
    
    def _find_neckline(self, segment: pd.DataFrame, left: int, head: int, right: int) -> float:
        """Find neckline level for head and shoulders pattern"""
        # Find the lows between shoulders and head
        left_valley = segment['low'].iloc[left:head].min()
        right_valley = segment['low'].iloc[head:right].min()
        
        # Neckline is the average of the two valleys
        return (left_valley + right_valley) / 2
    
    def _calculate_trendline_slope(self, data: np.array) -> float:
        """Calculate the slope of a trendline through data points"""
        x = np.arange(len(data)).reshape(-1, 1)
        y = data.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model.coef_[0][0]
    
    def _project_trendline(self, data: np.array, steps: int) -> float:
        """Project trendline forward"""
        x = np.arange(len(data)).reshape(-1, 1)
        y = data.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        
        projection = model.predict([[steps]])[0][0]
        return projection
