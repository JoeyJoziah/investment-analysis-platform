"""
Harmonic Pattern Recognition Engine

Identifies and analyzes harmonic patterns:
- Gartley patterns (222 pattern)
- Butterfly patterns (B-point at 78.6% retracement)
- Bat patterns (B-point at 38.2% or 50% retracement)
- Crab patterns (B-point at 38.2% or 61.8% retracement)
- ABCD patterns
- Cypher patterns
- Shark patterns

Uses precise Fibonacci ratios and geometric analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class HarmonicType(Enum):
    GARTLEY_BULLISH = "gartley_bullish"
    GARTLEY_BEARISH = "gartley_bearish"
    BUTTERFLY_BULLISH = "butterfly_bullish"
    BUTTERFLY_BEARISH = "butterfly_bearish"
    BAT_BULLISH = "bat_bullish"
    BAT_BEARISH = "bat_bearish"
    CRAB_BULLISH = "crab_bullish"
    CRAB_BEARISH = "crab_bearish"
    ABCD_BULLISH = "abcd_bullish"
    ABCD_BEARISH = "abcd_bearish"
    CYPHER_BULLISH = "cypher_bullish"
    CYPHER_BEARISH = "cypher_bearish"
    SHARK_BULLISH = "shark_bullish"
    SHARK_BEARISH = "shark_bearish"

@dataclass
class HarmonicPoint:
    """Point in harmonic pattern"""
    price: float
    time: datetime
    index: int
    label: str  # X, A, B, C, D

@dataclass
class HarmonicPattern:
    """Detected harmonic pattern"""
    pattern_type: HarmonicType
    points: Dict[str, HarmonicPoint]  # X, A, B, C, D points
    fibonacci_ratios: Dict[str, float]  # Actual ratios found
    target_ratios: Dict[str, Tuple[float, float]]  # Expected ratio ranges
    accuracy_score: float  # How well it matches ideal ratios
    completion_level: str  # forming, completed, active
    price_targets: List[float]  # Projected targets
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    confidence: float
    formed_at: datetime

@dataclass
class PatternValidation:
    """Pattern validation result"""
    is_valid: bool
    accuracy_score: float
    violations: List[str]
    fibonacci_accuracy: Dict[str, float]

class HarmonicPatternAnalyzer:
    """
    Advanced harmonic pattern recognition and analysis
    
    Features:
    - Precise Fibonacci ratio validation
    - Multiple harmonic pattern types
    - Pattern completion prediction
    - Price target calculation
    - Risk management levels
    - Pattern strength scoring
    - Historical pattern performance
    """
    
    def __init__(self, tolerance: float = 0.05):
        """
        Initialize harmonic pattern analyzer
        
        Args:
            tolerance: Fibonacci ratio tolerance (default 5%)
        """
        self.tolerance = tolerance
        
        # Define ideal Fibonacci ratios for each pattern
        self.pattern_ratios = self._define_pattern_ratios()
    
    def _define_pattern_ratios(self) -> Dict[HarmonicType, Dict[str, Tuple[float, float]]]:
        """Define Fibonacci ratios for each harmonic pattern"""
        return {
            # Gartley Pattern (222)
            HarmonicType.GARTLEY_BULLISH: {
                'AB_XA': (0.618 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (1.13 - self.tolerance, 1.618 + self.tolerance),
                'AD_XA': (0.786 - self.tolerance, 0.786 + self.tolerance)
            },
            HarmonicType.GARTLEY_BEARISH: {
                'AB_XA': (0.618 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (1.13 - self.tolerance, 1.618 + self.tolerance),
                'AD_XA': (0.786 - self.tolerance, 0.786 + self.tolerance)
            },
            
            # Butterfly Pattern
            HarmonicType.BUTTERFLY_BULLISH: {
                'AB_XA': (0.786 - self.tolerance, 0.786 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (1.618 - self.tolerance, 2.618 + self.tolerance),
                'AD_XA': (1.27 - self.tolerance, 1.618 + self.tolerance)
            },
            HarmonicType.BUTTERFLY_BEARISH: {
                'AB_XA': (0.786 - self.tolerance, 0.786 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (1.618 - self.tolerance, 2.618 + self.tolerance),
                'AD_XA': (1.27 - self.tolerance, 1.618 + self.tolerance)
            },
            
            # Bat Pattern
            HarmonicType.BAT_BULLISH: {
                'AB_XA': (0.382 - self.tolerance, 0.5 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (1.618 - self.tolerance, 2.618 + self.tolerance),
                'AD_XA': (0.886 - self.tolerance, 0.886 + self.tolerance)
            },
            HarmonicType.BAT_BEARISH: {
                'AB_XA': (0.382 - self.tolerance, 0.5 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (1.618 - self.tolerance, 2.618 + self.tolerance),
                'AD_XA': (0.886 - self.tolerance, 0.886 + self.tolerance)
            },
            
            # Crab Pattern
            HarmonicType.CRAB_BULLISH: {
                'AB_XA': (0.382 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (2.24 - self.tolerance, 3.618 + self.tolerance),
                'AD_XA': (1.618 - self.tolerance, 1.618 + self.tolerance)
            },
            HarmonicType.CRAB_BEARISH: {
                'AB_XA': (0.382 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (0.382 - self.tolerance, 0.886 + self.tolerance),
                'CD_BC': (2.24 - self.tolerance, 3.618 + self.tolerance),
                'AD_XA': (1.618 - self.tolerance, 1.618 + self.tolerance)
            },
            
            # ABCD Pattern
            HarmonicType.ABCD_BULLISH: {
                'BC_AB': (0.618 - self.tolerance, 0.786 + self.tolerance),
                'CD_AB': (1.27 - self.tolerance, 1.618 + self.tolerance)
            },
            HarmonicType.ABCD_BEARISH: {
                'BC_AB': (0.618 - self.tolerance, 0.786 + self.tolerance),
                'CD_AB': (1.27 - self.tolerance, 1.618 + self.tolerance)
            },
            
            # Cypher Pattern
            HarmonicType.CYPHER_BULLISH: {
                'AB_XA': (0.382 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (1.13 - self.tolerance, 1.414 + self.tolerance),
                'CD_BC': (0.618 - self.tolerance, 0.786 + self.tolerance),
                'AD_XA': (0.786 - self.tolerance, 0.786 + self.tolerance)
            },
            HarmonicType.CYPHER_BEARISH: {
                'AB_XA': (0.382 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (1.13 - self.tolerance, 1.414 + self.tolerance),
                'CD_BC': (0.618 - self.tolerance, 0.786 + self.tolerance),
                'AD_XA': (0.786 - self.tolerance, 0.786 + self.tolerance)
            },
            
            # Shark Pattern
            HarmonicType.SHARK_BULLISH: {
                'AB_OX': (0.382 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (1.13 - self.tolerance, 1.618 + self.tolerance),
                'CD_BC': (1.618 - self.tolerance, 2.24 + self.tolerance),
                'AD_OX': (0.886 - self.tolerance, 1.13 + self.tolerance)
            },
            HarmonicType.SHARK_BEARISH: {
                'AB_OX': (0.382 - self.tolerance, 0.618 + self.tolerance),
                'BC_AB': (1.13 - self.tolerance, 1.618 + self.tolerance),
                'CD_BC': (1.618 - self.tolerance, 2.24 + self.tolerance),
                'AD_OX': (0.886 - self.tolerance, 1.13 + self.tolerance)
            }
        }
    
    def detect_harmonic_patterns(
        self,
        price_data: pd.DataFrame,
        min_pattern_bars: int = 10,
        lookback_periods: int = 100
    ) -> List[HarmonicPattern]:
        """
        Detect harmonic patterns in price data
        
        Args:
            price_data: DataFrame with OHLC data
            min_pattern_bars: Minimum bars between pattern points
            lookback_periods: Number of periods to analyze
            
        Returns:
            List of detected harmonic patterns
        """
        if len(price_data) < min_pattern_bars * 4:  # Need at least 4 points
            return []
        
        patterns = []
        
        try:
            # Use most recent data
            recent_data = price_data.tail(lookback_periods).copy()
            
            # Find significant pivot points
            pivot_points = self._find_pivot_points(recent_data, min_pattern_bars)
            
            if len(pivot_points) < 4:
                return patterns
            
            # Try to form patterns with different point combinations
            for i in range(len(pivot_points) - 4):
                for j in range(i + 1, len(pivot_points) - 3):
                    for k in range(j + 1, len(pivot_points) - 2):
                        for l in range(k + 1, len(pivot_points) - 1):
                            for m in range(l + 1, len(pivot_points)):
                                # Try to form 5-point patterns (X-A-B-C-D)
                                points = {
                                    'X': pivot_points[i],
                                    'A': pivot_points[j],
                                    'B': pivot_points[k],
                                    'C': pivot_points[l],
                                    'D': pivot_points[m]
                                }
                                
                                # Test for each pattern type
                                detected_patterns = self._test_pattern_formation(points)
                                patterns.extend(detected_patterns)
                                
                                # Also test 4-point ABCD patterns
                                if i >= 1:  # Need at least one point before A
                                    abcd_points = {
                                        'A': pivot_points[j],
                                        'B': pivot_points[k],
                                        'C': pivot_points[l],
                                        'D': pivot_points[m]
                                    }
                                    abcd_patterns = self._test_abcd_pattern(abcd_points)
                                    patterns.extend(abcd_patterns)
            
            # Filter and rank patterns
            patterns = self._filter_overlapping_patterns(patterns)
            patterns.sort(key=lambda x: x.accuracy_score, reverse=True)
            
            return patterns[:10]  # Return top 10 patterns
            
        except Exception as e:
            logger.error(f"Error detecting harmonic patterns: {e}")
            return []
    
    def _find_pivot_points(
        self,
        data: pd.DataFrame,
        min_distance: int = 5
    ) -> List[HarmonicPoint]:
        """Find significant pivot points (highs and lows)"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find local maxima and minima
        high_indices = argrelextrema(highs, np.greater, order=min_distance)[0]
        low_indices = argrelextrema(lows, np.less, order=min_distance)[0]
        
        pivot_points = []
        
        # Add high points
        for idx in high_indices:
            if idx < len(data):
                pivot_points.append(HarmonicPoint(
                    price=highs[idx],
                    time=data.index[idx],
                    index=idx,
                    label='H'
                ))
        
        # Add low points
        for idx in low_indices:
            if idx < len(data):
                pivot_points.append(HarmonicPoint(
                    price=lows[idx],
                    time=data.index[idx],
                    index=idx,
                    label='L'
                ))
        
        # Sort by time
        pivot_points.sort(key=lambda x: x.index)
        
        return pivot_points
    
    def _test_pattern_formation(self, points: Dict[str, HarmonicPoint]) -> List[HarmonicPattern]:
        """Test if points form valid harmonic patterns"""
        patterns = []
        
        # Ensure proper point order
        point_list = [points['X'], points['A'], points['B'], points['C'], points['D']]
        if not all(point_list[i].index < point_list[i+1].index for i in range(4)):
            return patterns
        
        # Test for each pattern type
        for pattern_type in HarmonicType:
            if 'ABCD' in pattern_type.value:
                continue  # Handle separately
                
            validation = self._validate_pattern(points, pattern_type)
            
            if validation.is_valid:
                # Calculate price targets and risk levels
                price_targets = self._calculate_price_targets(points, pattern_type)
                stop_loss = self._calculate_stop_loss(points, pattern_type)
                risk_reward = self._calculate_risk_reward(
                    points['D'].price, price_targets, stop_loss
                )
                
                pattern = HarmonicPattern(
                    pattern_type=pattern_type,
                    points=points.copy(),
                    fibonacci_ratios=self._calculate_actual_ratios(points, pattern_type),
                    target_ratios=self.pattern_ratios[pattern_type],
                    accuracy_score=validation.accuracy_score,
                    completion_level=self._determine_completion_level(points),
                    price_targets=price_targets,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward,
                    confidence=self._calculate_pattern_confidence(validation),
                    formed_at=points['D'].time
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _test_abcd_pattern(self, points: Dict[str, HarmonicPoint]) -> List[HarmonicPattern]:
        """Test for ABCD patterns specifically"""
        patterns = []
        
        # Test both bullish and bearish ABCD
        for pattern_type in [HarmonicType.ABCD_BULLISH, HarmonicType.ABCD_BEARISH]:
            validation = self._validate_abcd_pattern(points, pattern_type)
            
            if validation.is_valid:
                # Calculate targets for ABCD pattern
                price_targets = self._calculate_abcd_targets(points, pattern_type)
                stop_loss = self._calculate_abcd_stop_loss(points, pattern_type)
                risk_reward = self._calculate_risk_reward(
                    points['D'].price, price_targets, stop_loss
                )
                
                pattern = HarmonicPattern(
                    pattern_type=pattern_type,
                    points=points.copy(),
                    fibonacci_ratios=self._calculate_abcd_ratios(points),
                    target_ratios=self.pattern_ratios[pattern_type],
                    accuracy_score=validation.accuracy_score,
                    completion_level='completed',
                    price_targets=price_targets,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward,
                    confidence=self._calculate_pattern_confidence(validation),
                    formed_at=points['D'].time
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _validate_pattern(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> PatternValidation:
        """Validate if points form the specified harmonic pattern"""
        ratios = self.pattern_ratios[pattern_type]
        actual_ratios = self._calculate_actual_ratios(points, pattern_type)
        
        violations = []
        fibonacci_accuracy = {}
        valid_ratios = 0
        total_ratios = len(ratios)
        
        for ratio_name, (min_val, max_val) in ratios.items():
            if ratio_name in actual_ratios:
                actual_val = actual_ratios[ratio_name]
                
                if min_val <= actual_val <= max_val:
                    valid_ratios += 1
                    # Calculate how close to ideal ratio
                    ideal_ratio = (min_val + max_val) / 2
                    accuracy = 1 - abs(actual_val - ideal_ratio) / ideal_ratio
                    fibonacci_accuracy[ratio_name] = accuracy
                else:
                    violations.append(f"{ratio_name}: {actual_val:.3f} outside range [{min_val:.3f}, {max_val:.3f}]")
                    fibonacci_accuracy[ratio_name] = 0
        
        # Pattern is valid if most ratios are within tolerance
        is_valid = valid_ratios >= total_ratios * 0.75  # At least 75% of ratios must be valid
        accuracy_score = valid_ratios / total_ratios if total_ratios > 0 else 0
        
        # Additional validation for pattern structure
        if is_valid:
            structure_valid = self._validate_pattern_structure(points, pattern_type)
            if not structure_valid:
                is_valid = False
                violations.append("Pattern structure invalid")
        
        return PatternValidation(
            is_valid=is_valid,
            accuracy_score=accuracy_score,
            violations=violations,
            fibonacci_accuracy=fibonacci_accuracy
        )
    
    def _validate_abcd_pattern(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> PatternValidation:
        """Validate ABCD pattern specifically"""
        ratios = self.pattern_ratios[pattern_type]
        actual_ratios = self._calculate_abcd_ratios(points)
        
        violations = []
        fibonacci_accuracy = {}
        valid_ratios = 0
        
        for ratio_name, (min_val, max_val) in ratios.items():
            if ratio_name in actual_ratios:
                actual_val = actual_ratios[ratio_name]
                
                if min_val <= actual_val <= max_val:
                    valid_ratios += 1
                    ideal_ratio = (min_val + max_val) / 2
                    accuracy = 1 - abs(actual_val - ideal_ratio) / ideal_ratio
                    fibonacci_accuracy[ratio_name] = accuracy
                else:
                    violations.append(f"{ratio_name}: {actual_val:.3f} outside range [{min_val:.3f}, {max_val:.3f}]")
                    fibonacci_accuracy[ratio_name] = 0
        
        is_valid = valid_ratios == len(ratios)  # All ratios must be valid for ABCD
        accuracy_score = valid_ratios / len(ratios) if ratios else 0
        
        return PatternValidation(
            is_valid=is_valid,
            accuracy_score=accuracy_score,
            violations=violations,
            fibonacci_accuracy=fibonacci_accuracy
        )
    
    def _calculate_actual_ratios(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> Dict[str, float]:
        """Calculate actual Fibonacci ratios for the pattern"""
        ratios = {}
        
        try:
            x_price = points['X'].price
            a_price = points['A'].price
            b_price = points['B'].price
            c_price = points['C'].price
            d_price = points['D'].price
            
            # Calculate standard ratios
            xa_range = abs(a_price - x_price)
            ab_range = abs(b_price - a_price)
            bc_range = abs(c_price - b_price)
            cd_range = abs(d_price - c_price)
            ad_range = abs(d_price - a_price)
            
            if xa_range > 0:
                ratios['AB_XA'] = ab_range / xa_range
                ratios['AD_XA'] = ad_range / xa_range
            
            if ab_range > 0:
                ratios['BC_AB'] = bc_range / ab_range
            
            if bc_range > 0:
                ratios['CD_BC'] = cd_range / bc_range
            
            # Pattern-specific ratios
            if 'SHARK' in pattern_type.value:
                # Shark pattern uses different reference points
                ox_range = xa_range  # In shark pattern, O=X typically
                if ox_range > 0:
                    ratios['AB_OX'] = ab_range / ox_range
                    ratios['AD_OX'] = ad_range / ox_range
                    
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
        
        return ratios
    
    def _calculate_abcd_ratios(self, points: Dict[str, HarmonicPoint]) -> Dict[str, float]:
        """Calculate ratios for ABCD pattern"""
        ratios = {}
        
        try:
            a_price = points['A'].price
            b_price = points['B'].price
            c_price = points['C'].price
            d_price = points['D'].price
            
            ab_range = abs(b_price - a_price)
            bc_range = abs(c_price - b_price)
            cd_range = abs(d_price - c_price)
            
            if ab_range > 0:
                ratios['BC_AB'] = bc_range / ab_range
                ratios['CD_AB'] = cd_range / ab_range
                
        except Exception as e:
            logger.error(f"Error calculating ABCD ratios: {e}")
        
        return ratios
    
    def _validate_pattern_structure(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> bool:
        """Validate the geometric structure of the pattern"""
        try:
            # Check alternating high-low structure
            if 'BULLISH' in pattern_type.value:
                # Bullish: X(low) -> A(high) -> B(low) -> C(high) -> D(low)
                return (points['X'].price < points['A'].price and
                        points['A'].price > points['B'].price and
                        points['B'].price < points['C'].price and
                        points['C'].price > points['D'].price)
            else:
                # Bearish: X(high) -> A(low) -> B(high) -> C(low) -> D(high)
                return (points['X'].price > points['A'].price and
                        points['A'].price < points['B'].price and
                        points['B'].price > points['C'].price and
                        points['C'].price < points['D'].price)
                        
        except Exception as e:
            logger.error(f"Error validating pattern structure: {e}")
            return False
    
    def _calculate_price_targets(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> List[float]:
        """Calculate price targets for the pattern"""
        targets = []
        
        try:
            d_price = points['D'].price
            c_price = points['C'].price
            a_price = points['A'].price
            x_price = points['X'].price
            
            cd_range = abs(d_price - c_price)
            xa_range = abs(x_price - a_price)
            
            if 'BULLISH' in pattern_type.value:
                # Bullish targets (price moving up from D)
                targets.append(d_price + cd_range * 0.382)  # 38.2% retracement
                targets.append(d_price + cd_range * 0.618)  # 61.8% retracement
                targets.append(c_price)  # Return to C level
                
                # Extended targets
                if xa_range > 0:
                    targets.append(d_price + xa_range * 0.618)
                    targets.append(d_price + xa_range * 1.0)
            else:
                # Bearish targets (price moving down from D)
                targets.append(d_price - cd_range * 0.382)
                targets.append(d_price - cd_range * 0.618)
                targets.append(c_price)  # Return to C level
                
                if xa_range > 0:
                    targets.append(d_price - xa_range * 0.618)
                    targets.append(d_price - xa_range * 1.0)
            
            # Remove duplicates and sort
            targets = list(set(targets))
            targets.sort(key=lambda x: abs(x - d_price))
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {e}")
        
        return targets[:3]  # Return top 3 targets
    
    def _calculate_abcd_targets(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> List[float]:
        """Calculate price targets for ABCD pattern"""
        targets = []
        
        try:
            a_price = points['A'].price
            b_price = points['B'].price
            c_price = points['C'].price
            d_price = points['D'].price
            
            ab_range = abs(b_price - a_price)
            cd_range = abs(d_price - c_price)
            
            if 'BULLISH' in pattern_type.value:
                # Bullish ABCD targets
                targets.append(d_price + cd_range * 0.382)
                targets.append(d_price + cd_range * 0.618)
                targets.append(c_price)  # BC retracement
                targets.append(d_price + ab_range)  # AB projection
            else:
                # Bearish ABCD targets
                targets.append(d_price - cd_range * 0.382)
                targets.append(d_price - cd_range * 0.618)
                targets.append(c_price)
                targets.append(d_price - ab_range)
            
        except Exception as e:
            logger.error(f"Error calculating ABCD targets: {e}")
        
        return targets[:3]
    
    def _calculate_stop_loss(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> Optional[float]:
        """Calculate stop loss level for the pattern"""
        try:
            d_price = points['D'].price
            x_price = points['X'].price
            
            if 'BULLISH' in pattern_type.value:
                # Bullish stop: below X or D minus small buffer
                buffer = abs(d_price - x_price) * 0.05  # 5% buffer
                return min(x_price, d_price - buffer)
            else:
                # Bearish stop: above X or D plus small buffer
                buffer = abs(d_price - x_price) * 0.05
                return max(x_price, d_price + buffer)
                
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None
    
    def _calculate_abcd_stop_loss(
        self,
        points: Dict[str, HarmonicPoint],
        pattern_type: HarmonicType
    ) -> Optional[float]:
        """Calculate stop loss for ABCD pattern"""
        try:
            a_price = points['A'].price
            d_price = points['D'].price
            
            buffer_pct = 0.02  # 2% buffer
            
            if 'BULLISH' in pattern_type.value:
                return d_price * (1 - buffer_pct)
            else:
                return d_price * (1 + buffer_pct)
                
        except Exception as e:
            logger.error(f"Error calculating ABCD stop loss: {e}")
            return None
    
    def _calculate_risk_reward(
        self,
        entry_price: float,
        targets: List[float],
        stop_loss: Optional[float]
    ) -> Optional[float]:
        """Calculate risk/reward ratio"""
        if not targets or stop_loss is None:
            return None
        
        try:
            risk = abs(entry_price - stop_loss)
            if risk == 0:
                return None
            
            # Use first target for risk/reward calculation
            reward = abs(targets[0] - entry_price)
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward: {e}")
            return None
    
    def _determine_completion_level(self, points: Dict[str, HarmonicPoint]) -> str:
        """Determine if pattern is forming, completed, or active"""
        # For now, assume all detected patterns are completed
        # In real-time analysis, this would check current price vs D point
        return 'completed'
    
    def _calculate_pattern_confidence(self, validation: PatternValidation) -> float:
        """Calculate overall confidence in the pattern"""
        base_confidence = validation.accuracy_score
        
        # Adjust based on Fibonacci accuracy
        if validation.fibonacci_accuracy:
            avg_fib_accuracy = np.mean(list(validation.fibonacci_accuracy.values()))
            return (base_confidence + avg_fib_accuracy) / 2
        
        return base_confidence
    
    def _filter_overlapping_patterns(
        self,
        patterns: List[HarmonicPattern]
    ) -> List[HarmonicPattern]:
        """Filter out overlapping patterns, keeping the best ones"""
        if not patterns:
            return patterns
        
        # Sort by accuracy score
        patterns.sort(key=lambda x: x.accuracy_score, reverse=True)
        
        filtered = []
        used_points = set()
        
        for pattern in patterns:
            # Check if this pattern uses any points already used by a better pattern
            pattern_points = set(p.index for p in pattern.points.values())
            
            if not pattern_points.intersection(used_points):
                filtered.append(pattern)
                used_points.update(pattern_points)
        
        return filtered
    
    def analyze_pattern_performance(
        self,
        patterns: List[HarmonicPattern],
        price_data: pd.DataFrame,
        lookforward_periods: int = 20
    ) -> Dict:
        """Analyze historical performance of detected patterns"""
        if not patterns:
            return {'total_patterns': 0}
        
        performance_stats = {
            'total_patterns': len(patterns),
            'bullish_patterns': 0,
            'bearish_patterns': 0,
            'successful_patterns': 0,
            'average_success_rate': 0,
            'pattern_type_performance': {},
            'risk_reward_analysis': {}
        }
        
        successful_count = 0
        
        for pattern in patterns:
            # Determine if bullish or bearish
            if 'BULLISH' in pattern.pattern_type.value:
                performance_stats['bullish_patterns'] += 1
            else:
                performance_stats['bearish_patterns'] += 1
            
            # Check if pattern achieved targets (simplified analysis)
            d_point = pattern.points['D']
            d_index = d_point.index
            
            if d_index + lookforward_periods < len(price_data):
                future_prices = price_data.iloc[d_index:d_index + lookforward_periods]
                
                success = False
                if pattern.price_targets:
                    first_target = pattern.price_targets[0]
                    
                    if 'BULLISH' in pattern.pattern_type.value:
                        success = future_prices['High'].max() >= first_target
                    else:
                        success = future_prices['Low'].min() <= first_target
                
                if success:
                    successful_count += 1
                    
                # Track by pattern type
                pattern_name = pattern.pattern_type.value
                if pattern_name not in performance_stats['pattern_type_performance']:
                    performance_stats['pattern_type_performance'][pattern_name] = {
                        'count': 0, 'successful': 0
                    }
                
                performance_stats['pattern_type_performance'][pattern_name]['count'] += 1
                if success:
                    performance_stats['pattern_type_performance'][pattern_name]['successful'] += 1
        
        performance_stats['successful_patterns'] = successful_count
        performance_stats['average_success_rate'] = successful_count / len(patterns) if patterns else 0
        
        # Calculate success rates by pattern type
        for pattern_name, stats in performance_stats['pattern_type_performance'].items():
            if stats['count'] > 0:
                stats['success_rate'] = stats['successful'] / stats['count']
        
        return performance_stats