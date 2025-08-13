"""
Market Profile Analysis Engine

Analyzes market profile concepts:
- Point of Control (POC) - highest volume price level
- Value Area High (VAH) and Low (VAL) - 70% of volume
- Volume Profile analysis
- Time Price Opportunities (TPO)
- Market Profile shapes (Normal, Trending, Non-Normal)
- Developing vs Composite profiles
- Naked POCs and Single Prints

Provides institutional-level market structure analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class ProfileType(Enum):
    NORMAL_DAY = "normal_day"
    TREND_DAY = "trend_day"
    NORMAL_VARIATION = "normal_variation"
    NEUTRAL_DAY = "neutral_day"
    NON_TREND_DAY = "non_trend_day"
    DOUBLE_DISTRIBUTION = "double_distribution"

class MarketContext(Enum):
    BALANCE = "balance"
    DIRECTIONAL_MOVEMENT = "directional_movement"
    INITIATIVE_ACTIVITY = "initiative_activity"
    RESPONSIVE_ACTIVITY = "responsive_activity"

@dataclass
class VolumeNode:
    """Volume at price level"""
    price: float
    volume: int
    tpo_count: int  # Time Price Opportunity count
    time_periods: List[str]

@dataclass
class ValueArea:
    """Value area definition (70% of volume)"""
    high: float
    low: float
    poc: float  # Point of Control
    volume_percentage: float
    total_volume: int

@dataclass
class MarketProfile:
    """Complete market profile analysis"""
    date: datetime
    value_area: ValueArea
    profile_type: ProfileType
    market_context: MarketContext
    
    # Key levels
    poc: float
    vah: float  # Value Area High
    val: float  # Value Area Low
    
    # Volume analysis
    volume_nodes: List[VolumeNode]
    high_volume_nodes: List[VolumeNode]
    low_volume_nodes: List[VolumeNode]
    
    # Profile characteristics
    range_high: float
    range_low: float
    initial_balance_high: float
    initial_balance_low: float
    
    # Advanced metrics
    single_prints: List[float]  # Prices that traded only once
    poor_highs: List[float]  # Highs with low volume
    poor_lows: List[float]   # Lows with low volume
    excess: List[Tuple[str, float]]  # Buying/Selling excess
    
    # Distribution metrics
    volume_distribution: Dict[float, int]
    tpo_distribution: Dict[float, int]
    
    # Summary stats
    total_volume: int
    trading_range: float
    value_area_range: float
    balance_area: Tuple[float, float]

class MarketProfileAnalyzer:
    """
    Advanced Market Profile analysis engine
    
    Features:
    - Point of Control identification
    - Value Area calculation (70% volume rule)
    - Profile type classification
    - Market context analysis
    - Single print identification
    - Excess identification
    - Volume distribution analysis
    - Multi-timeframe profile analysis
    """
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
    
    def analyze_market_profile(
        self,
        intraday_data: pd.DataFrame,
        session_start: str = "09:30",
        session_end: str = "16:00",
        initial_balance_minutes: int = 60
    ) -> MarketProfile:
        """
        Create comprehensive market profile analysis
        
        Args:
            intraday_data: Minute-by-minute OHLCV data
            session_start: Session start time
            session_end: Session end time
            initial_balance_minutes: Minutes for initial balance calculation
            
        Returns:
            Complete market profile analysis
        """
        try:
            # Filter session data
            session_data = self._filter_session_data(intraday_data, session_start, session_end)
            
            if session_data.empty:
                return self._create_empty_profile()
            
            # Build volume profile
            volume_nodes = self._build_volume_profile(session_data)
            
            if not volume_nodes:
                return self._create_empty_profile()
            
            # Calculate value area and POC
            value_area = self._calculate_value_area(volume_nodes)
            
            # Determine profile type
            profile_type = self._classify_profile_type(session_data, value_area)
            
            # Determine market context
            market_context = self._determine_market_context(session_data, value_area)
            
            # Calculate initial balance
            ib_high, ib_low = self._calculate_initial_balance(
                session_data, initial_balance_minutes
            )
            
            # Identify market profile features
            single_prints = self._identify_single_prints(volume_nodes)
            poor_highs = self._identify_poor_highs(session_data, volume_nodes)
            poor_lows = self._identify_poor_lows(session_data, volume_nodes)
            excess = self._identify_excess(session_data, volume_nodes)
            
            # Classify volume nodes
            high_volume_nodes = self._identify_high_volume_nodes(volume_nodes)
            low_volume_nodes = self._identify_low_volume_nodes(volume_nodes)
            
            # Calculate distributions
            volume_dist = {node.price: node.volume for node in volume_nodes}
            tpo_dist = {node.price: node.tpo_count for node in volume_nodes}
            
            # Calculate range metrics
            range_high = session_data['High'].max()
            range_low = session_data['Low'].min()
            trading_range = range_high - range_low
            value_area_range = value_area.high - value_area.low
            
            # Determine balance area (simplified)
            balance_area = (value_area.low * 0.99, value_area.high * 1.01)
            
            profile = MarketProfile(
                date=session_data.index[0].date() if len(session_data) > 0 else datetime.now().date(),
                value_area=value_area,
                profile_type=profile_type,
                market_context=market_context,
                poc=value_area.poc,
                vah=value_area.high,
                val=value_area.low,
                volume_nodes=volume_nodes,
                high_volume_nodes=high_volume_nodes,
                low_volume_nodes=low_volume_nodes,
                range_high=range_high,
                range_low=range_low,
                initial_balance_high=ib_high,
                initial_balance_low=ib_low,
                single_prints=single_prints,
                poor_highs=poor_highs,
                poor_lows=poor_lows,
                excess=excess,
                volume_distribution=volume_dist,
                tpo_distribution=tpo_dist,
                total_volume=sum(node.volume for node in volume_nodes),
                trading_range=trading_range,
                value_area_range=value_area_range,
                balance_area=balance_area
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing market profile: {e}")
            return self._create_empty_profile()
    
    def _filter_session_data(
        self,
        data: pd.DataFrame,
        start_time: str,
        end_time: str
    ) -> pd.DataFrame:
        """Filter data for trading session"""
        try:
            # Convert to datetime index if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Filter by time
            session_data = data.between_time(start_time, end_time)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Error filtering session data: {e}")
            return pd.DataFrame()
    
    def _build_volume_profile(self, data: pd.DataFrame) -> List[VolumeNode]:
        """Build volume profile from intraday data"""
        volume_nodes = []
        
        try:
            # Create price levels based on tick size
            min_price = data['Low'].min()
            max_price = data['High'].max()
            
            # Round to nearest tick
            min_price = np.floor(min_price / self.tick_size) * self.tick_size
            max_price = np.ceil(max_price / self.tick_size) * self.tick_size
            
            # Create price levels
            price_levels = np.arange(min_price, max_price + self.tick_size, self.tick_size)
            
            for price_level in price_levels:
                total_volume = 0
                tpo_count = 0
                time_periods = []
                
                # Check each bar to see if it traded at this price level
                for idx, row in data.iterrows():
                    if row['Low'] <= price_level <= row['High']:
                        # Price level was touched in this time period
                        tpo_count += 1
                        time_periods.append(idx.strftime('%H:%M'))
                        
                        # Estimate volume at this price level
                        # Simple approximation: distribute volume evenly across price range
                        price_range = row['High'] - row['Low']
                        if price_range > 0:
                            volume_at_level = row['Volume'] / (price_range / self.tick_size)
                        else:
                            volume_at_level = row['Volume']
                        
                        total_volume += int(volume_at_level)
                
                if total_volume > 0:
                    volume_nodes.append(VolumeNode(
                        price=round(price_level, 2),
                        volume=total_volume,
                        tpo_count=tpo_count,
                        time_periods=time_periods
                    ))
            
            # Sort by price
            volume_nodes.sort(key=lambda x: x.price)
            
            return volume_nodes
            
        except Exception as e:
            logger.error(f"Error building volume profile: {e}")
            return []
    
    def _calculate_value_area(self, volume_nodes: List[VolumeNode]) -> ValueArea:
        """Calculate value area (70% of volume) and POC"""
        if not volume_nodes:
            return ValueArea(0, 0, 0, 0, 0)
        
        # Find POC (Point of Control) - highest volume node
        poc_node = max(volume_nodes, key=lambda x: x.volume)
        poc_price = poc_node.price
        
        total_volume = sum(node.volume for node in volume_nodes)
        target_volume = int(total_volume * 0.70)  # 70% of volume
        
        # Build value area around POC
        value_area_volume = poc_node.volume
        value_area_nodes = [poc_node]
        
        # Sort nodes by distance from POC
        other_nodes = [node for node in volume_nodes if node.price != poc_price]
        other_nodes.sort(key=lambda x: abs(x.price - poc_price))
        
        # Add nodes to value area until we reach 70% of volume
        for node in other_nodes:
            if value_area_volume >= target_volume:
                break
            value_area_nodes.append(node)
            value_area_volume += node.volume
        
        # Find value area high and low
        value_area_prices = [node.price for node in value_area_nodes]
        vah = max(value_area_prices)
        val = min(value_area_prices)
        
        volume_percentage = (value_area_volume / total_volume) * 100 if total_volume > 0 else 0
        
        return ValueArea(
            high=vah,
            low=val,
            poc=poc_price,
            volume_percentage=volume_percentage,
            total_volume=value_area_volume
        )
    
    def _classify_profile_type(
        self,
        data: pd.DataFrame,
        value_area: ValueArea
    ) -> ProfileType:
        """Classify the market profile type"""
        try:
            session_high = data['High'].max()
            session_low = data['Low'].min()
            session_range = session_high - session_low
            value_range = value_area.high - value_area.low
            
            # Opening and closing prices
            open_price = data['Open'].iloc[0]
            close_price = data['Close'].iloc[-1]
            
            # Calculate some metrics
            value_area_ratio = value_range / session_range if session_range > 0 else 0
            poc_position = (value_area.poc - session_low) / session_range if session_range > 0 else 0.5
            
            # Classification logic
            if value_area_ratio > 0.6:
                # Wide value area suggests normal day
                if 0.3 < poc_position < 0.7:
                    return ProfileType.NORMAL_DAY
                else:
                    return ProfileType.NORMAL_VARIATION
            elif value_area_ratio < 0.3:
                # Narrow value area
                if abs(close_price - open_price) / session_range > 0.7:
                    return ProfileType.TREND_DAY
                else:
                    return ProfileType.NEUTRAL_DAY
            else:
                # Medium value area
                if abs(close_price - open_price) / session_range > 0.5:
                    return ProfileType.NON_TREND_DAY
                else:
                    return ProfileType.DOUBLE_DISTRIBUTION
            
        except Exception as e:
            logger.error(f"Error classifying profile type: {e}")
            return ProfileType.NORMAL_DAY
    
    def _determine_market_context(
        self,
        data: pd.DataFrame,
        value_area: ValueArea
    ) -> MarketContext:
        """Determine market context"""
        try:
            # Analyze price action relative to value area
            closes_above_va = (data['Close'] > value_area.high).sum()
            closes_below_va = (data['Close'] < value_area.low).sum()
            closes_in_va = len(data) - closes_above_va - closes_below_va
            
            total_bars = len(data)
            above_va_ratio = closes_above_va / total_bars
            below_va_ratio = closes_below_va / total_bars
            in_va_ratio = closes_in_va / total_bars
            
            if in_va_ratio > 0.6:
                return MarketContext.BALANCE
            elif above_va_ratio > 0.4 or below_va_ratio > 0.4:
                return MarketContext.DIRECTIONAL_MOVEMENT
            elif above_va_ratio > 0.2 and below_va_ratio > 0.2:
                return MarketContext.INITIATIVE_ACTIVITY
            else:
                return MarketContext.RESPONSIVE_ACTIVITY
                
        except Exception as e:
            logger.error(f"Error determining market context: {e}")
            return MarketContext.BALANCE
    
    def _calculate_initial_balance(
        self,
        data: pd.DataFrame,
        minutes: int
    ) -> Tuple[float, float]:
        """Calculate initial balance high and low"""
        try:
            # Get first N minutes of data
            ib_data = data.head(minutes)
            
            if ib_data.empty:
                return 0, 0
                
            ib_high = ib_data['High'].max()
            ib_low = ib_data['Low'].min()
            
            return ib_high, ib_low
            
        except Exception as e:
            logger.error(f"Error calculating initial balance: {e}")
            return 0, 0
    
    def _identify_single_prints(self, volume_nodes: List[VolumeNode]) -> List[float]:
        """Identify single prints (prices that traded only in one time period)"""
        single_prints = []
        
        for node in volume_nodes:
            if node.tpo_count == 1:  # Only one TPO
                single_prints.append(node.price)
        
        return single_prints
    
    def _identify_poor_highs(
        self,
        data: pd.DataFrame,
        volume_nodes: List[VolumeNode]
    ) -> List[float]:
        """Identify poor highs (highs with low volume)"""
        poor_highs = []
        
        try:
            # Find local highs
            highs = data['High'].values
            high_indices = find_peaks(highs, distance=5)[0]
            
            # Check volume at these highs
            avg_volume = np.mean([node.volume for node in volume_nodes])
            
            for idx in high_indices:
                high_price = highs[idx]
                
                # Find volume node closest to this high
                closest_node = min(volume_nodes, key=lambda x: abs(x.price - high_price))
                
                if closest_node.volume < avg_volume * 0.5:  # Less than 50% of average volume
                    poor_highs.append(high_price)
            
        except Exception as e:
            logger.error(f"Error identifying poor highs: {e}")
        
        return poor_highs
    
    def _identify_poor_lows(
        self,
        data: pd.DataFrame,
        volume_nodes: List[VolumeNode]
    ) -> List[float]:
        """Identify poor lows (lows with low volume)"""
        poor_lows = []
        
        try:
            # Find local lows
            lows = data['Low'].values
            low_indices = find_peaks(-lows, distance=5)[0]
            
            # Check volume at these lows
            avg_volume = np.mean([node.volume for node in volume_nodes])
            
            for idx in low_indices:
                low_price = lows[idx]
                
                # Find volume node closest to this low
                closest_node = min(volume_nodes, key=lambda x: abs(x.price - low_price))
                
                if closest_node.volume < avg_volume * 0.5:
                    poor_lows.append(low_price)
            
        except Exception as e:
            logger.error(f"Error identifying poor lows: {e}")
        
        return poor_lows
    
    def _identify_excess(
        self,
        data: pd.DataFrame,
        volume_nodes: List[VolumeNode]
    ) -> List[Tuple[str, float]]:
        """Identify buying and selling excess"""
        excess = []
        
        try:
            session_high = data['High'].max()
            session_low = data['Low'].min()
            
            # Find volume nodes near extremes
            high_threshold = session_high - (session_high - session_low) * 0.1
            low_threshold = session_low + (session_high - session_low) * 0.1
            
            high_nodes = [node for node in volume_nodes if node.price >= high_threshold]
            low_nodes = [node for node in volume_nodes if node.price <= low_threshold]
            
            # Check for selling excess (low volume at highs)
            if high_nodes:
                avg_high_volume = np.mean([node.volume for node in high_nodes])
                total_volume = sum(node.volume for node in volume_nodes)
                avg_total_volume = total_volume / len(volume_nodes)
                
                if avg_high_volume < avg_total_volume * 0.3:  # Very low volume at highs
                    excess.append(('selling_excess', session_high))
            
            # Check for buying excess (low volume at lows)
            if low_nodes:
                avg_low_volume = np.mean([node.volume for node in low_nodes])
                total_volume = sum(node.volume for node in volume_nodes)
                avg_total_volume = total_volume / len(volume_nodes)
                
                if avg_low_volume < avg_total_volume * 0.3:  # Very low volume at lows
                    excess.append(('buying_excess', session_low))
            
        except Exception as e:
            logger.error(f"Error identifying excess: {e}")
        
        return excess
    
    def _identify_high_volume_nodes(self, volume_nodes: List[VolumeNode]) -> List[VolumeNode]:
        """Identify high volume nodes (above 75th percentile)"""
        if not volume_nodes:
            return []
            
        volumes = [node.volume for node in volume_nodes]
        threshold = np.percentile(volumes, 75)
        
        return [node for node in volume_nodes if node.volume >= threshold]
    
    def _identify_low_volume_nodes(self, volume_nodes: List[VolumeNode]) -> List[VolumeNode]:
        """Identify low volume nodes (below 25th percentile)"""
        if not volume_nodes:
            return []
            
        volumes = [node.volume for node in volume_nodes]
        threshold = np.percentile(volumes, 25)
        
        return [node for node in volume_nodes if node.volume <= threshold]
    
    def _create_empty_profile(self) -> MarketProfile:
        """Create empty profile for error cases"""
        return MarketProfile(
            date=datetime.now().date(),
            value_area=ValueArea(0, 0, 0, 0, 0),
            profile_type=ProfileType.NORMAL_DAY,
            market_context=MarketContext.BALANCE,
            poc=0,
            vah=0,
            val=0,
            volume_nodes=[],
            high_volume_nodes=[],
            low_volume_nodes=[],
            range_high=0,
            range_low=0,
            initial_balance_high=0,
            initial_balance_low=0,
            single_prints=[],
            poor_highs=[],
            poor_lows=[],
            excess=[],
            volume_distribution={},
            tpo_distribution={},
            total_volume=0,
            trading_range=0,
            value_area_range=0,
            balance_area=(0, 0)
        )
    
    def compare_profiles(
        self,
        current_profile: MarketProfile,
        previous_profiles: List[MarketProfile]
    ) -> Dict:
        """Compare current profile with previous profiles"""
        if not previous_profiles:
            return {'comparison': 'no_previous_data'}
        
        comparison = {
            'poc_relationship': self._analyze_poc_relationship(current_profile, previous_profiles),
            'value_area_relationship': self._analyze_value_area_relationship(current_profile, previous_profiles),
            'volume_comparison': self._analyze_volume_comparison(current_profile, previous_profiles),
            'range_comparison': self._analyze_range_comparison(current_profile, previous_profiles),
            'profile_type_trend': self._analyze_profile_type_trend(current_profile, previous_profiles)
        }
        
        return comparison
    
    def _analyze_poc_relationship(
        self,
        current: MarketProfile,
        previous: List[MarketProfile]
    ) -> str:
        """Analyze POC relationship with previous sessions"""
        prev_poc = previous[-1].poc
        
        if current.poc > prev_poc * 1.01:
            return 'above_previous_poc'
        elif current.poc < prev_poc * 0.99:
            return 'below_previous_poc'
        else:
            return 'near_previous_poc'
    
    def _analyze_value_area_relationship(
        self,
        current: MarketProfile,
        previous: List[MarketProfile]
    ) -> str:
        """Analyze value area relationship"""
        prev_va = previous[-1].value_area
        
        if current.val > prev_va.high:
            return 'above_previous_value_area'
        elif current.vah < prev_va.low:
            return 'below_previous_value_area'
        elif current.val >= prev_va.low and current.vah <= prev_va.high:
            return 'overlapping_value_area'
        else:
            return 'partially_overlapping_value_area'
    
    def _analyze_volume_comparison(
        self,
        current: MarketProfile,
        previous: List[MarketProfile]
    ) -> Dict:
        """Compare volume metrics"""
        prev_volumes = [p.total_volume for p in previous[-5:]]  # Last 5 sessions
        avg_prev_volume = np.mean(prev_volumes) if prev_volumes else 0
        
        if avg_prev_volume > 0:
            volume_ratio = current.total_volume / avg_prev_volume
        else:
            volume_ratio = 1.0
        
        return {
            'volume_ratio': volume_ratio,
            'volume_trend': 'above_average' if volume_ratio > 1.2 else 'below_average' if volume_ratio < 0.8 else 'average'
        }
    
    def _analyze_range_comparison(
        self,
        current: MarketProfile,
        previous: List[MarketProfile]
    ) -> Dict:
        """Compare trading range"""
        prev_ranges = [p.trading_range for p in previous[-5:]]
        avg_prev_range = np.mean(prev_ranges) if prev_ranges else 0
        
        if avg_prev_range > 0:
            range_ratio = current.trading_range / avg_prev_range
        else:
            range_ratio = 1.0
        
        return {
            'range_ratio': range_ratio,
            'range_trend': 'expanding' if range_ratio > 1.2 else 'contracting' if range_ratio < 0.8 else 'normal'
        }
    
    def _analyze_profile_type_trend(
        self,
        current: MarketProfile,
        previous: List[MarketProfile]
    ) -> Dict:
        """Analyze profile type trends"""
        recent_types = [p.profile_type.value for p in previous[-5:]]
        recent_types.append(current.profile_type.value)
        
        type_counts = {}
        for ptype in recent_types:
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        
        return {
            'dominant_type': dominant_type,
            'type_distribution': type_counts
        }
    
    def generate_trading_insights(self, profile: MarketProfile) -> List[str]:
        """Generate actionable trading insights from market profile"""
        insights = []
        
        # POC insights
        insights.append(f"Point of Control at {profile.poc:.2f} represents the fairest price")
        
        # Value area insights
        va_range_pct = (profile.value_area_range / profile.trading_range * 100) if profile.trading_range > 0 else 0
        insights.append(f"Value area covers {va_range_pct:.1f}% of trading range ({profile.val:.2f} - {profile.vah:.2f})")
        
        # Profile type insights
        if profile.profile_type == ProfileType.TREND_DAY:
            insights.append("Trend day profile suggests strong directional conviction")
        elif profile.profile_type == ProfileType.NORMAL_DAY:
            insights.append("Normal day profile indicates balanced auction process")
        elif profile.profile_type == ProfileType.NEUTRAL_DAY:
            insights.append("Neutral day suggests lack of directional conviction")
        
        # Market context insights
        if profile.market_context == MarketContext.BALANCE:
            insights.append("Market in balance - expect price to rotate around value area")
        elif profile.market_context == MarketContext.DIRECTIONAL_MOVEMENT:
            insights.append("Directional movement detected - breakout potential")
        
        # Single print insights
        if profile.single_prints:
            insights.append(f"{len(profile.single_prints)} single print areas identified - potential support/resistance")
        
        # Excess insights
        if profile.excess:
            for excess_type, price in profile.excess:
                insights.append(f"{excess_type.replace('_', ' ').title()} at {price:.2f}")
        
        # Volume insights
        if profile.high_volume_nodes:
            high_vol_prices = [node.price for node in profile.high_volume_nodes[:3]]
            insights.append(f"High volume areas: {', '.join(f'{p:.2f}' for p in high_vol_prices)}")
        
        # Initial balance insights
        ib_range = profile.initial_balance_high - profile.initial_balance_low
        if ib_range > 0:
            ib_break_pct = ((max(profile.range_high - profile.initial_balance_high, 
                                profile.initial_balance_low - profile.range_low)) / ib_range * 100)
            if ib_break_pct > 100:
                insights.append(f"Initial balance broken by {ib_break_pct:.0f}% - strong directional move")
        
        return insights