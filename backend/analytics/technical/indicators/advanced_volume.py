"""
Advanced Volume Analysis Indicators

Comprehensive volume analysis including:
- Volume Price Trend (VPT)
- Accumulation/Distribution Line improvements
- Chaikin Money Flow enhancements
- Volume Rate of Change (VROC)
- Price Volume Trend (PVT)
- Klinger Oscillator
- Elder Force Index
- Ease of Movement (EOM)
- Volume Profile analysis
- On Balance Volume variations
- Smart Money indicators
- Volume Thrust indicators

Provides institutional-grade volume analysis
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class AdvancedVolumeAnalyzer:
    """
    Advanced volume analysis with institutional-grade indicators
    
    Features:
    - Multiple volume-price indicators
    - Smart money flow detection
    - Volume pattern recognition
    - Accumulation/Distribution analysis
    - Volume thrust identification
    - Money flow analysis
    - Volume profile insights
    """
    
    def __init__(self):
        pass
    
    def calculate_all_volume_indicators(
        self,
        price_data: pd.DataFrame,
        periods: Dict[str, int] = None
    ) -> Dict:
        """
        Calculate all advanced volume indicators
        
        Args:
            price_data: DataFrame with OHLCV data
            periods: Dictionary of periods for different indicators
            
        Returns:
            Dictionary of all volume indicators
        """
        if periods is None:
            periods = {
                'cmf': 21,
                'klinger': 34,
                'force_index': 13,
                'eom': 14,
                'vroc': 25,
                'vpt_ma': 20
            }
        
        try:
            indicators = {}
            
            # Basic volume indicators
            indicators.update(self._calculate_basic_volume_indicators(price_data))
            
            # Advanced oscillators
            indicators.update(self._calculate_volume_oscillators(price_data, periods))
            
            # Money flow indicators
            indicators.update(self._calculate_money_flow_indicators(price_data, periods))
            
            # Volume patterns
            indicators.update(self._identify_volume_patterns(price_data))
            
            # Smart money indicators
            indicators.update(self._calculate_smart_money_indicators(price_data, periods))
            
            # Volume thrust analysis
            indicators.update(self._analyze_volume_thrusts(price_data))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def _calculate_basic_volume_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate basic volume indicators with improvements"""
        indicators = {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # Enhanced On Balance Volume (OBV)
            obv = np.zeros(len(data))
            for i in range(1, len(data)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv[i] = obv[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv[i] = obv[i-1] - volume.iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            indicators['obv'] = obv
            indicators['obv_ma'] = pd.Series(obv).rolling(20).mean().values
            
            # Volume Price Trend (VPT)
            price_change_pct = close.pct_change()
            vpt = (price_change_pct * volume).cumsum()
            indicators['vpt'] = vpt.values
            indicators['vpt_ma'] = vpt.rolling(20).mean().values
            
            # Price Volume Trend (PVT)
            pvt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
            indicators['pvt'] = pvt.values
            indicators['pvt_ma'] = pvt.rolling(20).mean().values
            
            # Enhanced Accumulation/Distribution Line
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_multiplier = money_flow_multiplier.fillna(0)
            money_flow_volume = money_flow_multiplier * volume
            ad_line = money_flow_volume.cumsum()
            indicators['ad_line'] = ad_line.values
            indicators['ad_line_ma'] = ad_line.rolling(20).mean().values
            
            # Volume Weighted Average Price (VWAP)
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            indicators['vwap'] = vwap.values
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating basic volume indicators: {e}")
            return {}
    
    def _calculate_volume_oscillators(self, data: pd.DataFrame, periods: Dict) -> Dict:
        """Calculate volume oscillators"""
        indicators = {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # Chaikin Money Flow (CMF)
            cmf_period = periods.get('cmf', 21)
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_multiplier = money_flow_multiplier.fillna(0)
            money_flow_volume = money_flow_multiplier * volume
            
            cmf = money_flow_volume.rolling(cmf_period).sum() / volume.rolling(cmf_period).sum()
            indicators['cmf'] = cmf.values
            
            # Klinger Oscillator
            klinger_period = periods.get('klinger', 34)
            klinger_signal_period = 13
            
            # Trend calculation for Klinger
            hlc3 = (high + low + close) / 3
            trend = np.where(hlc3 > hlc3.shift(1), 1, -1)
            
            # Volume Force
            volume_force = volume * trend * abs((hlc3 - hlc3.shift(1)) / hlc3.shift(1) * 100)
            volume_force = pd.Series(volume_force, index=data.index).fillna(0)
            
            # Klinger Oscillator
            klinger_fast = volume_force.ewm(span=klinger_period).mean()
            klinger_slow = volume_force.ewm(span=klinger_period*3).mean()
            klinger = klinger_fast - klinger_slow
            klinger_signal = klinger.ewm(span=klinger_signal_period).mean()
            
            indicators['klinger'] = klinger.values
            indicators['klinger_signal'] = klinger_signal.values
            indicators['klinger_histogram'] = (klinger - klinger_signal).values
            
            # Elder Force Index
            force_period = periods.get('force_index', 13)
            raw_force = (close - close.shift(1)) * volume
            force_index = raw_force.ewm(span=force_period).mean()
            indicators['force_index'] = force_index.values
            
            # Ease of Movement (EOM)
            eom_period = periods.get('eom', 14)
            distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
            box_height = volume / (high - low)
            box_height = box_height.replace([np.inf, -np.inf], 0).fillna(0)
            
            eom_raw = distance_moved / box_height
            eom_raw = eom_raw.replace([np.inf, -np.inf], 0).fillna(0)
            eom = eom_raw.rolling(eom_period).mean()
            indicators['eom'] = eom.values
            
            # Volume Rate of Change (VROC)
            vroc_period = periods.get('vroc', 25)
            vroc = volume.pct_change(vroc_period) * 100
            indicators['vroc'] = vroc.values
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating volume oscillators: {e}")
            return {}
    
    def _calculate_money_flow_indicators(self, data: pd.DataFrame, periods: Dict) -> Dict:
        """Calculate money flow indicators"""
        indicators = {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # Money Flow Index (MFI) - Enhanced version
            mfi_period = periods.get('mfi', 14)
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            # Positive and negative money flow
            positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
            negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
            
            positive_flow_sum = pd.Series(positive_flow).rolling(mfi_period).sum()
            negative_flow_sum = pd.Series(negative_flow).rolling(mfi_period).sum()
            
            money_ratio = positive_flow_sum / negative_flow_sum
            mfi = 100 - (100 / (1 + money_ratio))
            indicators['mfi'] = mfi.values
            
            # Chaikin A/D Oscillator
            ad_line = indicators.get('ad_line', np.zeros(len(data)))
            if len(ad_line) > 0:
                ad_series = pd.Series(ad_line)
                chaikin_fast = ad_series.ewm(span=3).mean()
                chaikin_slow = ad_series.ewm(span=10).mean()
                chaikin_oscillator = chaikin_fast - chaikin_slow
                indicators['chaikin_oscillator'] = chaikin_oscillator.values
            
            # Positive Volume Index (PVI) and Negative Volume Index (NVI)
            pvi = np.ones(len(data)) * 100
            nvi = np.ones(len(data)) * 100
            
            for i in range(1, len(data)):
                price_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
                
                if volume.iloc[i] > volume.iloc[i-1]:
                    pvi[i] = pvi[i-1] * (1 + price_change)
                    nvi[i] = nvi[i-1]
                elif volume.iloc[i] < volume.iloc[i-1]:
                    pvi[i] = pvi[i-1]
                    nvi[i] = nvi[i-1] * (1 + price_change)
                else:
                    pvi[i] = pvi[i-1]
                    nvi[i] = nvi[i-1]
            
            indicators['pvi'] = pvi
            indicators['nvi'] = nvi
            
            # PVI/NVI Moving Averages
            indicators['pvi_ma'] = pd.Series(pvi).rolling(255).mean().values
            indicators['nvi_ma'] = pd.Series(nvi).rolling(255).mean().values
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating money flow indicators: {e}")
            return {}
    
    def _identify_volume_patterns(self, data: pd.DataFrame) -> Dict:
        """Identify volume patterns"""
        patterns = {}
        
        try:
            volume = data['Volume']
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # Volume spikes (>2 standard deviations above mean)
            vol_mean = volume.rolling(20).mean()
            vol_std = volume.rolling(20).std()
            volume_spikes = volume > (vol_mean + 2 * vol_std)
            patterns['volume_spikes'] = volume_spikes.values.astype(int)
            
            # Climax volume (very high volume on reversal)
            price_change = close.pct_change().abs()
            volume_percentile = volume.rolling(50).rank(pct=True)
            price_change_percentile = price_change.rolling(50).rank(pct=True)
            
            climax_volume = (volume_percentile > 0.95) & (price_change_percentile > 0.9)
            patterns['climax_volume'] = climax_volume.values.astype(int)
            
            # Volume dry up (very low volume)
            volume_dry_up = volume < (vol_mean - vol_std)
            patterns['volume_dry_up'] = volume_dry_up.values.astype(int)
            
            # Volume confirmation (volume increases with price trend)
            price_trend = close > close.shift(1)
            volume_trend = volume > volume.shift(1)
            volume_confirmation = price_trend == volume_trend
            patterns['volume_confirmation'] = volume_confirmation.values.astype(int)
            
            # Volume divergence
            price_higher_high = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
            volume_lower_high = (volume < volume.shift(1)) & (volume.shift(1) < volume.shift(2))
            bullish_divergence = price_higher_high & volume_lower_high
            
            price_lower_low = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
            volume_higher_low = (volume > volume.shift(1)) & (volume.shift(1) > volume.shift(2))
            bearish_divergence = price_lower_low & volume_higher_low
            
            patterns['bullish_volume_divergence'] = bullish_divergence.values.astype(int)
            patterns['bearish_volume_divergence'] = bearish_divergence.values.astype(int)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying volume patterns: {e}")
            return {}
    
    def _calculate_smart_money_indicators(self, data: pd.DataFrame, periods: Dict) -> Dict:
        """Calculate smart money indicators"""
        indicators = {}
        
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # Smart Money Flow Index (SMFI)
            smfi_period = periods.get('smfi', 20)
            
            # Calculate money flow for each bar
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            # Smart money assumes large volume + small price moves = accumulation
            price_range = high - low
            price_range_pct = price_range / close
            
            # Smart money conditions
            high_volume = volume > volume.rolling(20).mean()
            small_range = price_range_pct < price_range_pct.rolling(20).mean()
            
            smart_money_flow = np.where(high_volume & small_range, money_flow, 0)
            smfi = pd.Series(smart_money_flow).rolling(smfi_period).sum()
            indicators['smart_money_flow'] = smfi.values
            
            # Institutional Volume Indicator
            # Large volume on small price moves suggests institutional activity
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / volume_ma
            
            price_change_pct = close.pct_change().abs()
            price_volatility = price_change_pct.rolling(20).std()
            volatility_ratio = price_change_pct / price_volatility
            
            # High volume, low volatility = institutional
            institutional_indicator = (volume_ratio > 1.5) & (volatility_ratio < 0.5)
            indicators['institutional_volume'] = institutional_indicator.values.astype(int)
            
            # Money Flow Acceleration
            money_flow_change = money_flow.pct_change()
            money_flow_acceleration = money_flow_change.pct_change()
            indicators['money_flow_acceleration'] = money_flow_acceleration.values
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating smart money indicators: {e}")
            return {}
    
    def _analyze_volume_thrusts(self, data: pd.DataFrame) -> Dict:
        """Analyze volume thrust patterns"""
        thrusts = {}
        
        try:
            volume = data['Volume']
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # Volume thrust: high volume with significant price move
            volume_ma = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            
            price_change_pct = close.pct_change()
            price_ma = price_change_pct.rolling(20).mean()
            price_std = price_change_pct.rolling(20).std()
            
            # Conditions for volume thrust
            high_volume = volume > (volume_ma + 1.5 * volume_std)
            significant_move = abs(price_change_pct) > (abs(price_ma) + 1.5 * price_std)
            
            volume_thrust = high_volume & significant_move
            thrusts['volume_thrust'] = volume_thrust.values.astype(int)
            
            # Upthrust and downthrust
            upthrust = volume_thrust & (price_change_pct > 0)
            downthrust = volume_thrust & (price_change_pct < 0)
            
            thrusts['upthrust'] = upthrust.values.astype(int)
            thrusts['downthrust'] = downthrust.values.astype(int)
            
            # Volume thrust strength
            volume_strength = (volume - volume_ma) / volume_std
            price_strength = abs(price_change_pct - price_ma) / price_std
            
            thrust_strength = (volume_strength + price_strength) / 2
            thrusts['thrust_strength'] = thrust_strength.values
            
            # Follow-through analysis
            # Check if volume thrust is followed by continuation
            followthrough_periods = 3
            followthrough = np.zeros(len(data))
            
            for i in range(len(data) - followthrough_periods):
                if volume_thrust.iloc[i]:
                    # Check next few periods
                    if upthrust.iloc[i]:
                        # For upthrust, check if price continues higher
                        future_high = high.iloc[i+1:i+1+followthrough_periods].max()
                        if future_high > high.iloc[i]:
                            followthrough[i] = 1
                    elif downthrust.iloc[i]:
                        # For downthrust, check if price continues lower
                        future_low = low.iloc[i+1:i+1+followthrough_periods].min()
                        if future_low < low.iloc[i]:
                            followthrough[i] = 1
            
            thrusts['thrust_followthrough'] = followthrough
            
            return thrusts
            
        except Exception as e:
            logger.error(f"Error analyzing volume thrusts: {e}")
            return {}
    
    def calculate_volume_profile(
        self,
        data: pd.DataFrame,
        bins: int = 50
    ) -> Dict:
        """Calculate volume profile"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # Price range
            min_price = low.min()
            max_price = high.max()
            
            # Create price bins
            price_bins = np.linspace(min_price, max_price, bins + 1)
            volume_profile = np.zeros(bins)
            
            # Distribute volume across price levels
            for i, row in data.iterrows():
                bar_high = row['High']
                bar_low = row['Low']
                bar_volume = row['Volume']
                
                # Find which bins this bar touches
                start_bin = np.digitize(bar_low, price_bins) - 1
                end_bin = np.digitize(bar_high, price_bins) - 1
                
                start_bin = max(0, min(start_bin, bins - 1))
                end_bin = max(0, min(end_bin, bins - 1))
                
                # Distribute volume evenly across touched bins
                if start_bin == end_bin:
                    volume_profile[start_bin] += bar_volume
                else:
                    bins_touched = end_bin - start_bin + 1
                    volume_per_bin = bar_volume / bins_touched
                    for bin_idx in range(start_bin, end_bin + 1):
                        volume_profile[bin_idx] += volume_per_bin
            
            # Find Point of Control (POC) - price level with highest volume
            poc_bin = np.argmax(volume_profile)
            poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
            
            # Find Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            value_area_volume = total_volume * 0.70
            
            # Start from POC and expand outward
            value_area_bins = [poc_bin]
            current_volume = volume_profile[poc_bin]
            
            while current_volume < value_area_volume and len(value_area_bins) < bins:
                # Find adjacent bins
                left_bin = min(value_area_bins) - 1
                right_bin = max(value_area_bins) + 1
                
                left_volume = volume_profile[left_bin] if left_bin >= 0 else 0
                right_volume = volume_profile[right_bin] if right_bin < bins else 0
                
                # Add the bin with higher volume
                if left_volume >= right_volume and left_bin >= 0:
                    value_area_bins.append(left_bin)
                    current_volume += left_volume
                elif right_bin < bins:
                    value_area_bins.append(right_bin)
                    current_volume += right_volume
                else:
                    break
            
            # Value Area High and Low
            val_bin = min(value_area_bins)
            vah_bin = max(value_area_bins)
            
            val_price = price_bins[val_bin]
            vah_price = price_bins[vah_bin + 1]
            
            return {
                'price_bins': price_bins,
                'volume_profile': volume_profile,
                'poc_price': poc_price,
                'val_price': val_price,
                'vah_price': vah_price,
                'value_area_volume_pct': current_volume / total_volume * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {}
    
    def generate_volume_insights(
        self,
        indicators: Dict,
        current_price: float
    ) -> List[str]:
        """Generate insights from volume analysis"""
        insights = []
        
        try:
            # CMF insights
            cmf = indicators.get('cmf', [])
            if len(cmf) > 0 and not np.isnan(cmf[-1]):
                if cmf[-1] > 0.1:
                    insights.append("Strong money flow into the security (CMF > 0.1)")
                elif cmf[-1] < -0.1:
                    insights.append("Strong money flow out of the security (CMF < -0.1)")
            
            # MFI insights
            mfi = indicators.get('mfi', [])
            if len(mfi) > 0 and not np.isnan(mfi[-1]):
                if mfi[-1] > 80:
                    insights.append("Money Flow Index indicates overbought conditions (MFI > 80)")
                elif mfi[-1] < 20:
                    insights.append("Money Flow Index indicates oversold conditions (MFI < 20)")
            
            # Volume spike insights
            volume_spikes = indicators.get('volume_spikes', [])
            if len(volume_spikes) > 5 and volume_spikes[-1]:
                insights.append("Unusual volume spike detected - potential significant move")
            
            # Smart money insights
            institutional_volume = indicators.get('institutional_volume', [])
            if len(institutional_volume) > 0 and institutional_volume[-1]:
                insights.append("Institutional volume pattern detected")
            
            # Volume thrust insights
            upthrust = indicators.get('upthrust', [])
            downthrust = indicators.get('downthrust', [])
            
            if len(upthrust) > 0 and upthrust[-1]:
                insights.append("Upward volume thrust - strong buying pressure")
            elif len(downthrust) > 0 and downthrust[-1]:
                insights.append("Downward volume thrust - strong selling pressure")
            
            # Volume divergence insights
            bull_div = indicators.get('bullish_volume_divergence', [])
            bear_div = indicators.get('bearish_volume_divergence', [])
            
            if len(bull_div) > 0 and bull_div[-1]:
                insights.append("Bullish volume divergence - potential reversal signal")
            elif len(bear_div) > 0 and bear_div[-1]:
                insights.append("Bearish volume divergence - potential reversal signal")
            
            # Klinger insights
            klinger = indicators.get('klinger', [])
            klinger_signal = indicators.get('klinger_signal', [])
            
            if (len(klinger) > 0 and len(klinger_signal) > 0 and 
                not np.isnan(klinger[-1]) and not np.isnan(klinger_signal[-1])):
                if klinger[-1] > klinger_signal[-1] and klinger[-2] <= klinger_signal[-2]:
                    insights.append("Klinger Oscillator bullish crossover")
                elif klinger[-1] < klinger_signal[-1] and klinger[-2] >= klinger_signal[-2]:
                    insights.append("Klinger Oscillator bearish crossover")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating volume insights: {e}")
            return ["Error generating volume insights"]