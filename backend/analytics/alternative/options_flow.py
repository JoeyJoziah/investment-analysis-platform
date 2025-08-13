"""
Options Flow Analysis Engine

Analyzes options market activity using free CBOE data and other sources:
- CBOE VIX and volatility indices
- Options volume from Yahoo Finance
- Put/Call ratios
- Options chain analysis
- Unusual options activity detection
- Options sentiment indicators

All data from free sources to maintain budget constraints
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf

import pandas as pd
import numpy as np
from scipy import stats

from backend.utils.cache import CacheManager
from backend.utils.rate_limiter import RateLimiter
from backend.utils.cost_monitor import CostMonitor

logger = logging.getLogger(__name__)

@dataclass
class OptionsContract:
    """Options contract data"""
    symbol: str
    option_type: str  # call or put
    strike: float
    expiration: str
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

@dataclass
class UnusualActivity:
    """Unusual options activity"""
    symbol: str
    contract: OptionsContract
    activity_type: str  # volume_spike, iv_spike, flow_direction
    magnitude: float  # How unusual (multiple of normal)
    description: str
    timestamp: datetime

@dataclass
class OptionsFlowSentiment:
    """Options flow sentiment analysis"""
    sentiment_score: float  # -1 (very bearish) to 1 (very bullish)
    confidence: float
    put_call_ratio: float
    call_volume_ratio: float  # Call volume / Total volume
    unusual_activity_score: float
    iv_trend: str  # rising, falling, stable
    flow_direction: str  # bullish, bearish, neutral
    key_levels: List[float]  # Important strike levels
    insights: List[str]

class OptionsFlowAnalyzer:
    """
    Comprehensive options flow analysis using free data sources
    
    Features:
    - Options volume and open interest analysis
    - Put/call ratio calculations
    - Unusual activity detection
    - Implied volatility analysis
    - Options sentiment scoring
    - Key strike level identification
    - Flow direction analysis
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = CacheManager()
        self.cost_monitor = CostMonitor()
        
        # Rate limiters
        self.yahoo_limiter = RateLimiter(calls=100, period=60)
        self.cboe_limiter = RateLimiter(calls=50, period=60)
        
    async def analyze_options_flow(
        self,
        symbol: str,
        days_back: int = 5,
        min_volume: int = 100
    ) -> Dict:
        """
        Analyze options flow for a stock
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back for historical comparison
            min_volume: Minimum volume threshold for analysis
            
        Returns:
            Comprehensive options flow analysis
        """
        # Check cache first
        cache_key = f"options_flow:{symbol}:{days_back}:{min_volume}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            results = {
                'symbol': symbol,
                'analysis_date': datetime.now(),
                'options_data': {},
                'sentiment': None,
                'unusual_activity': [],
                'key_strikes': [],
                'volatility_analysis': {},
                'insights': []
            }
            
            # Get options chain data
            options_data = await self._get_options_chain(symbol)
            results['options_data'] = options_data
            
            if not options_data or not options_data.get('calls') and not options_data.get('puts'):
                results['insights'] = ['No options data available']
                return results
            
            # Analyze options sentiment
            sentiment = await self._calculate_options_sentiment(symbol, options_data)
            results['sentiment'] = sentiment
            
            # Detect unusual activity
            unusual_activity = await self._detect_unusual_options_activity(
                symbol, options_data, min_volume
            )
            results['unusual_activity'] = unusual_activity
            
            # Identify key strike levels
            key_strikes = self._identify_key_strikes(options_data)
            results['key_strikes'] = key_strikes
            
            # Analyze volatility
            volatility_analysis = await self._analyze_volatility(symbol)
            results['volatility_analysis'] = volatility_analysis
            
            # Generate insights
            results['insights'] = self._generate_options_insights(
                sentiment, unusual_activity, key_strikes, volatility_analysis
            )
            
            # Cache results
            await self.cache.set(cache_key, results, expire=900)  # 15 minutes
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing options flow for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _get_options_chain(self, symbol: str) -> Dict:
        """Get options chain data from Yahoo Finance"""
        await self.yahoo_limiter.acquire()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return {}
            
            options_data = {
                'calls': [],
                'puts': [],
                'expirations': expirations,
                'underlying_price': None
            }
            
            # Get current stock price
            try:
                info = ticker.info
                options_data['underlying_price'] = info.get('currentPrice') or info.get('regularMarketPrice')
            except:
                # Fallback to historical data
                hist = ticker.history(period='1d')
                if not hist.empty:
                    options_data['underlying_price'] = hist['Close'].iloc[-1]
            
            # Get options data for near-term expirations (next 2-3 expirations)
            for expiration in expirations[:3]:
                try:
                    option_chain = ticker.option_chain(expiration)
                    
                    # Process calls
                    if hasattr(option_chain, 'calls') and not option_chain.calls.empty:
                        for _, call in option_chain.calls.iterrows():
                            options_data['calls'].append(OptionsContract(
                                symbol=symbol,
                                option_type='call',
                                strike=call.get('strike', 0),
                                expiration=expiration,
                                last_price=call.get('lastPrice', 0),
                                bid=call.get('bid', 0),
                                ask=call.get('ask', 0),
                                volume=call.get('volume', 0) or 0,
                                open_interest=call.get('openInterest', 0) or 0,
                                implied_volatility=call.get('impliedVolatility', 0) or 0
                            ))
                    
                    # Process puts
                    if hasattr(option_chain, 'puts') and not option_chain.puts.empty:
                        for _, put in option_chain.puts.iterrows():
                            options_data['puts'].append(OptionsContract(
                                symbol=symbol,
                                option_type='put',
                                strike=put.get('strike', 0),
                                expiration=expiration,
                                last_price=put.get('lastPrice', 0),
                                bid=put.get('bid', 0),
                                ask=put.get('ask', 0),
                                volume=put.get('volume', 0) or 0,
                                open_interest=put.get('openInterest', 0) or 0,
                                implied_volatility=put.get('impliedVolatility', 0) or 0
                            ))
                            
                except Exception as e:
                    logger.warning(f"Error getting options for expiration {expiration}: {e}")
                    continue
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return {}
    
    async def _calculate_options_sentiment(
        self,
        symbol: str,
        options_data: Dict
    ) -> OptionsFlowSentiment:
        """Calculate options sentiment based on flow analysis"""
        
        calls = options_data.get('calls', [])
        puts = options_data.get('puts', [])
        underlying_price = options_data.get('underlying_price', 0)
        
        if not calls and not puts:
            return OptionsFlowSentiment(
                sentiment_score=0,
                confidence=0,
                put_call_ratio=1,
                call_volume_ratio=0.5,
                unusual_activity_score=0,
                iv_trend='stable',
                flow_direction='neutral',
                key_levels=[],
                insights=[]
            )
        
        # Calculate volume metrics
        total_call_volume = sum(call.volume for call in calls)
        total_put_volume = sum(put.volume for put in puts)
        total_volume = total_call_volume + total_put_volume
        
        # Put/Call ratio
        put_call_ratio = total_put_volume / max(total_call_volume, 1)
        call_volume_ratio = total_call_volume / max(total_volume, 1)
        
        # Analyze money flow direction
        sentiment_score = 0
        flow_direction = 'neutral'
        
        if underlying_price > 0:
            # ITM/OTM analysis
            itm_call_volume = sum(call.volume for call in calls if call.strike < underlying_price)
            otm_call_volume = sum(call.volume for call in calls if call.strike > underlying_price)
            itm_put_volume = sum(put.volume for put in puts if put.strike > underlying_price)
            otm_put_volume = sum(put.volume for put in puts if put.strike < underlying_price)
            
            # Bullish signals
            bullish_signals = 0
            if otm_call_volume > itm_call_volume * 1.5:  # Heavy OTM call buying
                bullish_signals += 1
                sentiment_score += 0.3
            if put_call_ratio < 0.7:  # Low put/call ratio
                bullish_signals += 1
                sentiment_score += 0.2
            if itm_put_volume < otm_put_volume * 0.5:  # Limited ITM put buying
                bullish_signals += 1
                sentiment_score += 0.1
                
            # Bearish signals
            bearish_signals = 0
            if itm_put_volume > otm_put_volume * 1.5:  # Heavy ITM put buying
                bearish_signals += 1
                sentiment_score -= 0.3
            if put_call_ratio > 1.5:  # High put/call ratio
                bearish_signals += 1
                sentiment_score -= 0.2
            if otm_call_volume < itm_call_volume * 0.5:  # Limited OTM call buying
                bearish_signals += 1
                sentiment_score -= 0.1
            
            # Determine flow direction
            if bullish_signals > bearish_signals and sentiment_score > 0.2:
                flow_direction = 'bullish'
            elif bearish_signals > bullish_signals and sentiment_score < -0.2:
                flow_direction = 'bearish'
        
        # Analyze implied volatility trend
        iv_trend = self._analyze_iv_trend(calls + puts)
        
        # Calculate unusual activity score
        unusual_activity_score = self._calculate_unusual_activity_score(calls + puts)
        
        # Calculate confidence based on volume and consistency
        confidence = min(1.0, total_volume / 1000) * 0.5  # Volume component
        if abs(sentiment_score) > 0.3:  # Strong signal component
            confidence += 0.3
        if unusual_activity_score > 2:  # Unusual activity component
            confidence += 0.2
        confidence = min(1.0, confidence)
        
        # Generate insights
        insights = []
        if put_call_ratio < 0.5:
            insights.append("Very low put/call ratio indicates bullish sentiment")
        elif put_call_ratio > 2.0:
            insights.append("Very high put/call ratio indicates bearish sentiment")
            
        if unusual_activity_score > 3:
            insights.append("Significant unusual options activity detected")
            
        if total_volume > 10000:
            insights.append(f"High options volume: {total_volume:,} contracts")
        elif total_volume < 100:
            insights.append("Low options volume - limited liquidity")
        
        return OptionsFlowSentiment(
            sentiment_score=max(-1, min(1, sentiment_score)),
            confidence=confidence,
            put_call_ratio=put_call_ratio,
            call_volume_ratio=call_volume_ratio,
            unusual_activity_score=unusual_activity_score,
            iv_trend=iv_trend,
            flow_direction=flow_direction,
            key_levels=self._identify_key_strikes(options_data),
            insights=insights
        )
    
    def _analyze_iv_trend(self, contracts: List[OptionsContract]) -> str:
        """Analyze implied volatility trend"""
        if not contracts:
            return 'stable'
        
        # Group by expiration and calculate average IV
        iv_by_expiration = {}
        for contract in contracts:
            if contract.implied_volatility > 0:
                if contract.expiration not in iv_by_expiration:
                    iv_by_expiration[contract.expiration] = []
                iv_by_expiration[contract.expiration].append(contract.implied_volatility)
        
        # Calculate average IV for each expiration
        avg_ivs = []
        for expiration in sorted(iv_by_expiration.keys()):
            avg_iv = np.mean(iv_by_expiration[expiration])
            avg_ivs.append(avg_iv)
        
        if len(avg_ivs) < 2:
            return 'stable'
        
        # Analyze trend
        if avg_ivs[-1] > avg_ivs[0] * 1.1:  # 10% increase
            return 'rising'
        elif avg_ivs[-1] < avg_ivs[0] * 0.9:  # 10% decrease
            return 'falling'
        else:
            return 'stable'
    
    def _calculate_unusual_activity_score(self, contracts: List[OptionsContract]) -> float:
        """Calculate unusual activity score based on volume and open interest"""
        if not contracts:
            return 0
        
        unusual_score = 0
        
        for contract in contracts:
            if contract.volume > 0 and contract.open_interest > 0:
                # Volume to open interest ratio
                vol_oi_ratio = contract.volume / contract.open_interest
                
                # High volume relative to open interest is unusual
                if vol_oi_ratio > 2.0:  # Volume > 2x open interest
                    unusual_score += 2
                elif vol_oi_ratio > 1.0:  # Volume > open interest
                    unusual_score += 1
            
            # Very high absolute volume
            if contract.volume > 1000:
                unusual_score += 1
            elif contract.volume > 5000:
                unusual_score += 2
        
        return unusual_score
    
    async def _detect_unusual_options_activity(
        self,
        symbol: str,
        options_data: Dict,
        min_volume: int
    ) -> List[UnusualActivity]:
        """Detect unusual options activity"""
        unusual_activities = []
        
        calls = options_data.get('calls', [])
        puts = options_data.get('puts', [])
        all_contracts = calls + puts
        
        # Filter by minimum volume
        active_contracts = [c for c in all_contracts if c.volume >= min_volume]
        
        if not active_contracts:
            return unusual_activities
        
        # Calculate volume statistics for comparison
        volumes = [c.volume for c in active_contracts]
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        
        # Calculate open interest statistics
        open_interests = [c.open_interest for c in active_contracts if c.open_interest > 0]
        oi_mean = np.mean(open_interests) if open_interests else 0
        
        for contract in active_contracts:
            # Volume spike detection
            if contract.volume > volume_mean + 2 * volume_std and contract.volume > 500:
                magnitude = contract.volume / max(volume_mean, 1)
                unusual_activities.append(UnusualActivity(
                    symbol=symbol,
                    contract=contract,
                    activity_type='volume_spike',
                    magnitude=magnitude,
                    description=f"Unusually high {contract.option_type} volume at ${contract.strike} strike",
                    timestamp=datetime.now()
                ))
            
            # High volume relative to open interest
            if contract.open_interest > 0:
                vol_oi_ratio = contract.volume / contract.open_interest
                if vol_oi_ratio > 3.0:  # Volume > 3x open interest
                    unusual_activities.append(UnusualActivity(
                        symbol=symbol,
                        contract=contract,
                        activity_type='flow_direction',
                        magnitude=vol_oi_ratio,
                        description=f"Heavy {contract.option_type} flow at ${contract.strike} ({vol_oi_ratio:.1f}x OI)",
                        timestamp=datetime.now()
                    ))
            
            # Implied volatility spike detection
            if contract.implied_volatility > 0.5:  # IV > 50%
                unusual_activities.append(UnusualActivity(
                    symbol=symbol,
                    contract=contract,
                    activity_type='iv_spike',
                    magnitude=contract.implied_volatility,
                    description=f"High implied volatility {contract.option_type} at ${contract.strike} ({contract.implied_volatility:.1%} IV)",
                    timestamp=datetime.now()
                ))
        
        # Sort by magnitude (most unusual first)
        unusual_activities.sort(key=lambda x: x.magnitude, reverse=True)
        
        return unusual_activities[:10]  # Return top 10 most unusual activities
    
    def _identify_key_strikes(self, options_data: Dict) -> List[float]:
        """Identify key strike levels based on volume and open interest"""
        calls = options_data.get('calls', [])
        puts = options_data.get('puts', [])
        
        # Combine volume and open interest data by strike
        strike_data = {}
        
        for contract in calls + puts:
            strike = contract.strike
            if strike not in strike_data:
                strike_data[strike] = {'volume': 0, 'open_interest': 0, 'contracts': 0}
            
            strike_data[strike]['volume'] += contract.volume
            strike_data[strike]['open_interest'] += contract.open_interest
            strike_data[strike]['contracts'] += 1
        
        # Score each strike level
        strike_scores = []
        for strike, data in strike_data.items():
            # Weight volume more heavily than open interest
            score = data['volume'] * 2 + data['open_interest']
            strike_scores.append((strike, score))
        
        # Sort by score and return top strikes
        strike_scores.sort(key=lambda x: x[1], reverse=True)
        key_strikes = [strike for strike, score in strike_scores[:5]]
        
        return sorted(key_strikes)
    
    async def _analyze_volatility(self, symbol: str) -> Dict:
        """Analyze volatility metrics and VIX correlation"""
        try:
            # Get VIX data for market volatility context
            vix_data = await self._get_vix_data()
            
            # Get stock's historical volatility
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d')
            
            volatility_analysis = {
                'current_vix': vix_data.get('current_vix'),
                'vix_trend': vix_data.get('trend'),
                'historical_volatility': None,
                'volatility_rank': None,
                'volatility_trend': 'stable'
            }
            
            if not hist.empty:
                # Calculate 30-day historical volatility
                returns = hist['Close'].pct_change().dropna()
                hist_vol = returns.std() * np.sqrt(252)  # Annualized
                volatility_analysis['historical_volatility'] = hist_vol
                
                # Calculate recent vs longer-term volatility
                recent_returns = returns[-10:]  # Last 10 days
                recent_vol = recent_returns.std() * np.sqrt(252)
                
                if recent_vol > hist_vol * 1.2:
                    volatility_analysis['volatility_trend'] = 'rising'
                elif recent_vol < hist_vol * 0.8:
                    volatility_analysis['volatility_trend'] = 'falling'
            
            return volatility_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}
    
    async def _get_vix_data(self) -> Dict:
        """Get VIX data from Yahoo Finance"""
        await self.yahoo_limiter.acquire()
        
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period='5d')
            
            if hist.empty:
                return {}
            
            current_vix = hist['Close'].iloc[-1]
            
            # Determine VIX trend
            if len(hist) >= 2:
                prev_vix = hist['Close'].iloc[-2]
                if current_vix > prev_vix * 1.05:
                    trend = 'rising'
                elif current_vix < prev_vix * 0.95:
                    trend = 'falling'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            return {
                'current_vix': current_vix,
                'trend': trend,
                'level': 'high' if current_vix > 25 else 'low' if current_vix < 15 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return {}
    
    def _generate_options_insights(
        self,
        sentiment: Optional[OptionsFlowSentiment],
        unusual_activity: List[UnusualActivity],
        key_strikes: List[float],
        volatility_analysis: Dict
    ) -> List[str]:
        """Generate actionable insights from options analysis"""
        insights = []
        
        if not sentiment:
            return ["Insufficient options data for analysis"]
        
        # Sentiment insights
        if sentiment.confidence > 0.6:
            if sentiment.sentiment_score > 0.3:
                insights.append(f"Bullish options flow detected (score: {sentiment.sentiment_score:.2f})")
            elif sentiment.sentiment_score < -0.3:
                insights.append(f"Bearish options flow detected (score: {sentiment.sentiment_score:.2f})")
        
        # Put/call ratio insights
        if sentiment.put_call_ratio < 0.5:
            insights.append(f"Very bullish put/call ratio: {sentiment.put_call_ratio:.2f}")
        elif sentiment.put_call_ratio > 2.0:
            insights.append(f"Very bearish put/call ratio: {sentiment.put_call_ratio:.2f}")
        
        # Flow direction insights
        if sentiment.flow_direction != 'neutral':
            insights.append(f"Options flow direction: {sentiment.flow_direction}")
        
        # Unusual activity insights
        if unusual_activity:
            insights.append(f"{len(unusual_activity)} unusual options activities detected")
            
            # Highlight most significant activities
            for activity in unusual_activity[:3]:  # Top 3
                insights.append(f"• {activity.description}")
        
        # Key strike insights
        if key_strikes:
            insights.append(f"Key strike levels: {', '.join(f'${strike:.0f}' for strike in key_strikes[:3])}")
        
        # Volatility insights
        vix_level = volatility_analysis.get('level')
        if vix_level == 'high':
            insights.append("High market volatility environment (VIX > 25)")
        elif vix_level == 'low':
            insights.append("Low market volatility environment (VIX < 15)")
        
        vol_trend = volatility_analysis.get('volatility_trend')
        if vol_trend == 'rising':
            insights.append("Stock volatility is increasing")
        elif vol_trend == 'falling':
            insights.append("Stock volatility is decreasing")
        
        # IV trend insights
        if sentiment.iv_trend == 'rising':
            insights.append("Implied volatility is increasing")
        elif sentiment.iv_trend == 'falling':
            insights.append("Implied volatility is decreasing")
        
        # Add specific insights from sentiment
        insights.extend(sentiment.insights)
        
        return insights or ["No significant options flow patterns detected"]
    
    async def get_options_alerts(self, symbol: str, thresholds: Dict) -> List[Dict]:
        """Generate options flow alerts"""
        analysis = await self.analyze_options_flow(symbol)
        
        alerts = []
        
        if 'error' in analysis:
            return alerts
            
        sentiment = analysis.get('sentiment')
        unusual_activity = analysis.get('unusual_activity', [])
        
        if sentiment:
            # High conviction flow alerts
            if sentiment.confidence > thresholds.get('confidence_threshold', 0.7):
                if sentiment.sentiment_score > thresholds.get('bullish_threshold', 0.4):
                    alerts.append({
                        'type': 'bullish_options_flow',
                        'message': f'Strong bullish options flow detected for {symbol}',
                        'sentiment_score': sentiment.sentiment_score,
                        'confidence': sentiment.confidence,
                        'put_call_ratio': sentiment.put_call_ratio,
                        'urgency': 'high'
                    })
                elif sentiment.sentiment_score < thresholds.get('bearish_threshold', -0.4):
                    alerts.append({
                        'type': 'bearish_options_flow',
                        'message': f'Strong bearish options flow detected for {symbol}',
                        'sentiment_score': sentiment.sentiment_score,
                        'confidence': sentiment.confidence,
                        'put_call_ratio': sentiment.put_call_ratio,
                        'urgency': 'high'
                    })
            
            # Extreme put/call ratio alerts
            if sentiment.put_call_ratio > thresholds.get('high_pcr_threshold', 3.0):
                alerts.append({
                    'type': 'extreme_put_call_ratio',
                    'message': f'Extremely high put/call ratio for {symbol}: {sentiment.put_call_ratio:.2f}',
                    'put_call_ratio': sentiment.put_call_ratio,
                    'urgency': 'medium'
                })
            elif sentiment.put_call_ratio < thresholds.get('low_pcr_threshold', 0.3):
                alerts.append({
                    'type': 'extreme_put_call_ratio',
                    'message': f'Extremely low put/call ratio for {symbol}: {sentiment.put_call_ratio:.2f}',
                    'put_call_ratio': sentiment.put_call_ratio,
                    'urgency': 'medium'
                })
        
        # Unusual activity alerts
        high_magnitude_activities = [
            activity for activity in unusual_activity 
            if activity.magnitude > thresholds.get('unusual_magnitude_threshold', 5.0)
        ]
        
        if high_magnitude_activities:
            alerts.append({
                'type': 'unusual_options_activity',
                'message': f'Unusual options activity detected for {symbol}',
                'activities': [
                    {
                        'type': activity.activity_type,
                        'description': activity.description,
                        'magnitude': activity.magnitude
                    } for activity in high_magnitude_activities[:3]
                ],
                'urgency': 'medium'
            })
        
        return alerts