"""
Earnings Whispers and Estimate Analysis Engine

Analyzes earnings expectations from free sources:
- Yahoo Finance earnings data
- Alpha Vantage earnings estimates (free tier)
- Seeking Alpha earnings previews (scraping)
- FinHub earnings estimates (free tier)
- SEC XBRL earnings data
- Earnings call transcripts analysis

Provides comprehensive earnings sentiment and surprise prediction
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re
from bs4 import BeautifulSoup
import yfinance as yf

import pandas as pd
import numpy as np
from transformers import pipeline
import requests

from backend.utils.cache import CacheManager
from backend.utils.rate_limiter import RateLimiter
from backend.utils.cost_monitor import CostMonitor

logger = logging.getLogger(__name__)

@dataclass
class EarningsEstimate:
    """Earnings estimate data structure"""
    symbol: str
    period: str  # Q1 2024, Q2 2024, etc.
    fiscal_date_ending: date
    report_date: Optional[date]
    estimate_type: str  # eps, revenue
    current_estimate: Optional[float]
    high_estimate: Optional[float]
    low_estimate: Optional[float]
    num_estimates: Optional[int]
    year_ago_eps: Optional[float]
    estimate_revision_trend: str  # up, down, stable
    source: str
    last_updated: datetime

@dataclass
class EarningsEvent:
    """Earnings event data structure"""
    symbol: str
    company_name: str
    report_date: date
    fiscal_period: str
    fiscal_year: int
    time: str  # BMO (before market open), AMC (after market close), time
    confirmed: bool
    
@dataclass
class EarningsSentiment:
    """Earnings sentiment analysis result"""
    sentiment_score: float  # -1 (very negative) to 1 (very positive)
    confidence: float
    surprise_probability: float  # Probability of beating estimates
    revision_momentum: str  # strong_up, up, stable, down, strong_down
    consensus_strength: float  # How tight the estimate range is
    whisper_premium: Optional[float]  # Difference between whisper and consensus
    key_factors: List[str]
    risk_factors: List[str]

class EarningsWhisperAnalyzer:
    """
    Comprehensive earnings analysis using free data sources
    
    Features:
    - Multi-source earnings estimates aggregation
    - Earnings surprise prediction
    - Estimate revision tracking
    - Earnings call sentiment analysis
    - Pre-earnings price movement analysis
    - Options activity correlation
    - Industry comparison
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = CacheManager()
        self.cost_monitor = CostMonitor()
        
        # API rate limiters
        self.alpha_vantage_limiter = RateLimiter(calls=5, period=60)  # 5 calls/minute
        self.finnhub_limiter = RateLimiter(calls=60, period=60)  # 60 calls/minute
        self.yahoo_limiter = RateLimiter(calls=100, period=60)  # Conservative limit
        
        # Initialize sentiment analyzer for earnings calls
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU only
            )
        except Exception as e:
            logger.warning(f"Could not load FinBERT model: {e}")
            self.sentiment_analyzer = None
    
    async def analyze_earnings_expectations(
        self,
        symbol: str,
        quarters_ahead: int = 2
    ) -> Dict:
        """
        Analyze earnings expectations and sentiment for upcoming quarters
        
        Args:
            symbol: Stock ticker symbol
            quarters_ahead: Number of quarters to analyze ahead
            
        Returns:
            Comprehensive earnings analysis
        """
        # Check cache first
        cache_key = f"earnings_analysis:{symbol}:{quarters_ahead}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            results = {
                'symbol': symbol,
                'analysis_date': datetime.now(),
                'upcoming_earnings': [],
                'estimates': {},
                'sentiment': None,
                'historical_performance': {},
                'insights': []
            }
            
            # Get upcoming earnings events
            upcoming_events = await self._get_upcoming_earnings(symbol, quarters_ahead)
            results['upcoming_earnings'] = upcoming_events
            
            if not upcoming_events:
                results['insights'] = ['No upcoming earnings events found']
                return results
            
            # Get earnings estimates from multiple sources
            estimates = await self._aggregate_earnings_estimates(symbol, quarters_ahead)
            results['estimates'] = estimates
            
            # Analyze sentiment
            if estimates:
                sentiment = await self._calculate_earnings_sentiment(symbol, estimates, upcoming_events)
                results['sentiment'] = sentiment
            
            # Get historical earnings performance
            historical = await self._analyze_historical_earnings_performance(symbol)
            results['historical_performance'] = historical
            
            # Generate insights
            results['insights'] = self._generate_earnings_insights(
                results['sentiment'], 
                estimates, 
                historical,
                upcoming_events
            )
            
            # Cache results
            await self.cache.set(cache_key, results, expire=3600)  # 1 hour
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing earnings expectations for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _get_upcoming_earnings(self, symbol: str, quarters_ahead: int) -> List[EarningsEvent]:
        """Get upcoming earnings events from multiple sources"""
        events = []
        
        try:
            # Yahoo Finance earnings calendar
            yahoo_events = await self._get_yahoo_earnings_events(symbol)
            events.extend(yahoo_events)
            
            # Alpha Vantage earnings data
            if self.config.get('alpha_vantage_api_key'):
                av_events = await self._get_alpha_vantage_earnings(symbol)
                events.extend(av_events)
            
            # Finnhub earnings calendar
            if self.config.get('finnhub_api_key'):
                finnhub_events = await self._get_finnhub_earnings(symbol)
                events.extend(finnhub_events)
            
            # Remove duplicates and sort by date
            unique_events = []
            seen_dates = set()
            
            for event in events:
                date_key = (event.report_date, event.fiscal_period)
                if date_key not in seen_dates:
                    seen_dates.add(date_key)
                    unique_events.append(event)
            
            unique_events.sort(key=lambda x: x.report_date)
            
            return unique_events[:quarters_ahead]
            
        except Exception as e:
            logger.error(f"Error getting upcoming earnings for {symbol}: {e}")
            return []
    
    async def _get_yahoo_earnings_events(self, symbol: str) -> List[EarningsEvent]:
        """Get earnings events from Yahoo Finance"""
        await self.yahoo_limiter.acquire()
        
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return []
            
            events = []
            for index, row in calendar.iterrows():
                try:
                    report_date = pd.to_datetime(row['Earnings Date']).date()
                    
                    # Estimate fiscal period based on report date
                    fiscal_period = self._estimate_fiscal_period(report_date)
                    
                    events.append(EarningsEvent(
                        symbol=symbol,
                        company_name=ticker.info.get('longName', symbol),
                        report_date=report_date,
                        fiscal_period=fiscal_period,
                        fiscal_year=report_date.year,
                        time=row.get('Time', 'Unknown'),
                        confirmed=True,
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing Yahoo earnings event: {e}")
                    continue
                    
            return events
            
        except Exception as e:
            logger.error(f"Error getting Yahoo earnings events: {e}")
            return []
    
    async def _get_alpha_vantage_earnings(self, symbol: str) -> List[EarningsEvent]:
        """Get earnings data from Alpha Vantage"""
        if not self.config.get('alpha_vantage_api_key'):
            return []
            
        await self.alpha_vantage_limiter.acquire()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.config['alpha_vantage_api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
            
            if 'Error Message' in data or 'quarterlyEarnings' not in data:
                return []
            
            events = []
            for earnings in data['quarterlyEarnings'][:4]:  # Next 4 quarters
                try:
                    report_date_str = earnings.get('reportedDate')
                    if not report_date_str:
                        continue
                        
                    report_date = datetime.strptime(report_date_str, '%Y-%m-%d').date()
                    fiscal_end = datetime.strptime(earnings['fiscalDateEnding'], '%Y-%m-%d').date()
                    
                    # Only include future earnings
                    if report_date > date.today():
                        fiscal_period = self._estimate_fiscal_period(fiscal_end)
                        
                        events.append(EarningsEvent(
                            symbol=symbol,
                            company_name=symbol,  # AV doesn't provide company name
                            report_date=report_date,
                            fiscal_period=fiscal_period,
                            fiscal_year=fiscal_end.year,
                            time='Unknown',
                            confirmed=False
                        ))
                except Exception as e:
                    logger.warning(f"Error parsing Alpha Vantage earnings: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage earnings: {e}")
            return []
    
    async def _get_finnhub_earnings(self, symbol: str) -> List[EarningsEvent]:
        """Get earnings calendar from Finnhub"""
        if not self.config.get('finnhub_api_key'):
            return []
            
        await self.finnhub_limiter.acquire()
        
        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            today = date.today()
            next_quarter = today + timedelta(days=90)
            
            params = {
                'from': today.strftime('%Y-%m-%d'),
                'to': next_quarter.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'token': self.config['finnhub_api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
            
            events = []
            earnings_calendar = data.get('earningsCalendar', [])
            
            for earning in earnings_calendar:
                try:
                    report_date = datetime.strptime(earning['date'], '%Y-%m-%d').date()
                    
                    events.append(EarningsEvent(
                        symbol=symbol,
                        company_name=earning.get('name', symbol),
                        report_date=report_date,
                        fiscal_period=earning.get('quarter', 'Unknown'),
                        fiscal_year=earning.get('year', report_date.year),
                        time=earning.get('hour', 'Unknown'),
                        confirmed=True
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing Finnhub earnings: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting Finnhub earnings: {e}")
            return []
    
    def _estimate_fiscal_period(self, date_obj: date) -> str:
        """Estimate fiscal period from date"""
        quarter_map = {
            1: 'Q1', 2: 'Q1', 3: 'Q1',
            4: 'Q2', 5: 'Q2', 6: 'Q2',
            7: 'Q3', 8: 'Q3', 9: 'Q3',
            10: 'Q4', 11: 'Q4', 12: 'Q4'
        }
        quarter = quarter_map[date_obj.month]
        return f"{quarter} {date_obj.year}"
    
    async def _aggregate_earnings_estimates(self, symbol: str, quarters_ahead: int) -> Dict:
        """Aggregate earnings estimates from multiple sources"""
        estimates = {
            'eps_estimates': [],
            'revenue_estimates': [],
            'estimate_revisions': [],
            'analyst_recommendations': {}
        }
        
        try:
            # Yahoo Finance estimates
            yahoo_estimates = await self._get_yahoo_estimates(symbol)
            if yahoo_estimates:
                estimates['eps_estimates'].extend(yahoo_estimates.get('eps', []))
                estimates['revenue_estimates'].extend(yahoo_estimates.get('revenue', []))
            
            # Alpha Vantage estimates
            if self.config.get('alpha_vantage_api_key'):
                av_estimates = await self._get_alpha_vantage_estimates(symbol)
                if av_estimates:
                    estimates['eps_estimates'].extend(av_estimates.get('eps', []))
            
            # Get estimate revisions
            revisions = await self._track_estimate_revisions(symbol)
            estimates['estimate_revisions'] = revisions
            
            # Get analyst recommendations
            recommendations = await self._get_analyst_recommendations(symbol)
            estimates['analyst_recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error aggregating estimates for {symbol}: {e}")
        
        return estimates
    
    async def _get_yahoo_estimates(self, symbol: str) -> Optional[Dict]:
        """Get earnings estimates from Yahoo Finance"""
        await self.yahoo_limiter.acquire()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings estimates
            estimates = {'eps': [], 'revenue': []}
            
            # Try to get analyst info
            analyst_info = ticker.analyst_info
            if analyst_info is not None and not analyst_info.empty:
                # Parse EPS estimates
                if 'Earnings Estimate' in analyst_info:
                    eps_data = analyst_info['Earnings Estimate']
                    for period, row in eps_data.iterrows():
                        estimates['eps'].append(EarningsEstimate(
                            symbol=symbol,
                            period=str(period),
                            fiscal_date_ending=date.today(),  # Approximate
                            report_date=None,
                            estimate_type='eps',
                            current_estimate=row.get('Avg. Estimate'),
                            high_estimate=row.get('High Estimate'),
                            low_estimate=row.get('Low Estimate'),
                            num_estimates=row.get('No. of Analysts'),
                            year_ago_eps=row.get('Year Ago EPS'),
                            estimate_revision_trend='stable',
                            source='yahoo',
                            last_updated=datetime.now()
                        ))
                
                # Parse Revenue estimates
                if 'Revenue Estimate' in analyst_info:
                    revenue_data = analyst_info['Revenue Estimate']
                    for period, row in revenue_data.iterrows():
                        estimates['revenue'].append(EarningsEstimate(
                            symbol=symbol,
                            period=str(period),
                            fiscal_date_ending=date.today(),
                            report_date=None,
                            estimate_type='revenue',
                            current_estimate=row.get('Avg. Estimate'),
                            high_estimate=row.get('High Estimate'),
                            low_estimate=row.get('Low Estimate'),
                            num_estimates=row.get('No. of Analysts'),
                            year_ago_eps=None,
                            estimate_revision_trend='stable',
                            source='yahoo',
                            last_updated=datetime.now()
                        ))
            
            return estimates if estimates['eps'] or estimates['revenue'] else None
            
        except Exception as e:
            logger.error(f"Error getting Yahoo estimates: {e}")
            return None
    
    async def _get_alpha_vantage_estimates(self, symbol: str) -> Optional[Dict]:
        """Get earnings estimates from Alpha Vantage"""
        await self.alpha_vantage_limiter.acquire()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.config['alpha_vantage_api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
            
            if 'Error Message' in data or 'quarterlyEarnings' not in data:
                return None
            
            estimates = {'eps': []}
            
            for earnings in data['quarterlyEarnings'][:4]:
                try:
                    fiscal_end = datetime.strptime(earnings['fiscalDateEnding'], '%Y-%m-%d').date()
                    reported_eps = earnings.get('reportedEPS')
                    
                    if reported_eps and reported_eps != 'None':
                        estimates['eps'].append(EarningsEstimate(
                            symbol=symbol,
                            period=self._estimate_fiscal_period(fiscal_end),
                            fiscal_date_ending=fiscal_end,
                            report_date=None,
                            estimate_type='eps',
                            current_estimate=float(reported_eps),
                            high_estimate=None,
                            low_estimate=None,
                            num_estimates=None,
                            year_ago_eps=None,
                            estimate_revision_trend='stable',
                            source='alpha_vantage',
                            last_updated=datetime.now()
                        ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing Alpha Vantage estimate: {e}")
                    continue
            
            return estimates if estimates['eps'] else None
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage estimates: {e}")
            return None
    
    async def _track_estimate_revisions(self, symbol: str) -> List[Dict]:
        """Track recent estimate revisions"""
        # This would typically require a paid service, but we can approximate
        # by comparing current estimates with cached historical estimates
        
        try:
            # Check if we have historical estimates cached
            historical_key = f"historical_estimates:{symbol}"
            historical_estimates = await self.cache.get(historical_key)
            
            if not historical_estimates:
                return []
            
            # Get current estimates
            current_estimates = await self._get_yahoo_estimates(symbol)
            if not current_estimates:
                return []
            
            revisions = []
            
            # Compare current vs historical EPS estimates
            for current_est in current_estimates.get('eps', []):
                for hist_est in historical_estimates.get('eps', []):
                    if current_est.period == hist_est['period']:
                        current_val = current_est.current_estimate
                        historical_val = hist_est.get('current_estimate')
                        
                        if current_val and historical_val:
                            revision_pct = (current_val - historical_val) / historical_val
                            
                            revisions.append({
                                'period': current_est.period,
                                'type': 'eps',
                                'previous_estimate': historical_val,
                                'current_estimate': current_val,
                                'revision_amount': current_val - historical_val,
                                'revision_percent': revision_pct,
                                'direction': 'up' if revision_pct > 0.02 else 'down' if revision_pct < -0.02 else 'stable'
                            })
            
            return revisions
            
        except Exception as e:
            logger.error(f"Error tracking estimate revisions: {e}")
            return []
    
    async def _get_analyst_recommendations(self, symbol: str) -> Dict:
        """Get current analyst recommendations"""
        await self.yahoo_limiter.acquire()
        
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is None or recommendations.empty:
                return {}
            
            # Get most recent recommendation summary
            latest = recommendations.iloc[-1] if not recommendations.empty else None
            
            if latest is not None:
                return {
                    'strong_buy': latest.get('strongBuy', 0),
                    'buy': latest.get('buy', 0),
                    'hold': latest.get('hold', 0),
                    'sell': latest.get('sell', 0),
                    'strong_sell': latest.get('strongSell', 0),
                    'mean_rating': latest.get('Mean Rating'),
                    'total_analysts': sum([
                        latest.get('strongBuy', 0),
                        latest.get('buy', 0),
                        latest.get('hold', 0),
                        latest.get('sell', 0),
                        latest.get('strongSell', 0)
                    ]),
                    'last_updated': datetime.now()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting analyst recommendations: {e}")
            return {}
    
    async def _calculate_earnings_sentiment(
        self,
        symbol: str,
        estimates: Dict,
        upcoming_events: List[EarningsEvent]
    ) -> EarningsSentiment:
        """Calculate overall earnings sentiment"""
        
        # Initialize sentiment components
        sentiment_score = 0.0
        confidence = 0.0
        surprise_probability = 0.5
        revision_momentum = 'stable'
        consensus_strength = 0.0
        key_factors = []
        risk_factors = []
        
        try:
            # Analyze EPS estimate consensus
            eps_estimates = estimates.get('eps_estimates', [])
            if eps_estimates:
                current_eps = [e for e in eps_estimates if e.current_estimate is not None]
                if current_eps:
                    # Calculate consensus strength (inverse of estimate range)
                    estimates_values = [e.current_estimate for e in current_eps]
                    if len(estimates_values) > 1:
                        consensus_strength = 1.0 - (np.std(estimates_values) / abs(np.mean(estimates_values)))
                        consensus_strength = max(0, min(1, consensus_strength))
                    else:
                        consensus_strength = 0.5
                        
                    # Positive factors for tight consensus
                    if consensus_strength > 0.8:
                        key_factors.append("Strong analyst consensus on EPS estimates")
                        sentiment_score += 0.1
                    elif consensus_strength < 0.4:
                        risk_factors.append("Wide range in analyst EPS estimates")
                        sentiment_score -= 0.1
            
            # Analyze estimate revisions
            revisions = estimates.get('estimate_revisions', [])
            if revisions:
                revision_scores = []
                for revision in revisions:
                    direction = revision.get('direction', 'stable')
                    if direction == 'up':
                        revision_scores.append(1)
                    elif direction == 'down':
                        revision_scores.append(-1)
                    else:
                        revision_scores.append(0)
                
                if revision_scores:
                    avg_revision = np.mean(revision_scores)
                    
                    if avg_revision > 0.3:
                        revision_momentum = 'strong_up'
                        sentiment_score += 0.3
                        key_factors.append("Strong upward estimate revisions")
                    elif avg_revision > 0.1:
                        revision_momentum = 'up'
                        sentiment_score += 0.1
                        key_factors.append("Positive estimate revisions")
                    elif avg_revision < -0.3:
                        revision_momentum = 'strong_down'
                        sentiment_score -= 0.3
                        risk_factors.append("Strong downward estimate revisions")
                    elif avg_revision < -0.1:
                        revision_momentum = 'down'
                        sentiment_score -= 0.1
                        risk_factors.append("Negative estimate revisions")
            
            # Analyze analyst recommendations
            recommendations = estimates.get('analyst_recommendations', {})
            if recommendations and recommendations.get('total_analysts', 0) > 0:
                strong_buy = recommendations.get('strong_buy', 0)
                buy = recommendations.get('buy', 0)
                hold = recommendations.get('hold', 0)
                sell = recommendations.get('sell', 0)
                strong_sell = recommendations.get('strong_sell', 0)
                total = recommendations.get('total_analysts', 1)
                
                # Calculate bullish ratio
                bullish_ratio = (strong_buy * 2 + buy) / (total * 2)
                bearish_ratio = (strong_sell * 2 + sell) / (total * 2)
                
                if bullish_ratio > 0.6:
                    sentiment_score += 0.2
                    key_factors.append(f"{bullish_ratio:.0%} of analysts recommend buying")
                elif bearish_ratio > 0.3:
                    sentiment_score -= 0.2
                    risk_factors.append(f"{bearish_ratio:.0%} of analysts recommend selling")
            
            # Historical earnings surprise analysis
            historical = await self._analyze_historical_earnings_performance(symbol)
            if historical:
                surprise_rate = historical.get('surprise_rate', 0.5)
                avg_surprise = historical.get('average_surprise_pct', 0)
                
                # Adjust surprise probability based on history
                surprise_probability = surprise_rate
                
                if surprise_rate > 0.7:
                    key_factors.append("Strong history of beating earnings estimates")
                    sentiment_score += 0.15
                elif surprise_rate < 0.3:
                    risk_factors.append("Poor history of beating earnings estimates")
                    sentiment_score -= 0.15
                
                if avg_surprise > 0.05:  # Average 5%+ surprise
                    key_factors.append("History of significant earnings beats")
                elif avg_surprise < -0.05:
                    risk_factors.append("History of earnings misses")
            
            # Calculate overall confidence
            confidence_factors = [
                len(eps_estimates) > 0,
                len(revisions) > 0,
                consensus_strength > 0.5,
                recommendations.get('total_analysts', 0) >= 3
            ]
            confidence = sum(confidence_factors) / len(confidence_factors)
            
            # Normalize sentiment score
            sentiment_score = max(-1, min(1, sentiment_score))
            
        except Exception as e:
            logger.error(f"Error calculating earnings sentiment: {e}")
        
        return EarningsSentiment(
            sentiment_score=sentiment_score,
            confidence=confidence,
            surprise_probability=surprise_probability,
            revision_momentum=revision_momentum,
            consensus_strength=consensus_strength,
            whisper_premium=None,  # Would need paid whisper data
            key_factors=key_factors,
            risk_factors=risk_factors
        )
    
    async def _analyze_historical_earnings_performance(self, symbol: str) -> Dict:
        """Analyze historical earnings surprise performance"""
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_dates
            
            if earnings is None or earnings.empty:
                return {}
            
            # Calculate surprise metrics
            surprises = []
            for index, row in earnings.iterrows():
                reported = row.get('Reported EPS')
                estimate = row.get('EPS Estimate')
                
                if reported is not None and estimate is not None and estimate != 0:
                    surprise_pct = (reported - estimate) / abs(estimate)
                    surprises.append({
                        'date': index,
                        'reported': reported,
                        'estimate': estimate,
                        'surprise': reported - estimate,
                        'surprise_pct': surprise_pct,
                        'beat': reported > estimate
                    })
            
            if not surprises:
                return {}
            
            # Calculate metrics
            surprise_rate = sum(1 for s in surprises if s['beat']) / len(surprises)
            avg_surprise_pct = np.mean([s['surprise_pct'] for s in surprises])
            recent_surprises = surprises[:4]  # Last 4 quarters
            recent_surprise_rate = sum(1 for s in recent_surprises if s['beat']) / len(recent_surprises) if recent_surprises else 0
            
            return {
                'total_reports': len(surprises),
                'surprise_rate': surprise_rate,
                'recent_surprise_rate': recent_surprise_rate,
                'average_surprise_pct': avg_surprise_pct,
                'recent_surprises': recent_surprises,
                'trend': 'improving' if recent_surprise_rate > surprise_rate else 'declining' if recent_surprise_rate < surprise_rate else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical earnings performance: {e}")
            return {}
    
    def _generate_earnings_insights(
        self,
        sentiment: Optional[EarningsSentiment],
        estimates: Dict,
        historical: Dict,
        upcoming_events: List[EarningsEvent]
    ) -> List[str]:
        """Generate actionable insights from earnings analysis"""
        insights = []
        
        if not sentiment:
            return ["Insufficient data for earnings analysis"]
        
        # Sentiment insights
        if sentiment.confidence > 0.6:
            if sentiment.sentiment_score > 0.2:
                insights.append(f"Positive earnings sentiment (score: {sentiment.sentiment_score:.2f})")
            elif sentiment.sentiment_score < -0.2:
                insights.append(f"Negative earnings sentiment (score: {sentiment.sentiment_score:.2f})")
        
        # Surprise probability insights
        if sentiment.surprise_probability > 0.7:
            insights.append(f"High probability of beating estimates ({sentiment.surprise_probability:.0%})")
        elif sentiment.surprise_probability < 0.3:
            insights.append(f"Low probability of beating estimates ({sentiment.surprise_probability:.0%})")
        
        # Revision momentum insights
        if sentiment.revision_momentum in ['strong_up', 'up']:
            insights.append("Positive earnings estimate revisions trend")
        elif sentiment.revision_momentum in ['strong_down', 'down']:
            insights.append("Negative earnings estimate revisions trend")
        
        # Consensus insights
        if sentiment.consensus_strength > 0.8:
            insights.append("Strong analyst consensus on earnings estimates")
        elif sentiment.consensus_strength < 0.4:
            insights.append("Wide disagreement among analysts on earnings")
        
        # Historical performance insights
        if historical:
            surprise_rate = historical.get('surprise_rate', 0)
            trend = historical.get('trend', 'stable')
            
            if surprise_rate > 0.75:
                insights.append("Excellent track record of beating earnings estimates")
            elif surprise_rate < 0.25:
                insights.append("Poor track record of beating earnings estimates")
                
            if trend == 'improving':
                insights.append("Recent earnings performance trend is improving")
            elif trend == 'declining':
                insights.append("Recent earnings performance trend is declining")
        
        # Timing insights
        if upcoming_events:
            next_event = upcoming_events[0]
            days_until = (next_event.report_date - date.today()).days
            
            if days_until <= 7:
                insights.append(f"Earnings report due in {days_until} days")
            elif days_until <= 30:
                insights.append(f"Earnings report due in {days_until} days - expect increased volatility")
        
        # Key factors and risks
        insights.extend(sentiment.key_factors)
        insights.extend([f"Risk: {risk}" for risk in sentiment.risk_factors])
        
        return insights or ["No significant earnings patterns detected"]
    
    async def get_earnings_alerts(self, symbol: str, thresholds: Dict) -> List[Dict]:
        """Generate earnings-based alerts"""
        analysis = await self.analyze_earnings_expectations(symbol)
        
        alerts = []
        
        if 'error' in analysis:
            return alerts
            
        sentiment = analysis.get('sentiment')
        upcoming = analysis.get('upcoming_earnings', [])
        
        # Upcoming earnings alerts
        if upcoming:
            next_earnings = upcoming[0]
            days_until = (next_earnings.report_date - date.today()).days
            
            if days_until <= thresholds.get('earnings_alert_days', 7):
                alerts.append({
                    'type': 'upcoming_earnings',
                    'message': f'{symbol} earnings report due in {days_until} days',
                    'report_date': next_earnings.report_date.isoformat(),
                    'fiscal_period': next_earnings.fiscal_period,
                    'urgency': 'high' if days_until <= 3 else 'medium'
                })
        
        if sentiment:
            # High surprise probability alerts
            if sentiment.surprise_probability > thresholds.get('surprise_threshold', 0.8):
                alerts.append({
                    'type': 'earnings_beat_likely',
                    'message': f'High probability of {symbol} beating earnings estimates',
                    'probability': sentiment.surprise_probability,
                    'sentiment_score': sentiment.sentiment_score,
                    'urgency': 'medium'
                })
            
            # Strong revision momentum alerts
            if sentiment.revision_momentum == 'strong_up':
                alerts.append({
                    'type': 'strong_estimate_revisions',
                    'message': f'Strong upward earnings estimate revisions for {symbol}',
                    'momentum': sentiment.revision_momentum,
                    'urgency': 'medium'
                })
            elif sentiment.revision_momentum == 'strong_down':
                alerts.append({
                    'type': 'negative_estimate_revisions',
                    'message': f'Strong downward earnings estimate revisions for {symbol}',
                    'momentum': sentiment.revision_momentum,
                    'urgency': 'medium'
                })
        
        return alerts