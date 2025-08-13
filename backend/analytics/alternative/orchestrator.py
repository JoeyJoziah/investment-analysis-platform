"""
Alternative Data Orchestrator

Coordinates all alternative data analysis modules:
- Social sentiment analysis
- Insider trading analysis  
- Earnings whispers analysis
- Options flow analysis
- Macro economic indicators
- Supply chain intelligence

Provides unified alternative data scoring and insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from backend.utils.cache import CacheManager
from backend.utils.cost_monitor import CostMonitor

from .social_sentiment import SocialSentimentAnalyzer
from .insider_trading import InsiderTradingAnalyzer
from .earnings_whispers import EarningsWhisperAnalyzer
from .options_flow import OptionsFlowAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class AlternativeDataScore:
    """Consolidated alternative data score"""
    overall_score: float  # -1 (very negative) to 1 (very positive)
    confidence: float
    component_scores: Dict[str, float]
    component_weights: Dict[str, float]
    key_signals: List[str]
    risk_factors: List[str]
    data_quality: float
    last_updated: datetime

class AlternativeDataOrchestrator:
    """
    Orchestrates comprehensive alternative data analysis
    
    Features:
    - Coordinates multiple alternative data sources
    - Provides weighted composite scoring
    - Manages data freshness and quality
    - Generates unified insights and alerts
    - Optimizes API usage across modules
    - Provides fallback when data sources unavailable
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = CacheManager()
        self.cost_monitor = CostMonitor()
        
        # Initialize component analyzers
        self.social_analyzer = SocialSentimentAnalyzer(config)
        self.insider_analyzer = InsiderTradingAnalyzer(config)
        self.earnings_analyzer = EarningsWhisperAnalyzer(config)
        self.options_analyzer = OptionsFlowAnalyzer(config)
        
        # Component weights (can be adjusted based on effectiveness)
        self.default_weights = {
            'social_sentiment': 0.2,
            'insider_trading': 0.3,
            'earnings_whispers': 0.25,
            'options_flow': 0.25
        }
    
    async def analyze_alternative_data(
        self,
        symbol: str,
        components: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Perform comprehensive alternative data analysis
        
        Args:
            symbol: Stock ticker symbol
            components: List of components to analyze (default: all)
            custom_weights: Custom weights for components
            
        Returns:
            Comprehensive alternative data analysis
        """
        if components is None:
            components = ['social_sentiment', 'insider_trading', 'earnings_whispers', 'options_flow']
            
        # Check cache first
        cache_key = f"alternative_data:{symbol}:{'-'.join(sorted(components))}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            results = {
                'symbol': symbol,
                'analysis_date': datetime.now(),
                'components': {},
                'overall_score': None,
                'unified_insights': [],
                'alerts': [],
                'data_quality': {},
                'cost_impact': {}
            }
            
            # Track costs and performance
            start_time = datetime.now()
            component_results = {}
            
            # Run analyses in parallel with error handling
            tasks = []
            
            if 'social_sentiment' in components:
                tasks.append(self._run_social_analysis(symbol))
            if 'insider_trading' in components:
                tasks.append(self._run_insider_analysis(symbol))
            if 'earnings_whispers' in components:
                tasks.append(self._run_earnings_analysis(symbol))
            if 'options_flow' in components:
                tasks.append(self._run_options_analysis(symbol))
            
            # Execute all analyses
            component_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(component_results_list):
                component = components[i] if i < len(components) else f"component_{i}"
                
                if isinstance(result, Exception):
                    logger.error(f"Error in {component} analysis: {result}")
                    component_results[component] = {'error': str(result)}
                else:
                    component_results[component] = result
            
            results['components'] = component_results
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                component_results, 
                custom_weights or self.default_weights
            )
            results['overall_score'] = overall_score
            
            # Generate unified insights
            unified_insights = self._generate_unified_insights(component_results, overall_score)
            results['unified_insights'] = unified_insights
            
            # Collect all alerts
            all_alerts = []
            for component, data in component_results.items():
                if isinstance(data, dict) and 'alerts' in data:
                    for alert in data['alerts']:
                        alert['source'] = component
                        all_alerts.append(alert)
            results['alerts'] = all_alerts
            
            # Calculate data quality metrics
            data_quality = self._assess_data_quality(component_results)
            results['data_quality'] = data_quality
            
            # Track cost impact
            execution_time = (datetime.now() - start_time).total_seconds()
            results['cost_impact'] = {
                'execution_time_seconds': execution_time,
                'components_analyzed': len([r for r in component_results.values() if 'error' not in r]),
                'api_calls_estimated': self._estimate_api_calls(components)
            }
            
            # Cache results
            cache_duration = self._determine_cache_duration(data_quality)
            await self.cache.set(cache_key, results, expire=cache_duration)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in alternative data analysis for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _run_social_analysis(self, symbol: str) -> Dict:
        """Run social sentiment analysis with error handling"""
        try:
            result = await self.social_analyzer.analyze_stock_sentiment(symbol, days_back=7)
            
            # Get alerts
            alerts = await self.social_analyzer.get_sentiment_alerts(
                symbol, 
                {'bullish_threshold': 0.4, 'bearish_threshold': -0.4, 'momentum_threshold': 0.3}
            )
            result['alerts'] = alerts
            
            return result
        except Exception as e:
            logger.error(f"Social sentiment analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _run_insider_analysis(self, symbol: str) -> Dict:
        """Run insider trading analysis with error handling"""
        try:
            result = await self.insider_analyzer.analyze_insider_activity(symbol, days_back=90)
            
            # Get alerts
            alerts = await self.insider_analyzer.get_insider_alerts(
                symbol,
                {'confidence_threshold': 0.7, 'bullish_threshold': 0.4, 'bearish_threshold': -0.4}
            )
            result['alerts'] = alerts
            
            return result
        except Exception as e:
            logger.error(f"Insider trading analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _run_earnings_analysis(self, symbol: str) -> Dict:
        """Run earnings whispers analysis with error handling"""
        try:
            result = await self.earnings_analyzer.analyze_earnings_expectations(symbol, quarters_ahead=2)
            
            # Get alerts
            alerts = await self.earnings_analyzer.get_earnings_alerts(
                symbol,
                {'earnings_alert_days': 7, 'surprise_threshold': 0.8}
            )
            result['alerts'] = alerts
            
            return result
        except Exception as e:
            logger.error(f"Earnings analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _run_options_analysis(self, symbol: str) -> Dict:
        """Run options flow analysis with error handling"""
        try:
            result = await self.options_analyzer.analyze_options_flow(symbol, days_back=5)
            
            # Get alerts
            alerts = await self.options_analyzer.get_options_alerts(
                symbol,
                {
                    'confidence_threshold': 0.7,
                    'bullish_threshold': 0.4,
                    'bearish_threshold': -0.4,
                    'high_pcr_threshold': 3.0,
                    'low_pcr_threshold': 0.3,
                    'unusual_magnitude_threshold': 5.0
                }
            )
            result['alerts'] = alerts
            
            return result
        except Exception as e:
            logger.error(f"Options flow analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(
        self,
        component_results: Dict,
        weights: Dict[str, float]
    ) -> Optional[AlternativeDataScore]:
        """Calculate weighted overall alternative data score"""
        
        component_scores = {}
        component_confidences = {}
        valid_components = []
        
        # Extract scores from each component
        for component, data in component_results.items():
            if 'error' in data:
                continue
                
            score = None
            confidence = 0
            
            if component == 'social_sentiment':
                aggregate = data.get('aggregate', {})
                if aggregate:
                    score = aggregate.get('compound_score', 0)
                    confidence = aggregate.get('confidence', 0)
                    
            elif component == 'insider_trading':
                sentiment = data.get('sentiment')
                if sentiment:
                    score = sentiment.sentiment_score
                    confidence = sentiment.confidence
                    
            elif component == 'earnings_whispers':
                sentiment = data.get('sentiment')
                if sentiment:
                    score = sentiment.sentiment_score
                    confidence = sentiment.confidence
                    
            elif component == 'options_flow':
                sentiment = data.get('sentiment')
                if sentiment:
                    score = sentiment.sentiment_score
                    confidence = sentiment.confidence
            
            if score is not None:
                component_scores[component] = score
                component_confidences[component] = confidence
                valid_components.append(component)
        
        if not component_scores:
            return None
        
        # Calculate weighted score
        total_weighted_score = 0
        total_weight = 0
        
        for component in valid_components:
            weight = weights.get(component, 0)
            if weight > 0:
                # Adjust weight by confidence
                adjusted_weight = weight * component_confidences[component]
                total_weighted_score += component_scores[component] * adjusted_weight
                total_weight += adjusted_weight
        
        if total_weight == 0:
            return None
        
        overall_score = total_weighted_score / total_weight
        overall_confidence = np.mean([component_confidences[c] for c in valid_components])
        
        # Generate key signals and risk factors
        key_signals = []
        risk_factors = []
        
        for component, score in component_scores.items():
            confidence = component_confidences[component]
            
            if confidence > 0.6:  # High confidence signals
                if score > 0.3:
                    key_signals.append(f"Strong positive signal from {component.replace('_', ' ')}")
                elif score < -0.3:
                    risk_factors.append(f"Strong negative signal from {component.replace('_', ' ')}")
        
        # Calculate data quality
        data_quality = len(valid_components) / len(component_results)
        
        return AlternativeDataScore(
            overall_score=overall_score,
            confidence=overall_confidence,
            component_scores=component_scores,
            component_weights={c: weights.get(c, 0) for c in valid_components},
            key_signals=key_signals,
            risk_factors=risk_factors,
            data_quality=data_quality,
            last_updated=datetime.now()
        )
    
    def _generate_unified_insights(
        self,
        component_results: Dict,
        overall_score: Optional[AlternativeDataScore]
    ) -> List[str]:
        """Generate unified insights from all components"""
        insights = []
        
        if overall_score is None:
            return ["Insufficient alternative data for analysis"]
        
        # Overall sentiment insight
        if overall_score.confidence > 0.6:
            if overall_score.overall_score > 0.3:
                insights.append(f"Strong positive alternative data sentiment (score: {overall_score.overall_score:.2f})")
            elif overall_score.overall_score < -0.3:
                insights.append(f"Strong negative alternative data sentiment (score: {overall_score.overall_score:.2f})")
            else:
                insights.append(f"Neutral alternative data sentiment (score: {overall_score.overall_score:.2f})")
        
        # Component-specific insights
        strongest_component = max(
            overall_score.component_scores.items(), 
            key=lambda x: abs(x[1])
        )
        insights.append(f"Strongest signal from {strongest_component[0].replace('_', ' ')}")
        
        # Data quality insight
        if overall_score.data_quality > 0.8:
            insights.append("High quality alternative data coverage")
        elif overall_score.data_quality < 0.5:
            insights.append("Limited alternative data coverage - results may be less reliable")
        
        # Consensus vs divergence
        scores = list(overall_score.component_scores.values())
        if len(scores) > 1:
            score_std = np.std(scores)
            if score_std < 0.2:
                insights.append("Strong consensus across alternative data sources")
            elif score_std > 0.5:
                insights.append("Divergent signals across alternative data sources")
        
        # Add key signals and risk factors
        insights.extend(overall_score.key_signals)
        insights.extend([f"Risk: {risk}" for risk in overall_score.risk_factors])
        
        # Component-specific insights
        for component, data in component_results.items():
            if 'error' not in data and 'insights' in data:
                component_insights = data['insights'][:2]  # Top 2 insights per component
                for insight in component_insights:
                    if insight not in insights:  # Avoid duplicates
                        insights.append(f"{component.replace('_', ' ').title()}: {insight}")
        
        return insights
    
    def _assess_data_quality(self, component_results: Dict) -> Dict:
        """Assess data quality across all components"""
        quality_metrics = {
            'components_available': 0,
            'components_with_data': 0,
            'overall_quality_score': 0,
            'data_freshness': 'unknown',
            'coverage_gaps': []
        }
        
        total_components = len(component_results)
        successful_components = 0
        components_with_meaningful_data = 0
        
        for component, data in component_results.items():
            quality_metrics['components_available'] += 1
            
            if 'error' not in data:
                successful_components += 1
                
                # Check for meaningful data
                has_meaningful_data = False
                
                if component == 'social_sentiment':
                    aggregate = data.get('aggregate', {})
                    if aggregate and aggregate.get('total_posts_analyzed', 0) > 5:
                        has_meaningful_data = True
                        
                elif component == 'insider_trading':
                    if data.get('transactions_analyzed', 0) > 0:
                        has_meaningful_data = True
                        
                elif component == 'earnings_whispers':
                    if data.get('estimates', {}) and any(data['estimates'].values()):
                        has_meaningful_data = True
                        
                elif component == 'options_flow':
                    options_data = data.get('options_data', {})
                    if options_data.get('calls') or options_data.get('puts'):
                        has_meaningful_data = True
                
                if has_meaningful_data:
                    components_with_meaningful_data += 1
                else:
                    quality_metrics['coverage_gaps'].append(f"Limited {component.replace('_', ' ')} data")
        
        quality_metrics['components_with_data'] = components_with_meaningful_data
        quality_metrics['overall_quality_score'] = components_with_meaningful_data / total_components if total_components > 0 else 0
        
        # Assess data freshness (simplified - would be more sophisticated in production)
        if components_with_meaningful_data > total_components * 0.75:
            quality_metrics['data_freshness'] = 'good'
        elif components_with_meaningful_data > total_components * 0.5:
            quality_metrics['data_freshness'] = 'fair'
        else:
            quality_metrics['data_freshness'] = 'poor'
        
        return quality_metrics
    
    def _estimate_api_calls(self, components: List[str]) -> int:
        """Estimate number of API calls used"""
        # Conservative estimates based on component complexity
        call_estimates = {
            'social_sentiment': 10,  # Reddit + Twitter + StockTwits
            'insider_trading': 5,    # SEC EDGAR calls
            'earnings_whispers': 3,  # Yahoo + Alpha Vantage
            'options_flow': 2        # Yahoo options chain
        }
        
        total_calls = sum(call_estimates.get(component, 1) for component in components)
        return total_calls
    
    def _determine_cache_duration(self, data_quality: Dict) -> int:
        """Determine cache duration based on data quality"""
        base_duration = 1800  # 30 minutes
        
        quality_score = data_quality.get('overall_quality_score', 0.5)
        
        # Higher quality data can be cached longer
        if quality_score > 0.8:
            return base_duration * 2  # 60 minutes
        elif quality_score < 0.3:
            return base_duration // 2  # 15 minutes
        else:
            return base_duration
    
    async def get_alternative_data_alerts(
        self, 
        symbol: str,
        alert_thresholds: Optional[Dict] = None
    ) -> List[Dict]:
        """Get comprehensive alternative data alerts"""
        if alert_thresholds is None:
            alert_thresholds = {
                'overall_score_threshold': 0.4,
                'confidence_threshold': 0.6,
                'component_agreement_threshold': 0.7
            }
        
        analysis = await self.analyze_alternative_data(symbol)
        
        alerts = []
        
        if 'error' in analysis:
            return alerts
        
        # Overall score alerts
        overall_score = analysis.get('overall_score')
        if overall_score and overall_score.confidence > alert_thresholds['confidence_threshold']:
            if abs(overall_score.overall_score) > alert_thresholds['overall_score_threshold']:
                direction = 'bullish' if overall_score.overall_score > 0 else 'bearish'
                alerts.append({
                    'type': 'alternative_data_consensus',
                    'message': f'Strong {direction} alternative data consensus for {symbol}',
                    'score': overall_score.overall_score,
                    'confidence': overall_score.confidence,
                    'components': len(overall_score.component_scores),
                    'urgency': 'high'
                })
        
        # Component alerts (already collected in main analysis)
        component_alerts = analysis.get('alerts', [])
        alerts.extend(component_alerts)
        
        # Data quality alerts
        data_quality = analysis.get('data_quality', {})
        if data_quality.get('overall_quality_score', 1) < 0.3:
            alerts.append({
                'type': 'data_quality_warning',
                'message': f'Limited alternative data coverage for {symbol}',
                'quality_score': data_quality.get('overall_quality_score'),
                'gaps': data_quality.get('coverage_gaps', []),
                'urgency': 'low'
            })
        
        return alerts