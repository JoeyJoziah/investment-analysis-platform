"""
World-Class Investment Recommendation Engine
Combines all analysis types to generate actionable recommendations
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import json

from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.models.ml_models import ModelManager, PredictionResult
# from backend.data_ingestion.market_scanner import MarketScanner
# from backend.utils.risk_manager import RiskManager
# from backend.utils.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


class RecommendationAction(Enum):
    """Recommendation actions"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StockRecommendation:
    """Complete stock recommendation"""
    ticker: str
    action: RecommendationAction
    confidence: float
    priority: int  # 1-10
    
    # Price targets
    entry_price: float
    target_price: float
    stop_loss: float
    expected_return: float
    time_horizon_days: int
    
    # Risk metrics
    risk_score: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Analysis scores
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    ml_prediction_score: float
    
    # Detailed analysis
    technical_analysis: Dict
    fundamental_analysis: Dict
    sentiment_analysis: Dict
    ml_predictions: Dict[str, PredictionResult]
    
    # Reasoning
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]
    catalysts: List[str]
    
    # Metadata
    generated_at: datetime
    valid_until: datetime
    
    # Position sizing
    recommended_allocation: float  # Percentage of portfolio
    max_position_size: float  # Dollar amount
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'ticker': self.ticker,
            'action': self.action.value,
            'confidence': self.confidence,
            'priority': self.priority,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'expected_return': self.expected_return,
            'time_horizon_days': self.time_horizon_days,
            'risk_score': self.risk_score,
            'volatility': self.volatility,
            'beta': self.beta,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'technical_score': self.technical_score,
            'fundamental_score': self.fundamental_score,
            'sentiment_score': self.sentiment_score,
            'ml_prediction_score': self.ml_prediction_score,
            'key_factors': self.key_factors,
            'risks': self.risks,
            'opportunities': self.opportunities,
            'catalysts': self.catalysts,
            'generated_at': self.generated_at.isoformat(),
            'valid_until': self.valid_until.isoformat(),
            'recommended_allocation': self.recommended_allocation,
            'max_position_size': self.max_position_size
        }


class RecommendationEngine:
    """
    Master recommendation engine that orchestrates all analysis
    """
    
    def __init__(self):
        self.technical_engine = TechnicalAnalysisEngine()
        self.fundamental_engine = FundamentalAnalysisEngine()
        self.sentiment_engine = SentimentAnalysisEngine()
        self.model_manager = ModelManager()
        # self.market_scanner = MarketScanner()  # Disabled - missing module
        # self.risk_manager = RiskManager()  # Disabled - missing module
        # self.portfolio_optimizer = PortfolioOptimizer()  # Disabled - missing module
        
        # Recommendation thresholds
        self.thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'hold': 0.4,
            'sell': 0.2,
            'strong_sell': 0.0
        }
    
    async def initialize(self):
        """Initialize all components"""
        await self.model_manager.load_models()
        await self.market_scanner.initialize()
        logger.info("Recommendation engine initialized")
    
    async def generate_daily_recommendations(
        self,
        max_recommendations: int = 50,
        risk_tolerance: str = 'moderate',
        sectors: Optional[List[str]] = None,
        market_cap_range: Optional[Tuple[float, float]] = None
    ) -> List[StockRecommendation]:
        """
        Generate daily recommendations for all stocks
        """
        logger.info("Starting daily recommendation generation...")
        
        # Step 1: Scan market for candidates
        candidates = await self.market_scanner.scan_market(
            sectors=sectors,
            market_cap_range=market_cap_range,
            max_stocks=500  # Pre-filter to top 500
        )
        
        logger.info(f"Found {len(candidates)} candidate stocks")
        
        # Step 2: Analyze each candidate
        recommendations = []
        
        # Process in batches for efficiency
        batch_size = 20
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            batch_tasks = [
                self.analyze_stock(stock['ticker'], stock)
                for stock in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing stock: {result}")
                    continue
                
                if result and self._should_recommend(result, risk_tolerance):
                    recommendations.append(result)
        
        # Step 3: Rank and filter recommendations
        ranked_recommendations = self._rank_recommendations(recommendations)
        
        # Step 4: Apply portfolio optimization
        optimized_recommendations = await self._optimize_recommendations(
            ranked_recommendations[:max_recommendations * 2],  # Consider 2x for optimization
            risk_tolerance
        )
        
        # Step 5: Final selection
        final_recommendations = optimized_recommendations[:max_recommendations]
        
        logger.info(f"Generated {len(final_recommendations)} recommendations")
        
        return final_recommendations
    
    async def analyze_stock(
        self,
        ticker: str,
        market_data: Optional[Dict] = None
    ) -> Optional[StockRecommendation]:
        """
        Comprehensive analysis of a single stock
        """
        try:
            logger.info(f"Analyzing {ticker}...")
            
            # Fetch all required data
            stock_data = await self._fetch_stock_data(ticker, market_data)
            
            if not stock_data:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Run all analysis types in parallel
            analysis_tasks = [
                self._run_technical_analysis(stock_data),
                self._run_fundamental_analysis(stock_data),
                self._run_sentiment_analysis(ticker, stock_data),
                self._run_ml_predictions(ticker, stock_data)
            ]
            
            results = await asyncio.gather(*analysis_tasks)
            
            technical_analysis = results[0]
            fundamental_analysis = results[1]
            sentiment_analysis = results[2]
            ml_predictions = results[3]
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(stock_data, ml_predictions)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                ticker=ticker,
                stock_data=stock_data,
                technical_analysis=technical_analysis,
                fundamental_analysis=fundamental_analysis,
                sentiment_analysis=sentiment_analysis,
                ml_predictions=ml_predictions,
                risk_metrics=risk_metrics
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    async def _fetch_stock_data(
        self,
        ticker: str,
        market_data: Optional[Dict]
    ) -> Optional[Dict]:
        """Fetch all required data for analysis"""
        
        # Get from market scanner if not provided
        if not market_data:
            market_data = await self.market_scanner.get_stock_data(ticker)
        
        if not market_data:
            return None
        
        # Ensure we have required data
        required_fields = ['price_history', 'fundamentals', 'market_cap']
        
        for field in required_fields:
            if field not in market_data:
                logger.warning(f"Missing {field} for {ticker}")
                return None
        
        return market_data
    
    async def _run_technical_analysis(self, stock_data: Dict) -> Dict:
        """Run technical analysis"""
        price_df = stock_data.get('price_history')
        
        if price_df is None or len(price_df) < 200:
            return {}
        
        return self.technical_engine.analyze_stock(price_df)
    
    async def _run_fundamental_analysis(self, stock_data: Dict) -> Dict:
        """Run fundamental analysis"""
        financials = stock_data.get('fundamentals', {})
        market_data = {
            'market_cap': stock_data.get('market_cap', 0),
            'price': stock_data.get('current_price', 0),
            'beta': stock_data.get('beta', 1.0)
        }
        
        # Get peer data if available
        peer_data = stock_data.get('peer_data')
        
        return await self.fundamental_engine.analyze_company(
            ticker=stock_data.get('ticker'),
            financials=financials,
            market_data=market_data,
            peer_data=peer_data
        )
    
    async def _run_sentiment_analysis(self, ticker: str, stock_data: Dict) -> Dict:
        """Run sentiment analysis"""
        # Get text data from various sources
        text_data = []
        
        # News
        if 'news' in stock_data:
            for article in stock_data['news'][:50]:  # Last 50 articles
                text_data.append({
                    'text': f"{article.get('headline', '')} {article.get('summary', '')}",
                    'source': 'news',
                    'timestamp': article.get('datetime', datetime.utcnow())
                })
        
        # Social media
        if 'social_mentions' in stock_data:
            for mention in stock_data['social_mentions'][:100]:  # Last 100 mentions
                text_data.append({
                    'text': mention.get('text', ''),
                    'source': mention.get('platform', 'social'),
                    'timestamp': mention.get('timestamp', datetime.utcnow())
                })
        
        # Analyst reports
        if 'analyst_opinions' in stock_data:
            for opinion in stock_data['analyst_opinions']:
                text_data.append({
                    'text': opinion.get('summary', ''),
                    'source': 'analyst',
                    'timestamp': opinion.get('date', datetime.utcnow())
                })
        
        if not text_data:
            return {
                'overall_sentiment': {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
            }
        
        return await self.sentiment_engine.analyze_sentiment(ticker, text_data)
    
    async def _run_ml_predictions(self, ticker: str, stock_data: Dict) -> Dict[str, PredictionResult]:
        """Run ML predictions"""
        price_df = stock_data.get('price_history')
        
        if price_df is None or len(price_df) < 60:
            return {}
        
        # Add fundamental features if available
        if 'fundamentals' in stock_data:
            for key, value in stock_data['fundamentals'].items():
                if isinstance(value, (int, float)):
                    price_df[f'fundamental_{key}'] = value
        
        # Add sentiment features if available
        if 'sentiment_history' in stock_data:
            price_df['sentiment_score'] = stock_data['sentiment_history']
        
        # Get predictions for multiple horizons
        predictions = {}
        
        for horizon in [5, 20, 60]:  # 1 week, 1 month, 3 months
            horizon_predictions = await self.model_manager.predict(
                ticker=ticker,
                current_data=price_df,
                horizon=horizon
            )
            
            # Store the ensemble prediction
            if 'ensemble' in horizon_predictions:
                predictions[f'horizon_{horizon}'] = horizon_predictions['ensemble']
        
        return predictions
    
    async def _calculate_risk_metrics(
        self,
        stock_data: Dict,
        ml_predictions: Dict
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        price_history = stock_data.get('price_history')
        
        if price_history is None or len(price_history) < 30:
            return {
                'risk_score': 0.5,
                'volatility': 0.0,
                'beta': 1.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        returns = price_history['close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Beta (if market data available)
        beta = stock_data.get('beta', 1.0)
        
        # Sharpe Ratio
        risk_free_rate = 0.045  # Current treasury rate
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk
        cvar_95 = returns[returns <= var_95].mean()
        
        # ML prediction uncertainty
        if ml_predictions:
            prediction_std = np.std([
                pred.predicted_return for pred in ml_predictions.values()
                if hasattr(pred, 'predicted_return')
            ])
        else:
            prediction_std = 0.0
        
        # Combined risk score (0-1, higher is riskier)
        risk_components = [
            min(volatility / 0.5, 1.0),  # Normalize to 50% annual vol
            min(abs(max_drawdown) / 0.3, 1.0),  # Normalize to 30% drawdown
            min(prediction_std / 0.1, 1.0),  # Normalize to 10% prediction std
            max(0, 1 - sharpe_ratio / 2)  # Inverse Sharpe, normalize to 2.0
        ]
        
        risk_score = np.mean(risk_components)
        
        return {
            'risk_score': risk_score,
            'volatility': volatility,
            'beta': beta,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _generate_recommendation(
        self,
        ticker: str,
        stock_data: Dict,
        technical_analysis: Dict,
        fundamental_analysis: Dict,
        sentiment_analysis: Dict,
        ml_predictions: Dict[str, PredictionResult],
        risk_metrics: Dict[str, float]
    ) -> StockRecommendation:
        """
        Generate final recommendation based on all analysis
        """
        
        # Get current price
        current_price = stock_data.get('current_price', 0)
        if current_price == 0:
            price_history = stock_data.get('price_history')
            if price_history is not None and len(price_history) > 0:
                current_price = price_history['close'].iloc[-1]
        
        # Calculate component scores (0-1)
        technical_score = self._normalize_score(
            technical_analysis.get('composite_score', 0), -1, 1
        )
        
        fundamental_score = self._normalize_score(
            fundamental_analysis.get('composite_score', 50), 0, 100
        ) if fundamental_analysis else 0.5
        
        sentiment_score = self._normalize_score(
            sentiment_analysis.get('overall_sentiment', {}).get('score', 0), -1, 1
        )
        
        # ML prediction score (average across horizons)
        ml_scores = []
        for pred in ml_predictions.values():
            if hasattr(pred, 'predicted_return'):
                # Convert return to score
                ml_scores.append(self._normalize_score(pred.predicted_return, -0.2, 0.2))
        
        ml_prediction_score = np.mean(ml_scores) if ml_scores else 0.5
        
        # Weight the scores
        weights = {
            'technical': 0.25,
            'fundamental': 0.30,
            'sentiment': 0.15,
            'ml_prediction': 0.30
        }
        
        # Calculate overall score
        overall_score = (
            weights['technical'] * technical_score +
            weights['fundamental'] * fundamental_score +
            weights['sentiment'] * sentiment_score +
            weights['ml_prediction'] * ml_prediction_score
        )
        
        # Adjust for risk
        risk_adjusted_score = overall_score * (1 - risk_metrics['risk_score'] * 0.3)
        
        # Determine action
        action = self._determine_action(risk_adjusted_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            technical_analysis,
            fundamental_analysis,
            sentiment_analysis,
            ml_predictions,
            risk_metrics
        )
        
        # Set price targets
        price_targets = self._calculate_price_targets(
            current_price,
            ml_predictions,
            technical_analysis,
            risk_metrics
        )
        
        # Determine time horizon
        time_horizon = self._determine_time_horizon(
            action,
            technical_analysis,
            ml_predictions
        )
        
        # Extract key factors
        key_factors = self._extract_key_factors(
            technical_analysis,
            fundamental_analysis,
            sentiment_analysis,
            ml_predictions
        )
        
        # Identify risks and opportunities
        risks = self._identify_risks(
            fundamental_analysis,
            risk_metrics,
            sentiment_analysis
        )
        
        opportunities = self._identify_opportunities(
            fundamental_analysis,
            technical_analysis,
            sentiment_analysis
        )
        
        # Find catalysts
        catalysts = self._find_catalysts(
            stock_data,
            sentiment_analysis,
            fundamental_analysis
        )
        
        # Calculate position sizing
        position_sizing = self._calculate_position_sizing(
            confidence,
            risk_metrics,
            action
        )
        
        # Set priority
        priority = self._calculate_priority(
            risk_adjusted_score,
            confidence,
            opportunities
        )
        
        return StockRecommendation(
            ticker=ticker,
            action=action,
            confidence=confidence,
            priority=priority,
            entry_price=current_price,
            target_price=price_targets['target'],
            stop_loss=price_targets['stop_loss'],
            expected_return=price_targets['expected_return'],
            time_horizon_days=time_horizon,
            risk_score=risk_metrics['risk_score'],
            volatility=risk_metrics['volatility'],
            beta=risk_metrics['beta'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            ml_prediction_score=ml_prediction_score,
            technical_analysis=technical_analysis,
            fundamental_analysis=fundamental_analysis,
            sentiment_analysis=sentiment_analysis,
            ml_predictions=ml_predictions,
            key_factors=key_factors,
            risks=risks,
            opportunities=opportunities,
            catalysts=catalysts,
            generated_at=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=1),
            recommended_allocation=position_sizing['allocation'],
            max_position_size=position_sizing['max_size']
        )
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range"""
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))
    
    def _determine_action(self, score: float) -> RecommendationAction:
        """Determine recommendation action based on score"""
        if score >= self.thresholds['strong_buy']:
            return RecommendationAction.STRONG_BUY
        elif score >= self.thresholds['buy']:
            return RecommendationAction.BUY
        elif score >= self.thresholds['hold']:
            return RecommendationAction.HOLD
        elif score >= self.thresholds['sell']:
            return RecommendationAction.SELL
        else:
            return RecommendationAction.STRONG_SELL
    
    def _calculate_confidence(
        self,
        technical: Dict,
        fundamental: Dict,
        sentiment: Dict,
        ml_predictions: Dict,
        risk_metrics: Dict
    ) -> float:
        """Calculate overall confidence in recommendation"""
        
        confidence_factors = []
        
        # Technical confidence
        if technical:
            # Strong signals increase confidence
            signal_count = len(technical.get('signals', []))
            pattern_count = len(technical.get('pattern_recognition', {}).get('candlestick_patterns', {}))
            
            tech_confidence = min(1.0, (signal_count + pattern_count) / 10)
            confidence_factors.append(tech_confidence)
        
        # Fundamental confidence
        if fundamental:
            quality_score = fundamental.get('quality_score', {}).get('overall_score', 50) / 100
            confidence_factors.append(quality_score)
        
        # Sentiment confidence
        if sentiment:
            sentiment_confidence = sentiment.get('overall_sentiment', {}).get('confidence', 0.5)
            confidence_factors.append(sentiment_confidence)
        
        # ML model confidence
        if ml_predictions:
            ml_confidences = [
                pred.model_confidence for pred in ml_predictions.values()
                if hasattr(pred, 'model_confidence')
            ]
            if ml_confidences:
                confidence_factors.append(np.mean(ml_confidences))
        
        # Risk adjustment
        risk_penalty = risk_metrics['risk_score'] * 0.2
        
        # Calculate overall confidence
        if confidence_factors:
            base_confidence = np.mean(confidence_factors)
            adjusted_confidence = max(0.1, base_confidence - risk_penalty)
            return adjusted_confidence
        
        return 0.5
    
    def _calculate_price_targets(
        self,
        current_price: float,
        ml_predictions: Dict,
        technical_analysis: Dict,
        risk_metrics: Dict
    ) -> Dict[str, float]:
        """Calculate price targets and stop loss"""
        
        # Get ML target
        ml_targets = []
        for pred in ml_predictions.values():
            if hasattr(pred, 'predicted_price'):
                ml_targets.append(pred.predicted_price)
        
        if ml_targets:
            ml_target = np.mean(ml_targets)
        else:
            ml_target = current_price
        
        # Get technical targets
        tech_resistance = technical_analysis.get('support_resistance', {}).get('primary_resistance', current_price * 1.1)
        tech_support = technical_analysis.get('support_resistance', {}).get('primary_support', current_price * 0.9)
        
        # Combine targets
        target_price = 0.7 * ml_target + 0.3 * tech_resistance
        
        # Calculate stop loss based on support and volatility
        volatility_stop = current_price * (1 - 2 * risk_metrics['volatility'] / np.sqrt(252) * 5)  # 5-day vol
        support_stop = tech_support * 0.98  # 2% below support
        
        stop_loss = max(volatility_stop, support_stop)
        
        # Expected return
        expected_return = (target_price - current_price) / current_price
        
        return {
            'target': target_price,
            'stop_loss': stop_loss,
            'expected_return': expected_return,
            'risk_reward_ratio': abs(expected_return) / abs((stop_loss - current_price) / current_price)
        }
    
    def _determine_time_horizon(
        self,
        action: RecommendationAction,
        technical_analysis: Dict,
        ml_predictions: Dict
    ) -> int:
        """Determine investment time horizon in days"""
        
        # Base horizon on action
        base_horizons = {
            RecommendationAction.STRONG_BUY: 60,
            RecommendationAction.BUY: 30,
            RecommendationAction.HOLD: 20,
            RecommendationAction.SELL: 10,
            RecommendationAction.STRONG_SELL: 5
        }
        
        base_horizon = base_horizons.get(action, 20)
        
        # Adjust based on technical patterns
        if technical_analysis:
            patterns = technical_analysis.get('pattern_recognition', {}).get('chart_patterns', {})
            
            # Longer-term patterns extend horizon
            if any(p in patterns for p in ['cup_and_handle', 'ascending_triangle']):
                base_horizon = int(base_horizon * 1.5)
            
            # Short-term patterns reduce horizon
            elif any(p in patterns for p in ['flag', 'pennant']):
                base_horizon = int(base_horizon * 0.7)
        
        # Consider ML predictions
        if ml_predictions:
            # If longer-term predictions are stronger, extend horizon
            if 'horizon_60' in ml_predictions and 'horizon_5' in ml_predictions:
                long_return = ml_predictions['horizon_60'].predicted_return
                short_return = ml_predictions['horizon_5'].predicted_return
                
                if abs(long_return) > abs(short_return) * 2:
                    base_horizon = 60
        
        return base_horizon
    
    def _extract_key_factors(
        self,
        technical: Dict,
        fundamental: Dict,
        sentiment: Dict,
        ml_predictions: Dict
    ) -> List[str]:
        """Extract key factors driving the recommendation"""
        
        factors = []
        
        # Technical factors
        if technical:
            # Trend
            trend = technical.get('market_structure', {}).get('trend', '')
            if 'uptrend' in trend:
                factors.append("Strong technical uptrend")
            elif 'downtrend' in trend:
                factors.append("Technical downtrend warning")
            
            # Patterns
            patterns = technical.get('pattern_recognition', {}).get('candlestick_patterns', {})
            if patterns:
                pattern_names = list(patterns.keys())[:2]
                factors.append(f"Technical patterns: {', '.join(pattern_names)}")
            
            # Momentum
            rsi = technical.get('momentum_indicators', {}).get('rsi_14', 50)
            if rsi < 30:
                factors.append("Oversold conditions (RSI < 30)")
            elif rsi > 70:
                factors.append("Overbought conditions (RSI > 70)")
        
        # Fundamental factors
        if fundamental:
            # Valuation
            valuation = fundamental.get('valuation_models', {})
            upside = valuation.get('upside_potential', 0)
            if upside > 30:
                factors.append(f"Significant undervaluation ({upside:.0f}% upside)")
            elif upside < -20:
                factors.append(f"Overvaluation concern ({abs(upside):.0f}% downside)")
            
            # Quality
            quality = fundamental.get('quality_score', {}).get('overall_score', 0)
            if quality > 80:
                factors.append("Exceptional business quality")
            
            # Growth
            growth = fundamental.get('growth_analysis', {}).get('growth_drivers', [])
            if growth:
                factors.append(f"Growth drivers: {', '.join(growth[:2])}")
        
        # Sentiment factors
        if sentiment:
            overall = sentiment.get('overall_sentiment', {})
            if overall.get('score', 0) > 0.5:
                factors.append("Positive market sentiment")
            elif overall.get('score', 0) < -0.5:
                factors.append("Negative sentiment warning")
            
            # Analyst sentiment
            analyst = sentiment.get('source_breakdown', {}).get('analyst', {})
            if analyst.get('average_sentiment', 0) > 0.6:
                factors.append("Bullish analyst consensus")
        
        # ML factors
        if ml_predictions:
            # Strong predictions
            strong_predictions = [
                pred for pred in ml_predictions.values()
                if hasattr(pred, 'predicted_return') and abs(pred.predicted_return) > 0.1
            ]
            
            if strong_predictions:
                avg_return = np.mean([p.predicted_return for p in strong_predictions])
                factors.append(f"ML models predict {avg_return*100:.1f}% return")
        
        return factors[:5]  # Top 5 factors
    
    def _identify_risks(
        self,
        fundamental: Dict,
        risk_metrics: Dict,
        sentiment: Dict
    ) -> List[str]:
        """Identify key risks"""
        
        risks = []
        
        # Fundamental risks
        if fundamental:
            fund_risks = fundamental.get('risks', [])
            for risk in fund_risks[:2]:
                risks.append(risk.get('description', ''))
            
            # Financial health risks
            health = fundamental.get('financial_health', {})
            z_score = health.get('altman_z_score', {}).get('score', 3)
            if z_score < 1.8:
                risks.append("Financial distress risk (low Altman Z-Score)")
        
        # Market risks
        if risk_metrics['volatility'] > 0.4:
            risks.append(f"High volatility ({risk_metrics['volatility']*100:.0f}% annual)")
        
        if risk_metrics['beta'] > 1.5:
            risks.append(f"High market sensitivity (Beta: {risk_metrics['beta']:.1f})")
        
        if risk_metrics['max_drawdown'] < -0.3:
            risks.append(f"Significant drawdown risk ({risk_metrics['max_drawdown']*100:.0f}%)")
        
        # Sentiment risks
        if sentiment:
            anomalies = sentiment.get('anomaly_detection', {})
            if anomalies.get('anomalies_detected'):
                risks.append("Unusual sentiment patterns detected")
        
        return risks[:4]  # Top 4 risks
    
    def _identify_opportunities(
        self,
        fundamental: Dict,
        technical: Dict,
        sentiment: Dict
    ) -> List[str]:
        """Identify opportunities"""
        
        opportunities = []
        
        # Fundamental opportunities
        if fundamental:
            fund_opps = fundamental.get('opportunities', [])
            for opp in fund_opps[:2]:
                opportunities.append(opp.get('description', ''))
            
            # Moat
            moat = fundamental.get('moat_analysis', {})
            if moat.get('rating') == 'wide':
                opportunities.append("Wide economic moat provides competitive advantage")
        
        # Technical opportunities
        if technical:
            # Support nearby
            sr = technical.get('support_resistance', {})
            if sr.get('current_price') and sr.get('primary_support'):
                support_distance = (sr['current_price'] - sr['primary_support']) / sr['current_price']
                if support_distance < 0.05:
                    opportunities.append("Trading near strong support level")
            
            # Breakout potential
            patterns = technical.get('pattern_recognition', {}).get('chart_patterns', {})
            if 'ascending_triangle' in patterns or 'cup_and_handle' in patterns:
                opportunities.append("Bullish breakout pattern forming")
        
        # Sentiment opportunities
        if sentiment:
            momentum = sentiment.get('temporal_analysis', {}).get('momentum', 0)
            if momentum > 0.2:
                opportunities.append("Improving sentiment momentum")
        
        return opportunities[:3]  # Top 3 opportunities
    
    def _find_catalysts(
        self,
        stock_data: Dict,
        sentiment: Dict,
        fundamental: Dict
    ) -> List[str]:
        """Identify potential catalysts"""
        
        catalysts = []
        
        # Earnings catalyst
        next_earnings = stock_data.get('next_earnings_date')
        if next_earnings:
            days_to_earnings = (next_earnings - datetime.utcnow()).days
            if 0 < days_to_earnings < 30:
                catalysts.append(f"Earnings report in {days_to_earnings} days")
        
        # News catalysts
        if sentiment:
            keywords = sentiment.get('keyword_analysis', {}).get('top_positive', [])
            if any(k in ['merger', 'acquisition', 'partnership', 'fda', 'approval'] for k in keywords):
                catalysts.append("Potential M&A or regulatory catalyst")
        
        # Product launches
        if 'product' in str(sentiment.get('keyword_analysis', {})).lower():
            catalysts.append("New product launch catalyst")
        
        # Sector rotation
        sector = stock_data.get('sector')
        if sector and fundamental:
            growth = fundamental.get('growth_analysis', {})
            if 'market_growth' in str(growth):
                catalysts.append(f"{sector} sector rotation opportunity")
        
        return catalysts[:3]  # Top 3 catalysts
    
    def _calculate_position_sizing(
        self,
        confidence: float,
        risk_metrics: Dict,
        action: RecommendationAction
    ) -> Dict[str, float]:
        """Calculate recommended position sizing"""
        
        # Base allocation on Kelly Criterion with safety margin
        # Kelly fraction = (p * b - q) / b
        # where p = probability of winning, b = odds, q = probability of losing
        
        # Simplified: use confidence as probability
        p = confidence
        q = 1 - p
        b = 2  # Assume 2:1 reward/risk ratio
        
        kelly_fraction = (p * b - q) / b
        
        # Apply safety factor (use 25% of Kelly)
        safe_kelly = kelly_fraction * 0.25
        
        # Adjust for risk
        risk_adjustment = 1 - risk_metrics['risk_score'] * 0.5
        
        # Action-based limits
        max_allocations = {
            RecommendationAction.STRONG_BUY: 0.10,  # 10% max
            RecommendationAction.BUY: 0.07,         # 7% max
            RecommendationAction.HOLD: 0.05,        # 5% max
            RecommendationAction.SELL: 0.0,         # No new position
            RecommendationAction.STRONG_SELL: 0.0   # No new position
        }
        
        max_allocation = max_allocations.get(action, 0.05)
        
        # Final allocation
        allocation = min(safe_kelly * risk_adjustment, max_allocation)
        allocation = max(0, allocation)  # Ensure non-negative
        
        # Dollar amount (assuming $100k portfolio)
        portfolio_size = 100000
        max_size = allocation * portfolio_size
        
        return {
            'allocation': allocation,
            'max_size': max_size,
            'kelly_fraction': kelly_fraction,
            'risk_adjusted_allocation': allocation
        }
    
    def _calculate_priority(
        self,
        score: float,
        confidence: float,
        opportunities: List[str]
    ) -> int:
        """Calculate recommendation priority (1-10)"""
        
        # Base priority on score
        base_priority = int(score * 10)
        
        # Boost for high confidence
        if confidence > 0.8:
            base_priority += 1
        
        # Boost for multiple opportunities
        if len(opportunities) >= 3:
            base_priority += 1
        
        # Ensure within bounds
        return max(1, min(10, base_priority))
    
    def _should_recommend(
        self,
        recommendation: StockRecommendation,
        risk_tolerance: str
    ) -> bool:
        """Determine if recommendation meets criteria"""
        
        # Risk tolerance thresholds
        risk_thresholds = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.8
        }
        
        max_risk = risk_thresholds.get(risk_tolerance, 0.5)
        
        # Filter by risk
        if recommendation.risk_score > max_risk:
            return False
        
        # Filter by confidence
        if recommendation.confidence < 0.5:
            return False
        
        # Filter by action
        if recommendation.action in [RecommendationAction.SELL, RecommendationAction.STRONG_SELL]:
            return False  # Only recommend buys
        
        # Filter by expected return
        if recommendation.expected_return < 0.05:  # Less than 5%
            return False
        
        return True
    
    def _rank_recommendations(
        self,
        recommendations: List[StockRecommendation]
    ) -> List[StockRecommendation]:
        """Rank recommendations by multiple criteria"""
        
        # Calculate composite ranking score
        for rec in recommendations:
            # Components of ranking
            return_score = min(rec.expected_return / 0.3, 1.0)  # Normalize to 30% return
            confidence_score = rec.confidence
            risk_score = 1 - rec.risk_score  # Invert so lower risk is better
            sharpe_score = min(rec.sharpe_ratio / 2, 1.0)  # Normalize to 2.0 Sharpe
            
            # Weighted ranking
            rec.ranking_score = (
                0.3 * return_score +
                0.2 * confidence_score +
                0.2 * risk_score +
                0.3 * sharpe_score
            )
        
        # Sort by ranking score
        ranked = sorted(
            recommendations,
            key=lambda x: x.ranking_score,
            reverse=True
        )
        
        # Update priorities based on ranking
        for i, rec in enumerate(ranked):
            rec.priority = min(10, max(1, 10 - i // 5))
        
        return ranked
    
    async def _optimize_recommendations(
        self,
        recommendations: List[StockRecommendation],
        risk_tolerance: str
    ) -> List[StockRecommendation]:
        """Apply portfolio optimization to recommendations"""
        
        if len(recommendations) < 2:
            return recommendations
        
        # Prepare data for optimization
        expected_returns = np.array([rec.expected_return for rec in recommendations])
        
        # Create covariance matrix (simplified - in production, calculate from returns)
        n_assets = len(recommendations)
        correlations = np.eye(n_assets) * 0.5  # Assume 0.5 correlation
        volatilities = np.array([rec.volatility for rec in recommendations])
        
        # Convert correlation to covariance
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Risk tolerance parameters
        risk_params = {
            'conservative': {'max_volatility': 0.15, 'min_sharpe': 1.0},
            'moderate': {'max_volatility': 0.25, 'min_sharpe': 0.7},
            'aggressive': {'max_volatility': 0.40, 'min_sharpe': 0.5}
        }
        
        params = risk_params.get(risk_tolerance, risk_params['moderate'])
        
        # Run portfolio optimization
        optimal_weights = await self.portfolio_optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints={
                'max_volatility': params['max_volatility'],
                'min_sharpe': params['min_sharpe'],
                'max_position': 0.10,  # Max 10% per position
                'min_position': 0.02   # Min 2% per position
            }
        )
        
        # Update allocations based on optimization
        optimized_recs = []
        
        for i, (rec, weight) in enumerate(zip(recommendations, optimal_weights)):
            if weight > 0.01:  # Only include if > 1% allocation
                rec.recommended_allocation = weight
                rec.max_position_size = weight * 100000  # Assume $100k portfolio
                optimized_recs.append(rec)
        
        # Re-rank based on optimal weights
        optimized_recs.sort(key=lambda x: x.recommended_allocation, reverse=True)
        
        return optimized_recs
    
    async def monitor_recommendations(
        self,
        active_recommendations: List[StockRecommendation]
    ) -> List[Dict]:
        """Monitor active recommendations and generate alerts"""
        
        alerts = []
        
        for rec in active_recommendations:
            # Get current data
            current_data = await self.market_scanner.get_stock_data(rec.ticker)
            
            if not current_data:
                continue
            
            current_price = current_data.get('current_price', 0)
            
            # Check stop loss
            if current_price <= rec.stop_loss:
                alerts.append({
                    'type': 'stop_loss',
                    'ticker': rec.ticker,
                    'message': f"{rec.ticker} hit stop loss at ${current_price:.2f}",
                    'action': 'sell',
                    'urgency': 'high'
                })
            
            # Check target
            elif current_price >= rec.target_price:
                alerts.append({
                    'type': 'target_reached',
                    'ticker': rec.ticker,
                    'message': f"{rec.ticker} reached target at ${current_price:.2f}",
                    'action': 'consider_profit_taking',
                    'urgency': 'medium'
                })
            
            # Check if recommendation expired
            elif datetime.utcnow() > rec.valid_until:
                alerts.append({
                    'type': 'recommendation_expired',
                    'ticker': rec.ticker,
                    'message': f"{rec.ticker} recommendation needs refresh",
                    'action': 'reanalyze',
                    'urgency': 'low'
                })
            
            # Check for significant changes
            price_change = (current_price - rec.entry_price) / rec.entry_price
            
            if abs(price_change) > 0.1:  # 10% move
                alerts.append({
                    'type': 'significant_move',
                    'ticker': rec.ticker,
                    'message': f"{rec.ticker} moved {price_change*100:.1f}% since recommendation",
                    'action': 'review_position',
                    'urgency': 'medium'
                })
        
        return alerts
    
    async def generate_report(
        self,
        recommendations: List[StockRecommendation],
        format: str = 'json'
    ) -> Any:
        """Generate recommendation report in various formats"""
        
        if format == 'json':
            return {
                'generated_at': datetime.utcnow().isoformat(),
                'recommendation_count': len(recommendations),
                'recommendations': [rec.to_dict() for rec in recommendations],
                'summary': self._generate_summary(recommendations)
            }
        
        elif format == 'pdf':
            # Generate PDF report (would use reportlab or similar)
            pass
        
        elif format == 'excel':
            # Generate Excel report (would use openpyxl)
            pass
        
        return None
    
    def _generate_summary(self, recommendations: List[StockRecommendation]) -> Dict:
        """Generate summary statistics"""
        
        if not recommendations:
            return {}
        
        returns = [rec.expected_return for rec in recommendations]
        risks = [rec.risk_score for rec in recommendations]
        
        return {
            'total_recommendations': len(recommendations),
            'average_expected_return': np.mean(returns),
            'average_risk_score': np.mean(risks),
            'by_action': {
                action.value: sum(1 for rec in recommendations if rec.action == action)
                for action in RecommendationAction
            },
            'top_sectors': self._get_top_sectors(recommendations),
            'total_allocation': sum(rec.recommended_allocation for rec in recommendations)
        }
    
    def _get_top_sectors(self, recommendations: List[StockRecommendation]) -> List[Dict]:
        """Get top sectors by recommendation count"""
        
        sector_counts = {}
        
        for rec in recommendations:
            # Would get sector from stock data
            sector = "Technology"  # Placeholder
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        sorted_sectors = sorted(
            sector_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'sector': sector, 'count': count}
            for sector, count in sorted_sectors[:5]
        ]