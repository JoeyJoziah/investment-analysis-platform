"""
Daily Stock Scanner
Scans 6000+ stocks daily for opportunities
"""

import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

from backend.analytics.technical.technical_analysis import TechnicalAnalyzer
from backend.analytics.fundamental.fundamental_analysis import FundamentalAnalyzer
from backend.analytics.sentiment.sentiment_analysis import SentimentAnalyzer
from backend.ml.model_manager import get_model_manager
from backend.utils.cache import get_redis
from backend.utils.database import get_db

logger = logging.getLogger(__name__)


class DailyStockScanner:
    """Scans all stocks daily for trading opportunities"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ml_manager = get_model_manager()
        self.redis_client = None
        self.scan_results = []
        
    async def initialize(self):
        """Initialize scanner resources"""
        self.redis_client = await get_redis()
        await self.ml_manager.load_models()
        
    async def scan_all_stocks(self, stock_list: List[str] = None) -> List[Dict]:
        """
        Scan all stocks or provided list
        
        Args:
            stock_list: Optional list of stock symbols to scan
            
        Returns:
            List of scan results with recommendations
        """
        if stock_list is None:
            stock_list = await self._get_all_stock_symbols()
        
        logger.info(f"Starting daily scan of {len(stock_list)} stocks")
        
        # Process stocks in batches for efficiency
        batch_size = 100
        all_results = []
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            batch_results = await self._process_batch(batch)
            all_results.extend(batch_results)
            
            # Update progress
            progress = (i + batch_size) / len(stock_list) * 100
            logger.info(f"Scan progress: {progress:.1f}%")
        
        # Rank and filter results
        ranked_results = self._rank_opportunities(all_results)
        
        # Cache results
        await self._cache_results(ranked_results)
        
        return ranked_results
    
    async def _get_all_stock_symbols(self) -> List[str]:
        """Get list of all tradeable stock symbols"""
        # In production, this would query the database
        # For now, return sample list
        
        # Get S&P 500 stocks as example
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        symbols = sp500['Symbol'].tolist()
        
        # Add NASDAQ 100
        nasdaq100 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 
                    'PYPL', 'INTC', 'NFLX', 'ADBE', 'CMCSA', 'PEP', 'CSCO', 'AVGO']
        
        # Combine and deduplicate
        all_symbols = list(set(symbols + nasdaq100))
        
        return all_symbols[:100]  # Limit for testing
    
    async def _process_batch(self, symbols: List[str]) -> List[Dict]:
        """Process a batch of stocks concurrently"""
        tasks = []
        
        for symbol in symbols:
            task = self._analyze_stock(symbol)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error processing stock: {result}")
        
        return valid_results
    
    async def _analyze_stock(self, symbol: str) -> Dict:
        """Comprehensive analysis of a single stock"""
        try:
            # Check cache first
            cached = await self._get_cached_analysis(symbol)
            if cached:
                return cached
            
            # Fetch stock data
            stock_data = await self._fetch_stock_data(symbol)
            
            if stock_data is None:
                return None
            
            # Run parallel analysis
            technical_task = self._run_technical_analysis(stock_data)
            fundamental_task = self._run_fundamental_analysis(symbol)
            sentiment_task = self._run_sentiment_analysis(symbol)
            ml_task = self._run_ml_prediction(stock_data)
            
            # Wait for all analyses
            technical_score, fundamental_score, sentiment_score, ml_prediction = await asyncio.gather(
                technical_task,
                fundamental_task, 
                sentiment_task,
                ml_task
            )
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                technical_score,
                fundamental_score,
                sentiment_score,
                ml_prediction
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(composite_score)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': stock_data['close'].iloc[-1],
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'sentiment_score': sentiment_score,
                'ml_prediction': ml_prediction,
                'composite_score': composite_score,
                'recommendation': recommendation,
                'signals': self._identify_signals(stock_data, technical_score)
            }
            
            # Cache the result
            await self._cache_analysis(symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def _fetch_stock_data(self, symbol: str, period: str = '6mo') -> pd.DataFrame:
        """Fetch stock price data"""
        try:
            # Use yfinance for data (in production, use data ingestion clients)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def _run_technical_analysis(self, stock_data: pd.DataFrame) -> float:
        """Run technical analysis and return score"""
        try:
            indicators = self.technical_analyzer.calculate_all_indicators(stock_data)
            
            # Calculate technical score based on indicators
            score = 0.0
            weight_sum = 0.0
            
            # RSI signal
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30:
                    score += 1.0 * 0.2  # Oversold - bullish
                elif rsi > 70:
                    score -= 1.0 * 0.2  # Overbought - bearish
                else:
                    score += (50 - abs(rsi - 50)) / 50 * 0.2  # Neutral zone
                weight_sum += 0.2
            
            # MACD signal
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_diff = indicators['macd'].iloc[-1] - indicators['macd_signal'].iloc[-1]
                if macd_diff > 0:
                    score += 0.5 * 0.3  # Bullish crossover
                else:
                    score -= 0.5 * 0.3  # Bearish crossover
                weight_sum += 0.3
            
            # Moving average signals
            if 'sma_20' in indicators and 'sma_50' in indicators:
                current_price = stock_data['close'].iloc[-1]
                if current_price > indicators['sma_20'].iloc[-1] > indicators['sma_50'].iloc[-1]:
                    score += 0.8 * 0.3  # Strong uptrend
                elif current_price < indicators['sma_20'].iloc[-1] < indicators['sma_50'].iloc[-1]:
                    score -= 0.8 * 0.3  # Strong downtrend
                weight_sum += 0.3
            
            # Bollinger Bands
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                current_price = stock_data['close'].iloc[-1]
                bb_position = (current_price - indicators['bb_lower'].iloc[-1]) / \
                             (indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1])
                
                if bb_position < 0.2:
                    score += 0.6 * 0.2  # Near lower band - oversold
                elif bb_position > 0.8:
                    score -= 0.6 * 0.2  # Near upper band - overbought
                weight_sum += 0.2
            
            # Normalize score to 0-100
            if weight_sum > 0:
                normalized_score = (score / weight_sum + 1) * 50
            else:
                normalized_score = 50
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return 50.0  # Neutral score on error
    
    async def _run_fundamental_analysis(self, symbol: str) -> float:
        """Run fundamental analysis and return score"""
        try:
            # In production, this would use SEC filings and financial data
            # For now, use basic metrics from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            score = 50.0  # Start with neutral
            
            # P/E Ratio
            pe_ratio = info.get('trailingPE', None)
            if pe_ratio:
                if pe_ratio < 15:
                    score += 10  # Undervalued
                elif pe_ratio > 30:
                    score -= 10  # Overvalued
            
            # P/B Ratio
            pb_ratio = info.get('priceToBook', None)
            if pb_ratio:
                if pb_ratio < 1:
                    score += 10  # Trading below book value
                elif pb_ratio > 3:
                    score -= 10  # Expensive
            
            # Profit Margin
            profit_margin = info.get('profitMargins', None)
            if profit_margin:
                if profit_margin > 0.2:
                    score += 10  # High margin
                elif profit_margin < 0.05:
                    score -= 10  # Low margin
            
            # Revenue Growth
            revenue_growth = info.get('revenueGrowth', None)
            if revenue_growth:
                if revenue_growth > 0.15:
                    score += 10  # Strong growth
                elif revenue_growth < 0:
                    score -= 10  # Declining revenue
            
            # Ensure score is within bounds
            score = max(0, min(100, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {e}")
            return 50.0
    
    async def _run_sentiment_analysis(self, symbol: str) -> float:
        """Run sentiment analysis and return score"""
        try:
            # This would analyze news, social media, etc.
            # For now, return a simulated score
            sentiment = await self.sentiment_analyzer.analyze_symbol(symbol)
            
            # Convert sentiment to score (0-100)
            if sentiment:
                return sentiment.get('composite_score', 50.0)
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return 50.0
    
    async def _run_ml_prediction(self, stock_data: pd.DataFrame) -> Dict:
        """Run ML models for prediction"""
        try:
            # Prepare features
            features = self._prepare_ml_features(stock_data)
            
            if features is None:
                return {'prediction': 'HOLD', 'confidence': 0.5}
            
            # Get prediction from ML model
            prediction = await self.ml_manager.predict(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {'prediction': 'HOLD', 'confidence': 0.5}
    
    def _prepare_ml_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            # Calculate technical indicators as features
            features = pd.DataFrame()
            
            # Price features
            features['returns_1d'] = stock_data['close'].pct_change(1)
            features['returns_5d'] = stock_data['close'].pct_change(5)
            features['returns_20d'] = stock_data['close'].pct_change(20)
            
            # Volume features
            features['volume_ratio'] = stock_data['volume'] / stock_data['volume'].rolling(20).mean()
            
            # Volatility
            features['volatility'] = stock_data['close'].pct_change().rolling(20).std()
            
            # Remove NaN values
            features = features.dropna()
            
            if features.empty:
                return None
            
            # Return last row (most recent features)
            return features.iloc[[-1]]
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
    
    def _calculate_composite_score(self, technical: float, fundamental: float,
                                  sentiment: float, ml_prediction: Dict) -> float:
        """Calculate weighted composite score"""
        
        # Convert ML prediction to score
        ml_score = 50.0
        if ml_prediction['prediction'] == 'BUY':
            ml_score = 50 + ml_prediction['confidence'] * 50
        elif ml_prediction['prediction'] == 'SELL':
            ml_score = 50 - ml_prediction['confidence'] * 50
        
        # Weighted average
        weights = {
            'technical': 0.3,
            'fundamental': 0.25,
            'sentiment': 0.2,
            'ml': 0.25
        }
        
        composite = (
            technical * weights['technical'] +
            fundamental * weights['fundamental'] +
            sentiment * weights['sentiment'] +
            ml_score * weights['ml']
        )
        
        return composite
    
    def _generate_recommendation(self, composite_score: float) -> Dict:
        """Generate trading recommendation based on composite score"""
        
        if composite_score >= 70:
            action = 'STRONG_BUY'
            confidence = (composite_score - 70) / 30
        elif composite_score >= 60:
            action = 'BUY'
            confidence = (composite_score - 60) / 10
        elif composite_score <= 30:
            action = 'STRONG_SELL'
            confidence = (30 - composite_score) / 30
        elif composite_score <= 40:
            action = 'SELL'
            confidence = (40 - composite_score) / 10
        else:
            action = 'HOLD'
            confidence = 1 - abs(composite_score - 50) / 10
        
        return {
            'action': action,
            'confidence': min(1.0, confidence),
            'score': composite_score
        }
    
    def _identify_signals(self, stock_data: pd.DataFrame, technical_score: float) -> List[str]:
        """Identify specific trading signals"""
        signals = []
        
        # Price breakout
        high_20d = stock_data['high'].rolling(20).max()
        if stock_data['close'].iloc[-1] > high_20d.iloc[-2]:
            signals.append('BREAKOUT_20D_HIGH')
        
        # Volume spike
        avg_volume = stock_data['volume'].rolling(20).mean()
        if stock_data['volume'].iloc[-1] > avg_volume.iloc[-1] * 2:
            signals.append('VOLUME_SPIKE')
        
        # Technical signals
        if technical_score > 70:
            signals.append('STRONG_TECHNICAL')
        elif technical_score < 30:
            signals.append('WEAK_TECHNICAL')
        
        return signals
    
    def _rank_opportunities(self, results: List[Dict]) -> List[Dict]:
        """Rank and filter scan results"""
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        # Sort by composite score
        sorted_results = sorted(valid_results, 
                              key=lambda x: x['composite_score'], 
                              reverse=True)
        
        # Add ranking
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1
        
        # Return top opportunities
        return sorted_results[:100]  # Top 100 opportunities
    
    async def _get_cached_analysis(self, symbol: str) -> Optional[Dict]:
        """Get cached analysis if available"""
        if self.redis_client:
            try:
                key = f"analysis:{symbol}:{datetime.now().date()}"
                cached = await self.redis_client.get(key)
                if cached:
                    import json
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Cache retrieval error: {e}")
        return None
    
    async def _cache_analysis(self, symbol: str, analysis: Dict) -> None:
        """Cache analysis results"""
        if self.redis_client:
            try:
                key = f"analysis:{symbol}:{datetime.now().date()}"
                import json
                await self.redis_client.setex(
                    key,
                    86400,  # 24 hours
                    json.dumps(analysis)
                )
            except Exception as e:
                logger.error(f"Cache storage error: {e}")
    
    async def _cache_results(self, results: List[Dict]) -> None:
        """Cache scan results"""
        if self.redis_client:
            try:
                key = f"scan_results:{datetime.now().date()}"
                import json
                await self.redis_client.setex(
                    key,
                    86400,  # 24 hours
                    json.dumps(results)
                )
            except Exception as e:
                logger.error(f"Cache storage error: {e}")
