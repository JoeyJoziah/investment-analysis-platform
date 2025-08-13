"""
Predictive Cache Warming System
Advanced cache warming with machine learning predictions and market event awareness.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

from backend.utils.cache_warming import CacheWarmingStrategy, WarmingPriority
from backend.utils.enhanced_cache_config import intelligent_cache_manager, StockTier
from backend.utils.monitoring import metrics

logger = logging.getLogger(__name__)


class MarketEvent:
    """Market events that trigger cache warming."""
    
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    EARNINGS_SEASON = "earnings_season"
    HIGH_VOLATILITY = "high_volatility"
    NEWS_SPIKE = "news_spike"
    VOLUME_SPIKE = "volume_spike"


class PredictiveWarmingModel:
    """
    Machine learning model for predicting which stocks to warm.
    """
    
    def __init__(self):
        self.access_predictor = None
        self.volume_predictor = None
        self.is_trained = False
        self.feature_history = defaultdict(deque)
        self.access_history = defaultdict(deque)
        self.last_training = None
        self.training_interval = timedelta(hours=24)
        
    def collect_features(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        """Collect features for prediction model."""
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        
        # Time-based features
        features = {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'is_market_hours': 1.0 if 9.5 <= hour <= 16 else 0.0,
            'is_premarket': 1.0 if 4 <= hour < 9.5 else 0.0,
            'is_afterhours': 1.0 if 16 < hour <= 20 else 0.0,
            'is_weekend': 1.0 if day_of_week >= 5 else 0.0
        }
        
        # Historical access patterns
        recent_accesses = self.get_recent_access_count(symbol, hours=24)
        features['recent_access_count'] = float(recent_accesses)
        features['access_trend'] = self.calculate_access_trend(symbol)
        
        # Market context features
        features['market_volatility'] = self.get_market_volatility_score()
        features['earnings_season'] = self.is_earnings_season()
        
        return features
    
    def record_access(self, symbol: str, timestamp: datetime):
        """Record access for training data."""
        self.access_history[symbol].append(timestamp)
        
        # Keep only recent history
        cutoff = timestamp - timedelta(days=7)
        while (self.access_history[symbol] and 
               self.access_history[symbol][0] < cutoff):
            self.access_history[symbol].popleft()
    
    def get_recent_access_count(self, symbol: str, hours: int = 24) -> int:
        """Get recent access count for a symbol."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(1 for ts in self.access_history[symbol] if ts >= cutoff)
    
    def calculate_access_trend(self, symbol: str) -> float:
        """Calculate access trend (increasing/decreasing)."""
        if len(self.access_history[symbol]) < 10:
            return 0.0
        
        # Get hourly access counts for last 24 hours
        now = datetime.now()
        hourly_counts = []
        
        for i in range(24):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)
            count = sum(1 for ts in self.access_history[symbol] 
                       if hour_start <= ts < hour_end)
            hourly_counts.append(count)
        
        # Calculate trend using linear regression
        if sum(hourly_counts) == 0:
            return 0.0
        
        x = np.arange(len(hourly_counts)).reshape(-1, 1)
        y = np.array(hourly_counts)
        
        try:
            model = LinearRegression().fit(x, y)
            return float(model.coef_[0])
        except Exception:
            return 0.0
    
    def get_market_volatility_score(self) -> float:
        """Get current market volatility score (0-1)."""
        # This would integrate with market data
        # For now, return a placeholder
        return 0.5
    
    def is_earnings_season(self) -> float:
        """Check if it's earnings season."""
        now = datetime.now()
        # Earnings seasons: Jan, Apr, Jul, Oct
        earnings_months = [1, 4, 7, 10]
        return 1.0 if now.month in earnings_months else 0.0
    
    async def train_model(self):
        """Train the predictive model with historical data."""
        try:
            # Collect training data
            features_list = []
            targets_list = []
            
            for symbol in self.access_history:
                if len(self.access_history[symbol]) < 50:
                    continue
                
                # Create training samples
                accesses = list(self.access_history[symbol])
                for i in range(1, len(accesses)):
                    prev_time = accesses[i-1]
                    curr_time = accesses[i]
                    
                    # Features from previous access
                    features = self.collect_features(symbol, prev_time)
                    features_list.append(list(features.values()))
                    
                    # Target: time until next access (normalized)
                    time_diff = (curr_time - prev_time).total_seconds() / 3600  # hours
                    target = min(time_diff, 24)  # Cap at 24 hours
                    targets_list.append(target)
            
            if len(features_list) < 100:
                logger.warning("Not enough training data for predictive model")
                return
            
            # Train model
            X = np.array(features_list)
            y = np.array(targets_list)
            
            self.access_predictor = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            
            # Convert regression to classification (predict if access within next hour)
            y_binary = (y <= 1.0).astype(int)
            self.access_predictor.fit(X, y_binary)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            logger.info(f"Trained predictive warming model with {len(features_list)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train predictive model: {e}")
    
    def predict_warming_candidates(self, symbols: List[str], 
                                 top_n: int = 100) -> List[Tuple[str, float]]:
        """Predict which symbols should be warmed."""
        if not self.is_trained or not self.access_predictor:
            return [(s, 0.5) for s in symbols[:top_n]]
        
        try:
            predictions = []
            now = datetime.now()
            
            for symbol in symbols:
                features = self.collect_features(symbol, now)
                feature_vector = np.array([list(features.values())])
                
                # Get prediction probability
                prob = self.access_predictor.predict_proba(feature_vector)[0][1]
                predictions.append((symbol, prob))
            
            # Sort by prediction probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:top_n]
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return [(s, 0.5) for s in symbols[:top_n]]


class MarketEventDetector:
    """
    Detect market events that should trigger cache warming.
    """
    
    def __init__(self):
        self.event_history = defaultdict(list)
        self.volume_baseline = {}
        self.volatility_baseline = {}
        
    def detect_events(self, market_data: Dict[str, Any]) -> List[str]:
        """Detect current market events."""
        events = []
        now = datetime.now()
        
        # Market open/close events
        if self.is_market_open_time():
            events.append(MarketEvent.MARKET_OPEN)
        elif self.is_market_close_time():
            events.append(MarketEvent.MARKET_CLOSE)
        
        # Volume spikes
        if self.detect_volume_spike(market_data):
            events.append(MarketEvent.VOLUME_SPIKE)
        
        # Volatility spikes
        if self.detect_volatility_spike(market_data):
            events.append(MarketEvent.HIGH_VOLATILITY)
        
        # Earnings season
        if self.is_earnings_season():
            events.append(MarketEvent.EARNINGS_SEASON)
        
        return events
    
    def is_market_open_time(self) -> bool:
        """Check if it's near market open time."""
        now = datetime.now()
        # Market opens at 9:30 AM EST
        return now.hour == 9 and 25 <= now.minute <= 35
    
    def is_market_close_time(self) -> bool:
        """Check if it's near market close time."""
        now = datetime.now()
        # Market closes at 4:00 PM EST
        return now.hour == 16 and 0 <= now.minute <= 10
    
    def detect_volume_spike(self, market_data: Dict[str, Any]) -> bool:
        """Detect unusual volume spikes."""
        # Placeholder implementation
        return False
    
    def detect_volatility_spike(self, market_data: Dict[str, Any]) -> bool:
        """Detect high volatility periods."""
        # Placeholder implementation
        return False
    
    def is_earnings_season(self) -> bool:
        """Check if it's earnings season."""
        now = datetime.now()
        earnings_months = [1, 4, 7, 10]
        return now.month in earnings_months


class PredictiveCacheWarming(CacheWarmingStrategy):
    """
    Enhanced cache warming with predictive capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_model = PredictiveWarmingModel()
        self.event_detector = MarketEventDetector()
        self.warming_scheduler = WarmingScheduler()
        
        # Enhanced configuration
        self.predictive_warming_ratio = 0.3  # 30% of warming based on predictions
        self.event_driven_warming_ratio = 0.2  # 20% event-driven
        self.tier_warming_ratio = 0.5  # 50% tier-based
        
        # Performance tracking
        self.predictive_hits = 0
        self.predictive_misses = 0
        
    async def initialize(self):
        """Initialize predictive warming system."""
        # Load existing model if available
        await self.load_model()
        
        # Start background tasks
        asyncio.create_task(self.model_training_loop())
        asyncio.create_task(self.event_monitoring_loop())
        
        logger.info("Predictive cache warming system initialized")
    
    async def warm_predictive_caches(self) -> Dict[str, Any]:
        """
        Perform predictive cache warming based on ML predictions.
        """
        results = {'warmed': [], 'failed': [], 'predicted': []}
        
        try:
            # Get all available stocks
            all_stocks = await self.get_all_stocks()
            
            # Get predictive warming candidates
            if self.prediction_model.is_trained:
                predicted_stocks = self.prediction_model.predict_warming_candidates(
                    all_stocks, top_n=200
                )
                results['predicted'] = [stock for stock, prob in predicted_stocks 
                                      if prob > 0.6]
            else:
                # Fallback to high-frequency stocks
                results['predicted'] = intelligent_cache_manager.get_high_frequency_stocks()
            
            # Warm predicted stocks
            if results['predicted']:
                batch_results = await self._warm_batch(
                    results['predicted'][:50],  # Limit batch size
                    data_types=['price', 'fundamentals'],
                    priority=WarmingPriority.HIGH
                )
                
                results['warmed'].extend(batch_results['success'])
                results['failed'].extend(batch_results['failed'])
                
                # Track predictions
                for stock in batch_results['success']:
                    self.predictive_hits += 1
                    intelligent_cache_manager.track_access(stock, 'predictive_warm')
            
            logger.info(f"Predictive warming: {len(results['warmed'])} warmed, "
                       f"{len(results['predicted'])} predicted")
            
        except Exception as e:
            logger.error(f"Predictive warming failed: {e}")
        
        return results
    
    async def warm_event_driven_caches(self, events: List[str]) -> Dict[str, Any]:
        """
        Warm caches based on detected market events.
        """
        results = {'warmed': [], 'failed': [], 'events': events}
        
        try:
            warming_stocks = set()
            
            for event in events:
                if event == MarketEvent.MARKET_OPEN:
                    # Warm high-tier stocks before market open
                    critical_stocks = list(intelligent_cache_manager.tier_mappings.items())
                    critical_stocks = [s for s, t in critical_stocks 
                                     if t == StockTier.CRITICAL][:100]
                    warming_stocks.update(critical_stocks)
                
                elif event == MarketEvent.EARNINGS_SEASON:
                    # Warm stocks with upcoming earnings
                    earnings_stocks = await self.get_earnings_stocks()
                    warming_stocks.update(earnings_stocks[:50])
                
                elif event == MarketEvent.VOLUME_SPIKE:
                    # Warm high-volume stocks
                    volume_stocks = await self.get_high_volume_stocks()
                    warming_stocks.update(volume_stocks[:30])
            
            # Perform warming
            if warming_stocks:
                batch_results = await self._warm_batch(
                    list(warming_stocks),
                    data_types=['price', 'technical', 'news'],
                    priority=WarmingPriority.CRITICAL
                )
                
                results['warmed'].extend(batch_results['success'])
                results['failed'].extend(batch_results['failed'])
                
                self._metrics['market_events_processed'] += len(events)
            
            logger.info(f"Event-driven warming for {events}: "
                       f"{len(results['warmed'])} stocks warmed")
            
        except Exception as e:
            logger.error(f"Event-driven warming failed: {e}")
        
        return results
    
    async def adaptive_warming_strategy(self) -> Dict[str, Any]:
        """
        Adaptive warming strategy combining predictions, events, and tiers.
        """
        results = {
            'predictive': {'warmed': [], 'failed': []},
            'event_driven': {'warmed': [], 'failed': []},
            'tier_based': {'warmed': [], 'failed': []}
        }
        
        # Detect current market events
        market_data = await self.get_current_market_data()
        events = self.event_detector.detect_events(market_data)
        
        # Allocate warming budget based on ratios
        total_budget = 500  # Maximum stocks to warm per cycle
        
        predictive_budget = int(total_budget * self.predictive_warming_ratio)
        event_budget = int(total_budget * self.event_driven_warming_ratio)
        tier_budget = int(total_budget * self.tier_warming_ratio)
        
        # Execute warming strategies
        if predictive_budget > 0:
            results['predictive'] = await self.warm_predictive_caches()
        
        if event_budget > 0 and events:
            results['event_driven'] = await self.warm_event_driven_caches(events)
        
        if tier_budget > 0:
            results['tier_based'] = await self.warm_critical_caches()
        
        # Update metrics
        total_warmed = (len(results['predictive']['warmed']) + 
                       len(results['event_driven']['warmed']) + 
                       len(results['tier_based']['warmed']))
        
        self._metrics['caches_warmed'] += total_warmed
        self._metrics['last_warming'] = datetime.now()
        
        return results
    
    async def model_training_loop(self):
        """Background loop for model training."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Train model if needed
                if (not self.prediction_model.last_training or 
                    datetime.now() - self.prediction_model.last_training > 
                    self.prediction_model.training_interval):
                    
                    await self.prediction_model.train_model()
                    await self.save_model()
                
            except Exception as e:
                logger.error(f"Model training loop error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def event_monitoring_loop(self):
        """Background loop for event monitoring."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Get market data and detect events
                market_data = await self.get_current_market_data()
                events = self.event_detector.detect_events(market_data)
                
                # Trigger event-driven warming if needed
                if events:
                    asyncio.create_task(self.warm_event_driven_caches(events))
                
            except Exception as e:
                logger.error(f"Event monitoring error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def get_all_stocks(self) -> List[str]:
        """Get list of all available stocks."""
        # This would integrate with the stock database
        return list(intelligent_cache_manager.tier_mappings.keys())
    
    async def get_earnings_stocks(self) -> List[str]:
        """Get stocks with upcoming earnings."""
        # Placeholder implementation
        return []
    
    async def get_high_volume_stocks(self) -> List[str]:
        """Get stocks with high volume."""
        # Placeholder implementation
        return []
    
    async def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for event detection."""
        # Placeholder implementation
        return {}
    
    async def load_model(self):
        """Load saved prediction model."""
        try:
            # Implementation would load from file system
            pass
        except Exception as e:
            logger.info(f"No saved model found: {e}")
    
    async def save_model(self):
        """Save prediction model."""
        try:
            # Implementation would save to file system
            pass
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class WarmingScheduler:
    """
    Schedule cache warming based on market calendar and patterns.
    """
    
    def __init__(self):
        self.warming_schedule = {}
        self.next_warming_time = None
    
    def calculate_optimal_warming_time(self) -> datetime:
        """Calculate optimal time for next warming cycle."""
        now = datetime.now()
        
        # Schedule before market open (9:30 AM EST)
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        
        # If market already opened today, schedule for tomorrow
        if now >= market_open:
            market_open += timedelta(days=1)
        
        # Warm 1 hour before market open
        warming_time = market_open - timedelta(hours=1)
        
        return warming_time


# Global predictive cache warming instance
predictive_cache_warmer = PredictiveCacheWarming()