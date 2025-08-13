"""
Social Sentiment Analysis Engine

Analyzes sentiment from free social media APIs:
- Reddit (via PRAW - free)
- Twitter/X (via API v2 free tier - 2M tweets/month)
- StockTwits (free tier)
- Discord (if configured)

Uses FinBERT and VADER for sentiment analysis
"""

import asyncio
import aiohttp
import praw
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import json

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy

from backend.utils.cache import CacheManager
from backend.utils.rate_limiter import RateLimiter
from backend.utils.cost_monitor import CostMonitor

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Social media post data structure"""
    platform: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    engagement: int  # likes, upvotes, retweets
    followers: Optional[int] = None
    verified: bool = False
    
@dataclass 
class SentimentResult:
    """Sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    label: str

class SocialSentimentAnalyzer:
    """
    Comprehensive social sentiment analysis using free APIs
    
    Features:
    - Multi-platform data collection (Reddit, Twitter, StockTwits)
    - Multiple sentiment models (FinBERT, VADER, TextBlob)
    - Real-time and historical analysis
    - Influencer weighting
    - Sentiment trend analysis
    - Cost-optimized API usage
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = CacheManager()
        self.cost_monitor = CostMonitor()
        
        # Initialize rate limiters for each platform
        self.reddit_limiter = RateLimiter(calls=100, period=60)  # Reddit: 100/min
        self.twitter_limiter = RateLimiter(calls=300, period=900)  # Twitter: 300/15min
        self.stocktwits_limiter = RateLimiter(calls=200, period=3600)  # StockTwits: 200/hour
        
        # Initialize sentiment analyzers
        self._init_sentiment_models()
        
        # Initialize API clients
        self._init_api_clients()
        
    def _init_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # FinBERT for financial sentiment (best for stocks)
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=-1  # CPU only to save costs
            )
            
            # VADER for social media sentiment
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info("Sentiment models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
            # Fallback to VADER only
            self.finbert_analyzer = None
            self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def _init_api_clients(self):
        """Initialize social media API clients"""
        # Reddit API (free)
        try:
            self.reddit = praw.Reddit(
                client_id=self.config.get('reddit_client_id'),
                client_secret=self.config.get('reddit_client_secret'),
                user_agent=self.config.get('reddit_user_agent', 'InvestmentAnalyzer/1.0')
            )
            logger.info("Reddit API client initialized")
        except Exception as e:
            logger.warning(f"Reddit API initialization failed: {e}")
            self.reddit = None
            
        # Twitter API v2 (free tier)
        try:
            self.twitter = tweepy.Client(
                bearer_token=self.config.get('twitter_bearer_token'),
                wait_on_rate_limit=True
            )
            logger.info("Twitter API client initialized")
        except Exception as e:
            logger.warning(f"Twitter API initialization failed: {e}")
            self.twitter = None
    
    async def analyze_stock_sentiment(
        self,
        symbol: str,
        days_back: int = 7,
        platforms: List[str] = None
    ) -> Dict:
        """
        Analyze social sentiment for a stock across multiple platforms
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            platforms: List of platforms to analyze ['reddit', 'twitter', 'stocktwits']
            
        Returns:
            Comprehensive sentiment analysis results
        """
        if platforms is None:
            platforms = ['reddit', 'twitter', 'stocktwits']
            
        # Check cache first
        cache_key = f"social_sentiment:{symbol}:{days_back}:{'-'.join(platforms)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        results = {
            'symbol': symbol,
            'analysis_date': datetime.now(),
            'platforms': {},
            'aggregate': {},
            'trends': {},
            'insights': []
        }
        
        # Collect data from each platform
        tasks = []
        if 'reddit' in platforms and self.reddit:
            tasks.append(self._analyze_reddit_sentiment(symbol, days_back))
        if 'twitter' in platforms and self.twitter:
            tasks.append(self._analyze_twitter_sentiment(symbol, days_back))
        if 'stocktwits' in platforms:
            tasks.append(self._analyze_stocktwits_sentiment(symbol, days_back))
            
        platform_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results from each platform
        valid_results = []
        for i, result in enumerate(platform_results):
            platform = platforms[i] if i < len(platforms) else f"platform_{i}"
            if isinstance(result, Exception):
                logger.error(f"Error analyzing {platform}: {result}")
                continue
            if result:
                results['platforms'][platform] = result
                valid_results.append(result)
        
        if not valid_results:
            logger.warning(f"No valid sentiment data found for {symbol}")
            return results
            
        # Calculate aggregate sentiment
        results['aggregate'] = self._calculate_aggregate_sentiment(valid_results)
        
        # Analyze trends
        results['trends'] = self._analyze_sentiment_trends(valid_results, days_back)
        
        # Generate insights
        results['insights'] = self._generate_sentiment_insights(results)
        
        # Cache results
        await self.cache.set(cache_key, results, expire=900)  # 15 minutes
        
        return results
    
    async def _analyze_reddit_sentiment(self, symbol: str, days_back: int) -> Dict:
        """Analyze Reddit sentiment for a stock"""
        if not self.reddit:
            return None
            
        await self.reddit_limiter.acquire()
        
        try:
            posts = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Search multiple subreddits
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting', 
                         'wallstreetbets', 'StockMarket', 'SecurityAnalysis']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for mentions of the stock
                    search_queries = [symbol, f"${symbol}", symbol.lower()]
                    
                    for query in search_queries:
                        for post in subreddit.search(query, time_filter='week', limit=50):
                            post_date = datetime.fromtimestamp(post.created_utc)
                            if post_date < cutoff_date:
                                continue
                                
                            posts.append(SocialPost(
                                platform='reddit',
                                post_id=post.id,
                                author=post.author.name if post.author else 'deleted',
                                content=f"{post.title} {post.selftext}",
                                timestamp=post_date,
                                engagement=post.score + post.num_comments,
                                verified=False
                            ))
                            
                except Exception as e:
                    logger.warning(f"Error searching subreddit {subreddit_name}: {e}")
                    continue
            
            if not posts:
                return {'platform': 'reddit', 'posts_analyzed': 0, 'sentiment': None}
            
            # Analyze sentiment of collected posts
            sentiment_scores = []
            for post in posts[:100]:  # Limit to avoid API costs
                sentiment = await self._analyze_text_sentiment(post.content)
                sentiment_scores.append({
                    'post_id': post.post_id,
                    'sentiment': sentiment,
                    'engagement': post.engagement,
                    'timestamp': post.timestamp
                })
            
            # Calculate weighted sentiment (by engagement)
            weighted_sentiment = self._calculate_weighted_sentiment(sentiment_scores)
            
            return {
                'platform': 'reddit',
                'posts_analyzed': len(posts),
                'sentiment': weighted_sentiment,
                'raw_scores': sentiment_scores,
                'top_posts': sorted(posts, key=lambda x: x.engagement, reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
            return None
    
    async def _analyze_twitter_sentiment(self, symbol: str, days_back: int) -> Dict:
        """Analyze Twitter sentiment for a stock"""
        if not self.twitter:
            return None
            
        await self.twitter_limiter.acquire()
        
        try:
            # Search for tweets about the stock
            query = f"${symbol} OR {symbol} -is:retweet lang:en"
            start_time = datetime.now() - timedelta(days=days_back)
            
            tweets = tweepy.Paginator(
                self.twitter.search_recent_tweets,
                query=query,
                start_time=start_time,
                tweet_fields=['created_at', 'author_id', 'public_metrics'],
                max_results=100
            ).flatten(limit=500)  # Stay within free tier limits
            
            posts = []
            for tweet in tweets:
                posts.append(SocialPost(
                    platform='twitter',
                    post_id=tweet.id,
                    author=tweet.author_id,
                    content=tweet.text,
                    timestamp=tweet.created_at,
                    engagement=tweet.public_metrics['like_count'] + 
                              tweet.public_metrics['retweet_count'],
                    verified=False
                ))
            
            if not posts:
                return {'platform': 'twitter', 'posts_analyzed': 0, 'sentiment': None}
            
            # Analyze sentiment
            sentiment_scores = []
            for post in posts:
                sentiment = await self._analyze_text_sentiment(post.content)
                sentiment_scores.append({
                    'post_id': post.post_id,
                    'sentiment': sentiment,
                    'engagement': post.engagement,
                    'timestamp': post.timestamp
                })
            
            weighted_sentiment = self._calculate_weighted_sentiment(sentiment_scores)
            
            return {
                'platform': 'twitter',
                'posts_analyzed': len(posts),
                'sentiment': weighted_sentiment,
                'raw_scores': sentiment_scores,
                'top_posts': sorted(posts, key=lambda x: x.engagement, reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
            return None
    
    async def _analyze_stocktwits_sentiment(self, symbol: str, days_back: int) -> Dict:
        """Analyze StockTwits sentiment using their free API"""
        await self.stocktwits_limiter.acquire()
        
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"StockTwits API error: {response.status}")
                        return None
                        
                    data = await response.json()
                    
            if 'messages' not in data:
                return {'platform': 'stocktwits', 'posts_analyzed': 0, 'sentiment': None}
            
            posts = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for message in data['messages']:
                created_at = datetime.strptime(message['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                if created_at < cutoff_date:
                    continue
                    
                posts.append(SocialPost(
                    platform='stocktwits',
                    post_id=str(message['id']),
                    author=message['user']['username'],
                    content=message['body'],
                    timestamp=created_at,
                    engagement=message.get('likes', {}).get('total', 0),
                    followers=message['user'].get('followers', 0),
                    verified=message['user'].get('verified', False)
                ))
            
            if not posts:
                return {'platform': 'stocktwits', 'posts_analyzed': 0, 'sentiment': None}
            
            # Analyze sentiment
            sentiment_scores = []
            for post in posts:
                sentiment = await self._analyze_text_sentiment(post.content)
                sentiment_scores.append({
                    'post_id': post.post_id,
                    'sentiment': sentiment,
                    'engagement': post.engagement,
                    'timestamp': post.timestamp
                })
            
            weighted_sentiment = self._calculate_weighted_sentiment(sentiment_scores)
            
            return {
                'platform': 'stocktwits',
                'posts_analyzed': len(posts),
                'sentiment': weighted_sentiment,
                'raw_scores': sentiment_scores,
                'top_posts': sorted(posts, key=lambda x: x.engagement, reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing StockTwits sentiment: {e}")
            return None
    
    async def _analyze_text_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text using multiple models"""
        # Clean text
        text = self._clean_text(text)
        
        # Use FinBERT if available (better for financial text)
        if self.finbert_analyzer and len(text.split()) > 3:
            try:
                finbert_result = self.finbert_analyzer(text[:512])  # Truncate to avoid token limits
                finbert_score = finbert_result[0]
                
                # Convert to standardized format
                if finbert_score['label'] == 'positive':
                    positive = finbert_score['score']
                    negative = 0.0
                elif finbert_score['label'] == 'negative':
                    positive = 0.0
                    negative = finbert_score['score']
                else:
                    positive = negative = 0.0
                    
                neutral = 1.0 - positive - negative
                compound = positive - negative
                
                return SentimentResult(
                    positive=positive,
                    negative=negative,
                    neutral=neutral,
                    compound=compound,
                    confidence=finbert_score['score'],
                    label=finbert_score['label']
                )
                
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {e}")
        
        # Fallback to VADER
        scores = self.vader_analyzer.polarity_scores(text)
        return SentimentResult(
            positive=scores['pos'],
            negative=scores['neg'],
            neutral=scores['neu'],
            compound=scores['compound'],
            confidence=abs(scores['compound']),
            label='positive' if scores['compound'] > 0.1 else 'negative' if scores['compound'] < -0.1 else 'neutral'
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags for cleaner analysis
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove emojis (they can skew FinBERT)
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _calculate_weighted_sentiment(self, sentiment_scores: List[Dict]) -> Dict:
        """Calculate weighted sentiment based on engagement"""
        if not sentiment_scores:
            return None
            
        total_weight = 0
        weighted_positive = 0
        weighted_negative = 0
        weighted_neutral = 0
        weighted_compound = 0
        
        for score in sentiment_scores:
            weight = max(1, score['engagement'])  # Minimum weight of 1
            sentiment = score['sentiment']
            
            total_weight += weight
            weighted_positive += sentiment.positive * weight
            weighted_negative += sentiment.negative * weight
            weighted_neutral += sentiment.neutral * weight
            weighted_compound += sentiment.compound * weight
        
        if total_weight == 0:
            return None
            
        return {
            'positive': weighted_positive / total_weight,
            'negative': weighted_negative / total_weight,
            'neutral': weighted_neutral / total_weight,
            'compound': weighted_compound / total_weight,
            'confidence': abs(weighted_compound / total_weight),
            'total_engagement': total_weight,
            'sample_size': len(sentiment_scores)
        }
    
    def _calculate_aggregate_sentiment(self, platform_results: List[Dict]) -> Dict:
        """Calculate aggregate sentiment across all platforms"""
        if not platform_results:
            return None
            
        total_posts = 0
        total_engagement = 0
        weighted_compound = 0
        weighted_positive = 0
        weighted_negative = 0
        
        platform_weights = {'reddit': 1.0, 'twitter': 0.8, 'stocktwits': 1.2}  # StockTwits weighted higher for financial relevance
        
        for result in platform_results:
            if not result.get('sentiment'):
                continue
                
            platform = result['platform']
            sentiment = result['sentiment']
            posts = result['posts_analyzed']
            engagement = sentiment.get('total_engagement', posts)
            
            # Apply platform weight
            weight = platform_weights.get(platform, 1.0) * engagement
            
            total_posts += posts
            total_engagement += engagement
            weighted_compound += sentiment['compound'] * weight
            weighted_positive += sentiment['positive'] * weight
            weighted_negative += sentiment['negative'] * weight
        
        if total_engagement == 0:
            return None
            
        compound_score = weighted_compound / total_engagement
        
        return {
            'compound_score': compound_score,
            'positive_ratio': weighted_positive / total_engagement,
            'negative_ratio': weighted_negative / total_engagement,
            'neutral_ratio': 1.0 - (weighted_positive + weighted_negative) / total_engagement,
            'confidence': abs(compound_score),
            'label': 'bullish' if compound_score > 0.1 else 'bearish' if compound_score < -0.1 else 'neutral',
            'total_posts_analyzed': total_posts,
            'total_engagement': total_engagement,
            'platforms_analyzed': len(platform_results)
        }
    
    def _analyze_sentiment_trends(self, platform_results: List[Dict], days_back: int) -> Dict:
        """Analyze sentiment trends over time"""
        # Combine all posts from all platforms
        all_posts = []
        for result in platform_results:
            if 'raw_scores' in result:
                for score_data in result['raw_scores']:
                    all_posts.append(score_data)
        
        if not all_posts:
            return {}
            
        # Group by day and calculate daily sentiment
        daily_sentiment = {}
        for post in all_posts:
            date_key = post['timestamp'].date()
            if date_key not in daily_sentiment:
                daily_sentiment[date_key] = []
            daily_sentiment[date_key].append(post['sentiment'].compound)
        
        # Calculate trend metrics
        dates = sorted(daily_sentiment.keys())
        if len(dates) < 2:
            return {'trend': 'insufficient_data'}
            
        daily_averages = [np.mean(daily_sentiment[date]) for date in dates]
        
        # Calculate trend direction
        if len(daily_averages) >= 3:
            trend_slope = np.polyfit(range(len(daily_averages)), daily_averages, 1)[0]
            trend_direction = 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
        else:
            trend_direction = 'stable'
            
        # Calculate volatility
        sentiment_volatility = np.std(daily_averages) if len(daily_averages) > 1 else 0
        
        return {
            'trend_direction': trend_direction,
            'sentiment_volatility': float(sentiment_volatility),
            'daily_averages': {str(date): avg for date, avg in zip(dates, daily_averages)},
            'latest_sentiment': daily_averages[-1] if daily_averages else 0,
            'sentiment_momentum': daily_averages[-1] - daily_averages[0] if len(daily_averages) >= 2 else 0
        }
    
    def _generate_sentiment_insights(self, results: Dict) -> List[str]:
        """Generate actionable insights from sentiment analysis"""
        insights = []
        
        aggregate = results.get('aggregate', {})
        trends = results.get('trends', {})
        platforms = results.get('platforms', {})
        
        if not aggregate:
            return ['Insufficient data for sentiment analysis']
            
        # Overall sentiment insights
        compound = aggregate.get('compound_score', 0)
        confidence = aggregate.get('confidence', 0)
        
        if confidence > 0.3:  # High confidence insights
            if compound > 0.2:
                insights.append(f"Strong bullish sentiment detected (score: {compound:.2f})")
            elif compound < -0.2:
                insights.append(f"Strong bearish sentiment detected (score: {compound:.2f})")
                
        # Trend insights
        trend_direction = trends.get('trend_direction')
        if trend_direction == 'improving':
            insights.append("Sentiment trend is improving over the analysis period")
        elif trend_direction == 'declining':
            insights.append("Sentiment trend is declining over the analysis period")
            
        # Platform-specific insights
        platform_names = list(platforms.keys())
        if len(platform_names) > 1:
            # Find platform with strongest sentiment
            strongest_platform = max(
                platforms.items(),
                key=lambda x: abs(x[1].get('sentiment', {}).get('compound', 0))
            )
            insights.append(f"Strongest sentiment signal comes from {strongest_platform[0]}")
            
        # Volume insights
        total_posts = aggregate.get('total_posts_analyzed', 0)
        if total_posts > 100:
            insights.append(f"High discussion volume detected ({total_posts} posts analyzed)")
        elif total_posts < 20:
            insights.append("Low discussion volume - sentiment may be less reliable")
            
        # Volatility insights
        volatility = trends.get('sentiment_volatility', 0)
        if volatility > 0.3:
            insights.append("High sentiment volatility indicates divided opinion")
            
        return insights
    
    async def get_sentiment_alerts(self, symbol: str, thresholds: Dict) -> List[Dict]:
        """Generate sentiment-based alerts"""
        sentiment_data = await self.analyze_stock_sentiment(symbol, days_back=1)
        
        alerts = []
        aggregate = sentiment_data.get('aggregate', {})
        
        if not aggregate:
            return alerts
            
        compound = aggregate.get('compound_score', 0)
        confidence = aggregate.get('confidence', 0)
        
        # Extreme sentiment alerts
        if confidence > 0.5:
            if compound > thresholds.get('bullish_threshold', 0.4):
                alerts.append({
                    'type': 'extreme_bullish_sentiment',
                    'message': f"Extreme bullish sentiment detected for {symbol}",
                    'score': compound,
                    'confidence': confidence,
                    'urgency': 'high'
                })
            elif compound < thresholds.get('bearish_threshold', -0.4):
                alerts.append({
                    'type': 'extreme_bearish_sentiment',
                    'message': f"Extreme bearish sentiment detected for {symbol}",
                    'score': compound,
                    'confidence': confidence,
                    'urgency': 'high'
                })
        
        # Sentiment shift alerts
        trends = sentiment_data.get('trends', {})
        momentum = trends.get('sentiment_momentum', 0)
        
        if abs(momentum) > thresholds.get('momentum_threshold', 0.3):
            alerts.append({
                'type': 'sentiment_shift',
                'message': f"Significant sentiment shift detected for {symbol}",
                'momentum': momentum,
                'direction': 'positive' if momentum > 0 else 'negative',
                'urgency': 'medium'
            })
        
        return alerts