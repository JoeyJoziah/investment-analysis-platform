"""
Advanced Sentiment Analysis Engine using FinBERT and Multi-Source Data
Analyzes news, social media, analyst reports, and alternative text sources
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import re
from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    source: str
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # positive, negative, neutral
    confidence: float
    timestamp: datetime
    keywords: List[str]
    entities: List[str]
    relevance_score: float


class SentimentAnalysisEngine:
    """
    Multi-model sentiment analysis engine optimized for financial text
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.vader = SentimentIntensityAnalyzer()
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.7,
            top=10
        )
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all sentiment models"""
        try:
            # FinBERT - Primary model for financial sentiment
            self.tokenizers['finbert'] = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert",
                cache_dir="./models/finbert"
            )
            self.models['finbert'] = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                cache_dir="./models/finbert"
            ).to(self.device)
            
            # Create pipeline for easier inference
            self.pipelines['finbert'] = pipeline(
                "sentiment-analysis",
                model=self.models['finbert'],
                tokenizer=self.tokenizers['finbert'],
                device=0 if self.device == "cuda" else -1
            )
            
            # Twitter-specific sentiment model
            self.pipelines['twitter'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Sentiment models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
            # Fallback to rule-based only
            self.models = {}
            self.pipelines = {}
    
    async def analyze_sentiment(
        self,
        ticker: str,
        text_data: List[Dict],
        source_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis across multiple sources
        """
        if not source_weights:
            source_weights = {
                'news': 0.4,
                'analyst': 0.3,
                'social': 0.2,
                'insider': 0.1
            }
        
        # Analyze each text item
        all_sentiments = []
        source_sentiments = defaultdict(list)
        
        for item in text_data:
            sentiment = await self._analyze_single_item(ticker, item)
            if sentiment:
                all_sentiments.append(sentiment)
                source_sentiments[sentiment.source].append(sentiment)
        
        # Aggregate results
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.utcnow().isoformat(),
            'item_count': len(all_sentiments),
            'overall_sentiment': self._calculate_overall_sentiment(all_sentiments, source_weights),
            'source_breakdown': self._analyze_by_source(source_sentiments),
            'temporal_analysis': self._analyze_temporal_sentiment(all_sentiments),
            'keyword_analysis': self._analyze_keywords(all_sentiments),
            'entity_analysis': self._analyze_entities(all_sentiments),
            'anomaly_detection': self._detect_sentiment_anomalies(all_sentiments),
            'market_impact_score': self._calculate_market_impact(all_sentiments),
            'signals': self._generate_sentiment_signals(all_sentiments)
        }
        
        return analysis
    
    async def _analyze_single_item(self, ticker: str, item: Dict) -> Optional[SentimentResult]:
        """Analyze sentiment of a single text item"""
        text = item.get('text', '')
        if not text or len(text) < 10:
            return None
        
        source = item.get('source', 'unknown')
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Get sentiment from multiple models
        sentiments = {}
        
        # FinBERT for financial text
        if source in ['news', 'analyst', 'filing'] and self.pipelines.get('finbert'):
            sentiments['finbert'] = self._get_finbert_sentiment(cleaned_text)
        
        # Twitter model for social media
        elif source in ['twitter', 'reddit', 'stocktwits'] and self.pipelines.get('twitter'):
            sentiments['twitter'] = self._get_twitter_sentiment(cleaned_text)
        
        # VADER for general sentiment
        sentiments['vader'] = self._get_vader_sentiment(cleaned_text)
        
        # TextBlob as additional signal
        sentiments['textblob'] = self._get_textblob_sentiment(cleaned_text)
        
        # Combine sentiments
        final_sentiment = self._combine_sentiments(sentiments, source)
        
        # Extract keywords and entities
        keywords = self._extract_keywords(cleaned_text)
        entities = self._extract_entities(cleaned_text, ticker)
        
        # Calculate relevance to ticker
        relevance = self._calculate_relevance(cleaned_text, ticker, entities)
        
        return SentimentResult(
            source=source,
            text=text[:500],  # Truncate for storage
            sentiment_score=final_sentiment['score'],
            sentiment_label=final_sentiment['label'],
            confidence=final_sentiment['confidence'],
            timestamp=item.get('timestamp', datetime.utcnow()),
            keywords=keywords,
            entities=entities,
            relevance_score=relevance
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep sentiment indicators
        text = re.sub(r'[^\w\s\-\!\?\.\,\;\:\$\%]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit length for model input
        if len(text) > 512:
            # Keep beginning and end
            text = text[:256] + ' ... ' + text[-256:]
        
        return text
    
    def _get_finbert_sentiment(self, text: str) -> Dict:
        """Get sentiment using FinBERT model"""
        try:
            results = self.pipelines['finbert'](text, max_length=512, truncation=True)
            
            # FinBERT returns positive, negative, neutral
            sentiment_map = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            # Get the top prediction
            top_result = results[0]
            label = top_result['label'].lower()
            score = sentiment_map.get(label, 0.0)
            confidence = top_result['score']
            
            return {
                'score': score,
                'confidence': confidence,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'label': 'neutral'}
    
    def _get_twitter_sentiment(self, text: str) -> Dict:
        """Get sentiment using Twitter-trained model"""
        try:
            results = self.pipelines['twitter'](text, max_length=512, truncation=True)
            
            # Map labels to scores
            label_map = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            top_result = results[0]
            label = top_result['label'].lower()
            
            # Handle different label formats
            if 'pos' in label:
                score = 1.0
                label = 'positive'
            elif 'neg' in label:
                score = -1.0
                label = 'negative'
            else:
                score = 0.0
                label = 'neutral'
            
            confidence = top_result['score']
            
            return {
                'score': score,
                'confidence': confidence,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'label': 'neutral'}
    
    def _get_vader_sentiment(self, text: str) -> Dict:
        """Get sentiment using VADER (rule-based)"""
        scores = self.vader.polarity_scores(text)
        
        # Determine label and confidence
        if scores['compound'] >= 0.05:
            label = 'positive'
            confidence = scores['pos']
        elif scores['compound'] <= -0.05:
            label = 'negative'
            confidence = scores['neg']
        else:
            label = 'neutral'
            confidence = scores['neu']
        
        return {
            'score': scores['compound'],  # -1 to 1
            'confidence': confidence,
            'label': label
        }
    
    def _get_textblob_sentiment(self, text: str) -> Dict:
        """Get sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # TextBlob subjectivity as inverse of confidence
            confidence = 1.0 - blob.sentiment.subjectivity
            
            return {
                'score': polarity,
                'confidence': confidence,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"TextBlob error: {e}")
            return {'score': 0.0, 'confidence': 0.5, 'label': 'neutral'}
    
    def _combine_sentiments(self, sentiments: Dict, source: str) -> Dict:
        """Combine multiple sentiment scores intelligently"""
        if not sentiments:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'neutral'}
        
        # Weight models based on source type
        model_weights = {
            'news': {'finbert': 0.6, 'vader': 0.2, 'textblob': 0.2},
            'analyst': {'finbert': 0.7, 'vader': 0.2, 'textblob': 0.1},
            'twitter': {'twitter': 0.5, 'vader': 0.3, 'textblob': 0.2},
            'reddit': {'twitter': 0.4, 'vader': 0.4, 'textblob': 0.2},
            'filing': {'finbert': 0.5, 'vader': 0.3, 'textblob': 0.2},
            'unknown': {'vader': 0.5, 'textblob': 0.5}
        }
        
        weights = model_weights.get(source, model_weights['unknown'])
        
        # Calculate weighted average
        total_score = 0.0
        total_confidence = 0.0
        total_weight = 0.0
        
        for model, sentiment in sentiments.items():
            weight = weights.get(model, 0.0)
            if weight > 0:
                total_score += sentiment['score'] * weight * sentiment['confidence']
                total_confidence += sentiment['confidence'] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = total_score / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0
        
        # Determine label
        if final_score >= 0.1:
            label = 'positive'
        elif final_score <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': final_score,
            'confidence': final_confidence,
            'label': label
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        keywords = []
        
        # Use YAKE for keyword extraction
        try:
            extracted = self.keyword_extractor.extract_keywords(text)
            keywords = [kw[0] for kw in extracted[:5]]  # Top 5 keywords
        except:
            pass
        
        # Add financial terms if present
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'beat', 'miss', 'guidance', 'forecast', 'upgrade', 'downgrade',
            'buy', 'sell', 'hold', 'outperform', 'underperform'
        ]
        
        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower and term not in keywords:
                keywords.append(term)
        
        return keywords[:10]  # Limit to 10 keywords
    
    def _extract_entities(self, text: str, ticker: str) -> List[str]:
        """Extract named entities relevant to analysis"""
        entities = [ticker]  # Always include the ticker
        
        # Extract dollar amounts
        dollar_pattern = r'\$[\d,]+\.?\d*[KMBTkmbt]?'
        dollars = re.findall(dollar_pattern, text)
        entities.extend(dollars[:3])  # Top 3 dollar amounts
        
        # Extract percentages
        percent_pattern = r'\d+\.?\d*%'
        percentages = re.findall(percent_pattern, text)
        entities.extend(percentages[:3])  # Top 3 percentages
        
        # Extract company names (simplified)
        # In production, use NER model
        company_pattern = r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|Corp|Company|Ltd|LLC|Group)\b'
        companies = re.findall(company_pattern, text)
        entities.extend(companies[:2])  # Top 2 company names
        
        # Extract executive names
        exec_pattern = r'\b(?:CEO|CFO|President|Chairman)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        executives = re.findall(exec_pattern, text)
        entities.extend(executives[:2])
        
        return list(set(entities))  # Remove duplicates
    
    def _calculate_relevance(self, text: str, ticker: str, entities: List[str]) -> float:
        """Calculate how relevant the text is to the ticker"""
        relevance = 0.0
        text_lower = text.lower()
        
        # Check ticker mentions
        ticker_lower = ticker.lower()
        ticker_count = text_lower.count(ticker_lower)
        relevance += min(ticker_count * 0.2, 0.6)  # Cap at 0.6
        
        # Check company name mentions (would need company name mapping)
        # For now, assume ticker is mentioned
        
        # Check if it's about the company vs just mentioning it
        company_patterns = [
            f"{ticker_lower} (?:reported|announced|beat|missed|raised|lowered)",
            f"(?:upgrade|downgrade|buy|sell|hold) {ticker_lower}",
            f"{ticker_lower}'s (?:earnings|revenue|profit|guidance)"
        ]
        
        for pattern in company_patterns:
            if re.search(pattern, text_lower):
                relevance += 0.2
        
        # Check entity relevance
        if len(entities) > 2:  # Multiple relevant entities
            relevance += 0.1
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    def _calculate_overall_sentiment(
        self,
        sentiments: List[SentimentResult],
        source_weights: Dict[str, float]
    ) -> Dict:
        """Calculate weighted overall sentiment"""
        if not sentiments:
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'direction': 'neutral'
            }
        
        # Weight by source, relevance, and recency
        weighted_sum = 0.0
        weight_total = 0.0
        confidence_sum = 0.0
        
        current_time = datetime.utcnow()
        
        for sentiment in sentiments:
            # Source weight
            source_weight = source_weights.get(sentiment.source, 0.1)
            
            # Relevance weight
            relevance_weight = sentiment.relevance_score
            
            # Recency weight (decay over time)
            hours_old = (current_time - sentiment.timestamp).total_seconds() / 3600
            recency_weight = np.exp(-hours_old / 24)  # Half-life of 24 hours
            
            # Combined weight
            weight = source_weight * relevance_weight * recency_weight * sentiment.confidence
            
            weighted_sum += sentiment.sentiment_score * weight
            confidence_sum += sentiment.confidence * weight
            weight_total += weight
        
        if weight_total > 0:
            overall_score = weighted_sum / weight_total
            overall_confidence = confidence_sum / weight_total
        else:
            overall_score = 0.0
            overall_confidence = 0.0
        
        # Determine label and direction
        if overall_score >= 0.2:
            label = 'positive'
            direction = 'bullish'
        elif overall_score <= -0.2:
            label = 'negative'
            direction = 'bearish'
        elif overall_score >= 0.05:
            label = 'slightly_positive'
            direction = 'neutral_bullish'
        elif overall_score <= -0.05:
            label = 'slightly_negative'
            direction = 'neutral_bearish'
        else:
            label = 'neutral'
            direction = 'neutral'
        
        return {
            'score': overall_score,
            'label': label,
            'confidence': overall_confidence,
            'direction': direction,
            'item_count': len(sentiments)
        }
    
    def _analyze_by_source(self, source_sentiments: Dict[str, List[SentimentResult]]) -> Dict:
        """Analyze sentiment breakdown by source"""
        breakdown = {}
        
        for source, sentiments in source_sentiments.items():
            if not sentiments:
                continue
            
            scores = [s.sentiment_score for s in sentiments]
            confidences = [s.confidence for s in sentiments]
            
            breakdown[source] = {
                'count': len(sentiments),
                'average_sentiment': np.mean(scores),
                'sentiment_std': np.std(scores),
                'average_confidence': np.mean(confidences),
                'positive_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
                'negative_ratio': sum(1 for s in scores if s < -0.1) / len(scores),
                'recent_trend': self._calculate_recent_trend(sentiments)
            }
        
        return breakdown
    
    def _analyze_temporal_sentiment(self, sentiments: List[SentimentResult]) -> Dict:
        """Analyze sentiment trends over time"""
        if len(sentiments) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort by timestamp
        sorted_sentiments = sorted(sentiments, key=lambda x: x.timestamp)
        
        # Group by time periods
        hourly = defaultdict(list)
        daily = defaultdict(list)
        
        for sentiment in sorted_sentiments:
            hour_key = sentiment.timestamp.strftime('%Y-%m-%d %H:00')
            day_key = sentiment.timestamp.strftime('%Y-%m-%d')
            
            hourly[hour_key].append(sentiment.sentiment_score)
            daily[day_key].append(sentiment.sentiment_score)
        
        # Calculate trends
        hourly_trend = []
        for hour, scores in sorted(hourly.items()):
            hourly_trend.append({
                'time': hour,
                'average_sentiment': np.mean(scores),
                'volume': len(scores)
            })
        
        daily_trend = []
        for day, scores in sorted(daily.items()):
            daily_trend.append({
                'date': day,
                'average_sentiment': np.mean(scores),
                'volume': len(scores),
                'volatility': np.std(scores) if len(scores) > 1 else 0
            })
        
        # Calculate momentum
        if len(daily_trend) >= 2:
            recent_sentiment = daily_trend[-1]['average_sentiment']
            previous_sentiment = daily_trend[-2]['average_sentiment']
            momentum = recent_sentiment - previous_sentiment
        else:
            momentum = 0.0
        
        return {
            'hourly_trend': hourly_trend[-24:],  # Last 24 hours
            'daily_trend': daily_trend[-7:],  # Last 7 days
            'momentum': momentum,
            'trend_direction': 'improving' if momentum > 0.1 else 'deteriorating' if momentum < -0.1 else 'stable'
        }
    
    def _analyze_keywords(self, sentiments: List[SentimentResult]) -> Dict:
        """Analyze keywords and their sentiment associations"""
        keyword_sentiments = defaultdict(list)
        
        for sentiment in sentiments:
            for keyword in sentiment.keywords:
                keyword_sentiments[keyword.lower()].append(sentiment.sentiment_score)
        
        # Calculate keyword impact
        keyword_analysis = {}
        for keyword, scores in keyword_sentiments.items():
            if len(scores) >= 2:  # Only include keywords that appear multiple times
                keyword_analysis[keyword] = {
                    'frequency': len(scores),
                    'average_sentiment': np.mean(scores),
                    'sentiment_impact': np.mean(scores) * len(scores)  # Frequency-weighted
                }
        
        # Sort by impact
        sorted_keywords = sorted(
            keyword_analysis.items(),
            key=lambda x: abs(x[1]['sentiment_impact']),
            reverse=True
        )[:20]  # Top 20 keywords
        
        return {
            'top_positive': [k for k, v in sorted_keywords if v['average_sentiment'] > 0.1][:10],
            'top_negative': [k for k, v in sorted_keywords if v['average_sentiment'] < -0.1][:10],
            'most_frequent': sorted(
                keyword_analysis.items(),
                key=lambda x: x[1]['frequency'],
                reverse=True
            )[:10]
        }
    
    def _analyze_entities(self, sentiments: List[SentimentResult]) -> Dict:
        """Analyze entities and their sentiment associations"""
        entity_sentiments = defaultdict(list)
        
        for sentiment in sentiments:
            for entity in sentiment.entities:
                entity_sentiments[entity].append(sentiment.sentiment_score)
        
        # Analyze each entity
        entity_analysis = {}
        for entity, scores in entity_sentiments.items():
            entity_analysis[entity] = {
                'mentions': len(scores),
                'average_sentiment': np.mean(scores),
                'sentiment_range': max(scores) - min(scores) if len(scores) > 1 else 0
            }
        
        return entity_analysis
    
    def _detect_sentiment_anomalies(self, sentiments: List[SentimentResult]) -> Dict:
        """Detect unusual sentiment patterns"""
        if len(sentiments) < 10:
            return {'anomalies_detected': False}
        
        scores = [s.sentiment_score for s in sentiments]
        timestamps = [s.timestamp for s in sentiments]
        
        # Statistical anomaly detection
        mean_sentiment = np.mean(scores)
        std_sentiment = np.std(scores)
        
        anomalies = []
        
        for i, (score, timestamp) in enumerate(zip(scores, timestamps)):
            # Z-score method
            z_score = abs(score - mean_sentiment) / std_sentiment if std_sentiment > 0 else 0
            
            if z_score > 2.5:  # 2.5 standard deviations
                anomalies.append({
                    'timestamp': timestamp.isoformat(),
                    'sentiment_score': score,
                    'z_score': z_score,
                    'type': 'statistical_outlier'
                })
        
        # Detect sudden sentiment shifts
        sorted_sentiments = sorted(sentiments, key=lambda x: x.timestamp)
        
        for i in range(1, len(sorted_sentiments)):
            current = sorted_sentiments[i]
            previous = sorted_sentiments[i-1]
            
            time_diff = (current.timestamp - previous.timestamp).total_seconds() / 3600
            
            if time_diff < 1:  # Within 1 hour
                sentiment_diff = abs(current.sentiment_score - previous.sentiment_score)
                
                if sentiment_diff > 0.5:  # Large shift
                    anomalies.append({
                        'timestamp': current.timestamp.isoformat(),
                        'sentiment_shift': sentiment_diff,
                        'type': 'sudden_shift'
                    })
        
        # Volume anomalies
        hourly_volumes = defaultdict(int)
        for sentiment in sentiments:
            hour_key = sentiment.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_volumes[hour_key] += 1
        
        if hourly_volumes:
            mean_volume = np.mean(list(hourly_volumes.values()))
            std_volume = np.std(list(hourly_volumes.values()))
            
            for hour, volume in hourly_volumes.items():
                if std_volume > 0:
                    z_score = (volume - mean_volume) / std_volume
                    if z_score > 2:
                        anomalies.append({
                            'timestamp': hour,
                            'volume': volume,
                            'z_score': z_score,
                            'type': 'volume_spike'
                        })
        
        return {
            'anomalies_detected': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies[:10]  # Top 10 anomalies
        }
    
    def _calculate_market_impact(self, sentiments: List[SentimentResult]) -> float:
        """
        Calculate potential market impact score (0-100)
        """
        if not sentiments:
            return 0.0
        
        impact_factors = []
        
        # 1. Sentiment extremity
        scores = [s.sentiment_score for s in sentiments]
        extremity = np.mean([abs(s) for s in scores])
        impact_factors.append(extremity * 30)  # Max 30 points
        
        # 2. Consensus (low std = high consensus)
        if len(scores) > 1:
            consensus = 1 - min(np.std(scores), 1.0)
            impact_factors.append(consensus * 20)  # Max 20 points
        
        # 3. Volume of mentions
        volume_score = min(len(sentiments) / 50, 1.0)  # Normalize to 50 mentions
        impact_factors.append(volume_score * 20)  # Max 20 points
        
        # 4. Source authority
        authority_scores = {
            'analyst': 1.0,
            'news': 0.8,
            'filing': 0.9,
            'insider': 1.0,
            'social': 0.4
        }
        
        avg_authority = np.mean([
            authority_scores.get(s.source, 0.5) for s in sentiments
        ])
        impact_factors.append(avg_authority * 15)  # Max 15 points
        
        # 5. Recency
        current_time = datetime.utcnow()
        recency_scores = []
        for sentiment in sentiments:
            hours_old = (current_time - sentiment.timestamp).total_seconds() / 3600
            recency = max(0, 1 - hours_old / 24)  # Decay over 24 hours
            recency_scores.append(recency)
        
        avg_recency = np.mean(recency_scores)
        impact_factors.append(avg_recency * 15)  # Max 15 points
        
        return sum(impact_factors)
    
    def _generate_sentiment_signals(self, sentiments: List[SentimentResult]) -> List[Dict]:
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        if not sentiments:
            return signals
        
        # Overall sentiment signal
        scores = [s.sentiment_score for s in sentiments]
        avg_sentiment = np.mean(scores)
        sentiment_std = np.std(scores) if len(scores) > 1 else 0
        
        if avg_sentiment > 0.5 and sentiment_std < 0.3:
            signals.append({
                'type': 'sentiment',
                'name': 'Strong Positive Sentiment Consensus',
                'strength': 'strong',
                'action': 'buy',
                'confidence': 0.8
            })
        elif avg_sentiment < -0.5 and sentiment_std < 0.3:
            signals.append({
                'type': 'sentiment',
                'name': 'Strong Negative Sentiment Consensus',
                'strength': 'strong',
                'action': 'sell',
                'confidence': 0.8
            })
        
        # Momentum signal
        recent_sentiments = [s for s in sentiments if 
                           (datetime.utcnow() - s.timestamp).total_seconds() < 86400]
        
        if len(recent_sentiments) >= 5:
            recent_avg = np.mean([s.sentiment_score for s in recent_sentiments])
            older_sentiments = [s for s in sentiments if s not in recent_sentiments]
            
            if older_sentiments:
                older_avg = np.mean([s.sentiment_score for s in older_sentiments])
                momentum = recent_avg - older_avg
                
                if momentum > 0.3:
                    signals.append({
                        'type': 'sentiment_momentum',
                        'name': 'Improving Sentiment Momentum',
                        'strength': 'medium',
                        'action': 'buy',
                        'confidence': 0.6
                    })
                elif momentum < -0.3:
                    signals.append({
                        'type': 'sentiment_momentum',
                        'name': 'Deteriorating Sentiment Momentum',
                        'strength': 'medium',
                        'action': 'sell',
                        'confidence': 0.6
                    })
        
        # Analyst sentiment signal
        analyst_sentiments = [s for s in sentiments if s.source == 'analyst']
        if len(analyst_sentiments) >= 3:
            analyst_avg = np.mean([s.sentiment_score for s in analyst_sentiments])
            
            if analyst_avg > 0.6:
                signals.append({
                    'type': 'analyst_sentiment',
                    'name': 'Positive Analyst Consensus',
                    'strength': 'strong',
                    'action': 'buy',
                    'confidence': 0.7
                })
            elif analyst_avg < -0.6:
                signals.append({
                    'type': 'analyst_sentiment',
                    'name': 'Negative Analyst Consensus',
                    'strength': 'strong',
                    'action': 'sell',
                    'confidence': 0.7
                })
        
        # Social sentiment divergence signal
        social_sentiments = [s for s in sentiments if s.source in ['twitter', 'reddit', 'stocktwits']]
        news_sentiments = [s for s in sentiments if s.source == 'news']
        
        if len(social_sentiments) >= 5 and len(news_sentiments) >= 3:
            social_avg = np.mean([s.sentiment_score for s in social_sentiments])
            news_avg = np.mean([s.sentiment_score for s in news_sentiments])
            
            divergence = social_avg - news_avg
            
            if divergence > 0.4:
                signals.append({
                    'type': 'sentiment_divergence',
                    'name': 'Social Sentiment More Positive Than News',
                    'strength': 'medium',
                    'action': 'monitor',
                    'confidence': 0.5
                })
            elif divergence < -0.4:
                signals.append({
                    'type': 'sentiment_divergence',
                    'name': 'Social Sentiment More Negative Than News',
                    'strength': 'medium',
                    'action': 'monitor',
                    'confidence': 0.5
                })
        
        return signals
    
    def _calculate_recent_trend(self, sentiments: List[SentimentResult]) -> str:
        """Calculate recent sentiment trend"""
        if len(sentiments) < 2:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_sentiments = sorted(sentiments, key=lambda x: x.timestamp)
        
        # Split into recent and older
        midpoint = len(sorted_sentiments) // 2
        older_half = sorted_sentiments[:midpoint]
        recent_half = sorted_sentiments[midpoint:]
        
        older_avg = np.mean([s.sentiment_score for s in older_half])
        recent_avg = np.mean([s.sentiment_score for s in recent_half])
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    async def analyze_earnings_call(self, transcript: str, ticker: str) -> Dict:
        """
        Specialized analysis for earnings call transcripts
        """
        # Split transcript into sections (management discussion vs Q&A)
        sections = self._split_earnings_transcript(transcript)
        
        analysis = {
            'ticker': ticker,
            'overall_tone': {},
            'management_sentiment': {},
            'analyst_questions_sentiment': {},
            'key_topics': {},
            'guidance_sentiment': {},
            'red_flags': []
        }
        
        # Analyze management discussion
        if sections.get('management'):
            mgmt_sentiment = await self._analyze_single_item(
                ticker,
                {'text': sections['management'], 'source': 'earnings_call'}
            )
            
            analysis['management_sentiment'] = {
                'score': mgmt_sentiment.sentiment_score,
                'confidence': mgmt_sentiment.confidence,
                'key_phrases': self._extract_key_phrases(sections['management'])
            }
        
        # Analyze Q&A section
        if sections.get('qa'):
            qa_sentiments = []
            qa_parts = sections['qa'].split('\n\n')  # Split by questions
            
            for qa in qa_parts[:10]:  # Analyze first 10 Q&As
                qa_sentiment = await self._analyze_single_item(
                    ticker,
                    {'text': qa, 'source': 'earnings_qa'}
                )
                if qa_sentiment:
                    qa_sentiments.append(qa_sentiment)
            
            if qa_sentiments:
                analysis['analyst_questions_sentiment'] = {
                    'average_score': np.mean([s.sentiment_score for s in qa_sentiments]),
                    'question_count': len(qa_sentiments),
                    'difficult_questions': sum(1 for s in qa_sentiments if s.sentiment_score < -0.3)
                }
        
        # Extract guidance sentiment
        guidance_text = self._extract_guidance(transcript)
        if guidance_text:
            guidance_sentiment = await self._analyze_single_item(
                ticker,
                {'text': guidance_text, 'source': 'guidance'}
            )
            
            analysis['guidance_sentiment'] = {
                'score': guidance_sentiment.sentiment_score,
                'raised': 'raise' in guidance_text.lower() or 'increase' in guidance_text.lower(),
                'lowered': 'lower' in guidance_text.lower() or 'decrease' in guidance_text.lower(),
                'maintained': 'maintain' in guidance_text.lower() or 'reiterate' in guidance_text.lower()
            }
        
        # Detect red flags
        analysis['red_flags'] = self._detect_earnings_red_flags(transcript)
        
        # Overall tone
        all_sentiments = []
        if sections.get('management'):
            all_sentiments.append(analysis['management_sentiment']['score'])
        if analysis.get('analyst_questions_sentiment'):
            all_sentiments.append(analysis['analyst_questions_sentiment']['average_score'])
        if analysis.get('guidance_sentiment'):
            all_sentiments.append(analysis['guidance_sentiment']['score'])
        
        if all_sentiments:
            analysis['overall_tone'] = {
                'score': np.mean(all_sentiments),
                'consistency': 1 - np.std(all_sentiments) if len(all_sentiments) > 1 else 1.0
            }
        
        return analysis
    
    def _split_earnings_transcript(self, transcript: str) -> Dict[str, str]:
        """Split earnings transcript into sections"""
        sections = {}
        
        # Common patterns for Q&A section start
        qa_patterns = [
            r'question.and.answer',
            r'q\s*&\s*a',
            r'question.+session',
            r'now.+questions'
        ]
        
        # Find Q&A section
        qa_start = len(transcript)
        for pattern in qa_patterns:
            match = re.search(pattern, transcript.lower())
            if match:
                qa_start = min(qa_start, match.start())
        
        if qa_start < len(transcript):
            sections['management'] = transcript[:qa_start]
            sections['qa'] = transcript[qa_start:]
        else:
            sections['management'] = transcript
        
        return sections
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that indicate sentiment"""
        key_phrases = []
        
        # Positive phrases
        positive_patterns = [
            r'record.{0,20}(?:revenue|earnings|quarter)',
            r'exceed.{0,20}expectations',
            r'strong.{0,20}(?:growth|performance|demand)',
            r'(?:raise|increase).{0,20}guidance',
            r'confident.{0,20}(?:in|about)',
            r'momentum.{0,20}continu'
        ]
        
        # Negative phrases
        negative_patterns = [
            r'challeng.{0,20}(?:environment|conditions|quarter)',
            r'below.{0,20}expectations',
            r'(?:lower|decrease).{0,20}guidance',
            r'disappoint.{0,20}(?:results|performance)',
            r'weak.{0,20}(?:demand|sales|growth)',
            r'uncertain.{0,20}(?:environment|outlook)'
        ]
        
        text_lower = text.lower()
        
        for pattern in positive_patterns:
            matches = re.findall(pattern, text_lower)
            key_phrases.extend([f"Positive: {m}" for m in matches])
        
        for pattern in negative_patterns:
            matches = re.findall(pattern, text_lower)
            key_phrases.extend([f"Negative: {m}" for m in matches])
        
        return key_phrases[:10]  # Top 10 phrases
    
    def _extract_guidance(self, transcript: str) -> str:
        """Extract guidance-related text from transcript"""
        guidance_patterns = [
            r'(?:full.year|annual|fy).{0,100}guidance',
            r'guidance.{0,100}(?:fiscal|full.year|annual)',
            r'(?:expect|anticipate|project).{0,50}(?:revenue|earnings|eps)',
            r'outlook.{0,100}(?:remain|improve|deteriorat)'
        ]
        
        guidance_text = []
        
        for pattern in guidance_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 100)
                end = min(len(transcript), match.end() + 100)
                guidance_text.append(transcript[start:end])
        
        return ' '.join(guidance_text)
    
    def _detect_earnings_red_flags(self, transcript: str) -> List[str]:
        """Detect potential red flags in earnings calls"""
        red_flags = []
        
        # Evasive language patterns
        evasive_patterns = {
            'deflection': r'(?:let me|I\'ll|we\'ll).{0,20}get back to you',
            'vague_timeline': r'(?:eventually|at some point|in due course)',
            'blame_external': r'(?:market conditions|macro environment|supply chain).{0,30}(?:challenge|difficult)',
            'accounting_changes': r'(?:change|adjust|restate).{0,30}(?:accounting|policy|method)'
        }
        
        text_lower = transcript.lower()
        
        for flag_type, pattern in evasive_patterns.items():
            if re.search(pattern, text_lower):
                red_flags.append(flag_type)
        
        # Check for excessive use of qualifying language
        qualifiers = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'potentially']
        qualifier_count = sum(text_lower.count(q) for q in qualifiers)
        
        if qualifier_count > len(transcript.split()) * 0.02:  # More than 2% qualifiers
            red_flags.append('excessive_qualifiers')
        
        # Check for sudden topic changes
        if re.search(r'(?:but|however).{0,20}(?:let\'s|moving on|different topic)', text_lower):
            red_flags.append('topic_avoidance')
        
        return red_flags