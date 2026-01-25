"""
Hybrid Sentiment Analysis Engine
Combines FinBERT-based analysis with fallback to lexicon approach.
Uses HuggingFace ProsusAI/finbert for accurate financial sentiment.
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
import os

logger = logging.getLogger(__name__)

# Try to import FinBERT analyzer
try:
    from backend.analytics.finbert_analyzer import (
        FinBERTAnalyzer,
        FinBERTInference,
        FinBERTResult,
        HAS_TRANSFORMERS
    )
    FINBERT_AVAILABLE = HAS_TRANSFORMERS
except ImportError:
    FINBERT_AVAILABLE = False
    logger.info("FinBERT not available, using lexicon-based sentiment only")

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    label: str  # 'positive', 'negative', 'neutral'
    breakdown: Dict[str, float]  # Individual component scores
    keywords: List[str]  # Key extracted keywords
    sources_analyzed: int  # Number of sources analyzed
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'confidence': self.confidence,
            'label': self.label,
            'breakdown': self.breakdown,
            'keywords': self.keywords,
            'sources_analyzed': self.sources_analyzed,
            'timestamp': self.timestamp.isoformat()
        }

class SentimentAnalysisEngine:
    """
    Hybrid sentiment analysis engine with FinBERT and lexicon fallback.
    Uses FinBERT (ProsusAI/finbert) as primary analyzer when available.
    """

    def __init__(self, use_finbert: bool = True):
        """
        Initialize sentiment analysis engine.

        Args:
            use_finbert: Whether to use FinBERT (default True, falls back if unavailable)
        """
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        self._finbert_analyzer = None
        self._finbert_inference = None

        # Simple word lists for basic sentiment analysis (fallback)
        self.positive_words = {
            'buy', 'bull', 'bullish', 'gain', 'gains', 'growth', 'profit', 'profits',
            'rise', 'rising', 'surge', 'soar', 'strong', 'strength', 'outperform',
            'upgrade', 'upside', 'positive', 'beat', 'beats', 'exceed', 'exceeds',
            'good', 'great', 'excellent', 'outstanding', 'impressive', 'solid',
            'robust', 'healthy', 'optimistic', 'confident', 'momentum', 'recovery'
        }
        
        self.negative_words = {
            'sell', 'bear', 'bearish', 'loss', 'losses', 'decline', 'drop', 'drops',
            'fall', 'falling', 'crash', 'weak', 'weakness', 'underperform',
            'downgrade', 'downside', 'negative', 'miss', 'misses', 'disappoint',
            'disappoints', 'bad', 'poor', 'terrible', 'awful', 'concerning',
            'worrying', 'risk', 'risks', 'threat', 'threats', 'volatile', 'uncertainty'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'significantly': 1.5,
            'substantially': 1.5, 'dramatically': 2.0, 'sharply': 1.5,
            'slightly': 0.5, 'somewhat': 0.7, 'moderately': 0.8
        }

        if self.use_finbert:
            logger.info("Sentiment analysis engine initialized with FinBERT")
        else:
            logger.info("Sentiment analysis engine initialized (lexicon-only mode)")

    async def _ensure_finbert_loaded(self) -> bool:
        """Lazy load FinBERT model."""
        if not self.use_finbert:
            return False

        if self._finbert_analyzer is None:
            try:
                self._finbert_analyzer = FinBERTAnalyzer()
                # Initialize in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    None,
                    self._finbert_analyzer.initialize
                )
                if success:
                    self._finbert_inference = FinBERTInference(
                        self._finbert_analyzer,
                        batch_size=16
                    )
                    return True
                else:
                    logger.warning("FinBERT init failed, using lexicon fallback")
                    self.use_finbert = False
                    return False
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                self.use_finbert = False
                return False
        return True
    
    async def analyze_sentiment(self, text: str, source: str = "unknown") -> SentimentResult:
        """
        Analyze sentiment of given text.
        Uses FinBERT if available, falls back to lexicon approach.
        """
        # Try FinBERT first
        if await self._ensure_finbert_loaded():
            try:
                result = await self._analyze_with_finbert(text, source)
                return result
            except Exception as e:
                logger.warning(f"FinBERT analysis failed, using fallback: {e}")

        # Fallback to lexicon-based approach
        return await self._analyze_with_lexicon(text, source)

    async def _analyze_with_finbert(self, text: str, source: str) -> SentimentResult:
        """Analyze using FinBERT model."""
        result = await self._finbert_inference.analyze_single_async(text)

        return SentimentResult(
            score=result.score,
            confidence=result.confidence,
            label=result.label,
            breakdown={
                'model': 'finbert',
                'probabilities': result.probabilities
            },
            keywords=self._extract_keywords(text.lower()),
            sources_analyzed=1,
            timestamp=datetime.utcnow()
        )

    async def _analyze_with_lexicon(self, text: str, source: str) -> SentimentResult:
        """Analyze using lexicon-based approach (fallback)."""
        try:
            # Basic text preprocessing
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            # Calculate basic sentiment scores
            pos_score = 0
            neg_score = 0
            word_count = len(words)
            
            for i, word in enumerate(words):
                # Check for intensifiers
                multiplier = 1.0
                if i > 0 and words[i-1] in self.intensifiers:
                    multiplier = self.intensifiers[words[i-1]]
                
                if word in self.positive_words:
                    pos_score += multiplier
                elif word in self.negative_words:
                    neg_score += multiplier
            
            # Calculate overall sentiment
            total_sentiment_words = pos_score + neg_score
            if total_sentiment_words == 0:
                sentiment_score = 0.0
                confidence = 0.3  # Low confidence for neutral
                label = 'neutral'
            else:
                sentiment_score = (pos_score - neg_score) / max(word_count, 1)
                confidence = min(total_sentiment_words / max(word_count, 1), 1.0)
                
                if sentiment_score > 0.1:
                    label = 'positive'
                elif sentiment_score < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
            
            # Extract basic keywords (most frequent meaningful words)
            keywords = self._extract_keywords(text_lower)
            
            # Create result
            result = SentimentResult(
                score=max(-1.0, min(1.0, sentiment_score)),  # Clamp to [-1, 1]
                confidence=confidence,
                label=label,
                breakdown={
                    'model': 'lexicon',
                    'positive_score': pos_score,
                    'negative_score': neg_score,
                    'word_count': word_count,
                    'sentiment_words': total_sentiment_words
                },
                keywords=keywords,
                sources_analyzed=1,
                timestamp=datetime.utcnow()
            )

            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label='neutral',
                breakdown={},
                keywords=[],
                sources_analyzed=0,
                timestamp=datetime.utcnow()
            )
    
    async def analyze_stock_sentiment(self, ticker: str, texts: List[str]) -> SentimentResult:
        """
        Analyze sentiment for multiple texts related to a stock.
        Uses batch processing with FinBERT for efficiency.
        """
        if not texts:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label='neutral',
                breakdown={},
                keywords=[],
                sources_analyzed=0,
                timestamp=datetime.utcnow()
            )

        # Try FinBERT batch processing for efficiency
        if await self._ensure_finbert_loaded():
            try:
                return await self._batch_analyze_finbert(ticker, texts)
            except Exception as e:
                logger.warning(f"FinBERT batch analysis failed: {e}")

        # Fallback to sequential lexicon analysis
        results = []
        for text in texts:
            result = await self._analyze_with_lexicon(text, "stock_analysis")
            results.append(result)
        
        # Aggregate results
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        all_keywords = []
        for r in results:
            all_keywords.extend(r.keywords)
        
        # Calculate weighted average
        if confidences and sum(confidences) > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            weighted_score = sum(scores) / len(scores) if scores else 0.0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine label
        if weighted_score > 0.1:
            label = 'positive'
        elif weighted_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Get most common keywords
        keyword_counts = defaultdict(int)
        for keyword in all_keywords:
            keyword_counts[keyword] += 1
        
        top_keywords = sorted(keyword_counts.keys(), 
                            key=lambda k: keyword_counts[k], 
                            reverse=True)[:10]
        
        return SentimentResult(
            score=max(-1.0, min(1.0, weighted_score)),
            confidence=avg_confidence,
            label=label,
            breakdown={
                'model': 'lexicon',
                'individual_scores': scores,
                'individual_confidences': confidences,
                'text_count': len(texts)
            },
            keywords=top_keywords,
            sources_analyzed=len(texts),
            timestamp=datetime.utcnow()
        )

    async def _batch_analyze_finbert(
        self, ticker: str, texts: List[str]
    ) -> SentimentResult:
        """Batch analyze texts with FinBERT for efficiency."""
        # Run batch analysis
        results = await self._finbert_inference.analyze_batch_async(texts)

        # Aggregate results
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]

        # Confidence-weighted aggregation
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight
        else:
            weighted_score = sum(scores) / len(scores) if scores else 0.0

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine overall label
        if weighted_score > 0.1:
            label = 'positive'
        elif weighted_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        # Extract keywords from all texts
        all_text = ' '.join(texts)
        keywords = self._extract_keywords(all_text.lower())

        return SentimentResult(
            score=max(-1.0, min(1.0, weighted_score)),
            confidence=avg_confidence,
            label=label,
            breakdown={
                'model': 'finbert',
                'individual_scores': scores,
                'individual_confidences': confidences,
                'text_count': len(texts)
            },
            keywords=keywords,
            sources_analyzed=len(texts),
            timestamp=datetime.utcnow()
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract basic keywords from text"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'that', 'this', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'there',
            'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just', 'now'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter meaningful words
        meaningful_words = []
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                word.isalpha() and
                not word.isdigit()):
                meaningful_words.append(word)
        
        # Count frequency and return top words
        word_counts = defaultdict(int)
        for word in meaningful_words:
            word_counts[word] += 1
        
        # Return top 5 most frequent keywords
        top_words = sorted(word_counts.keys(), 
                          key=lambda w: word_counts[w], 
                          reverse=True)[:5]
        
        return top_words
    
    async def get_news_sentiment(self, ticker: str, limit: int = 10) -> SentimentResult:
        """
        Get sentiment from news articles (simplified - returns neutral for now)
        In a full implementation, this would fetch and analyze real news data
        """
        # Placeholder implementation
        return SentimentResult(
            score=0.0,
            confidence=0.5,
            label='neutral',
            breakdown={'source': 'news_placeholder'},
            keywords=[ticker.lower(), 'news'],
            sources_analyzed=0,
            timestamp=datetime.utcnow()
        )
    
    async def get_social_sentiment(self, ticker: str, limit: int = 50) -> SentimentResult:
        """
        Get sentiment from social media (simplified - returns neutral for now)
        In a full implementation, this would fetch and analyze real social media data
        """
        # Placeholder implementation
        return SentimentResult(
            score=0.0,
            confidence=0.5,
            label='neutral',
            breakdown={'source': 'social_placeholder'},
            keywords=[ticker.lower(), 'social'],
            sources_analyzed=0,
            timestamp=datetime.utcnow()
        )
    
    async def analyze_comprehensive_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis combining multiple sources
        """
        try:
            # In a full implementation, this would gather data from multiple sources
            # For now, return a basic structure
            news_sentiment = await self.get_news_sentiment(ticker)
            social_sentiment = await self.get_social_sentiment(ticker)
            
            # Combine sentiments (simplified averaging)
            combined_score = (news_sentiment.score + social_sentiment.score) / 2
            combined_confidence = (news_sentiment.confidence + social_sentiment.confidence) / 2
            
            if combined_score > 0.1:
                combined_label = 'positive'
            elif combined_score < -0.1:
                combined_label = 'negative'
            else:
                combined_label = 'neutral'
            
            return {
                'ticker': ticker,
                'overall_sentiment': {
                    'score': combined_score,
                    'label': combined_label,
                    'confidence': combined_confidence
                },
                'news_sentiment': news_sentiment.to_dict(),
                'social_sentiment': social_sentiment.to_dict(),
                'timestamp': datetime.utcnow().isoformat(),
                'sources_analyzed': news_sentiment.sources_analyzed + social_sentiment.sources_analyzed
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive sentiment analysis: {e}")
            return {
                'ticker': ticker,
                'overall_sentiment': {
                    'score': 0.0,
                    'label': 'neutral',
                    'confidence': 0.0
                },
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'sources_analyzed': 0
            }