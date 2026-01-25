"""
FinBERT Sentiment Analyzer
Uses HuggingFace's ProsusAI/finbert model for financial sentiment analysis.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Check if transformers is available
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not available. FinBERT will use fallback mode.")


@dataclass
class FinBERTResult:
    """Result from FinBERT sentiment analysis."""
    score: float        # -1 to 1
    confidence: float   # 0 to 1
    label: str         # positive/negative/neutral
    probabilities: Dict[str, float]  # Raw probabilities per class


class FinancialTextPreprocessor:
    """Preprocessor optimized for financial news and reports."""

    def preprocess(self, text: str) -> str:
        """Preprocess financial text for FinBERT."""
        if not text:
            return ""

        import re

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags but keep word
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        # Truncate to reasonable length (FinBERT max is 512 tokens)
        words = text.split()
        if len(words) > 400:
            text = ' '.join(words[:400])

        return text.strip()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]

    def combine_headline_summary(self, headline: str, summary: str) -> str:
        """Combine headline and summary for analysis."""
        parts = []
        if headline:
            parts.append(headline.strip())
        if summary:
            parts.append(summary.strip())
        return ' '.join(parts)


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    Uses ProsusAI/finbert from HuggingFace (free, ~400MB).
    """

    _instance: Optional['FinBERTAnalyzer'] = None
    MODEL_NAME = "ProsusAI/finbert"

    # FinBERT label mapping: 0=positive, 1=negative, 2=neutral
    LABEL_MAP = {0: 'positive', 1: 'negative', 2: 'neutral'}

    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._model = None
        self._tokenizer = None
        self._device = None
        self._preprocessor = FinancialTextPreprocessor()

    def initialize(
        self,
        model_cache_dir: Optional[str] = None,
        force_cpu: bool = False
    ) -> bool:
        """
        Initialize the FinBERT model with lazy loading.

        Args:
            model_cache_dir: Directory to cache downloaded models
            force_cpu: Force CPU inference even if GPU available

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        if not HAS_TRANSFORMERS:
            logger.error("transformers library not available")
            return False

        try:
            # Determine device
            if force_cpu or not torch.cuda.is_available():
                self._device = torch.device('cpu')
                logger.info("FinBERT will run on CPU")
            else:
                self._device = torch.device('cuda')
                logger.info("FinBERT will run on GPU")

            # Set cache directory
            cache_dir = model_cache_dir or os.getenv('MODEL_CACHE_DIR', 'ml_models/finbert')

            logger.info(f"Loading FinBERT model from {self.MODEL_NAME}...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                cache_dir=cache_dir
            )

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                cache_dir=cache_dir
            ).to(self._device)

            self._model.eval()  # Set to evaluation mode
            self._initialized = True

            logger.info("FinBERT model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def analyze_single(self, text: str) -> FinBERTResult:
        """Analyze a single text."""
        results = self.analyze_batch([text])
        return results[0] if results else self._neutral_result()

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[FinBERTResult]:
        """
        Analyze a batch of texts efficiently.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts per batch

        Returns:
            List of FinBERTResult objects
        """
        if not texts:
            return []

        if not self._initialized:
            if not self.initialize():
                # Return neutral results if model fails to load
                return [self._neutral_result() for _ in texts]

        # Preprocess all texts
        processed_texts = self._preprocessor.preprocess_batch(texts)

        results = []

        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)

        return results

    def _process_batch(self, texts: List[str]) -> List[FinBERTResult]:
        """Process a single batch through the model."""

        # Handle empty texts
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)

        if not valid_texts:
            return [self._neutral_result() for _ in texts]

        try:
            # Tokenize
            inputs = self._tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self._device)

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probabilities = probabilities.cpu().numpy()

            # Convert to results
            valid_results = []
            for probs in probabilities:
                result = self._probabilities_to_result(probs)
                valid_results.append(result)

            # Reconstruct full results list
            full_results = [self._neutral_result() for _ in texts]
            for idx, result in zip(valid_indices, valid_results):
                full_results[idx] = result

            return full_results

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [self._neutral_result() for _ in texts]

    def _probabilities_to_result(self, probs: np.ndarray) -> FinBERTResult:
        """Convert raw probabilities to FinBERTResult."""

        # FinBERT probs order: [positive, negative, neutral]
        positive_prob = float(probs[0])
        negative_prob = float(probs[1])
        neutral_prob = float(probs[2])

        # Calculate sentiment score (-1 to 1)
        # Positive contributes positively, negative contributes negatively
        score = positive_prob - negative_prob

        # Confidence is the max probability
        confidence = float(max(probs))

        # Label is the argmax
        label_idx = int(np.argmax(probs))
        label = self.LABEL_MAP[label_idx]

        return FinBERTResult(
            score=score,
            confidence=confidence,
            label=label,
            probabilities={
                'positive': positive_prob,
                'negative': negative_prob,
                'neutral': neutral_prob
            }
        )

    def _neutral_result(self) -> FinBERTResult:
        """Return neutral result for empty/invalid text."""
        return FinBERTResult(
            score=0.0,
            confidence=0.33,
            label='neutral',
            probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        )


class FinBERTInference:
    """
    High-level inference interface for FinBERT.
    Provides async methods and caching.
    """

    def __init__(
        self,
        analyzer: Optional[FinBERTAnalyzer] = None,
        batch_size: int = 16,
        enable_cache: bool = True
    ):
        self.analyzer = analyzer or FinBERTAnalyzer()
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self._cache: Dict[str, FinBERTResult] = {}

    async def analyze_single_async(self, text: str) -> FinBERTResult:
        """Async wrapper for single text analysis."""
        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.analyzer.analyze_single,
            text
        )

        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    async def analyze_batch_async(self, texts: List[str]) -> List[FinBERTResult]:
        """Async wrapper for batch analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyzer.analyze_batch,
            texts,
            self.batch_size
        )

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def clear_cache(self):
        """Clear the sentiment cache."""
        self._cache.clear()


def download_finbert_model(cache_dir: str = 'ml_models/finbert') -> bool:
    """
    Download and cache FinBERT model for offline use.

    Args:
        cache_dir: Directory to save the model

    Returns:
        True if download successful
    """
    if not HAS_TRANSFORMERS:
        logger.error("transformers library required to download FinBERT")
        return False

    try:
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Downloading FinBERT model to {cache_dir}...")

        tokenizer = AutoTokenizer.from_pretrained(
            FinBERTAnalyzer.MODEL_NAME,
            cache_dir=cache_dir
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            FinBERTAnalyzer.MODEL_NAME,
            cache_dir=cache_dir
        )

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info("FinBERT download complete!")

        return True

    except Exception as e:
        logger.error(f"Failed to download FinBERT: {e}")
        return False


# Module-level convenience functions
_default_analyzer: Optional[FinBERTAnalyzer] = None


def get_finbert_analyzer() -> FinBERTAnalyzer:
    """Get or create the default FinBERT analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = FinBERTAnalyzer()
    return _default_analyzer


def analyze_sentiment(text: str) -> FinBERTResult:
    """Quick single-text sentiment analysis."""
    analyzer = get_finbert_analyzer()
    if not analyzer.is_initialized:
        analyzer.initialize()
    return analyzer.analyze_single(text)


def analyze_sentiment_batch(texts: List[str]) -> List[FinBERTResult]:
    """Quick batch sentiment analysis."""
    analyzer = get_finbert_analyzer()
    if not analyzer.is_initialized:
        analyzer.initialize()
    return analyzer.analyze_batch(texts)


if __name__ == '__main__':
    # Test the FinBERT analyzer
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--download':
        # Download model
        download_finbert_model()
    else:
        # Run test
        test_texts = [
            "Apple reports record quarterly earnings, beats analyst expectations",
            "Company faces major lawsuit, stock plunges 15%",
            "Market remains uncertain amid mixed economic data"
        ]

        print("Testing FinBERT Analyzer...")
        analyzer = FinBERTAnalyzer()

        if analyzer.initialize():
            results = analyzer.analyze_batch(test_texts)

            for text, result in zip(test_texts, results):
                print(f"\nText: {text[:60]}...")
                print(f"  Label: {result.label}")
                print(f"  Score: {result.score:.3f}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Probs: {result.probabilities}")
        else:
            print("Failed to initialize FinBERT")
