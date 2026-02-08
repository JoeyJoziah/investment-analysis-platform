"""
ML/Prediction Domain Contract (Domain 4).

Defines the interface for machine learning predictions, model management,
and backtesting capabilities.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ContractResult, DomainContract


class ModelType(Enum):
    """Types of ML models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1m"
    MONTH_3 = "3m"
    MONTH_6 = "6m"
    YEAR_1 = "1y"


class PredictionDirection(Enum):
    """Predicted price direction."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class ModelMetricsDTO:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mae: Optional[float] = None  # Mean Absolute Error
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    r2: Optional[float] = None   # R-squared
    sharpe_ratio: Optional[float] = None  # For trading models
    max_drawdown: Optional[float] = None


@dataclass(frozen=True)
class ModelDTO:
    """Data transfer object for ML model."""
    id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    metrics: ModelMetricsDTO
    features_used: List[str]
    target_column: str
    training_date: Optional[datetime] = None
    last_prediction_date: Optional[datetime] = None
    prediction_count: int = 0
    description: Optional[str] = None


@dataclass(frozen=True)
class PredictionDTO:
    """Data transfer object for a prediction."""
    id: int
    stock_id: int
    symbol: str
    model_id: str
    model_name: str
    prediction_date: datetime
    target_date: datetime
    horizon: PredictionHorizon
    predicted_price: Optional[Decimal] = None
    predicted_price_low: Optional[Decimal] = None
    predicted_price_high: Optional[Decimal] = None
    predicted_return: Optional[float] = None
    predicted_direction: PredictionDirection = PredictionDirection.NEUTRAL
    confidence: float = 0.0
    features_snapshot: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class PredictionValidationDTO:
    """Validation result for a past prediction."""
    prediction_id: int
    predicted_price: Decimal
    actual_price: Decimal
    prediction_error: float
    direction_correct: bool
    within_confidence_interval: bool
    validated_at: datetime


@dataclass(frozen=True)
class BacktestResultDTO:
    """Results of a backtest."""
    model_id: str
    start_date: date
    end_date: date
    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_return: float
    benchmark_return: float
    alpha: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trade_count: int
    execution_time_seconds: float


@dataclass(frozen=True)
class FeatureImportanceDTO:
    """Feature importance for a model."""
    model_id: str
    feature_name: str
    importance_score: float
    rank: int


@dataclass(frozen=True)
class TrainingJobDTO:
    """ML training job information."""
    id: str
    model_name: str
    model_type: ModelType
    status: str
    progress: float  # 0-100
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[ModelMetricsDTO] = None
    error_message: Optional[str] = None


class MLContract(DomainContract):
    """
    Contract for Domain 4: ML/Prediction.

    Provides machine learning predictions, model management, and backtesting
    capabilities for stock price forecasting.
    """

    @property
    def domain_name(self) -> str:
        return "ML"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_prediction",
            "generate_prediction",
            "batch_predict",
            "get_model",
            "list_models",
            "get_model_metrics",
            "get_feature_importance",
            "train_model",
            "get_training_status",
            "backtest_model",
            "validate_predictions",
            "get_prediction_history",
        ]

    # Prediction Operations

    @abstractmethod
    async def get_prediction(
        self,
        stock_id: int,
        horizon: PredictionHorizon = PredictionHorizon.WEEK_1,
        model_id: Optional[str] = None
    ) -> ContractResult[PredictionDTO]:
        """
        Get the latest prediction for a stock.

        Args:
            stock_id: Stock ID
            horizon: Prediction time horizon
            model_id: Specific model to use (None = best model)

        Returns:
            ContractResult containing prediction
        """
        pass

    @abstractmethod
    async def generate_prediction(
        self,
        stock_id: int,
        horizon: PredictionHorizon = PredictionHorizon.WEEK_1,
        model_id: Optional[str] = None
    ) -> ContractResult[PredictionDTO]:
        """
        Generate a new prediction for a stock.

        Args:
            stock_id: Stock ID
            horizon: Prediction time horizon
            model_id: Specific model to use (None = best model)

        Returns:
            ContractResult containing new prediction
        """
        pass

    @abstractmethod
    async def batch_predict(
        self,
        stock_ids: List[int],
        horizon: PredictionHorizon = PredictionHorizon.WEEK_1,
        model_id: Optional[str] = None
    ) -> ContractResult[List[PredictionDTO]]:
        """
        Generate predictions for multiple stocks.

        Args:
            stock_ids: List of stock IDs
            horizon: Prediction time horizon
            model_id: Specific model to use

        Returns:
            ContractResult containing list of predictions
        """
        pass

    @abstractmethod
    async def get_prediction_history(
        self,
        stock_id: int,
        start_date: datetime,
        end_date: datetime,
        horizon: Optional[PredictionHorizon] = None
    ) -> ContractResult[List[PredictionDTO]]:
        """
        Get historical predictions for a stock.

        Args:
            stock_id: Stock ID
            start_date: Start date
            end_date: End date
            horizon: Filter by horizon

        Returns:
            ContractResult containing list of predictions
        """
        pass

    # Model Management

    @abstractmethod
    async def get_model(self, model_id: str) -> ContractResult[ModelDTO]:
        """
        Get model information.

        Args:
            model_id: Model ID

        Returns:
            ContractResult containing model info
        """
        pass

    @abstractmethod
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> ContractResult[List[ModelDTO]]:
        """
        List available models.

        Args:
            model_type: Filter by type
            status: Filter by status

        Returns:
            ContractResult containing list of models
        """
        pass

    @abstractmethod
    async def get_model_metrics(
        self,
        model_id: str
    ) -> ContractResult[ModelMetricsDTO]:
        """
        Get model performance metrics.

        Args:
            model_id: Model ID

        Returns:
            ContractResult containing metrics
        """
        pass

    @abstractmethod
    async def get_feature_importance(
        self,
        model_id: str,
        top_n: int = 20
    ) -> ContractResult[List[FeatureImportanceDTO]]:
        """
        Get feature importance for a model.

        Args:
            model_id: Model ID
            top_n: Number of top features

        Returns:
            ContractResult containing feature importances
        """
        pass

    # Training Operations

    @abstractmethod
    async def train_model(
        self,
        name: str,
        model_type: ModelType,
        features: List[str],
        target: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ContractResult[TrainingJobDTO]:
        """
        Start model training.

        Args:
            name: Model name
            model_type: Type of model
            features: Feature columns
            target: Target column
            config: Training configuration

        Returns:
            ContractResult containing training job info
        """
        pass

    @abstractmethod
    async def get_training_status(
        self,
        job_id: str
    ) -> ContractResult[TrainingJobDTO]:
        """
        Get training job status.

        Args:
            job_id: Training job ID

        Returns:
            ContractResult containing job status
        """
        pass

    # Backtesting

    @abstractmethod
    async def backtest_model(
        self,
        model_id: str,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None,
        initial_capital: float = 100000.0
    ) -> ContractResult[BacktestResultDTO]:
        """
        Backtest a model on historical data.

        Args:
            model_id: Model ID
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: Symbols to test (None = all)
            initial_capital: Starting capital

        Returns:
            ContractResult containing backtest results
        """
        pass

    # Validation

    @abstractmethod
    async def validate_predictions(
        self,
        model_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> ContractResult[List[PredictionValidationDTO]]:
        """
        Validate past predictions against actual outcomes.

        Args:
            model_id: Model ID
            start_date: Start date
            end_date: End date

        Returns:
            ContractResult containing validation results
        """
        pass

    # Health Check

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        """Check ML service health."""
        return ContractResult.ok({
            "domain": self.domain_name,
            "version": self.version,
            "status": "healthy",
            "capabilities": self.capabilities,
        })
