"""
Inference-time feature engineering for ML price predictions (#208 item 1).

This module turns a *windowed* OHLCV price/volume history (as produced by
:class:`backend.ml.data_providers.PriceHistoryDataProvider`) into the exact
feature matrix shape each trained model expects:

* **LSTM** — a 3D tensor of shape ``(1, sequence_length, n_features)``.
* **XGBoost** — a 2D matrix of shape ``(n_rows, n_features)``.

The trained model bundles persist a ``config`` dict whose ``feature_columns``
list (and, for LSTM, ``sequence_length``) defines the input contract.  This
module computes the technical indicators it can derive from raw OHLCV and then
**aligns the result to ``config['feature_columns']``** — same columns, same
order, same width — so the model's persisted scaler and weights line up.

Fail-loud (Finding #200): if there is genuinely no price history, the caller
gets nothing back and must raise a 503 upstream.  This module never fabricates
prices or pads a missing window with synthetic rows.  Columns that cannot be
derived from OHLCV alone (e.g. fundamentals such as ``pe_ratio`` that are not
present in the price feed) are filled with ``0.0`` — exactly how the training
pipeline treats missing/early-window values via ``np.nan_to_num`` — rather than
invented.  No randomness is ever introduced.

The module is import-light on purpose (only numpy + pandas) so it can be
exercised source-level under ``--noconftest`` without dragging the SQLAlchemy
package graph into the test process.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Raw OHLCV columns we always start from.
_OHLCV = ("open", "high", "low", "close", "volume")


class InsufficientHistoryError(RuntimeError):
    """Raised when there is no usable price history to build features from.

    Fail-loud sentinel: the prediction path must refuse to serve rather than
    synthesise a window when the underlying history is empty.
    """


def _ohlcv_frame(records: Sequence[Any]) -> pd.DataFrame:
    """Build an ascending-by-date OHLCV ``DataFrame`` from price-history rows.

    Accepts either ``PriceHistory``-like objects exposing
    ``date/open/high/low/close/volume`` attributes, or mapping rows with those
    keys.  Raises :class:`InsufficientHistoryError` when ``records`` is empty.
    """
    if records is None or len(records) == 0:
        raise InsufficientHistoryError(
            "No price history available to engineer prediction features"
        )

    rows = []
    index = []
    for rec in records:
        if isinstance(rec, Mapping):
            get = rec.get
        else:
            get = lambda k, _r=rec: getattr(_r, k)  # noqa: E731 - tiny local
        index.append(pd.Timestamp(get("date")))
        rows.append(
            {
                "open": float(get("open")),
                "high": float(get("high")),
                "low": float(get("low")),
                "close": float(get("close")),
                "volume": float(get("volume")),
            }
        )

    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=list(_OHLCV))
    # Repository order is not guaranteed; sort ascending so rolling windows and
    # lag features are chronologically correct.
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Augment the OHLCV frame with the technical indicators we can derive.

    Only columns computable from raw OHLCV are produced.  Any model feature
    not produced here is later filled with ``0.0`` during alignment.  All
    indicators are deterministic functions of price/volume — no randomness.
    """
    out = frame.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # Moving averages
    for w in (5, 10, 20, 50, 200):
        out[f"sma_{w}"] = close.rolling(w, min_periods=1).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_7"] = _rsi(close, 7)

    # Bollinger bands (20, 2 std)
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
    out["bb_middle"] = bb_mid
    out["bb_upper"] = bb_mid + 2.0 * bb_std
    out["bb_lower"] = bb_mid - 2.0 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0.0, np.nan)
    out["bb_position"] = (close - out["bb_lower"]) / (
        (out["bb_upper"] - out["bb_lower"]).replace(0.0, np.nan)
    )

    # ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14, min_periods=1).mean()

    # Stochastic oscillator
    low_14 = low.rolling(14, min_periods=1).min()
    high_14 = high.rolling(14, min_periods=1).max()
    out["stoch_k"] = 100.0 * (close - low_14) / (high_14 - low_14).replace(0.0, np.nan)
    out["stoch_d"] = out["stoch_k"].rolling(3, min_periods=1).mean()

    # Williams %R
    out["williams_r"] = -100.0 * (high_14 - close) / (high_14 - low_14).replace(
        0.0, np.nan
    )

    # Rate of change / momentum
    out["roc_10"] = close.pct_change(10) * 100.0
    out["momentum_10"] = close - close.shift(10)

    # Volume-based
    out["obv"] = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
    out["volume_sma_20"] = volume.rolling(20, min_periods=1).mean()
    out["volume_ratio"] = volume / out["volume_sma_20"].replace(0.0, np.nan)
    typical = (high + low + close) / 3.0
    out["vwap"] = (typical * volume).cumsum() / volume.cumsum().replace(0.0, np.nan)

    # Support / resistance (rolling extremes)
    out["resistance_1"] = high.rolling(20, min_periods=1).max()
    out["support_1"] = low.rolling(20, min_periods=1).min()

    # Returns / volatility / candle geometry
    out["returns"] = close.pct_change()
    out["log_returns"] = np.log(close / close.shift(1))
    out["volatility_20"] = out["returns"].rolling(20, min_periods=1).std(ddof=0)
    out["intraday_range"] = (high - low) / close.replace(0.0, np.nan)
    out["close_to_open"] = (close - out["open"]) / out["open"].replace(0.0, np.nan)
    out["gap"] = (out["open"] - prev_close) / prev_close.replace(0.0, np.nan)

    # Trend flags
    out["trend_sma"] = (close > out["sma_50"]).astype(float)
    out["trend_macd"] = (out["macd"] > out["macd_signal"]).astype(float)
    out["above_sma_200"] = (close > out["sma_200"]).astype(float)

    # Lagged returns
    for lag in (1, 5, 10, 20):
        out[f"return_lag_{lag}"] = out["returns"].shift(lag)

    return out


def _align_to_contract(
    enriched: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Project the enriched frame onto the model's exact feature contract.

    Produces a frame whose columns are *exactly* ``feature_columns`` in order.
    Columns we could not derive from OHLCV are filled with ``0.0`` (the same
    treatment the training pipeline applies to missing/early-window values via
    ``np.nan_to_num``).  Never fabricates non-zero values.
    """
    aligned = pd.DataFrame(index=enriched.index)
    for col in feature_columns:
        if col in enriched.columns:
            aligned[col] = enriched[col]
        else:
            # Non-derivable (e.g. fundamentals not in the price feed). Zero-fill
            # rather than invent — matches training's nan_to_num behaviour.
            aligned[col] = 0.0
    # Replace inf/nan introduced by early-window indicators with 0.0.
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return aligned[list(feature_columns)]


def _resolve_feature_columns(config: Optional[Mapping[str, Any]]) -> List[str]:
    cols = (config or {}).get("feature_columns")
    if not cols:
        raise ValueError(
            "Model config is missing 'feature_columns'; cannot build a feature "
            "matrix that matches the trained model's input contract."
        )
    return list(cols)


def build_lstm_features(
    records: Sequence[Any],
    config: Optional[Mapping[str, Any]],
    *,
    sequence_length: Optional[int] = None,
) -> np.ndarray:
    """Build the LSTM input tensor ``(1, sequence_length, n_features)``.

    Args:
        records: Windowed price history (``PriceHistory``-like rows or mappings),
            in any date order.  Must be non-empty.
        config: The trained LSTM bundle's ``config`` dict.  Supplies
            ``feature_columns`` (defines ``n_features``) and ``sequence_length``.
        sequence_length: Override for the window length; falls back to
            ``config['sequence_length']`` then 60.

    Returns:
        ``np.ndarray`` of shape ``(1, sequence_length, n_features)`` (float32).

    Raises:
        InsufficientHistoryError: ``records`` is empty.
        ValueError: ``config`` lacks ``feature_columns``.
    """
    feature_columns = _resolve_feature_columns(config)
    seq_len = int(
        sequence_length
        if sequence_length is not None
        else (config or {}).get("sequence_length", 60)
    )
    if seq_len < 1:
        raise ValueError(f"sequence_length must be >= 1, got {seq_len}")

    frame = _ohlcv_frame(records)
    enriched = _compute_indicators(frame)
    aligned = _align_to_contract(enriched, feature_columns)

    matrix = aligned.to_numpy(dtype=np.float32)  # (n_rows, n_features)
    n_features = len(feature_columns)

    # Take the most recent ``seq_len`` rows. If history is shorter than the
    # window, left-pad with the earliest available row repeated — this keeps the
    # tensor shape exact without inventing *new* price points (the pad rows are
    # real, observed feature vectors, not synthetic prices).
    if matrix.shape[0] >= seq_len:
        window = matrix[-seq_len:]
    else:
        pad_count = seq_len - matrix.shape[0]
        pad = np.repeat(matrix[:1], pad_count, axis=0)
        window = np.vstack([pad, matrix])

    tensor = window.reshape(1, seq_len, n_features)
    return tensor


def build_xgboost_features(
    records: Sequence[Any],
    config: Optional[Mapping[str, Any]],
    *,
    n_rows: int = 1,
) -> np.ndarray:
    """Build the XGBoost input matrix ``(n_rows, n_features)``.

    XGBoost is row-wise (no temporal window): the most recent ``n_rows`` feature
    rows are returned.  ``n_features`` is dictated by ``config['feature_columns']``.

    Args:
        records: Windowed price history rows (non-empty).
        config: The trained XGBoost bundle's ``config`` dict.
        n_rows: Number of (most-recent) feature rows to return. Defaults to 1.

    Returns:
        ``np.ndarray`` of shape ``(n_rows, n_features)`` (float32).

    Raises:
        InsufficientHistoryError: ``records`` is empty.
        ValueError: ``config`` lacks ``feature_columns`` or ``n_rows`` < 1.
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")

    feature_columns = _resolve_feature_columns(config)
    frame = _ohlcv_frame(records)
    enriched = _compute_indicators(frame)
    aligned = _align_to_contract(enriched, feature_columns)

    matrix = aligned.to_numpy(dtype=np.float32)
    n_features = len(feature_columns)

    if matrix.shape[0] >= n_rows:
        rows = matrix[-n_rows:]
    else:
        pad_count = n_rows - matrix.shape[0]
        pad = np.repeat(matrix[-1:], pad_count, axis=0)
        rows = np.vstack([matrix, pad])

    return rows.reshape(n_rows, n_features)
