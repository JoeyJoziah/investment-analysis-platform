#!/usr/bin/env python3
"""
ML Training Data Generator
Generates real training data from yfinance for ML model training.
Uses free APIs with no rate limits.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# S&P 100 stocks - diversified across sectors (free to fetch from yfinance)
SP100_STOCKS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
    'INTC', 'AMD', 'ORCL', 'IBM', 'QCOM',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'SCHW', 'USB',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    # Consumer Discretionary
    'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'TGT', 'BKNG', 'CMG',
    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'MDLZ', 'KHC',
    # Industrial
    'CAT', 'BA', 'HON', 'UPS', 'RTX', 'GE', 'MMM', 'DE', 'LMT', 'UNP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG',
    # Materials
    'LIN', 'APD', 'ECL', 'SHW', 'FCX',
    # Communication
    'GOOG', 'DIS', 'CMCSA', 'NFLX', 'VZ', 'T', 'TMUS', 'CHTR',
]


class MLTrainingDataGenerator:
    """Generate ML training data from yfinance (free, no rate limits)."""

    def __init__(
        self,
        output_dir: str = 'data/ml_training',
        years_history: int = 2,
        stocks: Optional[List[str]] = None
    ):
        self.output_dir = Path(output_dir)
        self.years_history = years_history
        self.stocks = stocks or SP100_STOCKS

        # Create directories
        (self.output_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(parents=True, exist_ok=True)

    def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single stock using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.years_history * 365)

            df = stock.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data for {ticker}")
                return None

            # Reset index and standardize columns
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            df['ticker'] = ticker

            # Rename columns to match expected format
            df = df.rename(columns={'date': 'date'})

            # Get company info for fundamentals
            try:
                info = stock.info
                df['market_cap'] = info.get('marketCap', np.nan)
                df['pe_ratio'] = info.get('trailingPE', np.nan)
                df['beta'] = info.get('beta', np.nan)
                df['dividend_yield'] = info.get('dividendYield', np.nan)
                df['sector'] = info.get('sector', 'Unknown')
            except Exception:
                df['market_cap'] = np.nan
                df['pe_ratio'] = np.nan
                df['beta'] = np.nan
                df['dividend_yield'] = np.nan
                df['sector'] = 'Unknown'

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        if df.empty or len(df) < 50:
            return df

        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']

            # Moving Averages
            df['sma_5'] = close.rolling(window=5).mean()
            df['sma_10'] = close.rolling(window=10).mean()
            df['sma_20'] = close.rolling(window=20).mean()
            df['sma_50'] = close.rolling(window=50).mean()
            df['sma_200'] = close.rolling(window=200).mean()

            df['ema_12'] = close.ewm(span=12, adjust=False).mean()
            df['ema_26'] = close.ewm(span=26, adjust=False).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # RSI 7
            gain_7 = delta.where(delta > 0, 0).rolling(window=7).mean()
            loss_7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
            rs_7 = gain_7 / loss_7
            df['rsi_7'] = 100 - (100 / (1 + rs_7))

            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            bb_std = close.rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (close - df['bb_lower']) / df['bb_width']

            # ATR (Average True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14).mean()

            # Stochastic Oscillator
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Williams %R
            df['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14)

            # CCI (Commodity Channel Index)
            typical_price = (high + low + close) / 3
            tp_sma = typical_price.rolling(window=14).mean()
            tp_mad = typical_price.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
            df['cci'] = (typical_price - tp_sma) / (0.015 * tp_mad)

            # ADX (Average Directional Index) - simplified
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0).where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0).where(minus_dm > 0, 0)

            tr_smooth = tr.rolling(window=14).sum()
            plus_di = 100 * plus_dm.rolling(window=14).sum() / tr_smooth
            minus_di = 100 * minus_dm.rolling(window=14).sum() / tr_smooth
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()

            # OBV (On Balance Volume)
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            df['obv'] = obv

            # MFI (Money Flow Index)
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume

            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0)
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0)

            positive_flow_14 = positive_flow.rolling(window=14).sum()
            negative_flow_14 = negative_flow.rolling(window=14).sum()

            mfi_ratio = positive_flow_14 / negative_flow_14
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))

            # Rate of Change
            df['roc_10'] = close.pct_change(periods=10) * 100
            df['momentum_10'] = close.diff(periods=10)

            # VWAP
            df['vwap'] = (close * volume).cumsum() / volume.cumsum()

            # Support and Resistance
            df['resistance_1'] = high.rolling(window=20).max()
            df['support_1'] = low.rolling(window=20).min()

            # Price features
            df['returns'] = close.pct_change()
            df['log_returns'] = np.log(close / close.shift(1))
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['intraday_range'] = (high - low) / close
            df['close_to_open'] = (close - df['open']) / df['open']
            df['gap'] = (df['open'] - close.shift(1)) / close.shift(1)

            # Volume features
            df['volume_sma_20'] = volume.rolling(window=20).mean()
            df['volume_ratio'] = volume / df['volume_sma_20']

            # Trend indicators
            df['trend_sma'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['trend_macd'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            df['above_sma_200'] = np.where(close > df['sma_200'], 1, 0)

            # Lagged returns (for prediction)
            for lag in [1, 5, 10, 20]:
                df[f'return_lag_{lag}'] = df['returns'].shift(lag)

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

        return df

    def generate_labels(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Generate future return labels for supervised learning."""
        if df.empty:
            return df

        close = df['close']

        for horizon in horizons:
            # Future returns (what we want to predict)
            df[f'future_return_{horizon}d'] = close.shift(-horizon) / close - 1

            # Direction labels for classification
            df[f'direction_{horizon}d'] = (df[f'future_return_{horizon}d'] > 0).astype(int)

            # Risk-adjusted returns (Sharpe-like)
            rolling_vol = df['returns'].rolling(window=20).std()
            df[f'risk_adj_return_{horizon}d'] = df[f'future_return_{horizon}d'] / rolling_vol

        return df

    def create_time_series_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-series aware train/val/test splits (no look-ahead bias)."""

        # Sort by date
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)

        # Get unique dates
        unique_dates = df['date'].unique()
        n_dates = len(unique_dates)

        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        test_dates = unique_dates[val_end_idx:]

        train_df = df[df['date'].isin(train_dates)].copy()
        val_df = df[df['date'].isin(val_dates)].copy()
        test_df = df[df['date'].isin(test_dates)].copy()

        logger.info(f"Train: {len(train_df)} samples ({train_dates[0]} to {train_dates[-1]})")
        logger.info(f"Val: {len(val_df)} samples ({val_dates[0]} to {val_dates[-1]})")
        logger.info(f"Test: {len(test_df)} samples ({test_dates[0]} to {test_dates[-1]})")

        return train_df, val_df, test_df

    def run_data_generation(self, max_stocks: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Run the complete data generation pipeline."""
        logger.info("="*60)
        logger.info("Starting ML Training Data Generation")
        logger.info(f"Stocks: {len(self.stocks)}, Years: {self.years_history}")
        logger.info("="*60)

        stocks_to_process = self.stocks[:max_stocks] if max_stocks else self.stocks

        all_data = []
        successful = 0
        failed = 0

        # Fetch data for all stocks
        for i, ticker in enumerate(stocks_to_process):
            logger.info(f"Processing {i+1}/{len(stocks_to_process)}: {ticker}")

            df = self.fetch_stock_data(ticker)
            if df is not None and len(df) >= 200:  # Need at least 200 days
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df)

                # Generate labels
                df = self.generate_labels(df)

                # Remove rows with NaN in critical columns
                df = df.dropna(subset=['close', 'returns', 'rsi_14', 'macd'])

                if len(df) >= 100:  # Keep only if we have enough data after cleaning
                    all_data.append(df)
                    successful += 1
                else:
                    logger.warning(f"Insufficient data after cleaning for {ticker}")
                    failed += 1
            else:
                failed += 1

        if not all_data:
            logger.error("No data collected!")
            return {}

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df)} total samples from {successful} stocks")

        # Create train/val/test splits
        train_df, val_df, test_df = self.create_time_series_splits(combined_df)

        # Save raw combined data
        raw_path = self.output_dir / 'raw' / 'all_stocks_raw.parquet'
        combined_df.to_parquet(raw_path, index=False)
        logger.info(f"Saved raw data to {raw_path}")

        # Save processed splits
        train_path = self.output_dir / 'processed' / 'train_data.parquet'
        val_path = self.output_dir / 'processed' / 'val_data.parquet'
        test_path = self.output_dir / 'processed' / 'test_data.parquet'

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info(f"Saved train data: {train_path}")
        logger.info(f"Saved val data: {val_path}")
        logger.info(f"Saved test data: {test_path}")

        # Save metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'stocks_processed': successful,
            'stocks_failed': failed,
            'total_samples': len(combined_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'date_range': {
                'start': str(combined_df['date'].min()),
                'end': str(combined_df['date'].max())
            },
            'feature_columns': list(combined_df.columns),
            'label_columns': [c for c in combined_df.columns if 'future_return' in c or 'direction' in c]
        }

        import json
        metadata_path = self.output_dir / 'processed' / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info("="*60)
        logger.info(f"Data generation complete! {successful} stocks, {len(combined_df)} total samples")
        logger.info("="*60)

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'metadata': metadata
        }


def main():
    """Main entry point for data generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate ML training data')
    parser.add_argument('--output-dir', type=str, default='data/ml_training',
                        help='Output directory for training data')
    parser.add_argument('--years', type=int, default=2,
                        help='Years of historical data to fetch')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='Maximum number of stocks to process (None for all)')

    args = parser.parse_args()

    generator = MLTrainingDataGenerator(
        output_dir=args.output_dir,
        years_history=args.years
    )

    result = generator.run_data_generation(max_stocks=args.max_stocks)

    if result:
        print(f"\nData generation successful!")
        print(f"Train samples: {len(result['train'])}")
        print(f"Val samples: {len(result['val'])}")
        print(f"Test samples: {len(result['test'])}")


if __name__ == '__main__':
    main()
