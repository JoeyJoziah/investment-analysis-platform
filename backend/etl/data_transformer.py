"""
Data Transformation Module
Handles data cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    import pandas_ta as ta

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transform raw financial data into analysis-ready format"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        
    def transform_price_data(self, raw_data: Dict) -> pd.DataFrame:
        """Transform raw price data into structured DataFrame"""
        try:
            if 'sources' not in raw_data:
                logger.error("No sources in raw data")
                return pd.DataFrame()
            
            # Prioritize data sources (yfinance > polygon > finnhub)
            price_df = None
            
            # Try yfinance first
            if 'yfinance' in raw_data['sources']:
                yf_data = raw_data['sources']['yfinance']
                if 'price_data' in yf_data and 'history' in yf_data['price_data']:
                    price_df = pd.DataFrame(yf_data['price_data']['history'])
                    if 'Date' in price_df.columns:
                        price_df['date'] = pd.to_datetime(price_df['Date'])
                    else:
                        price_df['date'] = pd.to_datetime(price_df.index)
            
            # Fallback to Polygon
            if price_df is None and 'polygon' in raw_data['sources']:
                poly_data = raw_data['sources']['polygon']
                if 'aggregates' in poly_data and poly_data['aggregates']:
                    price_df = pd.DataFrame(poly_data['aggregates'])
                    price_df['date'] = pd.to_datetime(price_df['t'], unit='ms')
                    price_df.rename(columns={
                        'o': 'Open', 'h': 'High', 'l': 'Low', 
                        'c': 'Close', 'v': 'Volume'
                    }, inplace=True)
            
            if price_df is None:
                logger.warning(f"No price data available for {raw_data.get('ticker')}")
                return pd.DataFrame()
            
            # Standardize column names
            price_df.columns = price_df.columns.str.lower()
            
            # Ensure required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in price_df.columns:
                    logger.warning(f"Missing column: {col}")
                    if col != 'date':
                        price_df[col] = np.nan
            
            # Sort by date
            price_df = price_df.sort_values('date')
            
            # Add ticker column
            price_df['ticker'] = raw_data.get('ticker')
            
            # Clean data
            price_df = self.clean_price_data(price_df)
            
            # Add derived features
            price_df = self.add_price_features(price_df)
            
            return price_df
            
        except Exception as e:
            logger.error(f"Error transforming price data: {e}")
            return pd.DataFrame()
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Handle missing values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Forward fill for minor gaps
                df[col] = df[col].fillna(method='ffill', limit=2)
                # Interpolate for remaining
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # Remove outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df = self.remove_outliers(df, col)
        
        # Validate OHLC relationships
        df = df[
            (df['low'] <= df['high']) &
            (df['low'] <= df['open']) &
            (df['open'] <= df['high']) &
            (df['low'] <= df['close']) &
            (df['close'] <= df['high'])
        ]
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, column: str, z_threshold: float = 3) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        if column not in df.columns or df[column].isna().all():
            return df
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        
        # Mark outliers
        outlier_indices = df[column].dropna().index[z_scores > z_threshold]
        
        # Replace outliers with interpolated values
        df.loc[outlier_indices, column] = np.nan
        df[column] = df[column].interpolate(method='linear', limit_direction='both')
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived price features"""
        if df.empty:
            return df
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()
        
        # Volatility measures
        df['intraday_range'] = (df['high'] - df['low']) / df['open']
        df['close_to_open'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Price position
        df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap analysis
        df['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 20:
            logger.warning("Insufficient data for technical indicators")
            return df
        
        try:
            if HAS_TALIB:
                # Use TA-Lib if available
                close = df['close'].values
                high = df['high'].values
                low = df['low'].values
                volume = df['volume'].values
                
                # Moving Averages
                df['sma_5'] = talib.SMA(close, timeperiod=5)
                df['sma_10'] = talib.SMA(close, timeperiod=10)
                df['sma_20'] = talib.SMA(close, timeperiod=20)
                df['sma_50'] = talib.SMA(close, timeperiod=50)
                df['sma_200'] = talib.SMA(close, timeperiod=200)
                
                df['ema_12'] = talib.EMA(close, timeperiod=12)
                df['ema_26'] = talib.EMA(close, timeperiod=26)
                
                # MACD
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                    close, fastperiod=12, slowperiod=26, signalperiod=9
                )
                
                # RSI
                df['rsi_14'] = talib.RSI(close, timeperiod=14)
                df['rsi_7'] = talib.RSI(close, timeperiod=7)
                
                # Stochastic
                df['stoch_k'], df['stoch_d'] = talib.STOCH(
                    high, low, close, 
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                
                # Bollinger Bands
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                    close, timeperiod=20, nbdevup=2, nbdevdn=2
                )
                
                # ATR (Average True Range)
                df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
                
                # ADX (Average Directional Index)
                df['adx'] = talib.ADX(high, low, close, timeperiod=14)
                
                # CCI (Commodity Channel Index)
                df['cci'] = talib.CCI(high, low, close, timeperiod=14)
                
                # MFI (Money Flow Index)
                df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
                
                # OBV (On Balance Volume)
                df['obv'] = talib.OBV(close, volume)
                
                # Williams %R
                df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
                
                # Momentum indicators
                df['roc_10'] = talib.ROC(close, timeperiod=10)
                df['momentum_10'] = talib.MOM(close, timeperiod=10)
                
                # Pattern recognition (candlestick patterns)
                df['pattern_doji'] = talib.CDLDOJI(df['open'], high, low, close)
                df['pattern_hammer'] = talib.CDLHAMMER(df['open'], high, low, close)
                df['pattern_engulfing'] = talib.CDLENGULFING(df['open'], high, low, close)
                
            else:
                # Use pandas_ta as fallback
                # Moving Averages
                df['sma_5'] = df['close'].rolling(window=5).mean()
                df['sma_10'] = df['close'].rolling(window=10).mean()
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['sma_200'] = df['close'].rolling(window=200).mean()
                
                df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                
                # MACD
                macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if macd_result is not None and not macd_result.empty:
                    df['macd'] = macd_result.iloc[:, 0]
                    df['macd_hist'] = macd_result.iloc[:, 1]
                    df['macd_signal'] = macd_result.iloc[:, 2]
                
                # RSI
                df['rsi_14'] = ta.rsi(df['close'], length=14)
                df['rsi_7'] = ta.rsi(df['close'], length=7)
                
                # Stochastic
                stoch_result = ta.stoch(df['high'], df['low'], df['close'])
                if stoch_result is not None and not stoch_result.empty:
                    df['stoch_k'] = stoch_result.iloc[:, 0]
                    df['stoch_d'] = stoch_result.iloc[:, 1]
                
                # Bollinger Bands
                bb_result = ta.bbands(df['close'], length=20, std=2)
                if bb_result is not None and not bb_result.empty:
                    df['bb_lower'] = bb_result.iloc[:, 0]
                    df['bb_middle'] = bb_result.iloc[:, 1]
                    df['bb_upper'] = bb_result.iloc[:, 2]
                
                # ATR
                df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                
                # ADX
                df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
                
                # CCI
                df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
                
                # MFI
                df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
                
                # OBV
                df['obv'] = ta.obv(df['close'], df['volume'])
                
                # Williams %R
                df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
                
                # ROC and Momentum
                df['roc_10'] = ta.roc(df['close'], length=10)
                df['momentum_10'] = ta.mom(df['close'], length=10)
            
            # Common calculations (work with both libraries)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Support and Resistance levels
            df['resistance_1'] = df['high'].rolling(window=20).max()
            df['support_1'] = df['low'].rolling(window=20).min()
            
            # Trend indicators
            df['trend_sma'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['trend_macd'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return df
    
    def calculate_fundamental_features(self, df: pd.DataFrame, company_info: Dict) -> pd.DataFrame:
        """Add fundamental analysis features"""
        if not company_info:
            return df
        
        try:
            # Add static fundamental data
            df['market_cap'] = company_info.get('market_cap')
            df['pe_ratio'] = company_info.get('pe_ratio')
            df['dividend_yield'] = company_info.get('dividend_yield')
            df['beta'] = company_info.get('beta')
            
            # Calculate valuation metrics
            if 'pe_ratio' in company_info and company_info['pe_ratio']:
                df['earnings_yield'] = 1 / company_info['pe_ratio']
            
            # Sector and industry
            df['sector'] = company_info.get('sector')
            df['industry'] = company_info.get('industry')
            
        except Exception as e:
            logger.error(f"Error adding fundamental features: {e}")
        
        return df
    
    def transform_sentiment_data(self, sentiment_data: Dict) -> Dict:
        """Transform and aggregate sentiment data"""
        try:
            if not sentiment_data:
                return {
                    'sentiment_score': 0,
                    'sentiment_confidence': 0,
                    'article_count': 0
                }
            
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            article_count = sentiment_data.get('article_count', 0)
            
            # Calculate confidence based on article count
            confidence = min(article_count / 10, 1.0)  # Max confidence at 10+ articles
            
            # Normalize sentiment score
            normalized_score = np.tanh(sentiment_score)  # Bound between -1 and 1
            
            return {
                'sentiment_score': normalized_score,
                'sentiment_confidence': confidence,
                'article_count': article_count,
                'sentiment_trend': self.calculate_sentiment_trend(sentiment_data)
            }
            
        except Exception as e:
            logger.error(f"Error transforming sentiment data: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_confidence': 0,
                'article_count': 0
            }
    
    def calculate_sentiment_trend(self, sentiment_data: Dict) -> str:
        """Calculate sentiment trend from articles"""
        if 'articles' not in sentiment_data:
            return 'neutral'
        
        articles = sentiment_data['articles']
        if not articles:
            return 'neutral'
        
        # Sort by date and analyze trend
        recent_sentiments = []
        for article in articles[:5]:  # Last 5 articles
            text = article.get('title', '') + ' ' + article.get('description', '')
            # Simple sentiment analysis
            if any(word in text.lower() for word in ['upgrade', 'beat', 'strong', 'gain']):
                recent_sentiments.append(1)
            elif any(word in text.lower() for word in ['downgrade', 'miss', 'weak', 'loss']):
                recent_sentiments.append(-1)
            else:
                recent_sentiments.append(0)
        
        avg_sentiment = np.mean(recent_sentiments)
        
        if avg_sentiment > 0.3:
            return 'improving'
        elif avg_sentiment < -0.3:
            return 'declining'
        else:
            return 'neutral'
    
    def create_feature_matrix(self, price_df: pd.DataFrame, 
                            sentiment_data: Dict = None,
                            company_info: Dict = None) -> pd.DataFrame:
        """Create comprehensive feature matrix for ML models"""
        try:
            # Start with price and technical features
            features_df = price_df.copy()
            
            # Add technical indicators
            features_df = self.calculate_technical_indicators(features_df)
            
            # Add fundamental features
            if company_info:
                features_df = self.calculate_fundamental_features(features_df, company_info)
            
            # Add sentiment features
            if sentiment_data:
                sentiment_features = self.transform_sentiment_data(sentiment_data)
                for key, value in sentiment_features.items():
                    features_df[key] = value
            
            # Create lag features
            lag_periods = [1, 3, 5, 10]
            for period in lag_periods:
                features_df[f'return_{period}d'] = features_df['close'].pct_change(period)
                features_df[f'volume_ma_{period}d'] = features_df['volume'].rolling(period).mean()
            
            # Create interaction features
            features_df['price_volume'] = features_df['close'] * features_df['volume']
            features_df['rsi_volume'] = features_df['rsi_14'] * features_df['volume_ratio']
            
            # Handle missing values
            features_df = features_df.fillna(method='ffill', limit=2)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return price_df
    
    def normalize_features(self, df: pd.DataFrame, 
                          exclude_cols: List[str] = None) -> pd.DataFrame:
        """Normalize numerical features for ML models"""
        if exclude_cols is None:
            exclude_cols = ['date', 'ticker', 'sector', 'industry']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if cols_to_normalize:
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            target_col: str = 'next_day_return',
                            lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML model training"""
        try:
            # Create target variable
            df[target_col] = df['close'].shift(-1) / df['close'] - 1
            
            # Select features
            feature_cols = [col for col in df.columns if col not in [
                'date', 'ticker', 'sector', 'industry', target_col
            ]]
            
            # Create sequences for time series models
            X, y = [], []
            
            for i in range(lookback, len(df) - 1):
                X.append(df[feature_cols].iloc[i-lookback:i].values)
                y.append(df[target_col].iloc[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])


class DataAggregator:
    """Aggregate data from multiple sources and time periods"""
    
    @staticmethod
    def merge_price_data(data_sources: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge price data from multiple sources"""
        if not data_sources:
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined = pd.concat(data_sources, ignore_index=True)
        
        # Group by date and ticker, taking mean of numeric columns
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        
        aggregated = combined.groupby(['date', 'ticker']).agg({
            **{col: 'mean' for col in numeric_cols},
            'sector': 'first',
            'industry': 'first'
        }).reset_index()
        
        return aggregated
    
    @staticmethod
    def calculate_market_metrics(df: pd.DataFrame) -> Dict:
        """Calculate market-wide metrics"""
        metrics = {}
        
        if df.empty:
            return metrics
        
        # Market breadth
        latest_date = df['date'].max()
        latest_data = df[df['date'] == latest_date]
        
        if not latest_data.empty:
            metrics['advances'] = (latest_data['price_change'] > 0).sum()
            metrics['declines'] = (latest_data['price_change'] < 0).sum()
            metrics['unchanged'] = (latest_data['price_change'] == 0).sum()
            metrics['advance_decline_ratio'] = (
                metrics['advances'] / metrics['declines'] 
                if metrics['declines'] > 0 else np.inf
            )
            
            # Market momentum
            metrics['avg_rsi'] = latest_data['rsi_14'].mean()
            metrics['percent_above_sma20'] = (
                (latest_data['close'] > latest_data['sma_20']).sum() / 
                len(latest_data) * 100
            )
            
            # Volatility
            metrics['avg_atr'] = latest_data['atr_14'].mean()
            metrics['market_volatility'] = latest_data['price_change'].std()
        
        return metrics


if __name__ == "__main__":
    # Test the transformer
    transformer = DataTransformer()
    
    # Sample data for testing
    sample_data = {
        'ticker': 'AAPL',
        'sources': {
            'yfinance': {
                'price_data': {
                    'history': [
                        {'Date': '2024-01-01', 'Open': 100, 'High': 105, 
                         'Low': 99, 'Close': 103, 'Volume': 1000000},
                        {'Date': '2024-01-02', 'Open': 103, 'High': 106, 
                         'Low': 102, 'Close': 105, 'Volume': 1100000}
                    ]
                },
                'company_info': {
                    'market_cap': 3000000000000,
                    'pe_ratio': 30,
                    'sector': 'Technology'
                }
            }
        }
    }
    
    # Transform data
    price_df = transformer.transform_price_data(sample_data)
    print("Transformed price data:")
    print(price_df.head())
    
    # Calculate indicators
    features_df = transformer.create_feature_matrix(
        price_df, 
        company_info=sample_data['sources']['yfinance']['company_info']
    )
    print("\nFeature matrix:")
    print(features_df.head())