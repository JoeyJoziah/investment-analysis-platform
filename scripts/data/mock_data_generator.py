#!/usr/bin/env python3
"""
Mock Data Generator for Development
Generates realistic stock market data for testing without external API dependencies.
"""

import random
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

class MockDataGenerator:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'investment_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '9v1g^OV9XUwzUP6cEgCYgNOE')
        }
        
        # Sample stock symbols (S&P 100 subset)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK.B', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX',
            'CMCSA', 'XOM', 'CVX', 'PFE', 'KO', 'TMO', 'ABBV', 'WMT', 'PEP', 'AVGO',
            'CSCO', 'VZ', 'ABT', 'ORCL', 'NKE', 'ACN', 'MRK', 'COST', 'WFC', 'LLY',
            'INTC', 'MCD', 'T', 'DHR', 'TXN', 'MS', 'UNP', 'PM', 'NEE', 'HON',
            'AMD', 'BMY', 'LOW', 'QCOM', 'RTX', 'LIN', 'SBUX', 'SPGI', 'CVS', 'AMT',
            'GS', 'INTU', 'CAT', 'BA', 'ISRG', 'DE', 'BLK', 'MDLZ', 'MMM', 'GILD',
            'AMGN', 'AXP', 'SYK', 'TGT', 'BKNG', 'ADI', 'GE', 'NOW', 'VRTX', 'MO',
            'PLD', 'ZTS', 'LRCX', 'CHTR', 'TJX', 'TMUS', 'PNC', 'C', 'USB', 'DUK',
            'MMC', 'BDX', 'SO', 'CL', 'CI', 'SHW', 'SCHW', 'FIS', 'ITW', 'EL'
        ]
        
        self.company_names = {
            'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms Inc.', 'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation', 'BRK.B': 'Berkshire Hathaway Inc.', 'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson', 'V': 'Visa Inc.', 'PG': 'Procter & Gamble Co.',
            'UNH': 'UnitedHealth Group Inc.', 'HD': 'The Home Depot Inc.', 'MA': 'Mastercard Inc.'
        }
        
        self.sectors = [
            'Technology', 'Healthcare', 'Financial', 'Consumer Discretionary',
            'Consumer Staples', 'Energy', 'Industrials', 'Communication Services',
            'Utilities', 'Real Estate', 'Materials'
        ]
        
        self.industries = [
            'Software', 'Hardware', 'Biotechnology', 'Banking', 'Insurance',
            'Retail', 'E-commerce', 'Entertainment', 'Pharmaceuticals', 'Semiconductors',
            'Automotive', 'Aerospace', 'Telecommunications', 'Food & Beverage'
        ]
        
        self.exchanges = ['NYSE', 'NASDAQ', 'American Stock Exchange']
        
        self.news_headlines = [
            "Company Reports Strong Q{} Earnings, Beats Expectations",
            "Analysts Upgrade {} to Buy Rating",
            "New Product Launch Drives {} Stock Higher",
            "{} Announces Strategic Partnership",
            "Market Volatility Affects {} Performance",
            "{} CEO Discusses Growth Strategy",
            "Regulatory Changes Impact {} Outlook",
            "{} Expands into New Markets",
            "Technology Innovation at {} Shows Promise",
            "Investor Confidence in {} Remains Strong"
        ]
        
    def connect_db(self):
        """Connect to database"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def generate_price_series(self, base_price: float, days: int = 365) -> List[Dict]:
        """Generate realistic price series with volatility"""
        prices = []
        current_price = base_price
        current_date = datetime.now() - timedelta(days=days)
        
        for _ in range(days):
            # Random walk with drift
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily drift, 2% volatility
            current_price *= (1 + daily_return)
            
            # Generate OHLCV data
            open_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
            high_price = max(open_price, current_price) * (1 + np.random.uniform(0, 0.02))
            low_price = min(open_price, current_price) * (1 - np.random.uniform(0, 0.02))
            close_price = current_price
            volume = int(np.random.uniform(1000000, 50000000))
            
            prices.append({
                'date': current_date.date(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'adjusted_close': round(close_price * 0.98, 2)  # Simulate dividend adjustment
            })
            
            current_date += timedelta(days=1)
        
        return prices
    
    def generate_fundamentals(self, symbol: str) -> Dict:
        """Generate realistic fundamental data"""
        market_cap = random.uniform(10e9, 2000e9)  # $10B to $2T
        shares_outstanding = random.uniform(100e6, 10e9)  # 100M to 10B shares
        price = market_cap / shares_outstanding
        
        return {
            'symbol': symbol,
            'market_cap': round(market_cap, 0),
            'pe_ratio': round(random.uniform(10, 40), 2),
            'dividend_yield': round(random.uniform(0, 5), 2),
            'eps': round(random.uniform(1, 20), 2),
            'revenue': round(random.uniform(1e9, 500e9), 0),
            'profit_margin': round(random.uniform(5, 30), 2),
            'roe': round(random.uniform(5, 40), 2),
            'debt_to_equity': round(random.uniform(0.2, 2.5), 2),
            'current_ratio': round(random.uniform(0.8, 3), 2),
            'beta': round(random.uniform(0.5, 2), 2),
            '52_week_high': round(price * 1.3, 2),
            '52_week_low': round(price * 0.7, 2),
            'shares_outstanding': int(shares_outstanding),
            'float': int(shares_outstanding * 0.9)
        }
    
    def generate_technical_indicators(self, prices: List[Dict]) -> Dict:
        """Generate technical indicators from price data"""
        closes = [p['close'] for p in prices[-50:]]  # Last 50 days
        
        # Simple Moving Averages
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma_50 = np.mean(closes) if len(closes) >= 50 else closes[-1]
        
        # RSI calculation (simplified)
        if len(closes) >= 14:
            gains = []
            losses = []
            for i in range(1, 14):
                diff = closes[i] - closes[i-1]
                if diff > 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(diff))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 1
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50
        
        # MACD (simplified)
        if len(closes) >= 26:
            ema_12 = closes[-1]  # Simplified
            ema_26 = np.mean(closes[-26:])
            macd = ema_12 - ema_26
            signal = macd * 0.9  # Simplified signal line
        else:
            macd = 0
            signal = 0
        
        return {
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'ema_12': round(closes[-1] * 1.01, 2),
            'ema_26': round(closes[-1] * 0.99, 2),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'macd_signal': round(signal, 2),
            'bollinger_upper': round(sma_20 * 1.02, 2),
            'bollinger_lower': round(sma_20 * 0.98, 2),
            'volume_sma': int(np.mean([p['volume'] for p in prices[-20:]])),
            'atr': round(random.uniform(1, 5), 2),
            'stochastic_k': round(random.uniform(20, 80), 2),
            'stochastic_d': round(random.uniform(20, 80), 2)
        }
    
    def generate_sentiment_data(self, symbol: str) -> List[Dict]:
        """Generate news sentiment data"""
        sentiments = []
        for i in range(random.randint(3, 10)):
            headline = random.choice(self.news_headlines)
            if '{}' in headline:
                headline = headline.format(symbol)
            elif 'Q{}' in headline:
                headline = headline.format(random.randint(1, 4))
            
            sentiments.append({
                'symbol': symbol,
                'headline': headline,
                'source': random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'MarketWatch']),
                'url': f'https://example.com/news/{symbol.lower()}-{i}',
                'published_at': datetime.now() - timedelta(days=random.randint(0, 30)),
                'sentiment_score': round(random.uniform(-1, 1), 3),
                'relevance_score': round(random.uniform(0.5, 1), 3)
            })
        
        return sentiments
    
    def generate_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """Generate stock recommendations"""
        recommendations = []
        selected_symbols = random.sample(self.symbols[:50], min(num_recommendations, len(self.symbols)))
        
        for symbol in selected_symbols:
            recommendations.append({
                'symbol': symbol,
                'recommendation_type': random.choice(['BUY', 'STRONG_BUY', 'HOLD']),
                'confidence_score': round(random.uniform(0.6, 0.95), 3),
                'target_price': round(random.uniform(100, 500), 2),
                'stop_loss': round(random.uniform(80, 100), 2),
                'rationale': f"Based on strong fundamentals and positive technical indicators",
                'ml_score': round(random.uniform(0.6, 0.9), 3),
                'technical_score': round(random.uniform(0.5, 0.9), 3),
                'fundamental_score': round(random.uniform(0.5, 0.9), 3),
                'sentiment_score': round(random.uniform(0.4, 0.8), 3),
                'risk_level': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'time_horizon': random.choice(['SHORT', 'MEDIUM', 'LONG']),
                'generated_at': datetime.now()
            })
        
        return recommendations
    
    def populate_database(self, limit_symbols: int = 100):
        """Populate database with mock data"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Clear existing mock data
            logger.info("Clearing existing mock data...")
            cursor.execute("DELETE FROM recommendations WHERE reasoning LIKE '%mock data%'")
            cursor.execute("DELETE FROM news_sentiment WHERE source IN ('Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'MarketWatch')")
            
            # Insert exchanges
            logger.info("Inserting exchanges...")
            for exchange in self.exchanges:
                cursor.execute("""
                    INSERT INTO exchanges (name, code, timezone)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET 
                        timezone = EXCLUDED.timezone
                """, (exchange, exchange[:4], 'America/New_York'))
            
            # Get sector IDs first
            cursor.execute("SELECT id, name FROM sectors")
            sector_map = {name: id for id, name in cursor.fetchall()}
            
            # Map industries to appropriate sectors
            industry_sector_map = {
                'Software': 'Technology',
                'Hardware': 'Technology',
                'Semiconductors': 'Technology',
                'Biotechnology': 'Healthcare',
                'Pharmaceuticals': 'Healthcare',
                'Banking': 'Financials',
                'Insurance': 'Financials',
                'Retail': 'Consumer Discretionary',
                'E-commerce': 'Consumer Discretionary',
                'Entertainment': 'Communication Services',
                'Automotive': 'Consumer Discretionary',
                'Aerospace': 'Industrials',
                'Telecommunications': 'Communication Services',
                'Food & Beverage': 'Consumer Staples'
            }
            
            # Insert industries
            logger.info("Inserting industries...")
            for industry in self.industries:
                sector_name = industry_sector_map.get(industry, 'Technology')
                sector_id = sector_map.get(sector_name, sector_map['Technology'])
                cursor.execute("""
                    INSERT INTO industries (name, sector_id)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO NOTHING
                """, (industry, sector_id))
            
            # Get exchange and industry IDs
            cursor.execute("SELECT id, name FROM exchanges")
            exchange_map = {name: id for id, name in cursor.fetchall()}
            
            cursor.execute("SELECT id, name FROM industries")
            industry_map = {name: id for id, name in cursor.fetchall()}
            
            # Insert stocks
            logger.info(f"Inserting {min(limit_symbols, len(self.symbols))} stocks...")
            stock_data = []
            for symbol in self.symbols[:limit_symbols]:
                exchange_id = exchange_map[random.choice(self.exchanges)]
                industry_id = industry_map[random.choice(self.industries)]
                company_name = self.company_names.get(symbol, f"{symbol} Corporation")
                
                # Get sector_id for the industry
                cursor.execute("SELECT sector_id FROM industries WHERE id = %s", (industry_id,))
                sector_id = cursor.fetchone()[0]
                
                stock_data.append((
                    symbol, company_name, exchange_id, industry_id,
                    sector_id, True
                ))
            
            execute_batch(cursor, """
                INSERT INTO stocks (ticker, name, exchange_id, industry_id, sector_id, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    is_active = EXCLUDED.is_active
            """, stock_data)
            
            # Get stock IDs
            cursor.execute("SELECT id, ticker FROM stocks WHERE ticker = ANY(%s)", (self.symbols[:limit_symbols],))
            stock_map = {ticker: id for id, ticker in cursor.fetchall()}
            
            # Insert price history (last 30 days for speed)
            logger.info("Generating price history...")
            for symbol, stock_id in stock_map.items():
                prices = self.generate_price_series(random.uniform(50, 500), days=30)
                price_data = [
                    (stock_id, p['date'], p['open'], p['high'], p['low'], 
                     p['close'], p['volume'], p['adjusted_close'])
                    for p in prices
                ]
                
                execute_batch(cursor, """
                    INSERT INTO price_history 
                    (stock_id, date, open, high, low, close, volume, adjusted_close)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_id, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        adjusted_close = EXCLUDED.adjusted_close
                """, price_data)
            
            # Insert fundamentals
            logger.info("Generating fundamental data...")
            for symbol, stock_id in stock_map.items():
                fund = self.generate_fundamentals(symbol)
                cursor.execute("""
                    INSERT INTO fundamentals 
                    (stock_id, market_cap, pe_ratio, dividend_yield, eps, revenue,
                     profit_margin, roe, debt_to_equity, current_ratio, beta,
                     week_52_high, week_52_low, shares_outstanding, float)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_id) DO UPDATE SET
                        market_cap = EXCLUDED.market_cap,
                        pe_ratio = EXCLUDED.pe_ratio,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    stock_id, fund['market_cap'], fund['pe_ratio'], fund['dividend_yield'],
                    fund['eps'], fund['revenue'], fund['profit_margin'], fund['roe'],
                    fund['debt_to_equity'], fund['current_ratio'], fund['beta'],
                    fund['52_week_high'], fund['52_week_low'], fund['shares_outstanding'],
                    fund['float']
                ))
            
            # Insert technical indicators
            logger.info("Generating technical indicators...")
            for symbol, stock_id in stock_map.items():
                cursor.execute("""
                    SELECT date, close, volume FROM price_history 
                    WHERE stock_id = %s ORDER BY date DESC LIMIT 50
                """, (stock_id,))
                price_data = cursor.fetchall()
                
                if price_data:
                    prices = [{'date': d, 'close': float(c), 'volume': v} 
                             for d, c, v in price_data]
                    indicators = self.generate_technical_indicators(prices)
                    
                    cursor.execute("""
                        INSERT INTO technical_indicators
                        (stock_id, sma_20, sma_50, ema_12, ema_26, rsi, macd, macd_signal,
                         bollinger_upper, bollinger_lower, volume_sma, atr, 
                         stochastic_k, stochastic_d)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (stock_id) DO UPDATE SET
                            sma_20 = EXCLUDED.sma_20,
                            sma_50 = EXCLUDED.sma_50,
                            rsi = EXCLUDED.rsi,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        stock_id, indicators['sma_20'], indicators['sma_50'],
                        indicators['ema_12'], indicators['ema_26'], indicators['rsi'],
                        indicators['macd'], indicators['macd_signal'],
                        indicators['bollinger_upper'], indicators['bollinger_lower'],
                        indicators['volume_sma'], indicators['atr'],
                        indicators['stochastic_k'], indicators['stochastic_d']
                    ))
            
            # Insert sentiment data
            logger.info("Generating sentiment data...")
            for symbol, stock_id in list(stock_map.items())[:20]:  # Limit to 20 stocks for speed
                sentiments = self.generate_sentiment_data(symbol)
                for sent in sentiments:
                    cursor.execute("""
                        INSERT INTO news_sentiment
                        (stock_id, headline, source, url, published_at, 
                         sentiment_score, relevance_score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        stock_id, sent['headline'], sent['source'], sent['url'],
                        sent['published_at'], sent['sentiment_score'], sent['relevance_score']
                    ))
            
            # Insert recommendations
            logger.info("Generating recommendations...")
            recommendations = self.generate_recommendations(10)
            for rec in recommendations:
                if rec['symbol'] in stock_map:
                    stock_id = stock_map[rec['symbol']]
                    
                    # Map recommendation type to action field
                    action_map = {
                        'BUY': 'buy',
                        'STRONG_BUY': 'strong_buy',
                        'HOLD': 'hold'
                    }
                    
                    # Calculate time horizon in days
                    time_horizon_map = {
                        'SHORT': 30,
                        'MEDIUM': 90,
                        'LONG': 365
                    }
                    
                    cursor.execute("""
                        INSERT INTO recommendations
                        (stock_id, action, confidence, target_price, stop_loss, 
                         reasoning, technical_score, fundamental_score, sentiment_score,
                         risk_level, time_horizon_days, is_active, created_at,
                         overall_score, priority)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        stock_id, 
                        action_map[rec['recommendation_type']],
                        rec['confidence_score'],
                        rec['target_price'], 
                        rec['stop_loss'], 
                        rec['rationale'] + ' (mock data)',
                        rec['technical_score'], 
                        rec['fundamental_score'],
                        rec['sentiment_score'], 
                        rec['risk_level'], 
                        time_horizon_map[rec['time_horizon']],
                        True,
                        datetime.now(),
                        rec['ml_score'],  # Use ml_score as overall_score
                        int(rec['confidence_score'] * 10)  # Priority based on confidence
                    ))
            
            conn.commit()
            logger.info("✅ Mock data generation complete!")
            
            # Print summary
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
            stock_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM price_history")
            price_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM recommendations WHERE is_active = true")
            rec_count = cursor.fetchone()[0]
            
            logger.info(f"\nData Summary:")
            logger.info(f"  - Active Stocks: {stock_count}")
            logger.info(f"  - Price Records: {price_count}")
            logger.info(f"  - Active Recommendations: {rec_count}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error populating database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def export_to_json(self, output_dir: str = 'mock_data'):
        """Export mock data to JSON files for offline development"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate sample data
        sample_stock = self.symbols[0]
        prices = self.generate_price_series(150, days=365)
        fundamentals = self.generate_fundamentals(sample_stock)
        indicators = self.generate_technical_indicators(prices)
        sentiments = self.generate_sentiment_data(sample_stock)
        recommendations = self.generate_recommendations(5)
        
        # Save to JSON files
        with open(output_path / 'price_history.json', 'w') as f:
            json.dump(prices[:100], f, indent=2, default=str)
        
        with open(output_path / 'fundamentals.json', 'w') as f:
            json.dump(fundamentals, f, indent=2)
        
        with open(output_path / 'technical_indicators.json', 'w') as f:
            json.dump(indicators, f, indent=2)
        
        with open(output_path / 'sentiments.json', 'w') as f:
            json.dump(sentiments, f, indent=2, default=str)
        
        with open(output_path / 'recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"✅ Mock data exported to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock stock market data')
    parser.add_argument('--symbols', type=int, default=100, 
                       help='Number of symbols to generate (default: 100)')
    parser.add_argument('--export', action='store_true',
                       help='Export data to JSON files')
    parser.add_argument('--output-dir', type=str, default='mock_data',
                       help='Output directory for JSON export')
    
    args = parser.parse_args()
    
    generator = MockDataGenerator()
    
    logger.info("=" * 60)
    logger.info("Mock Data Generator")
    logger.info("=" * 60)
    
    if args.export:
        logger.info("Exporting mock data to JSON files...")
        generator.export_to_json(args.output_dir)
    else:
        logger.info(f"Generating mock data for {args.symbols} symbols...")
        generator.populate_database(args.symbols)
    
    logger.info("=" * 60)
    logger.info("✅ Complete!")


if __name__ == "__main__":
    main()