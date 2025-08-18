#!/usr/bin/env python3
"""Quick API Connection Test"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_apis():
    print("\n🔍 Testing API Connections...\n")
    
    # Test Finnhub
    try:
        from backend.data_ingestion.finnhub_client import FinnhubClient
        client = FinnhubClient()
        data = client.get_stock_data("AAPL")
        if data and 'current_price' in data:
            print(f"✅ Finnhub API: Connected - AAPL: ${data['current_price']:.2f}")
        else:
            print("⚠️ Finnhub API: Connected but no data received")
    except Exception as e:
        print(f"❌ Finnhub API: {str(e)[:100]}")
    
    # Test Alpha Vantage
    try:
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
        client = AlphaVantageClient()
        data = client.get_stock_data("AAPL")
        if data and 'current_price' in data:
            print(f"✅ Alpha Vantage API: Connected - AAPL: ${data['current_price']:.2f}")
        else:
            print("⚠️ Alpha Vantage API: Connected but no data received")
    except Exception as e:
        print(f"❌ Alpha Vantage API: {str(e)[:100]}")
    
    # Test Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis: Connected")
    except Exception as e:
        print(f"⚠️ Redis: Not connected (caching disabled)")
    
    # Test Cost Monitor
    try:
        from backend.utils.cost_monitor import CostMonitor
        monitor = CostMonitor()
        print(f"✅ Cost Monitor: Active (Budget: ${monitor.monthly_budget:.2f}/month)")
    except Exception as e:
        print(f"⚠️ Cost Monitor: {str(e)[:100]}")
    
    print("\n✨ API connections ready for data pipeline activation!\n")

if __name__ == "__main__":
    test_apis()