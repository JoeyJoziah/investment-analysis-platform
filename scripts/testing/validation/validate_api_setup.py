#!/usr/bin/env python3
"""
Quick API Setup Validation Script
Validates that API connections are working for production deployment
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


async def validate_setup() -> Dict[str, Any]:
    """Validate that all critical components are working"""
    results = {
        'dependencies': {},
        'services': {},
        'apis': {},
        'overall_status': 'UNKNOWN'
    }
    
    print("🔍 Investment Analysis Platform - API Setup Validation")
    print("=" * 60)
    
    # 1. Check Dependencies
    print("\n📦 Checking Dependencies...")
    
    try:
        import aiohttp
        results['dependencies']['aiohttp'] = f"✅ {aiohttp.__version__}"
        print(f"  ✅ aiohttp: {aiohttp.__version__}")
    except ImportError:
        results['dependencies']['aiohttp'] = "❌ Missing"
        print(f"  ❌ aiohttp: Missing - install with: pip install aiohttp")
    
    try:
        import backoff
        results['dependencies']['backoff'] = "✅ Available"
        print(f"  ✅ backoff: Available")
    except ImportError:
        results['dependencies']['backoff'] = "❌ Missing"
        print(f"  ❌ backoff: Missing - install with: pip install backoff")
    
    try:
        import requests
        results['dependencies']['requests'] = f"✅ {requests.__version__}"
        print(f"  ✅ requests: {requests.__version__}")
    except ImportError:
        results['dependencies']['requests'] = "❌ Missing"
        print(f"  ❌ requests: Missing - install with: pip install requests")
    
    # 2. Check Services
    print("\n🔧 Checking Services...")
    
    # Redis
    try:
        from backend.utils.cache import get_redis
        redis = await get_redis()
        await redis.set("test_key", "test_value", ex=10)
        value = await redis.get("test_key")
        await redis.delete("test_key")
        if value == "test_value":
            results['services']['redis'] = "✅ Connected"
            print(f"  ✅ Redis: Connected and operational")
        else:
            results['services']['redis'] = "❌ Not working"
            print(f"  ❌ Redis: Connected but operations failed")
    except Exception as e:
        results['services']['redis'] = f"❌ {str(e)}"
        print(f"  ❌ Redis: {e}")
    
    # Cost Monitor
    try:
        from backend.utils.cost_monitor import cost_monitor
        await cost_monitor.initialize()
        can_call = await cost_monitor.check_api_limit("test_provider", "test_endpoint")
        results['services']['cost_monitor'] = "✅ Working"
        print(f"  ✅ Cost Monitor: Initialized and working")
    except Exception as e:
        results['services']['cost_monitor'] = f"❌ {str(e)}"
        print(f"  ❌ Cost Monitor: {e}")
    
    # 3. Check APIs
    print("\n🌐 Checking API Connections...")
    
    # Finnhub
    try:
        from backend.data_ingestion.finnhub_client import FinnhubClient
        client = FinnhubClient()
        if not client.api_key:
            results['apis']['finnhub'] = "⚠️ No API key"
            print(f"  ⚠️ Finnhub: No API key (set FINNHUB_API_KEY)")
        else:
            async with client:
                quote = await client.get_quote('AAPL')
                if quote and 'current_price' in quote:
                    results['apis']['finnhub'] = f"✅ Working - AAPL: ${quote['current_price']}"
                    print(f"  ✅ Finnhub: Working - AAPL: ${quote['current_price']}")
                else:
                    results['apis']['finnhub'] = "❌ No data returned"
                    print(f"  ❌ Finnhub: Connected but no data returned")
    except Exception as e:
        results['apis']['finnhub'] = f"❌ {str(e)}"
        print(f"  ❌ Finnhub: {e}")
    
    # Alpha Vantage
    try:
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
        client = AlphaVantageClient()
        if not client.api_key:
            results['apis']['alpha_vantage'] = "⚠️ No API key"
            print(f"  ⚠️ Alpha Vantage: No API key (set ALPHA_VANTAGE_API_KEY)")
        else:
            async with client:
                quote = await client.get_quote('AAPL')
                if quote and 'price' in quote:
                    results['apis']['alpha_vantage'] = f"✅ Working - AAPL: ${quote['price']}"
                    print(f"  ✅ Alpha Vantage: Working - AAPL: ${quote['price']}")
                else:
                    results['apis']['alpha_vantage'] = "❌ No data returned"
                    print(f"  ❌ Alpha Vantage: Connected but no data returned")
    except Exception as e:
        results['apis']['alpha_vantage'] = f"❌ {str(e)}"
        print(f"  ❌ Alpha Vantage: {e}")
    
    # Polygon
    try:
        from backend.data_ingestion.polygon_client import PolygonClient
        client = PolygonClient()
        if not client.api_key:
            results['apis']['polygon'] = "⚠️ No API key"
            print(f"  ⚠️ Polygon: No API key (set POLYGON_API_KEY)")
        else:
            results['apis']['polygon'] = "✅ API key configured"
            print(f"  ✅ Polygon: API key configured (not tested to preserve rate limit)")
    except Exception as e:
        results['apis']['polygon'] = f"❌ {str(e)}"
        print(f"  ❌ Polygon: {e}")
    
    # 4. Overall Status
    print("\n📊 Overall Status")
    print("-" * 30)
    
    critical_components = [
        '✅' in str(results['dependencies']['aiohttp']),
        '✅' in str(results['dependencies']['backoff']),
        '✅' in str(results['services']['redis']),
        '✅' in str(results['services']['cost_monitor']),
        '✅' in str(results['apis']['finnhub']) or '✅' in str(results['apis']['alpha_vantage'])
    ]
    
    # Count working APIs for additional info
    working_apis = sum([
        '✅' in str(results['apis']['finnhub']),
        '✅' in str(results['apis']['alpha_vantage']),
        '✅' in str(results['apis']['polygon'])
    ])
    
    working_count = sum(critical_components)
    total_count = len(critical_components)
    
    if working_count == total_count:
        results['overall_status'] = "✅ READY FOR PRODUCTION"
        print(f"✅ Status: READY FOR PRODUCTION ({working_count}/{total_count} critical components working)")
        print(f"📊 APIs Connected: {working_apis}/3 (Finnhub, Alpha Vantage, Polygon)")
        if working_apis >= 2:
            print("🎉 Your investment analysis platform is ready to go!")
        else:
            print("💡 Consider adding more API keys for redundancy")
    elif working_count >= 4:
        results['overall_status'] = "⚠️ MOSTLY READY"
        print(f"⚠️ Status: MOSTLY READY ({working_count}/{total_count} critical components working)")
        print(f"📊 APIs Connected: {working_apis}/3 (Finnhub, Alpha Vantage, Polygon)")
        print("💡 Consider setting up missing components for full functionality")
    else:
        results['overall_status'] = "❌ NEEDS SETUP"
        print(f"❌ Status: NEEDS SETUP ({working_count}/{total_count} critical components working)")
        print(f"📊 APIs Connected: {working_apis}/3 (Finnhub, Alpha Vantage, Polygon)")
        print("🔧 Please fix the issues above before deployment")
    
    print("\n" + "=" * 60)
    
    return results


def main():
    """Main validation function"""
    try:
        results = asyncio.run(validate_setup())
        
        # Exit codes for automation
        if "✅ READY FOR PRODUCTION" in results['overall_status']:
            sys.exit(0)  # Success
        elif "⚠️ MOSTLY READY" in results['overall_status']:
            sys.exit(1)  # Warning - some issues
        else:
            sys.exit(2)  # Error - major issues
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()