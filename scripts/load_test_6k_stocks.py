"""
Load test: 6,000+ stock concurrent simulation.

Exercises the recommendation, watchlist, and stock-detail surfaces under realistic
6K-symbol load to validate the canonical performance target from IMPLEMENTATION_STATUS.

Usage:
    # Quick smoke (50 users, 60s):
    locust -f scripts/load_test_6k_stocks.py --host=http://localhost:8000 \
           --users=50 --spawn-rate=10 --run-time=60s --headless

    # Full target (500 users, 5min):
    locust -f scripts/load_test_6k_stocks.py --host=http://localhost:8000 \
           --users=500 --spawn-rate=25 --run-time=5m --headless --csv=load_results

Performance targets (from docs/IMPLEMENTATION_STATUS.md):
    API response p95: < 500ms
    Cache hit rate:   > 80%
    ML inference p95: < 100ms
"""
from __future__ import annotations

import random
from typing import List

from locust import HttpUser, between, events, task

# Realistic 6K-symbol distribution. Heavy tail mirrors real market;
# top 200 are
# ~80% of attention, bottom 5800 are ~20%. This hits the cache the way real users do.
TOP_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "XOM",
    "JNJ", "V", "WMT", "JPM", "MA", "PG", "HD", "CVX", "LLY", "ABBV", "MRK", "PEP",
    "AVGO", "KO", "COST", "ADBE", "BAC", "CSCO", "PFE", "TMO", "DIS", "ACN", "ABT",
    "MCD", "NFLX", "DHR", "WFC", "CMCSA", "VZ", "AMD", "NEE", "TXN", "PM", "RTX",
    "INTC", "HON", "QCOM", "T", "LIN", "BMY", "UPS", "LOW", "AMGN", "ORCL", "MS",
    "GS", "INTU", "IBM", "DE", "NKE", "ELV", "BA", "CAT", "MDT", "GE", "AXP", "C",
    "BLK", "BKNG", "SBUX", "ISRG", "TJX", "AMT", "ADI", "ZTS", "MMC", "VRTX", "GILD",
    "AMAT", "REGN", "SCHW", "CB", "PYPL", "MO", "DUK", "PLD", "EOG", "SLB", "ETN",
    "BSX", "SO", "ITW", "TGT", "PNC", "CI", "USB", "CME", "MU", "FI", "EQIX", "CVS",
]


def _full_universe() -> List[str]:
    """Synthesize a 6K-symbol universe by extending
    the realistic head with placeholders.
    """
    base = TOP_SYMBOLS * 1
    # Pad to 6,000 with placeholder symbols. Backend should treat unknown symbols
    # gracefully (404 or empty result), which still exercises routing, auth, validation.
    pad_count = 6000 - len(base)
    base.extend(f"SYN{i:04d}" for i in range(pad_count))
    return base


SYMBOLS = _full_universe()


def pick_symbol() -> str:
    """80/20 distribution: 80% chance of a top-100 symbol, 20% of the long tail."""
    if random.random() < 0.80:
        return random.choice(TOP_SYMBOLS)
    return random.choice(SYMBOLS)


class StockUser(HttpUser):
    """Simulates a typical user browsing the platform."""

    wait_time = between(0.5, 3.0)

    def on_start(self):
        # No auth in this scenario; if needed, set SMOKE_AUTH_TOKEN and uncomment:
        # token = os.getenv("LOAD_AUTH_TOKEN")
        # if token:
        #     self.client.headers.update({"Authorization": f"Bearer {token}"})
        pass

    @task(40)
    def view_stock_detail(self):
        sym = pick_symbol()
        with self.client.get(
            f"/api/v1/stocks/{sym}",
            name="/api/v1/stocks/{symbol}",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 404):
                r.success()
            else:
                r.failure(f"unexpected {r.status_code}")

    @task(20)
    def get_recommendation(self):
        sym = pick_symbol()
        with self.client.get(
            f"/api/v1/recommendations/{sym}",
            name="/api/v1/recommendations/{symbol}",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 404):
                r.success()
            else:
                r.failure(f"unexpected {r.status_code}")

    @task(10)
    def list_top_recommendations(self):
        with self.client.get(
            "/api/v1/recommendations/?limit=20",
            name="/api/v1/recommendations/?limit",
            catch_response=True,
        ) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"unexpected {r.status_code}")

    @task(10)
    def market_overview(self):
        with self.client.get(
            "/api/v1/stocks/?limit=50",
            name="/api/v1/stocks/?limit",
            catch_response=True,
        ) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"unexpected {r.status_code}")

    @task(5)
    def health_probe(self):
        self.client.get("/health", name="/health")


@events.test_stop.add_listener
def _print_summary(environment, **kwargs):
    stats = environment.stats.total
    p95 = stats.get_response_time_percentile(0.95) if stats.num_requests else 0
    p99 = stats.get_response_time_percentile(0.99) if stats.num_requests else 0
    print()
    print("=" * 60)
    print("6K-STOCK LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total requests:     {stats.num_requests}")
    print(f"Failures:           {stats.num_failures}")
    print(f"Failure rate:       {stats.fail_ratio*100:.2f}%")
    print(f"Median (p50):       {stats.median_response_time} ms")
    print(f"p95:                {p95:.0f} ms      (target: < 500 ms)")
    print(f"p99:                {p99:.0f} ms")
    print(f"Avg req/sec:        {stats.total_rps:.1f}")
    print("=" * 60)
    print("PASS" if p95 < 500 and stats.fail_ratio < 0.01 else "FAIL")
    print("=" * 60)
