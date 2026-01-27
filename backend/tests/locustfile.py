"""
Locust Load Testing Configuration

Simulates realistic user behavior with 100 concurrent users
performing common platform operations: dashboard viewing,
portfolio checking, and recommendation retrieval.

Usage:
    locust -f backend/tests/locustfile.py --host=http://localhost:8000
    locust -f backend/tests/locustfile.py --host=http://localhost:8000 -u 100 -r 10 --run-time 300s
"""

from locust import HttpUser, TaskSet, task, between, events
from datetime import datetime, timedelta
import random
import json
import logging
from typing import Dict, List, Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track performance metrics during load testing"""

    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = datetime.now()

    def record_response(self, response_time: float, success: bool):
        """Record a response"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def get_percentile(self, percentile: int) -> float:
        """Calculate response time percentile"""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * (percentile / 100.0))
        return sorted_times[min(index, len(sorted_times) - 1)]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        duration = (datetime.now() - self.start_time).total_seconds()
        total_requests = self.success_count + self.error_count

        return {
            'duration_seconds': duration,
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'error_rate': self.error_count / max(total_requests, 1),
            'avg_response_time': sum(self.response_times) / max(len(self.response_times), 1),
            'p50_response_time': self.get_percentile(50),
            'p95_response_time': self.get_percentile(95),
            'p99_response_time': self.get_percentile(99),
            'requests_per_second': total_requests / max(duration, 1),
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        }


# Global metrics tracking
global_metrics = PerformanceMetrics()


class DashboardTasks(TaskSet):
    """Dashboard-related tasks"""

    @task(1)
    def view_dashboard(self):
        """View main dashboard"""
        with self.client.get(
            "/api/dashboard",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        with self.client.get(
            "/api/portfolio/summary",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_market_overview(self):
        """Get market overview"""
        with self.client.get(
            "/api/market/overview",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)


class PortfolioTasks(TaskSet):
    """Portfolio-related tasks"""

    @task(2)
    def list_holdings(self):
        """List portfolio holdings"""
        with self.client.get(
            "/api/portfolio/holdings",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
                try:
                    data = response.json()
                    if data.get('data'):
                        # Simulate selecting a random stock for detail view
                        self.user.selected_stock = random.choice(data['data']).get('ticker')
                except Exception as e:
                    logger.warning(f"Failed to parse holdings response: {e}")
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(2)
    def get_holding_details(self):
        """Get details for a holding"""
        if hasattr(self.user, 'selected_stock') and self.user.selected_stock:
            ticker = self.user.selected_stock
        else:
            ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

        with self.client.get(
            f"/api/portfolio/holdings/{ticker}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_performance(self):
        """Get portfolio performance"""
        with self.client.get(
            "/api/portfolio/performance",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)


class RecommendationTasks(TaskSet):
    """Recommendation-related tasks"""

    @task(3)
    def get_recommendations(self):
        """Get stock recommendations"""
        params = {
            'limit': random.randint(5, 20),
            'sector': random.choice(['Technology', 'Healthcare', 'Finance', 'Energy', 'All']),
            'min_score': random.uniform(0.5, 0.8)
        }

        with self.client.get(
            "/api/recommendations",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_recommendation_details(self):
        """Get detailed recommendation"""
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD'])

        with self.client.get(
            f"/api/recommendations/{ticker}",
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:  # 404 is acceptable for missing recommendations
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_ai_analysis(self):
        """Get AI analysis for a stock"""
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

        with self.client.get(
            f"/api/analysis/{ticker}/ai",
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)


class SearchTasks(TaskSet):
    """Search-related tasks"""

    @task(2)
    def search_stocks(self):
        """Search for stocks"""
        search_terms = ['tech', 'finance', 'healthcare', 'apple', 'microsoft']
        query = random.choice(search_terms)

        with self.client.get(
            f"/api/search/stocks",
            params={'q': query},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def autocomplete_search(self):
        """Test search autocomplete"""
        with self.client.get(
            "/api/search/autocomplete",
            params={'q': random.choice(['AAP', 'MSF', 'GOO', 'AMA', 'TSL'])},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)


class AnalyticsTasks(TaskSet):
    """Analytics-related tasks"""

    @task(1)
    def get_stock_metrics(self):
        """Get stock metrics"""
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

        with self.client.get(
            f"/api/stocks/{ticker}/metrics",
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_price_history(self):
        """Get price history"""
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        period = random.choice(['1M', '3M', '6M', '1Y'])

        with self.client.get(
            f"/api/stocks/{ticker}/prices",
            params={'period': period},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)

    @task(1)
    def get_correlation_analysis(self):
        """Get correlation analysis"""
        with self.client.get(
            "/api/analytics/correlation",
            params={'tickers': 'AAPL,MSFT,GOOGL'},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
                global_metrics.record_response(response.elapsed.total_seconds(), True)
            else:
                response.failure(f"Failed with status {response.status_code}")
                global_metrics.record_response(response.elapsed.total_seconds(), False)


class InvestmentAnalysisUser(HttpUser):
    """Represents a user performing investment analysis tasks"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    tasks = {
        DashboardTasks: 1,
        PortfolioTasks: 2,
        RecommendationTasks: 3,
        SearchTasks: 1,
        AnalyticsTasks: 1
    }

    def on_start(self):
        """Called when a user starts"""
        self.selected_stock = None
        logger.info(f"User {self.client_id} started")

    def on_stop(self):
        """Called when a user stops"""
        logger.info(f"User {self.client_id} stopped")


# Event handlers for metrics collection
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Handle request events"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Handle test completion"""
    summary = global_metrics.get_summary()

    logger.info("\n" + "="*80)
    logger.info("LOAD TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Duration: {summary['duration_seconds']:.1f}s")
    logger.info(f"Total Requests: {summary['total_requests']}")
    logger.info(f"Successful: {summary['successful_requests']}")
    logger.info(f"Failed: {summary['failed_requests']}")
    logger.info(f"Error Rate: {summary['error_rate']:.3f}")
    logger.info(f"Avg Response Time: {summary['avg_response_time']:.3f}s")
    logger.info(f"P50 Response Time: {summary['p50_response_time']:.3f}s")
    logger.info(f"P95 Response Time: {summary['p95_response_time']:.3f}s")
    logger.info(f"P99 Response Time: {summary['p99_response_time']:.3f}s")
    logger.info(f"Requests/Second: {summary['requests_per_second']:.2f}")
    logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']:.3f}")
    logger.info("="*80 + "\n")
