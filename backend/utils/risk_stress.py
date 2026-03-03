"""
Stress Testing Utilities

Provides historical and custom stress-test scenario analysis for portfolios.

Historical scenarios included:
- 2008 Financial Crisis
- COVID-19 March 2020 Crash
- Dot-Com Bubble Burst (2000)
- European Debt Crisis (2011)
- Black Monday 1987
- 2022 Rate Hike Cycle

Custom scenarios can be created with user-defined shocks.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class StressTestResult:
    """Result of a stress test."""
    scenario_name: str
    portfolio_loss: float
    asset_impacts: Dict[str, float]
    var_breach: bool
    description: str
    historical_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Historical scenario catalog
# ---------------------------------------------------------------------------

# Historical stress scenarios with typical asset class impacts
HISTORICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "2008_financial_crisis": {
        "name": "2008 Financial Crisis",
        "description": "Global financial crisis triggered by subprime mortgage collapse",
        "date": "2008-09-15",
        "equity_shock": -0.50,
        "bond_shock": 0.05,
        "commodity_shock": -0.35,
        "volatility_multiplier": 3.0,
        "correlation_shock": 0.3,  # Correlations increase in crisis
    },
    "2020_covid_crash": {
        "name": "COVID-19 March 2020 Crash",
        "description": "Rapid market decline due to pandemic fears",
        "date": "2020-03-16",
        "equity_shock": -0.34,
        "bond_shock": 0.02,
        "commodity_shock": -0.40,
        "volatility_multiplier": 4.0,
        "correlation_shock": 0.4,
    },
    "2000_dotcom_burst": {
        "name": "Dot-Com Bubble Burst",
        "description": "Technology sector collapse",
        "date": "2000-03-10",
        "equity_shock": -0.40,
        "tech_shock": -0.75,
        "bond_shock": 0.08,
        "commodity_shock": -0.15,
        "volatility_multiplier": 2.5,
        "correlation_shock": 0.2,
    },
    "2011_european_debt": {
        "name": "European Debt Crisis",
        "description": "Sovereign debt crisis in Europe",
        "date": "2011-08-05",
        "equity_shock": -0.20,
        "bond_shock": -0.05,
        "commodity_shock": -0.15,
        "volatility_multiplier": 2.0,
        "correlation_shock": 0.25,
    },
    "1987_black_monday": {
        "name": "Black Monday 1987",
        "description": "Largest single-day stock market crash",
        "date": "1987-10-19",
        "equity_shock": -0.22,
        "bond_shock": 0.03,
        "commodity_shock": -0.10,
        "volatility_multiplier": 5.0,
        "correlation_shock": 0.5,
    },
    "2022_rate_hike": {
        "name": "2022 Rate Hike Cycle",
        "description": "Aggressive Fed rate hikes to combat inflation",
        "date": "2022-06-13",
        "equity_shock": -0.25,
        "bond_shock": -0.15,
        "commodity_shock": -0.10,
        "volatility_multiplier": 2.0,
        "correlation_shock": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Stress test functions
# ---------------------------------------------------------------------------

def stress_test(
    portfolio: Dict[str, float],
    scenario: str,
    max_portfolio_var: float = 0.02,
    asset_betas: Optional[Dict[str, float]] = None,
    sector_mappings: Optional[Dict[str, str]] = None
) -> StressTestResult:
    """
    Apply a historical stress scenario to a portfolio.

    Args:
        portfolio: Dictionary mapping tickers to weights
        scenario: Scenario name (e.g., '2008_financial_crisis')
        max_portfolio_var: Maximum acceptable portfolio VaR for breach check
        asset_betas: Dictionary mapping tickers to beta values
        sector_mappings: Dictionary mapping tickers to sectors

    Returns:
        StressTestResult with scenario impact details
    """
    if scenario not in HISTORICAL_SCENARIOS:
        available = ", ".join(HISTORICAL_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {scenario}. Available: {available}")

    scenario_data = HISTORICAL_SCENARIOS[scenario]
    asset_betas = asset_betas or {}
    sector_mappings = sector_mappings or {}

    asset_impacts: Dict[str, float] = {}
    portfolio_loss = 0.0

    for ticker, weight in portfolio.items():
        beta = asset_betas.get(ticker, 1.0)
        sector = sector_mappings.get(ticker, 'equity')

        # Determine base shock based on sector
        if sector.lower() == 'tech' and 'tech_shock' in scenario_data:
            base_shock = scenario_data['tech_shock']
        elif sector.lower() == 'bond':
            base_shock = scenario_data['bond_shock']
        elif sector.lower() == 'commodity':
            base_shock = scenario_data['commodity_shock']
        else:
            base_shock = scenario_data['equity_shock']

        # Adjust shock by beta
        adjusted_shock = base_shock * beta

        asset_impacts[ticker] = adjusted_shock
        portfolio_loss += weight * adjusted_shock

    # Check if VaR is breached
    var_breach = abs(portfolio_loss) > max_portfolio_var

    return StressTestResult(
        scenario_name=scenario_data['name'],
        portfolio_loss=float(portfolio_loss),
        asset_impacts=asset_impacts,
        var_breach=var_breach,
        description=scenario_data['description'],
        historical_date=scenario_data['date']
    )


def stress_test_custom(
    portfolio: Dict[str, float],
    shocks: Dict[str, float],
    max_portfolio_var: float = 0.02,
    scenario_name: str = "Custom Scenario",
    description: str = "User-defined stress test"
) -> StressTestResult:
    """
    Apply custom shocks to a portfolio.

    Args:
        portfolio: Dictionary mapping tickers to weights
        shocks: Dictionary mapping tickers to shock values
        max_portfolio_var: Maximum acceptable portfolio VaR for breach check
        scenario_name: Name for the custom scenario
        description: Description of the scenario

    Returns:
        StressTestResult with custom scenario impact
    """
    asset_impacts: Dict[str, float] = {}
    portfolio_loss = 0.0

    for ticker, weight in portfolio.items():
        shock = shocks.get(ticker, 0.0)
        asset_impacts[ticker] = shock
        portfolio_loss += weight * shock

    var_breach = abs(portfolio_loss) > max_portfolio_var

    return StressTestResult(
        scenario_name=scenario_name,
        portfolio_loss=float(portfolio_loss),
        asset_impacts=asset_impacts,
        var_breach=var_breach,
        description=description,
        historical_date=None
    )


def stress_test_all_scenarios(
    portfolio: Dict[str, float],
    max_portfolio_var: float = 0.02,
    asset_betas: Optional[Dict[str, float]] = None,
    sector_mappings: Optional[Dict[str, str]] = None
) -> List[StressTestResult]:
    """
    Run all historical stress scenarios on a portfolio.

    Args:
        portfolio: Dictionary mapping tickers to weights
        max_portfolio_var: Maximum acceptable portfolio VaR for breach check
        asset_betas: Dictionary mapping tickers to beta values
        sector_mappings: Dictionary mapping tickers to sectors

    Returns:
        List of StressTestResult for all scenarios, sorted worst-first
    """
    results = []
    for scenario_name in HISTORICAL_SCENARIOS.keys():
        try:
            result = stress_test(
                portfolio, scenario_name, max_portfolio_var,
                asset_betas, sector_mappings
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error in stress test {scenario_name}: {e}")

    # Sort by portfolio loss (worst first)
    results.sort(key=lambda x: x.portfolio_loss)

    return results


def get_available_scenarios() -> List[Dict[str, str]]:
    """Get list of available stress test scenarios."""
    return [
        {
            'id': scenario_id,
            'name': data['name'],
            'description': data['description'],
            'date': data['date']
        }
        for scenario_id, data in HISTORICAL_SCENARIOS.items()
    ]


def create_custom_scenario(
    name: str,
    equity_shock: float,
    bond_shock: float = 0.0,
    commodity_shock: float = 0.0,
    tech_shock: Optional[float] = None,
    volatility_multiplier: float = 2.0,
    correlation_shock: float = 0.2,
    description: str = ""
) -> Dict[str, Any]:
    """
    Create a custom stress scenario dictionary.

    Args:
        name: Scenario name
        equity_shock: Shock to equity prices (e.g., -0.20 for 20% decline)
        bond_shock: Shock to bond prices
        commodity_shock: Shock to commodity prices
        tech_shock: Optional separate shock for tech stocks
        volatility_multiplier: How much volatility increases
        correlation_shock: How much correlations increase
        description: Scenario description

    Returns:
        Scenario dictionary compatible with stress_test()
    """
    scenario: Dict[str, Any] = {
        'name': name,
        'description': description or f"Custom scenario: {name}",
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'equity_shock': equity_shock,
        'bond_shock': bond_shock,
        'commodity_shock': commodity_shock,
        'volatility_multiplier': volatility_multiplier,
        'correlation_shock': correlation_shock,
    }

    if tech_shock is not None:
        scenario['tech_shock'] = tech_shock

    return scenario
