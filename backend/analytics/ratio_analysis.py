"""
Financial Ratio Analysis
Calculates all core financial metrics and ratios from raw financial statement data.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """Container for financial metrics"""
    # Profitability
    gross_margin: float
    operating_margin: float
    net_margin: float
    roe: float  # Return on Equity
    roa: float  # Return on Assets
    roic: float  # Return on Invested Capital

    # Growth
    revenue_growth: float
    earnings_growth: float
    fcf_growth: float

    # Valuation
    pe_ratio: float
    peg_ratio: float
    price_to_book: float
    price_to_sales: float
    ev_to_ebitda: float
    fcf_yield: float

    # Financial Health
    current_ratio: float
    quick_ratio: float
    debt_to_equity: float
    interest_coverage: float

    # Efficiency
    asset_turnover: float
    inventory_turnover: float
    receivables_turnover: float


def calculate_growth_rate(values: List[float]) -> float:
    """Calculate CAGR from a list of values."""
    if len(values) < 2 or values[0] <= 0:
        return 0
    years = len(values) - 1
    return (pow(values[-1] / values[0], 1 / years) - 1) * 100


def calculate_cagr(values: List[float]) -> float:
    """Calculate Compound Annual Growth Rate."""
    return calculate_growth_rate(values)


def get_quality_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return 'A+'
    elif score >= 85:
        return 'A'
    elif score >= 80:
        return 'A-'
    elif score >= 75:
        return 'B+'
    elif score >= 70:
        return 'B'
    elif score >= 65:
        return 'B-'
    elif score >= 60:
        return 'C+'
    elif score >= 55:
        return 'C'
    elif score >= 50:
        return 'C-'
    else:
        return 'D'


def calculate_margin_stability(financials: Dict) -> float:
    """Calculate margin stability score."""
    margin_history = financials.get('operating_margin_history', [])
    if len(margin_history) < 3:
        return 0.5

    # Calculate coefficient of variation
    mean_margin = np.mean(margin_history)
    std_margin = np.std(margin_history)

    if mean_margin > 0:
        cv = std_margin / mean_margin
        # Lower CV = more stable
        stability = max(0, 1 - cv)
        return stability

    return 0.5


def evaluate_buyback_effectiveness(financials: Dict) -> float:
    """Evaluate share buyback effectiveness."""
    buyback_price = financials.get('avg_buyback_price', 0)
    current_price = financials.get('current_price', 0)

    if buyback_price > 0 and current_price > 0:
        # Good if bought below current price
        effectiveness = min(1.0, current_price / buyback_price - 0.5)
        return max(0, effectiveness)

    return 0.5


def calculate_financial_metrics(financials: Dict, market_data: Dict) -> FinancialMetrics:
    """
    Calculate all financial metrics from raw data.

    Args:
        financials: Raw financial statement data.
        market_data: Market price and valuation data.
    """
    # Extract key values
    revenue = financials.get('revenue', 0)
    gross_profit = financials.get('gross_profit', 0)
    operating_income = financials.get('operating_income', 0)
    net_income = financials.get('net_income', 0)
    total_assets = financials.get('total_assets', 0)
    total_equity = financials.get('total_equity', 0)
    total_debt = financials.get('total_debt', 0)
    current_assets = financials.get('current_assets', 0)
    current_liabilities = financials.get('current_liabilities', 0)
    cash = financials.get('cash', 0)
    inventory = financials.get('inventory', 0)
    receivables = financials.get('receivables', 0)
    free_cash_flow = financials.get('free_cash_flow', 0)
    shares_outstanding = financials.get('shares_outstanding', 1)

    # Market data
    market_cap = market_data.get('market_cap', 0)
    enterprise_value = market_data.get('enterprise_value', market_cap + total_debt - cash)
    stock_price = market_data.get('price', 0)

    # Calculate margins
    gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
    operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
    net_margin = (net_income / revenue * 100) if revenue > 0 else 0

    # Calculate returns
    roe = (net_income / total_equity * 100) if total_equity > 0 else 0
    roa = (net_income / total_assets * 100) if total_assets > 0 else 0

    # ROIC = NOPAT / Invested Capital
    tax_rate = financials.get('tax_rate', 0.21)  # Default corporate tax rate
    nopat = operating_income * (1 - tax_rate)
    invested_capital = total_equity + total_debt - cash
    roic = (nopat / invested_capital * 100) if invested_capital > 0 else 0

    # Growth rates (need historical data)
    revenue_growth = calculate_growth_rate(
        financials.get('revenue_history', [])
    )
    earnings_growth = calculate_growth_rate(
        financials.get('earnings_history', [])
    )
    fcf_growth = calculate_growth_rate(
        financials.get('fcf_history', [])
    )

    # Valuation ratios
    eps = net_income / shares_outstanding if shares_outstanding > 0 else 0
    pe_ratio = stock_price / eps if eps > 0 else 0
    peg_ratio = pe_ratio / earnings_growth if earnings_growth > 0 else 0

    book_value = total_equity / shares_outstanding if shares_outstanding > 0 else 0
    price_to_book = stock_price / book_value if book_value > 0 else 0

    price_to_sales = market_cap / revenue if revenue > 0 else 0

    ebitda = financials.get('ebitda', operating_income * 1.2)  # Rough estimate if not provided
    ev_to_ebitda = enterprise_value / ebitda if ebitda > 0 else 0

    fcf_yield = (free_cash_flow / market_cap * 100) if market_cap > 0 else 0

    # Financial health ratios
    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
    quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities > 0 else 0
    debt_to_equity = total_debt / total_equity if total_equity > 0 else 0

    interest_expense = financials.get('interest_expense', 0)
    interest_coverage = operating_income / interest_expense if interest_expense > 0 else 999

    # Efficiency ratios
    asset_turnover = revenue / total_assets if total_assets > 0 else 0
    inventory_turnover = financials.get('cogs', revenue * 0.7) / inventory if inventory > 0 else 0
    receivables_turnover = revenue / receivables if receivables > 0 else 0

    return FinancialMetrics(
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        roe=roe,
        roa=roa,
        roic=roic,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        fcf_growth=fcf_growth,
        pe_ratio=pe_ratio,
        peg_ratio=peg_ratio,
        price_to_book=price_to_book,
        price_to_sales=price_to_sales,
        ev_to_ebitda=ev_to_ebitda,
        fcf_yield=fcf_yield,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        debt_to_equity=debt_to_equity,
        interest_coverage=interest_coverage,
        asset_turnover=asset_turnover,
        inventory_turnover=inventory_turnover,
        receivables_turnover=receivables_turnover,
    )
