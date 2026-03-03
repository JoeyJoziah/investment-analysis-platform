"""
Valuation Helpers
Multiple intrinsic-value models: DCF, DDM, Residual Income, Asset-Based, EPV, and SOTP.
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-module helpers from ratio_analysis
# Try relative import first; fall back to loading the sibling file directly.
# ---------------------------------------------------------------------------
try:
    from .ratio_analysis import calculate_growth_rate
except ImportError:
    import importlib.util
    import pathlib
    import sys as _sys

    def _load_ratio_analysis():
        cache_key = "backend.analytics.ratio_analysis"
        if cache_key in _sys.modules:
            return _sys.modules[cache_key]
        this_dir = pathlib.Path(__file__).parent
        spec = importlib.util.spec_from_file_location(cache_key, this_dir / "ratio_analysis.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[cache_key] = mod
        spec.loader.exec_module(mod)
        return mod

    _ra = _load_ratio_analysis()
    calculate_growth_rate = _ra.calculate_growth_rate

# ---- constants used across models ----
_RISK_FREE_RATE = 0.045      # 10-year treasury proxy
_MARKET_RISK_PREMIUM = 0.08  # Historical equity risk premium


def calculate_wacc(
    financials: Dict,
    market_data: Dict,
    risk_free_rate: float = _RISK_FREE_RATE,
    market_risk_premium: float = _MARKET_RISK_PREMIUM,
) -> float:
    """Calculate Weighted Average Cost of Capital."""
    # Market values
    market_cap = market_data.get('market_cap', 0)
    debt_value = financials.get('total_debt', 0)
    total_value = market_cap + debt_value

    if total_value == 0:
        return 0.10  # Default 10%

    # Weights
    equity_weight = market_cap / total_value
    debt_weight = debt_value / total_value

    # Cost of equity (CAPM)
    beta = market_data.get('beta', 1.0)
    cost_of_equity = risk_free_rate + beta * market_risk_premium

    # Cost of debt
    interest_expense = financials.get('interest_expense', 0)
    cost_of_debt = interest_expense / debt_value if debt_value > 0 else 0.04

    # Tax rate
    tax_rate = financials.get('tax_rate', 0.21)

    # WACC
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))

    return wacc


def calculate_dcf(
    financials: Dict,
    market_data: Dict,
    risk_free_rate: float = _RISK_FREE_RATE,
    market_risk_premium: float = _MARKET_RISK_PREMIUM,
) -> Dict:
    """Discounted Cash Flow model."""
    # Get inputs
    fcf = financials.get('free_cash_flow', 0)
    growth_rate = financials.get('fcf_growth', 0.05)  # 5% default
    terminal_growth = 0.03  # 3% perpetual growth
    shares_outstanding = financials.get('shares_outstanding', 1)

    # Calculate WACC
    wacc = calculate_wacc(financials, market_data, risk_free_rate, market_risk_premium)

    # Project cash flows for 10 years
    projected_fcf = []
    for year in range(1, 11):
        if year <= 5:
            # Higher growth for first 5 years
            fcf_year = fcf * (1 + growth_rate) ** year
        else:
            # Decay to terminal growth
            decay_rate = (growth_rate - terminal_growth) * (10 - year) / 5
            fcf_year = fcf * (1 + growth_rate) ** 5 * (1 + terminal_growth + decay_rate) ** (year - 5)

        projected_fcf.append(fcf_year)

    # Calculate present value of projected cash flows
    pv_fcf = sum([cf / (1 + wacc) ** i for i, cf in enumerate(projected_fcf, 1)])

    # Terminal value
    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 10

    # Enterprise value
    enterprise_value = pv_fcf + pv_terminal

    # Equity value
    cash = financials.get('cash', 0)
    debt = financials.get('total_debt', 0)
    equity_value = enterprise_value + cash - debt

    # Value per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'value': value_per_share,
        'enterprise_value': enterprise_value,
        'wacc': wacc * 100,
        'terminal_growth': terminal_growth * 100,
        'confidence': 0.8,  # High confidence in DCF
    }


def calculate_ddm(
    financials: Dict,
    market_data: Dict,
    risk_free_rate: float = _RISK_FREE_RATE,
    market_risk_premium: float = _MARKET_RISK_PREMIUM,
) -> Dict:
    """Dividend Discount Model."""
    dividend_per_share = financials.get('dividend_per_share', 0)

    if dividend_per_share == 0:
        return {'value': 0, 'confidence': 0}

    # Get dividend growth rate
    dividend_history = financials.get('dividend_history', [])
    if len(dividend_history) >= 3:
        growth_rate = calculate_growth_rate(dividend_history)
    else:
        # Estimate based on earnings growth
        growth_rate = financials.get('earnings_growth', 0.03)

    # Required return (using CAPM)
    beta = market_data.get('beta', 1.0)
    required_return = risk_free_rate + beta * market_risk_premium

    # Gordon growth model
    if required_return > growth_rate:
        value = dividend_per_share * (1 + growth_rate) / (required_return - growth_rate)
    else:
        # Two-stage model if growth > required return
        high_growth_years = 5
        terminal_growth = 0.03

        # Stage 1: High growth
        pv_dividends_stage1 = sum([
            dividend_per_share * (1 + growth_rate) ** i / (1 + required_return) ** i
            for i in range(1, high_growth_years + 1)
        ])

        # Stage 2: Terminal value
        terminal_dividend = dividend_per_share * (1 + growth_rate) ** high_growth_years * (1 + terminal_growth)
        terminal_value = terminal_dividend / (required_return - terminal_growth)
        pv_terminal = terminal_value / (1 + required_return) ** high_growth_years

        value = pv_dividends_stage1 + pv_terminal

    return {
        'value': value,
        'dividend_yield': dividend_per_share / market_data.get('price', 1) * 100,
        'growth_rate': growth_rate * 100,
        'confidence': 0.7,
    }


def calculate_residual_income(
    financials: Dict,
    market_data: Dict,
    risk_free_rate: float = _RISK_FREE_RATE,
    market_risk_premium: float = _MARKET_RISK_PREMIUM,
) -> Dict:
    """Residual Income Model."""
    book_value = financials.get('book_value_per_share', 0)
    roe = financials.get('roe', 0)

    # Required return
    beta = market_data.get('beta', 1.0)
    required_return = risk_free_rate + beta * market_risk_premium

    # Project residual income
    years = 10
    terminal_growth = 0.03

    current_bv = book_value
    pv_residual_income = 0
    residual_income = 0

    for year in range(1, years + 1):
        # Expected earnings
        expected_earnings = current_bv * roe

        # Required earnings
        required_earnings = current_bv * required_return

        # Residual income
        residual_income = expected_earnings - required_earnings

        # Present value
        pv_residual_income += residual_income / (1 + required_return) ** year

        # Update book value
        retention_ratio = 1 - financials.get('payout_ratio', 0.3)
        current_bv = current_bv * (1 + roe * retention_ratio)

    # Terminal value
    terminal_ri = residual_income * (1 + terminal_growth)
    terminal_value = terminal_ri / (required_return - terminal_growth)
    pv_terminal = terminal_value / (1 + required_return) ** years

    # Total value
    value = book_value + pv_residual_income + pv_terminal

    return {
        'value': value,
        'book_value': book_value,
        'residual_income': pv_residual_income,
        'confidence': 0.6,
    }


def calculate_asset_based_value(financials: Dict) -> Dict:
    """Asset-based valuation."""
    # Net asset value
    total_assets = financials.get('total_assets', 0)
    total_liabilities = financials.get('total_liabilities', 0)
    shares_outstanding = financials.get('shares_outstanding', 1)

    net_assets = total_assets - total_liabilities
    nav_per_share = net_assets / shares_outstanding if shares_outstanding > 0 else 0

    # Adjusted for intangibles
    intangibles = financials.get('intangible_assets', 0)
    tangible_nav = (net_assets - intangibles) / shares_outstanding if shares_outstanding > 0 else 0

    # Liquidation value (conservative)
    current_assets = financials.get('current_assets', 0)
    ppe = financials.get('property_plant_equipment', 0)

    # Apply haircuts
    liquidation_value = (
        current_assets * 0.9 +  # 90% of current assets
        ppe * 0.5 -              # 50% of PP&E
        total_liabilities
    )

    liquidation_per_share = liquidation_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'value': nav_per_share,
        'tangible_nav': tangible_nav,
        'liquidation_value': liquidation_per_share,
        'confidence': 0.5,
    }


def calculate_epv(
    financials: Dict,
    market_data: Dict,
    risk_free_rate: float = _RISK_FREE_RATE,
    market_risk_premium: float = _MARKET_RISK_PREMIUM,
) -> Dict:
    """Earnings Power Value (Greenwald method)."""
    # Normalized earnings
    operating_income = financials.get('operating_income', 0)
    tax_rate = financials.get('tax_rate', 0.21)

    # Normalize for business cycle
    normalized_ebit = operating_income  # Could adjust for cycle

    # After-tax earnings
    normalized_earnings = normalized_ebit * (1 - tax_rate)

    # Cost of capital
    wacc = calculate_wacc(financials, market_data, risk_free_rate, market_risk_premium)

    # EPV = Normalized Earnings / WACC (no growth)
    epv = normalized_earnings / wacc if wacc > 0 else 0

    # Add excess cash, subtract debt
    cash = financials.get('cash', 0)
    debt = financials.get('total_debt', 0)
    equity_epv = epv + cash - debt

    shares_outstanding = financials.get('shares_outstanding', 1)
    epv_per_share = equity_epv / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'value': epv_per_share,
        'enterprise_epv': epv,
        'no_growth_assumption': True,
        'confidence': 0.7,
    }


def get_industry_multiple(industry: str) -> float:
    """Get typical EV/EBITDA multiple for industry."""
    industry_multiples = {
        'technology': 20,
        'software': 25,
        'healthcare': 15,
        'consumer': 12,
        'industrial': 10,
        'financial': 8,
        'utilities': 7,
        'energy': 6,
        'materials': 8,
        'realestate': 15,
        'general': 10,
    }
    return industry_multiples.get(industry.lower(), 10)


def calculate_sum_of_parts(financials: Dict) -> Dict:
    """Sum of the parts valuation for conglomerates."""
    segments = financials.get('segments', [])

    if not segments:
        return {'value': 0, 'confidence': 0}

    total_value = 0
    segment_values = {}

    for segment in segments:
        # Value each segment separately
        segment_ebitda = segment.get('ebitda', segment.get('operating_income', 0) * 1.2)

        # Apply industry multiples
        industry = segment.get('industry', 'general')
        ev_ebitda_multiple = get_industry_multiple(industry)

        segment_value = segment_ebitda * ev_ebitda_multiple
        segment_values[segment.get('name', 'unknown')] = segment_value
        total_value += segment_value

    # Add holding company discount/premium
    holding_adjustment = 0.9  # 10% conglomerate discount
    adjusted_value = total_value * holding_adjustment

    # Add cash, subtract debt
    cash = financials.get('cash', 0)
    debt = financials.get('total_debt', 0)
    equity_value = adjusted_value + cash - debt

    shares_outstanding = financials.get('shares_outstanding', 1)
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'value': value_per_share,
        'segment_values': segment_values,
        'total_enterprise_value': total_value,
        'confidence': 0.6,
    }


def run_valuation_models(
    financials: Dict,
    market_data: Dict,
    risk_free_rate: float = _RISK_FREE_RATE,
    market_risk_premium: float = _MARKET_RISK_PREMIUM,
) -> Dict:
    """
    Run all valuation models and aggregate results.

    Args:
        financials: Raw financial statement data.
        market_data: Market price and valuation data.
        risk_free_rate: Risk-free rate for CAPM / WACC.
        market_risk_premium: Equity risk premium for CAPM / WACC.
    """
    valuations = {}

    # 1. Discounted Cash Flow (DCF)
    valuations['dcf'] = calculate_dcf(financials, market_data, risk_free_rate, market_risk_premium)

    # 2. Dividend Discount Model (DDM)
    valuations['ddm'] = calculate_ddm(financials, market_data, risk_free_rate, market_risk_premium)

    # 3. Residual Income Model
    valuations['rim'] = calculate_residual_income(financials, market_data, risk_free_rate, market_risk_premium)

    # 4. Asset-Based Valuation
    valuations['asset_based'] = calculate_asset_based_value(financials)

    # 5. Earnings Power Value (EPV)
    valuations['epv'] = calculate_epv(financials, market_data, risk_free_rate, market_risk_premium)

    # 6. Sum of Parts Valuation
    valuations['sotp'] = calculate_sum_of_parts(financials)

    # Calculate average and range
    valid_values = [v['value'] for v in valuations.values() if v and v.get('value', 0) > 0]

    if valid_values:
        valuations['average'] = np.mean(valid_values)
        valuations['median'] = np.median(valid_values)
        valuations['range'] = {
            'min': min(valid_values),
            'max': max(valid_values),
        }

        current_price = market_data.get('price', 0)
        valuations['upside_potential'] = (
            (valuations['average'] - current_price) / current_price * 100
        ) if current_price > 0 else 0

    return valuations
