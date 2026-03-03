"""
Quality Scoring and Financial Health
Piotroski F-Score, Altman Z-Score, Beneish M-Score, and quality scoring.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-module helpers from ratio_analysis
# Try relative import first; fall back to loading the sibling file directly.
# ---------------------------------------------------------------------------
try:
    from .ratio_analysis import (
        calculate_margin_stability,
        evaluate_buyback_effectiveness,
        get_quality_grade,
    )
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
    calculate_margin_stability = _ra.calculate_margin_stability
    evaluate_buyback_effectiveness = _ra.evaluate_buyback_effectiveness
    get_quality_grade = _ra.get_quality_grade


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def score_profitability(financials: Dict) -> float:
    """Score profitability quality (0-100)."""
    score = 0

    # Positive net income (20 points)
    if financials.get('net_income', 0) > 0:
        score += 20

    # Positive operating cash flow (20 points)
    if financials.get('operating_cash_flow', 0) > 0:
        score += 20

    # Cash flow > Net income (quality earnings) (20 points)
    if financials.get('operating_cash_flow', 0) > financials.get('net_income', 0):
        score += 20

    # ROA improvement (20 points)
    roa_current = financials.get('roa', 0)
    roa_previous = financials.get('roa_previous', 0)
    if roa_current > roa_previous:
        score += 20

    # High ROIC (20 points)
    roic = financials.get('roic', 0)
    if roic > 15:
        score += 20
    elif roic > 10:
        score += 10

    return score


def score_balance_sheet(financials: Dict) -> float:
    """Score balance sheet quality (0-100)."""
    score = 0

    # Low leverage (25 points)
    debt_to_equity = financials.get('debt_to_equity', 0)
    if debt_to_equity < 0.5:
        score += 25
    elif debt_to_equity < 1.0:
        score += 15

    # Good liquidity (25 points)
    current_ratio = financials.get('current_ratio', 0)
    if current_ratio > 2.0:
        score += 25
    elif current_ratio > 1.5:
        score += 15

    # Interest coverage (25 points)
    interest_coverage = financials.get('interest_coverage', 0)
    if interest_coverage > 5:
        score += 25
    elif interest_coverage > 3:
        score += 15

    # Asset quality (25 points)
    # Low intangibles relative to total assets
    intangibles_ratio = financials.get('intangibles_to_assets', 0)
    if intangibles_ratio < 0.2:
        score += 25
    elif intangibles_ratio < 0.4:
        score += 15

    return score


def score_earnings_quality(financials: Dict) -> float:
    """Score earnings quality (0-100)."""
    score = 0

    # Low accruals (30 points)
    total_accruals = (
        financials.get('net_income', 0) -
        financials.get('operating_cash_flow', 0)
    ) / financials.get('total_assets', 1)

    if abs(total_accruals) < 0.05:
        score += 30
    elif abs(total_accruals) < 0.10:
        score += 15

    # Consistent margins (25 points)
    margin_stability = calculate_margin_stability(financials)
    if margin_stability > 0.9:
        score += 25
    elif margin_stability > 0.8:
        score += 15

    # Revenue recognition quality (25 points)
    # Days sales outstanding trend
    dso_trend = financials.get('dso_trend', 0)
    if dso_trend <= 0:  # Stable or improving
        score += 25
    elif dso_trend < 0.1:  # Slight increase
        score += 15

    # Low one-time items (20 points)
    exceptional_items_ratio = abs(
        financials.get('exceptional_items', 0) /
        financials.get('operating_income', 1)
    )
    if exceptional_items_ratio < 0.05:
        score += 20
    elif exceptional_items_ratio < 0.10:
        score += 10

    return score


def score_growth_quality(financials: Dict) -> float:
    """Score growth quality (0-100)."""
    score = 0

    # Sustainable revenue growth (30 points)
    revenue_growth = financials.get('revenue_growth', 0)
    if 5 <= revenue_growth <= 20:  # Sustainable range
        score += 30
    elif 0 < revenue_growth < 5:
        score += 20
    elif revenue_growth > 20:  # Might be too high
        score += 15

    # Margin expansion (25 points)
    margin_trend = financials.get('operating_margin_trend', 0)
    if margin_trend > 0:
        score += 25
    elif margin_trend == 0:
        score += 15

    # Market share gains (25 points)
    market_share_change = financials.get('market_share_change', 0)
    if market_share_change > 0:
        score += 25
    elif market_share_change == 0:
        score += 15

    # R&D efficiency (20 points)
    rd_to_revenue = financials.get('rd_to_revenue', 0)
    if 0.05 <= rd_to_revenue <= 0.15:  # Healthy R&D spending
        score += 20
    elif 0 < rd_to_revenue < 0.05:
        score += 10

    return score


def score_capital_allocation(financials: Dict) -> float:
    """Score capital allocation quality (0-100)."""
    score = 0

    # ROIC vs WACC (30 points)
    roic = financials.get('roic', 0)
    wacc = financials.get('wacc', 10)
    if roic > wacc * 1.5:
        score += 30
    elif roic > wacc:
        score += 20

    # Dividend policy (25 points)
    payout_ratio = financials.get('payout_ratio', 0)
    if 0.2 <= payout_ratio <= 0.6:  # Balanced payout
        score += 25
    elif 0 < payout_ratio < 0.2:
        score += 15

    # Share buybacks at good prices (25 points)
    buyback_effectiveness = evaluate_buyback_effectiveness(financials)
    score += buyback_effectiveness * 25

    # Acquisition track record (20 points)
    acquisition_returns = financials.get('acquisition_roic', 0)
    if acquisition_returns > 15:
        score += 20
    elif acquisition_returns > 10:
        score += 10

    return score


def calculate_quality_score(financials: Dict) -> Dict:
    """
    Calculate quality score based on multiple factors.

    Returns a dict with 'overall_score', 'scores', and 'grade'.
    """
    scores = {}

    # 1. Profitability Quality (Piotroski F-Score components)
    scores['profitability'] = score_profitability(financials)

    # 2. Balance Sheet Quality
    scores['balance_sheet'] = score_balance_sheet(financials)

    # 3. Earnings Quality
    scores['earnings_quality'] = score_earnings_quality(financials)

    # 4. Growth Quality
    scores['growth_quality'] = score_growth_quality(financials)

    # 5. Capital Allocation
    scores['capital_allocation'] = score_capital_allocation(financials)

    # Overall quality score (0-100)
    overall_score = np.mean(list(scores.values()))

    return {
        'overall_score': overall_score,
        'scores': scores,
        'grade': get_quality_grade(overall_score),
    }


# ---------------------------------------------------------------------------
# Financial health scores (Altman, Piotroski, Beneish)
# ---------------------------------------------------------------------------

def calculate_altman_z_score(financials: Dict) -> Dict:
    """Calculate Altman Z-Score for bankruptcy prediction."""
    # Get required values
    working_capital = financials.get('current_assets', 0) - financials.get('current_liabilities', 0)
    total_assets = financials.get('total_assets', 1)
    retained_earnings = financials.get('retained_earnings', 0)
    ebit = financials.get('ebit', financials.get('operating_income', 0))
    market_cap = financials.get('market_cap', 0)
    total_liabilities = financials.get('total_liabilities', 0)
    revenue = financials.get('revenue', 0)

    # Calculate ratios
    x1 = (working_capital / total_assets) if total_assets > 0 else 0
    x2 = (retained_earnings / total_assets) if total_assets > 0 else 0
    x3 = (ebit / total_assets) if total_assets > 0 else 0
    x4 = (market_cap / total_liabilities) if total_liabilities > 0 else 0
    x5 = (revenue / total_assets) if total_assets > 0 else 0

    # Calculate Z-Score (for public companies)
    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # Interpret score
    if z_score > 2.99:
        zone = 'safe'
        risk = 'low'
    elif z_score > 1.81:
        zone = 'grey'
        risk = 'medium'
    else:
        zone = 'distress'
        risk = 'high'

    return {
        'score': z_score,
        'zone': zone,
        'bankruptcy_risk': risk,
        'components': {
            'working_capital_ratio': x1,
            'retained_earnings_ratio': x2,
            'ebit_ratio': x3,
            'market_to_book': x4,
            'sales_to_assets': x5,
        },
    }


def calculate_piotroski_score(financials: Dict) -> Dict:
    """Calculate Piotroski F-Score (0-9)."""
    score = 0
    criteria = {}

    # Profitability (4 points)
    # 1. Positive net income
    if financials.get('net_income', 0) > 0:
        score += 1
        criteria['positive_net_income'] = True

    # 2. Positive operating cash flow
    if financials.get('operating_cash_flow', 0) > 0:
        score += 1
        criteria['positive_ocf'] = True

    # 3. Growing ROA
    if financials.get('roa', 0) > financials.get('roa_previous', 0):
        score += 1
        criteria['improving_roa'] = True

    # 4. Quality of earnings (OCF > NI)
    if financials.get('operating_cash_flow', 0) > financials.get('net_income', 0):
        score += 1
        criteria['quality_earnings'] = True

    # Leverage/Liquidity (3 points)
    # 5. Decreasing leverage
    if financials.get('debt_to_assets', 1) < financials.get('debt_to_assets_previous', 1):
        score += 1
        criteria['decreasing_leverage'] = True

    # 6. Improving current ratio
    if financials.get('current_ratio', 0) > financials.get('current_ratio_previous', 0):
        score += 1
        criteria['improving_liquidity'] = True

    # 7. No new equity issuance
    if financials.get('shares_outstanding', 0) <= financials.get('shares_outstanding_previous', 0):
        score += 1
        criteria['no_dilution'] = True

    # Operating Efficiency (2 points)
    # 8. Improving gross margin
    if financials.get('gross_margin', 0) > financials.get('gross_margin_previous', 0):
        score += 1
        criteria['improving_gross_margin'] = True

    # 9. Improving asset turnover
    if financials.get('asset_turnover', 0) > financials.get('asset_turnover_previous', 0):
        score += 1
        criteria['improving_efficiency'] = True

    return {
        'score': score,
        'criteria': criteria,
        'strength': 'strong' if score >= 7 else 'moderate' if score >= 4 else 'weak',
    }


def _calculate_asset_quality_index(financials: Dict) -> float:
    """Calculate asset quality index for M-Score."""
    non_current_assets = financials.get('total_assets', 0) - financials.get('current_assets', 0)
    ppe = financials.get('property_plant_equipment', 0)

    if non_current_assets > 0:
        aqi_current = 1 - (ppe / non_current_assets)
    else:
        aqi_current = 0

    non_current_assets_prev = (
        financials.get('total_assets_previous', 0) -
        financials.get('current_assets_previous', 0)
    )
    ppe_prev = financials.get('property_plant_equipment_previous', 0)

    if non_current_assets_prev > 0:
        aqi_previous = 1 - (ppe_prev / non_current_assets_prev)
    else:
        aqi_previous = 0

    return aqi_current / aqi_previous if aqi_previous > 0 else 1


def _calculate_depreciation_index(financials: Dict) -> float:
    """Calculate depreciation index for M-Score."""
    dep_rate = financials.get('depreciation', 0) / financials.get('property_plant_equipment', 1)
    dep_rate_prev = (
        financials.get('depreciation_previous', 0) /
        financials.get('property_plant_equipment_previous', 1)
    )

    return dep_rate_prev / dep_rate if dep_rate > 0 else 1


def _calculate_sga_index(financials: Dict) -> float:
    """Calculate SG&A index for M-Score."""
    sga_rate = financials.get('sga_expenses', 0) / financials.get('revenue', 1)
    sga_rate_prev = financials.get('sga_expenses_previous', 0) / financials.get('revenue_previous', 1)

    return sga_rate / sga_rate_prev if sga_rate_prev > 0 else 1


def _calculate_total_accruals(financials: Dict) -> float:
    """Calculate total accruals to total assets."""
    total_accruals = (
        financials.get('net_income', 0) -
        financials.get('operating_cash_flow', 0)
    )
    total_assets = financials.get('total_assets', 1)

    return total_accruals / total_assets


def calculate_beneish_m_score(financials: Dict) -> Dict:
    """Calculate Beneish M-Score for earnings manipulation detection."""
    # Calculate 8 variables
    # 1. Days Sales in Receivables Index
    dsr_current = (financials.get('receivables', 0) / financials.get('revenue', 1)) * 365
    dsr_previous = (financials.get('receivables_previous', 0) / financials.get('revenue_previous', 1)) * 365
    dsri = dsr_current / dsr_previous if dsr_previous > 0 else 1

    # 2. Gross Margin Index
    gm_previous = financials.get('gross_margin_previous', 1)
    gm_current = financials.get('gross_margin', 1)
    gmi = gm_previous / gm_current if gm_current > 0 else 1

    # 3. Asset Quality Index
    aqi = _calculate_asset_quality_index(financials)

    # 4. Sales Growth Index
    sgi = financials.get('revenue', 1) / financials.get('revenue_previous', 1)

    # 5. Depreciation Index
    depi = _calculate_depreciation_index(financials)

    # 6. SG&A Index
    sgai = _calculate_sga_index(financials)

    # 7. Leverage Index
    lvgi = financials.get('debt_to_assets', 0) / financials.get('debt_to_assets_previous', 1)

    # 8. Total Accruals to Total Assets
    tata = _calculate_total_accruals(financials)

    # Calculate M-Score
    m_score = (
        -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi +
        0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    )

    # Interpret score
    if m_score > -2.22:
        likelihood = 'high'
        risk = 'Likely earnings manipulator'
    else:
        likelihood = 'low'
        risk = 'Unlikely earnings manipulator'

    return {
        'score': m_score,
        'likelihood': likelihood,
        'interpretation': risk,
        'components': {
            'dsri': dsri,
            'gmi': gmi,
            'aqi': aqi,
            'sgi': sgi,
            'depi': depi,
            'sgai': sgai,
            'lvgi': lvgi,
            'tata': tata,
        },
    }


# ---------------------------------------------------------------------------
# Liquidity / solvency / cash-flow grading helpers
# ---------------------------------------------------------------------------

def grade_liquidity(financials: Dict) -> str:
    """Grade liquidity position."""
    current_ratio = financials.get('current_ratio', 0)
    quick_ratio = financials.get('quick_ratio', 0)

    if current_ratio > 2 and quick_ratio > 1.5:
        return 'A'
    elif current_ratio > 1.5 and quick_ratio > 1:
        return 'B'
    elif current_ratio > 1 and quick_ratio > 0.7:
        return 'C'
    elif current_ratio > 0.7:
        return 'D'
    else:
        return 'F'


def grade_solvency(financials: Dict) -> str:
    """Grade solvency position."""
    debt_to_equity = financials.get('debt_to_equity', 999)
    interest_coverage = financials.get('interest_coverage', 0)

    if debt_to_equity < 0.5 and interest_coverage > 5:
        return 'A'
    elif debt_to_equity < 1 and interest_coverage > 3:
        return 'B'
    elif debt_to_equity < 2 and interest_coverage > 2:
        return 'C'
    elif interest_coverage > 1:
        return 'D'
    else:
        return 'F'


def grade_cash_flows(financials: Dict) -> str:
    """Grade cash flow quality."""
    ocf = financials.get('operating_cash_flow', 0)
    fcf = financials.get('free_cash_flow', 0)
    net_income = financials.get('net_income', 0)

    if ocf > 0 and fcf > 0 and ocf > net_income:
        return 'A'
    elif ocf > 0 and fcf > 0:
        return 'B'
    elif ocf > 0:
        return 'C'
    elif ocf > -net_income * 0.5:
        return 'D'
    else:
        return 'F'


def analyze_liquidity(financials: Dict) -> Dict:
    """Analyze liquidity position."""
    return {
        'current_ratio': financials.get('current_ratio', 0),
        'quick_ratio': financials.get('quick_ratio', 0),
        'cash_ratio': financials.get('cash_ratio', 0),
        'working_capital': financials.get('working_capital', 0),
        'cash_conversion_cycle': financials.get('cash_conversion_cycle', 0),
        'liquidity_grade': grade_liquidity(financials),
    }


def analyze_solvency(financials: Dict) -> Dict:
    """Analyze solvency position."""
    return {
        'debt_to_equity': financials.get('debt_to_equity', 0),
        'debt_to_assets': financials.get('debt_to_assets', 0),
        'interest_coverage': financials.get('interest_coverage', 0),
        'debt_service_coverage': financials.get('debt_service_coverage', 0),
        'solvency_grade': grade_solvency(financials),
    }


def analyze_cash_flows(financials: Dict) -> Dict:
    """Analyze cash flow patterns."""
    return {
        'operating_cash_flow': financials.get('operating_cash_flow', 0),
        'free_cash_flow': financials.get('free_cash_flow', 0),
        'fcf_conversion': financials.get('fcf_to_net_income', 0),
        'capex_to_revenue': financials.get('capex_to_revenue', 0),
        'cash_flow_grade': grade_cash_flows(financials),
    }


def calculate_overall_health_score(health_metrics: Dict) -> float:
    """Calculate overall financial health score."""
    scores = []

    # Altman Z-Score contribution
    z_score = health_metrics.get('altman_z_score', {}).get('score', 0)
    if z_score > 3:
        scores.append(100)
    elif z_score > 1.8:
        scores.append(60)
    else:
        scores.append(20)

    # Piotroski score contribution
    f_score = health_metrics.get('piotroski_f_score', {}).get('score', 0)
    scores.append(f_score / 9 * 100)

    # Liquidity score
    liquidity_grade = health_metrics.get('liquidity_analysis', {}).get('liquidity_grade', 'C')
    grade_scores = {'A': 90, 'B': 70, 'C': 50, 'D': 30, 'F': 10}
    scores.append(grade_scores.get(liquidity_grade, 50))

    return np.mean(scores)


def assess_financial_health(financials: Dict) -> Dict:
    """
    Assess financial health and solvency.

    Returns a dict with altman z-score, piotroski f-score, beneish m-score,
    liquidity/solvency/cash-flow analyses, and an overall health score.
    """
    health_metrics = {
        'altman_z_score': calculate_altman_z_score(financials),
        'piotroski_f_score': calculate_piotroski_score(financials),
        'beneish_m_score': calculate_beneish_m_score(financials),
        'liquidity_analysis': analyze_liquidity(financials),
        'solvency_analysis': analyze_solvency(financials),
        'cash_flow_analysis': analyze_cash_flows(financials),
    }

    # Overall health score
    health_metrics['overall_health'] = calculate_overall_health_score(health_metrics)

    return health_metrics
