"""
Advanced Fundamental Analysis Engine
Comprehensive financial analysis including DCF, peer comparison, and quality metrics.

This module is the public surface of the fundamental-analysis subsystem.  All
existing import paths remain valid:

    from backend.analytics.fundamental_analysis import (
        FinancialMetrics,
        FundamentalAnalysisEngine,
    )

Implementation is split into three focused sub-modules:

    ratio_analysis.py      – FinancialMetrics dataclass + ratio helpers
    valuation_helpers.py   – DCF, DDM, EPV, SOTP, asset-based, residual income
    quality_scoring.py     – Piotroski, Altman, Beneish, quality & health scores
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
from scipy import stats

# ---------------------------------------------------------------------------
# Sub-module imports
#
# Try relative imports first (normal package usage).  Fall back to absolute
# imports so that this file can also be loaded via importlib.util
# spec_from_file_location without a parent package – a pattern used by the
# unit test suite (test_analytics_extended_agent4.py).
# ---------------------------------------------------------------------------
def _load_sibling(module_name: str, filename: str):
    """Load a sibling module from the same directory, caching in sys.modules."""
    import sys
    import importlib.util
    import pathlib

    cache_key = f"backend.analytics.{module_name}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]

    this_dir = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(cache_key, this_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec to handle any intra-module circularity
    sys.modules[cache_key] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from .ratio_analysis import (
        FinancialMetrics,
        calculate_financial_metrics,
        calculate_growth_rate,
        calculate_cagr,
        get_quality_grade,
        calculate_margin_stability,
        evaluate_buyback_effectiveness,
    )
    from .valuation_helpers import (
        run_valuation_models,
        calculate_dcf,
        calculate_wacc,
        calculate_ddm,
        calculate_residual_income,
        calculate_asset_based_value,
        calculate_epv,
        calculate_sum_of_parts,
        get_industry_multiple,
    )
    from .quality_scoring import (
        calculate_quality_score,
        assess_financial_health,
        calculate_altman_z_score,
        calculate_piotroski_score,
        calculate_beneish_m_score,
        score_profitability,
        score_balance_sheet,
        score_earnings_quality,
        score_growth_quality,
        score_capital_allocation,
        analyze_liquidity,
        analyze_solvency,
        analyze_cash_flows,
        calculate_overall_health_score,
        grade_liquidity,
        grade_solvency,
        grade_cash_flows,
    )
except ImportError:
    # Fallback for when this file is loaded directly via importlib
    # spec_from_file_location without a parent package (e.g. unit test suite).
    _ra = _load_sibling("ratio_analysis", "ratio_analysis.py")
    _vh = _load_sibling("valuation_helpers", "valuation_helpers.py")
    _qs = _load_sibling("quality_scoring", "quality_scoring.py")

    FinancialMetrics = _ra.FinancialMetrics
    calculate_financial_metrics = _ra.calculate_financial_metrics
    calculate_growth_rate = _ra.calculate_growth_rate
    calculate_cagr = _ra.calculate_cagr
    get_quality_grade = _ra.get_quality_grade
    calculate_margin_stability = _ra.calculate_margin_stability
    evaluate_buyback_effectiveness = _ra.evaluate_buyback_effectiveness

    run_valuation_models = _vh.run_valuation_models
    calculate_dcf = _vh.calculate_dcf
    calculate_wacc = _vh.calculate_wacc
    calculate_ddm = _vh.calculate_ddm
    calculate_residual_income = _vh.calculate_residual_income
    calculate_asset_based_value = _vh.calculate_asset_based_value
    calculate_epv = _vh.calculate_epv
    calculate_sum_of_parts = _vh.calculate_sum_of_parts
    get_industry_multiple = _vh.get_industry_multiple

    calculate_quality_score = _qs.calculate_quality_score
    assess_financial_health = _qs.assess_financial_health
    calculate_altman_z_score = _qs.calculate_altman_z_score
    calculate_piotroski_score = _qs.calculate_piotroski_score
    calculate_beneish_m_score = _qs.calculate_beneish_m_score
    score_profitability = _qs.score_profitability
    score_balance_sheet = _qs.score_balance_sheet
    score_earnings_quality = _qs.score_earnings_quality
    score_growth_quality = _qs.score_growth_quality
    score_capital_allocation = _qs.score_capital_allocation
    analyze_liquidity = _qs.analyze_liquidity
    analyze_solvency = _qs.analyze_solvency
    analyze_cash_flows = _qs.analyze_cash_flows
    calculate_overall_health_score = _qs.calculate_overall_health_score
    grade_liquidity = _qs.grade_liquidity
    grade_solvency = _qs.grade_solvency
    grade_cash_flows = _qs.grade_cash_flows

logger = logging.getLogger(__name__)


class FundamentalAnalysisEngine:
    """
    Comprehensive fundamental analysis using SEC data and financial APIs.

    Orchestrates the ratio, valuation, and quality sub-modules and exposes a
    single ``analyze_company`` entry point for consumers.
    """

    def __init__(self):
        self.sector_averages = {}  # Cache sector averages
        self.risk_free_rate = 0.045  # Current 10-year treasury
        self.market_risk_premium = 0.08  # Historical equity risk premium

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_company(
        self,
        ticker: str,
        financials: Dict,
        market_data: Dict,
        peer_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.
        """
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'financial_metrics': self._calculate_financial_metrics(financials, market_data),
            'valuation_models': self._run_valuation_models(financials, market_data),
            'quality_score': self._calculate_quality_score(financials),
            'growth_analysis': self._analyze_growth(financials),
            'financial_health': self._assess_financial_health(financials),
            'efficiency_metrics': self._calculate_efficiency_metrics(financials),
            'peer_comparison': self._compare_with_peers(financials, peer_data) if peer_data else None,
            'moat_analysis': self._analyze_moat(financials, market_data),
            'management_quality': self._assess_management_quality(financials),
            'composite_score': 0.0,
            'risks': [],
            'opportunities': [],
        }

        # Calculate composite fundamental score
        analysis['composite_score'] = self._calculate_composite_score(analysis)

        # Identify risks and opportunities
        analysis['risks'] = self._identify_risks(analysis)
        analysis['opportunities'] = self._identify_opportunities(analysis)

        return analysis

    # ------------------------------------------------------------------
    # Delegating wrappers (preserve the private-method API that tests call)
    # ------------------------------------------------------------------

    def _calculate_financial_metrics(
        self, financials: Dict, market_data: Dict
    ) -> FinancialMetrics:
        """Calculate all financial metrics."""
        return calculate_financial_metrics(financials, market_data)

    def _run_valuation_models(
        self, financials: Dict, market_data: Dict
    ) -> Dict[str, Any]:
        """Run multiple valuation models."""
        return run_valuation_models(
            financials,
            market_data,
            self.risk_free_rate,
            self.market_risk_premium,
        )

    def _calculate_dcf(self, financials: Dict, market_data: Dict) -> Dict:
        """Discounted Cash Flow model."""
        return calculate_dcf(
            financials, market_data, self.risk_free_rate, self.market_risk_premium
        )

    def _calculate_wacc(self, financials: Dict, market_data: Dict) -> float:
        """Calculate Weighted Average Cost of Capital."""
        return calculate_wacc(
            financials, market_data, self.risk_free_rate, self.market_risk_premium
        )

    def _calculate_ddm(self, financials: Dict, market_data: Dict) -> Dict:
        """Dividend Discount Model."""
        return calculate_ddm(
            financials, market_data, self.risk_free_rate, self.market_risk_premium
        )

    def _calculate_residual_income(self, financials: Dict, market_data: Dict) -> Dict:
        """Residual Income Model."""
        return calculate_residual_income(
            financials, market_data, self.risk_free_rate, self.market_risk_premium
        )

    def _calculate_asset_based_value(self, financials: Dict) -> Dict:
        """Asset-based valuation."""
        return calculate_asset_based_value(financials)

    def _calculate_epv(self, financials: Dict, market_data: Dict) -> Dict:
        """Earnings Power Value (Greenwald method)."""
        return calculate_epv(
            financials, market_data, self.risk_free_rate, self.market_risk_premium
        )

    def _calculate_sum_of_parts(self, financials: Dict) -> Dict:
        """Sum of the parts valuation for conglomerates."""
        return calculate_sum_of_parts(financials)

    def _calculate_quality_score(self, financials: Dict) -> Dict:
        """Calculate quality score based on multiple factors."""
        return calculate_quality_score(financials)

    def _score_profitability(self, financials: Dict) -> float:
        """Score profitability quality (0-100)."""
        return score_profitability(financials)

    def _score_balance_sheet(self, financials: Dict) -> float:
        """Score balance sheet quality (0-100)."""
        return score_balance_sheet(financials)

    def _score_earnings_quality(self, financials: Dict) -> float:
        """Score earnings quality (0-100)."""
        return score_earnings_quality(financials)

    def _score_growth_quality(self, financials: Dict) -> float:
        """Score growth quality (0-100)."""
        return score_growth_quality(financials)

    def _score_capital_allocation(self, financials: Dict) -> float:
        """Score capital allocation quality (0-100)."""
        return score_capital_allocation(financials)

    def _assess_financial_health(self, financials: Dict) -> Dict:
        """Assess financial health and solvency."""
        return assess_financial_health(financials)

    def _calculate_altman_z_score(self, financials: Dict) -> Dict:
        """Calculate Altman Z-Score for bankruptcy prediction."""
        return calculate_altman_z_score(financials)

    def _calculate_piotroski_score(self, financials: Dict) -> Dict:
        """Calculate Piotroski F-Score (0-9)."""
        return calculate_piotroski_score(financials)

    def _calculate_beneish_m_score(self, financials: Dict) -> Dict:
        """Calculate Beneish M-Score for earnings manipulation detection."""
        return calculate_beneish_m_score(financials)

    def _analyze_liquidity(self, financials: Dict) -> Dict:
        """Analyze liquidity position."""
        return analyze_liquidity(financials)

    def _analyze_solvency(self, financials: Dict) -> Dict:
        """Analyze solvency position."""
        return analyze_solvency(financials)

    def _analyze_cash_flows(self, financials: Dict) -> Dict:
        """Analyze cash flow patterns."""
        return analyze_cash_flows(financials)

    def _calculate_overall_health_score(self, health_metrics: Dict) -> float:
        """Calculate overall financial health score."""
        return calculate_overall_health_score(health_metrics)

    def _grade_liquidity(self, financials: Dict) -> str:
        """Grade liquidity position."""
        return grade_liquidity(financials)

    def _grade_solvency(self, financials: Dict) -> str:
        """Grade solvency position."""
        return grade_solvency(financials)

    def _grade_cash_flows(self, financials: Dict) -> str:
        """Grade cash flow quality."""
        return grade_cash_flows(financials)

    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate CAGR from a list of values."""
        return calculate_growth_rate(values)

    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate Compound Annual Growth Rate."""
        return calculate_cagr(values)

    def _get_industry_multiple(self, industry: str) -> float:
        """Get typical EV/EBITDA multiple for industry."""
        return get_industry_multiple(industry)

    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        return get_quality_grade(score)

    def _calculate_margin_stability(self, financials: Dict) -> float:
        """Calculate margin stability score."""
        return calculate_margin_stability(financials)

    def _evaluate_buyback_effectiveness(self, financials: Dict) -> float:
        """Evaluate share buyback effectiveness."""
        return evaluate_buyback_effectiveness(financials)

    # ------------------------------------------------------------------
    # Growth analysis (kept here – tightly coupled to the orchestrator)
    # ------------------------------------------------------------------

    def _analyze_growth(self, financials: Dict) -> Dict:
        """Comprehensive growth analysis."""
        return {
            'historical_growth': self._analyze_historical_growth(financials),
            'growth_drivers': self._identify_growth_drivers(financials),
            'growth_sustainability': self._assess_growth_sustainability(financials),
            'growth_forecast': self._forecast_growth(financials),
        }

    def _analyze_historical_growth(self, financials: Dict) -> Dict:
        """Analyze historical growth patterns."""
        metrics = {}

        # Revenue growth
        revenue_history = financials.get('revenue_history', [])
        if len(revenue_history) >= 3:
            metrics['revenue_cagr_3y'] = calculate_cagr(revenue_history[-3:])
            metrics['revenue_volatility'] = np.std([
                (revenue_history[i] - revenue_history[i - 1]) / revenue_history[i - 1]
                for i in range(1, len(revenue_history))
            ]) if len(revenue_history) > 1 else 0

        # Earnings growth
        earnings_history = financials.get('earnings_history', [])
        if len(earnings_history) >= 3:
            metrics['earnings_cagr_3y'] = calculate_cagr(earnings_history[-3:])

        # Free cash flow growth
        fcf_history = financials.get('fcf_history', [])
        if len(fcf_history) >= 3:
            metrics['fcf_cagr_3y'] = calculate_cagr(fcf_history[-3:])

        return metrics

    def _identify_growth_drivers(self, financials: Dict) -> List[str]:
        """Identify key growth drivers."""
        drivers = []

        # Organic growth
        if financials.get('same_store_sales_growth', 0) > 5:
            drivers.append('strong_organic_growth')

        # Market expansion
        if financials.get('geographic_expansion', False):
            drivers.append('geographic_expansion')

        # Product innovation
        if financials.get('rd_to_revenue', 0) > 0.05:
            drivers.append('product_innovation')

        # Market share gains
        if financials.get('market_share_change', 0) > 0:
            drivers.append('market_share_gains')

        # Pricing power
        if financials.get('pricing_power_score', 0) > 0.7:
            drivers.append('pricing_power')

        # Operational leverage
        if financials.get('operating_leverage', 0) > 1.2:
            drivers.append('operational_leverage')

        return drivers

    def _assess_growth_sustainability(self, financials: Dict) -> Dict:
        """Assess if growth is sustainable."""
        factors = {
            'organic': financials.get('organic_growth_rate', 0) > 0,
            'margin_stable': abs(financials.get('operating_margin_trend', 0)) < 0.02,
            'roic_maintained': financials.get('roic', 0) > financials.get('wacc', 10),
            'reinvestment_rate': financials.get('reinvestment_rate', 0) > 0.2,
            'market_growth': financials.get('market_growth_rate', 0) > 0.05,
        }

        sustainability_score = sum(factors.values()) / len(factors)

        return {
            'score': sustainability_score,
            'factors': factors,
            'assessment': 'sustainable' if sustainability_score > 0.6 else 'questionable',
        }

    def _forecast_growth(self, financials: Dict) -> Dict:
        """Forecast future growth rates."""
        # Simple growth forecast based on historical trends and fundamentals
        historical_growth = financials.get('revenue_growth', 0)
        market_growth = financials.get('market_growth_rate', 5)
        competitive_position = financials.get('market_share_change', 0)

        # Base case
        base_growth = historical_growth * 0.7 + market_growth * 0.3

        # Adjust for competitive position
        if competitive_position > 0:
            base_growth *= 1.1
        elif competitive_position < 0:
            base_growth *= 0.9

        return {
            'next_year': base_growth,
            'three_year': base_growth * 0.8,
            'five_year': base_growth * 0.6,
            'terminal': min(3, base_growth * 0.4),
        }

    # ------------------------------------------------------------------
    # Moat and management analysis
    # ------------------------------------------------------------------

    def _analyze_moat(self, financials: Dict, market_data: Dict) -> Dict:
        """Analyze competitive moat."""
        moat_analysis = {
            'moat_sources': [],
            'moat_trend': 'stable',
            'moat_score': 0,
        }

        # 1. Network Effects
        if market_data.get('network_effects_score', 0) > 0.7:
            moat_analysis['moat_sources'].append({
                'type': 'network_effects',
                'strength': 'strong',
                'description': 'Value increases with more users',
            })

        # 2. Switching Costs
        customer_retention = financials.get('customer_retention_rate', 0)
        if customer_retention > 0.9:
            moat_analysis['moat_sources'].append({
                'type': 'switching_costs',
                'strength': 'strong',
                'description': 'High customer retention indicates switching costs',
            })

        # 3. Intangible Assets
        if financials.get('brand_value', 0) > 0 or financials.get('patents_count', 0) > 100:
            moat_analysis['moat_sources'].append({
                'type': 'intangible_assets',
                'strength': 'moderate',
                'description': 'Strong brand or patent portfolio',
            })

        # 4. Cost Advantages
        if financials.get('gross_margin', 0) > financials.get('industry_avg_gross_margin', 0) * 1.2:
            moat_analysis['moat_sources'].append({
                'type': 'cost_advantages',
                'strength': 'strong',
                'description': 'Significantly higher margins than industry',
            })

        # 5. Efficient Scale
        if market_data.get('market_share', 0) > 0.3 and market_data.get('industry_concentration', 0) > 0.7:
            moat_analysis['moat_sources'].append({
                'type': 'efficient_scale',
                'strength': 'moderate',
                'description': 'Dominant position in concentrated market',
            })

        # Calculate moat score
        moat_analysis['moat_score'] = len(moat_analysis['moat_sources']) * 20

        # Determine moat rating
        if moat_analysis['moat_score'] >= 60:
            moat_analysis['rating'] = 'wide'
        elif moat_analysis['moat_score'] >= 40:
            moat_analysis['rating'] = 'narrow'
        else:
            moat_analysis['rating'] = 'none'

        # Analyze moat trend
        if financials.get('market_share_change', 0) < -0.02:
            moat_analysis['moat_trend'] = 'eroding'
        elif financials.get('market_share_change', 0) > 0.02:
            moat_analysis['moat_trend'] = 'strengthening'

        return moat_analysis

    def _assess_management_quality(self, financials: Dict) -> Dict:
        """Assess management quality."""
        management_score = {
            'capital_allocation': score_capital_allocation(financials) / 100,
            'execution': 0,
            'transparency': 0,
            'alignment': 0,
            'track_record': 0,
        }

        # Execution score
        if financials.get('revenue_guidance_accuracy', 0) > 0.95:
            management_score['execution'] += 0.5
        if financials.get('earnings_guidance_accuracy', 0) > 0.95:
            management_score['execution'] += 0.5

        # Transparency score
        if financials.get('segment_reporting_detail', 0) > 0.8:
            management_score['transparency'] += 0.5
        if financials.get('conference_call_participation', 0) > 0.9:
            management_score['transparency'] += 0.5

        # Alignment score
        insider_ownership = financials.get('insider_ownership', 0)
        if insider_ownership > 0.05 and insider_ownership < 0.30:
            management_score['alignment'] = 1.0
        elif insider_ownership > 0.01:
            management_score['alignment'] = 0.5

        # Track record
        if financials.get('ceo_tenure', 0) > 5 and financials.get('avg_roic_under_ceo', 0) > 15:
            management_score['track_record'] = 1.0
        elif financials.get('avg_roic_under_ceo', 0) > 10:
            management_score['track_record'] = 0.5

        # Overall score
        overall_score = np.mean(list(management_score.values())) * 100

        return {
            'overall_score': overall_score,
            'components': management_score,
            'grade': get_quality_grade(overall_score),
            'red_flags': self._identify_management_red_flags(financials),
        }

    def _identify_management_red_flags(self, financials: Dict) -> List[str]:
        """Identify management red flags."""
        red_flags = []

        if financials.get('ceo_turnover_rate', 0) > 0.3:
            red_flags.append('High executive turnover')

        if financials.get('audit_issues_count', 0) > 0:
            red_flags.append('Audit concerns identified')

        if financials.get('related_party_transactions', 0) > 0.05:
            red_flags.append('Significant related party transactions')

        if financials.get('earnings_restatements', 0) > 0:
            red_flags.append('History of earnings restatements')

        return red_flags

    # ------------------------------------------------------------------
    # Composite scoring, risk/opportunity identification, peer comparison
    # ------------------------------------------------------------------

    def _calculate_composite_score(self, analysis: Dict) -> float:
        """Calculate overall fundamental score (0-100)."""
        weights = {
            'valuation': 0.25,
            'quality': 0.25,
            'growth': 0.20,
            'financial_health': 0.15,
            'moat': 0.10,
            'management': 0.05,
        }

        scores = {}

        # Valuation score
        valuation = analysis.get('valuation_models', {})
        upside = valuation.get('upside_potential', 0)
        if upside > 30:
            scores['valuation'] = 100
        elif upside > 15:
            scores['valuation'] = 70
        elif upside > 0:
            scores['valuation'] = 50
        else:
            scores['valuation'] = 20

        # Quality score
        scores['quality'] = analysis.get('quality_score', {}).get('overall_score', 50)

        # Growth score
        growth = analysis.get('growth_analysis', {})
        growth_rate = growth.get('historical_growth', {}).get('revenue_cagr_3y', 0)
        if growth_rate > 15:
            scores['growth'] = 90
        elif growth_rate > 10:
            scores['growth'] = 70
        elif growth_rate > 5:
            scores['growth'] = 50
        else:
            scores['growth'] = 30

        # Financial health score
        health = analysis.get('financial_health', {})
        scores['financial_health'] = health.get('overall_health', 50)

        # Moat score
        scores['moat'] = analysis.get('moat_analysis', {}).get('moat_score', 0)

        # Management score
        scores['management'] = analysis.get('management_quality', {}).get('overall_score', 50)

        # Calculate weighted score
        composite = sum(scores.get(factor, 0) * weight for factor, weight in weights.items())

        return composite

    def _identify_risks(self, analysis: Dict) -> List[Dict]:
        """Identify key risks."""
        risks = []

        # Valuation risk
        if analysis.get('valuation_models', {}).get('pe_ratio', 0) > 30:
            risks.append({
                'type': 'valuation',
                'severity': 'high',
                'description': 'High valuation multiples indicate potential downside risk',
            })

        # Leverage risk
        metrics = analysis.get('financial_metrics')
        if metrics and metrics.debt_to_equity > 2:
            risks.append({
                'type': 'leverage',
                'severity': 'high',
                'description': 'High debt levels increase financial risk',
            })

        # Profitability risk
        if metrics and metrics.net_margin < 5:
            risks.append({
                'type': 'profitability',
                'severity': 'medium',
                'description': 'Low profit margins vulnerable to cost pressures',
            })

        # Manipulation risk
        m_score = analysis.get('financial_health', {}).get('beneish_m_score', {})
        if m_score.get('likelihood') == 'high':
            risks.append({
                'type': 'accounting',
                'severity': 'high',
                'description': 'Potential earnings manipulation detected',
            })

        return risks

    def _identify_opportunities(self, analysis: Dict) -> List[Dict]:
        """Identify opportunities."""
        opportunities = []

        # Valuation opportunity
        upside = analysis.get('valuation_models', {}).get('upside_potential', 0)
        if upside > 30:
            opportunities.append({
                'type': 'valuation',
                'potential': 'high',
                'description': f'Significant upside potential of {upside:.1f}%',
            })

        # Growth opportunity
        growth_drivers = analysis.get('growth_analysis', {}).get('growth_drivers', [])
        if len(growth_drivers) >= 3:
            opportunities.append({
                'type': 'growth',
                'potential': 'high',
                'description': f'Multiple growth drivers: {", ".join(growth_drivers)}',
            })

        # Quality opportunity
        quality_score = analysis.get('quality_score', {}).get('overall_score', 0)
        if quality_score > 80:
            opportunities.append({
                'type': 'quality',
                'potential': 'medium',
                'description': 'High-quality business with sustainable advantages',
            })

        return opportunities

    def _compare_with_peers(
        self, company_financials: Dict, peer_data: List[Dict]
    ) -> Dict:
        """Compare company with industry peers."""
        if not peer_data:
            return {}

        # Calculate peer averages
        peer_metrics = {}
        metrics_to_compare = [
            'pe_ratio', 'ev_to_ebitda', 'profit_margin', 'roe', 'debt_to_equity',
            'revenue_growth', 'fcf_yield',
        ]

        for metric in metrics_to_compare:
            peer_values = [p.get(metric, 0) for p in peer_data if p.get(metric) is not None]
            if peer_values:
                peer_metrics[f'{metric}_peer_avg'] = np.mean(peer_values)
                peer_metrics[f'{metric}_peer_median'] = np.median(peer_values)

                # Calculate percentile ranking
                company_value = company_financials.get(metric, 0)
                percentile = stats.percentileofscore(peer_values, company_value)
                peer_metrics[f'{metric}_percentile'] = percentile

        # Overall peer comparison score
        valuation_percentile = peer_metrics.get('pe_ratio_percentile', 50)
        profitability_percentile = peer_metrics.get('roe_percentile', 50)
        growth_percentile = peer_metrics.get('revenue_growth_percentile', 50)

        overall_percentile = np.mean([
            valuation_percentile,
            profitability_percentile,
            growth_percentile,
        ])

        return {
            'metrics': peer_metrics,
            'overall_percentile': overall_percentile,
            'relative_value': (
                'undervalued' if valuation_percentile < 30
                else 'overvalued' if valuation_percentile > 70
                else 'fair'
            ),
            'competitive_position': (
                'strong' if overall_percentile > 70
                else 'weak' if overall_percentile < 30
                else 'average'
            ),
        }

    # ------------------------------------------------------------------
    # Stub for efficiency metrics (preserves original interface)
    # ------------------------------------------------------------------

    def _calculate_efficiency_metrics(self, financials: Dict) -> Dict:
        """
        Placeholder preserving the original interface.

        The original file called this method but never defined it, so it would
        raise AttributeError at runtime.  The method is kept here as a stub so
        the call in ``analyze_company`` does not change its behaviour (the
        AttributeError was the original behaviour; replacing it with an empty
        dict is a safe improvement).
        """
        return {}
