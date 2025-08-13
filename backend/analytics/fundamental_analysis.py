"""
Advanced Fundamental Analysis Engine
Comprehensive financial analysis including DCF, peer comparison, and quality metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats

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


class FundamentalAnalysisEngine:
    """
    Comprehensive fundamental analysis using SEC data and financial APIs
    """
    
    def __init__(self):
        self.sector_averages = {}  # Cache sector averages
        self.risk_free_rate = 0.045  # Current 10-year treasury
        self.market_risk_premium = 0.08  # Historical equity risk premium
    
    async def analyze_company(
        self,
        ticker: str,
        financials: Dict,
        market_data: Dict,
        peer_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis
        """
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.utcnow().isoformat(),
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
            'opportunities': []
        }
        
        # Calculate composite fundamental score
        analysis['composite_score'] = self._calculate_composite_score(analysis)
        
        # Identify risks and opportunities
        analysis['risks'] = self._identify_risks(analysis)
        analysis['opportunities'] = self._identify_opportunities(analysis)
        
        return analysis
    
    def _calculate_financial_metrics(
        self,
        financials: Dict,
        market_data: Dict
    ) -> FinancialMetrics:
        """
        Calculate all financial metrics
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
        revenue_growth = self._calculate_growth_rate(
            financials.get('revenue_history', [])
        )
        earnings_growth = self._calculate_growth_rate(
            financials.get('earnings_history', [])
        )
        fcf_growth = self._calculate_growth_rate(
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
            receivables_turnover=receivables_turnover
        )
    
    def _run_valuation_models(
        self,
        financials: Dict,
        market_data: Dict
    ) -> Dict[str, Any]:
        """
        Run multiple valuation models
        """
        valuations = {}
        
        # 1. Discounted Cash Flow (DCF)
        valuations['dcf'] = self._calculate_dcf(financials, market_data)
        
        # 2. Dividend Discount Model (DDM)
        valuations['ddm'] = self._calculate_ddm(financials, market_data)
        
        # 3. Residual Income Model
        valuations['rim'] = self._calculate_residual_income(financials, market_data)
        
        # 4. Asset-Based Valuation
        valuations['asset_based'] = self._calculate_asset_based_value(financials)
        
        # 5. Earnings Power Value (EPV)
        valuations['epv'] = self._calculate_epv(financials, market_data)
        
        # 6. Sum of Parts Valuation
        valuations['sotp'] = self._calculate_sum_of_parts(financials)
        
        # Calculate average and range
        valid_values = [v['value'] for v in valuations.values() if v and v.get('value', 0) > 0]
        
        if valid_values:
            valuations['average'] = np.mean(valid_values)
            valuations['median'] = np.median(valid_values)
            valuations['range'] = {
                'min': min(valid_values),
                'max': max(valid_values)
            }
            
            current_price = market_data.get('price', 0)
            valuations['upside_potential'] = ((valuations['average'] - current_price) / current_price * 100) if current_price > 0 else 0
        
        return valuations
    
    def _calculate_dcf(self, financials: Dict, market_data: Dict) -> Dict:
        """
        Discounted Cash Flow model
        """
        # Get inputs
        fcf = financials.get('free_cash_flow', 0)
        growth_rate = financials.get('fcf_growth', 0.05)  # 5% default
        terminal_growth = 0.03  # 3% perpetual growth
        shares_outstanding = financials.get('shares_outstanding', 1)
        
        # Calculate WACC
        wacc = self._calculate_wacc(financials, market_data)
        
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
        pv_fcf = sum([fcf / (1 + wacc) ** i for i, fcf in enumerate(projected_fcf, 1)])
        
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
            'confidence': 0.8  # High confidence in DCF
        }
    
    def _calculate_wacc(self, financials: Dict, market_data: Dict) -> float:
        """
        Calculate Weighted Average Cost of Capital
        """
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
        cost_of_equity = self.risk_free_rate + beta * self.market_risk_premium
        
        # Cost of debt
        interest_expense = financials.get('interest_expense', 0)
        cost_of_debt = interest_expense / debt_value if debt_value > 0 else 0.04
        
        # Tax rate
        tax_rate = financials.get('tax_rate', 0.21)
        
        # WACC
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        
        return wacc
    
    def _calculate_ddm(self, financials: Dict, market_data: Dict) -> Dict:
        """
        Dividend Discount Model
        """
        dividend_per_share = financials.get('dividend_per_share', 0)
        
        if dividend_per_share == 0:
            return {'value': 0, 'confidence': 0}
        
        # Get dividend growth rate
        dividend_history = financials.get('dividend_history', [])
        if len(dividend_history) >= 3:
            growth_rate = self._calculate_growth_rate(dividend_history)
        else:
            # Estimate based on earnings growth
            growth_rate = financials.get('earnings_growth', 0.03)
        
        # Required return (using CAPM)
        beta = market_data.get('beta', 1.0)
        required_return = self.risk_free_rate + beta * self.market_risk_premium
        
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
            'confidence': 0.7
        }
    
    def _calculate_residual_income(self, financials: Dict, market_data: Dict) -> Dict:
        """
        Residual Income Model
        """
        book_value = financials.get('book_value_per_share', 0)
        roe = financials.get('roe', 0)
        
        # Required return
        beta = market_data.get('beta', 1.0)
        required_return = self.risk_free_rate + beta * self.market_risk_premium
        
        # Project residual income
        years = 10
        terminal_growth = 0.03
        
        current_bv = book_value
        pv_residual_income = 0
        
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
            'confidence': 0.6
        }
    
    def _calculate_asset_based_value(self, financials: Dict) -> Dict:
        """
        Asset-based valuation
        """
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
            ppe * 0.5 -  # 50% of PP&E
            total_liabilities
        )
        
        liquidation_per_share = liquidation_value / shares_outstanding if shares_outstanding > 0 else 0
        
        return {
            'value': nav_per_share,
            'tangible_nav': tangible_nav,
            'liquidation_value': liquidation_per_share,
            'confidence': 0.5
        }
    
    def _calculate_epv(self, financials: Dict, market_data: Dict) -> Dict:
        """
        Earnings Power Value (Greenwald method)
        """
        # Normalized earnings
        operating_income = financials.get('operating_income', 0)
        tax_rate = financials.get('tax_rate', 0.21)
        
        # Normalize for business cycle
        normalized_ebit = operating_income  # Could adjust for cycle
        
        # After-tax earnings
        normalized_earnings = normalized_ebit * (1 - tax_rate)
        
        # Cost of capital
        wacc = self._calculate_wacc(financials, market_data)
        
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
            'confidence': 0.7
        }
    
    def _calculate_sum_of_parts(self, financials: Dict) -> Dict:
        """
        Sum of the parts valuation for conglomerates
        """
        segments = financials.get('segments', [])
        
        if not segments:
            return {'value': 0, 'confidence': 0}
        
        total_value = 0
        segment_values = {}
        
        for segment in segments:
            # Value each segment separately
            segment_revenue = segment.get('revenue', 0)
            segment_ebitda = segment.get('ebitda', segment.get('operating_income', 0) * 1.2)
            
            # Apply industry multiples
            industry = segment.get('industry', 'general')
            ev_ebitda_multiple = self._get_industry_multiple(industry)
            
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
            'confidence': 0.6
        }
    
    def _calculate_quality_score(self, financials: Dict) -> Dict:
        """
        Calculate quality score based on multiple factors
        """
        scores = {}
        
        # 1. Profitability Quality (Piotroski F-Score components)
        scores['profitability'] = self._score_profitability(financials)
        
        # 2. Balance Sheet Quality
        scores['balance_sheet'] = self._score_balance_sheet(financials)
        
        # 3. Earnings Quality
        scores['earnings_quality'] = self._score_earnings_quality(financials)
        
        # 4. Growth Quality
        scores['growth_quality'] = self._score_growth_quality(financials)
        
        # 5. Capital Allocation
        scores['capital_allocation'] = self._score_capital_allocation(financials)
        
        # Overall quality score (0-100)
        overall_score = np.mean(list(scores.values()))
        
        return {
            'overall_score': overall_score,
            'scores': scores,
            'grade': self._get_quality_grade(overall_score)
        }
    
    def _score_profitability(self, financials: Dict) -> float:
        """Score profitability quality (0-100)"""
        score = 0
        max_score = 100
        
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
    
    def _score_balance_sheet(self, financials: Dict) -> float:
        """Score balance sheet quality (0-100)"""
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
    
    def _score_earnings_quality(self, financials: Dict) -> float:
        """Score earnings quality (0-100)"""
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
        margin_stability = self._calculate_margin_stability(financials)
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
    
    def _score_growth_quality(self, financials: Dict) -> float:
        """Score growth quality (0-100)"""
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
    
    def _score_capital_allocation(self, financials: Dict) -> float:
        """Score capital allocation quality (0-100)"""
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
        buyback_effectiveness = self._evaluate_buyback_effectiveness(financials)
        score += buyback_effectiveness * 25
        
        # Acquisition track record (20 points)
        acquisition_returns = financials.get('acquisition_roic', 0)
        if acquisition_returns > 15:
            score += 20
        elif acquisition_returns > 10:
            score += 10
        
        return score
    
    def _analyze_growth(self, financials: Dict) -> Dict:
        """
        Comprehensive growth analysis
        """
        growth_analysis = {
            'historical_growth': self._analyze_historical_growth(financials),
            'growth_drivers': self._identify_growth_drivers(financials),
            'growth_sustainability': self._assess_growth_sustainability(financials),
            'growth_forecast': self._forecast_growth(financials)
        }
        
        return growth_analysis
    
    def _analyze_historical_growth(self, financials: Dict) -> Dict:
        """Analyze historical growth patterns"""
        metrics = {}
        
        # Revenue growth
        revenue_history = financials.get('revenue_history', [])
        if len(revenue_history) >= 3:
            metrics['revenue_cagr_3y'] = self._calculate_cagr(revenue_history[-3:])
            metrics['revenue_volatility'] = np.std([
                (revenue_history[i] - revenue_history[i-1]) / revenue_history[i-1]
                for i in range(1, len(revenue_history))
            ]) if len(revenue_history) > 1 else 0
        
        # Earnings growth
        earnings_history = financials.get('earnings_history', [])
        if len(earnings_history) >= 3:
            metrics['earnings_cagr_3y'] = self._calculate_cagr(earnings_history[-3:])
        
        # Free cash flow growth
        fcf_history = financials.get('fcf_history', [])
        if len(fcf_history) >= 3:
            metrics['fcf_cagr_3y'] = self._calculate_cagr(fcf_history[-3:])
        
        return metrics
    
    def _identify_growth_drivers(self, financials: Dict) -> List[str]:
        """Identify key growth drivers"""
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
    
    def _assess_financial_health(self, financials: Dict) -> Dict:
        """
        Assess financial health and solvency
        """
        health_metrics = {
            'altman_z_score': self._calculate_altman_z_score(financials),
            'piotroski_f_score': self._calculate_piotroski_score(financials),
            'beneish_m_score': self._calculate_beneish_m_score(financials),
            'liquidity_analysis': self._analyze_liquidity(financials),
            'solvency_analysis': self._analyze_solvency(financials),
            'cash_flow_analysis': self._analyze_cash_flows(financials)
        }
        
        # Overall health score
        health_metrics['overall_health'] = self._calculate_overall_health_score(health_metrics)
        
        return health_metrics
    
    def _calculate_altman_z_score(self, financials: Dict) -> Dict:
        """
        Calculate Altman Z-Score for bankruptcy prediction
        """
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
                'sales_to_assets': x5
            }
        }
    
    def _calculate_piotroski_score(self, financials: Dict) -> Dict:
        """
        Calculate Piotroski F-Score (0-9)
        """
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
            'strength': 'strong' if score >= 7 else 'moderate' if score >= 4 else 'weak'
        }
    
    def _calculate_beneish_m_score(self, financials: Dict) -> Dict:
        """
        Calculate Beneish M-Score for earnings manipulation detection
        """
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
        aqi = self._calculate_asset_quality_index(financials)
        
        # 4. Sales Growth Index
        sgi = financials.get('revenue', 1) / financials.get('revenue_previous', 1)
        
        # 5. Depreciation Index
        depi = self._calculate_depreciation_index(financials)
        
        # 6. SG&A Index
        sgai = self._calculate_sga_index(financials)
        
        # 7. Leverage Index
        lvgi = financials.get('debt_to_assets', 0) / financials.get('debt_to_assets_previous', 1)
        
        # 8. Total Accruals to Total Assets
        tata = self._calculate_total_accruals(financials)
        
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
                'tata': tata
            }
        }
    
    def _analyze_moat(self, financials: Dict, market_data: Dict) -> Dict:
        """
        Analyze competitive moat
        """
        moat_analysis = {
            'moat_sources': [],
            'moat_trend': 'stable',
            'moat_score': 0
        }
        
        # 1. Network Effects
        if market_data.get('network_effects_score', 0) > 0.7:
            moat_analysis['moat_sources'].append({
                'type': 'network_effects',
                'strength': 'strong',
                'description': 'Value increases with more users'
            })
        
        # 2. Switching Costs
        customer_retention = financials.get('customer_retention_rate', 0)
        if customer_retention > 0.9:
            moat_analysis['moat_sources'].append({
                'type': 'switching_costs',
                'strength': 'strong',
                'description': 'High customer retention indicates switching costs'
            })
        
        # 3. Intangible Assets
        if financials.get('brand_value', 0) > 0 or financials.get('patents_count', 0) > 100:
            moat_analysis['moat_sources'].append({
                'type': 'intangible_assets',
                'strength': 'moderate',
                'description': 'Strong brand or patent portfolio'
            })
        
        # 4. Cost Advantages
        if financials.get('gross_margin', 0) > financials.get('industry_avg_gross_margin', 0) * 1.2:
            moat_analysis['moat_sources'].append({
                'type': 'cost_advantages',
                'strength': 'strong',
                'description': 'Significantly higher margins than industry'
            })
        
        # 5. Efficient Scale
        if market_data.get('market_share', 0) > 0.3 and market_data.get('industry_concentration', 0) > 0.7:
            moat_analysis['moat_sources'].append({
                'type': 'efficient_scale',
                'strength': 'moderate',
                'description': 'Dominant position in concentrated market'
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
        """
        Assess management quality
        """
        management_score = {
            'capital_allocation': self._score_capital_allocation(financials) / 100,
            'execution': 0,
            'transparency': 0,
            'alignment': 0,
            'track_record': 0
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
            'grade': self._get_quality_grade(overall_score),
            'red_flags': self._identify_management_red_flags(financials)
        }
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """
        Calculate overall fundamental score (0-100)
        """
        weights = {
            'valuation': 0.25,
            'quality': 0.25,
            'growth': 0.20,
            'financial_health': 0.15,
            'moat': 0.10,
            'management': 0.05
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
        """Identify key risks"""
        risks = []
        
        # Valuation risk
        if analysis.get('valuation_models', {}).get('pe_ratio', 0) > 30:
            risks.append({
                'type': 'valuation',
                'severity': 'high',
                'description': 'High valuation multiples indicate potential downside risk'
            })
        
        # Leverage risk
        metrics = analysis.get('financial_metrics')
        if metrics and metrics.debt_to_equity > 2:
            risks.append({
                'type': 'leverage',
                'severity': 'high',
                'description': 'High debt levels increase financial risk'
            })
        
        # Profitability risk
        if metrics and metrics.net_margin < 5:
            risks.append({
                'type': 'profitability',
                'severity': 'medium',
                'description': 'Low profit margins vulnerable to cost pressures'
            })
        
        # Manipulation risk
        m_score = analysis.get('financial_health', {}).get('beneish_m_score', {})
        if m_score.get('likelihood') == 'high':
            risks.append({
                'type': 'accounting',
                'severity': 'high',
                'description': 'Potential earnings manipulation detected'
            })
        
        return risks
    
    def _identify_opportunities(self, analysis: Dict) -> List[Dict]:
        """Identify opportunities"""
        opportunities = []
        
        # Valuation opportunity
        upside = analysis.get('valuation_models', {}).get('upside_potential', 0)
        if upside > 30:
            opportunities.append({
                'type': 'valuation',
                'potential': 'high',
                'description': f'Significant upside potential of {upside:.1f}%'
            })
        
        # Growth opportunity
        growth_drivers = analysis.get('growth_analysis', {}).get('growth_drivers', [])
        if len(growth_drivers) >= 3:
            opportunities.append({
                'type': 'growth',
                'potential': 'high',
                'description': f'Multiple growth drivers: {", ".join(growth_drivers)}'
            })
        
        # Quality opportunity
        quality_score = analysis.get('quality_score', {}).get('overall_score', 0)
        if quality_score > 80:
            opportunities.append({
                'type': 'quality',
                'potential': 'medium',
                'description': 'High-quality business with sustainable advantages'
            })
        
        return opportunities
    
    # Helper methods
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate CAGR from a list of values"""
        if len(values) < 2 or values[0] <= 0:
            return 0
        
        years = len(values) - 1
        return (pow(values[-1] / values[0], 1/years) - 1) * 100
    
    def _calculate_cagr(self, values: List[float]) -> float:
        """Calculate Compound Annual Growth Rate"""
        return self._calculate_growth_rate(values)
    
    def _get_industry_multiple(self, industry: str) -> float:
        """Get typical EV/EBITDA multiple for industry"""
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
            'general': 10
        }
        return industry_multiples.get(industry.lower(), 10)
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
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
    
    def _calculate_margin_stability(self, financials: Dict) -> float:
        """Calculate margin stability score"""
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
    
    def _evaluate_buyback_effectiveness(self, financials: Dict) -> float:
        """Evaluate share buyback effectiveness"""
        buyback_price = financials.get('avg_buyback_price', 0)
        current_price = financials.get('current_price', 0)
        
        if buyback_price > 0 and current_price > 0:
            # Good if bought below current price
            effectiveness = min(1.0, current_price / buyback_price - 0.5)
            return max(0, effectiveness)
        
        return 0.5
    
    def _calculate_asset_quality_index(self, financials: Dict) -> float:
        """Calculate asset quality index for M-Score"""
        non_current_assets = financials.get('total_assets', 0) - financials.get('current_assets', 0)
        ppe = financials.get('property_plant_equipment', 0)
        
        if non_current_assets > 0:
            aqi_current = 1 - (ppe / non_current_assets)
        else:
            aqi_current = 0
        
        non_current_assets_prev = financials.get('total_assets_previous', 0) - financials.get('current_assets_previous', 0)
        ppe_prev = financials.get('property_plant_equipment_previous', 0)
        
        if non_current_assets_prev > 0:
            aqi_previous = 1 - (ppe_prev / non_current_assets_prev)
        else:
            aqi_previous = 0
        
        return aqi_current / aqi_previous if aqi_previous > 0 else 1
    
    def _calculate_depreciation_index(self, financials: Dict) -> float:
        """Calculate depreciation index for M-Score"""
        dep_rate = financials.get('depreciation', 0) / financials.get('property_plant_equipment', 1)
        dep_rate_prev = financials.get('depreciation_previous', 0) / financials.get('property_plant_equipment_previous', 1)
        
        return dep_rate_prev / dep_rate if dep_rate > 0 else 1
    
    def _calculate_sga_index(self, financials: Dict) -> float:
        """Calculate SG&A index for M-Score"""
        sga_rate = financials.get('sga_expenses', 0) / financials.get('revenue', 1)
        sga_rate_prev = financials.get('sga_expenses_previous', 0) / financials.get('revenue_previous', 1)
        
        return sga_rate / sga_rate_prev if sga_rate_prev > 0 else 1
    
    def _calculate_total_accruals(self, financials: Dict) -> float:
        """Calculate total accruals to total assets"""
        total_accruals = (
            financials.get('net_income', 0) - 
            financials.get('operating_cash_flow', 0)
        )
        total_assets = financials.get('total_assets', 1)
        
        return total_accruals / total_assets
    
    def _analyze_liquidity(self, financials: Dict) -> Dict:
        """Analyze liquidity position"""
        return {
            'current_ratio': financials.get('current_ratio', 0),
            'quick_ratio': financials.get('quick_ratio', 0),
            'cash_ratio': financials.get('cash_ratio', 0),
            'working_capital': financials.get('working_capital', 0),
            'cash_conversion_cycle': financials.get('cash_conversion_cycle', 0),
            'liquidity_grade': self._grade_liquidity(financials)
        }
    
    def _analyze_solvency(self, financials: Dict) -> Dict:
        """Analyze solvency position"""
        return {
            'debt_to_equity': financials.get('debt_to_equity', 0),
            'debt_to_assets': financials.get('debt_to_assets', 0),
            'interest_coverage': financials.get('interest_coverage', 0),
            'debt_service_coverage': financials.get('debt_service_coverage', 0),
            'solvency_grade': self._grade_solvency(financials)
        }
    
    def _analyze_cash_flows(self, financials: Dict) -> Dict:
        """Analyze cash flow patterns"""
        return {
            'operating_cash_flow': financials.get('operating_cash_flow', 0),
            'free_cash_flow': financials.get('free_cash_flow', 0),
            'fcf_conversion': financials.get('fcf_to_net_income', 0),
            'capex_to_revenue': financials.get('capex_to_revenue', 0),
            'cash_flow_grade': self._grade_cash_flows(financials)
        }
    
    def _calculate_overall_health_score(self, health_metrics: Dict) -> float:
        """Calculate overall financial health score"""
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
    
    def _grade_liquidity(self, financials: Dict) -> str:
        """Grade liquidity position"""
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
    
    def _grade_solvency(self, financials: Dict) -> str:
        """Grade solvency position"""
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
    
    def _grade_cash_flows(self, financials: Dict) -> str:
        """Grade cash flow quality"""
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
    
    def _assess_growth_sustainability(self, financials: Dict) -> Dict:
        """Assess if growth is sustainable"""
        factors = {
            'organic': financials.get('organic_growth_rate', 0) > 0,
            'margin_stable': abs(financials.get('operating_margin_trend', 0)) < 0.02,
            'roic_maintained': financials.get('roic', 0) > financials.get('wacc', 10),
            'reinvestment_rate': financials.get('reinvestment_rate', 0) > 0.2,
            'market_growth': financials.get('market_growth_rate', 0) > 0.05
        }
        
        sustainability_score = sum(factors.values()) / len(factors)
        
        return {
            'score': sustainability_score,
            'factors': factors,
            'assessment': 'sustainable' if sustainability_score > 0.6 else 'questionable'
        }
    
    def _forecast_growth(self, financials: Dict) -> Dict:
        """Forecast future growth rates"""
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
            'terminal': min(3, base_growth * 0.4)
        }
    
    def _identify_management_red_flags(self, financials: Dict) -> List[str]:
        """Identify management red flags"""
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
    
    def _compare_with_peers(self, company_financials: Dict, peer_data: List[Dict]) -> Dict:
        """Compare company with industry peers"""
        if not peer_data:
            return {}
        
        # Calculate peer averages
        peer_metrics = {}
        metrics_to_compare = [
            'pe_ratio', 'ev_to_ebitda', 'profit_margin', 'roe', 'debt_to_equity',
            'revenue_growth', 'fcf_yield'
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
            growth_percentile
        ])
        
        return {
            'metrics': peer_metrics,
            'overall_percentile': overall_percentile,
            'relative_value': 'undervalued' if valuation_percentile < 30 else 'overvalued' if valuation_percentile > 70 else 'fair',
            'competitive_position': 'strong' if overall_percentile > 70 else 'weak' if overall_percentile < 30 else 'average'
        }