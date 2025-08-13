"""
Discounted Cash Flow (DCF) Valuation Model
Implements comprehensive DCF analysis for stock valuation
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DCFInputs:
    """Input parameters for DCF model"""
    revenue_growth_rates: List[float]  # Year-over-year growth rates
    operating_margin: float
    tax_rate: float
    capex_as_percent_revenue: float
    nwc_as_percent_revenue: float  # Net Working Capital
    terminal_growth_rate: float
    wacc: float  # Weighted Average Cost of Capital
    shares_outstanding: float
    current_share_price: float
    debt: float
    cash: float


@dataclass
class DCFResult:
    """Results from DCF analysis"""
    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    current_price: float
    upside_potential: float
    terminal_value: float
    pv_of_cash_flows: float
    sensitivity_analysis: Dict[str, Dict[str, float]]


class DCFModel:
    """Discounted Cash Flow valuation model"""
    
    def __init__(self, company_data: Dict):
        self.company_data = company_data
        self.financial_statements = self._parse_financial_data()
    
    def calculate_dcf(self, inputs: DCFInputs, projection_years: int = 5) -> DCFResult:
        """Calculate DCF valuation"""
        
        # Project free cash flows
        fcf_projections = self._project_free_cash_flows(inputs, projection_years)
        
        # Calculate terminal value
        terminal_value = self._calculate_terminal_value(
            fcf_projections[-1],
            inputs.terminal_growth_rate,
            inputs.wacc
        )
        
        # Discount cash flows to present value
        pv_cash_flows = self._discount_cash_flows(fcf_projections, inputs.wacc)
        pv_terminal_value = terminal_value / ((1 + inputs.wacc) ** projection_years)
        
        # Calculate enterprise and equity value
        enterprise_value = pv_cash_flows + pv_terminal_value
        equity_value = enterprise_value - inputs.debt + inputs.cash
        fair_value_per_share = equity_value / inputs.shares_outstanding
        
        # Calculate upside potential
        upside = ((fair_value_per_share - inputs.current_share_price) / 
                  inputs.current_share_price) * 100
        
        # Perform sensitivity analysis
        sensitivity = self._sensitivity_analysis(inputs, projection_years)
        
        return DCFResult(
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            fair_value_per_share=fair_value_per_share,
            current_price=inputs.current_share_price,
            upside_potential=upside,
            terminal_value=terminal_value,
            pv_of_cash_flows=pv_cash_flows,
            sensitivity_analysis=sensitivity
        )
    
    def _parse_financial_data(self) -> Dict:
        """Parse financial statements from company data"""
        # Implementation would parse actual financial data
        return {
            'revenue': self.company_data.get('revenue', []),
            'operating_income': self.company_data.get('operating_income', []),
            'capex': self.company_data.get('capex', []),
            'working_capital': self.company_data.get('working_capital', [])
        }
    
    def _project_free_cash_flows(self, inputs: DCFInputs, years: int) -> List[float]:
        """Project free cash flows for specified years"""
        fcf_projections = []
        
        # Get base revenue (last known revenue)
        base_revenue = self.financial_statements['revenue'][-1] if \
                      self.financial_statements['revenue'] else 1000000000  # Default $1B
        
        for year in range(years):
            # Project revenue
            if year < len(inputs.revenue_growth_rates):
                growth_rate = inputs.revenue_growth_rates[year]
            else:
                # Use average of provided rates for remaining years
                growth_rate = np.mean(inputs.revenue_growth_rates)
            
            revenue = base_revenue * (1 + growth_rate) ** (year + 1)
            
            # Calculate components
            ebit = revenue * inputs.operating_margin
            tax = ebit * inputs.tax_rate
            nopat = ebit - tax  # Net Operating Profit After Tax
            
            # Calculate capital investments
            capex = revenue * inputs.capex_as_percent_revenue
            nwc_change = revenue * inputs.nwc_as_percent_revenue * growth_rate
            
            # Free Cash Flow = NOPAT - CapEx - Change in NWC
            fcf = nopat - capex - nwc_change
            fcf_projections.append(fcf)
        
        return fcf_projections
    
    def _calculate_terminal_value(self, final_fcf: float, 
                                  terminal_growth: float, wacc: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        if wacc <= terminal_growth:
            raise ValueError("WACC must be greater than terminal growth rate")
        
        terminal_value = (final_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        return terminal_value
    
    def _discount_cash_flows(self, cash_flows: List[float], discount_rate: float) -> float:
        """Discount future cash flows to present value"""
        pv_total = 0
        for year, cf in enumerate(cash_flows, 1):
            pv = cf / ((1 + discount_rate) ** year)
            pv_total += pv
        return pv_total
    
    def _sensitivity_analysis(self, inputs: DCFInputs, years: int) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on key variables"""
        sensitivity = {}
        
        # WACC sensitivity
        wacc_range = [inputs.wacc - 0.02, inputs.wacc - 0.01, 
                      inputs.wacc, inputs.wacc + 0.01, inputs.wacc + 0.02]
        sensitivity['wacc'] = {}
        
        for wacc in wacc_range:
            temp_inputs = DCFInputs(**inputs.__dict__)
            temp_inputs.wacc = wacc
            result = self.calculate_dcf(temp_inputs, years)
            sensitivity['wacc'][f"{wacc:.1%}"] = result.fair_value_per_share
        
        # Terminal growth rate sensitivity
        tg_range = [inputs.terminal_growth_rate - 0.01, inputs.terminal_growth_rate,
                   inputs.terminal_growth_rate + 0.01]
        sensitivity['terminal_growth'] = {}
        
        for tg in tg_range:
            temp_inputs = DCFInputs(**inputs.__dict__)
            temp_inputs.terminal_growth_rate = tg
            result = self.calculate_dcf(temp_inputs, years)
            sensitivity['terminal_growth'][f"{tg:.1%}"] = result.fair_value_per_share
        
        # Revenue growth sensitivity
        sensitivity['revenue_growth'] = {}
        for adjustment in [-0.02, -0.01, 0, 0.01, 0.02]:
            temp_inputs = DCFInputs(**inputs.__dict__)
            temp_inputs.revenue_growth_rates = [
                rate + adjustment for rate in inputs.revenue_growth_rates
            ]
            result = self.calculate_dcf(temp_inputs, years)
            sensitivity['revenue_growth'][f"{adjustment:+.1%}"] = result.fair_value_per_share
        
        return sensitivity
    
    def calculate_wacc(self, 
                      risk_free_rate: float,
                      market_return: float,
                      beta: float,
                      cost_of_debt: float,
                      tax_rate: float,
                      debt_value: float,
                      equity_value: float) -> float:
        """Calculate Weighted Average Cost of Capital"""
        
        # Cost of equity using CAPM
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        
        # Total value
        total_value = debt_value + equity_value
        
        # WACC calculation
        wacc = (equity_value / total_value) * cost_of_equity + \
               (debt_value / total_value) * cost_of_debt * (1 - tax_rate)
        
        return wacc
    
    def monte_carlo_dcf(self, inputs: DCFInputs, 
                       simulations: int = 10000) -> Dict[str, float]:
        """Run Monte Carlo simulation for DCF valuation"""
        results = []
        
        for _ in range(simulations):
            # Create random variations of inputs
            sim_inputs = DCFInputs(
                revenue_growth_rates=[
                    rate + np.random.normal(0, 0.02) for rate in inputs.revenue_growth_rates
                ],
                operating_margin=inputs.operating_margin + np.random.normal(0, 0.02),
                tax_rate=inputs.tax_rate,
                capex_as_percent_revenue=inputs.capex_as_percent_revenue + np.random.normal(0, 0.01),
                nwc_as_percent_revenue=inputs.nwc_as_percent_revenue + np.random.normal(0, 0.005),
                terminal_growth_rate=inputs.terminal_growth_rate + np.random.normal(0, 0.005),
                wacc=inputs.wacc + np.random.normal(0, 0.01),
                shares_outstanding=inputs.shares_outstanding,
                current_share_price=inputs.current_share_price,
                debt=inputs.debt,
                cash=inputs.cash
            )
            
            try:
                result = self.calculate_dcf(sim_inputs)
                results.append(result.fair_value_per_share)
            except:
                continue
        
        return {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'percentile_25': np.percentile(results, 25),
            'percentile_75': np.percentile(results, 75)
        }
