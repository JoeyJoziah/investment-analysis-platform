"""
Discounted Cash Flow (DCF) Model Implementation

Implements DCF valuation for intrinsic value calculation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DCFResult:
    """Result of DCF valuation."""
    intrinsic_value: float
    current_price: float
    upside_potential: float
    margin_of_safety: float
    free_cash_flows: List[float]
    terminal_value: float
    discount_rate: float


class DCFModel:
    """
    Discounted Cash Flow model for intrinsic value calculation.

    Estimates the present value of future cash flows to determine
    the fair value of a stock.
    """

    def __init__(
        self,
        projection_years: int = 5,
        terminal_growth_rate: float = 0.025,
        default_discount_rate: float = 0.10
    ):
        """
        Initialize the DCF model.

        Args:
            projection_years: Number of years to project cash flows
            terminal_growth_rate: Long-term growth rate for terminal value
            default_discount_rate: Default WACC if not calculated
        """
        self.projection_years = projection_years
        self.terminal_growth_rate = terminal_growth_rate
        self.default_discount_rate = default_discount_rate

    def calculate_intrinsic_value(
        self,
        free_cash_flow: float,
        growth_rates: Optional[List[float]] = None,
        discount_rate: Optional[float] = None,
        shares_outstanding: float = 1.0,
        current_price: float = 0.0,
        terminal_growth_rate: Optional[float] = None,
    ) -> DCFResult:
        """
        Calculate intrinsic value using DCF model.

        Args:
            free_cash_flow: Current year free cash flow
            growth_rates: Projected growth rates for each year
            discount_rate: Discount rate (WACC)
            shares_outstanding: Number of shares outstanding
            current_price: Current stock price
            terminal_growth_rate: Override the instance-level
                ``self.terminal_growth_rate`` for this call only.
                Added for F-09-003 so sensitivity_analysis can vary the
                terminal growth rate without mutating shared instance
                state.

        Returns:
            DCFResult with valuation metrics
        """
        if discount_rate is None:
            discount_rate = self.default_discount_rate

        if growth_rates is None:
            # Default declining growth assumption
            growth_rates = [0.15, 0.12, 0.10, 0.08, 0.05]

        # F-09-003: prefer the per-call override; fall back to the
        # instance default. Reading via a local keeps the rest of the
        # method side-effect-free.
        _tgr = (
            terminal_growth_rate
            if terminal_growth_rate is not None
            else self.terminal_growth_rate
        )

        # Project future free cash flows
        projected_fcf = []
        fcf = free_cash_flow

        for i, rate in enumerate(growth_rates[:self.projection_years]):
            fcf = fcf * (1 + rate)
            projected_fcf.append(fcf)

        # Pad with terminal growth if needed
        while len(projected_fcf) < self.projection_years:
            fcf = projected_fcf[-1] * (1 + _tgr)
            projected_fcf.append(fcf)

        # Calculate terminal value (Gordon Growth Model)
        terminal_fcf = projected_fcf[-1] * (1 + _tgr)
        terminal_value = terminal_fcf / (discount_rate - _tgr)

        # Calculate present values
        pv_fcfs = 0
        for i, fcf in enumerate(projected_fcf):
            pv_fcfs += fcf / ((1 + discount_rate) ** (i + 1))

        # Present value of terminal value
        pv_terminal = terminal_value / ((1 + discount_rate) ** self.projection_years)

        # Enterprise value
        enterprise_value = pv_fcfs + pv_terminal

        # Intrinsic value per share
        intrinsic_value = enterprise_value / shares_outstanding

        # Calculate upside potential
        if current_price > 0:
            upside_potential = (intrinsic_value - current_price) / current_price
            margin_of_safety = max(0, (intrinsic_value - current_price) / intrinsic_value)
        else:
            upside_potential = 0.0
            margin_of_safety = 0.0

        return DCFResult(
            intrinsic_value=intrinsic_value,
            current_price=current_price,
            upside_potential=upside_potential,
            margin_of_safety=margin_of_safety,
            free_cash_flows=projected_fcf,
            terminal_value=terminal_value,
            discount_rate=discount_rate
        )

    def calculate_wacc(
        self,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float,
        equity_weight: float,
        debt_weight: float
    ) -> float:
        """
        Calculate Weighted Average Cost of Capital (WACC).

        Args:
            cost_of_equity: Cost of equity capital
            cost_of_debt: Cost of debt capital
            tax_rate: Corporate tax rate
            equity_weight: Proportion of equity in capital structure
            debt_weight: Proportion of debt in capital structure

        Returns:
            WACC as decimal
        """
        after_tax_debt = cost_of_debt * (1 - tax_rate)
        wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_debt)
        return wacc

    def sensitivity_analysis(
        self,
        free_cash_flow: float,
        discount_rates: List[float],
        growth_rates: List[float],
        shares_outstanding: float = 1.0
    ) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis on DCF inputs.

        Args:
            free_cash_flow: Base free cash flow
            discount_rates: List of discount rates to test
            growth_rates: List of terminal growth rates to test
            shares_outstanding: Shares outstanding

        Returns:
            Matrix of intrinsic values for each combination
        """
        results = {
            'discount_rates': discount_rates,
            'growth_rates': growth_rates,
            'values': []
        }

        # F-09-003: previously mutated ``self.terminal_growth_rate``
        # mid-loop which left the instance in a non-deterministic state
        # after sensitivity_analysis returned. Pass ``gr`` per call
        # instead so ``self`` is unchanged on exit.
        for dr in discount_rates:
            row = []
            for gr in growth_rates:
                result = self.calculate_intrinsic_value(
                    free_cash_flow=free_cash_flow,
                    discount_rate=dr,
                    shares_outstanding=shares_outstanding,
                    terminal_growth_rate=gr,
                )
                row.append(result.intrinsic_value)
            results['values'].append(row)

        return results
