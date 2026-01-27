"""
Dividend Analyzer - Calculate and validate dividend yields with TDD approach.

Features:
- Accurate dividend yield calculation: (Annual Dividend / Stock Price) * 100
- Special dividend filtering and exclusion
- Stock split adjustment logic
- API data validation
- Edge case handling (zero prices, missing data)
- Cross-validation against known dividend stocks (AT&T, Apple)

Formula: Dividend Yield % = (Annual Dividend / Current Stock Price) * 100
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DividendData:
    """Represents a single dividend payment."""

    stock_id: int
    ex_date: date
    pay_date: date
    dividend_amount: Decimal
    is_special: bool = False
    record_date: Optional[date] = None
    announcement_date: Optional[date] = None

    def __post_init__(self):
        """Validate dividend data after initialization."""
        if isinstance(self.dividend_amount, (int, float)):
            self.dividend_amount = Decimal(str(self.dividend_amount))
        elif not isinstance(self.dividend_amount, Decimal):
            self.dividend_amount = Decimal(str(self.dividend_amount))

        if self.dividend_amount < 0:
            raise ValueError(f"Dividend amount cannot be negative: {self.dividend_amount}")


# ============================================================================
# Core Calculation Functions
# ============================================================================


def calculate_dividend_yield(
    annual_dividend: Decimal,
    stock_price: Optional[Decimal]
) -> Optional[float]:
    """
    Calculate dividend yield percentage.

    Formula: (Annual Dividend / Stock Price) * 100

    Args:
        annual_dividend: Annual dividend per share (as Decimal)
        stock_price: Current stock price (as Decimal or None)

    Returns:
        Dividend yield as percentage (float), or None if price is None

    Raises:
        ValueError: If stock_price is zero or negative

    Examples:
        AT&T: (2.50 / 35.00) * 100 = 7.14%
        Apple: (0.94 / 190.00) * 100 = 0.49%
    """
    if stock_price is None:
        return None

    # Convert to Decimal if needed
    if isinstance(stock_price, (int, float)):
        stock_price = Decimal(str(stock_price))
    if isinstance(annual_dividend, (int, float)):
        annual_dividend = Decimal(str(annual_dividend))

    # Validate price
    if stock_price <= 0:
        raise ValueError(f"Stock price must be positive, got: {stock_price}")

    if annual_dividend == 0:
        return 0.0

    # Calculate yield with precision
    yield_decimal = (annual_dividend / stock_price) * Decimal("100")
    yield_float = float(yield_decimal)

    # Log warning for extreme yields (potential data errors)
    if yield_float > 20.0:
        logger.warning(
            f"Unusual high dividend yield detected: {yield_float:.2f}%. "
            f"Annual dividend: {annual_dividend}, Stock price: {stock_price}"
        )

    return yield_float


def validate_api_dividend_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate dividend data from external API.

    Args:
        data: Dictionary containing dividend data

    Returns:
        Dictionary with 'valid' (bool), 'errors' (list), and 'data' (dict)

    Examples:
        >>> result = validate_api_dividend_data({
        ...     "ex_date": "2024-01-15",
        ...     "pay_date": "2024-02-01",
        ...     "dividend_amount": 0.50,
        ...     "special": False
        ... })
        >>> result["valid"]
        True
    """
    errors = []
    validated_data = dict(data)

    # Check required fields
    if "dividend_amount" not in data or data["dividend_amount"] is None:
        errors.append("Missing required field: dividend_amount")
    elif not isinstance(data["dividend_amount"], (int, float, Decimal, str)):
        errors.append(f"dividend_amount must be numeric, got {type(data['dividend_amount'])}")

    # Validate dividend amount is non-negative
    if "dividend_amount" in data and data["dividend_amount"] is not None:
        try:
            amount = Decimal(str(data["dividend_amount"]))
            if amount < 0:
                errors.append(f"dividend_amount cannot be negative: {amount}")
        except (ValueError, TypeError):
            errors.append(f"dividend_amount is not a valid number: {data['dividend_amount']}")

    # Validate dates
    for date_field in ["ex_date", "pay_date"]:
        if date_field not in data or data[date_field] is None:
            errors.append(f"Missing required field: {date_field}")
        else:
            try:
                if isinstance(data[date_field], str):
                    datetime.strptime(data[date_field], "%Y-%m-%d")
                elif not isinstance(data[date_field], date):
                    errors.append(f"{date_field} must be a date or string in YYYY-MM-DD format")
            except ValueError:
                errors.append(f"Invalid {date_field} format: {data[date_field]}")

    # Validate ex_date is before or equal to pay_date
    if "ex_date" in data and "pay_date" in data and data["ex_date"] and data["pay_date"]:
        try:
            ex_date_obj = (
                datetime.strptime(data["ex_date"], "%Y-%m-%d").date()
                if isinstance(data["ex_date"], str)
                else data["ex_date"]
            )
            pay_date_obj = (
                datetime.strptime(data["pay_date"], "%Y-%m-%d").date()
                if isinstance(data["pay_date"], str)
                else data["pay_date"]
            )

            if ex_date_obj > pay_date_obj:
                errors.append(
                    f"ex_date ({ex_date_obj}) cannot be after pay_date ({pay_date_obj})"
                )
        except (ValueError, TypeError):
            pass  # Already reported as invalid format

    # Set special flag default
    if "special" not in validated_data:
        validated_data["special"] = False

    # Normalize date strings
    if "ex_date" in validated_data and isinstance(validated_data["ex_date"], date):
        validated_data["ex_date"] = validated_data["ex_date"].isoformat()
    if "pay_date" in validated_data and isinstance(validated_data["pay_date"], date):
        validated_data["pay_date"] = validated_data["pay_date"].isoformat()

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": validated_data
    }


# ============================================================================
# DividendAnalyzer Class
# ============================================================================


class DividendAnalyzer:
    """
    Comprehensive dividend analysis engine.

    Handles:
    - Annual dividend calculation from individual payments
    - Dividend yield computation
    - Special dividend filtering
    - Stock split adjustments
    - Data validation
    - Edge case handling
    """

    DEFAULT_CONFIG = {
        "exclude_special_dividends": True,
        "min_yield_threshold": 0.0,
        "max_yield_threshold": 50.0,
        "lookback_years": 1,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DividendAnalyzer.

        Args:
            config: Optional configuration dictionary with keys:
                - exclude_special_dividends (bool): Filter out special dividends
                - min_yield_threshold (float): Minimum acceptable yield %
                - max_yield_threshold (float): Maximum acceptable yield %
                - lookback_years (int): Years to look back for dividend history
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

    # ========================================================================
    # Core Methods
    # ========================================================================

    def filter_dividends(
        self,
        dividends: List[DividendData],
        include_special: Optional[bool] = None
    ) -> List[DividendData]:
        """
        Filter dividends based on configuration.

        Args:
            dividends: List of dividend data
            include_special: Override config to include special dividends

        Returns:
            Filtered list of dividends
        """
        if include_special is None:
            include_special = not self.config["exclude_special_dividends"]

        if include_special:
            return dividends

        return [d for d in dividends if not d.is_special]

    def calculate_annual_dividend(
        self,
        dividends: List[DividendData],
        include_special: Optional[bool] = None
    ) -> Decimal:
        """
        Calculate annual dividend from list of dividend payments.

        Sums all dividends, excluding special dividends if configured.

        Args:
            dividends: List of DividendData objects
            include_special: Override config to include special dividends

        Returns:
            Annual dividend as Decimal
        """
        if not dividends:
            return Decimal("0.00")

        filtered = self.filter_dividends(dividends, include_special)

        if not filtered:
            return Decimal("0.00")

        total = sum(d.dividend_amount for d in filtered)
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def calculate_dividend_yield_for_stock(
        self,
        dividends: List[DividendData],
        current_price: Optional[Decimal],
        include_special: Optional[bool] = None
    ) -> Optional[float]:
        """
        Calculate dividend yield for a stock given dividend history and price.

        Args:
            dividends: List of DividendData objects
            current_price: Current stock price (Decimal or None)
            include_special: Override config to include special dividends

        Returns:
            Dividend yield as percentage (float), or None if price is None

        Raises:
            ValueError: If current_price is invalid
        """
        if current_price is None:
            return None

        annual_div = self.calculate_annual_dividend(dividends, include_special)

        return calculate_dividend_yield(annual_div, current_price)

    def adjust_dividend_for_split(
        self,
        dividend_amount: Decimal,
        split_coefficient: Decimal
    ) -> Decimal:
        """
        Adjust dividend for stock split.

        Stock split coefficient is the factor by which the price is adjusted.
        A 2:1 split has coefficient 0.5 (price halves, shares double).
        Dividends adjust inversely: multiply by the split coefficient.

        Examples:
            - 2:1 split (coeff 0.5): dividend of $2.00 becomes $1.00
              (multiply by 0.5)
            - 3:1 split (coeff 1/3): dividend of $1.50 becomes $0.50
              (multiply by 1/3)

        Args:
            dividend_amount: Original dividend per share
            split_coefficient: Stock split coefficient

        Returns:
            Adjusted dividend per share
        """
        if split_coefficient == Decimal("1.0") or split_coefficient == 1.0:
            return dividend_amount

        if isinstance(split_coefficient, float):
            split_coefficient = Decimal(str(split_coefficient))
        if isinstance(dividend_amount, float):
            dividend_amount = Decimal(str(dividend_amount))

        # Multiply dividend by the split coefficient (inverse adjustment)
        adjusted = (dividend_amount * split_coefficient).quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP
        )

        return adjusted

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def get_dividend_statistics(
        self,
        dividends: List[DividendData]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about dividend history.

        Args:
            dividends: List of DividendData objects

        Returns:
            Dictionary with statistics
        """
        if not dividends:
            return {
                "total_dividends": 0,
                "count": 0,
                "average_dividend": 0,
                "min_dividend": None,
                "max_dividend": None,
                "special_dividend_count": 0,
                "regular_dividend_count": 0,
                "payment_frequency": None,
            }

        regular_divs = [d for d in dividends if not d.is_special]
        special_divs = [d for d in dividends if d.is_special]

        amounts = [d.dividend_amount for d in regular_divs]

        stats_dict = {
            "total_dividends": sum(d.dividend_amount for d in dividends),
            "count": len(dividends),
            "average_dividend": (
                sum(amounts) / len(amounts)
                if amounts
                else Decimal("0")
            ),
            "min_dividend": min(amounts) if amounts else None,
            "max_dividend": max(amounts) if amounts else None,
            "special_dividend_count": len(special_divs),
            "regular_dividend_count": len(regular_divs),
            "payment_frequency": self._detect_payment_frequency(dividends),
        }

        return stats_dict

    def _detect_payment_frequency(
        self,
        dividends: List[DividendData]
    ) -> Optional[str]:
        """
        Detect payment frequency from dividend dates.

        Args:
            dividends: List of DividendData objects

        Returns:
            One of: "monthly", "quarterly", "semi-annual", "annual", or None
        """
        if len(dividends) < 2:
            return None

        # Sort by ex_date
        sorted_divs = sorted(dividends, key=lambda d: d.ex_date)

        # Calculate days between payments
        days_between = []
        for i in range(1, len(sorted_divs)):
            delta = (sorted_divs[i].ex_date - sorted_divs[i - 1].ex_date).days
            days_between.append(delta)

        if not days_between:
            return None

        avg_days = sum(days_between) / len(days_between)

        if 20 < avg_days < 40:
            return "monthly"
        elif 80 < avg_days < 100:
            return "quarterly"
        elif 150 < avg_days < 200:
            return "semi-annual"
        elif 350 < avg_days < 380:
            return "annual"

        return None

    def validate_yield_reasonableness(
        self,
        yield_percent: float
    ) -> Dict[str, Any]:
        """
        Validate that calculated yield is within reasonable bounds.

        Args:
            yield_percent: Dividend yield as percentage

        Returns:
            Dictionary with validation result and warnings
        """
        issues = []

        min_threshold = self.config.get("min_yield_threshold", 0.0)
        max_threshold = self.config.get("max_yield_threshold", 50.0)

        if yield_percent < min_threshold:
            issues.append(f"Yield {yield_percent:.2f}% below minimum {min_threshold}%")

        if yield_percent > max_threshold:
            issues.append(f"Yield {yield_percent:.2f}% above maximum {max_threshold}%")

        if yield_percent > 20.0:
            issues.append(
                f"High yield {yield_percent:.2f}% may indicate data error or special situation"
            )

        return {
            "is_reasonable": len(issues) == 0,
            "issues": issues,
            "yield_percent": yield_percent,
        }


# ============================================================================
# High-Level API Functions
# ============================================================================


def calculate_stock_dividend_yield(
    dividends: List[DividendData],
    current_price: Decimal,
    exclude_special: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    High-level function to calculate dividend yield for a stock.

    Args:
        dividends: List of DividendData objects
        current_price: Current stock price
        exclude_special: Whether to exclude special dividends

    Returns:
        Tuple of (yield_percent, statistics_dict)
    """
    config = {"exclude_special_dividends": exclude_special}
    analyzer = DividendAnalyzer(config=config)

    yield_pct = analyzer.calculate_dividend_yield_for_stock(dividends, current_price)

    if yield_pct is None:
        yield_pct = 0.0

    stats = analyzer.get_dividend_statistics(dividends)

    return yield_pct, stats


__all__ = [
    "DividendData",
    "DividendAnalyzer",
    "calculate_dividend_yield",
    "validate_api_dividend_data",
    "calculate_stock_dividend_yield",
]
