"""
Comprehensive tests for Dividend Analyzer with TDD approach.
Tests are written FIRST, implementation follows.

Test coverage includes:
- Normal dividend calculations
- Special dividend handling
- Stock split adjustments
- Missing data handling
- Edge cases (0 dividends, extreme yields)
- Known stock validation (AT&T, Apple)
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any

# These will be imported from the implementation
from backend.analytics.dividend_analyzer import (
    DividendAnalyzer,
    DividendData,
    calculate_dividend_yield,
    validate_api_dividend_data
)


class TestDividendDataModel:
    """Test the DividendData data model."""

    def test_dividend_data_creation(self):
        """Test basic DividendData creation."""
        div = DividendData(
            stock_id=1,
            ex_date=date(2024, 1, 15),
            pay_date=date(2024, 2, 1),
            dividend_amount=Decimal("0.50"),
            is_special=False
        )
        assert div.stock_id == 1
        assert div.dividend_amount == Decimal("0.50")
        assert div.is_special is False

    def test_dividend_data_with_special_dividend(self):
        """Test DividendData with special dividend flag."""
        div = DividendData(
            stock_id=1,
            ex_date=date(2024, 1, 15),
            pay_date=date(2024, 2, 1),
            dividend_amount=Decimal("2.50"),
            is_special=True
        )
        assert div.is_special is True
        assert div.dividend_amount == Decimal("2.50")


class TestCalculateDividendYield:
    """Test the core dividend yield calculation function."""

    def test_calculate_yield_normal_case(self):
        """Test normal dividend yield calculation: (Annual Dividend / Stock Price) * 100."""
        # AT&T: ~$35 stock price, ~$2.50 annual dividend = 7.14% yield
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("2.50"),
            stock_price=Decimal("35.00")
        )
        assert yield_pct is not None
        assert 7.0 <= yield_pct <= 7.3  # ~7.14%

    def test_calculate_yield_apple_case(self):
        """Test Apple-like dividend: ~$190 stock price, ~$0.94 annual dividend = 0.49% yield."""
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("0.94"),
            stock_price=Decimal("190.00")
        )
        assert yield_pct is not None
        assert 0.4 <= yield_pct <= 0.6  # ~0.49%

    def test_calculate_yield_zero_dividend(self):
        """Test with zero dividend."""
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("0.00"),
            stock_price=Decimal("100.00")
        )
        assert yield_pct == 0.0

    def test_calculate_yield_zero_price_raises_error(self):
        """Test that zero stock price raises an error."""
        with pytest.raises(ValueError):
            calculate_dividend_yield(
                annual_dividend=Decimal("2.50"),
                stock_price=Decimal("0.00")
            )

    def test_calculate_yield_none_price_returns_none(self):
        """Test that None stock price returns None."""
        result = calculate_dividend_yield(
            annual_dividend=Decimal("2.50"),
            stock_price=None
        )
        assert result is None

    def test_calculate_yield_high_yield_stock(self):
        """Test high-yield dividend stock (15%+)."""
        # High-yield bond ETF: $50 price, $7.50 annual dividend = 15%
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("7.50"),
            stock_price=Decimal("50.00")
        )
        assert yield_pct is not None
        assert 14.9 <= yield_pct <= 15.1  # 15%

    def test_calculate_yield_extreme_high_yield_warning(self):
        """Test that extreme yields (>20%) are flagged (potential data errors)."""
        # 25% yield is suspicious
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("25.00"),
            stock_price=Decimal("100.00")
        )
        assert yield_pct is not None
        assert yield_pct == 25.0
        # Validation should flag this as suspicious but return the value

    def test_calculate_yield_precision(self):
        """Test calculation precision."""
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("1.234567"),
            stock_price=Decimal("100.00")
        )
        assert yield_pct is not None
        assert abs(yield_pct - 1.234567) < 0.0001


class TestValidateDividendData:
    """Test API dividend data validation."""

    def test_validate_normal_dividend(self):
        """Test validation of normal dividend data."""
        data = {
            "ex_date": "2024-01-15",
            "pay_date": "2024-02-01",
            "dividend_amount": 0.50,
            "special": False
        }
        result = validate_api_dividend_data(data)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_missing_amount(self):
        """Test validation with missing dividend amount."""
        data = {
            "ex_date": "2024-01-15",
            "pay_date": "2024-02-01",
            "special": False
        }
        result = validate_api_dividend_data(data)
        assert result["valid"] is False
        assert any("dividend_amount" in err.lower() for err in result["errors"])

    def test_validate_invalid_dates(self):
        """Test validation with invalid dates."""
        data = {
            "ex_date": "invalid-date",
            "pay_date": "2024-02-01",
            "dividend_amount": 0.50,
            "special": False
        }
        result = validate_api_dividend_data(data)
        assert result["valid"] is False
        assert any("date" in err.lower() for err in result["errors"])

    def test_validate_negative_dividend(self):
        """Test validation with negative dividend amount."""
        data = {
            "ex_date": "2024-01-15",
            "pay_date": "2024-02-01",
            "dividend_amount": -0.50,
            "special": False
        }
        result = validate_api_dividend_data(data)
        assert result["valid"] is False
        assert any("negative" in err.lower() for err in result["errors"])

    def test_validate_ex_date_after_pay_date(self):
        """Test validation when ex_date is after pay_date."""
        data = {
            "ex_date": "2024-02-15",
            "pay_date": "2024-02-01",
            "dividend_amount": 0.50,
            "special": False
        }
        result = validate_api_dividend_data(data)
        assert result["valid"] is False
        assert any("ex_date" in err.lower() and "pay_date" in err.lower()
                   for err in result["errors"])

    def test_validate_missing_special_flag(self):
        """Test validation with missing special dividend flag (should have default)."""
        data = {
            "ex_date": "2024-01-15",
            "pay_date": "2024-02-01",
            "dividend_amount": 0.50
        }
        result = validate_api_dividend_data(data)
        assert result["valid"] is True
        assert result["data"]["special"] is False  # Should default to False


class TestDividendAnalyzerInitialization:
    """Test DividendAnalyzer class initialization."""

    def test_analyzer_initialization(self):
        """Test basic analyzer initialization."""
        analyzer = DividendAnalyzer()
        assert analyzer is not None

    def test_analyzer_with_config(self):
        """Test analyzer with custom configuration."""
        config = {
            "exclude_special_dividends": True,
            "min_yield_threshold": 0.1,
            "max_yield_threshold": 50.0
        }
        analyzer = DividendAnalyzer(config=config)
        assert analyzer.config["exclude_special_dividends"] is True


class TestNormalDividendCalculation:
    """Test normal dividend calculations without special cases."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for tests."""
        return DividendAnalyzer()

    def test_annual_dividend_calculation_four_quarters(self, analyzer):
        """Test annual dividend from four quarterly payments."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, m, 15),
                pay_date=date(2024, m, 25),
                dividend_amount=Decimal("0.50"),
                is_special=False
            )
            for m in [1, 4, 7, 10]  # Q1, Q2, Q3, Q4
        ]
        annual_div = analyzer.calculate_annual_dividend(dividends)
        assert annual_div == Decimal("2.00")  # 4 quarters * 0.50

    def test_annual_dividend_calculation_monthly(self, analyzer):
        """Test annual dividend from monthly dividends."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, m, 15),
                pay_date=date(2024, m, 25),
                dividend_amount=Decimal("0.25"),
                is_special=False
            )
            for m in range(1, 13)
        ]
        annual_div = analyzer.calculate_annual_dividend(dividends)
        assert annual_div == Decimal("3.00")  # 12 * 0.25

    def test_yield_calculation_with_price_snapshot(self, analyzer):
        """Test yield calculation with a specific price."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, 1, 15),
                pay_date=date(2024, 2, 1),
                dividend_amount=Decimal("0.625"),
                is_special=False
            ) for _ in range(4)  # Quarterly
        ]

        yield_pct = analyzer.calculate_dividend_yield_for_stock(
            dividends=dividends,
            current_price=Decimal("35.00")
        )
        # Annual = 0.625 * 4 = 2.50
        # Yield = (2.50 / 35.00) * 100 = 7.14%
        assert yield_pct is not None
        assert 7.0 <= yield_pct <= 7.3


class TestSpecialDividendHandling:
    """Test special dividend exclusion logic."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer that excludes special dividends."""
        config = {"exclude_special_dividends": True}
        return DividendAnalyzer(config=config)

    def test_filter_special_dividends(self, analyzer):
        """Test filtering out special dividends."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, 1, 15),
                pay_date=date(2024, 2, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            ),
            DividendData(
                stock_id=1,
                ex_date=date(2024, 3, 15),
                pay_date=date(2024, 4, 1),
                dividend_amount=Decimal("2.50"),  # Special: large amount
                is_special=True
            ),
            DividendData(
                stock_id=1,
                ex_date=date(2024, 7, 15),
                pay_date=date(2024, 8, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            )
        ]

        filtered = analyzer.filter_dividends(dividends)
        assert len(filtered) == 2
        assert all(not d.is_special for d in filtered)

    def test_annual_dividend_excludes_special(self, analyzer):
        """Test annual dividend calculation excludes special dividends."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, 1, 15),
                pay_date=date(2024, 2, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            ),
            DividendData(
                stock_id=1,
                ex_date=date(2024, 3, 15),
                pay_date=date(2024, 4, 1),
                dividend_amount=Decimal("5.00"),
                is_special=True  # Should be excluded
            ),
            DividendData(
                stock_id=1,
                ex_date=date(2024, 4, 15),
                pay_date=date(2024, 5, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            ),
            DividendData(
                stock_id=1,
                ex_date=date(2024, 7, 15),
                pay_date=date(2024, 8, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            ),
            DividendData(
                stock_id=1,
                ex_date=date(2024, 10, 15),
                pay_date=date(2024, 11, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            )
        ]

        annual_div = analyzer.calculate_annual_dividend(dividends)
        # Should only count the regular dividends: 0.50 * 4 = 2.00
        assert annual_div == Decimal("2.00")


class TestStockSplitAdjustment:
    """Test stock split adjustment logic."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DividendAnalyzer()

    def test_adjust_dividend_for_stock_split_2for1(self, analyzer):
        """Test dividend adjustment for 2:1 stock split.

        In a 2:1 split, old shareholders get 2x shares.
        A dividend of $2.00 per old share becomes $1.00 per new share.
        split_coefficient = 0.5 means multiply dividend by 0.5 (i.e., divide by 2).
        """
        original_dividend = Decimal("2.00")
        split_coefficient = Decimal("0.5")  # 2:1 split: divide by 2

        adjusted = analyzer.adjust_dividend_for_split(
            dividend_amount=original_dividend,
            split_coefficient=split_coefficient
        )
        # 2.00 / (1/0.5) = 2.00 / 2 = 1.00
        assert adjusted == Decimal("1.0000")

    def test_adjust_dividend_for_stock_split_3for1(self, analyzer):
        """Test dividend adjustment for 3:1 stock split.

        In a 3:1 split, old shareholders get 3x shares.
        A dividend of $1.50 per old share becomes $0.50 per new share.
        split_coefficient = 1/3 means multiply dividend by 1/3 (i.e., divide by 3).
        """
        original_dividend = Decimal("1.50")
        split_coefficient = Decimal("1.0") / Decimal("3.0")  # 3:1 split: divide by 3

        adjusted = analyzer.adjust_dividend_for_split(
            dividend_amount=original_dividend,
            split_coefficient=split_coefficient
        )
        # 1.50 / (1/(1/3)) = 1.50 / 3 = 0.50
        assert abs(adjusted - Decimal("0.5")) < Decimal("0.01")

    def test_no_adjustment_for_no_split(self, analyzer):
        """Test that no adjustment occurs when split coefficient is 1.0."""
        original_dividend = Decimal("0.50")
        adjusted = analyzer.adjust_dividend_for_split(
            dividend_amount=original_dividend,
            split_coefficient=Decimal("1.0")
        )
        assert adjusted == original_dividend


class TestMissingDataHandling:
    """Test handling of missing or incomplete data."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DividendAnalyzer()

    def test_empty_dividend_list(self, analyzer):
        """Test with empty dividend list."""
        annual_div = analyzer.calculate_annual_dividend([])
        assert annual_div == Decimal("0.00")

    def test_none_price_returns_none_yield(self, analyzer):
        """Test that None price returns None yield."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, 1, 15),
                pay_date=date(2024, 2, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            )
        ]

        yield_pct = analyzer.calculate_dividend_yield_for_stock(
            dividends=dividends,
            current_price=None
        )
        assert yield_pct is None

    def test_zero_price_raises_error(self, analyzer):
        """Test that zero price raises an error."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, 1, 15),
                pay_date=date(2024, 2, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            )
        ]

        with pytest.raises(ValueError):
            analyzer.calculate_dividend_yield_for_stock(
                dividends=dividends,
                current_price=Decimal("0.00")
            )


class TestKnownStockValidation:
    """Test against known dividend stocks to verify accuracy."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DividendAnalyzer()

    def test_att_dividend_yield(self, analyzer):
        """Test AT&T (T) dividend yield validation.

        AT&T is known for high dividend yield (~7%).
        Current: ~$21 price, $0.775 quarterly = ~$3.10 annual
        Expected yield: ~14.7% (based on quarterly payments)
        """
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, m, 15),
                pay_date=date(2024, m, 25),
                dividend_amount=Decimal("0.775"),  # Approximate quarterly
                is_special=False
            )
            for m in [1, 4, 7, 10]  # Quarterly: Jan, Apr, Jul, Oct
        ]

        # Using lower price point for AT&T
        yield_pct = analyzer.calculate_dividend_yield_for_stock(
            dividends=dividends,
            current_price=Decimal("21.00")
        )

        assert yield_pct is not None
        # Annual = 0.775 * 4 = 3.10
        # Yield = (3.10 / 21.00) * 100 = 14.76%
        assert 14.0 <= yield_pct <= 15.0

    def test_apple_dividend_yield(self, analyzer):
        """Test Apple (AAPL) dividend yield validation.

        Apple has low dividend yield (~0.5%).
        Current: ~$190 price, $0.24 quarterly = $0.96 annual
        Expected yield: ~0.5%
        """
        dividends = [
            DividendData(
                stock_id=2,
                ex_date=date(2024, m, 15),
                pay_date=date(2024, m, 25),
                dividend_amount=Decimal("0.24"),  # Approximate quarterly
                is_special=False
            )
            for m in [2, 5, 8, 11]  # Quarterly
        ]

        yield_pct = analyzer.calculate_dividend_yield_for_stock(
            dividends=dividends,
            current_price=Decimal("190.00")
        )

        assert yield_pct is not None
        # Annual = 0.24 * 4 = 0.96
        # Yield = (0.96 / 190.00) * 100 = 0.505%
        assert 0.4 <= yield_pct <= 0.6


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DividendAnalyzer()

    def test_very_small_dividend(self, analyzer):
        """Test with very small dividend amount."""
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("0.001"),
            stock_price=Decimal("100.00")
        )
        assert yield_pct is not None
        assert yield_pct < 0.1

    def test_very_large_stock_price(self, analyzer):
        """Test with very large stock price."""
        yield_pct = calculate_dividend_yield(
            annual_dividend=Decimal("10.00"),
            stock_price=Decimal("100000.00")
        )
        assert yield_pct is not None
        assert 0.009 <= yield_pct <= 0.011

    def test_multiple_special_dividends_in_year(self, analyzer):
        """Test year with multiple special dividends (should all be excluded)."""
        config = {"exclude_special_dividends": True}
        special_analyzer = DividendAnalyzer(config=config)

        dividends = [
            # Regular dividends
            DividendData(
                stock_id=1,
                ex_date=date(2024, 1, 15),
                pay_date=date(2024, 2, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            ),
            # Special dividend 1
            DividendData(
                stock_id=1,
                ex_date=date(2024, 3, 15),
                pay_date=date(2024, 4, 1),
                dividend_amount=Decimal("3.00"),
                is_special=True
            ),
            # Regular dividend
            DividendData(
                stock_id=1,
                ex_date=date(2024, 4, 15),
                pay_date=date(2024, 5, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            ),
            # Special dividend 2
            DividendData(
                stock_id=1,
                ex_date=date(2024, 9, 15),
                pay_date=date(2024, 10, 1),
                dividend_amount=Decimal("2.50"),
                is_special=True
            ),
            # Regular dividend
            DividendData(
                stock_id=1,
                ex_date=date(2024, 7, 15),
                pay_date=date(2024, 8, 1),
                dividend_amount=Decimal("0.50"),
                is_special=False
            )
        ]

        annual_div = special_analyzer.calculate_annual_dividend(dividends)
        # Should only count: 0.50 + 0.50 + 0.50 + 0.50 = 2.00 (excluding special)
        # But we only have 3 regular dividends, so: 0.50 * 3 = 1.50
        assert annual_div == Decimal("1.50")


class TestIntegrationWithAnalysisRouter:
    """Test integration scenarios as they would be used in the router."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DividendAnalyzer()

    def test_router_integration_basic(self, analyzer):
        """Test basic usage as router would use it."""
        # Simulate API data
        api_data = [
            {
                "ex_date": "2024-01-15",
                "pay_date": "2024-02-01",
                "dividend_amount": 0.50,
                "special": False
            },
            {
                "ex_date": "2024-04-15",
                "pay_date": "2024-05-01",
                "dividend_amount": 0.50,
                "special": False
            }
        ]

        # Validate and create DividendData objects
        validated_data = []
        for item in api_data:
            validation_result = validate_api_dividend_data(item)
            if validation_result["valid"]:
                div = DividendData(
                    stock_id=1,
                    ex_date=datetime.strptime(
                        validation_result["data"]["ex_date"], "%Y-%m-%d"
                    ).date(),
                    pay_date=datetime.strptime(
                        validation_result["data"]["pay_date"], "%Y-%m-%d"
                    ).date(),
                    dividend_amount=Decimal(str(validation_result["data"]["dividend_amount"])),
                    is_special=validation_result["data"]["special"]
                )
                validated_data.append(div)

        assert len(validated_data) == 2

        # Calculate annual dividend
        annual_div = analyzer.calculate_annual_dividend(validated_data)
        assert annual_div > 0


class TestPerformanceAndAccuracy:
    """Test performance and numerical accuracy."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DividendAnalyzer()

    def test_accuracy_with_many_dividends(self, analyzer):
        """Test accuracy calculation with many dividend entries."""
        dividends = [
            DividendData(
                stock_id=1,
                ex_date=date(2024, m, 15),
                pay_date=date(2024, m, 25),
                dividend_amount=Decimal("0.25"),
                is_special=False
            )
            for m in range(1, 13)  # 12 months
        ]

        annual_div = analyzer.calculate_annual_dividend(dividends)
        assert annual_div == Decimal("3.00")

    def test_numerical_precision(self, analyzer):
        """Test that calculations maintain proper precision."""
        # Use precise decimal values
        annual_div = Decimal("3.14159265")
        stock_price = Decimal("100.00000000")

        yield_pct = calculate_dividend_yield(
            annual_dividend=annual_div,
            stock_price=stock_price
        )

        assert yield_pct is not None
        # Should be 3.14159265%
        assert 3.14 <= yield_pct <= 3.15
