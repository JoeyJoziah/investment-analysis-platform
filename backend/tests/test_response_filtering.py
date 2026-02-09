"""
Tests for Response Field Filtering

Tests for API response field filtering utilities and endpoint integration.

Created: 2026-02-08
Part of: Issue #12 - API Response Time Optimization
"""

import pytest
from typing import Dict, List, Any

from backend.utils.response_utils import (
    filter_response_fields,
    parse_field_list,
    validate_fields,
    _filter_dict_fields
)


class TestFilterResponseFields:
    """Tests for filter_response_fields function"""

    def test_filter_single_object_basic_fields(self):
        """Test filtering basic fields from a single object"""
        data = {
            "ticker": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "change": 2.5,
            "change_percent": 1.7
        }

        result = filter_response_fields(data, "ticker,price")

        assert result == {"ticker": "AAPL", "price": 150.0}

    def test_filter_list_of_objects(self):
        """Test filtering fields from a list of objects"""
        data = [
            {"ticker": "AAPL", "price": 150.0, "volume": 1000000},
            {"ticker": "GOOGL", "price": 2800.0, "volume": 500000},
            {"ticker": "MSFT", "price": 320.0, "volume": 750000}
        ]

        result = filter_response_fields(data, "ticker,price")

        assert len(result) == 3
        assert result[0] == {"ticker": "AAPL", "price": 150.0}
        assert result[1] == {"ticker": "GOOGL", "price": 2800.0}
        assert result[2] == {"ticker": "MSFT", "price": 320.0}

    def test_filter_no_fields_returns_full_response(self):
        """Test that no fields parameter returns full response"""
        data = {"ticker": "AAPL", "price": 150.0, "volume": 1000000}

        result = filter_response_fields(data, None)
        assert result == data

        result = filter_response_fields(data, "")
        assert result == data

        result = filter_response_fields(data, "   ")
        assert result == data

    def test_filter_invalid_fields_ignored(self):
        """Test that invalid field names are ignored silently"""
        data = {"ticker": "AAPL", "price": 150.0, "volume": 1000000}

        result = filter_response_fields(data, "ticker,invalid_field,price")

        # Should only include valid fields
        assert result == {"ticker": "AAPL", "price": 150.0}

    def test_filter_empty_fields_param(self):
        """Test that empty fields param returns full response"""
        data = {"ticker": "AAPL", "price": 150.0}

        result = filter_response_fields(data, "")
        assert result == data

        result = filter_response_fields(data, ",,,")
        assert result == data

    def test_filter_whitespace_handling(self):
        """Test that whitespace in field names is handled correctly"""
        data = {"ticker": "AAPL", "price": 150.0, "volume": 1000000}

        result = filter_response_fields(data, "  ticker  , price  ,  volume  ")

        assert result == data

    def test_filter_nested_object_basic(self):
        """Test filtering nested object fields"""
        data = {
            "ticker": "AAPL",
            "price": 150.0,
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30
            }
        }

        result = filter_response_fields(data, "ticker,user.name,user.email")

        assert result == {
            "ticker": "AAPL",
            "user": {
                "name": "John Doe",
                "email": "john@example.com"
            }
        }

    def test_filter_nested_object_deep(self):
        """Test filtering deeply nested fields"""
        data = {
            "ticker": "AAPL",
            "company": {
                "name": "Apple Inc.",
                "details": {
                    "ceo": "Tim Cook",
                    "founded": 1976,
                    "hq": "Cupertino, CA"
                }
            }
        }

        result = filter_response_fields(data, "ticker,company.details.ceo")

        assert result == {
            "ticker": "AAPL",
            "company": {
                "details": {
                    "ceo": "Tim Cook"
                }
            }
        }

    def test_filter_list_of_nested_objects(self):
        """Test filtering list of objects with nested fields"""
        data = [
            {
                "ticker": "AAPL",
                "user": {"name": "John", "email": "john@example.com"}
            },
            {
                "ticker": "GOOGL",
                "user": {"name": "Jane", "email": "jane@example.com"}
            }
        ]

        result = filter_response_fields(data, "ticker,user.name")

        assert len(result) == 2
        assert result[0] == {"ticker": "AAPL", "user": {"name": "John"}}
        assert result[1] == {"ticker": "GOOGL", "user": {"name": "Jane"}}

    def test_filter_nested_list_of_objects(self):
        """Test filtering nested list of objects"""
        data = {
            "ticker": "AAPL",
            "holders": [
                {"name": "John", "shares": 100},
                {"name": "Jane", "shares": 200}
            ]
        }

        result = filter_response_fields(data, "ticker,holders.name")

        assert result == {
            "ticker": "AAPL",
            "holders": [
                {"name": "John"},
                {"name": "Jane"}
            ]
        }

    def test_filter_nonexistent_nested_field(self):
        """Test filtering nonexistent nested fields"""
        data = {
            "ticker": "AAPL",
            "price": 150.0
        }

        result = filter_response_fields(data, "ticker,user.name")

        # Should only include ticker since user doesn't exist
        assert result == {"ticker": "AAPL"}

    def test_filter_mixed_nested_and_direct_fields(self):
        """Test filtering mix of nested and direct fields"""
        data = {
            "ticker": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "user": {
                "name": "John",
                "email": "john@example.com"
            }
        }

        result = filter_response_fields(data, "ticker,price,user.name")

        assert result == {
            "ticker": "AAPL",
            "price": 150.0,
            "user": {"name": "John"}
        }

    def test_filter_preserves_data_types(self):
        """Test that filtering preserves data types"""
        data = {
            "ticker": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "is_active": True,
            "metadata": None
        }

        result = filter_response_fields(data, "ticker,price,volume,is_active,metadata")

        assert isinstance(result["ticker"], str)
        assert isinstance(result["price"], float)
        assert isinstance(result["volume"], int)
        assert isinstance(result["is_active"], bool)
        assert result["metadata"] is None

    def test_filter_non_dict_data_returns_unchanged(self):
        """Test that non-dict/non-list data is returned unchanged"""
        data = "simple string"
        result = filter_response_fields(data, "field1,field2")
        assert result == data

        data = 12345
        result = filter_response_fields(data, "field1,field2")
        assert result == data


class TestParseFieldList:
    """Tests for parse_field_list function"""

    def test_parse_basic_field_list(self):
        """Test parsing basic comma-separated fields"""
        result = parse_field_list("ticker,price,volume")
        assert result == ["ticker", "price", "volume"]

    def test_parse_with_whitespace(self):
        """Test parsing with whitespace"""
        result = parse_field_list("  ticker , price  , volume  ")
        assert result == ["ticker", "price", "volume"]

    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_field_list("")
        assert result == []

        result = parse_field_list("   ")
        assert result == []

    def test_parse_none(self):
        """Test parsing None"""
        result = parse_field_list(None)
        assert result == []

    def test_parse_single_field(self):
        """Test parsing single field"""
        result = parse_field_list("ticker")
        assert result == ["ticker"]

    def test_parse_nested_fields(self):
        """Test parsing nested field notation"""
        result = parse_field_list("ticker,user.name,user.email")
        assert result == ["ticker", "user.name", "user.email"]

    def test_parse_removes_empty_fields(self):
        """Test that empty fields are removed"""
        result = parse_field_list("ticker,,price,,")
        assert result == ["ticker", "price"]


class TestValidateFields:
    """Tests for validate_fields function"""

    def test_validate_all_valid_fields(self):
        """Test validation with all valid fields"""
        allowed = {"ticker", "price", "volume", "change"}
        is_valid, error = validate_fields("ticker,price", allowed)

        assert is_valid is True
        assert error is None

    def test_validate_invalid_fields(self):
        """Test validation with invalid fields"""
        allowed = {"ticker", "price", "volume"}
        is_valid, error = validate_fields("ticker,invalid,price", allowed)

        assert is_valid is False
        assert "invalid" in error.lower()

    def test_validate_multiple_invalid_fields(self):
        """Test validation with multiple invalid fields"""
        allowed = {"ticker", "price"}
        is_valid, error = validate_fields("ticker,invalid1,invalid2", allowed)

        assert is_valid is False
        assert "invalid1" in error
        assert "invalid2" in error

    def test_validate_none_fields(self):
        """Test validation with None fields"""
        allowed = {"ticker", "price"}
        is_valid, error = validate_fields(None, allowed)

        assert is_valid is True
        assert error is None

    def test_validate_empty_fields(self):
        """Test validation with empty fields"""
        allowed = {"ticker", "price"}
        is_valid, error = validate_fields("", allowed)

        assert is_valid is True
        assert error is None


class TestFilterDictFields:
    """Tests for _filter_dict_fields internal function"""

    def test_filter_basic_fields(self):
        """Test filtering basic fields"""
        data = {"a": 1, "b": 2, "c": 3}
        result = _filter_dict_fields(data, ["a", "b"])

        assert result == {"a": 1, "b": 2}

    def test_filter_nested_single_level(self):
        """Test filtering nested fields (single level)"""
        data = {
            "ticker": "AAPL",
            "user": {"name": "John", "age": 30}
        }
        result = _filter_dict_fields(data, ["ticker", "user.name"])

        assert result == {
            "ticker": "AAPL",
            "user": {"name": "John"}
        }

    def test_filter_multiple_nested_same_parent(self):
        """Test filtering multiple fields from same nested parent"""
        data = {
            "ticker": "AAPL",
            "user": {"name": "John", "email": "john@example.com", "age": 30}
        }
        result = _filter_dict_fields(data, ["ticker", "user.name", "user.email"])

        assert result == {
            "ticker": "AAPL",
            "user": {
                "name": "John",
                "email": "john@example.com"
            }
        }

    def test_filter_empty_result(self):
        """Test filtering with no matching fields"""
        data = {"ticker": "AAPL", "price": 150.0}
        result = _filter_dict_fields(data, ["nonexistent"])

        assert result == {}


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""

    def test_stock_quote_filtering(self):
        """Test filtering stock quote response"""
        quote = {
            "symbol": "AAPL",
            "price": 150.0,
            "change": 2.5,
            "change_percent": 1.7,
            "volume": 1000000,
            "open": 148.0,
            "high": 151.0,
            "low": 147.5,
            "previous_close": 147.5,
            "market_cap": 2500000000,
            "pe_ratio": 25.3,
            "timestamp": "2026-02-08T12:00:00Z"
        }

        # Client only wants ticker, price, and change
        result = filter_response_fields(quote, "symbol,price,change")

        assert result == {
            "symbol": "AAPL",
            "price": 150.0,
            "change": 2.5
        }

    def test_stock_list_filtering(self):
        """Test filtering stock list response"""
        stocks = [
            {"symbol": "AAPL", "name": "Apple Inc.", "price": 150.0, "market_cap": 2500000000},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "price": 2800.0, "market_cap": 1800000000},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "price": 320.0, "market_cap": 2400000000}
        ]

        # Client only wants symbol and price
        result = filter_response_fields(stocks, "symbol,price")

        assert len(result) == 3
        assert all("symbol" in stock and "price" in stock for stock in result)
        assert all(len(stock) == 2 for stock in result)

    def test_portfolio_with_nested_positions(self):
        """Test filtering portfolio with nested positions"""
        portfolio = {
            "id": "portfolio-1",
            "total_value": 100000.0,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "value": 15000.0},
                {"symbol": "GOOGL", "quantity": 20, "value": 56000.0}
            ],
            "metadata": {
                "created_at": "2026-01-01",
                "last_updated": "2026-02-08"
            }
        }

        # Client wants portfolio ID and only symbol/value from positions
        result = filter_response_fields(portfolio, "id,positions.symbol,positions.value")

        assert result == {
            "id": "portfolio-1",
            "positions": [
                {"symbol": "AAPL", "value": 15000.0},
                {"symbol": "GOOGL", "value": 56000.0}
            ]
        }

    def test_api_response_wrapper_filtering(self):
        """Test filtering with API response wrapper structure"""
        response = {
            "success": True,
            "data": {
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000000
            },
            "timestamp": "2026-02-08T12:00:00Z"
        }

        # Filter nested data fields
        result = filter_response_fields(response, "success,data.symbol,data.price")

        assert result == {
            "success": True,
            "data": {
                "symbol": "AAPL",
                "price": 150.0
            }
        }
