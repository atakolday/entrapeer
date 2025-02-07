import sys
import os
import unittest
from unittest.mock import patch

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import unittest
from unittest.mock import patch
from financial_query_handler import FinancialQueryHandler

class TestFinancialQueryHandler(unittest.TestCase):

    @patch("financial_query_handler.yf.Ticker")  # Mock yfinance
    def test_fetch_stock_data(self, mock_ticker):
        """Test that fetch_stock_data returns expected stock information."""
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.info = {
            "currentPrice": 150.0,
            "marketCap": 250000000000,
            "trailingPE": 25.3,
            "dividendYield": 0.015,
            "fiftyTwoWeekHigh": 180.0,
            "fiftyTwoWeekLow": 120.0
        }

        handler = FinancialQueryHandler()
        stock_data = handler.fetch_stock_data("AAPL")

        expected_output = {
            "price": 150.0,
            "market_cap": 250000000000,
            "pe_ratio": 25.3,
            "dividend_yield": 0.015,
            "52_week_high": 180.0,
            "52_week_low": 120.0
        }

        self.assertEqual(stock_data, expected_output)

    @patch("financial_query_handler.ChatOpenAI.invoke")
    def test_get_ticker(self, mock_model):
        """Test that get_ticker returns the correct stock ticker."""
        handler = FinancialQueryHandler()
        
        # Case 1: Valid ticker
        mock_model.return_value.content = "AAPL"
        self.assertEqual(handler.get_ticker("Apple"), "AAPL")

        # Case 2: Company not publicly traded
        mock_model.return_value.content = "Not publicly traded"
        self.assertEqual(handler.get_ticker("Small Private Company"), "Not publicly traded")

    @patch("financial_query_handler.format_sources")
    @patch("financial_query_handler.FinancialQueryHandler.fetch_stock_data")
    @patch("financial_query_handler.FinancialQueryHandler.get_ticker")
    def test_analyze_stock(self, mock_get_ticker, mock_fetch_stock_data, mock_format_sources):
        """Test full stock analysis, ensuring all dependencies are called correctly."""
        handler = FinancialQueryHandler(user_query="What is Apple's stock price?")
        
        # Mock dependencies
        mock_get_ticker.return_value = "AAPL"
        mock_fetch_stock_data.return_value = {
            "price": 150.0,
            "market_cap": 250000000000,
            "pe_ratio": 25.3,
            "dividend_yield": 0.015,
            "52_week_high": 180.0,
            "52_week_low": 120.0
        }
        mock_format_sources.return_value = "As of today, Apple's stock is $150 (Source: Yahoo Finance)."

        # Call function
        result = handler.analyze_stock("Apple")

        # Verify dependencies were called
        mock_get_ticker.assert_called_once_with("Apple")
        mock_fetch_stock_data.assert_called_once_with("AAPL")
        mock_format_sources.assert_called()

        # Check output
        self.assertEqual(result, "As of today, Apple's stock is $150 (Source: Yahoo Finance).")

if __name__ == "__main__":
    unittest.main()