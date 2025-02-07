import sys
import os
import unittest
from unittest.mock import patch

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils import evaluate_response, hyperlink, extract_source_names, format_sources

class TestUtils(unittest.TestCase):

    def test_hyperlink(self):
        url = "https://example.com"
        text = "Example"
        expected_output = "\x1b]8;;https://example.com\x1b\\Example\x1b]8;;\x1b\\"
        self.assertEqual(hyperlink(url, text), expected_output)

    def test_extract_source_names(self):
        urls = ["https://businessinsider.com/news", "https://nytimes.com/articles"]
        expected_output = [("Business Insider", "https://businessinsider.com/news"), 
                           ("NY Times", "https://nytimes.com/articles")]
        self.assertEqual(extract_source_names(urls), expected_output)

    def test_format_sources_with_ticker(self):
        text = "Stock price update (Source: Yahoo Finance)."
        ticker = "AAPL"
        expected_output = "Stock price update (Source: \x1b]8;;https://finance.yahoo.com/quote/AAPL\x1b\\Yahoo Finance\x1b]8;;\x1b\\)."
        self.assertEqual(format_sources(text, ticker), expected_output)

    def test_format_sources_without_ticker(self):
        text = "Latest news (https://news.com, https://example.com)."
        expected_output = "Latest news (Source: \x1b]8;;https://news.com\x1b\\News\x1b]8;;\x1b\\, \x1b]8;;https://example.com\x1b\\Example\x1b]8;;\x1b\\)."
        self.assertEqual(format_sources(text), expected_output)

    @patch("utils.ChatOpenAI.invoke")
    def test_evaluate_response(self, mock_model):
        mock_model.return_value.content = "sufficient"
        self.assertEqual(evaluate_response("What is the capital of France?", "Paris"), "sufficient")

        mock_model.return_value.content = "irrelevant"
        self.assertEqual(evaluate_response("What is the capital of France?", "New York"), "irrelevant")

        mock_model.return_value.content = "incomplete"
        self.assertEqual(evaluate_response("Tell me about quantum mechanics", "It's a field of physics"), "incomplete")

        mock_model.return_value.content = "unexpected_output"
        self.assertEqual(evaluate_response("What is 2+2?", "4"), "incomplete")  # Default fallback

if __name__ == "__main__":
    unittest.main()