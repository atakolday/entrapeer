import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from verification_search_handler import VerificationSearchHandler
from utils import format_sources

class TestVerificationSearchHandler(unittest.TestCase):
    
    @patch("verification_search_handler.TavilySearchResults.run")
    def test_search_tavily(self, mock_tavily_run):
        """Test that Tavily search correctly extracts content and sources."""
        handler = VerificationSearchHandler()
        
        # Mock response from Tavily
        mock_tavily_run.return_value = [
            {"content": "Tesla is an electric vehicle company.", "url": "https://tesla.com"},
            {"content": "Tesla's CEO is Elon Musk.", "url": "https://example.com/tesla"}
        ]
        
        text, sources = handler.search_tavily("Tesla company information")
        self.assertEqual(text, "Tesla is an electric vehicle company. Tesla's CEO is Elon Musk.")
        self.assertEqual(sources, ["https://tesla.com", "https://example.com/tesla"])

    @patch("verification_search_handler.GoogleSerperAPIWrapper.results")
    def test_search_serper(self, mock_serper_results):
        """Test that Serper search correctly extracts content and sources."""
        handler = VerificationSearchHandler()
        
        # Mock response from Serper
        mock_serper_results.return_value = {
            "organic": [
                {"link": "https://news.com/tesla", "snippet": "Tesla recently announced a new vehicle."},
                {"link": "https://finance.com/tesla", "snippet": "Tesla stock prices have risen significantly."}
            ]
        }
        
        text, sources = handler.search_serper("Tesla latest news")
        self.assertEqual(text, "Tesla recently announced a new vehicle. Tesla stock prices have risen significantly.")
        self.assertEqual(sources, ["https://news.com/tesla", "https://finance.com/tesla"])
    
    @patch("verification_search_handler.VerificationSearchHandler.search_tavily")
    @patch("verification_search_handler.VerificationSearchHandler.search_serper")
    @patch("verification_search_handler.ChatOpenAI.invoke")
    def test_combined_search(self, mock_model_invoke, mock_search_serper, mock_search_tavily):
        """Test that combined_search correctly merges and validates results."""
        handler = VerificationSearchHandler()
        
        mock_search_tavily.return_value = ("Tesla is an EV company.", ["https://tesla.com"])
        mock_search_serper.return_value = ("Tesla's stock is performing well.", ["https://finance.com"])
        mock_model_invoke.return_value = MagicMock(content="Tesla is a leading EV company and its stock is performing well. (Source: Tesla, Finance)")
        
        result = handler.combined_search("Tell me about Tesla")
        self.assertEqual(result, "Tesla is a leading EV company and its stock is performing well. (Source: Tesla, Finance)")

    @patch("verification_search_handler.VerificationSearchHandler.search_tavily")
    @patch("verification_search_handler.VerificationSearchHandler.search_serper")
    @patch("verification_search_handler.ChatOpenAI.invoke")
    def test_verify_auxiliary_response(self, mock_model_invoke, mock_search_serper, mock_search_tavily):
        """Test that verify_auxiliary_response validates auxiliary sources correctly."""
        handler = VerificationSearchHandler()
        
        mock_search_tavily.return_value = ("Tesla produces electric cars.", ["https://tesla.com"])
        mock_search_serper.return_value = ("Tesla is a well-known EV brand.", ["https://news.com"])
        mock_model_invoke.return_value = MagicMock(content="valid")
        
        response = handler.verify_auxiliary_response(
            query = "What does Tesla do?", 
            auxiliary_response = "Tesla is an electric vehicle company.", 
            first_text = "Tesla produces electric cars.", 
            second_text = "Tesla is a well-known EV brand.", 
            sources = ["https://tesla.com", "https://news.com"], 
            aux_source = "https://wikipedia.com/wiki/Tesla_Inc."
        )
        
        expected_output = format_sources("Tesla is an electric vehicle company (https://wikipedia.com/wiki/Tesla_Inc., https://tesla.com, https://news.com).")
        self.assertEqual(response, expected_output)

if __name__ == "__main__":
    unittest.main()