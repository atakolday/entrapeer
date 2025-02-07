import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from wikipedia_query_handler import WikipediaQueryHandler

class TestWikipediaQueryHandler(unittest.TestCase):
    
    @patch("wikipedia_query_handler.WikipediaAPIWrapper.run")
    @patch("wikipedia_query_handler.WikipediaAPIWrapper.lazy_load")
    def test_search_wikipedia(self, mock_lazy_load, mock_run):
        """Test Wikipedia search function for different intents."""
        handler = WikipediaQueryHandler(company="Apple")
        
        # Mock Wikipedia responses
        mock_run.return_value = "Apple Inc. is located in Cupertino, California."
        mock_lazy_load.return_value = iter(["Apple Inc. designs and manufactures consumer electronics."])
        
        # Test 'location' intent
        self.assertEqual(handler.search_wikipedia("location", "Apple"), "Apple Inc. is located in Cupertino, California.")
        
        # Test general search
        result = handler.search_wikipedia("general", "Apple")
        self.assertEqual(next(result), "Apple Inc. designs and manufactures consumer electronics.")
    
    @patch("wikipedia_query_handler.WikipediaQueryHandler.search_wikipedia")
    @patch("wikipedia_query_handler.ChatOpenAI.invoke")
    def test_generate_response(self, mock_model_invoke, mock_search_wikipedia):
        """Test Wikipedia response generation using LLM."""
        handler = WikipediaQueryHandler(intent="general", company="Apple")

        # Mock Wikipedia and LLM responses
        mock_search_wikipedia.return_value = iter(["Apple Inc. is a multinational technology company."])
        
        # Ensure mock returns a string, not a MagicMock
        mock_model_invoke.return_value = "Apple Inc. is a major player in the tech industry."

        result = handler.generate_response("What is Apple?")
        self.assertEqual(result, "Apple Inc. is a major player in the tech industry.")

        # Test case where no relevant context is found
        mock_model_invoke.return_value = "The context provided does not mention Apple."
        mock_search_wikipedia.return_value = iter([])  # Simulate no results
        result = handler.generate_response("What is Apple?")
        self.assertEqual(result, "No relevant Wikipedia data found for What is Apple?.")

if __name__ == "__main__":
    unittest.main()
