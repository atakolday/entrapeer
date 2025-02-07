import sys
import os
from io import StringIO
import unittest
from unittest.mock import patch, MagicMock

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from query_disambiguator import QueryDisambiguator

class TestQueryDisambiguator(unittest.TestCase):

    def setUp(self):
        """Redirect stdout to suppress print statements."""
        self.suppress_output = StringIO()
        sys.stdout = self.suppress_output

    def tearDown(self):
        """Restore stdout after each test."""
        sys.stdout = sys.__stdout__

    @patch("query_disambiguator.ChatOpenAI.invoke")
    def test_detect_ambiguity(self, mock_model_invoke):
        """Test if detect_ambiguity correctly identifies ambiguous and unambiguous queries."""
        handler = QueryDisambiguator()
        
        # Mock response for ambiguous query
        mock_model_invoke.return_value = MagicMock(content='{"ambiguous": true, "follow_up": "Are you referring to Tesla headquarters or a specific location?"}')
        self.assertEqual(handler.detect_ambiguity("Where is Tesla?"), 
                         {"ambiguous": True, "follow_up": "Are you referring to Tesla headquarters or a specific location?"})

        # Mock response for unambiguous query
        mock_model_invoke.return_value = MagicMock(content='{"ambiguous": false, "follow_up": null}')
        self.assertEqual(handler.detect_ambiguity("Where is Tesla headquarters?"), 
                         {"ambiguous": False, "follow_up": None})

    @patch("query_disambiguator.ChatOpenAI.invoke")
    def test_clarify_query(self, mock_model_invoke):
        """Test if clarify_query correctly refines a vague query."""
        handler = QueryDisambiguator()
        mock_model_invoke.return_value = MagicMock(content="Tesla headquarters location")
        
        self.assertEqual(handler.clarify_query("Where is Tesla?", "I'm asking about their headquarters"), 
                         "Tesla headquarters location")

    @patch("query_disambiguator.ChatOpenAI.invoke")
    def test_extract_company_and_intent(self, mock_model_invoke):
        """Test if extract_company_and_intent properly extracts structured data."""
        handler = QueryDisambiguator()
        mock_model_invoke.return_value = MagicMock(content='{"company": "Tesla Inc.", "intent": "location", "details": "headquarters", "time_reference": ""}')
        
        self.assertEqual(handler.extract_company_and_intent("Where is Tesla headquarters?"),
                         {"company": "Tesla Inc.", "intent": "location", "details": "headquarters", "time_reference": ""})

    @patch("query_disambiguator.QueryDisambiguator.extract_company_and_intent")
    def test_refine_query_for_tools(self, mock_extract):
        """Test if refine_query_for_tools generates optimized tool-specific queries."""
        handler = QueryDisambiguator()
        mock_extract.return_value = {"company": "Tesla Inc.", "intent": "location", "details": "headquarters", "time_reference": ""}
        
        self.assertEqual(handler.refine_query_for_tools("Where is Tesla?"), "Tesla Inc. headquarters location")

    @patch("query_disambiguator.QueryDisambiguator.detect_ambiguity")
    @patch("query_disambiguator.QueryDisambiguator.clarify_query")
    @patch("query_disambiguator.QueryDisambiguator.refine_query_for_tools")
    @patch("builtins.input", return_value="I'm asking about Tesla headquarters")  # Mock user input
    def test_resolve_query(self, mock_input, mock_refine, mock_clarify, mock_detect):
        """Test if resolve_query correctly handles ambiguity and refines queries."""
        handler = QueryDisambiguator()
        
        # Case 1: Ambiguous query, requiring clarification
        mock_detect.return_value = {"ambiguous": True, "follow_up": "Are you referring to Tesla headquarters or a specific location?"}
        mock_clarify.return_value = "Tesla headquarters location"
        mock_refine.return_value = "Tesla Inc. headquarters location"
        
        self.assertEqual(handler.resolve_query("Where is Tesla?"), "Tesla Inc. headquarters location")

        # Case 2: Unambiguous query
        mock_detect.return_value = {"ambiguous": False, "follow_up": None}
        mock_refine.return_value = "Tesla Inc. headquarters location"
        
        self.assertEqual(handler.resolve_query("Where is Tesla headquarters?"), "Tesla Inc. headquarters location")

if __name__ == "__main__":
    unittest.main()