from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from pydantic import BaseModel, Field

# Define input schema for Wikipedia search
class QueryInput(BaseModel):
    search_query: str = Field(description="The query string to search for on the wrapper.")

# Define the Wikipedia search function
def search_wikipedia(search_query: str) -> str:
    """Fetches relevant Wikipedia content based on the query intent."""
    wiki = WikipediaAPIWrapper()
    return next(wiki.lazy_load(search_query), f"No relevant Wikipedia data found for {search_query}.")

# def search_serper(search_query: str) -> List[dict[str: str]]:
#     serper = GoogleSerperAPIWrapper()
#     serper_results = serper.results(search_query)
#     return serper_results['organic']
def search_serper(search_query: str) -> List[dict[str, str]]:
    serper = GoogleSerperAPIWrapper()
    serper_results = serper.results(search_query)

    if not serper_results or "organic" not in serper_results:
        return [{"title": "No results found", "content": "No relevant search results for this query."}]

    return serper_results

# Convert the function into a structured tool
WikipediaQueryTool= StructuredTool.from_function(
    func=search_wikipedia,
    name="WikipediaQueryTool",
    description="Searches Wikipedia for relevant information based on query.",
    args_schema=QueryInput,
    return_direct=False,
)

SerperQueryTool = StructuredTool.from_function(
    func=search_serper,
    name="SerperQueryTool",
    description="Searches Google via Serper API for relevant information based on query.",
    args_schema=QueryInput,
    return_direct=False,
)
