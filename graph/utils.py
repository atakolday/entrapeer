from typing import List
import json
from urllib.parse import urlparse
import wordninja
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from tools import WikipediaQueryTool, SerperQueryTool

def extract_source_name(url: str):
    """
    Extracts a readable source name from a URL.
    Example: "https://finance.yahoo.com/quote/NVDA/news/" → "Yahoo Finance"
    """
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split(".")
    
    def wordninja_helper(domain_parts: List[str]) -> str:
        if len(domain_parts) > 2:
            domain, subdomain = domain_parts[1], domain_parts[0]
            source: List = wordninja.split(domain) + wordninja.split(subdomain)

        else:
            source: List = wordninja.split(domain_parts[0])

        # Remove duplicate words while preserving order
        seen = set()
        unique = [word for word in source if not (word in seen or seen.add(word))]

        formatted_source = " ".join(word.upper() if len(word) <= 3 else word.title() for word in unique)
        return formatted_source
    
    return wordninja_helper(domain_parts)

def hyperlink(url: str, text: str) -> str:
    """
    Returns a clickable hyperlink (if supported by the terminal) for the given URL and text.
    """
    ESC = "\033"
    return f"{ESC}]8;;{url}{ESC}\\{text}{ESC}]8;;{ESC}\\"

def determine_tool_for_task(task: str, model=ChatOpenAI(model='gpt-4o-mini', temperature=0)):
    """
    Uses an LLM to determine which tools should be used for the task.
    """
    prompt = ChatPromptTemplate.from_messages([
        "You are a smart assistant that selects the best search tools for a given task. \
        You have access to the following tools: \
         - WikipediaQueryTool (for structured information like general knowledge, company profiles, historical/product info, location) \
         - SerperQueryTool (for up-to-date news, financial queries, Google-like search) \
         - TavilySearchResults (as a fallback, for web-wide search, including blogs, analysis, and lists).\
        Given the following user task: \
        {task} \
        Return a JSON list with the best tools for this task, STRICYLY in the following format \
        {{\"tools\": [\"<tool1>\", \"<tool2>\"]}}."
        # {{[\"WikipediaQueryTool\", \"SerperQueryTool\"]}}"
    ])

    formatted_prompt = prompt.format_messages(task=task)

    response = model.invoke(formatted_prompt).content.strip()
    
    try:
        response_json = json.loads(response.replace("\n", "").strip())
        if not isinstance(response_json, dict):
            raise ValueError("Response is not a valid JSON object")
        
        tool_list = response_json['tools']
        tools_map = {
            "WikipediaQueryTool": WikipediaQueryTool,  
            "SerperQueryTool": SerperQueryTool,  
            "TavilySearchResults": TavilySearchResults(max_results=3)  
        }
        return [tools_map[tool] for tool in tool_list if tool in tools_map]  # e.g. [WikipediaQueryTool, TavilySearchResults(max_results=3)]

    except (json.JSONDecodeError, ValueError) as e:
        # print("❌ Error parsing extraction response:", e)
        return [TavilySearchResults(max_results=3)]

if __name__ == "__main__":
    result = determine_tool_for_task("Where is OpenAI located?")
    print("\n > Result: ", result)