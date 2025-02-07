from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import re
import wordninja
import tldextract

def evaluate_response(user_query: str, retrieved_response: str, model=ChatOpenAI(model='openai-4o-mini')) -> str:
    """Evaluates if a retrieved response sufficiently answers the user's question."""
    
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluation assistant that determines whether a retrieved response completely and accurately answers the user's question. \
                    Evaluation Criteria: \
                    1. Relevance: Does the information directly address the user's specific question (e.g., user question: 'Apple stock price' --> response includes 'Apple', 'stock' and its price in $)? \
                    2. Completeness: Is the answer detailed enough to answer the user query? \
                    Decision Rules: \
                    - If the retrieved response is relevant for and adequetly answers the user question, return 'sufficient'. \
                    - If the retrieved response is not relevant to the user question, return 'irrelevant'. \
                    - If the retrieved response is not complete or enough to answer the user question, return 'incomplete'.\
                    ONLY return 'sufficient', 'irrelevant', 'incomplete'."),
        ("user", "User Question: {user_query}\nRetrieved Response: {retrieved_response}")
    ])
    
    formatted_prompt = evaluation_prompt.format_messages(
        user_query=user_query, 
        retrieved_response=retrieved_response
    )
    
    evaluation_result = model.invoke(formatted_prompt).content.strip().lower()
    
    if evaluation_result in ["sufficient", "irrelevant", "incomplete"]:
        return evaluation_result
    else:
        # print(" > Unexpected evaluation output:", evaluation_result)  # DEBUG
        return "incomplete"  # Default to insufficient if response is unclear
    
def hyperlink(url, text):
    """
    Returns a clickable hyperlink (if supported by the terminal) for the given URL and display text.
    """
    ESC = "\033"
    return f"{ESC}]8;;{url}{ESC}\\{text}{ESC}]8;;{ESC}\\"

def extract_source_names(source_list):
    """
    Extracts domain names from URLs and returns a list of (formatted_source, url) tuples.
    """
    source_names = []
    for url in source_list:
        extracted = tldextract.extract(url)
        domain = extracted.domain

        # Split concatenated words (e.g., "businessinsider" -> ["business", "insider"])
        words = wordninja.split(domain)

        # Capitalize words individually based on length
        formatted_source = " ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in words)

        source_names.append((formatted_source, url))
    
    return source_names

def format_sources(text, ticker=None):
    """
    If there is a ticker, the source is coming a Yahoo Finance, so plugs in the custom Yahoo Finance URL.
    Otherwise, finds a list of source URLs inside parentheses at the end of `text` and replaces them with
    hyperlinked, formatted source names.
    """
    match = re.search(r"\(([^)]+)\)\.?\s*$", text)

    if match:
        hyperlinks = []
       
        # If a ticker is provided, the source is Yahoo Finance. 
        if ticker:                                                   
            url = f"https://finance.yahoo.com/quote/{ticker}"
            hyperlinks.append(hyperlink(url, "Yahoo Finance"))

        # If a ticker is not provided, then this is VerificationSearchHandler calling this function.
        else:
            # Assume the sources inside the parentheses are comma-separated URLs.
            sources = match.group(1).split(", ")
            formatted_sources = extract_source_names(sources)
            
            # Remove duplicate URLs while preserving order.
            seen_names = set()

            for name, url in formatted_sources:
                if name not in seen_names:
                    seen_names.add(name)
                    hyperlinks.append(hyperlink(url, name))
                
        new_source_text = f"(Source: {', '.join(hyperlinks)})."
        formatted_text = re.sub(r"\([^()]*\)\.?\s*$", lambda m: new_source_text, text)
        # print(" - Text with formatted sources: ", formatted_text) # DEBUG
        return formatted_text

    return text