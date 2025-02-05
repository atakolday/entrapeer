from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper

import re
import tldextract

class VerificationSearchHandler:
    """A combined search handler that integrates Tavily and Serper for robust verification."""

    def __init__(self, model=ChatOpenAI(model="gpt-4o-mini")):
        self.model = model
        self.tavily = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
        )
        self.serper = GoogleSerperAPIWrapper()

    def search_tavily(self, query: str):
        """Perform a search using Tavily and extract actual sources."""
        tavily_response = self.tavily.run(query)

        extracted_content = []
        sources = []

        for entry in tavily_response:
            extracted_content.append(entry['content'])
            sources.append(entry['url'])

        return " ".join(extracted_content[:3]), sources  

    def search_serper(self, query: str):
        """Perform a search using Serper and extract actual sources."""
        serper_results = self.serper.results(query)
        organic_results = serper_results.get("organic", [])

        extracted_content = []
        sources = []

        for entry in organic_results:
            if "link" in entry and "snippet" in entry:
                sources.append(f"{entry['link']}")
                extracted_content.append(entry["snippet"])

        return " ".join(extracted_content[:3]), sources  

    def combined_search(self, user_query: str, auxiliary_response: str = None, aux_source: str = None):
        """
        Run both search tools and use LLM to merge results into a refined answer.
        If an auxiliary response (e.g. Wikipedia) is provided, verify it first.
        """
        tavily_text, tavily_sources = self.search_tavily(user_query)
        serper_text, serper_sources = self.search_serper(user_query)

        all_sources = list(set(tavily_sources + serper_sources))  # Merge sources, remove duplicates

        # If an auxiliary response exists, verify it
        if auxiliary_response:
            print(' >> detected auxiliary response.') # debugging
            return self.verify_auxiliary_response(user_query, auxiliary_response, tavily_text, serper_text, all_sources, aux_source)

        # Otherwise, generate a combined answer
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that combines and validates search results from a user query, and returns a combined answer. \
                        Compare the responses from two separate web searches. Your task is to combine the information \
                        from both search results based on the user query, and return a concise and complete 1-2 sentence answer. \
                        ONLY provide a DIRECT answer the query. For example, if the user asks for a list of companies, MAKE SURE to include specific companies in your response. \
                        Extract the source names IN PROPER FORMAT from the sources gathered through both sets of search results \
                        (e.g., https://gbtimes.com/ → GB Times, https://businessinsider.com/ → Business Insider, 'Sfstandard → SF Standard) \
                        and ONLY include those source names in parentheses at the end of your combined response like this: \
                        (e.g. [Your answer here] (Source: Tesla Official Website, Yahoo Finance))."),
            ("user", "User Query: {query}\\nFirst search: {tavily_text}\\nSecond search: {serper_text}\\nSources: {all_sources}")
        ])

        formatted_prompt = verification_prompt.format_messages(
            query=user_query, 
            tavily_text=tavily_text, 
            serper_text=serper_text,
            all_sources=", ".join(all_sources[:5])  
        )

        return self.model.invoke(formatted_prompt).content.strip()

    @staticmethod
    def extract_source_names(source_list):
        """Extracts domain names from URLs to make readable source names."""
        source_names = []
        
        for url in source_list:
            extracted = tldextract.extract(url)
            source_name = extracted.domain                                             # Extract only the domain part
            formatted_source = source_name.capitalize()                                # Capitalize first letter for readability
            source_names.append(formatted_source)
        
        return source_names

    @staticmethod
    def format_sources(text):
        """Replaces URLs with extracted source names inside the parentheses at the end of the sentence."""
        match = re.search(r"\((.*?)\)\s*$", text)
        
        if match:
            sources = match.group(1).split(", ")                                        # Split sources
            formatted_sources = VerificationSearchHandler.extract_source_names(sources)  # Convert URLs to readable names
            new_source_text = f"(Source: {', '.join(formatted_sources)})"

            formatted_text = re.sub(r"\(.*?\)\s*$", new_source_text, text)              # Replace sources in the original text
            # print(formatted_text)
            return formatted_text                                                       # Replace sources in the original text
        
        return text                                                                     # Return original if no parentheses found

    def verify_auxiliary_response(self, query: str, auxiliary_response: str, first_text: str, second_text: str, sources: list, aux_source: str = None):
        """
        Verify an auxiliary response (e.g., Wikipedia) against Tavily and Serper.
        If the auxiliary response is validated, return it with proper citations.
        If it is contradicted, return a refined answer based on search results.
        """

        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that validates whether an auxiliary response is accurate, \
                        using search results from web searches First search and Second search. Respond based on the following: \
                        - If the auxiliary response contains factually correct and relevant information based on the search results, respond with 'valid'.\
                        - ONLY if the auxiliary response is inaccurate, respond with 'invalid'.\
                        Respond with either 'valid' or 'invalid'."),
            ("user", "Query: {query}\\nAuxiliary Response: {auxiliary_response}\\nFirst search: {first_text}\\nSecond search: {second_text}")
        ])

        formatted_prompt = verification_prompt.format_messages(
            query=query,
            auxiliary_response=auxiliary_response,
            first_text=first_text,
            second_text=second_text
        )

        validation_result = self.model.invoke(formatted_prompt).content.strip().lower()

        # For debugging
        print(' >> Validation result: ', validation_result)

        if validation_result == "valid":
            unique_sources = ", ".join(dict.fromkeys([aux_source] + sources[:5])) # Remove duplicates from sources
            response = f"{auxiliary_response} ({unique_sources})"
            formatted_response = VerificationSearchHandler.format_sources(response)
            return formatted_response
        else:
            return self.combined_search(query)  # Generate a new response if Wikipedia is invalid