from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper

from utils import format_sources

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
            # print(' >> detected auxiliary response.') # debugging
            return self.verify_auxiliary_response(user_query, auxiliary_response, tavily_text, serper_text, all_sources, aux_source)

        # Otherwise, generate a combined answer
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that synthesizes and validates search results for a user query. \
                        Given two separate web searches, your task is to produce a DIRECT, concise (one sentence) \
                        answer that combines the key information from both results. Follow these rules: \
                        1. Your answer must address the query directly without additional commentary. \
                        2. If the query requests a list (e.g., companies), include specific, concrete examples. \
                        3. At the end of your answer, append the source names in the following format: (Source: Source1, Source2, ..., Source n) for ALL relevant sources. \
                        4. Format each source as a **separate clickable hyperlink** using ANSI escape sequences, ensuring that links are correctly separated. \
                        Use this structure for each source: '\033]8;;<source_url>\033\\<source_name>\033]8;;\033\\' \
                        When listing multiple sources, separate them with `, ` (a comma and a space), ensuring **NO ANSI escape characters touch each other**. \
                        Example: (Source: \033]8;;https://businessinsider.com/\033\\Business Insider\033]8;;\033\\, \
                                          \033]8;;https://reuters.com/\033\\Reuters\033]8;;\033\\). \
                        5. If the two sources conflict, rely on the Second search. \
                        Provide ONLY the final answer in the specified format."),
            ("user", "User Query: {query}\\nFirst search: {tavily_text}\\nSecond search: {serper_text}\\nSources: {all_sources}")
        ]) 

        formatted_prompt = verification_prompt.format_messages(
            query=user_query, 
            tavily_text=tavily_text, 
            serper_text=serper_text,
            all_sources=", ".join(all_sources[:5])  
        )

        return self.model.invoke(formatted_prompt).content.strip()

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
        # print(' >> Validation result: ', validation_result)

        if validation_result == "valid":
            all_sources = ", ".join([aux_source] + sources[:5])                           # Add the auxiliary source (Wikipedia or Yahoo Finance) 
            response = f"{auxiliary_response[:-1]} ({all_sources})"                       # Add sources to the end of the response, before the period
            formatted_response = format_sources(response)                                 # Format the sources for readability
            return formatted_response
        else:
            return self.combined_search(query)                                            # Generate a new response if Wikipedia is invalid