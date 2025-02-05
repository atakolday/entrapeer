from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class WikipediaQueryHandler:
    """Handles Wikipedia-based searches for company-related information."""
    def __init__(self, intent=None, company=None, model=ChatOpenAI(model='gpt-4o-mini'), **kwargs):
        self.wiki = WikipediaAPIWrapper(**kwargs)
        self.model = model
        self.name = "Wikipedia"  # For debugging purposes
        self.company = company
        self.intent = intent

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context. \
                        If the context does not mention the user's question, \
                        return 'The context provided does not mention {question}.' \
                        ONLY provide a one-sentence answer that directly answers the question."),
            ("user", "Question: {question}\nContext: {context}")
        ])
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.model | self.output_parser

    def search_wikipedia(self, intent: str, search_query: str):
        """Fetches relevant Wikipedia context using lazy loading."""
        if intent == 'location':
            return self.wiki.run(f'{self.company}')  # Easier for Wikipedia to extract location data from just the company name

        return self.wiki.lazy_load(search_query)     # Returns a generator object, better for resource efficiency

    def generate_response(self, question: str):
        """Uses Wikipedia context to generate an LLM-based response efficiently."""
        wiki_generator = self.search_wikipedia(intent=self.intent, search_query=question)
        
        if type(wiki_generator) == str: # If the generator returns a string, it's a summary
            context = wiki_generator
            response = self.chain.invoke({"question": question, "context": context}).strip()
            if not response.startswith("The context provided does not mention"):
                # print(f" - Wikipedia Response (context=summary): {response}")
                return response
            else:
                # conduct lazy_load search if the company name is insufficient
                wiki_generator = self.search_wikipedia(intent=None, search_query=question) 

        try:
            while True:
                context = next(wiki_generator)  # Get next result from generator
                response = self.chain.invoke({"question": question, "context": context}).strip()
                
                if not response.startswith("The context provided does not mention"):
                    # print(f" - Wikipedia Response (context=lazy_load): {response}")
                    return response  # Return the first valid response
        except StopIteration:
            return f"No relevant Wikipedia data found for {question}."
            