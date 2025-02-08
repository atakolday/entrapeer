import os
import getpass
import time
import sys
from dotenv import load_dotenv
load_dotenv()

import warnings
from bs4 import GuessedAtParserWarning

from langsmith import utils
from langchain_openai import ChatOpenAI

from query_disambiguator import QueryDisambiguator
from financial_query_handler import FinancialQueryHandler
from verification_search_handler import VerificationSearchHandler
from wikipedia_query_handler import WikipediaQueryHandler
from utils import evaluate_response

# LangChain's WikipediaAPI has a weird error that's due to a lack of maintenance, it doesn't cause any problems so just ignored
warnings.simplefilter("ignore", GuessedAtParserWarning)

# Function to retrieve API key (from .env or user input)
def get_key(env_var, prompt):
    key = os.getenv(env_var)
    if key:
        return key  # Use key from .env if available
    return getpass.getpass(f"Enter {prompt}: ")  # Otherwise, ask user

# LangChain environment
os.environ["LANGCHAIN_API_KEY"] = get_key("LANGCHAIN_API_KEY", "LangChain API Key")
os.environ["LANGCHAIN_TRACING_V2"] = get_key("LANGCHAIN_TRACING_V2", "LangChain Tracing [true/false]")
os.environ["LANGCHAIN_PROJECT"] = get_key("LANGCHAIN_PROJECT", "LangChain Project Name")

# Load API Keys dynamically, stores the keys if found in the environment, otherwise asks for user input
os.environ["OPENAI_API_KEY"] = get_key("OPENAI_API_KEY", "OpenAI API Key")
os.environ["TAVILY_API_KEY"] = get_key("TAVILY_API_KEY", "Tavily API Key")
os.environ["SERPER_API_KEY"] = get_key("SERPER_API_KEY", "Serper API Key")

if not utils.tracing_is_enabled():
    print(">>> LangSmith Tracing NOT enabled!")

def main(retry=False, user_input=None):
    """Main execution flow for handling user queries dynamically."""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if not retry:
        user_input = input(" >> So, what would you like to look up today?  ")
    
    # Step 1: Query Disambiguation & Refinement
    disambiguator = QueryDisambiguator(model)
    refined_query = disambiguator.resolve_query(user_input)
    intent = disambiguator.intent     # extracted user intent
    company = disambiguator.company   # extracted company name
    
    # print(f"\n >> Refined Query: {refined_query}\n") # debugging
    
    # Step 2: Get initial response (Wikipedia, Yahoo Finance)
    if intent == 'stock':
        # print(" >> Searching Yahoo Finance...")
        first_tool = FinancialQueryHandler(user_query=refined_query, model=model)
        first_result = first_tool.analyze_stock(company)

        # BREAK: If Yahoo Finance couldn't find the ticker, it returned a tuple, with the second element being its message
        if type(first_result) != str:  
            print(f" >> It looks like {first_result[1]}")
            something_else = input(" >> Would you like to search something else? (y/n)  ")
            if something_else.lower() in ['y', 'yes']:
                return main()
            else:
                time.sleep(1)
                return " >> Understood. Have a great day!"
            
        return first_result  # Yahoo Finance response has the most up-to-date information, verification system fails

    else:
        # print(" >> Searching Wikipedia...")
        first_tool = WikipediaQueryHandler(
            model = model,
            intent = intent,
            company = company,
            doc_content_chars_max = 5000, 
            top_k_results = 5
        )
        first_result = first_tool.generate_response(refined_query) # Wikipedia response
    
    # Step 3: Response Evaluation
    evaluation = evaluate_response(refined_query, first_result, model)
    # print(f" >> Evaluation: {evaluation}") # debugging
    if evaluation == 'sufficient':
        verification_handler = VerificationSearchHandler(model)
        
        # Step 4: Response Verification
        verified_result = verification_handler.combined_search(
            user_query = refined_query,
            auxiliary_response = first_result,
            aux_source=first_tool.url
        )

    else:
        # print(f" >> Hmm, {first_tool.name} didn't have enough information.") # first_tool.name will print either Yahoo Finance or Wikipedia

        # print(" >> Searching Tavily and Serper...")
        verification_handler = VerificationSearchHandler(model)
        verified_result = verification_handler.combined_search(user_query = refined_query) # combined search with Tavily and Serper
        
        # Step 5: Retry & Refine if Evaluation Fails
        evaluation = evaluate_response(refined_query, verified_result, model)  # evaluate the search results
        if evaluation != 'sufficient':
            return main(retry=True, user_input=refined_query)

    return verified_result


if __name__ == "__main__":
    # Check if running interactively
    if sys.stdin.isatty():  
        print("""
Welcome! I am an AI assistant that will help you with your company-related queries.
I can provide information about a company you want, including:
 • General information (e.g. location, history, products, investment portfolio)
 • Financial information (e.g. stocks, market performance, projections)
 • Recent news and updates
          
After I answer your question, I will cite my sources as hyperlinks so that you can check for more details. 
 • On Mac: Press \033[1mCommand (⌘) + Click\033[0m on a link to open it in your browser.
 • On Windows (PowerShell, Windows Terminal): Press \033[1mCtrl + Click\033[0m to access the source directly.
 • On Windows Command Prompt (cmd.exe): Hyperlinks are not supported, so please \033[1mcopy and paste\033[0m the link into your browser.

Start by asking me a question about a company, and I'll do my best to help you out!
""")
        final_result = main()
        print(f"\n{final_result}\n")
    
    else:
        # Exit if not interactive
        time.sleep(1)
        print("\n >> Hey there! This program requires user input. You should run the container with `-it` flag.\n")  
        time.sleep(1)
        sys.exit(1)
   