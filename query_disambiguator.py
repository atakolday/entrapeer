from dotenv import load_dotenv
load_dotenv()

import json
import datetime
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class QueryDisambiguator:
    def __init__(self, model=ChatOpenAI(model='gpt-4o-mini')):
        """Initialize the disambiguation system with an LLM model and query executor."""
        self.model = model
        self.company = None   # will be dynamically added later
        self.intent = None    # will be dynamically added later

    def detect_ambiguity(self, user_input: str):
        """Detects if a user query is ambiguous and requires clarification."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant whose sole task is to determine whether a company-related query is ambiguous. Follow these steps strictly: \
                        1. Identify the company name mentioned in the query. \
                        2. Check if this company name could refer to more than one business entity. If so, it is ambiguous. Example: 'Midas' could refer to 'Midas Investments' or 'Midas Automotive Service'. \
                        3. Determine if the query is vague about what aspect of the company is being asked (e.g., location, business model, history, etc.). \
                        4. If any of these conditions are met, the query is ambiguous. Otherwise, it is not. \
                        \
                        If ambiguous, output exactly in JSON format: \
                        {{\"ambiguous\": true, \"follow_up\": \"Clarification question\"}}. \
                        If not ambiguous, output exactly: \
                        {{\"ambiguous\": false, \"follow_up\": null}}"),
            ("user", "Query: {user_input}")
        ])
        formatted_prompt = prompt.format(user_input=user_input)

        response = self.model.invoke(formatted_prompt).content.strip()

        # Debugging: Print the raw response to see if the model is behaving correctly
        # print(" - Raw Model Response:", response)

        try:
            response_json = json.loads(response)
            return response_json  # Returns structured JSON output
        except json.JSONDecodeError:
            print("Error: Model did not return valid JSON. Using default response.")
            return {"ambiguous": False, "follow_up": ""}

    def clarify_query(self, original_query: str, clarification: str):
        """Uses the LLM to intelligently refine the original query based on the clarification."""
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that refines a user query based on clarification input. "
                       "Ensure that the refined query is clear, precise, and correctly structured."),
            ("user", "Original Query: {original_query}\nClarification: {clarification}\nRefined Query:")
        ])
        formatted_prompt = refinement_prompt.format(original_query=original_query, clarification=clarification)

        refined_response = self.model.invoke(formatted_prompt).content.strip()
        return refined_response

    def extract_company_and_intent(self, user_query: str):
        """Extracts company name and intent from a user query using LLM."""
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that extracts structured information from user queries about companies. Follow these instructions: \
                        1. Identify the full company name (e.g., 'Sequoia' → 'Sequoia Capital', 'Apple' → 'Apple, Inc.'). \
                        2. Determine the user's intent from this list: general information, location, business model, investments, stock, news, products, history. \
                        3. If a specific time, year, or relative time expression (e.g., “recently,” “latest,” “current”) is mentioned, extract it in the 'time_reference' field; otherwise, leave it blank. \
                        4. For the 'details' field, extract any REMAINING modifier that refines or specifies the main intent (e.g., 'price' in 'stock price', 'headquarters' in 'headquarters location'). Do not repeat the company name or generic phrases. \
                        Output your answer strictly in JSON format as: \
                        {{\"company\": \"<company>\", \"intent\": \"<intent>\", \"details\": \"<details>\", \"time_reference\": \"<time_reference>\"}}."),
            ("user", "Query: {user_query}")
        ])
        formatted_messages = extraction_prompt.format_messages(user_query=user_query)
        response = self.model.invoke(formatted_messages).content.strip()

        # For debugging
        # print("🛠 Raw Extraction Response:", response)
        
        try:
            response_json = json.loads(response.replace("\n", "").strip())
            if not isinstance(response_json, dict):
                raise ValueError("Response is not a valid JSON object")
            return response_json  
        except (json.JSONDecodeError, ValueError) as e:
            print("❌ Error parsing extraction response:", e)
            return {"company": "Unknown", "intent": "Unknown", "details": "", "time_reference": ""} 

    def refine_query_for_tools(self, user_query: str, retry=False):
        """Refine the user query into an optimized tool-specific query based on intent, company, and time reference."""
        result = self.extract_company_and_intent(user_query)
        intent, company, details, time_reference = result['intent'], result['company'], result['details'], result['time_reference']

        if retry:
            new_query = f"{company} {details} {time_reference}".strip()
            return new_query

        details = re.sub(rf"\b{intent}\b", "", details).strip() # removes the intent from the details to avoid repetition in the query
        
        self.intent = intent    # store the intent for future reference
        self.company = company  # store the company name for future reference

        # If no specific time is mentioned, use the current year for "recent" queries
        RELATIVE_TIME_WORDS = {"recently", "latest", "current", "today", "this year"}
        if time_reference and any(word in time_reference.lower() for word in RELATIVE_TIME_WORDS):
            time_reference = str(datetime.datetime.now().year)  # Example: "2025"
            # print(f" ! No time reference found, parsing from datetime (year: {time_reference}).") # --> Debugging

        query_map = {
            "general information": f"{company} history and products overview", 
            'location': f'{company} headquarters location'.strip() if details == "" else f'{company} {details} location'.strip(), 
            "business model": f"{company} revenue model",
            "investments": f"{company} investment portfolio {time_reference}",
            "stock": f"{company} stock {details}",
            "news": f"Latest news on {company} {time_reference}",
            "products": f"{company} product lineup {time_reference}",
            "history": f"{company} history overview {time_reference}"
        }
        
        new_query = query_map.get(intent, f"{company} {details} {time_reference}").strip() # if no pre-coded intent, ensure that it isn't empty
        refined_query = re.sub(r"\s+", " ", new_query).strip() # remove the whitespaces, clean-up
        
        return refined_query

    def resolve_query(self, user_query: str, retry=False):
        """Handles ambiguity, refines query, and returns the final refined query."""
        if retry:
            print("\nHmm, your query didn't yield any search results. Could you provide more information?")
            retry_clarification = input(" >> Your clarification: ")
            refined_query = self.clarify_query(user_query, retry_clarification)

            return self.refine_query_for_tools(refined_query)
        
        detection_response = self.detect_ambiguity(user_query)
        
        # For debugging
        # print("✅ Parsed Ambiguity Response:", detection_response)
        
        if detection_response.get("ambiguous", False):
            clarification_question = detection_response.get("follow_up", "Could you clarify?")
            print(f"\n >> Hmm, I need some clarification. {clarification_question}")
            user_clarification = input(" >> Your clarification: ")
            refined_query = self.clarify_query(user_query, user_clarification)
        else:
            refined_query = user_query
        
        return self.refine_query_for_tools(refined_query)