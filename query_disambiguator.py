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
            ("system", "You are an assistant that detects ambiguity in queries, asking something about a COMPANY. \
                        Your task is to determine if that query has multiple interpretations. \
                        ONLY detect ambiguity in the following scenarios: \
                        - If multiple companies exist with the name in the query. However, pass this check if the company is a well-known global corporation (e.g., Sequoia Capital, Apple, Tesla, Amazon). \
                        - If the query mentions a company name, but does not specify what SPECIFIC aspect of the company they wish to know. \
                        - If the query's intent has multiple interpretations (location --> headquarters, or any other relevant location associated with the company's business). \
                        If ambiguity exists, provide a clarification question. \
                        Respond in strict JSON format: \
                        {{\"ambiguous\": true/false, \"follow_up\": Clarification question if needed}}"),
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
            ("system", "You are an assistant that extracts structured information from user queries. \
                        Identify the full company name (e.g., 'Sequoia' → 'Sequoia Capital', 'Google' → 'Alphabet, Inc.'), \
                        user intent (general information, location, business model, investments, stock, news, products, history, etc.), \
                        and any specific details about the user intent (e.g., 'San Francisco store', 'headquarters', 'stock performance'). \
                        If you can't detect a clear intent, leave it blank. Do not include the user intent in the details, only auxiliary information. \
                        If an explicit time reference is mentioned, extract it. Otherwise, leave it blank for future use. \
                        Respond strictly in JSON format: \
                        {{\"company\": \"\", \"intent\": \"\", \"details\": \"\", \"time_reference\": \"\"}}"),
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
        if not time_reference and ("recently" in user_query.lower() or "latest" in user_query.lower()):
            time_reference = str(datetime.datetime.now().year)  # Example: "2025"
            # print(f" ! No time reference found, parsing from datetime (year: {time_reference}).") # --> Debugging

        query_map = {
            "general information": f"{company} history and products overview", 
            "location": f"{company} {details} location", 
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
            print(f"\nHmm, I need some clarification. {clarification_question}")
            user_clarification = input(" >> Your clarification: ")
            refined_query = self.clarify_query(user_query, user_clarification)
        else:
            refined_query = user_query
        
        return self.refine_query_for_tools(refined_query)