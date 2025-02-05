from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from verification_search_handler import VerificationSearchHandler

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
        print(" > Unexpected evaluation output:", evaluation_result)
        return "incomplete"  # Default to insufficient if response is unclear