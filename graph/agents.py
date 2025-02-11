from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import json
import operator
from typing import Annotated, List, Tuple, TypedDict, Union, Optional
from pydantic import BaseModel, Field

from base_models import Response, DetectAmbiguity, RefinedQuery

# Define State
class AgentState(TypedDict):
    """State tracking for the agent throughout execution."""
    
    query: str  # Original user query
    refined_query: Optional[str] = None  # Query after refinement (if ambiguity detected)
    
    plan: List[str] 
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    
    is_ambiguous: bool = False  # Whether the query required disambiguation
    follow_up: Optional[str] = None  # Follow-up question for user clarification
    clarification: Optional[str] = None  # User clarification for ambiguous query
    response: str

# Parent class for all agents
class BaseAgents:
    def __init__(self, state: Optional[AgentState] = None, model_name="gpt-4o-mini", temperature=0, output_schema: BaseModel = None):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.state = state or AgentState(query="", plan=[], past_steps=[])  # Default empty state
        self.raw_response = None  # Store raw response for debugging
        self.output_schema = output_schema  # Custom BaseModel for structured output

        self.prompt = self._define_prompt()  # Subclasses override this
        self.chain = self._define_chain()

    def _define_prompt(self):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def _define_chain(self):
        """Defines the LLM chain using the specified output schema."""
        if not self.output_schema:
            raise ValueError("Output schema (BaseModel) must be provided.")
        return self.prompt | self.model.with_structured_output(self.output_schema)

    async def _run_chain(self):
        """Runs the LLM and stores the raw response for debugging."""
        # self.raw_response = await self.chain.ainvoke(**kwargs)
        raise NotImplementedError

    async def execute(self):
        """To be implemented by subclasses."""
        raise NotImplementedError
    
class PlannerAgent(BaseAgents):
    def _define_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent AI that generates step-by-step search queries to efficiently gather factual information. "
                        "Each step must be directly searchable. \n\n"
             
                        "**Instructions:**\n"
                        "1. Each step must be **a well-formed search query**.\n"
                        "2. Do **not** include vague steps like 'Identify the topic' or 'Find information'.\n"
                        "3. Assume a search tool will execute each step—write queries accordingly.\n"
                        "4. Always be **specific and structured**.\n\n"
                        
                        "**Examples:**\n"
                        "- ❌ Bad: 'Research Sequoia Capital's investments.'\n"
                        "- ✅ Good: 'What companies has Sequoia Capital invested in since 2023?'\n"
                        "- ❌ Bad: 'Look up OpenAI's location.'\n"
                        "- ✅ Good: 'Where is OpenAI's headquarters located?'\n"
                        "- ❌ Bad: 'Check for recent updates on Tesla.'\n"
                        "- ✅ Good: 'What are the latest news articles about Tesla in 2024?'\n\n"

                        "Generate a precise **step-by-step search plan** for the query."
                    ),
            ("placeholder", "{messages}")
        ])

    async def _run_chain(self, state: AgentState):
        self.raw_response = await self.chain.ainvoke(state)
        return self.raw_response

    async def execute(self, state_or_query: Union[AgentState, str]):
        """Generates an execution plan for the given state."""
        # Call LLM to generate a plan asynchronously
        if isinstance(state_or_query, str):
            state = AgentState(query=state_or_query, plan=[], past_steps=[])
        else:
            state = state_or_query
        
        output = await self._run_chain(
            {"messages": [("user", state["query"])]}
        )

        # Return updated AgentState with the generated plan
        # print(" >> Current state: ", state)
        return {"plan": output.steps}
        
class RePlannerAgent(BaseAgents):
    def _define_prompt(self):
        return ChatPromptTemplate.from_template(
            "For the given task, come up with a simple, step-by-step plan. Do NOT add any superfulous steps. "
            "This plan should involve individual tasks that, if executed correctly, will yield the correct answer. "
            "The result of the final step should be the final answer. "
            "Make sure that each step has all the information needed—do not skip steps.\n\n"
            
            "Your objective was this:\n"
            "{query}\n\n"

            "Your original plan was this:\n"
            "{plan}\n\n"

            "You have currently done the following steps:\n"
            "{past_steps}\n\n"

            "If the question has been sufficiently answered, **return the final answer immediately**. "
            "Do not add an extra step like 'Provide the final answer'—just return the answer as the response.\n\n"

            "### **IMPORTANT RESPONSE FORMAT**\n"
            "- **First**, provide a **concise ONE SENTENCE final answer**.\n"
            "- **Second**, include a **citation list** for all used sources in your answer, with the **source names and URLs** formatted as:\n"
            "  - [Source Name] Source URL \n"
            "If additional verification is needed, include only the necessary remaining steps."
        )
    
    async def _run_chain(self, state: AgentState):
        self.raw_response = await self.chain.ainvoke(state)
        return self.raw_response

    async def execute(self, state: AgentState):
        """
        Revises the plan based on execution results.
        If no more steps are needed, returns a final response.
        """
        # Call LLM asynchronously with state.query
        output = await self._run_chain(state)  

        # If the response indicates that execution is complete, return final response
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        
        # Otherwise, update the plan
        return {"plan": output.action.steps}
    
class AmbiguityDetectorAgent(BaseAgents):
    def _define_prompt(self):
        return ChatPromptTemplate.from_template(
            "You are an assistant whose sole task is to determine whether a company-related query is ambiguous. "
            "Query: {query}\n"
            "Follow these steps strictly:\n"
            "1. Identify the intended company name mentioned in the query (e.g. Tesla, Apple, Google, etc.).\n"
            "2. If this company name could refer to more than one company, it is ambiguous. "
            "For example: The company name 'Midas' could refer to 'Midas Investments' or 'Midas Automotive Service'.\n"
            "3. Determine if the query is vague about what aspect of the company is being asked "
            "(e.g., location of a store vs. headquarters, business model, history, etc.).\n"
            "4. If any of these conditions are met, the query is ambiguous. Otherwise, it is not.\n\n"
            "IF ambiguous, reply with a clarification question and respond **ONLY** in the following JSON format:\n"
            '{{\"is_ambiguous\": true/false, \"follow_up\": \"Clarification question if needed \"}}\n'
            # "if not, respond **ONLY** in the following JSON format without any extra text: \n"
            # '{{\"is_ambiguous\": false, \"follow_up\": null}}'
        )

    async def _run_chain(self, state: AgentState):
        """Runs the LLM to detect ambiguity in the query."""
        self.raw_response = await self.chain.ainvoke({"query": state["query"]})
        return self.raw_response

    # async def execute(self, state: AgentState):
    #     """Checks if the query is ambiguous and updates the state."""
    #     output = await self._run_chain(state)

    #     # ✅ Ensure output is properly converted to JSON string
    #     if isinstance(output, dict):  # If LLM already returns a dict
    #         parsed_output = output
    #     elif isinstance(output, str):  # If LLM returns a string
    #         try:
    #             parsed_output = json.loads(output)
    #         except json.JSONDecodeError:
    #             return {"is_ambiguous": False, "follow_up": None}  # Fallback if LLM output isn't valid JSON
    #     else:
    #         return {"is_ambiguous": False, "follow_up": None}  # Fallback for unexpected output types

    #     return {
    #         "is_ambiguous": parsed_output.get("is_ambiguous", False),
    #         "follow_up": parsed_output.get("follow_up", None),
    #     }
    async def execute(self, state: AgentState):
        """Checks if the query is ambiguous and updates the state."""
        output = await self._run_chain(state)
        # print("🔍 LLM Raw Output:", output)  # Debugging

        # If output is a DetectAmbiguity object, extract fields
        if isinstance(output, DetectAmbiguity):
            return {
                "is_ambiguous": output.is_ambiguous,
                "follow_up": output.follow_up,
            }

        # ✅ If output is a string (raw JSON), parse it
        try:
            parsed_output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            print("❌ JSON Decode Error! LLM did not return valid JSON.")
            return {"is_ambiguous": False, "follow_up": None}

        return {
            "is_ambiguous": parsed_output.get("is_ambiguous", False),
            "follow_up": parsed_output.get("follow_up", None),
        }
    
class QueryClarifierAgent(BaseAgents):
    def _define_prompt(self):
        return ChatPromptTemplate.from_template(
            "You are an assistant that refines a user query based on clarification input. "
            "Ensure that the refined query is clear, precise, and correctly structured.\n\n"
            "**Output your response strictly in JSON format:**\n"
            '{{\"refined_query\": \"Your refined query here\"}}\n\n'
            "---\n\n"
            "Original Query: {query}\n"
            "Clarification: {clarification}\n\n"
        )

    async def _run_chain(self, state: AgentState):
        """Uses the LLM to refine the query based on the user's clarification."""
        self.raw_response = await self.chain.ainvoke({"query": state["query"], "clarification": state["clarification"]})
        # Ensure the LLM outputs structured JSON
        return self.raw_response

    async def execute(self, state: AgentState):
        """Refines the query based on user clarification and updates the state."""
        if not state["is_ambiguous"]:
            return {}  # ✅ No ambiguity detected, no refinement needed

        clarification_question = state.get("follow_up", "Could you provide more details?")
        print(f"\n >> Hmm, I need some clarification. {clarification_question}")
        user_clarification = input(" >> Your clarification: ")
        state["clarification"] = user_clarification

        output = await self._run_chain(state)

        # If output is a RefinedQuery object, extract fields
        if isinstance(output, RefinedQuery):
            return {"refined_query": output.refined_query,}

        # ✅ If output is a string (raw JSON), parse it
        try:
            parsed_output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            print("❌ JSON Decode Error! LLM did not return valid JSON.")
            return {"refined_query": state["query"]}

        return {"refined_query": parsed_output.get("refined_query", state["query"]),}