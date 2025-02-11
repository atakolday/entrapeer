import os
import getpass
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langsmith import utils
from langgraph.graph import StateGraph, START, END
from agents import AgentState, PlannerAgent, RePlannerAgent, QueryClarifierAgent, AmbiguityDetectorAgent
from base_models import Plan, Act, Response, DetectAmbiguity, RefinedQuery
from langgraph.prebuilt import create_react_agent
import asyncio

from langchain_community.tools.tavily_search import TavilySearchResults
from tools import WikipediaQueryTool, SerperQueryTool
from utils import determine_tool_for_task, extract_source_name

import argparse

parser = argparse.ArgumentParser(description='Helper for the AI Application.')
parser.add_argument('--debug', '-d', action='store_true')
args = parser.parse_args()

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
    print("\n>>> LangSmith Tracing NOT enabled!")

# Initialize agents
planner = PlannerAgent(output_schema=Plan)
replanner = RePlannerAgent(output_schema=Act)
ambiguity_detector = AmbiguityDetectorAgent(output_schema=DetectAmbiguity)
query_clarifier = QueryClarifierAgent(output_schema=RefinedQuery)

# Initialize the model and tools
model = ChatOpenAI(model='gpt-4o', temperature=0)
tools = [WikipediaQueryTool, SerperQueryTool, TavilySearchResults(max_results=3)]
prompt = "You are a helpful assistant."
executor_agent = create_react_agent(model, tools, prompt=prompt)

### ✅ Step 3: Define Execution Flow ###
async def execute_step(state: AgentState):
    """Runs a single step from the plan and executes applicable tools in parallel."""
    if not state["plan"]:
        return {"response": "No plan steps available for execution."}

    # Fetch the next unexecuted task
    past_step_tasks = {step[0] for step in state["past_steps"]}
    task = next((step for step in state["plan"] if step not in past_step_tasks), None)
    
    if task is None:
        return {"response": "All steps in the plan have been executed."}

    # Determine applicable tools for the task
    applicable_tools = determine_tool_for_task(task)
    # print(f"\n >> Task: {task}\n >> Applicable Tools: {applicable_tools}")

    # Fallback to LLM execution if no tools apply
    if not applicable_tools:
        agent_response = await executor_agent.ainvoke({"messages": [("user", task)]})
        result = agent_response["messages"][-1].content
        return {"past_steps": [(task, result)]}

    async def run_tool(tool):
        """Runs an individual tool and extracts responses + actual source URLs."""
        if tool in [WikipediaQueryTool, SerperQueryTool]:
            formatted_input = {"search_query": task}  # Proper input formatting
        else:
            formatted_input = task  # Tavily can take raw string

        agent_response = await tool.ainvoke(formatted_input)

        # ✅ Extract content and source URLs from structured data
        response_texts = []
        response_urls = []

        if isinstance(agent_response, list):  # Handle list outputs (Serper, Tavily)
            for res in agent_response:
                if isinstance(res, dict):
                    content = res.get("content") or res.get("snippet", "")
                    source_url = res.get("link") or res.get("source", "")

                    if content:
                        response_texts.append(content)
                    if source_url:
                        response_urls.append(source_url)

                elif isinstance(res, Document):  # If it's a LangChain Document
                    response_texts.append(res.page_content)  # Extract text
                    response_urls.append(res.metadata.get("source", ""))  # Extract source if available

        elif isinstance(agent_response, Document):  # Single Document response
            response_texts.append(agent_response.page_content)
            response_urls.append(agent_response.metadata.get("source", ""))

        else:  # Standard string response
            response_texts.append(str(agent_response))

        response_text = "\n".join(response_texts)
        
        return tool.name, response_text, response_urls  # ✅ Return cleaned response

    # ✅ Run all applicable tools asynchronously
    tool_responses = await asyncio.gather(*[run_tool(tool) for tool in applicable_tools])

    # ✅ Merge responses and format sources correctly
    merged_results = []
    sources = []

    for tool_name, result, urls in tool_responses:
        merged_results.append(result)  # ✅ Ensure all results are strings now
        sources.extend(urls)  # ✅ Collect all URLs

    # ✅ Convert final response to string safely
    final_response = "\n".join(str(res) for res in merged_results)
    formatted_sources = "\n".join(f"- [{extract_source_name(url)}]({url})" for url in sources if url)

    # ✅ Return past steps with properly formatted sources
    return {
        "past_steps": state["past_steps"] + [(task, f"{final_response}\n\nSources:\n{formatted_sources}")]
    }

### ✅ Step 4: Define Query Handling Flow ###
async def handle_query(state: AgentState):
    """Detects ambiguity, refines query if necessary, then proceeds with planning."""
    # Step 1: Detect ambiguity
    detection_result = await ambiguity_detector.execute(state)
    if not detection_result["is_ambiguous"]:
        return {}  # ✅ Proceed without refinement if not ambiguous

    # Step 2: Ask for clarification
    clarification_question = detection_result.get("follow_up", "Can you clarify?")
    print(f"\n >> Hmm, I need some clarification. {clarification_question}")
    user_clarification = input(" >> Your clarification: ")

    # Step 3: Refine the query
    refined_query = await query_clarifier.execute({**state, **detection_result, "user_input": user_clarification})
    return refined_query  # ✅ Updated query is now part of state

### ✅ Step 5: Define End Condition ###
def should_end(state: AgentState):
    """Determines if execution should stop."""
    if "response" in state and state["response"]:
        return END
    last_step = state["past_steps"][-1][0] if state["past_steps"] else ""
    return END if last_step.lower().startswith(("confirm", "provide the final answer")) and not state["plan"] else "agent"


### ✅ Step 6: Build Execution Graph ###
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("ambiguity_detection", handle_query)
workflow.add_node("planner", planner.execute)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replanner.execute)

# Define execution flow
workflow.add_edge(START, "ambiguity_detection")  # First check for ambiguity
workflow.add_edge("ambiguity_detection", "planner")  # Proceed to planning
workflow.add_edge("planner", "agent")  # Execute tasks
workflow.add_edge("agent", "replan")  # Replan if needed

# Define stop conditions
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

# Compile workflow
app = workflow.compile()

if __name__ == "__main__":
    config = {"recursion_limit": 50}
    user_query = input("\n >> What would you like to search today?  ")
    # user_query = "What is Apple's current stock price?"
    inputs = {"query": user_query}

    async def debug():
        async for event in app.astream(inputs, config=config):
            for k, v in event.items():
                print(f"{k}: {v}")

    async def run_app():
        response = None  # Initialize response storage

        async for event in app.astream(inputs, config=config):
            if "replan" in event and "response" in event["replan"]:  # Extract response correctly
                response = event["replan"]["response"]

        if response:
            print(f'\n >> {response}')  # Print only the final response
        else:
            print("No response generated.")
        # asyncio.run(run_app())

    if args.debug:
        asyncio.run(debug())
    else:
        asyncio.run(run_app())