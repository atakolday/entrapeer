# Intelligent Company Information Retrieval System

**The Intelligent Company Information Retrieval System** is designed to automate the process of handling user queries about companies and providing accurate, up-to-date information. Supported query types include:

- General information (e.g. location, history, products, investment portfolio, etc.)
- Recent news and updates about a company, and
- Financial information (e.g. stocks, market performance, projections)

The main workflow is implemented in `main.py`, which coordinates the processing and resolution of incoming queries.

## Project Structure

```
.
├── main.py                         # Entry point of the application
├── query_disambiguator.py          # Handles query disambiguation, classification, and refinement
├── verification_search_handler.py  # Manages web-based and verification-related searches through Tavily and Google Serper API
├── financial_query_handler.py      # Handles finance-related queries through Yahoo Finance API
├── wikipedia_query_handler.py      # Handles query search through Wikipedia API
├── utils.py                        # Utility functions used throughout the project (expandable)
```

## Installation

### Prerequisites
- Python 3.8 or later
- Required dependencies listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <repo_name>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start processing queries:
```sh
python main.py
```

## Storing API Keys
To use this application, you will need 4 API keys:
- `LangChain API`: to manage LangSmith integration and query tracing
- `OpenAI API`: to handle LLM requests within LangChain
- `Tavily API`: to conduct general, web-based searches within LangChain
- `Serper Google Search API`: to conduct low-cost Google searches within LangChain

Since API kets should not be hardcoded or pushed to version control, users have two secure ways to store them:
1. Using a `.env` file:
- In your project root, create a file named .env
- Add the required keys in the following format:
  ```sh
  LANGCHAIN_API_KEY=<Your LangChain API Key here>
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_PROJECT=<Your project name here>

  OPENAI_API_KEY=<Your OpenAI API Key here>
  TAVILY_API_KEY=<Your Tavily API Key here>
  SERPER_API_KEY=<Your Serper API Key here>
  ```
- Specifying your `LANGCHAIN_PROJECT` and setting `LANGCHAIN_TRACING_V2=true` will enable LangChain tracing through your designated project name.

2. Using Secure Input via `getpass`:
- Instead of storing the key in a file, users can also enter it securely at runtime.
- If the required API keys are not in the environment, the following lines in `main.py` will automatically prompt the user to securely input their API keys:
  ```sh
  # LangChain environment
  os.environ["LANGCHAIN_API_KEY"] = get_key("LANGCHAIN_API_KEY", "LangChain API Key")
  os.environ["LANGCHAIN_TRACING_V2"] = get_key("LANGCHAIN_TRACING_V2", "LangChain Tracing [true/false]")
  os.environ["LANGCHAIN_PROJECT"] = get_key("LANGCHAIN_PROJECT", "LangChain Project Name")

  # Load API Keys dynamically, stores the keys if found in the environment, otherwise asks for user input
  os.environ["OPENAI_API_KEY"] = get_key("OPENAI_API_KEY", "OpenAI API Key")
  os.environ["TAVILY_API_KEY"] = get_key("TAVILY_API_KEY", "Tavily API Key")
  os.environ["SERPER_API_KEY"] = get_key("SERPER_API_KEY", "Serper API Key")
  ```

### Components

#### `main.py`
- The central orchestrator of the query processing system.
- Routes queries to the appropriate handlers based on their type.
- Uses `query_disambiguator.py` to resolve ambiguous queries.

#### `query_disambiguator.py`
- Receives the raw incoming query.
- Detects ambiguity in the query, prompts the user for clarification.
- Extracts the company name, user intent, any relevant details, and timeframe details.
- Refines the query based on clarification input and relevant query details.
- Returns the refined query.

#### `verification_search_handler.py`
- Handles queries that require external verification.
- Handles web-search functionality using `Tavily` and `Google Serper`
- Retrieves and verifies information from authoritative sources.

#### `financial_query_handler.py`
- Specializes in parsing and processing financial-related queries.
- Fetches stock data, financial reports, or other related financial information.
- Generates an LLM-response based on retrieved information from Yahoo Finance.

#### `wikipedia_query_handler.py`
- Searches Wikipedia for relevant results.
- Extracts useful information from Wikipedia pages.

#### `utils.py`
- Contains helper functions used across different modules.
- Currently, only includes `evaluate_response()` which evaluates the retrieved response based on the user query, focusing on (1) relevance, and (2) completeness.

## End-to-End Execution Flow
1. `main.py` receives a query through user input.
2. The query is passed to `query_disambiguator.py` for refinement and classification.
3. Based on classification, the query is routed to the appropriate handler:
   - `financial_query_handler.py` for finance-related queries through Yahoo Finance.
   - `wikipedia_query_handler.py` for all other queries.
4. The appropriate handler processes the query and returns the first result.
5. The returned result is evaluated for relevance and completeness.
   - If the first result from Yahoo Finance or Wikipedia is sufficient, `verification_search_handler.py` verifies the information through a web search.
   - The application returns the verified result.
5. If the first result failed the relevance or completeness check, the query is given to `verification_search_handler.py` for a real-time web search using Tavily and Google Serper.
   - the result is evaluated for relevance and completeness.
   - if the result is sufficient, the application returns the verified result.
   - if the result is insufficient or incomplete, the application returns to step 2, and asks the user for further details with `main(retry=True, user_input=refined_input)`
5. `main.py` formats and outputs the verified result.


## License
This project is licensed under the MIT License. See `LICENSE` for more details.
