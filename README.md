# Intelligent Company Information Retrieval System

**The Intelligent Company Information Retrieval System** is designed to automate the process of handling user queries about companies and providing accurate, up-to-date information. Supported query types include:

- General information (e.g. location, history, products, investment portfolio, etc.)
- Recent news and updates about a company, and
- Financial information (e.g. stocks, market performance, projections)

The main workflow is implemented in `main.py`, which coordinates the processing and resolution of incoming queries.

---
### ⚠️ LangGraph Note

You may find a LangGraph integration of this application in `graph/`, which contains all the necessary code for the application to run on LangGraph. However, some features (e.g. Dockerfile, unittests) are not present yet.

- You can still run the app using `graph/main.py` after following the local installation steps here: [🔧 Local Installation](#local-installation).
- The resulting agent will receive a LangSmith trace of all the executed steps.
- You can view the executed sample query's LangSmith trace here: [Link](https://smith.langchain.com/public/da6fa2b2-ee84-46be-9961-b2a21bb46117/r)
- **LangGraph App Execution Flow:**

  ![LangGraph agents graph](https://github.com/atakolday/entrapeer/blob/main/static/graph_structure.png)
  
---

## 📂 Project Structure

```
.
├── main.py                         # Entry point of the application
├── query_disambiguator.py          # Handles query disambiguation, classification, and refinement
├── verification_search_handler.py  # Manages web-based and verification-related searches through Tavily and Google Serper API
├── financial_query_handler.py      # Handles finance-related queries through Yahoo Finance API
├── wikipedia_query_handler.py      # Handles query search through Wikipedia API
├── utils.py                        # Utility functions used throughout the project (expandable)
├── unittests/                      # Folder containing all unit tests
│   ├── test_utils.py
│   ├── test_financial_query_handler.py
│   ├── test_query_disambiguator.py
│   ├── test_verification_search_handler.py
│   ├── test_wikipedia_query_handler.py
├── Dockerfile                      # Docker configuration for containerized deployment
├── graph/                          # [In progress] LangGraph integration
│   ├── base_models.py              # The base model structures used with structured agent output
│   ├── agents.py                   # The state structure and all agents used in the main execution flow
│   ├── tools.py                    # Contains all search tools (Wikipedia, Google Serper, Tavily)
│   ├── utils.py                    # Some utility functions
│   ├── main.py                     # Main LangGraph execution flow
├── static/                         # Miscellaneous
```

## 🚀 End-to-End Execution Flow
1. `main.py` receives a query through user input.
2. The query is passed to `query_disambiguator.py` for refinement and classification.
3. Based on classification, the query is routed to the appropriate handler:
   - `financial_query_handler.py` for finance-related queries through Yahoo Finance.
   - `wikipedia_query_handler.py` for all other queries.
4. The appropriate handler processes the query and returns the first result.
   - If the Yahoo Finance API couldn't find a ticker associated with the company name in the query, the LLM returns the reason for why that company does not have a ticker (e.g. OpenAI is not a publicly traded company).
   - The application asks the user if they wish to search for something else (`[y/n]`):
      - If the user inputs `yes` or `y`, `main()` is called again, and the application returns to **Step 1**.
      - Otherwise, the system exits with a goodbye message.
5. The returned result is evaluated for relevance and completeness.
   - If the first result from Yahoo Finance or Wikipedia is sufficient, `verification_search_handler.py` verifies the information through a web search.
   - The application returns the verified result.
5. If the first result failed the relevance or completeness check, the query is given to `verification_search_handler.py` for a real-time web search using Tavily and Google Serper.
   - The result is verified internally by cross-checking raw results from `Tavily` and `Google Serper`.
   - The verified result is evaluated for relevance and completeness.
   - If the result is sufficient, the application returns the verified result.
   - If the result is insufficient or incomplete, the application returns to Step 1, and asks the user for further details with `main(retry=True, user_input=refined_input)`:
      - At this point, the application takes the already refined input as the raw user input and asks for any additional details from the user.
      - The query is **not** mapped to a predetermined value this time, it is passed along as `{company_name} {details} {time_reference}`.

## 🔧 Local Installation

### Prerequisites
- Python 3.8 or later
- Required dependencies listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/atakolday/entrapeer.git
   cd entrapeer
   ```
2. Set up virtual environment:

   Using Python virtual environment (`venv`)
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
   Using Conda environment
   ```sh
   conda create --name entrapeer-env python=3.12
   conda activate entrapeer-env  
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Storing API Keys
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

### Usage

Run the main script to start processing queries:
```sh
python main.py
```
---

## 🐳 Running with Docker

This project supports containerized execution via Docker.

### Build the Docker Image
```sh
docker build -t entrapeer-app .
```

### Run the Container (Interactive Mode)

To run the app with interactive input, use:

```sh
docker run --rm -it --env-file .env entrapeer-app
```

### Running Tests Inside Docker (Optional)

If you want to run tests inside the container, you can execute:

```sh
docker run --rm -it entrapeer-app python -m unittest discover unittests
```

### Stop & Clean Up Docker Containers

```sh
docker ps -a           # List running containers
docker stop <ID>       # Stop a specific container
docker system prune -a # Remove unused containers and images
```

## 🧪 Running Unit Tests
All unit tests are located in the `unittests/` folder. Run them using:
```sh
python -m unittest discover unittests
```
## 🧩 Components

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
- `evaluate_response`: Evaluates the retrieved response based on the user query, focusing on (1) relevance, and (2) completeness.
- `hyperlink`: Creates hyperlinks with source URL, using ASCII escape sequences for the sources listed at the end of the system response.
- `extract_source_names`: Extracts domain names from URLs and returns a list of (formatted_source, url) tuples.
- `format_sources`: If there is a ticker, returns the hyperlinked Yahoo Finance URL. Otherwise, finds a list of source URLs inside parentheses at the end of the given text and replaces them with hyperlinked, formatted source names.
