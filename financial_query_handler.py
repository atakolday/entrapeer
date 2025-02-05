from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yfinance as yf

import datetime

class FinancialQueryHandler:
    """A tool to look up financial information using yfinance and integrate LLM reasoning."""
    def __init__(self, user_query, model=ChatOpenAI(model='gpt-4o-mini')):
        self.name = "Yahoo Finance"
        self.query = user_query
        self.company = None   # will be dynamically added below
        self.ticker = None    # will be dynamically added below
        
        self.model = model
        self.prompt = self._initialize_prompt()
        self.ticker_lookup_prompt = self._initialize_ticker_lookup_prompt()
    
    def _initialize_prompt(self):
        """Set up a structured prompt for LLM-based financial insights."""
        today = datetime.datetime.today().strftime("%B {day}, %Y").format(day=datetime.datetime.today().day)
        return ChatPromptTemplate.from_messages([
            ("system", f"You are a financial assistant that analyzes stock data and provides insights. \
                        Provide a succinct, 1-2 sentence summary, that ONLY directly answers the user question. \
                        Start your response 'As of {today}', include the company's name and the ticker in parentheses \
                        (e.g., Tesla, Inc. ($TSLA) ...), avoid excessive details, and focus only on valuable information. \
                        End your response with (Source: Yahoo Finance)."),
            ("user", "Stock Symbol: {symbol}\nCurrent Data: {data}\nUser Question: {query}")
        ])
    
    def _initialize_ticker_lookup_prompt(self):
        """Set up a structured prompt for LLM-based company-to-ticker lookup."""
        return ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that maps company names to their corresponding stock ticker symbols. \
             ONLY respond with the stock ticker (e.g. 'Apple' → 'AAPL')"),
            ("user", "Company Name: {company}\nWhat is the stock ticker?")
        ])
    
    def fetch_stock_data(self, symbol: str) -> dict:
        """Fetch stock data from yfinance."""
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        info = {
            "price": stock.info.get("currentPrice"),
            "market_cap": stock.info.get("marketCap"),
            "pe_ratio": stock.info.get("trailingPE"),
            "dividend_yield": stock.info.get("dividendYield"),
            "52_week_high": stock.info.get("fiftyTwoWeekHigh"),
            "52_week_low": stock.info.get("fiftyTwoWeekLow"),
        }
        return info
    
    def get_ticker(self, company_name: str) -> str:
        """Uses LLM to find the stock ticker symbol for a given company name."""
        formatted_prompt = self.ticker_lookup_prompt.format_messages(company=company_name)
        response = self.model.invoke(formatted_prompt).content.strip()
        
        return response
    
    def analyze_stock(self, company_name: str) -> str:
        """Uses LLM to find the stock ticker, fetch stock data, and generate insights."""
        ticker = self.get_ticker(company_name)
        if len(ticker) > 5:
            return False, ticker # When the ticker is not found, the system returns a str with information for why (e.g. "Not publicly traded")
        
        self.company = company_name
        self.ticker = ticker

        # Debug
        print(f">> Company name: {company_name}, Ticker: ${ticker}")
        
        data = self.fetch_stock_data(ticker)
        formatted_prompt = self.prompt.format_messages(
            symbol = ticker, data = data, query = self.query
        )

        # print("Formatted LLM Prompt:", formatted_prompt)  # Debugging
        response = self.model.invoke(formatted_prompt).content.strip()
        # print("Raw LLM Response:", response)  # Debugging
        # response = self.model.invoke(formatted_prompt).content.strip()
        if not response:
            print("Response is empty.")
        else:
            return response
        # return self.model.invoke(formatted_prompt).content.strip()
        