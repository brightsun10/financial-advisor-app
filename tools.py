import yfinance as yf
from langchain.tools import tool
from typing import List
import ast  # to safely evaluate string list
from typing import Union
import redis
import os

@tool
def get_stock_price(tickers: Union[str, list]) -> dict:
    """
    Fetch current stock prices. Accepts a list of tickers or a stringified list.
    """
    if isinstance(tickers, str):
        try:
            tickers = ast.literal_eval(tickers)
        except Exception as e:
            return {"error": f"Could not parse input: {tickers}. Error: {e}"}

    if not isinstance(tickers, list):
        return {"error": "tickers must be a list"}

    import yfinance as yf
    results = {}
    for ticker_symbol in tickers:
        try:
            stock = yf.Ticker(ticker_symbol)
            price = stock.info.get('regularMarketPrice')
            if price:
                results[ticker_symbol] = f"${price:.2f}"
            else:
                results[ticker_symbol] = f"No data found for '{ticker_symbol}'"
        except Exception as e:
            results[ticker_symbol] = f"Error fetching '{ticker_symbol}': {e}"
    return results



redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(redis_url)
        
@tool
def update_user_profile(profile: dict) -> str:
    """
    Stores user's financial profile into Redis.
    Must include 'income', 'expenses', 'risk_tolerance', and 'username'.
    """
    try:
        import json
        username = profile.get("username")
        if not username:
            return "Username missing in profile."
        key = f"profile:{username}"
        r.set(key, json.dumps(profile))
        return f"Profile saved for {username}"
    except Exception as e:
        return f"Failed to save profile: {e}"



# You can add more financial tools here in later phases.
# For example:
# @tool
# def get_financial_news(company_name: str) -> List[str]:
#     """Fetches recent financial news for a given company."""
#     # ... implementation using a news API or web scraping ...
#     pass

import requests

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from typing import List

@tool
def get_financial_news(company_name: str) -> List[str]:
    """
    Scrape recent news headlines related to a company from Google News.
    """
    try:
        query = company_name + " stock"
        url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        headlines = []

        # Each headline is inside <a> within an <article>
        articles = soup.find_all("article")
        for article in articles[:5]:  # Get top 5 headlines
            title_tag = article.find("a")
            if title_tag:
                title = title_tag.text.strip()
                headlines.append(title)

        if not headlines:
            return [f"No recent headlines found for {company_name}."]

        return headlines

    except Exception as e:
        return [f"Error fetching news for {company_name}: {e}"]


@tool
def get_portfolio_analysis(session_id: str) -> str:
    """
    Analyzes user's portfolio and provides summary (dummy logic).
    """
    try:
        import json
        key = f"profile:{session_id}"
        profile = r.get(key)

        if not profile:
            return "User profile not found."

        profile = json.loads(profile)
        income = profile.get("income")
        expenses = profile.get("expenses")
        risk = profile.get("risk_tolerance")

        savings = income - expenses
        suggestion = {
            "Low": "Consider fixed deposits or government bonds.",
            "Medium": "You could invest in balanced mutual funds.",
            "High": "You can explore stocks, crypto, or high-growth funds."
        }.get(risk, "No suggestion available.")

        return f"""
                Portfolio Summary:
                - Savings: ₹{savings}
                - Risk Tolerance: {risk}
                - Suggested Strategy: {suggestion}
                """
    except Exception as e:
        return f"Error in portfolio analysis: {e}"



