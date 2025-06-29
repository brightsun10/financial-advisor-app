# AI Financial Advisor 

📈This project is a prototype of an autonomous financial advisor built with Python. It uses a powerful agentic AI framework to provide financial information and advice through an interactive chat interface.This agent is powered by Google's Gemini LLM, structured with the LangChain framework, and deployed as a user-friendly web application using Streamlit.

## Phase 1 FeaturesInteractive Chat Interface: A clean web UI built with Streamlit to chat with the agent.Agentic AI Core: Utilizes the LangChain framework to create a "ReAct" (Reasoning and Acting) agent that can make decisions.Custom Tools: The agent is equipped with a tool to fetch real-time stock prices using the yfinance library.Persistent Memory: It remembers your conversation history within a session using a Redis database, allowing for contextual follow-up questions.

## Tech Stack

- LLM: Google Gemini (gemini-1.5-flash)
- Agent Framework: LangChain
- Memory: Redis
- Tools: yfinance for stock data
- Frontend/Deployment: Streamlit

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- Python 3.8+
- Redis: You must have a Redis instance running.

The easiest way to get one is with Docker:docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

- Google AI API Key: You need a valid API key for the Gemini model. You can get one from Google AI Studio.


## Project Setup

Follow these steps to get the application running on your local machine.

1. Create Project DirectoryCreate a folder for your project and place the app.py and tools.py files inside it.mkdir financial_advisor
cd financial_advisor
2. Create a Virtual EnvironmentIt is highly recommended to use a virtual environment to manage dependencies.

Create the environment

python -m venv venv

Activate it

On Windows:

venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

3. Install Dependencies

Create a requirements.txt file with the contents from the document provided, and then install all the necessary packages.

pip install -r requirements.txt

Configure Environment VariablesCreate a file named .env in the root of your project directory (financial_advisor/). 

This file will securely store your API key and Redis connection URL.Your .env file must contain the following:# .env

Your Google AI API key for Gemini

GOOGLE_API_KEY="your-google-api-key-here"

The connection URL for your running Redis instance

REDIS_URL="redis://localhost:6379/0"

5. Run the Application

You are now ready to start the AI Financial Advisor! Run the following command in your terminal:

streamlit run app.py

Your web browser should automatically open a new tab with the application running. If not, navigate to http://localhost:8501.How to UseSimply type your financial questions into the chat box at the bottom of the page. For this initial phase, the agent is best equipped to handle stock price lookups.Example questions:"What is the current stock price of Google?""Can you get me the prices for TSLA and AMZN?""How much is a share of AAPL?"

Project Structure

financial_advisor/

├── venv/                   # Virtual environment directory

├── .env                    # Environment variables (API Key, Redis URL)

├── app.py                  # The main Streamlit application file

├── tools.py                # Contains custom tools for the LangChain agent

└── requirements.txt        # List of Python dependencies
