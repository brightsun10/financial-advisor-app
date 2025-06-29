import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAIError, RateLimitError, AuthenticationError
from langchain.agents import AgentExecutor, create_react_agent
# from langchain.agents.agent_toolkits.react.base import ReActPrompt
from langchain_core.prompts import ChatPromptTemplate
# from langchain.memory import RedisChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import hashlib
import json
import redis

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(redis_url)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    key = f"user:{username}"
    if r.exists(key):
        return False, "Username already exists."
    r.set(key, json.dumps({
        "username": username,
        "email": email,
        "password": hash_password(password)
    }))
    return True, "Registration successful."

def login_user(username, password):
    key = f"user:{username}"
    if not r.exists(key):
        return False, "User does not exist."
    user_data = json.loads(r.get(key))
    if user_data["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, user_data


# Import our custom tool
from tools import (
    get_stock_price,
    update_user_profile,
    get_financial_news,
    get_portfolio_analysis,
)

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Ensure you have GOOGLE_API_KEY and REDIS_URL in your .env file
# Example REDIS_URL: redis://localhost:6379/0
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- AGENT SETUP ---

# 1. Define the tools the agent can use
tools = [
    get_stock_price,
    update_user_profile,
    get_financial_news,
    get_portfolio_analysis,
]

# 2. Create the LLM instance
# We use Gemini Flash for speed and cost-effectiveness
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)



def get_llm():
    #try:
     #   if openai_api_key:
     #       st.info("🔌 Trying OpenAI GPT-3.5...")
     #       return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    #except (AuthenticationError, OpenAIError, RateLimitError, Exception) as e:
    #    st.warning(f"OpenAI failed: {e}. Switching to Gemini...")

    if GOOGLE_API_KEY:
        st.info("🔁 Using Gemini 1.5 Flash")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

    raise ValueError("❌ No working LLM API key available.")

llm = get_llm()

# 3. Create the prompt template
# This is a crucial step to instruct the agent on how to behave.
# It uses the ReAct (Reasoning and Acting) framework prompt structure.
prompt_template = """
You are a helpful Financial Advisor AI. Your goal is to provide accurate and timely financial information.  

You can:
- Analyze user profiles
- Fetch stock prices
- Fetch financial news
- Generate weekly summaries

Always consider:
- User's profile (income, expenses, risk tolerance) — accessible via update_user_profile and get_portfolio_analysis. Use this to tailor investment suggestions appropriately.
- Relevant company news — use get_financial_news

You have access to the following tools:

{tools}

Use the following format for your responses:

Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Be sure to output a Final Answer after using the necessary tools.

Here is an example:

Question: What is the current price of AAPL?
Thought: I should look up the price of AAPL
Action: get_stock_price
Action Input: ["AAPL"] or "AAPL" or AAPL or Apple
Observation: {{'AAPL': '$172.34'}}
Thought: I now know the final answer
Final Answer: The current price of AAPL is $172.34

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# prompt = ReActPrompt().from_llm_and_tools(llm=llm, tools=tools)

# 4. Create the ReAct Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)


# 5. Create the Agent Executor, which runs the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Add Memory to the Agent
# This allows the agent to remember past conversations.
# Each user gets a unique session_id.
def get_chat_history(session_id: str):
    return RedisChatMessageHistory(session_id, url=REDIS_URL)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Financial Advisor", page_icon="📈")
st.title("📈 AI Financial Advisor")
st.caption("Your personal AI assistant for financial queries.")

# Initialize or get the session ID
if "session_id" not in st.session_state:
    # A simple way to get a unique ID for the session
    st.session_state.session_id = str(os.urandom(24).hex())

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.subheader("👤 User Login / Register")

    auth_mode = st.radio("Choose", ["Login", "Register"])
    username = st.text_input("Username")
    email = st.text_input("Email") if auth_mode == "Register" else None
    password = st.text_input("Password", type="password")

    if st.button("🔐 Submit"):
        if auth_mode == "Register":
            success, message = register_user(username, email, password)
            st.success(message) if success else st.error(message)
        else:
            success, result = login_user(username, password)
            if success:
                st.session_state["username"] = result["username"]
                st.session_state["session_id"] = st.session_state.get("session_id") or os.urandom(16).hex()
                r.set(f"session:{st.session_state.session_id}", result["username"])
                st.success(f"Welcome back, {result['username']}!")
            else:
                st.error(result)

    st.header("🧾 Your Financial Profile")
    income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
    expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=20000)
    risk = st.select_slider("Risk Tolerance", options=["Low", "Medium", "High"])

    if st.button("💾 Save Profile"):
        if "username" not in st.session_state:
            st.warning("Please log in to save your profile.")
        else:
            profile_data = {
                "income": income,
                "expenses": expenses,
                "risk_tolerance": risk,
                "username": st.session_state["username"],
                "session_id": st.session_state["session_id"]
            }
            result = update_user_profile.invoke({"profile": profile_data})
            st.success(result)



# Handle new user input
if prompt := st.chat_input("Ask about stock prices (e.g., What's the price of TSLA and MSFT?)"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_with_chat_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            st.markdown(response["output"])
    
    # Add assistant response to UI
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

if st.button("📊 Generate Weekly Summary"):
    with st.spinner("Analyzing your financial profile..."):
        prompt = f"Generate a weekly financial summary for user with session ID {st.session_state.session_id}."
        response = agent_with_chat_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )
        st.chat_message("assistant").markdown(response["output"])
        st.session_state.messages.append({
            "role": "assistant", "content": response["output"]
        })
