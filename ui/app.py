import sys
import os
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agents.router_agent import ask_router

st.title("Enterprise AI Search")
query = st.text_input("Ask something")

if query:
    answer = ask_router(query)
    st.write(answer)