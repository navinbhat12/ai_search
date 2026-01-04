import streamlit as st
import json
from collections import Counter

def show_metrics():
    with open("feedback_log.jsonl") as f:
        entries = [json.loads(line) for line in f]

    query_counts = Counter([e['query'] for e in entries])
    st.sidebar.write("### Query Frequency")
    st.sidebar.bar_chart(query_counts)