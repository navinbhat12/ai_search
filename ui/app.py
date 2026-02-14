"""
Chatbot UI for the RAG search system. Ask questions and see the final answer,
with backend pipeline (query → agent routing → retrieval → prompt augmentation → response) visible.
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from agents.router_agent import (
    ask_router,
    get_router_trace_log,
    get_router_pipeline_steps,
)


def _truncate(s: str, max_len: int = 1200) -> str:
    if not isinstance(s, str) or len(s) <= max_len:
        return s
    return s[:max_len] + "\n\n... [truncated]"


st.set_page_config(page_title="RAG AI Search", page_icon="🔍", layout="centered")

st.title("🔍 RAG AI Search")
st.caption("Ask a question. Answers use SQL (basketball stats), vector (rules/backstories), or graph (teams/players/coaches).")

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_pipeline_steps" not in st.session_state:
    st.session_state.last_pipeline_steps = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input at bottom
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_router(prompt)
            st.session_state.last_trace = get_router_trace_log()
            st.session_state.last_pipeline_steps = get_router_pipeline_steps()
            st.session_state.last_response = answer
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Backend pipeline: query → agent → retrieval → prompt augmentation → final response
if st.session_state.last_trace:
    st.divider()
    with st.expander("📋 Backend: Query → Response pipeline", expanded=True):
        trace = st.session_state.last_trace
        pipeline_steps = st.session_state.last_pipeline_steps
        response = st.session_state.last_response

        # 1. Query received
        query_events = [e for e in trace if e.get("event_type") == "query_received"]
        if query_events:
            st.subheader("1. Query")
            st.text(query_events[0].get("data", {}).get("query", ""))

        # 2. Agent processing & retrieval (how the agent decides and what the retriever returns)
        if pipeline_steps:
            st.subheader("2. Agent routing & retrieval")
            for i, step in enumerate(pipeline_steps, 1):
                thought = step.get("thought", "")
                action = step.get("action", "")
                action_input = step.get("action_input", "")
                observation = step.get("observation", "")
                st.markdown(f"**Step {i}**")
                st.markdown("**Agent reasoning (Thought / Decision):**")
                st.text(_truncate(thought))
                st.markdown(f"**Chosen tool:** `{action}`  \n**Input to tool:** {action_input}")
                st.markdown("**Retrieval result (from DB):**")
                st.text(_truncate(observation))
                st.markdown("---")

        # 3. Prompt augmentation (if synthesis was used)
        aug_events = [e for e in trace if e.get("event_type") == "prompt_augmented"]
        if aug_events:
            st.subheader("3. Prompt augmentation")
            data = aug_events[0].get("data", {})
            st.markdown("**Original query:**")
            st.text(data.get("query", ""))
            st.markdown("**Retrieved info passed to LLM:**")
            st.text(_truncate(data.get("retrieved_info", "")))
            st.markdown("**Full prompt sent for formatting:**")
            st.text(_truncate(data.get("prompt_sent", "")))
        else:
            st.subheader("3. Prompt augmentation")
            st.caption("Final answer was produced directly by the agent (no separate formatting step).")

        # 4. Final response
        st.subheader("4. Final response")
        st.text(response)

        # Raw trace (collapsed) for debugging / console
        with st.expander("Raw trace (for debugging / console)", expanded=False):
            lines = []
            for event in trace:
                ts = event.get("timestamp", "")[:19].replace("T", " ")
                etype = event.get("event_type", "")
                data = event.get("data", {})
                lines.append(f"[{ts}] {etype}")
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > 300:
                        v = v[:300] + "..."
                    lines.append(f"  {k}: {v}")
                lines.append("")
            trace_text = "\n".join(lines)
            st.text(trace_text)
            st.caption("To log in browser console (F12 → Console):")
            st.code('console.log("=== RAG Trace ===\\n", ' + json.dumps(trace_text) + ');', language="javascript")
