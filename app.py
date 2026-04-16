import streamlit as st
from graph import build_graph

app = build_graph()

st.set_page_config(page_title="HR Assistant", layout="centered")

st.title("💼 AI HR Assistant")
st.caption("Ask about leave, salary, attendance, and policies.")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.chat.append(("user", user_input))

    state = {
        "question": user_input,
        "messages": [],
        "eval_retries": 0
    }

    result = app.invoke(state)
    answer = result["answer"]

    st.session_state.chat.append(("bot", answer))

for role, msg in st.session_state.chat:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)