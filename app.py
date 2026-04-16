import streamlit as st
from graph import build_graph

app = build_graph()

st.title("💼 AI HR Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask your HR question...")

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
    with st.chat_message(role):
        st.markdown(msg)