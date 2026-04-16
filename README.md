# 💼 AI HR Assistant

This is a retrieval-based HR chatbot built using LangGraph, ChromaDB, and Streamlit.

## Features
- Answer HR-related queries (leave, salary, attendance)
- Memory handling (remembers user name)
- Tool integration (current date/time)
- No hallucination (retrieval-based responses)

## Tech Stack
- LangGraph
- Sentence Transformers
- ChromaDB
- Streamlit

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
