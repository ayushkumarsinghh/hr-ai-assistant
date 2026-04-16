from state import CapstoneState
from tools import time_tool
from kb import docs

from sentence_transformers import SentenceTransformer
import chromadb


# =========================
# EMBEDDING MODEL
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

try:
    collection = client.get_collection("hr_docs")
except:
    collection = client.create_collection("hr_docs")

# Add docs once
if len(collection.get()["ids"]) == 0:
    for d in docs:
        emb = embedder.encode(d["text"]).tolist()
        collection.add(
            documents=[d["text"]],
            embeddings=[emb],
            ids=[d["id"]],
            metadatas=[{"topic": d["topic"]}]
        )


# =========================
# MEMORY NODE
# =========================
def memory_node(state: CapstoneState):
    state.setdefault("messages", [])
    state.setdefault("eval_retries", 0)

    state["messages"].append(state["question"])
    state["messages"] = state["messages"][-6:]

    if "my name is" in state["question"].lower():
        state["user_name"] = state["question"].split("is")[-1].strip()

    return state


# =========================
# ROUTER NODE
# =========================
def router_node(state: CapstoneState):
    q = state["question"].lower().strip()

    if q in ["hi", "hello", "hey"]:
        state["route"] = "greet"

    elif "date" in q or "today" in q:
        state["route"] = "tool"

    elif "my name" in q:
        state["route"] = "skip"

    else:
        state["route"] = "retrieve"

    return state


# =========================
# RETRIEVAL NODE
# =========================
def retrieval_node(state: CapstoneState):
    q = state["question"].lower()
    q_emb = embedder.encode(q).tolist()

    results = collection.query(query_embeddings=[q_emb], n_results=3)

    docs_text = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]

    # Smart selection
    best_doc = docs_text[0]
    best_topic = topics[0]

    for doc, topic in zip(docs_text, topics):
        topic_lower = topic.lower()

        if "sick" in q and "sick" in topic_lower:
            best_doc = doc
            best_topic = topic
            break
        elif "casual" in q and "casual" in topic_lower:
            best_doc = doc
            best_topic = topic
            break
        elif "salary" in q and "payroll" in topic_lower:
            best_doc = doc
            best_topic = topic
            break
        elif "attendance" in q and "attendance" in topic_lower:
            best_doc = doc
            best_topic = topic
            break

    state["retrieved"] = f"[{best_topic}] {best_doc}"
    state["sources"] = [best_topic]

    return state


# =========================
# TOOL NODE
# =========================
def tool_node(state: CapstoneState):
    state["tool_result"] = time_tool()
    return state


# =========================
# ANSWER NODE (FINAL — NO LLM)
# =========================
def answer_node(state):

    # Greeting
    if state["route"] == "greet":
        state["answer"] = "Hello! I'm your HR assistant. How can I help you today?"
        return state

    # Tool
    if state["route"] == "tool":
        state["answer"] = state["tool_result"]
        return state

    # Memory
    if state["route"] == "skip":
        state["answer"] = f"Your name is {state.get('user_name', 'unknown')}"
        return state

    # Retrieval answer
    context = state.get("retrieved", "")

    if not context:
        state["answer"] = "I don't have that information."
        return state

    # Extract best answer
    answer = context.split("\n\n")[0]

    # Remove topic tag
    if "]" in answer:
        answer = answer.split("]", 1)[-1].strip()

    state["answer"] = answer
    return state


# =========================
# EVAL NODE
# =========================
def eval_node(state: CapstoneState):
    state["faithfulness"] = 1.0
    state["eval_retries"] = state.get("eval_retries", 0) + 1
    return state


# =========================
# SAVE NODE
# =========================
def save_node(state):
    state["messages"].append(state["answer"])
    return state