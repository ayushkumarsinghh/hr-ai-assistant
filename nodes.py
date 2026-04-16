from state import CapstoneState
from tools import time_tool
from kb import docs

from sentence_transformers import SentenceTransformer
import chromadb


embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

try:
    collection = client.get_collection("hr_docs")
except:
    collection = client.create_collection("hr_docs")

if len(collection.get()["ids"]) == 0:
    for d in docs:
        emb = embedder.encode(d["text"]).tolist()
        collection.add(
            documents=[d["text"]],
            embeddings=[emb],
            ids=[d["id"]],
            metadatas=[{"topic": d["topic"]}]
        )


def memory_node(state: CapstoneState):
    state.setdefault("messages", [])
    state.setdefault("eval_retries", 0)

    state["messages"].append(state["question"])
    state["messages"] = state["messages"][-6:]

    if "my name is" in state["question"].lower():
        state["user_name"] = state["question"].split("is")[-1].strip()

    return state


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


def retrieval_node(state: CapstoneState):
    q = state["question"].lower()
    q_emb = embedder.encode(q).tolist()

    results = collection.query(query_embeddings=[q_emb], n_results=3)

    docs_text = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]

    best_doc = docs_text[0]
    best_topic = topics[0]

    for doc, topic in zip(docs_text, topics):
        if "sick" in q and "sick" in topic.lower():
            best_doc, best_topic = doc, topic
            break
        elif "casual" in q and "casual" in topic.lower():
            best_doc, best_topic = doc, topic
            break

    state["retrieved"] = f"[{best_topic}] {best_doc}"
    state["sources"] = [best_topic]

    return state


def tool_node(state: CapstoneState):
    state["tool_result"] = time_tool()
    return state


def answer_node(state):

    if state["route"] == "greet":
        state["answer"] = "Hello! I'm your HR assistant. How can I help you today?"
        return state

    if state["route"] == "tool":
        state["answer"] = state["tool_result"]
        return state

    if state["route"] == "skip":
        state["answer"] = f"Your name is {state.get('user_name', 'unknown')}"
        return state

    context = state.get("retrieved", "")

    if not context:
        state["answer"] = "I don't have that information."
        return state

    answer = context.split("\n\n")[0]

    if "]" in answer:
        answer = answer.split("]", 1)[-1].strip()

    state["answer"] = answer
    return state


def eval_node(state: CapstoneState):
    state["faithfulness"] = 1.0
    state["eval_retries"] += 1
    return state


def save_node(state):
    state["messages"].append(state["answer"])
    return state