from langgraph.graph import StateGraph, END
from state import CapstoneState
from nodes import *

# Route function
def route_decision(state):
    return state["route"]


def build_graph():
    graph = StateGraph(CapstoneState)

    # Nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    # Entry
    graph.set_entry_point("memory")

    # Flow
    graph.add_edge("memory", "router")

    # 🔥 FIXED ROUTING (includes greet)
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "skip": "answer",
            "greet": "answer"   # ✅ IMPORTANT FIX
        }
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("tool", "answer")

    graph.add_edge("answer", "eval")
    graph.add_edge("eval", "save")

    graph.add_edge("save", END)

    return graph.compile()