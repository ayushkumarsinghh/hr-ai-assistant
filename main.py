from graph import build_graph

app = build_graph()

def ask(q):
    state = {
        "question": q,
        "messages": [],
        "eval_retries": 0
    }
    result = app.invoke(state)
    return result["answer"]

while True:
    q = input("You: ")
    print("Bot:", ask(q))