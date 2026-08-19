from __future__ import annotations

from langgraph.graph import END, StateGraph

from src import config
from src.agent.nodes import generate_node, grade_node, retrieve_node, rewrite_node
from src.agent.state import GraphState


def _route_after_grade(state: GraphState) -> str:
    if state.get("grade_passed"):
        return "generate"
    if state.get("retry_count", 0) >= config.MAX_RETRIES:
        # Never let the loop run unbounded: force generate with an explicit
        # insufficient-context instruction instead of retrying again.
        return "generate"
    return "rewrite"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", _route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def ask(question: str) -> dict:
    app = get_app()
    initial: GraphState = {
        "question": question,
        "rewritten_question": None,
        "documents": [],
        "retry_count": 0,
        "answer": "",
        "citations": [],
    }
    return app.invoke(initial)
