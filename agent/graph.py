"""LangGraph definition for the Medical AI Chatbot.

- Uses ``StateGraph`` with the ``MedicalState`` TypedDict.
- Adds nodes: ``intent``, ``retrieve``, ``generate``, ``validate``, ``respond``.
- Connects them in the sequence:
    intent → retrieve → generate → validate → respond → END
- Adds a conditional edge from ``intent``: if ``state['intent'] == 'emergency'`` the graph skips the ``retrieve`` node and goes directly to ``generate``.

The compiled graph is returned by ``build_graph()`` and can be used with ``graph.ainvoke(state)``.
"""

from langgraph.graph import StateGraph, START, END

from .state import MedicalState
from .nodes.intent import intent_node
from .nodes.retrieve import retrieve_node
from .nodes.generate import generate_node
from .nodes.validate import validate_node
from .nodes.respond import respond_node


def build_graph():
    """Construct and compile the LangGraph for the medical chatbot.

    Returns:
        CompiledGraph: the ready‑to‑invoke graph.
    """
    graph = StateGraph(MedicalState)

    # Register nodes
    graph.add_node("intent", intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("respond", respond_node)

    # Normal flow
    graph.add_edge(START, "intent")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "respond")
    graph.add_edge("respond", END)

    # Conditional shortcut for emergency intent – skip retrieval
    def is_emergency(state: MedicalState) -> str:
        return "generate" if state.get("intent") == "emergency" else "retrieve"

    graph.add_conditional_edges("intent", is_emergency, {"retrieve": "retrieve", "generate": "generate"})

    # Compile the graph
    compiled = graph.compile()
    return compiled

# When the module is executed directly, print an ASCII representation of the graph for verification.
if __name__ == "__main__":
    g = build_graph()
    try:
        print(g.get_graph().draw_ascii())
    except Exception:
        # ``get_graph`` may not be available in some runtime versions; ignore.
        pass
