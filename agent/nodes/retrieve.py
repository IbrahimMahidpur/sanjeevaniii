"""Retrieve node for the medical chatbot.

* Uses the ``HybridRetriever`` (Phase 3) to fetch relevant documents based on the user query.
* If ``state['multimodal_context']`` is present, it is prepended to the query before retrieval.
* The retrieved documents are stored in ``state['retrieved_docs']`` as a list of dictionaries
  containing at least ``text`` and ``source`` fields.
"""

from typing import Any, Dict, List

from rag.retriever import HybridRetriever

async def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    # Prepend any multimodal context (e.g., parsed image or PDF text)
    multimodal = state.get("multimodal_context")
    if multimodal:
        query = f"{multimodal} {query}".strip()
    retriever = HybridRetriever()
    docs: List[Dict] = retriever.retrieve(query, top_k=5)
    state["retrieved_docs"] = docs
    return state
