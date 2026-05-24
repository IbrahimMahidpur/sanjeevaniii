"""Prompt builder placeholder for RAG.

Will format the final LLM prompt given query and retrieved docs.
"""

def build_medical_prompt(query: str, retrieved_docs: list, language: str = "en") -> str:
    # Simple placeholder implementation
    context_str = "\n".join([f"[Context {i+1}] {doc.get('text','')}``" for i, doc in enumerate(retrieved_docs)])
    prompt = (
        f"You are a helpful medical assistant. Answer the following question using the provided context.\n"
        f"{context_str}\n"
        f"Question: {query}\n"
        f"Note: Always consult a qualified doctor for medical advice."
    )
    return prompt
