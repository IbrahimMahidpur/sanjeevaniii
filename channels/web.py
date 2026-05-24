import json
import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")

SYSTEM_PROMPT = """You are Sanjeevani, an expert medical AI assistant.
IMPORTANT: Write ALL words completely - never split words (write "Fever" NOT "F ever").
Use proper markdown: ## for headers, ** for bold, - for bullets.
Be thorough, specific, and evidence-based.
Always end with: > Always consult a qualified healthcare provider."""

async def stream_response(state: dict):
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    multimodal = state.get("multimodal_context", "")

    print(f"[web.py] Model: {LLM_MODEL} | Query: {query[:50]}")

    # RAG context
    context = ""
    if retrieved_docs:
        parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):
            text = doc.get("text", doc.get("content", ""))
            if text:
                parts.append(f"[Reference {i}]: {text[:500]}")
        if parts:
            context = "\n\nMedical References:\n" + "\n\n".join(parts)

    # Multimodal context
    multimodal_section = ""
    if multimodal:
        multimodal_section = f"\n\nUploaded File Analysis:\n{multimodal}"

    user_message = f"""{context}{multimodal_section}

Patient Question: {query}

Provide a comprehensive response with:
## [Title]
### Sections (Symptoms/Causes/Treatment)
- **Bold** key terms
- Specific values and details
- Warning signs and when to seek care"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 2048,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "top_p": 0.9,
        },
    }

    full_response = ""
    try:
        print(f"[web.py] Connecting to {OLLAMA_BASE_URL}/api/chat")
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as response:
                print(f"[web.py] Status: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"[web.py] ERROR: {e}")
        yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"

    print(f"[web.py] Done. Length: {len(full_response)}")
    yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
    yield "data: [DONE]\n\n"
