# CLAUDE.md — Ollama Integration (gpt-oss:120b-cloud)

> This file updates the LLM backend from vLLM to Ollama with gpt-oss:120b-cloud.
> Read this alongside the main CLAUDE.md. This file OVERRIDES the LLM serving section.

---

## LLM Backend Change

| Old (vLLM) | New (Ollama) |
|---|---|
| vLLM server (Linux only) | Ollama (Windows native ✅) |
| Mistral-7B local | gpt-oss:120b-cloud (OpenAI hosted) |
| ~6GB VRAM needed | 0 VRAM (runs via cloud) |
| Complex setup | `ollama run gpt-oss:120b-cloud` |

---

## Prerequisites

### Step 1 — Install Ollama
Download from: https://ollama.com/download/windows
After install, verify:
```powershell
ollama --version
```

### Step 2 — Pull the model
```powershell
ollama pull gpt-oss:120b-cloud
```

### Step 3 — Add OpenAI API key
The `:cloud` tag routes through OpenAI's API. Add to `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key-here
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gpt-oss:120b-cloud
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
LLM_REASONING_EFFORT=medium
```

Get your OpenAI key at: https://platform.openai.com/api-keys

### Step 4 — Start Ollama server
```powershell
ollama serve
```
Verify it's running:
```powershell
curl http://localhost:11434/api/tags
```

---

## Code Changes Required

### 1. Update `agent/nodes/generate.py`

Replace the entire vLLM call with this Ollama implementation:

```python
"""
generate.py — LLM generation node using Ollama (gpt-oss:120b-cloud)
"""

import os
import httpx
import asyncio
from agent.state import MedicalState
from rag.prompt_builder import build_medical_prompt

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")
TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS", "1000"))
TIMEOUT         = 60


async def generate_node(state: MedicalState) -> MedicalState:
    """Call Ollama gpt-oss:120b-cloud and return generated response."""

    # Build prompt with RAG context
    prompt = build_medical_prompt(
        query=state["query"],
        retrieved_docs=state.get("retrieved_docs", []),
        language=state.get("user_language", "en"),
        multimodal_context=state.get("multimodal_context"),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Sanjeevani, a medical AI assistant. "
                "Provide accurate, evidence-based medical information. "
                "Always recommend consulting a qualified doctor. "
                "Never diagnose or prescribe. Be clear and compassionate."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Add conversation history if exists
    for msg in state.get("messages", [])[-6:]:  # last 3 turns
        if hasattr(msg, "type"):
            role = "user" if msg.type == "human" else "assistant"
            messages.insert(-1, {"role": role, "content": msg.content})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            # gpt-oss specific: reasoning effort (low/medium/high)
            "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "medium"),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            generated = data["message"]["content"]

    except httpx.TimeoutException:
        generated = (
            "I'm sorry, the response took too long. "
            "Please try again. If symptoms are severe, "
            "consult a doctor immediately."
        )
    except Exception as e:
        generated = (
            f"I encountered an error generating a response. "
            f"Please try again. For medical emergencies, call emergency services."
        )

    return {**state, "generated_response": generated}


async def generate_streaming_node(state: MedicalState):
    """
    Streaming version — yields tokens for SSE endpoint.
    Used by channels/web.py for real-time streaming to React frontend.
    """
    prompt = build_medical_prompt(
        query=state["query"],
        retrieved_docs=state.get("retrieved_docs", []),
        language=state.get("user_language", "en"),
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are Sanjeevani, a medical AI assistant. Always recommend consulting a qualified doctor.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "medium"),
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
```

---

### 2. Update `channels/web.py`

```python
"""
web.py — SSE streaming channel for React frontend
Uses Ollama streaming via generate_streaming_node
"""

import json
import asyncio
from typing import AsyncGenerator
from agent.nodes.generate import generate_streaming_node


async def stream_response(state: dict) -> AsyncGenerator[str, None]:
    """
    Stream tokens as SSE events.
    Format: data: {"token": "..."}\n\n
    Final: data: [DONE]\n\n
    """
    full_response = ""

    async for token in generate_streaming_node(state):
        full_response += token
        event = json.dumps({"token": token})
        yield f"data: {event}\n\n"

    # Send final complete response
    done_event = json.dumps({
        "done": True,
        "full_response": full_response
    })
    yield f"data: {done_event}\n\n"
    yield "data: [DONE]\n\n"
```

---

### 3. Update `api/routers/chat.py` — SSE endpoint

```python
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from channels.web import stream_response
from agent.graph import build_graph
from agent.state import MedicalState

router = APIRouter()
agent_graph = build_graph()


@router.post("/chat")
async def chat(body: dict):
    """Sync chat endpoint."""
    state = MedicalState(
        messages=[],
        query=body.get("message", ""),
        session_id=body.get("session_id", "default"),
        intent="",
        retrieved_docs=[],
        generated_response="",
        validated_response="",
        user_language="en",
        multimodal_context=None,
        error=None,
    )
    result = await agent_graph.ainvoke(state)
    return {
        "response": result["validated_response"],
        "session_id": body.get("session_id"),
        "intent": result.get("intent", "general"),
    }


@router.get("/chat/stream")
async def chat_stream(
    message: str = Query(...),
    session_id: str = Query("default"),
):
    """SSE streaming endpoint for React frontend."""
    state = {
        "query": message,
        "session_id": session_id,
        "messages": [],
        "retrieved_docs": [],
        "user_language": "en",
        "multimodal_context": None,
    }

    return StreamingResponse(
        stream_response(state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
```

---

### 4. Update `api/routers/health.py` — check Ollama

```python
import os
import httpx
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def ready():
    checks = {}

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            checks["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        checks["ollama"] = "unreachable"

    # Check Qdrant
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("http://localhost:6333/healthz")
            checks["qdrant"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        checks["qdrant"] = "unreachable"

    # Check Redis
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unreachable"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "services": checks}
```

---

## Running the Updated Stack

```powershell
# Terminal 1 — Start Ollama
ollama serve

# Terminal 2 — Verify model is available
ollama list
# Should show: gpt-oss:120b-cloud

# Terminal 3 — Start FastAPI
.\venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Terminal 4 — Start React frontend
cd frontend
npm run dev
```

---

## Test the Integration

```powershell
# Quick test via curl
curl -X POST http://localhost:8080/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\": \"What are the symptoms of diabetes?\", \"session_id\": \"test-1\"}"
```

Expected response:
```json
{
  "response": "Common symptoms of diabetes include... *Always consult a qualified doctor.*",
  "session_id": "test-1",
  "intent": "symptom_check"
}
```

---

## Why gpt-oss:120b-cloud for Medical Use

| Feature | Benefit for Medical Chatbot |
|---|---|
| 120B parameters | Higher accuracy on complex clinical questions |
| Chain-of-thought | Reasoning visible — easier to validate medical logic |
| Configurable reasoning effort | Use `high` for drug interactions, `low` for simple queries |
| Function calling | Native tool use for RAG retrieval integration |
| Structured outputs | Consistent JSON for intent classification |
| Apache 2.0 license | Safe for production deployment |

Set reasoning effort per intent in `agent/nodes/generate.py`:
```python
reasoning_map = {
    "emergency":    "high",
    "drug_info":    "high",
    "symptom_check":"medium",
    "lab_report":   "medium",
    "general":      "low",
}
effort = reasoning_map.get(state.get("intent", "general"), "medium")
```

---

## Resume Justification

With this setup your resume line becomes fully accurate:

```
"Built production Medical AI Chatbot (Sanjeevani): LangGraph multi-agent
pipeline with gpt-oss-120B (OpenAI open-weight, Apache 2.0), hybrid RAG
(dense BAAI/bge + BM25 + RRF reranking) on Qdrant, multimodal inputs
(PDF/voice/image via Whisper + PyMuPDF + LLaVA), multilingual support
(5 languages via NLLB-200), FastAPI async gateway with Redis sessions +
rate limiting + PII scrubbing, SSE token streaming, React/Tailwind frontend
— deployed via Docker Compose with Prometheus + Grafana monitoring."
```

Every word of this is buildable and verifiable with the current stack.

---

*LLM: gpt-oss:120b-cloud via Ollama | Backend: FastAPI | Frontend: React | RAG: Qdrant*
