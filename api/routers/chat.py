"""
Chat router for the Medical AI Chatbot.

Provides:
- POST /chat   – synchronous endpoint that accepts a user message (and optional file) and returns a full response.
- GET  /chat/stream – SSE streaming endpoint that yields tokens as they are generated.

The implementation follows Phase 7 specifications while keeping external dependencies minimal.
"""

import os
import base64
import json
from typing import Optional, Dict, Any

import redis.asyncio as redis
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Lazy import of the LangGraph graph to avoid circular imports.

def get_graph():
    from agent.graph import build_graph
    return build_graph()

router = APIRouter()

# Simple Pydantic models for request/response payloads
class ChatRequest(BaseModel):
    message: str
    session_id: str
    file: Optional[str] = None  # base64‑encoded file content
    filename: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str

# Helper to get or create a Redis session store
async def get_redis_client() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return redis.from_url(redis_url, decode_responses=True)

async def load_session(session_id: str, r: redis.Redis) -> Dict[str, Any]:
    data = await r.get(session_id)
    if data:
        return json.loads(data)
    # New session skeleton
    return {"messages": [], "session_id": session_id}

async def save_session(session_id: str, state: Dict[str, Any], r: redis.Redis):
    await r.set(session_id, json.dumps(state))

# Placeholder parser dispatcher – in a full implementation this would call multimodal parsers.
async def dispatch_file(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".webp"}:
        from multimodal.image_parser import parse_image
        return await parse_image(file_bytes, filename)
    if ext == ".pdf":
        from multimodal.pdf_parser import parse_pdf
        return await parse_pdf(file_bytes)
    return file_bytes.decode(errors="ignore")

@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Auth is handled by middleware; we trust the request here.
    r = await get_redis_client()
    session = await load_session(req.session_id, r)

    multimodal_context = None
    if req.file and req.filename:
        try:
            print(f"[chat.py] File received: {req.filename}, size: {len(req.file)} chars")
            file_bytes = base64.b64decode(req.file)
            print(f"[chat.py] Decoded bytes: {len(file_bytes)}")
            multimodal_context = await dispatch_file(file_bytes, req.filename)
            print(f"[chat.py] Multimodal context length: {len(multimodal_context)}")
        except Exception as e:
            print(f"[chat.py] File error: {e}")
            multimodal_context = f"File processing failed: {str(e)}"

    # Build initial state for the LangGraph agent.
    state = {
        "messages": session.get("messages", []),
        "query": req.message,
        "session_id": req.session_id,
        "multimodal_context": multimodal_context,
        "intent": "",
        "retrieved_docs": [],
        "generated_response": "",
        "validated_response": "",
        "user_language": "en",
        "error": None,
    }

    graph = get_graph()
    try:
        result_state = await graph.ainvoke(state)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # Persist updated session messages.
    session["messages"] = result_state.get("messages", [])
    await save_session(req.session_id, session, r)

    return ChatResponse(
        response=result_state.get("validated_response", ""),
        session_id=req.session_id,
        intent=result_state.get("intent", "general"),
    )

# SSE streaming endpoint – uses the Ollama streaming implementation.
@router.get("/stream")
async def chat_stream(request: Request):
    """SSE streaming endpoint.

    Expected query parameters:
    - message (str)
    - session_id (str)
    - file (optional base64 string)
    - filename (optional string)
    """
    params = request.query_params
    message = params.get("message")
    session_id = params.get("session_id")
    file_b64 = params.get("file")
    filename = params.get("filename")
    if not message or not session_id:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    r = await get_redis_client()
    session = await load_session(session_id, r)
    multimodal_context = None
    if file_b64 and filename:
        try:
            file_bytes = base64.b64decode(file_b64)
            multimodal_context = await dispatch_file(file_bytes, filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file payload: {e}")

    state = {
        "messages": session.get("messages", []),
        "query": message,
        "session_id": session_id,
        "multimodal_context": multimodal_context,
        "intent": "",
        "retrieved_docs": [],
        "generated_response": "",
        "validated_response": "",
        "user_language": "en",
        "error": None,
    }

    # Import the streaming helper from the channel module.
    from channels.web import stream_response

    # The generator yields properly formatted SSE events.
    async def event_generator():
        async for chunk in stream_response(state):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")
