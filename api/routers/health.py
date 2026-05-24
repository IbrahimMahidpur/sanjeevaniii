"""
Health check routers for the FastAPI app.

Provides:
- GET /health – basic liveness probe.
- GET /ready – readiness probe, verifies connectivity to Redis, Qdrant and Ollama.
"""

import os
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient

router = APIRouter()

@router.get("/health")
async def health() -> JSONResponse:
    """Simple liveness endpoint returning status and timestamp."""
    return JSONResponse({"status": "ok", "timestamp": int(time.time())})

@router.get("/ready")
async def ready() -> JSONResponse:
    """Readiness endpoint – ensures external services are reachable.

    Returns 200 if all checks pass, otherwise 503 with error details.
    """
    errors = []
    # Redis check
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url, decode_responses=True)
        await r.ping()
    except Exception as e:
        errors.append(f"Redis error: {e}")
    # Qdrant check
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        q = AsyncQdrantClient(url=qdrant_url)
        await q.get_collections()
    except Exception as e:
        errors.append(f"Qdrant error: {e}")
    # Ollama health check
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
    except Exception as e:
        errors.append(f"Ollama error: {e}")
    if errors:
        return JSONResponse({"status": "unhealthy", "errors": errors}, status_code=503)
    return JSONResponse({"status": "ready", "timestamp": int(time.time())})
