"""FastAPI application entry point for the Medical AI Chatbot.

Sets up the app with:
- CORS middleware (origins from CORS_ORIGINS env var)
- Custom middlewares: PIIScrubMiddleware, RateLimitMiddleware, AuthMiddleware
- Routers: health and chat
- Prometheus metrics endpoint at /metrics
"""

import os
import time
from typing import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .middleware import PIIScrubMiddleware, RateLimitMiddleware, AuthMiddleware
from .routers import health, chat

# Prometheus metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency seconds",
    ["method", "endpoint"]
)
LLM_RESPONSE_LATENCY = Histogram(
    "llm_response_latency_seconds",
    "LLM response latency seconds",
    []
)
RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "RAG retrieval latency seconds",
    []
)
ACTIVE_SESSIONS = Gauge("active_sessions", "Number of active chat sessions")

def create_app() -> FastAPI:
    app = FastAPI()

    # CORS — SABSE PEHLE add karo
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middlewares — CORS ke baad
    #app.add_middleware(PIIScrubMiddleware)
    #app.add_middleware(RateLimitMiddleware, limit=100, period_seconds=60)
    # app.add_middleware(AuthMiddleware)  # Disabled for dev

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics_endpoint():
        return generate_latest()

    # Middleware to record request metrics
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start = time.time()
        response = await call_next(request)
        process_time = time.time() - start
        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(process_time)
        return response

    # Lifespan events to track active sessions (simple heuristic)
    @app.on_event("startup")
    async def startup_event():
        # Could initialize connections here if needed
        pass

    @app.on_event("shutdown")
    async def shutdown_event():
        # Cleanup resources
        pass

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)
