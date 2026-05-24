"""
middleware.py — Simplified middleware for development
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class PIIScrubMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 100, period_seconds: int = 60):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)