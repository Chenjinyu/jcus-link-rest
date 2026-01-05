"""Simple in-memory rate limiter for MCP endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import settings


@dataclass
class _RateLimitState:
    window_start: float
    count: int = 0


class MCPRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply a basic per-IP rate limit to MCP routes."""

    def __init__(self, app, path_prefix: str = "/mcp") -> None:
        super().__init__(app)
        self._path_prefix = path_prefix
        self._states: Dict[str, _RateLimitState] = {}
        self._limit = settings.rate_limit_requests
        self._window = settings.rate_limit_period

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        if not request.url.path.startswith(self._path_prefix):
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        now = time.time()
        state = self._states.get(client_host)

        if state is None or now - state.window_start >= self._window:
            state = _RateLimitState(window_start=now, count=0)
            self._states[client_host] = state

        state.count += 1

        if state.count > self._limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "MCP rate limit exceeded",
                    "limit": self._limit,
                    "period_seconds": self._window,
                },
            )

        return await call_next(request)
