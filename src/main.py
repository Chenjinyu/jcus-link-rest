from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

from src.config.settings import settings
from src.api.health import router as health_router
from src.mcp.rate_limit import MCPRateLimitMiddleware
from src.mcp.prompts import register_prompts
from src.mcp.resources import register_resources
from src.mcp.tools import register_tools


def create_mcp() -> FastMCP:
    mcp = FastMCP(settings.app_name, version=settings.app_version)
    register_tools(mcp)
    register_prompts(mcp)
    register_resources(mcp)
    return mcp


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.add_middleware(MCPRateLimitMiddleware, path_prefix="/mcp")

    app.include_router(health_router, prefix="/api")

    mcp = create_mcp()
    app.state.mcp = mcp

    app.mount("/mcp", mcp.http_app(transport="streamable-http"))

    return app


app = create_app()
