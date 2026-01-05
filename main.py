from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

from src.config.settings import settings
from src.api.health import router as health_router


def create_mcp() -> FastMCP:
    return FastMCP(settings.app_name, version=settings.app_version)


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

    app.include_router(health_router)

    mcp = create_mcp()
    app.state.mcp = mcp

    mcp_app = getattr(mcp, "app", None)
    if mcp_app is None and hasattr(mcp, "asgi_app"):
        mcp_app = mcp.asgi_app()
    if mcp_app is None:
        mcp_app = mcp

    app.mount("/mcp", mcp_app)

    return app


app = create_app()
