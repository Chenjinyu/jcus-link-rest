from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP
from starlette.routing import Route

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
    app_env = settings.app_env or os.getenv("APP_ENV", "development")
    print(f"Starting {settings.app_name} (version: {settings.app_version}) in {app_env}")
    mcp = create_mcp()
    # FastMCP defaults to /mcp; when mounted at /mcp we need a root path here.
    mcp_app = mcp.http_app(
        path="/",
        transport="streamable-http",
        stateless_http=True, # FastMCP’s streamable HTTP transport is session‑based by default, so it expects an mcp-session-id. To make your curl calls work without sessions, I set it to stateless mode.
    )

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=mcp_app.lifespan,
    )
    # Disable automatic trailing-slash redirects for all routes (including mounts).
    app.router.redirect_slashes = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.add_middleware(MCPRateLimitMiddleware, path_prefix="/mcp")

    app.include_router(health_router, prefix="/api")

    app.state.mcp = mcp
    # Avoid 307 redirects on POST /mcp by disabling slash redirects on the sub-app.
    mcp_app.router.redirect_slashes = False
    # Allow both /mcp and /mcp/ without redirects by proxying /mcp to the sub-app root.
    class _McpRootProxy:
        def __init__(self, inner_app) -> None:
            self._app = inner_app

        async def __call__(self, scope, receive, send) -> None:
            # only the scope['schema'] has http and https
            if scope.get("type") == "http":
                scope = dict(scope)
                scope["path"] = "/"
                scope["raw_path"] = b"/"
            await self._app(scope, receive, send)
            
    # • FastMCP’s streamable HTTP transport uses multiple verbs:
    # - POST is for JSON‑RPC calls.
    # - GET is used for streamable polling/resume behavior.
    # - DELETE is used to terminate sessions.
    # - OPTIONS is for CORS preflight when browsers call POST.
    app.router.routes.append(
        Route(
            "/mcp",
            _McpRootProxy(mcp_app),
            methods=["GET", "POST", "DELETE", "OPTIONS"],
        )
    )
    app.mount("/mcp", mcp_app)
    return app

app = create_app()
