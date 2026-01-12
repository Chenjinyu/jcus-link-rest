#!/usr/bin/env python3
"""
Run the FastAPI REST server locally
"""
import uvicorn
from src.main import app
from src.config.settings import settings

if __name__ == "__main__":
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"📚 API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"🔧 MCP Endpoint: http://{settings.host}:{settings.port}/mcp")
    print(f"❤️  Health Check: http://{settings.host}:{settings.port}/api/health")
    print(f"🔑 LLM Provider: {settings.default_llm_provider}")
    print(f"💾 Vector DB: {settings.vector_db_type}")
    print("-" * 60)
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )

