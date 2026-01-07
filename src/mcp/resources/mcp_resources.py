"""MCP resource registrations."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server import Context

from src.config import settings

PROMPT_CATALOG: dict[str, dict[str, str]] = {
    "resume_generation_prompt": {
        "name": "resume_generation_prompt",
        "description": "Generate a tailored resume from a job description and matches.",
    },
    "job_analysis_prompt": {
        "name": "job_analysis_prompt",
        "description": "Analyze a job description into structured requirements.",
    },
    "resume_from_source_prompt": {
        "name": "resume_from_source_prompt",
        "description": "Generate a resume from structured resume source data.",
    },
}


def register_resources(mcp: FastMCP) -> None:
    @mcp.resource("resource://mcp/server-info")
    async def server_info(ctx: Context | None = None) -> dict[str, str]:
        if ctx:
            await ctx.info("Providing MCP server info")
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "protocol_version": settings.mcp_protocol_version,
        }

    @mcp.resource("resource://mcp/prompts")
    async def prompt_catalog(
        ctx: Context | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        if ctx:
            await ctx.info("Providing prompt catalog")
        return {"prompts": list(PROMPT_CATALOG.values())}

    @mcp.resource("resource://mcp/prompts/{prompt_name}")
    async def prompt_details(
        prompt_name: str,
        ctx: Context | None = None,
    ) -> dict[str, str]:
        if ctx:
            await ctx.info(f"Providing details for prompt {prompt_name}")
        prompt = PROMPT_CATALOG.get(prompt_name)
        if not prompt:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        return prompt
