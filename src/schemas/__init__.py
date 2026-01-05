# src/models/__init__.py
"""Data models module"""

from .domain_schema import (
    ResumeMatch,
    JobAnalysis,
    SearchMatchesRequest,
    SearchMatchesResponse,
    AnalyzeJobRequest,
    GenerateResumeRequest,
    UploadJobResponse,
    ResumeData,
)
from .mcp_schema import (
    MCPMessageType,
    MCPMethod,
    MCPRequest,
    MCPError,
    MCPResponse,
    ToolInput,
    Tool,
    TextContent,
    ImageContent,
    ToolResponse,
    ServerInfo,
    ServerCapabilities,
    InitializeResult,
    ToolsListResult,
    ToolCallParams,
)

__all__ = [
    # Domain Schemas
    "ResumeMatch",
    "JobAnalysis",
    "SearchMatchesRequest",
    "SearchMatchesResponse",
    "AnalyzeJobRequest",
    "GenerateResumeRequest",
    "UploadJobResponse",
    "ResumeData",
    # MCP Protocol Schemas
    "MCPMessageType",
    "MCPMethod",
    "MCPRequest",
    "MCPError",
    "MCPResponse",
    "ToolInput",
    "Tool",
    "TextContent",
    "ImageContent",
    "ToolResponse",
    "ServerInfo",
    "ServerCapabilities",
    "InitializeResult",
    "ToolsListResult",
    "ToolCallParams",
]