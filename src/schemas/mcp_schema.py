# src/models/mcp_schema.py
"""
MCP Protocol models and types (Pydantic models)
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List
from enum import Enum


class MCPMessageType(str, Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class MCPMethod(str, Enum):
    """MCP protocol methods"""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"


class MCPRequest(BaseModel):
    """MCP request message"""
    jsonrpc: str = "2.0"
    id: int
    method: str
    params: Dict[str, Any] | None = None


class MCPError(BaseModel):
    """MCP error object"""
    code: int
    message: str
    data: Any | None = None


class MCPResponse(BaseModel):
    """MCP response message"""
    jsonrpc: str = "2.0"
    id: int
    result: Dict[str, Any] | None = None
    error: MCPError | None = None


class ToolInput(BaseModel):
    """JSON Schema for tool input"""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class Tool(BaseModel):
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: ToolInput


class TextContent(BaseModel):
    """Text content in tool response"""
    type: str = "text"
    text: str


class ImageContent(BaseModel):
    """Image content in tool response"""
    type: str = "image"
    data: str
    mimeType: str


class ToolResponse(BaseModel):
    """Tool execution response"""
    content: List[TextContent | ImageContent]
    isError: bool = False


class ServerInfo(BaseModel):
    """Server information"""
    name: str
    version: str


class ServerCapabilities(BaseModel):
    """Server capabilities"""
    tools: Dict[str, Any] | None = Field(default_factory=dict)
    resources: Dict[str, Any] | None = None
    prompts: Dict[str, Any] | None = None


class InitializeResult(BaseModel):
    """Initialize method result"""
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: ServerInfo


class ToolsListResult(BaseModel):
    """tools/list result"""
    tools: List[Tool]


class ToolCallParams(BaseModel):
    """tools/call parameters"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

