# src/core/exceptions.py
"""
Custom exceptions for the application
"""


class MCPServerException(Exception):
    """Base exception for MCP server errors"""
    def __init__(self, message: str, code: int = -32603):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ToolNotFoundException(MCPServerException):
    """Tool not found exception"""
    def __init__(self, tool_name: str):
        super().__init__(
            message=f"Tool not found: {tool_name}",
            code=-32601
        )


class ToolExecutionException(MCPServerException):
    """Tool execution failed exception"""
    def __init__(self, tool_name: str, error: str):
        super().__init__(
            message=f"Tool execution failed for '{tool_name}': {error}",
            code=-32603
        )


class InvalidParametersException(MCPServerException):
    """Invalid parameters exception"""
    def __init__(self, message: str):
        super().__init__(
            message=f"Invalid parameters: {message}",
            code=-32602
        )


class VectorDatabaseException(MCPServerException):
    """Vector database operation failed"""
    def __init__(self, operation: str, error: str):
        super().__init__(
            message=f"Vector database {operation} failed: {error}",
            code=-32603
        )


class LLMServiceException(MCPServerException):
    """LLM service operation failed"""
    def __init__(self, operation: str, error: str):
        super().__init__(
            message=f"LLM service {operation} failed: {error}",
            code=-32603
        )


class FileUploadException(MCPServerException):
    """File upload failed"""
    def __init__(self, error: str):
        super().__init__(
            message=f"File upload failed: {error}",
            code=-32603
        )