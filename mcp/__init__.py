"""MCP (Model Context Protocol) package for agent coordination."""
from .coordinator import MCPServer

try:
    from .langgraph_coordinator import LangGraphCoordinator
    __all__ = ["MCPServer", "LangGraphCoordinator"]
except ImportError:
    __all__ = ["MCPServer"]

