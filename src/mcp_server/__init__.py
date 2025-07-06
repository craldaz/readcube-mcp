"""MCP Server for Readcube integration using Query2Label.

Model Context Protocol server that provides natural language query capabilities
for scientific paper databases, powered by the Query2Label algorithm.
"""

from .server import ReadcubeMCPServer
from .handlers import QueryHandler, PaperHandler

__version__ = "0.1.0"

__all__ = [
    "ReadcubeMCPServer",
    "QueryHandler", 
    "PaperHandler",
]