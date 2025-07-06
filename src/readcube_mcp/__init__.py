"""Readcube-MCP: MCP Server with Query2Label Integration.

Model Context Protocol server that provides natural language query capabilities
for scientific paper databases, powered by the Query2Label algorithm.
"""

# Import both sub-packages
from . import query2label
from . import mcp_server

# Main interfaces from query2label
from .query2label import BooleanQuery, QueryType, QueryResult
from .query2label import AdvancedQueryTranslator, PaperDataLoader, PaperFilter

# Main interfaces from mcp_server
from .mcp_server import ReadcubeMCPServer

# Legacy imports for backward compatibility
from .readcube_mcp import *

from ._version import __version__

__all__ = [
    # Query2Label exports
    "BooleanQuery",
    "QueryType", 
    "QueryResult",
    "AdvancedQueryTranslator",
    "PaperDataLoader",
    "PaperFilter",
    "query2label",
    # MCP Server exports
    "ReadcubeMCPServer",
    "mcp_server",
]
