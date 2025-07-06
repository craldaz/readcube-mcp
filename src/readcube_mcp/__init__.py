"""Readcube-MCP: Model Context Protocol Server for Scientific Paper Search.

An MCP server that provides natural language query capabilities for scientific 
paper databases, powered by the Query2Label algorithm with DSPy Refine.
"""

from ._version import __version__

# Import Query2Label algorithm as subpackage
from . import query2label

# Import main Query2Label types for convenience
from .query2label import BooleanQuery, QueryType, QueryResult, ParsedQuery, LabelMatch

# MCP Server components (when implemented)
try:
    from .server import ReadcubeMCPServer
    from .handlers import QueryHandler, PaperHandler
except ImportError:
    # MCP server components not implemented yet
    pass

__all__ = [
    "__version__",
    "query2label",
    "BooleanQuery",
    "QueryType",
    "QueryResult", 
    "ParsedQuery",
    "LabelMatch",
]