"""Query2Label: Natural Language to Boolean Query Translation with DSPy Refine.

A sophisticated system that translates natural language queries into structured boolean 
queries over paper labels, enabling semantic search across academic paper databases.
"""

# Core components
from . import core
from . import dspy_modules
from . import data
from . import utils

# Main interfaces (only what's implemented)
from .core import BooleanQuery, QueryType, QueryResult, ParsedQuery, LabelMatch

__version__ = "0.1.0"

__all__ = [
    "BooleanQuery",
    "QueryType", 
    "QueryResult",
    "ParsedQuery",
    "LabelMatch",
    "core",
    "dspy_modules", 
    "data",
    "utils",
]