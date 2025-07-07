"""Core data structures and types for Query2Label system."""

from .types import (
    BooleanQuery,
    QueryType,
    QueryResult,
    ParsedQuery,
    LabelMatch,
    FusionResult,
    FusionConfig,
)
from .exceptions import (
    Query2LabelError,
    QueryTranslationError,
    LabelValidationError,
    DataLoadingError,
    ConfigurationError,
    DSPyModuleError,
    PaperFilterError,
)

__all__ = [
    # Core types
    "BooleanQuery",
    "QueryType", 
    "QueryResult",
    "ParsedQuery",
    "LabelMatch",
    "FusionResult",
    "FusionConfig",
    # Exceptions
    "Query2LabelError",
    "QueryTranslationError",
    "LabelValidationError",
    "DataLoadingError",
    "ConfigurationError",
    "DSPyModuleError",
    "PaperFilterError",
]