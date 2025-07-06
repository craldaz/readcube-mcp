"""Core data structures and types for Query2Label system."""

from .types import BooleanQuery, QueryType, QueryResult
from .exceptions import (
    QueryTranslationError,
    LabelValidationError,
    DataLoadingError,
    ConfigurationError,
)

__all__ = [
    "BooleanQuery",
    "QueryType", 
    "QueryResult",
    "QueryTranslationError",
    "LabelValidationError",
    "DataLoadingError",
    "ConfigurationError",
]