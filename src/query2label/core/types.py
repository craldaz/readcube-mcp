"""Core data structures and types for Query2Label system."""

from dataclasses import dataclass
from typing import Set, Optional, Dict, Any
from enum import Enum


class QueryType(Enum):
    """Types of queries the system can handle."""
    SIMPLE = "simple"
    BOOLEAN = "boolean"
    COMPLEX = "complex"


@dataclass
class BooleanQuery:
    """Represents a boolean query over labels.
    
    This is the core data structure that represents how papers should be filtered
    based on label conditions. It supports three types of conditions:
    - MUST (AND): Papers must have all these labels
    - SHOULD (OR): Papers should have at least one of these labels  
    - NOT: Papers must not have any of these labels
    """
    must_have: Set[str]       # AND conditions - papers MUST have all these labels
    should_have: Set[str]     # OR conditions - papers SHOULD have at least one
    must_not_have: Set[str]   # NOT conditions - papers must NOT have any of these
    confidence: float = 1.0   # Confidence score for the translation

    def __str__(self) -> str:
        """Human-readable string representation of the boolean query."""
        parts = []
        if self.must_have:
            parts.append(f"MUST: {list(self.must_have)}")
        if self.should_have:
            parts.append(f"SHOULD: {list(self.should_have)}")
        if self.must_not_have:
            parts.append(f"NOT: {list(self.must_not_have)}")
        return " | ".join(parts) if parts else "Empty Query"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"BooleanQuery("
            f"must_have={self.must_have}, "
            f"should_have={self.should_have}, "
            f"must_not_have={self.must_not_have}, "
            f"confidence={self.confidence})"
        )

    def is_empty(self) -> bool:
        """Check if the query has any conditions."""
        return not (self.must_have or self.should_have or self.must_not_have)

    def has_positive_conditions(self) -> bool:
        """Check if the query has any positive conditions (MUST or SHOULD)."""
        return bool(self.must_have or self.should_have)

    def total_label_count(self) -> int:
        """Total number of unique labels referenced in this query."""
        all_labels = self.must_have | self.should_have | self.must_not_have
        return len(all_labels)


@dataclass
class QueryResult:
    """Result of query translation with metadata.
    
    Contains the translated boolean query along with metadata about
    the translation process and confidence metrics.
    """
    boolean_query: BooleanQuery
    query_type: QueryType
    confidence: float
    original_query: str
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"QueryResult(type={self.query_type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"query={self.boolean_query})"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"QueryResult("
            f"boolean_query={self.boolean_query!r}, "
            f"query_type={self.query_type}, "
            f"confidence={self.confidence}, "
            f"original_query={self.original_query!r}, "
            f"metadata={self.metadata})"
        )


@dataclass
class ParsedQuery:
    """Intermediate representation of a parsed natural language query.
    
    This represents the concepts extracted from natural language before
    they are matched to actual database labels.
    """
    main_concepts: list[str]
    required_concepts: list[str]
    optional_concepts: list[str] 
    excluded_concepts: list[str]
    original_query: str

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = []
        if self.main_concepts:
            parts.append(f"Main: {self.main_concepts}")
        if self.required_concepts:
            parts.append(f"Required: {self.required_concepts}")
        if self.optional_concepts:
            parts.append(f"Optional: {self.optional_concepts}")
        if self.excluded_concepts:
            parts.append(f"Excluded: {self.excluded_concepts}")
        return " | ".join(parts) if parts else "Empty ParsedQuery"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"ParsedQuery("
            f"main_concepts={self.main_concepts}, "
            f"required_concepts={self.required_concepts}, "
            f"optional_concepts={self.optional_concepts}, "
            f"excluded_concepts={self.excluded_concepts}, "
            f"original_query={self.original_query!r})"
        )

    def all_concepts(self) -> list[str]:
        """Get all concepts from all categories."""
        return (
            self.main_concepts + 
            self.required_concepts + 
            self.optional_concepts + 
            self.excluded_concepts
        )

    def has_concepts(self) -> bool:
        """Check if any concepts were extracted."""
        return bool(self.all_concepts())


@dataclass
class LabelMatch:
    """Result of matching a concept to database labels.
    
    Contains the matched labels along with confidence and reasoning
    information from the matching process.
    """
    concept: str
    matched_labels: list[str]
    confidence: float
    reasoning: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"'{self.concept}' → {self.matched_labels} (confidence: {self.confidence:.2f})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"LabelMatch("
            f"concept={self.concept!r}, "
            f"matched_labels={self.matched_labels}, "
            f"confidence={self.confidence}, "
            f"reasoning={self.reasoning!r})"
        )
    
    def is_valid(self) -> bool:
        """Check if this match has valid labels."""
        return bool(self.matched_labels) and self.confidence > 0