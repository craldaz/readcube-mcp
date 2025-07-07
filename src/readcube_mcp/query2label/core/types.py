"""Core data structures and types for Query2Label system."""

from dataclasses import dataclass
from typing import Set, Optional, Dict, Any, List
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


@dataclass
class FusionResult:
    """Results from RAG fusion processing.
    
    Contains the final fused papers along with metadata about the fusion process,
    including sub-queries used, individual results, and fusion scores.
    """
    papers: List[Dict[str, Any]]           # Final fused and ranked papers
    sub_queries: List[str]                 # Sub-queries that were processed
    boolean_queries: List[BooleanQuery]    # Translated boolean queries for each sub-query
    fusion_scores: Dict[str, float]        # Paper ID → fusion score mapping
    individual_results: Dict[str, List[Dict[str, Any]]]  # Sub-query → papers mapping
    strategy_used: str                     # "simple" or "fusion"
    total_papers_before_fusion: int        # Sum of all individual results
    
    def improvement_ratio(self) -> float:
        """Calculate improvement over best single sub-query.
        
        Returns the ratio of fused results to the best individual sub-query.
        A ratio > 1.0 indicates fusion found more papers than any single query.
        """
        if not self.individual_results:
            return 1.0
        max_single = max(len(papers) for papers in self.individual_results.values())
        return len(self.papers) / max(max_single, 1)
    
    def coverage_per_query(self) -> Dict[str, int]:
        """Get number of papers found by each sub-query."""
        return {query: len(papers) for query, papers in self.individual_results.items()}
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"FusionResult(strategy={self.strategy_used}, "
            f"papers={len(self.papers)}, "
            f"sub_queries={len(self.sub_queries)}, "
            f"improvement={self.improvement_ratio():.1f}x)"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"FusionResult("
            f"papers={len(self.papers)} items, "
            f"sub_queries={self.sub_queries}, "
            f"strategy_used={self.strategy_used!r}, "
            f"total_papers_before_fusion={self.total_papers_before_fusion}, "
            f"improvement_ratio={self.improvement_ratio():.2f})"
        )


@dataclass
class FusionConfig:
    """Configuration for fusion processing.
    
    Controls various aspects of the fusion algorithm including the RRF parameter,
    result limits, and sub-query constraints.
    """
    k: int = 60                           # RRF constant (higher = less weight on rank differences)
    max_results: int = 50                 # Maximum papers to return after fusion
    min_sub_queries: int = 2              # Minimum sub-queries needed to trigger fusion
    max_sub_queries: int = 5              # Maximum sub-queries to process
    complexity_threshold: float = 0.7     # Threshold for automatic fusion (0-1 scale)
    enable_deduplication: bool = True     # Whether to deduplicate similar sub-queries
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"FusionConfig(k={self.k}, "
            f"max_results={self.max_results}, "
            f"sub_queries={self.min_sub_queries}-{self.max_sub_queries})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"FusionConfig("
            f"k={self.k}, "
            f"max_results={self.max_results}, "
            f"min_sub_queries={self.min_sub_queries}, "
            f"max_sub_queries={self.max_sub_queries}, "
            f"complexity_threshold={self.complexity_threshold}, "
            f"enable_deduplication={self.enable_deduplication})"
        )