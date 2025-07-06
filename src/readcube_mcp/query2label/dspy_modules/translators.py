"""Main translation modules using DSPy for Query2Label system.

This module contains the core translation logic that converts natural language
queries into structured boolean queries over paper labels. It implements the
sophisticated DSPy Refine pattern with automatic feedback loops and label
count priors for robust, validated label matching.
"""

import dspy
import re
from typing import List, Set, Dict, Optional
from ..core.types import BooleanQuery, QueryType, QueryResult, ParsedQuery, LabelMatch
from ..core.exceptions import QueryTranslationError, DSPyModuleError
from .signatures import QueryParser, LabelMatcher, BooleanQueryParser
from .validators import (
    create_label_validation_reward, 
    format_labels_with_counts,
    generate_validation_feedback,
    validate_labels_against_database
)


class QueryToLabelsTranslator(dspy.Module):
    """Main translator from natural language to boolean label queries with DSPy Refine validation.
    
    This is the core Query2Label algorithm that implements sophisticated label matching
    with automatic feedback loops, label count priors, and validation to ensure all
    returned labels are valid. Uses DSPy Refine pattern for robust error handling.
    """

    def __init__(
        self, 
        available_labels: List[str], 
        label_counts: Dict[str, int], 
        max_retries: int = 3
    ):
        """Initialize the translator with label database and configuration.
        
        Args:
            available_labels: List of all valid labels in the database
            label_counts: Dictionary mapping labels to their usage frequencies
            max_retries: Maximum number of retries for DSPy Refine validation
        """
        super().__init__()
        self.available_labels = available_labels
        self.label_set = set(available_labels)
        self.label_counts = label_counts
        self.max_retries = max_retries

        # Create formatted label list with counts for LLM context
        self.labels_with_counts = format_labels_with_counts(label_counts)

        # DSPy modules
        self.query_parser = dspy.ChainOfThought(QueryParser)
        
        # Create reward function for label validation with bound label set
        validation_reward = create_label_validation_reward(self.label_set)
        
        # Wrap LabelMatcher with Refine for automatic feedback-based retries
        self.label_matcher = dspy.Refine(
            dspy.ChainOfThought(LabelMatcher),
            N=max_retries,
            reward_fn=validation_reward,
            threshold=1.0  # Stop when we get perfect validation (reward = 1.0)
        )

    def forward(self, query: str) -> BooleanQuery:
        """Translate natural language query to boolean label query.
        
        Args:
            query: Natural language query from user
            
        Returns:
            BooleanQuery: Structured boolean query with validated labels
            
        Raises:
            QueryTranslationError: If query processing fails
        """
        try:
            # Step 1: Parse the natural language query
            parsed = self.query_parser(
                query=query,
                available_labels=self.labels_with_counts
            )

            # Step 2: Match concepts to actual labels
            boolean_query = BooleanQuery(
                must_have=set(),
                should_have=set(),
                must_not_have=set()
            )

            # Process main concepts (these become SHOULD conditions)
            if parsed.main_concepts.strip():
                main_labels = self._match_concepts_to_labels(
                    parsed.main_concepts.split(',')
                )
                boolean_query.should_have.update(main_labels)

            # Process required concepts (these become MUST conditions)
            if parsed.required_concepts.strip():
                required_labels = self._match_concepts_to_labels(
                    parsed.required_concepts.split(',')
                )
                boolean_query.must_have.update(required_labels)

            # Process optional concepts (these become additional SHOULD conditions)
            if parsed.optional_concepts.strip():
                optional_labels = self._match_concepts_to_labels(
                    parsed.optional_concepts.split(',')
                )
                boolean_query.should_have.update(optional_labels)

            # Process excluded concepts (these become NOT conditions)
            if parsed.excluded_concepts.strip():
                excluded_labels = self._match_concepts_to_labels(
                    parsed.excluded_concepts.split(',')
                )
                boolean_query.must_not_have.update(excluded_labels)

            return boolean_query

        except Exception as e:
            raise QueryTranslationError(
                f"Failed to translate query: {str(e)}", 
                query, 
                {"error_type": type(e).__name__}
            )

    def _match_concepts_to_labels(self, concepts: List[str]) -> Set[str]:
        """Match a list of concepts to actual database labels using Refine validation.
        
        This is the core matching logic that uses DSPy Refine to ensure all
        returned labels are valid. It automatically provides feedback and retries
        when validation fails.
        
        Args:
            concepts: List of concepts extracted from the query
            
        Returns:
            Set of validated labels from the database
        """
        matched_labels = set()

        for concept in concepts:
            concept = concept.strip()
            if not concept:
                continue

            try:
                # Use Refine wrapped LabelMatcher - it will automatically provide feedback and retry
                matches = self.label_matcher(
                    concept=concept,
                    available_labels=self.labels_with_counts
                )

                # Parse matched labels - Refine ensures these are valid through feedback loop
                if matches.matched_labels.strip():
                    for label in matches.matched_labels.split(','):
                        label = label.strip()
                        if label and label in self.label_set:  # Double-check validity
                            matched_labels.add(label)

            except Exception as e:
                # Handle any errors (network, JSON parsing, etc.)
                print(f"Warning: Could not match concept '{concept}' to valid labels: {e}")
                continue

        return matched_labels

    def create_query_result(self, query: str, boolean_query: BooleanQuery) -> QueryResult:
        """Create a complete QueryResult with metadata.
        
        Args:
            query: Original natural language query
            boolean_query: Translated boolean query
            
        Returns:
            QueryResult with complete metadata
        """
        return QueryResult(
            boolean_query=boolean_query,
            query_type=QueryType.SIMPLE,
            confidence=1.0,  # High confidence since we use validation
            original_query=query,
            metadata={
                "label_count": boolean_query.total_label_count(),
                "has_positive_conditions": boolean_query.has_positive_conditions(),
                "translator": "QueryToLabelsTranslator"
            }
        )


class AdvancedQueryTranslator(dspy.Module):
    """Enhanced translator that handles complex boolean logic with Refine validation.
    
    This translator can handle both simple natural language queries and complex
    queries with explicit boolean operators (AND, OR, NOT). It uses the basic
    translator for simple cases and implements specialized logic for boolean queries.
    """

    def __init__(
        self, 
        available_labels: List[str], 
        label_counts: Dict[str, int], 
        max_retries: int = 3
    ):
        """Initialize the advanced translator.
        
        Args:
            available_labels: List of all valid labels in the database
            label_counts: Dictionary mapping labels to their usage frequencies
            max_retries: Maximum number of retries for DSPy Refine validation
        """
        super().__init__()
        self.basic_translator = QueryToLabelsTranslator(
            available_labels, label_counts, max_retries
        )
        self.available_labels = available_labels
        self.label_counts = label_counts
        self.max_retries = max_retries
        self.label_set = set(available_labels)

        # Additional module for boolean query parsing
        self.boolean_parser = dspy.ChainOfThought(BooleanQueryParser)

    def forward(self, query: str, force_basic: bool = False) -> BooleanQuery:
        """Handle complex queries with explicit boolean logic.
        
        Args:
            query: Natural language query, potentially with boolean operators
            force_basic: Force use of basic translator even for boolean queries
            
        Returns:
            BooleanQuery: Structured boolean query with validated labels
        """
        try:
            # Check for explicit boolean operators
            if not force_basic and self._has_explicit_boolean_logic(query):
                return self._parse_explicit_boolean_query(query)
            else:
                return self.basic_translator(query)
                
        except Exception as e:
            raise QueryTranslationError(
                f"Advanced query translation failed: {str(e)}", 
                query,
                {"translator": "AdvancedQueryTranslator"}
            )

    def _has_explicit_boolean_logic(self, query: str) -> bool:
        """Check if query contains explicit AND, OR, NOT operators.
        
        Args:
            query: Query string to check
            
        Returns:
            bool: True if explicit boolean operators are detected
        """
        boolean_keywords = ['AND', 'OR', 'NOT', 'but not', 'except', 'without']
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in boolean_keywords)

    def _parse_explicit_boolean_query(self, query: str) -> BooleanQuery:
        """Parse queries with explicit boolean logic.
        
        This method handles complex queries like:
        - "Machine learning AND protein folding"
        - "Drug discovery OR molecular dynamics but NOT computational chemistry"
        
        Args:
            query: Query with explicit boolean operators
            
        Returns:
            BooleanQuery: Parsed boolean query with validated labels
        """
        try:
            # Use DSPy to parse the boolean structure
            parsed = self.boolean_parser(
                query=query,
                available_labels=", ".join(self.available_labels)
            )

            boolean_query = BooleanQuery(set(), set(), set())

            # Extract and validate labels for each condition type
            if parsed.must_concepts.strip():
                must_labels = self._extract_labels_from_text(parsed.must_concepts)
                boolean_query.must_have.update(must_labels)

            if parsed.should_concepts.strip():
                should_labels = self._extract_labels_from_text(parsed.should_concepts)
                boolean_query.should_have.update(should_labels)

            if parsed.not_concepts.strip():
                not_labels = self._extract_labels_from_text(parsed.not_concepts)
                boolean_query.must_not_have.update(not_labels)

            return boolean_query

        except Exception as e:
            # Fallback to regex-based parsing if DSPy parsing fails
            return self._parse_explicit_boolean_query_regex(query)

    def _parse_explicit_boolean_query_regex(self, query: str) -> BooleanQuery:
        """Fallback regex-based parser for explicit boolean queries.
        
        This is a simpler parser that uses regex to split queries on boolean
        operators when the DSPy-based parser fails.
        
        Args:
            query: Query with boolean operators
            
        Returns:
            BooleanQuery: Parsed boolean query
        """
        query_lower = query.lower()

        # Split on boolean operators
        parts = re.split(
            r'\b(and|or|not|but not|except|without)\b', 
            query_lower
        )

        boolean_query = BooleanQuery(set(), set(), set())
        current_mode = 'should'  # Default mode

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part in ['and']:
                current_mode = 'must'
            elif part in ['or']:
                current_mode = 'should'
            elif part in ['not', 'but not', 'except', 'without']:
                current_mode = 'not'
            else:
                # Extract labels from this part
                part_labels = self._extract_labels_from_text(part)

                if current_mode == 'must':
                    boolean_query.must_have.update(part_labels)
                elif current_mode == 'should':
                    boolean_query.should_have.update(part_labels)
                elif current_mode == 'not':
                    boolean_query.must_not_have.update(part_labels)

        return boolean_query

    def _extract_labels_from_text(self, text: str) -> Set[str]:
        """Extract labels from a piece of text using the basic translator.
        
        Args:
            text: Text fragment to extract labels from
            
        Returns:
            Set of validated labels
        """
        try:
            # Use the basic translator on just this piece
            sub_query = self.basic_translator(text)
            return sub_query.should_have.union(sub_query.must_have)
        except Exception:
            # If extraction fails, return empty set
            return set()

    def create_query_result(self, query: str, boolean_query: BooleanQuery) -> QueryResult:
        """Create a complete QueryResult with metadata for advanced queries.
        
        Args:
            query: Original natural language query
            boolean_query: Translated boolean query
            
        Returns:
            QueryResult with complete metadata
        """
        query_type = (
            QueryType.BOOLEAN if self._has_explicit_boolean_logic(query) 
            else QueryType.SIMPLE
        )
        
        return QueryResult(
            boolean_query=boolean_query,
            query_type=query_type,
            confidence=1.0,
            original_query=query,
            metadata={
                "label_count": boolean_query.total_label_count(),
                "has_positive_conditions": boolean_query.has_positive_conditions(),
                "translator": "AdvancedQueryTranslator",
                "has_boolean_operators": self._has_explicit_boolean_logic(query)
            }
        )