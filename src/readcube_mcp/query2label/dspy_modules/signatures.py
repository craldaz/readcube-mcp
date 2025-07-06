"""DSPy signatures for query processing and translation.

This module contains the DSPy signature definitions used in the Query2Label system
for parsing natural language queries and matching concepts to database labels.
"""

import dspy
from typing import List


class QueryParser(dspy.Signature):
    """Parse natural language query into structured components using available labels.
    
    This signature takes a natural language query and breaks it down into different
    types of concepts that can be mapped to database labels. It considers the
    available labels to think broadly about different clusters and categories.
    """

    query = dspy.InputField(
        desc="Natural language query from user"
    )
    available_labels = dspy.InputField(
        desc="List of available labels in the database which will be used to match "
             "concepts in the query to actual labels (comma-separated)"
    )
    
    main_concepts = dspy.OutputField(
        desc="Primary concepts the user is looking for (comma-separated)"
    )
    required_concepts = dspy.OutputField(
        desc="Concepts that MUST be present (comma-separated, empty if none)"
    )
    optional_concepts = dspy.OutputField(
        desc="Concepts that SHOULD be present but not required (comma-separated, empty if none)"
    )
    excluded_concepts = dspy.OutputField(
        desc="Concepts that must NOT be present (comma-separated, empty if none)"
    )


class LabelMatcher(dspy.Signature):
    """Match parsed concepts to actual database labels with usage frequency awareness.
    
    This signature takes individual concepts extracted from natural language and
    matches them to the most appropriate labels in the database. It considers
    label usage frequencies to prefer commonly used labels when multiple matches
    are possible, ensuring alignment with existing labeling patterns.
    """

    concept = dspy.InputField(
        desc="A concept extracted from user query"
    )
    available_labels = dspy.InputField(
        desc="List of available labels with their usage counts in format 'label (count=X)'"
    )

    matched_labels = dspy.OutputField(
        desc="Best matching labels for this concept (comma-separated). MUST be from "
             "available_labels list only. Prefer labels with higher usage counts when "
             "multiple good matches exist."
    )
    confidence = dspy.OutputField(
        desc="Confidence score 0-1 for the matches"
    )


class BooleanQueryParser(dspy.Signature):
    """Parse queries with explicit boolean operators (AND, OR, NOT).
    
    This signature handles more complex queries that contain explicit boolean
    logic operators, extracting the different parts and their logical relationships.
    """
    
    query = dspy.InputField(
        desc="Natural language query with explicit boolean operators (AND, OR, NOT, etc.)"
    )
    available_labels = dspy.InputField(
        desc="List of available labels in the database (comma-separated)"
    )
    
    must_concepts = dspy.OutputField(
        desc="Concepts that must be present (AND conditions) (comma-separated)"
    )
    should_concepts = dspy.OutputField(
        desc="Concepts that should be present (OR conditions) (comma-separated)"
    )
    not_concepts = dspy.OutputField(
        desc="Concepts that must not be present (NOT conditions) (comma-separated)"
    )


class ConceptExtractor(dspy.Signature):
    """Extract meaningful concepts from text fragments.
    
    This signature is used to extract concepts from text parts when parsing
    complex boolean queries, helping to identify the key terms that need to
    be matched to database labels.
    """
    
    text_fragment = dspy.InputField(
        desc="A fragment of text from a larger query"
    )
    context = dspy.InputField(
        desc="Context about what type of concepts to extract (e.g., 'scientific terms', 'methods')"
    )
    
    extracted_concepts = dspy.OutputField(
        desc="Key concepts extracted from the text fragment (comma-separated)"
    )
    relevance = dspy.OutputField(
        desc="Relevance score 0-1 indicating how relevant these concepts are to the context"
    )


class LabelValidator(dspy.Signature):
    """Validate that proposed labels exist in the database.
    
    This signature provides validation feedback for label matching, helping
    to ensure that all matched labels are valid and providing suggestions
    for corrections when needed.
    """
    
    proposed_labels = dspy.InputField(
        desc="List of labels that were proposed as matches (comma-separated)"
    )
    available_labels = dspy.InputField(
        desc="Complete list of valid labels in the database (comma-separated)"
    )
    original_concept = dspy.InputField(
        desc="The original concept that was being matched"
    )
    
    valid_labels = dspy.OutputField(
        desc="Labels from proposed_labels that are actually valid (comma-separated)"
    )
    invalid_labels = dspy.OutputField(
        desc="Labels from proposed_labels that are not valid (comma-separated)"
    )
    suggestions = dspy.OutputField(
        desc="Suggested alternative labels for invalid ones (comma-separated, empty if none)"
    )
    validation_feedback = dspy.OutputField(
        desc="Detailed feedback about the validation results"
    )