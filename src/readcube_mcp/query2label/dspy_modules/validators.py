"""Reward functions and validation logic for DSPy Refine pattern.

This module contains validation functions used with DSPy Refine to ensure
that label matching produces valid results and provides feedback for 
automatic retries when validation fails.
"""

import dspy
from typing import Set, List, Dict, Tuple
from collections import Counter


def label_validation_reward(args, pred: dspy.Prediction, label_set: Set[str]) -> float:
    """Reward function for DSPy Refine that validates label matching results.
    
    This function is used with DSPy Refine to automatically validate that
    all matched labels exist in the available label set. It returns 1.0
    for perfect validation (all labels valid) and 0.0 for any invalid labels.
    
    Args:
        args: Arguments passed to the DSPy module (unused but required by DSPy)
        pred: DSPy prediction containing matched_labels field
        label_set: Set of valid labels to check against
        
    Returns:
        float: 1.0 if all labels are valid, 0.0 otherwise
    """
    try:
        matched_labels = []
        if hasattr(pred, 'matched_labels') and pred.matched_labels.strip():
            matched_labels = [
                label.strip() 
                for label in pred.matched_labels.split(',') 
                if label.strip()
            ]
        
        # Check if all labels are in available_labels
        valid_labels = all(label in label_set for label in matched_labels)
        return 1.0 if valid_labels else 0.0
        
    except Exception:
        # Any exception during validation counts as failure
        return 0.0


def create_label_validation_reward(label_set: Set[str]):
    """Create a label validation reward function with a specific label set.
    
    This factory function creates a validation reward function that has
    the label set bound to it, making it easier to use with DSPy Refine.
    
    Args:
        label_set: Set of valid labels to validate against
        
    Returns:
        Callable: Reward function suitable for DSPy Refine
    """
    def reward_fn(args, pred: dspy.Prediction) -> float:
        return label_validation_reward(args, pred, label_set)
    
    return reward_fn


def format_labels_with_counts(label_counts: Dict[str, int]) -> str:
    """Format labels with their usage counts for LLM context.
    
    This function creates a formatted string of labels with their usage
    frequencies, sorted by frequency (descending) to show the most
    commonly used labels first. This helps the LLM make better matching
    decisions based on actual usage patterns.
    
    Args:
        label_counts: Dictionary mapping label names to usage counts
        
    Returns:
        str: Formatted string like "label1 (count=50), label2 (count=25), ..."
    """
    # Sort labels by count (descending) to show most popular first
    sorted_labels = sorted(
        label_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Format as "label (count=X)"
    formatted_labels = [
        f"{label} (count={count})" 
        for label, count in sorted_labels
    ]
    
    return ", ".join(formatted_labels)


def validate_labels_against_database(
    proposed_labels: List[str], 
    available_labels: Set[str]
) -> Tuple[List[str], List[str]]:
    """Validate a list of proposed labels against the available database labels.
    
    Args:
        proposed_labels: List of labels to validate
        available_labels: Set of valid labels in the database
        
    Returns:
        Tuple of (valid_labels, invalid_labels)
    """
    valid_labels = []
    invalid_labels = []
    
    for label in proposed_labels:
        label = label.strip()
        if label in available_labels:
            valid_labels.append(label)
        else:
            invalid_labels.append(label)
    
    return valid_labels, invalid_labels


def generate_validation_feedback(
    invalid_labels: List[str],
    available_labels: Set[str],
    original_concept: str = ""
) -> str:
    """Generate helpful feedback for failed label validation.
    
    This function creates detailed feedback that can be used by DSPy Refine
    to provide better guidance for the next attempt when label matching fails.
    
    Args:
        invalid_labels: List of labels that failed validation
        available_labels: Set of valid labels in the database
        original_concept: The original concept that was being matched
        
    Returns:
        str: Detailed feedback message for improvement
    """
    if not invalid_labels:
        return "All labels are valid."
    
    feedback_parts = []
    
    feedback_parts.append(
        f"Invalid labels found: {', '.join(invalid_labels)}"
    )
    
    # Try to suggest similar labels
    suggestions = []
    for invalid_label in invalid_labels:
        # Simple similarity check - look for labels containing parts of the invalid label
        lower_invalid = invalid_label.lower()
        similar_labels = [
            label for label in available_labels 
            if any(
                part in label.lower() 
                for part in lower_invalid.split('-')
                if len(part) > 2
            )
        ]
        if similar_labels:
            suggestions.extend(similar_labels[:3])  # Limit to 3 suggestions per invalid label
    
    if suggestions:
        unique_suggestions = list(set(suggestions))[:5]  # Max 5 total suggestions
        feedback_parts.append(
            f"Consider these similar available labels: {', '.join(unique_suggestions)}"
        )
    
    if original_concept:
        feedback_parts.append(
            f"Remember to match the concept '{original_concept}' to labels that actually exist in the database."
        )
    
    feedback_parts.append(
        "Please select only from the available labels list provided."
    )
    
    return " ".join(feedback_parts)


def calculate_label_frequencies(papers: List[Dict]) -> Dict[str, int]:
    """Calculate label usage frequencies from a list of papers.
    
    Args:
        papers: List of paper dictionaries, each containing a 'labels' key
        
    Returns:
        Dict mapping label names to their usage frequencies
    """
    all_labels = []
    
    for paper in papers:
        labels = paper.get('labels', [])
        if isinstance(labels, list):
            all_labels.extend(labels)
        elif isinstance(labels, str):
            # Handle case where labels might be a string
            all_labels.append(labels)
    
    return dict(Counter(all_labels))


class LabelValidationError(Exception):
    """Exception raised when label validation fails completely."""
    
    def __init__(self, message: str, invalid_labels: List[str] = None):
        super().__init__(message)
        self.invalid_labels = invalid_labels or []