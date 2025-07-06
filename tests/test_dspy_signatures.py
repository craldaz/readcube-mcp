"""
Tests for DSPy signatures in the query2label package.
"""

import pytest
import dspy

# Test imports
from readcube_mcp.query2label.dspy_modules import (
    QueryParser,
    LabelMatcher,
    BooleanQueryParser,
    ConceptExtractor,
    LabelValidator,
)


def test_signatures_imported():
    """Test that all DSPy signatures can be imported."""
    assert QueryParser is not None
    assert LabelMatcher is not None
    assert BooleanQueryParser is not None
    assert ConceptExtractor is not None
    assert LabelValidator is not None


def test_signature_inheritance():
    """Test that all signature classes properly inherit from dspy.Signature."""
    signatures = [
        QueryParser,
        LabelMatcher,
        BooleanQueryParser,
        ConceptExtractor,
        LabelValidator,
    ]
    
    for signature_class in signatures:
        assert issubclass(signature_class, dspy.Signature)


def test_signature_docstrings():
    """Test that all signatures have proper docstrings."""
    signatures = [
        QueryParser,
        LabelMatcher,
        BooleanQueryParser,
        ConceptExtractor,
        LabelValidator,
    ]
    
    for signature in signatures:
        assert signature.__doc__ is not None
        assert len(signature.__doc__.strip()) > 0
        # Docstring should describe what the signature does
        assert any(word in signature.__doc__.lower() for word in ['parse', 'match', 'extract', 'validate'])


def test_query_parser_fields():
    """Test QueryParser has the expected field structure."""
    # Test that QueryParser has the expected string representation
    signature_str = str(QueryParser)
    
    # Should contain input field names
    assert 'query' in signature_str
    assert 'available_labels' in signature_str
    
    # Should contain output field names
    assert 'main_concepts' in signature_str
    assert 'required_concepts' in signature_str
    assert 'optional_concepts' in signature_str
    assert 'excluded_concepts' in signature_str


def test_label_matcher_fields():
    """Test LabelMatcher has the expected field structure."""
    signature_str = str(LabelMatcher)
    
    # Should contain input field names
    assert 'concept' in signature_str
    assert 'available_labels' in signature_str
    
    # Should contain output field names
    assert 'matched_labels' in signature_str
    assert 'confidence' in signature_str


def test_boolean_query_parser_fields():
    """Test BooleanQueryParser has the expected field structure."""
    signature_str = str(BooleanQueryParser)
    
    # Should contain input field names
    assert 'query' in signature_str
    assert 'available_labels' in signature_str
    
    # Should contain output field names
    assert 'must_concepts' in signature_str
    assert 'should_concepts' in signature_str
    assert 'not_concepts' in signature_str


def test_concept_extractor_fields():
    """Test ConceptExtractor has the expected field structure."""
    signature_str = str(ConceptExtractor)
    
    # Should contain input field names
    assert 'text_fragment' in signature_str
    assert 'context' in signature_str
    
    # Should contain output field names
    assert 'extracted_concepts' in signature_str
    assert 'relevance' in signature_str


def test_label_validator_fields():
    """Test LabelValidator has the expected field structure."""
    signature_str = str(LabelValidator)
    
    # Should contain input field names
    assert 'proposed_labels' in signature_str
    assert 'available_labels' in signature_str
    assert 'original_concept' in signature_str
    
    # Should contain output field names
    assert 'valid_labels' in signature_str
    assert 'invalid_labels' in signature_str
    assert 'suggestions' in signature_str
    assert 'validation_feedback' in signature_str


def test_signature_can_be_used_with_chain_of_thought():
    """Test that signatures can be wrapped with ChainOfThought."""
    # This tests that the signatures are properly structured for DSPy
    try:
        query_parser = dspy.ChainOfThought(QueryParser)
        label_matcher = dspy.ChainOfThought(LabelMatcher)
        boolean_parser = dspy.ChainOfThought(BooleanQueryParser)
        concept_extractor = dspy.ChainOfThought(ConceptExtractor)
        label_validator = dspy.ChainOfThought(LabelValidator)
        
        # If we can create these without errors, the signatures are well-formed
        assert query_parser is not None
        assert label_matcher is not None
        assert boolean_parser is not None
        assert concept_extractor is not None
        assert label_validator is not None
        
    except Exception as e:
        pytest.fail(f"Signatures should be compatible with ChainOfThought: {e}")


def test_signature_field_descriptions():
    """Test that signature fields have proper descriptions in their string representation."""
    # Check that the signatures contain field descriptions
    signature_str = str(QueryParser)
    assert 'Natural language query from user' in signature_str
    
    signature_str = str(LabelMatcher)
    assert 'concept extracted from user query' in signature_str
    
    # This ensures our field descriptions are properly set


@pytest.mark.parametrize("signature_class,expected_name", [
    (QueryParser, "QueryParser"),
    (LabelMatcher, "LabelMatcher"),
    (BooleanQueryParser, "BooleanQueryParser"),
    (ConceptExtractor, "ConceptExtractor"),
    (LabelValidator, "LabelValidator"),
])
def test_signature_names(signature_class, expected_name):
    """Test that signature classes have the expected names."""
    assert signature_class.__name__ == expected_name