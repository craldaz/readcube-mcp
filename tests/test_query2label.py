"""
Unit and regression tests for the query2label package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest

# Test imports
import query2label
from query2label.core import BooleanQuery, QueryType, QueryResult, ParsedQuery, LabelMatch
from query2label.core.exceptions import (
    Query2LabelError,
    QueryTranslationError,
    LabelValidationError,
    DataLoadingError,
    ConfigurationError,
)


def test_query2label_imported():
    """Test that query2label package can be imported."""
    assert "query2label" in sys.modules


def test_core_types_imported():
    """Test that core types can be imported."""
    # Main types
    assert BooleanQuery is not None
    assert QueryType is not None
    assert QueryResult is not None
    assert ParsedQuery is not None
    assert LabelMatch is not None
    
    # Exceptions
    assert Query2LabelError is not None
    assert QueryTranslationError is not None
    assert LabelValidationError is not None
    assert DataLoadingError is not None
    assert ConfigurationError is not None


def test_boolean_query_creation():
    """Test basic BooleanQuery creation and methods."""
    # Test empty query
    empty_query = BooleanQuery(set(), set(), set())
    assert empty_query.is_empty()
    assert not empty_query.has_positive_conditions()
    assert empty_query.total_label_count() == 0
    
    # Test query with conditions
    query = BooleanQuery(
        must_have={"machine-learning"},
        should_have={"protein-folding", "drug-discovery"},
        must_not_have={"computational-chemistry"}
    )
    assert not query.is_empty()
    assert query.has_positive_conditions()
    assert query.total_label_count() == 4
    
    # Test string representation
    query_str = str(query)
    assert "MUST:" in query_str
    assert "SHOULD:" in query_str
    assert "NOT:" in query_str
    assert "machine-learning" in query_str


def test_query_type_enum():
    """Test QueryType enum values."""
    assert QueryType.SIMPLE.value == "simple"
    assert QueryType.BOOLEAN.value == "boolean"
    assert QueryType.COMPLEX.value == "complex"


def test_query_result_creation():
    """Test QueryResult creation."""
    boolean_query = BooleanQuery(
        must_have={"test-label"},
        should_have=set(),
        must_not_have=set()
    )
    
    result = QueryResult(
        boolean_query=boolean_query,
        query_type=QueryType.SIMPLE,
        confidence=0.95,
        original_query="test query",
        metadata={"test": "data"}
    )
    
    assert result.boolean_query == boolean_query
    assert result.query_type == QueryType.SIMPLE
    assert result.confidence == 0.95
    assert result.original_query == "test query"
    assert result.metadata == {"test": "data"}
    
    # Test string representation
    result_str = str(result)
    assert "type=simple" in result_str
    assert "confidence=0.95" in result_str


def test_parsed_query_creation():
    """Test ParsedQuery creation and methods."""
    parsed = ParsedQuery(
        main_concepts=["machine learning", "protein"],
        required_concepts=["drug discovery"],
        optional_concepts=["chemistry"],
        excluded_concepts=["toxicity"],
        original_query="test query"
    )
    
    assert parsed.has_concepts()
    
    all_concepts = parsed.all_concepts()
    assert len(all_concepts) == 5
    assert "machine learning" in all_concepts
    assert "drug discovery" in all_concepts
    
    # Test string representation
    parsed_str = str(parsed)
    assert "Main:" in parsed_str
    assert "Required:" in parsed_str


def test_label_match_creation():
    """Test LabelMatch creation and validation."""
    match = LabelMatch(
        concept="machine learning",
        matched_labels=["machine-learning", "ml"],
        confidence=0.9,
        reasoning="Direct semantic match"
    )
    
    assert match.is_valid()
    assert match.concept == "machine learning"
    assert len(match.matched_labels) == 2
    assert match.confidence == 0.9
    
    # Test invalid match
    invalid_match = LabelMatch(
        concept="test",
        matched_labels=[],
        confidence=0.0
    )
    assert not invalid_match.is_valid()


def test_exceptions_can_be_raised():
    """Test that custom exceptions can be instantiated and raised."""
    # Test base exception
    with pytest.raises(Query2LabelError):
        raise Query2LabelError("Test error")
    
    # Test specific exceptions
    with pytest.raises(QueryTranslationError):
        raise QueryTranslationError("Translation failed", "test query")
    
    with pytest.raises(LabelValidationError):
        raise LabelValidationError("Invalid labels", ["bad-label"], ["good-label"])
    
    with pytest.raises(DataLoadingError):
        raise DataLoadingError("Data loading failed", "/path/to/file")
    
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Config error", "test_key", str)


def test_exception_attributes():
    """Test that exceptions store additional attributes correctly."""
    # Test QueryTranslationError
    query_error = QueryTranslationError("Failed", "test query", {"detail": "info"})
    assert query_error.original_query == "test query"
    assert query_error.details == {"detail": "info"}
    
    # Test LabelValidationError
    label_error = LabelValidationError("Invalid", ["bad"], ["good"])
    assert label_error.invalid_labels == ["bad"]
    assert label_error.available_labels == ["good"]
    
    # Test DataLoadingError
    data_error = DataLoadingError("Failed", "/path", {"size": 0})
    assert data_error.file_path == "/path"
    assert data_error.details == {"size": 0}
    
    # Test ConfigurationError
    config_error = ConfigurationError("Missing", "api_key", str)
    assert config_error.config_key == "api_key"
    assert config_error.expected_type == str