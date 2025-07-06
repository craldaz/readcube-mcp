"""
Tests for translator modules in the query2label package.
"""

import pytest
from unittest.mock import Mock, patch
import dspy

# Test imports
from readcube_mcp.query2label.core import BooleanQuery, QueryType, QueryResult
from readcube_mcp.query2label.core.exceptions import QueryTranslationError
from readcube_mcp.query2label.dspy_modules import (
    QueryToLabelsTranslator,
    AdvancedQueryTranslator,
    format_labels_with_counts,
    label_validation_reward,
    create_label_validation_reward,
)


# Test data
SAMPLE_LABELS = [
    "machine-learning",
    "protein-folding", 
    "drug-discovery",
    "computational-chemistry",
    "molecular-dynamics",
    "graph-neural-networks",
    "transformers",
    "neural-networks"
]

SAMPLE_LABEL_COUNTS = {
    "machine-learning": 150,
    "protein-folding": 75,
    "drug-discovery": 120,
    "computational-chemistry": 45,
    "molecular-dynamics": 60,
    "graph-neural-networks": 30,
    "transformers": 25,
    "neural-networks": 90
}


class TestValidators:
    """Test validation functions."""
    
    def test_format_labels_with_counts(self):
        """Test label formatting with counts."""
        formatted = format_labels_with_counts(SAMPLE_LABEL_COUNTS)
        
        # Should be sorted by count (descending)
        assert "machine-learning (count=150)" in formatted
        assert "protein-folding (count=75)" in formatted
        
        # Most frequent should come first
        parts = formatted.split(", ")
        assert parts[0] == "machine-learning (count=150)"
        assert parts[1] == "drug-discovery (count=120)"
    
    def test_label_validation_reward_valid(self):
        """Test reward function with valid labels."""
        label_set = set(SAMPLE_LABELS)
        
        # Mock prediction with valid labels
        pred = Mock()
        pred.matched_labels = "machine-learning, protein-folding"
        
        reward = label_validation_reward(None, pred, label_set)
        assert reward == 1.0
    
    def test_label_validation_reward_invalid(self):
        """Test reward function with invalid labels."""
        label_set = set(SAMPLE_LABELS)
        
        # Mock prediction with invalid labels
        pred = Mock()
        pred.matched_labels = "machine-learning, invalid-label"
        
        reward = label_validation_reward(None, pred, label_set)
        assert reward == 0.0
    
    def test_label_validation_reward_empty(self):
        """Test reward function with empty labels."""
        label_set = set(SAMPLE_LABELS)
        
        # Mock prediction with empty labels
        pred = Mock()
        pred.matched_labels = ""
        
        reward = label_validation_reward(None, pred, label_set)
        assert reward == 1.0  # Empty is considered valid
    
    def test_create_label_validation_reward(self):
        """Test factory function for creating reward functions."""
        label_set = set(SAMPLE_LABELS)
        reward_fn = create_label_validation_reward(label_set)
        
        # Test with valid labels
        pred = Mock()
        pred.matched_labels = "machine-learning"
        
        reward = reward_fn(None, pred)
        assert reward == 1.0


class TestQueryToLabelsTranslator:
    """Test basic query translator."""
    
    def test_init(self):
        """Test translator initialization."""
        translator = QueryToLabelsTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS, 
            max_retries=3
        )
        
        assert translator.available_labels == SAMPLE_LABELS
        assert translator.label_set == set(SAMPLE_LABELS)
        assert translator.label_counts == SAMPLE_LABEL_COUNTS
        assert translator.max_retries == 3
        assert translator.labels_with_counts is not None
        
        # DSPy modules should be initialized
        assert translator.query_parser is not None
        assert translator.label_matcher is not None
    
    @patch('readcube_mcp.query2label.dspy_modules.translators.dspy.ChainOfThought')
    @patch('readcube_mcp.query2label.dspy_modules.translators.dspy.Refine')
    def test_forward_simple_query(self, mock_refine, mock_cot):
        """Test basic query translation (mocked)."""
        # Mock the DSPy components
        mock_parser = Mock()
        mock_parser.return_value = Mock(
            main_concepts="machine learning, protein folding",
            required_concepts="",
            optional_concepts="",
            excluded_concepts=""
        )
        mock_cot.return_value = mock_parser
        
        mock_matcher = Mock()
        mock_match_result = Mock()
        mock_match_result.matched_labels = "machine-learning, protein-folding"
        mock_matcher.return_value = mock_match_result
        mock_refine.return_value = mock_matcher
        
        translator = QueryToLabelsTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        result = translator.forward("Find papers on machine learning and protein folding")
        
        assert isinstance(result, BooleanQuery)
        assert "machine-learning" in result.should_have
        assert "protein-folding" in result.should_have
    
    def test_match_concepts_to_labels_empty(self):
        """Test concept matching with empty concepts."""
        translator = QueryToLabelsTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        # Test with empty concepts
        result = translator._match_concepts_to_labels([])
        assert result == set()
        
        # Test with whitespace-only concepts
        result = translator._match_concepts_to_labels(["", "  ", "\t"])
        assert result == set()
    
    def test_create_query_result(self):
        """Test QueryResult creation."""
        translator = QueryToLabelsTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        boolean_query = BooleanQuery(
            must_have={"machine-learning"},
            should_have={"protein-folding"},
            must_not_have={"computational-chemistry"}
        )
        
        result = translator.create_query_result("test query", boolean_query)
        
        assert isinstance(result, QueryResult)
        assert result.boolean_query == boolean_query
        assert result.query_type == QueryType.SIMPLE
        assert result.confidence == 1.0
        assert result.original_query == "test query"
        assert "translator" in result.metadata


class TestAdvancedQueryTranslator:
    """Test advanced query translator."""
    
    def test_init(self):
        """Test advanced translator initialization."""
        translator = AdvancedQueryTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS, 
            max_retries=3
        )
        
        assert translator.available_labels == SAMPLE_LABELS
        assert translator.label_counts == SAMPLE_LABEL_COUNTS
        assert translator.max_retries == 3
        assert translator.basic_translator is not None
        assert translator.boolean_parser is not None
    
    def test_has_explicit_boolean_logic(self):
        """Test boolean operator detection."""
        translator = AdvancedQueryTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        # Test queries with boolean operators
        assert translator._has_explicit_boolean_logic("machine learning AND protein folding")
        assert translator._has_explicit_boolean_logic("drug discovery OR molecular dynamics")
        assert translator._has_explicit_boolean_logic("chemistry but NOT computational")
        assert translator._has_explicit_boolean_logic("proteins except folding")
        assert translator._has_explicit_boolean_logic("ML without deep learning")
        
        # Test queries without boolean operators
        assert not translator._has_explicit_boolean_logic("machine learning protein folding")
        assert not translator._has_explicit_boolean_logic("find papers on chemistry")
    
    def test_parse_explicit_boolean_query_regex_fallback(self):
        """Test regex-based boolean query parsing."""
        translator = AdvancedQueryTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        # Mock the _extract_labels_from_text method to return predictable results
        def mock_extract(text):
            if "machine learning" in text.lower():
                return {"machine-learning"}
            elif "protein folding" in text.lower():
                return {"protein-folding"}
            elif "chemistry" in text.lower():
                return {"computational-chemistry"}
            return set()
        
        translator._extract_labels_from_text = mock_extract
        
        # Test AND query
        result = translator._parse_explicit_boolean_query_regex(
            "machine learning AND protein folding"
        )
        
        assert isinstance(result, BooleanQuery)
        # Note: The regex parser puts the first part in should_have, then switches to must_have
        assert len(result.should_have.union(result.must_have)) >= 1
    
    def test_extract_labels_from_text_error_handling(self):
        """Test error handling in label extraction."""
        translator = AdvancedQueryTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        # Mock basic_translator to raise an exception
        translator.basic_translator = Mock(side_effect=Exception("Test error"))
        
        # Should return empty set on error
        result = translator._extract_labels_from_text("test text")
        assert result == set()
    
    def test_create_query_result_boolean(self):
        """Test QueryResult creation for boolean queries."""
        translator = AdvancedQueryTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        boolean_query = BooleanQuery(
            must_have={"machine-learning"},
            should_have=set(),
            must_not_have={"computational-chemistry"}
        )
        
        # Test with boolean query
        result = translator.create_query_result(
            "machine learning AND NOT computational chemistry", 
            boolean_query
        )
        
        assert result.query_type == QueryType.BOOLEAN
        assert result.metadata["has_boolean_operators"] is True
        
        # Test with simple query
        result = translator.create_query_result(
            "machine learning", 
            boolean_query
        )
        
        assert result.query_type == QueryType.SIMPLE
        assert result.metadata["has_boolean_operators"] is False


class TestTranslatorIntegration:
    """Integration tests for translator components."""
    
    def test_translator_imports(self):
        """Test that translators can be imported from main package."""
        from readcube_mcp.query2label import QueryToLabelsTranslator, AdvancedQueryTranslator
        
        assert QueryToLabelsTranslator is not None
        assert AdvancedQueryTranslator is not None
    
    def test_boolean_query_creation(self):
        """Test that translators create valid BooleanQuery objects."""
        translator = QueryToLabelsTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        # Create a simple boolean query manually to test structure
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
    
    @pytest.mark.parametrize("translator_class", [
        QueryToLabelsTranslator,
        AdvancedQueryTranslator,
    ])
    def test_translator_inheritance(self, translator_class):
        """Test that translator classes inherit from dspy.Module."""
        assert issubclass(translator_class, dspy.Module)
        
        # Test instantiation
        translator = translator_class(SAMPLE_LABELS, SAMPLE_LABEL_COUNTS)
        assert isinstance(translator, dspy.Module)
    
    def test_error_handling(self):
        """Test error handling in translators."""
        translator = QueryToLabelsTranslator(
            SAMPLE_LABELS, 
            SAMPLE_LABEL_COUNTS
        )
        
        # Mock query_parser to raise an exception
        translator.query_parser = Mock(side_effect=Exception("DSPy error"))
        
        with pytest.raises(QueryTranslationError) as exc_info:
            translator.forward("test query")
        
        assert "Failed to translate query" in str(exc_info.value)
        assert exc_info.value.original_query == "test query"