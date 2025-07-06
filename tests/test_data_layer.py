"""Tests for the data layer components in the query2label package."""

import pytest
import tempfile
import json
import pandas as pd
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Test imports
from readcube_mcp.query2label.core import BooleanQuery
from readcube_mcp.query2label.core.exceptions import DataProcessingError
from readcube_mcp.query2label.data import (
    PaperFilter,
    PaperDataLoader,
    LabelProcessor,
    TextProcessor,
    PaperDataProcessor,
    DataValidator
)


# Test data
SAMPLE_PAPERS = [
    {
        'id': 1,
        'title': 'Machine Learning for Drug Discovery',
        'abstract': 'This paper explores machine learning techniques for drug discovery.',
        'labels': ['machine-learning', 'drug-discovery', 'computational-chemistry'],
        'year': '2023',
        'journal': 'Nature',
        'authors': 'Smith, J.'
    },
    {
        'id': 2,
        'title': 'Protein Folding Prediction',
        'abstract': 'Advanced methods for predicting protein folding patterns.',
        'labels': ['protein-folding', 'computational-biology'],
        'year': '2022',
        'journal': 'Science',
        'authors': 'Johnson, A.'
    },
    {
        'id': 3,
        'title': 'Graph Neural Networks in Chemistry',
        'abstract': 'Application of graph neural networks to chemical problems.',
        'labels': ['graph-neural-networks', 'machine-learning', 'chemistry'],
        'year': '2023',
        'journal': 'Cell',
        'authors': 'Brown, K.'
    }
]


class TestPaperFilter:
    """Test PaperFilter functionality."""
    
    def test_init(self):
        """Test PaperFilter initialization."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        assert paper_filter.papers_db == SAMPLE_PAPERS
        assert paper_filter.get_papers_count() == 3
    
    def test_init_validation_error(self):
        """Test PaperFilter initialization with invalid data."""
        # Test with non-list input
        with pytest.raises(TypeError):
            PaperFilter("not a list")
        
        # Test with invalid paper structure
        invalid_papers = [{'id': 1, 'title': 'Test'}]  # Missing labels
        with pytest.raises(ValueError):
            PaperFilter(invalid_papers)
    
    def test_filter_papers_must_conditions(self):
        """Test filtering with MUST conditions."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        
        # Query that should match paper 1
        query = BooleanQuery(
            must_have={'machine-learning'},
            should_have=set(),
            must_not_have=set()
        )
        
        results = paper_filter.filter_papers(query)
        assert len(results) == 2  # Papers 1 and 3 have machine-learning
        assert results[0]['id'] in [1, 3]
        assert results[1]['id'] in [1, 3]
    
    def test_filter_papers_should_conditions(self):
        """Test filtering with SHOULD conditions."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        
        # Query with multiple SHOULD conditions
        query = BooleanQuery(
            must_have=set(),
            should_have={'protein-folding', 'chemistry'},
            must_not_have=set()
        )
        
        results = paper_filter.filter_papers(query)
        assert len(results) == 2  # Papers 2 and 3 match
        
        # Check relevance scores
        for paper in results:
            assert 'relevance_score' in paper
            assert 0.0 <= paper['relevance_score'] <= 1.0
    
    def test_filter_papers_not_conditions(self):
        """Test filtering with NOT conditions."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        
        # Exclude machine learning papers
        query = BooleanQuery(
            must_have=set(),
            should_have={'computational-biology', 'chemistry'},
            must_not_have={'machine-learning'}
        )
        
        results = paper_filter.filter_papers(query)
        assert len(results) == 1  # Only paper 2 should match
        assert results[0]['id'] == 2
    
    def test_filter_papers_complex_query(self):
        """Test filtering with complex boolean query."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        
        # Must have chemistry-related, should have ML, but not drug discovery
        query = BooleanQuery(
            must_have={'chemistry'},
            should_have={'machine-learning'},
            must_not_have={'drug-discovery'}
        )
        
        results = paper_filter.filter_papers(query)
        assert len(results) == 1  # Only paper 3 matches
        assert results[0]['id'] == 3
    
    def test_filter_papers_min_should_match(self):
        """Test min_should_match parameter."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        
        query = BooleanQuery(
            must_have=set(),
            should_have={'protein-folding', 'drug-discovery', 'nonexistent'},
            must_not_have=set()
        )
        
        # Default min_should_match=1
        results = paper_filter.filter_papers(query, min_should_match=1)
        assert len(results) == 2  # Papers 1 and 2
        
        # Require 2 matches
        results = paper_filter.filter_papers(query, min_should_match=2)
        assert len(results) == 0  # No paper has 2+ matches
    
    def test_get_unique_labels(self):
        """Test getting unique labels."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        unique_labels = paper_filter.get_unique_labels()
        
        expected_labels = {
            'machine-learning', 'drug-discovery', 'computational-chemistry',
            'protein-folding', 'computational-biology', 'graph-neural-networks', 'chemistry'
        }
        assert unique_labels == expected_labels
    
    def test_get_label_counts(self):
        """Test getting label counts."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        label_counts = paper_filter.get_label_counts()
        
        assert label_counts['machine-learning'] == 2
        assert label_counts['protein-folding'] == 1
        assert label_counts['chemistry'] == 1
    
    def test_get_statistics(self):
        """Test getting database statistics."""
        paper_filter = PaperFilter(SAMPLE_PAPERS)
        stats = paper_filter.get_statistics()
        
        assert stats['total_papers'] == 3
        assert stats['unique_labels_count'] == 7
        assert stats['avg_labels_per_paper'] == 8/3  # Total 8 labels across 3 papers
        assert len(stats['most_common_labels']) <= 10


class TestPaperDataLoader:
    """Test PaperDataLoader functionality."""
    
    def test_init(self):
        """Test PaperDataLoader initialization."""
        loader = PaperDataLoader()
        assert loader.label_delimiters == [';', ',', '|']
        assert loader.papers_db == []
        
        # Test custom delimiters
        loader = PaperDataLoader(label_delimiters=[';', '|'])
        assert loader.label_delimiters == [';', '|']
    
    def test_parse_labels(self):
        """Test label parsing with multiple delimiters."""
        loader = PaperDataLoader()
        
        # Test semicolon delimiter
        labels = loader._parse_labels("machine-learning; drug-discovery; chemistry")
        assert labels == ['machine-learning', 'drug-discovery', 'chemistry']
        
        # Test comma delimiter
        labels = loader._parse_labels("ml, ai, deep-learning")
        assert labels == ['ml', 'ai', 'deep-learning']
        
        # Test pipe delimiter
        labels = loader._parse_labels("protein|folding|prediction")
        assert labels == ['protein', 'folding', 'prediction']
        
        # Test single label (no delimiter)
        labels = loader._parse_labels("single-label")
        assert labels == ['single-label']
        
        # Test empty string
        labels = loader._parse_labels("")
        assert labels == []
    
    def test_load_papers_list(self):
        """Test loading papers from a list."""
        loader = PaperDataLoader()
        papers = loader.load_papers_list(SAMPLE_PAPERS)
        
        assert len(papers) == 3
        assert loader.get_papers_db() == papers
        assert len(loader.get_available_labels()) == 7
        
        # Check statistics
        stats = loader.get_statistics()
        assert stats['total_papers'] == 3
        assert stats['unique_labels'] == 7
    
    def test_load_csv(self):
        """Test loading papers from CSV file."""
        loader = PaperDataLoader()
        
        # Create a temporary CSV file
        csv_data = {
            'Tags': ['ml;ai', 'protein;folding', 'chemistry,drugs'],
            'Title': ['ML Paper', 'Protein Paper', 'Chemistry Paper'],
            'Abstract': ['ML abstract', 'Protein abstract', 'Chemistry abstract'],
            'year': ['2023', '2022', '2023']
        }
        df = pd.DataFrame(csv_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            f.flush()  # Ensure data is written
            
            papers = loader.load_csv(f.name)
            
        assert len(papers) == 3
        assert papers[0]['labels'] == ['ml', 'ai']
        assert papers[1]['labels'] == ['protein', 'folding']
        assert papers[2]['labels'] == ['chemistry', 'drugs']
    
    def test_load_json(self):
        """Test loading papers from JSON file."""
        loader = PaperDataLoader()
        
        json_data = {
            "papers": [
                {
                    "title": "Test Paper",
                    "labels": ["ml", "ai"],
                    "abstract": "Test abstract"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            f.flush()  # Ensure data is written
            
            papers = loader.load_json(f.name)
            
        assert len(papers) == 1
        assert papers[0]['labels'] == ['ml', 'ai']
    
    def test_export_to_json(self):
        """Test exporting papers to JSON."""
        loader = PaperDataLoader()
        loader.load_papers_list(SAMPLE_PAPERS)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            loader.export_to_json(f.name)
            
            # Read back the exported data
            with open(f.name, 'r') as read_f:
                exported_data = json.load(read_f)
        
        assert 'papers' in exported_data
        assert 'statistics' in exported_data
        assert 'label_counts' in exported_data
        assert len(exported_data['papers']) == 3
    
    def test_filter_papers_by_label_frequency(self):
        """Test filtering papers by label frequency."""
        loader = PaperDataLoader()
        loader.load_papers_list(SAMPLE_PAPERS)
        
        # Filter out labels that appear only once
        filtered_papers = loader.filter_papers_by_label_frequency(min_frequency=2)
        
        # Only machine-learning appears twice, so only papers with it should remain
        assert len(filtered_papers) == 2
        for paper in filtered_papers:
            assert 'machine-learning' in paper['labels']
    
    def test_get_sample_papers(self):
        """Test getting sample papers."""
        loader = PaperDataLoader()
        loader.load_papers_list(SAMPLE_PAPERS)
        
        sample = loader.get_sample_papers(n=2)
        assert len(sample) == 2
        
        # Test when n > available papers
        sample = loader.get_sample_papers(n=10)
        assert len(sample) == 3


class TestLabelProcessor:
    """Test LabelProcessor functionality."""
    
    def test_init(self):
        """Test LabelProcessor initialization."""
        processor = LabelProcessor()
        assert processor.normalize_case is True
        assert processor.remove_special_chars is True
        assert processor.min_label_length == 2
        assert processor.max_label_length == 100
    
    def test_process_single_label(self):
        """Test single label processing."""
        processor = LabelProcessor()
        
        # Test normal label
        result = processor._process_single_label("Machine Learning")
        assert result == "machine-learning"
        
        # Test label with special characters
        result = processor._process_single_label("AI/ML & Deep Learning!")
        assert result == "aiml-deep-learning"
        
        # Test label too short
        result = processor._process_single_label("A")
        assert result is None
        
        # Test empty label
        result = processor._process_single_label("")
        assert result is None
    
    def test_process_labels(self):
        """Test batch label processing."""
        processor = LabelProcessor()
        
        labels = ["Machine Learning", "AI/ML", "Deep Learning!", "", "X"]
        processed = processor.process_labels(labels)
        
        expected = ["machine-learning", "aiml", "deep-learning"]
        assert processed == expected
    
    def test_deduplicate_labels(self):
        """Test label deduplication."""
        processor = LabelProcessor()
        
        labels = ["ml", "ai", "ml", "deep-learning", "ai"]
        unique_labels = processor.deduplicate_labels(labels)
        
        assert unique_labels == ["ml", "ai", "deep-learning"]
    
    def test_get_label_statistics(self):
        """Test label statistics."""
        processor = LabelProcessor()
        
        all_labels = ["ml", "ai", "ml", "deep-learning", "ai", "ml"]
        stats = processor.get_label_statistics(all_labels)
        
        assert stats['total_labels'] == 6
        assert stats['unique_labels'] == 3
        assert stats['most_common'][0] == ('ml', 3)


class TestTextProcessor:
    """Test TextProcessor functionality."""
    
    def test_init(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor()
        assert processor.max_length is None
        assert processor.clean_html is True
        assert processor.normalize_whitespace is True
    
    def test_process_text(self):
        """Test text processing."""
        processor = TextProcessor()
        
        # Test normal text
        result = processor.process_text("This is a test.")
        assert result == "This is a test."
        
        # Test HTML removal
        result = processor.process_text("This <b>is</b> a <i>test</i>.")
        assert result == "This is a test."
        
        # Test whitespace normalization
        result = processor.process_text("This   is    a\n\ntest.")
        assert result == "This is a test."
        
        # Test empty text
        result = processor.process_text("")
        assert result == ""
    
    def test_process_text_with_max_length(self):
        """Test text processing with length limit."""
        processor = TextProcessor(max_length=10)
        
        result = processor.process_text("This is a very long text that should be truncated.")
        assert len(result) <= 10
        assert result == "This is a"
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        processor = TextProcessor()
        
        text = "This paper discusses machine learning and artificial intelligence methods."
        keywords = processor.extract_keywords(text)
        
        assert "machine" in keywords
        assert "learning" in keywords
        assert "artificial" in keywords
        assert "intelligence" in keywords
        assert "methods" in keywords
        
        # Stop words should be removed
        assert "this" not in keywords
        assert "and" not in keywords


class TestPaperDataProcessor:
    """Test PaperDataProcessor functionality."""
    
    def test_init(self):
        """Test PaperDataProcessor initialization."""
        processor = PaperDataProcessor()
        assert processor.label_processor is not None
        assert processor.text_processor is not None
    
    def test_process_single_paper(self):
        """Test single paper processing."""
        processor = PaperDataProcessor()
        
        paper = {
            'title': '  Machine Learning  ',
            'abstract': 'This <b>is</b> a test.',
            'labels': ['ML & AI', 'Deep Learning!'],
            'id': 1
        }
        
        processed = processor.process_single_paper(paper)
        
        assert processed['title'] == 'Machine Learning'
        assert processed['abstract'] == 'This is a test.'
        assert 'ml-ai' in processed['labels']
        assert 'deep-learning' in processed['labels']
    
    def test_process_paper_batch(self):
        """Test batch paper processing."""
        processor = PaperDataProcessor()
        
        papers = [
            {'title': 'Paper 1', 'labels': ['ML'], 'id': 1},
            {'title': 'Paper 2', 'labels': [], 'id': 2},  # Should be filtered out
            {'title': 'Paper 3', 'labels': ['AI'], 'id': 3}
        ]
        
        processed = processor.process_paper_batch(papers)
        
        # Paper 2 should be filtered out (no labels)
        assert len(processed) == 2
        assert processed[0]['id'] == 1
        assert processed[1]['id'] == 3
    
    def test_validate_paper_structure(self):
        """Test paper structure validation."""
        processor = PaperDataProcessor()
        
        # Valid paper with all required and recommended fields
        valid_paper = {'id': 1, 'title': 'Test', 'abstract': 'Test abstract', 'labels': ['ml']}
        is_valid, errors = processor.validate_paper_structure(valid_paper)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid paper (missing labels)
        invalid_paper = {'id': 1, 'title': 'Test'}
        is_valid, errors = processor.validate_paper_structure(invalid_paper)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_get_processing_statistics(self):
        """Test processing statistics."""
        processor = PaperDataProcessor()
        
        original_papers = [
            {'labels': ['ML', 'AI'], 'id': 1},
            {'labels': [], 'id': 2},
            {'labels': ['DL'], 'id': 3}
        ]
        
        processed_papers = [
            {'labels': ['ml', 'ai'], 'id': 1},
            {'labels': ['dl'], 'id': 3}
        ]
        
        stats = processor.get_processing_statistics(original_papers, processed_papers)
        
        assert stats['original_paper_count'] == 3
        assert stats['processed_paper_count'] == 2
        assert stats['papers_filtered_out'] == 1
        assert stats['retention_rate'] == 2/3


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def test_init(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        assert validator.min_papers == 1
        assert validator.min_labels_per_paper == 1
        assert validator.min_unique_labels == 1
    
    def test_validate_dataset_valid(self):
        """Test dataset validation with valid data."""
        validator = DataValidator()
        
        papers = [
            {'labels': ['ml', 'ai'], 'id': 1},
            {'labels': ['dl'], 'id': 2}
        ]
        
        is_valid, errors = validator.validate_dataset(papers)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_dataset_invalid(self):
        """Test dataset validation with invalid data."""
        validator = DataValidator(min_papers=3, min_unique_labels=5)
        
        papers = [
            {'labels': ['ml'], 'id': 1},
            {'labels': ['ai'], 'id': 2}
        ]
        
        is_valid, errors = validator.validate_dataset(papers)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_get_quality_metrics(self):
        """Test quality metrics calculation."""
        validator = DataValidator()
        
        papers = [
            {'labels': ['ml', 'ai'], 'title': 'Paper 1', 'abstract': 'Abstract 1'},
            {'labels': ['dl'], 'title': 'Paper 2', 'abstract': ''}
        ]
        
        metrics = validator.get_quality_metrics(papers)
        
        assert metrics['total_papers'] == 2
        assert metrics['total_labels'] == 3
        assert metrics['unique_labels'] == 3
        assert metrics['avg_labels_per_paper'] == 1.5
        assert metrics['title_coverage'] == 1.0
        assert metrics['abstract_coverage'] == 0.5


class TestDataLayerIntegration:
    """Integration tests for data layer components."""
    
    def test_full_pipeline(self):
        """Test complete data processing pipeline."""
        # Load data
        loader = PaperDataLoader()
        papers = loader.load_papers_list(SAMPLE_PAPERS)
        
        # Process data
        processor = PaperDataProcessor()
        processed_papers = processor.process_paper_batch(papers)
        
        # Validate data
        validator = DataValidator()
        is_valid, errors = validator.validate_dataset(processed_papers)
        assert is_valid is True
        
        # Filter data
        paper_filter = PaperFilter(processed_papers)
        query = BooleanQuery(
            must_have=set(),
            should_have={'machine-learning'},
            must_not_have=set()
        )
        
        filtered_papers = paper_filter.filter_papers(query)
        assert len(filtered_papers) > 0
    
    def test_error_handling(self):
        """Test error handling in data components."""
        # Test invalid file path
        loader = PaperDataLoader()
        with pytest.raises(DataProcessingError):
            loader.load_csv("nonexistent_file.csv")
        
        # Test invalid paper structure
        with pytest.raises(TypeError):
            PaperFilter([1, 2, 3])  # Not dictionaries
    
    def test_imports(self):
        """Test that all data components can be imported."""
        from readcube_mcp.query2label.data import (
            PaperFilter,
            PaperDataLoader,
            LabelProcessor,
            TextProcessor,
            PaperDataProcessor,
            DataValidator
        )
        
        # Verify all classes are importable
        assert PaperFilter is not None
        assert PaperDataLoader is not None
        assert LabelProcessor is not None
        assert TextProcessor is not None
        assert PaperDataProcessor is not None
        assert DataValidator is not None