"""Integration tests for end-to-end Query2Label functionality.

This module tests the complete pipeline from loading data to processing
queries and filtering results, using the real Acelot Library dataset.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import all the components we need
from readcube_mcp.query2label import (
    BooleanQuery, 
    QueryType,
    QueryResult,
    PaperFilter,
    PaperDataLoader
)
from readcube_mcp.query2label.dspy_modules import (
    QueryToLabelsTranslator,
    AdvancedQueryTranslator
)
from readcube_mcp.query2label.utils import get_config


class TestDataLoading:
    """Test data loading from CSV files."""
    
    def test_can_load_sample_data(self):
        """Test that we can load sample data for testing."""
        # Create sample data if Acelot Library.csv doesn't exist
        loader = PaperDataLoader()
        
        sample_papers = [
            {
                'id': 1,
                'title': 'Machine Learning for Drug Discovery',
                'abstract': 'This paper explores the use of machine learning techniques in drug discovery processes.',
                'labels': ['machine-learning', 'drug-discovery', 'computational-chemistry'],
                'year': '2023'
            },
            {
                'id': 2,
                'title': 'Protein Folding Prediction with Neural Networks',
                'abstract': 'We present a novel approach to protein folding prediction using deep neural networks.',
                'labels': ['protein-folding', 'neural-networks', 'computational-biology'],
                'year': '2022'
            },
            {
                'id': 3,
                'title': 'Graph Neural Networks in Chemistry',
                'abstract': 'Application of graph neural networks for molecular property prediction.',
                'labels': ['graph-neural-networks', 'chemistry', 'machine-learning'],
                'year': '2023'
            },
            {
                'id': 4,
                'title': 'Molecular Dynamics Simulations',
                'abstract': 'Computational methods for molecular dynamics simulations.',
                'labels': ['molecular-dynamics', 'computational-chemistry', 'simulation'],
                'year': '2021'
            },
            {
                'id': 5,
                'title': 'Transformers for Chemical Property Prediction',
                'abstract': 'Using transformer models for predicting chemical properties.',
                'labels': ['transformers', 'chemistry', 'deep-learning'],
                'year': '2023'
            }
        ]
        
        papers = loader.load_papers_list(sample_papers)
        
        assert len(papers) == 5
        assert loader.get_statistics()['total_papers'] == 5
        assert loader.get_statistics()['unique_labels'] > 0
        
        return papers, loader.get_available_labels(), loader.get_label_counts()


class TestQueryTranslation:
    """Test query translation functionality."""
    
    def setup_method(self):
        """Set up test data for query translation tests."""
        self.test_data = TestDataLoading()
        self.papers, self.available_labels, self.label_counts = self.test_data.test_can_load_sample_data()
        
        # Mock the DSPy components since we don't want to make actual API calls in tests
        self.mock_dspy_calls = True
    
    def _create_mock_translator(self):
        """Create a mock translator for testing without API calls."""
        translator = QueryToLabelsTranslator(self.available_labels, self.label_counts)
        
        # Mock the DSPy modules to return predictable results
        def mock_query_parser(query, available_labels):
            # Simple rule-based parsing for testing
            query_lower = query.lower()
            
            mock_result = Mock()
            
            if 'drug discovery' in query_lower or 'alzheimer' in query_lower:
                mock_result.main_concepts = 'drug-discovery'
                mock_result.required_concepts = ''
                mock_result.optional_concepts = ''
                mock_result.excluded_concepts = ''
            elif 'protein folding' in query_lower and 'machine learning' in query_lower:
                mock_result.main_concepts = 'protein-folding, machine-learning'
                mock_result.required_concepts = ''
                mock_result.optional_concepts = ''
                mock_result.excluded_concepts = ''
            elif 'graph neural networks' in query_lower:
                mock_result.main_concepts = 'graph-neural-networks'
                mock_result.required_concepts = ''
                mock_result.optional_concepts = 'drug-discovery'
                mock_result.excluded_concepts = ''
            elif 'machine learning' in query_lower and 'not' in query_lower:
                mock_result.main_concepts = 'machine-learning'
                mock_result.required_concepts = ''
                mock_result.optional_concepts = ''
                mock_result.excluded_concepts = 'computational-chemistry'
            elif 'deep learning' in query_lower and 'molecular dynamics' in query_lower:
                mock_result.main_concepts = 'deep-learning, molecular-dynamics'
                mock_result.required_concepts = ''
                mock_result.optional_concepts = ''
                mock_result.excluded_concepts = ''
            elif 'transformers' in query_lower or 'neural networks' in query_lower:
                mock_result.main_concepts = 'transformers, neural-networks'
                mock_result.required_concepts = ''
                mock_result.optional_concepts = 'chemistry'
                mock_result.excluded_concepts = ''
            else:
                mock_result.main_concepts = ''
                mock_result.required_concepts = ''
                mock_result.optional_concepts = ''
                mock_result.excluded_concepts = ''
            
            return mock_result
        
        def mock_label_matcher(concept, available_labels):
            # Simple concept to label matching for testing
            concept_lower = concept.lower()
            mock_result = Mock()
            
            # Map concepts to actual labels
            concept_mapping = {
                'drug-discovery': 'drug-discovery',
                'protein-folding': 'protein-folding', 
                'machine-learning': 'machine-learning',
                'graph-neural-networks': 'graph-neural-networks',
                'computational-chemistry': 'computational-chemistry',
                'deep-learning': 'deep-learning',
                'molecular-dynamics': 'molecular-dynamics',
                'transformers': 'transformers',
                'neural-networks': 'neural-networks',
                'chemistry': 'chemistry'
            }
            
            matched_labels = []
            for key, label in concept_mapping.items():
                if key in concept_lower and label in self.available_labels:
                    matched_labels.append(label)
            
            mock_result.matched_labels = ', '.join(matched_labels)
            return mock_result
        
        translator.query_parser = mock_query_parser
        translator.label_matcher = mock_label_matcher
        
        return translator
    
    def test_query_translation_simple(self):
        """Test simple query translation."""
        translator = self._create_mock_translator()
        
        query = "Find papers on protein folding using machine learning"
        result = translator(query)
        
        assert isinstance(result, BooleanQuery)
        assert 'protein-folding' in result.should_have or 'protein-folding' in result.must_have
        assert 'machine-learning' in result.should_have or 'machine-learning' in result.must_have
    
    def test_query_translation_with_exclusion(self):
        """Test query translation with exclusion."""
        translator = self._create_mock_translator()
        
        query = "Machine learning but not computational chemistry"
        result = translator(query)
        
        assert isinstance(result, BooleanQuery)
        assert 'machine-learning' in result.should_have or 'machine-learning' in result.must_have
        assert 'computational-chemistry' in result.must_not_have
    
    def test_advanced_query_translator(self):
        """Test advanced query translator with boolean logic."""
        translator = AdvancedQueryTranslator(self.available_labels, self.label_counts)
        
        # Mock the basic translator
        mock_basic = self._create_mock_translator()
        translator.basic_translator = mock_basic
        
        query = "Deep learning AND molecular dynamics"
        result = translator(query)
        
        assert isinstance(result, BooleanQuery)
        # Should detect boolean logic and process accordingly
    
    @pytest.mark.parametrize("query,expected_labels", [
        ("Find papers on the discovery of new alzheimer's drugs", ["drug-discovery"]),
        ("Find papers on protein folding using machine learning", ["protein-folding", "machine-learning"]),
        ("Drug discovery with graph neural networks", ["drug-discovery", "graph-neural-networks"]),
        ("Machine learning but not computational chemistry", ["machine-learning"]),
        ("Deep learning AND molecular dynamics", ["deep-learning", "molecular-dynamics"]),
        ("Transformers OR neural networks for chemistry", ["transformers", "neural-networks", "chemistry"])
    ])
    def test_demo_queries(self, query, expected_labels):
        """Test the actual queries from demo.ipynb."""
        translator = self._create_mock_translator()
        
        result = translator(query)
        
        assert isinstance(result, BooleanQuery)
        
        # Check that at least some expected labels are present
        all_labels = result.must_have.union(result.should_have)
        
        # At least one expected label should be found
        found_labels = [label for label in expected_labels if label in all_labels]
        assert len(found_labels) > 0, f"No expected labels found for query '{query}'. Got: {all_labels}"


class TestPaperFiltering:
    """Test paper filtering functionality."""
    
    def setup_method(self):
        """Set up test data for filtering tests."""
        self.test_data = TestDataLoading()
        self.papers, self.available_labels, self.label_counts = self.test_data.test_can_load_sample_data()
        self.paper_filter = PaperFilter(self.papers)
    
    def test_filter_with_must_conditions(self):
        """Test filtering with MUST conditions."""
        query = BooleanQuery(
            must_have={'machine-learning'},
            should_have=set(),
            must_not_have=set()
        )
        
        results = self.paper_filter.filter_papers(query)
        
        # Should find papers with machine-learning
        assert len(results) > 0
        for paper in results:
            assert 'machine-learning' in paper['labels']
    
    def test_filter_with_should_conditions(self):
        """Test filtering with SHOULD conditions."""
        query = BooleanQuery(
            must_have=set(),
            should_have={'protein-folding', 'drug-discovery'},
            must_not_have=set()
        )
        
        results = self.paper_filter.filter_papers(query)
        
        # Should find papers with either protein-folding or drug-discovery
        assert len(results) > 0
        for paper in results:
            has_target_label = any(label in paper['labels'] for label in ['protein-folding', 'drug-discovery'])
            assert has_target_label
    
    def test_filter_with_not_conditions(self):
        """Test filtering with NOT conditions."""
        query = BooleanQuery(
            must_have=set(),
            should_have={'machine-learning'},
            must_not_have={'computational-chemistry'}
        )
        
        results = self.paper_filter.filter_papers(query)
        
        # Should find machine-learning papers but exclude computational-chemistry
        for paper in results:
            assert 'machine-learning' in paper['labels']
            assert 'computational-chemistry' not in paper['labels']
    
    def test_relevance_scoring(self):
        """Test relevance scoring in filtered results."""
        query = BooleanQuery(
            must_have=set(),
            should_have={'machine-learning', 'chemistry', 'drug-discovery'},
            must_not_have=set()
        )
        
        results = self.paper_filter.filter_papers(query)
        
        # Results should be sorted by relevance score
        assert len(results) > 0
        
        # Check that results have relevance scores
        for paper in results:
            assert 'relevance_score' in paper
            assert 0.0 <= paper['relevance_score'] <= 1.0
        
        # Check that scores are in descending order
        scores = [paper['relevance_score'] for paper in results]
        assert scores == sorted(scores, reverse=True)


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""
    
    def setup_method(self):
        """Set up complete system for integration tests."""
        # Load test data
        self.test_data = TestDataLoading()
        self.papers, self.available_labels, self.label_counts = self.test_data.test_can_load_sample_data()
        
        # Set up components
        self.paper_filter = PaperFilter(self.papers)
        
        # Get configuration
        self.config = get_config()
    
    def _create_mock_query_system(self):
        """Create a complete mock query system for testing."""
        translator = QueryToLabelsTranslator(self.available_labels, self.label_counts)
        
        # Mock query processing to return reasonable results
        def mock_forward(query):
            query_lower = query.lower()
            
            must_have = set()
            should_have = set()
            must_not_have = set()
            
            # Simple rule-based query processing for testing
            if 'drug discovery' in query_lower or 'alzheimer' in query_lower:
                should_have.add('drug-discovery')
            
            if 'protein folding' in query_lower:
                should_have.add('protein-folding')
            
            if 'machine learning' in query_lower:
                should_have.add('machine-learning')
            
            if 'graph neural networks' in query_lower:
                should_have.add('graph-neural-networks')
            
            if 'computational chemistry' in query_lower and ('not' in query_lower or 'but' in query_lower):
                must_not_have.add('computational-chemistry')
            
            if 'deep learning' in query_lower:
                should_have.add('deep-learning')
            
            if 'molecular dynamics' in query_lower:
                should_have.add('molecular-dynamics')
            
            if 'transformers' in query_lower:
                should_have.add('transformers')
            
            if 'neural networks' in query_lower:
                should_have.add('neural-networks')
            
            if 'chemistry' in query_lower and 'not' not in query_lower:
                should_have.add('chemistry')
            
            # Handle AND logic
            if ' and ' in query_lower.lower():
                # Move should_have to must_have for AND queries
                must_have.update(should_have)
                should_have.clear()
            
            return BooleanQuery(must_have, should_have, must_not_have)
        
        translator.forward = mock_forward
        return translator
    
    def test_complete_query_pipeline(self):
        """Test the complete pipeline from query to results."""
        translator = self._create_mock_query_system()
        
        test_queries = [
            "Find papers on protein folding using machine learning",
            "Drug discovery with graph neural networks", 
            "Machine learning but not computational chemistry"
        ]
        
        for query in test_queries:
            # Step 1: Translate query
            boolean_query = translator(query)
            assert isinstance(boolean_query, BooleanQuery)
            
            # Step 2: Filter papers
            results = self.paper_filter.filter_papers(boolean_query)
            print(f"Found {len(results)} papers")

            # Step 3: Verify results are reasonable
            assert isinstance(results, list)
            # Results might be empty for some queries with our limited test data
            
            # If we have results, verify they have the expected structure
            for paper in results:
                assert 'id' in paper
                assert 'title' in paper
                assert 'labels' in paper
                assert 'library_url' in paper
                assert 'relevance_score' in paper

            # Show top 5 results
            for i, paper in enumerate(results[:5]):
                print(f"  {i+1}. {paper['title'][:80]}..." if len(paper['title']) > 80 else f"  {i+1}. {paper['title']}")
                print(f"     Score: {paper.get('relevance_score', 0):.2f} | Labels: {paper['labels'][:3]}")
                if paper.get('library_url'):
                    print(f"     URL: {paper['library_url']}")
                print()

            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more papers")

            
    
    @pytest.mark.parametrize("query", [
        "Find papers on the discovery of new alzheimer's drugs",
        "Find papers on protein folding using machine learning", 
        "Drug discovery with graph neural networks",
        "Machine learning but not computational chemistry",
        "Deep learning AND molecular dynamics",
        "Transformers OR neural networks for chemistry"
    ])
    def test_demo_queries_end_to_end(self, query):
        """Test all demo queries end-to-end."""
        translator = self._create_mock_query_system()
        
        # Process query
        boolean_query = translator(query)
        results = self.paper_filter.filter_papers(boolean_query)
        
        # Basic validation
        assert isinstance(boolean_query, BooleanQuery)
        assert isinstance(results, list)
        
        # Log query results for debugging
        print(f"\nQuery: {query}")
        print(f"Boolean Query: MUST={boolean_query.must_have}, SHOULD={boolean_query.should_have}, NOT={boolean_query.must_not_have}")
        print(f"Results: {len(results)} papers found")
        
        if results:
            print(f"Top result: {results[0]['title']} (score: {results[0]['relevance_score']:.2f})")
    
    def test_system_statistics(self):
        """Test that we can get system statistics."""
        stats = self.paper_filter.get_statistics()
        
        assert 'total_papers' in stats
        assert 'unique_labels_count' in stats
        assert 'avg_labels_per_paper' in stats
        assert stats['total_papers'] > 0
        assert stats['unique_labels_count'] > 0
    
    def test_configuration_integration(self):
        """Test that configuration is properly integrated."""
        config = get_config()
        
        # Verify config structure
        assert hasattr(config, 'dspy')
        assert hasattr(config, 'data')
        assert hasattr(config, 'logging')
        
        # Verify we can use config values
        assert config.data.label_delimiters == [';', ',', '|']
        assert config.dspy.max_retries >= 0
        
        print(f"\nSystem Configuration:")
        print(f"DSPy Model: {config.dspy.model}")
        print(f"Max Retries: {config.dspy.max_retries}")
        print(f"Label Delimiters: {config.data.label_delimiters}")
        print(f"Cache Enabled: {config.data.cache_enabled}")


class TestRealDataIntegration:
    """Test integration with real Acelot Library data if available."""
    
    def test_load_real_acelot_data(self):
        """Test loading real Acelot Library CSV if it exists."""
        # Look for the CSV file in common locations
        possible_paths = [
            Path.cwd() / "Acelot Library.csv",
            Path.cwd().parent / "Acelot Library.csv",
            Path.cwd() / "data" / "Acelot Library.csv"
        ]
        
        csv_file = None
        for path in possible_paths:
            if path.exists():
                csv_file = path
                break
        
        if csv_file is None:
            pytest.skip("Acelot Library.csv not found - skipping real data test")
        
        # Load the real data
        loader = PaperDataLoader()
        papers = loader.load_csv(csv_file)
        
        assert len(papers) > 0
        
        # Get statistics
        stats = loader.get_statistics()
        print(f"\nReal Acelot Library Statistics:")
        print(f"Total papers: {stats['total_papers']}")
        print(f"Unique labels: {stats['unique_labels']}")
        print(f"Avg labels per paper: {stats['avg_labels_per_paper']:.2f}")
        
        # Show top labels
        if 'top_labels' in stats:
            print("Top 10 labels:")
            for label, count in stats['top_labels'][:10]:
                print(f"  {label}: {count}")
        
        # Test that we can create a filter with the real data
        paper_filter = PaperFilter(papers)
        filter_stats = paper_filter.get_statistics()
        
        assert filter_stats['total_papers'] == len(papers)
        assert filter_stats['unique_labels_count'] > 0
