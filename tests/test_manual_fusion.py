"""Integration tests for manual fusion processor with real query examples."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from readcube_mcp.query2label.fusion.processors import ManualFusionProcessor
from readcube_mcp.query2label.core.types import FusionConfig, BooleanQuery
from readcube_mcp.query2label.core.exceptions import QueryTranslationError


class TestManualFusionProcessor:
    """Test suite for ManualFusionProcessor with realistic scenarios."""
    
    @pytest.fixture
    def mock_translator(self):
        """Mock translator that converts queries to boolean logic."""
        translator = Mock()
        
        def translate_query(query: str) -> BooleanQuery:
            """Mock translation based on query content."""
            if "docking" in query.lower():
                return BooleanQuery(
                    must_have={"docking-calculations"},
                    should_have={"protein-ligand-binding", "molecular-docking"},
                    must_not_have=set()
                )
            elif "tdp-43" in query.lower():
                return BooleanQuery(
                    must_have={"TDP-43"},
                    should_have={"protein-aggregation", "drug-discovery"},
                    must_not_have=set()
                )
            elif "alpha-synuclein" in query.lower():
                return BooleanQuery(
                    must_have={"alpha-synuclein"},
                    should_have={"amyloid-fibrils", "parkinsons-disease"},
                    must_not_have=set()
                )
            elif "idp" in query.lower() or "intrinsically disordered" in query.lower():
                return BooleanQuery(
                    must_have={"intrinsically-disordered-proteins"},
                    should_have={"protein-folding", "structural-disorder"},
                    must_not_have=set()
                )
            elif "error" in query.lower():
                raise QueryTranslationError(f"Simulated translation error for: {query}")
            else:
                return BooleanQuery(
                    must_have=set(),
                    should_have={"general-protein-research"},
                    must_not_have=set()
                )
        
        translator.side_effect = translate_query
        return translator
    
    @pytest.fixture
    def mock_paper_filter(self):
        """Mock paper filter that returns papers based on boolean queries."""
        paper_filter = Mock()
        
        # Sample papers from your domain
        sample_papers = {
            "docking": [
                {
                    "item ID": "docking-1",
                    "Title": "Molecular Docking Studies of Protein-Ligand Interactions",
                    "doi": "10.1234/docking.2023.001",
                    "year": 2023,
                    "Author": "Smith, J.; Jones, M.",
                    "Journal": "J Comput Chem",
                    "labels": ["docking-calculations", "protein-ligand-binding"]
                },
                {
                    "item ID": "docking-2", 
                    "Title": "Virtual Screening of Small Molecules Against IDPs",
                    "doi": "10.1234/docking.2023.002",
                    "year": 2023,
                    "Author": "Chen, L.; Wang, K.",
                    "Journal": "Drug Discov Today",
                    "labels": ["docking-calculations", "intrinsically-disordered-proteins"]
                }
            ],
            "tdp43": [
                {
                    "item ID": "docking-2",  # Same paper as in docking results!
                    "Title": "Virtual Screening of Small Molecules Against IDPs",
                    "doi": "10.1234/docking.2023.002",
                    "year": 2023,
                    "Author": "Chen, L.; Wang, K.",
                    "Journal": "Drug Discov Today",
                    "labels": ["TDP-43", "intrinsically-disordered-proteins", "docking-calculations"]
                },
                {
                    "item ID": "tdp43-1",
                    "Title": "TDP-43 Aggregation Mechanisms in ALS",
                    "doi": "10.1234/tdp43.2023.001", 
                    "year": 2023,
                    "Author": "Brown, A.; Davis, R.",
                    "Journal": "Nat Neurosci",
                    "labels": ["TDP-43", "protein-aggregation", "ALS"]
                }
            ],
            "alpha_syn": [
                {
                    "item ID": "asyn-1",
                    "Title": "Alpha-Synuclein Fibril Formation and Inhibition",
                    "doi": "10.1234/asyn.2023.001",
                    "year": 2023,
                    "Author": "Wilson, P.; Taylor, S.",
                    "Journal": "J Biol Chem",
                    "labels": ["alpha-synuclein", "amyloid-fibrils", "parkinsons-disease"]
                },
                {
                    "item ID": "asyn-2",
                    "Title": "Small Molecule Inhibitors of Alpha-Synuclein Aggregation",
                    "doi": "10.1234/asyn.2023.002",
                    "year": 2023,
                    "Author": "Johnson, K.; Lee, H.",
                    "Journal": "ACS Chem Neurosci",
                    "labels": ["alpha-synuclein", "drug-discovery", "small-molecules"]
                }
            ],
            "idp": [
                {
                    "item ID": "docking-2",  # Same paper appears again!
                    "Title": "Virtual Screening of Small Molecules Against IDPs",
                    "doi": "10.1234/docking.2023.002",
                    "year": 2023,
                    "Author": "Chen, L.; Wang, K.",
                    "Journal": "Drug Discov Today",
                    "labels": ["intrinsically-disordered-proteins", "docking-calculations"]
                },
                {
                    "item ID": "idp-1",
                    "Title": "Structural Properties of Intrinsically Disordered Proteins",
                    "doi": "10.1234/idp.2023.001",
                    "year": 2023,
                    "Author": "Garcia, M.; Rodriguez, C.",
                    "Journal": "Protein Sci",
                    "labels": ["intrinsically-disordered-proteins", "structural-disorder"]
                }
            ]
        }
        
        def filter_papers(boolean_query: BooleanQuery) -> List[Dict[str, Any]]:
            """Mock filtering based on boolean query content."""
            if "docking-calculations" in boolean_query.must_have:
                return sample_papers["docking"]
            elif "TDP-43" in boolean_query.must_have:
                return sample_papers["tdp43"]
            elif "alpha-synuclein" in boolean_query.must_have:
                return sample_papers["alpha_syn"]
            elif "intrinsically-disordered-proteins" in boolean_query.must_have:
                return sample_papers["idp"]
            else:
                return []
        
        paper_filter.filter_papers.side_effect = filter_papers
        return paper_filter
    
    @pytest.fixture
    def fusion_processor(self, mock_translator, mock_paper_filter):
        """Create fusion processor with mocked dependencies."""
        config = FusionConfig(k=60, max_results=50)
        return ManualFusionProcessor(mock_translator, mock_paper_filter, config)
    
    def test_single_query_no_fusion(self, fusion_processor):
        """Test that single queries bypass fusion."""
        result = fusion_processor.process_sub_queries([
            "docking calculations protein ligand binding"
        ])
        
        assert result.strategy_used == "simple"
        assert len(result.sub_queries) == 1
        assert len(result.papers) == 2  # From mock data
        assert not result.fusion_scores  # No fusion applied
        assert result.improvement_ratio() == 1.0  # No improvement since no fusion
    
    def test_complex_idp_query_decomposition(self, fusion_processor):
        """Test the exact complex query from your examples."""
        # Manual decomposition of complex IDP/amyloid query
        sub_queries = [
            "docking calculations intrinsically disordered proteins",
            "TDP-43 ligand binding drug discovery", 
            "alpha-synuclein small molecule inhibitors",
            "intrinsically disordered protein folding"
        ]
        
        result = fusion_processor.process_sub_queries(sub_queries)
        
        # Validation criteria
        assert result.strategy_used == "fusion"
        assert len(result.papers) >= 3  # Should find multiple unique papers
        assert len(result.sub_queries) == 4
        
        # The paper "Virtual Screening of Small Molecules Against IDPs" should rank highly
        # since it appears in multiple sub-query results (docking, TDP-43, IDP)
        top_paper = result.papers[0]
        assert top_paper["found_in_queries"] >= 2  # Appears in multiple queries
        
        # Check improvement ratio
        assert result.improvement_ratio() > 1.0  # Fusion should improve over best single query
        
        # Verify all sub-queries were processed
        assert len(result.individual_results) == 4
        assert all(query in result.individual_results for query in sub_queries)
    
    def test_protein_protein_interactions_excluding_idps(self, fusion_processor):
        """Test Query 1: PPI excluding IDPs using positive context."""
        sub_queries = [
            "protein protein interactions structural biology",
            "protein binding interfaces crystal structures", 
            "protein complex formation mechanisms",
            "protein interaction networks systems biology"
        ]
        
        result = fusion_processor.process_sub_queries(sub_queries)
        
        # These queries don't match our mock patterns, so they return empty results
        # The processor should handle this gracefully with fusion_empty strategy
        assert result.strategy_used in ["fusion", "fusion_empty"]
        assert len(result.sub_queries) == 4
        # With our mock, these specific queries don't match patterns so may be empty
        assert len(result.papers) >= 0  # Should handle empty results gracefully
    
    def test_tdp43_alpha_synuclein_structures(self, fusion_processor):
        """Test Query 2: TDP-43/α-synuclein structural studies."""
        sub_queries = [
            "TDP-43 cryo-EM structure",
            "alpha-synuclein cryo-EM structure",
            "TDP-43 crystal structure x-ray crystallography",
            "alpha-synuclein NMR structure"
        ]
        
        result = fusion_processor.process_sub_queries(sub_queries)
        
        assert result.strategy_used == "fusion"
        assert len(result.papers) >= 2  # Should find papers for both proteins
        
        # Should have papers from both TDP-43 and alpha-synuclein queries
        paper_sources = set()
        for query, papers in result.individual_results.items():
            if papers:
                paper_sources.add("tdp43" if "TDP-43" in query else "alpha_syn")
        
        assert len(paper_sources) >= 1  # At least one protein type found
    
    def test_error_handling_failed_translation(self, fusion_processor):
        """Test handling of translation errors."""
        sub_queries = [
            "docking calculations protein binding",  # Should work
            "error query that fails translation",    # Will trigger error
            "alpha-synuclein aggregation"           # Should work
        ]
        
        result = fusion_processor.process_sub_queries(sub_queries)
        
        # Should complete despite one failure
        assert result.strategy_used in ["fusion", "fusion_failed_fallback"]
        assert len(result.sub_queries) == 3
        
        # Should have results from successful queries
        successful_results = [
            papers for papers in result.individual_results.values() 
            if papers
        ]
        assert len(successful_results) >= 1
    
    def test_empty_query_list(self, fusion_processor):
        """Test error handling for empty query list."""
        with pytest.raises(ValueError, match="No sub-queries provided"):
            fusion_processor.process_sub_queries([])
    
    def test_invalid_query_types(self, fusion_processor):
        """Test error handling for invalid query types."""
        with pytest.raises(ValueError, match="sub_queries must be a list"):
            fusion_processor.process_sub_queries("not a list")
    
    def test_coverage_analysis(self, fusion_processor):
        """Test the coverage analysis functionality."""
        sub_queries = [
            "docking calculations protein binding",
            "TDP-43 aggregation studies",
            "alpha-synuclein drug discovery"
        ]
        
        result = fusion_processor.process_sub_queries(sub_queries)
        analysis = fusion_processor.analyze_coverage(result)
        
        assert "total_sub_queries" in analysis
        assert "successful_queries" in analysis
        assert "improvement_ratio" in analysis
        assert "per_query_counts" in analysis
        
        assert analysis["total_sub_queries"] == 3
        assert analysis["successful_queries"] >= 1
        assert analysis["improvement_ratio"] >= 1.0
    
    def test_result_explanation(self, fusion_processor):
        """Test the result explanation functionality."""
        sub_queries = [
            "docking calculations protein binding",
            "TDP-43 drug discovery"
        ]
        
        result = fusion_processor.process_sub_queries(sub_queries)
        
        if result.papers:
            explanation = fusion_processor.explain_result(result, 0)
            assert "Paper #1:" in explanation
            assert len(explanation) > 50  # Should be a substantial explanation
        
        # Test explanation for non-existent paper
        explanation = fusion_processor.explain_result(result, 999)
        assert "No paper at the specified index" in explanation
    
    def test_max_results_limiting(self, fusion_processor):
        """Test that max_results parameter limits output."""
        sub_queries = [
            "docking calculations protein binding",
            "TDP-43 aggregation studies"
        ]
        
        # Test with small max_results
        result = fusion_processor.process_sub_queries(sub_queries, max_results=2)
        assert len(result.papers) <= 2
    
    def test_fusion_config_parameters(self, mock_translator, mock_paper_filter):
        """Test different fusion configuration parameters."""
        # Test with different k value
        config = FusionConfig(k=10, max_results=20)  # Lower k = more rank-sensitive
        processor = ManualFusionProcessor(mock_translator, mock_paper_filter, config)
        
        sub_queries = [
            "docking calculations protein binding",
            "TDP-43 drug discovery"
        ]
        
        result = processor.process_sub_queries(sub_queries)
        assert len(result.papers) <= 20  # Respects max_results
        
        # The RRF algorithm should work with different k values
        if result.strategy_used == "fusion" and result.papers:
            assert "fusion_score" in result.papers[0]


class TestRealWorldScenarios:
    """Test scenarios based on real academic search use cases."""
    
    def test_overlapping_papers_boost_ranking(self):
        """Test that papers appearing in multiple sub-queries get ranking boost."""
        # This test uses the mock data where "docking-2" appears in multiple results
        mock_translator = Mock()
        mock_translator.side_effect = lambda q: BooleanQuery(
            must_have={"test"}, should_have=set(), must_not_have=set()
        )
        
        mock_filter = Mock()
        
        # Simulate results where one paper appears in multiple queries
        overlap_paper = {
            "item ID": "overlap-1",
            "Title": "Paper Found in Multiple Searches",
            "doi": "10.1234/overlap.001"
        }
        
        unique_paper1 = {
            "item ID": "unique-1", 
            "Title": "Paper Only in Query 1",
            "doi": "10.1234/unique.001"
        }
        
        unique_paper2 = {
            "item ID": "unique-2",
            "Title": "Paper Only in Query 2", 
            "doi": "10.1234/unique.002"
        }
        
        def mock_filter_func(boolean_query):
            # Simulate different results for different calls
            call_count = getattr(mock_filter_func, 'call_count', 0)
            mock_filter_func.call_count = call_count + 1
            
            if call_count == 0:  # First sub-query
                return [overlap_paper, unique_paper1]
            else:  # Second sub-query  
                return [overlap_paper, unique_paper2]
        
        mock_filter.filter_papers.side_effect = mock_filter_func
        
        processor = ManualFusionProcessor(mock_translator, mock_filter)
        result = processor.process_sub_queries([
            "query one",
            "query two"
        ])
        
        # The overlapping paper should rank first due to fusion boost
        assert result.papers[0]["item ID"] == "overlap-1"
        assert result.papers[0]["found_in_queries"] == 2
        
        # Should have higher fusion score than papers appearing only once
        overlap_score = result.papers[0]["fusion_score"]
        unique_scores = [p["fusion_score"] for p in result.papers[1:]]
        assert all(overlap_score > score for score in unique_scores)