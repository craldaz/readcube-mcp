"""Unit tests for the Reciprocal Rank Fusion algorithm."""

import pytest
from typing import Dict, List, Any

from readcube_mcp.query2label.core.fusion import ReciprocalRankFusion


class TestReciprocalRankFusion:
    """Test suite for RRF algorithm implementation."""
    
    def test_init_valid_k(self):
        """Test initialization with valid k values."""
        rrf = ReciprocalRankFusion(k=60)
        assert rrf.k == 60
        
        rrf = ReciprocalRankFusion(k=1)
        assert rrf.k == 1
        
        rrf = ReciprocalRankFusion(k=100)
        assert rrf.k == 100
    
    def test_init_invalid_k(self):
        """Test initialization with invalid k values."""
        with pytest.raises(ValueError, match="k must be positive"):
            ReciprocalRankFusion(k=0)
        
        with pytest.raises(ValueError, match="k must be positive"):
            ReciprocalRankFusion(k=-10)
    
    def test_empty_results(self):
        """Test fusion with empty results."""
        rrf = ReciprocalRankFusion()
        
        # Empty dict
        papers, scores = rrf.fuse_results({})
        assert papers == []
        assert scores == {}
        
        # Dict with empty lists
        papers, scores = rrf.fuse_results({"query1": [], "query2": []})
        assert papers == []
        assert scores == {}
    
    def test_single_query(self):
        """Test fusion with single query (no actual fusion needed)."""
        rrf = ReciprocalRankFusion(k=60)
        
        papers_list = [
            {"id": "paper1", "title": "First Paper"},
            {"id": "paper2", "title": "Second Paper"},
            {"id": "paper3", "title": "Third Paper"}
        ]
        
        results = {"query1": papers_list}
        fused_papers, scores = rrf.fuse_results(results)
        
        # All papers should be returned in original order
        assert len(fused_papers) == 3
        assert [p["id"] for p in fused_papers] == ["paper1", "paper2", "paper3"]
        
        # Check RRF scores (1/(rank+k))
        assert scores["paper1"] == pytest.approx(1.0 / (1 + 60))
        assert scores["paper2"] == pytest.approx(1.0 / (2 + 60))
        assert scores["paper3"] == pytest.approx(1.0 / (3 + 60))
        
        # Check metadata
        assert fused_papers[0]["fusion_rank"] == 1
        assert fused_papers[0]["found_in_queries"] == 1
    
    def test_multiple_queries_no_overlap(self):
        """Test fusion with multiple queries but no overlapping papers."""
        rrf = ReciprocalRankFusion(k=60)
        
        results = {
            "query1": [
                {"id": "A", "title": "Paper A"},
                {"id": "B", "title": "Paper B"}
            ],
            "query2": [
                {"id": "C", "title": "Paper C"},
                {"id": "D", "title": "Paper D"}
            ]
        }
        
        fused_papers, scores = rrf.fuse_results(results)
        
        # All 4 papers should be present
        assert len(fused_papers) == 4
        paper_ids = [p["id"] for p in fused_papers]
        assert set(paper_ids) == {"A", "B", "C", "D"}
        
        # Papers from same position should have same score
        assert scores["A"] == scores["C"]  # Both rank 1
        assert scores["B"] == scores["D"]  # Both rank 2
        
        # Higher ranked papers should have higher scores
        assert scores["A"] > scores["B"]
        assert scores["C"] > scores["D"]
    
    def test_multiple_queries_with_overlap(self):
        """Test fusion with overlapping papers (the main use case)."""
        rrf = ReciprocalRankFusion(k=60)
        
        results = {
            "query1": [
                {"id": "A", "title": "Paper A"},
                {"id": "B", "title": "Paper B"},
                {"id": "C", "title": "Paper C"}
            ],
            "query2": [
                {"id": "B", "title": "Paper B"},  # Rank 1 in query2
                {"id": "D", "title": "Paper D"},
                {"id": "A", "title": "Paper A"}   # Rank 3 in query2
            ],
            "query3": [
                {"id": "B", "title": "Paper B"},  # Rank 1 in query3
                {"id": "E", "title": "Paper E"}
            ]
        }
        
        fused_papers, scores = rrf.fuse_results(results)
        
        # Check total unique papers
        assert len(fused_papers) == 5
        
        # Paper B should rank first (appears in all 3 queries at high ranks)
        assert fused_papers[0]["id"] == "B"
        assert fused_papers[0]["found_in_queries"] == 3
        
        # Paper A should rank high (appears in 2 queries)
        paper_a_rank = next(i for i, p in enumerate(fused_papers) if p["id"] == "A")
        assert paper_a_rank < 3  # Should be in top 3
        
        # Check score calculation for Paper B
        expected_score_b = (1.0 / (2 + 60) +  # Rank 2 in query1
                           1.0 / (1 + 60) +   # Rank 1 in query2
                           1.0 / (1 + 60))    # Rank 1 in query3
        assert scores["B"] == pytest.approx(expected_score_b)
    
    def test_max_results_limit(self):
        """Test that max_results parameter limits output."""
        rrf = ReciprocalRankFusion()
        
        results = {
            "query1": [{"id": str(i), "title": f"Paper {i}"} for i in range(10)]
        }
        
        # Test with max_results=5
        fused_papers, scores = rrf.fuse_results(results, max_results=5)
        assert len(fused_papers) == 5
        assert len(scores) == 5
        
        # Should get top 5 papers
        assert [p["id"] for p in fused_papers] == ["0", "1", "2", "3", "4"]
    
    def test_paper_id_extraction(self):
        """Test various paper ID extraction strategies."""
        rrf = ReciprocalRankFusion()
        
        # Test with explicit id field
        paper = {"id": "12345", "title": "Test Paper"}
        assert rrf._get_paper_id(paper) == "12345"
        
        # Test with DOI
        paper = {"doi": "10.1234/test", "title": "Test Paper"}
        assert rrf._get_paper_id(paper) == "10.1234/test"
        
        # Test with title and year
        paper = {"title": "Machine Learning Paper", "year": 2023}
        assert rrf._get_paper_id(paper) == "Machine Learning Paper_2023"
        
        # Test with only title
        paper = {"title": "Another Paper"}
        assert rrf._get_paper_id(paper) == "Another Paper"
        
        # Test hash generation for paper without good ID
        paper = {"abstract": "Some abstract", "venue": "Conference"}
        paper_id = rrf._get_paper_id(paper)
        assert paper_id is not None
        assert len(paper_id) > 0
    
    def test_acelot_csv_field_extraction(self):
        """Test ID extraction with Acelot Library CSV field names."""
        rrf = ReciprocalRankFusion()
        
        # Test with Acelot CSV structure
        acelot_paper = {
            "item ID": "530a0578-23cf-4f07-9ab1-0af15e7dd766",
            "doi": "10.15252/embr.201642683",
            "pmid": "27979972",
            "pmcid": "PMC5210122",
            "Title": "Phosphorylation by NLK inhibits YAP interactions",
            "year": 2017,
            "Author": "Moon, Sungho; Kim, Wantae",
            "Journal": "EMBO reports"
        }
        
        # Should prefer item ID first
        assert rrf._get_paper_id(acelot_paper) == "530a0578-23cf-4f07-9ab1-0af15e7dd766"
        
        # Test fallback to DOI
        paper_no_item_id = acelot_paper.copy()
        del paper_no_item_id["item ID"]
        assert rrf._get_paper_id(paper_no_item_id) == "10.15252/embr.201642683"
        
        # Test fallback to PMID
        paper_no_doi = paper_no_item_id.copy()
        del paper_no_doi["doi"]
        assert rrf._get_paper_id(paper_no_doi) == "27979972"
        
        # Test fallback to Title + year
        paper_title_only = {
            "Title": "Phosphorylation by NLK inhibits YAP interactions",
            "year": 2017
        }
        assert rrf._get_paper_id(paper_title_only) == "Phosphorylation by NLK inhibits YAP interactions_2017"
    
    def test_different_k_values(self):
        """Test how different k values affect ranking."""
        results = {
            "query1": [
                {"id": "A", "title": "Paper A"},  # Rank 1
                {"id": "B", "title": "Paper B"},  # Rank 2
                {"id": "C", "title": "Paper C"}   # Rank 3
            ],
            "query2": [
                {"id": "B", "title": "Paper B"},  # Rank 1
                {"id": "D", "title": "Paper D"},  # Rank 2
                {"id": "E", "title": "Paper E"}   # Rank 3
            ]
        }
        
        # Low k (more weight on rank differences)
        rrf_low_k = ReciprocalRankFusion(k=10)
        papers_low_k, scores_low_k = rrf_low_k.fuse_results(results)
        
        # High k (less weight on rank differences)
        rrf_high_k = ReciprocalRankFusion(k=100)
        papers_high_k, scores_high_k = rrf_high_k.fuse_results(results)
        
        # Paper B appears in both lists (rank 2 in query1, rank 1 in query2)
        # It should rank first in both cases due to appearing twice
        assert papers_low_k[0]["id"] == "B"
        assert papers_high_k[0]["id"] == "B"
        
        # Calculate B's scores with different k values
        # B gets: 1/(2+k) from query1 + 1/(1+k) from query2
        expected_b_low = 1.0/(2+10) + 1.0/(1+10)  # = 1/12 + 1/11 ≈ 0.174
        expected_b_high = 1.0/(2+100) + 1.0/(1+100)  # = 1/102 + 1/101 ≈ 0.0198
        
        assert scores_low_k["B"] == pytest.approx(expected_b_low)
        assert scores_high_k["B"] == pytest.approx(expected_b_high)
        
        # Paper A only appears once at rank 1
        # A gets: 1/(1+k) from query1
        expected_a_low = 1.0/(1+10)  # = 1/11 ≈ 0.0909
        expected_a_high = 1.0/(1+100)  # = 1/101 ≈ 0.0099
        
        assert scores_low_k["A"] == pytest.approx(expected_a_low)
        assert scores_high_k["A"] == pytest.approx(expected_a_high)
    
    def test_explanation_generation(self):
        """Test the explanation generation for fusion results."""
        rrf = ReciprocalRankFusion(k=60)
        
        results = {
            "protein folding query": [
                {"id": "A", "title": "Protein Folding Study"},
                {"id": "B", "title": "Another Study"}
            ],
            "machine learning query": [
                {"id": "A", "title": "Protein Folding Study"}
            ]
        }
        
        fused_papers, _ = rrf.fuse_results(results)
        
        # Get explanation for top paper
        explanation = rrf.get_explanation(fused_papers[0])
        
        assert "Fusion rank: #1" in explanation
        assert "Found in 2 sub-queries" in explanation
        assert "protein folding query" in explanation
        assert "machine learning query" in explanation
        assert "contributed" in explanation
    
    def test_duplicate_papers_in_same_query(self):
        """Test handling of duplicate papers within the same query."""
        rrf = ReciprocalRankFusion()
        
        # Simulate a query that somehow returned duplicates
        results = {
            "query1": [
                {"id": "A", "title": "Paper A"},
                {"id": "A", "title": "Paper A"},  # Duplicate
                {"id": "B", "title": "Paper B"}
            ]
        }
        
        fused_papers, scores = rrf.fuse_results(results)
        
        # Should handle duplicates gracefully
        assert len(fused_papers) == 2
        assert set(p["id"] for p in fused_papers) == {"A", "B"}


class TestRRFIntegration:
    """Integration tests for RRF with realistic data."""
    
    def test_realistic_query_decomposition_example(self):
        """Test with a realistic example from query decomposition."""
        rrf = ReciprocalRankFusion(k=60)
        
        # Simulate results from decomposed sub-queries for IDP docking
        results = {
            "docking calculations intrinsically disordered proteins": [
                {"id": "idp-dock-1", "title": "Docking to IDPs: Challenges and Methods"},
                {"id": "idp-review", "title": "IDP Structure and Function Review"},
                {"id": "tdp43-dock", "title": "TDP-43 Small Molecule Docking"}
            ],
            "TDP-43 ligand binding drug discovery": [
                {"id": "tdp43-dock", "title": "TDP-43 Small Molecule Docking"},
                {"id": "tdp43-drug", "title": "Drug Discovery for TDP-43 Aggregation"},
                {"id": "als-therapy", "title": "ALS Therapeutic Targets"}
            ],
            "alpha-synuclein small molecule inhibitors": [
                {"id": "asyn-inhibit", "title": "α-Synuclein Aggregation Inhibitors"},
                {"id": "pd-drugs", "title": "Parkinson's Disease Drug Development"},
                {"id": "idp-review", "title": "IDP Structure and Function Review"}
            ]
        }
        
        fused_papers, scores = rrf.fuse_results(results, max_results=10)
        
        # Papers appearing in multiple queries should rank higher
        paper_ids = [p["id"] for p in fused_papers]
        
        # tdp43-dock appears in 2 queries, should be near top
        tdp43_rank = paper_ids.index("tdp43-dock")
        assert tdp43_rank < 3
        
        # idp-review appears in 2 queries, should also rank well
        review_rank = paper_ids.index("idp-review")
        assert review_rank < 4
        
        # Check metadata is correct
        tdp43_paper = next(p for p in fused_papers if p["id"] == "tdp43-dock")
        assert tdp43_paper["found_in_queries"] == 2
        
        # Should find all unique papers
        all_unique_ids = {"idp-dock-1", "idp-review", "tdp43-dock", 
                         "tdp43-drug", "als-therapy", "asyn-inhibit", "pd-drugs"}
        assert set(paper_ids) == all_unique_ids