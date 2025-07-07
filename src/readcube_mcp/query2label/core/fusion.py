"""Reciprocal Rank Fusion implementation for combining search results.

This module implements the RRF algorithm for merging ranked lists of papers
from multiple sub-queries into a single ranked list. RRF is particularly
effective for combining results from different search strategies while
maintaining relevance.
"""

from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import logging
import hashlib

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) algorithm for merging ranked lists.
    
    RRF combines multiple ranked lists by giving each item a score based on
    its rank in each list. The formula is: RRF_score = Σ(1 / (rank + k))
    where the sum is over all lists containing the item.
    
    The constant k controls the importance of rank vs. occurrence:
    - Lower k (e.g., 10): High-ranked items get much higher scores
    - Higher k (e.g., 60): More weight on items appearing in multiple lists
    - Default k=60 is commonly used in information retrieval
    
    Reference: Cormack, Clarke, and Buettcher. "Reciprocal rank fusion
    outperforms condorcet and individual rank learning methods." SIGIR 2009.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF with constant k.
        
        Args:
            k: RRF constant controlling rank vs. occurrence importance.
               Higher k = more weight on items appearing in multiple lists.
               Lower k = more weight on high-ranked items.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        logger.debug(f"Initialized RRF with k={k}")
    
    def fuse_results(
        self, 
        results_by_query: Dict[str, List[Dict[str, Any]]], 
        max_results: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Apply RRF to combine results from multiple queries.
        
        Args:
            results_by_query: Dictionary mapping sub-query strings to lists of papers.
                             Each paper should be a dict with at least a unique identifier.
            max_results: Maximum number of papers to return. If None, returns all.
            
        Returns:
            Tuple of (fused_papers, fusion_scores) where:
            - fused_papers: List of papers sorted by RRF score (highest first)
            - fusion_scores: Dict mapping paper IDs to their RRF scores
            
        Example:
            >>> rrf = ReciprocalRankFusion(k=60)
            >>> results = {
            ...     "query1": [{"id": "A", "title": "Paper A"}, {"id": "B", "title": "Paper B"}],
            ...     "query2": [{"id": "B", "title": "Paper B"}, {"id": "C", "title": "Paper C"}]
            ... }
            >>> papers, scores = rrf.fuse_results(results)
            >>> # Paper B appears in both lists, so gets higher score
        """
        if not results_by_query:
            logger.warning("No results to fuse - empty results_by_query")
            return [], {}
        
        # Track RRF scores and paper objects
        fused_scores = defaultdict(float)
        paper_objects = {}
        paper_occurrences = defaultdict(list)  # Track which queries found each paper
        
        # Calculate RRF scores for each paper
        total_papers_processed = 0
        for query, papers in results_by_query.items():
            if not papers:
                logger.debug(f"Query '{query[:50]}...' returned no results")
                continue
                
            for rank, paper in enumerate(papers):
                paper_id = self._get_paper_id(paper)
                if not paper_id:
                    logger.warning(f"Could not extract ID for paper: {paper}")
                    continue
                    
                # Store the paper object (last occurrence wins to handle duplicates)
                paper_objects[paper_id] = paper
                
                # RRF formula: 1 / (rank + k)
                # Note: rank is 0-indexed, so rank=0 is the top result
                rrf_score = 1.0 / (rank + 1 + self.k)
                fused_scores[paper_id] += rrf_score
                paper_occurrences[paper_id].append((query, rank + 1))
                
                total_papers_processed += 1
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Paper '{paper.get('title', paper_id)[:50]}...' "
                        f"rank {rank + 1} in query '{query[:30]}...' "
                        f"contributes RRF score {rrf_score:.4f}"
                    )
        
        logger.info(
            f"Processed {total_papers_processed} paper occurrences from "
            f"{len(results_by_query)} queries into {len(fused_scores)} unique papers"
        )
        
        # Sort by fused score (highest first)
        sorted_items = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Apply max_results limit if specified
        if max_results is not None and max_results > 0:
            sorted_items = sorted_items[:max_results]
        
        # Prepare final results with metadata
        fused_papers = []
        fusion_scores_dict = {}
        
        for rank, (paper_id, score) in enumerate(sorted_items):
            if paper_id not in paper_objects:
                logger.warning(f"Paper ID {paper_id} not found in paper_objects")
                continue
                
            # Create a copy to avoid modifying the original
            paper = paper_objects[paper_id].copy()
            
            # Add fusion metadata
            paper['fusion_score'] = score
            paper['fusion_rank'] = rank + 1
            paper['found_in_queries'] = len(paper_occurrences[paper_id])
            paper['query_ranks'] = paper_occurrences[paper_id]
            
            fused_papers.append(paper)
            fusion_scores_dict[paper_id] = score
        
        logger.info(
            f"RRF fusion complete: {len(results_by_query)} queries → "
            f"{sum(len(p) for p in results_by_query.values())} total papers → "
            f"{len(fused_papers)} unique fused results"
        )
        
        # Log top results for debugging
        if fused_papers and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Top 5 fusion results:")
            for i, paper in enumerate(fused_papers[:5]):
                logger.debug(
                    f"  {i+1}. {paper.get('title', 'Unknown')[:60]}... "
                    f"(score: {paper['fusion_score']:.3f}, "
                    f"found in {paper['found_in_queries']} queries)"
                )
        
        return fused_papers, fusion_scores_dict
    
    def _get_paper_id(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Extract a unique identifier for a paper.
        
        Tries multiple strategies to find a unique ID:
        1. Look for explicit ID fields (id, paper_id, doi, url)
        2. Use title if unique enough
        3. Generate hash from paper content as last resort
        
        Args:
            paper: Paper dictionary
            
        Returns:
            String identifier or None if paper is invalid
        """
        if not isinstance(paper, dict):
            return None
        
        # Strategy 1: Try explicit ID fields in order of preference
        # Optimized for Acelot Library CSV structure
        id_fields = [
            'item ID',      # Acelot Library UUID
            'id',           # Generic ID field
            'paper_id',     # Alternative ID field
            'doi',          # Digital Object Identifier
            'pmid',         # PubMed ID
            'pmcid',        # PubMed Central ID
            'arxiv',        # ArXiv ID
            'arxiv_id',     # Alternative ArXiv field
            'pubmed_id',    # Alternative PubMed field
            'Library URL',  # ReadCube library URL
            'url'           # Generic URL
        ]
        for field in id_fields:
            if field in paper and paper[field]:
                paper_id = str(paper[field]).strip()
                if paper_id:
                    return paper_id
        
        # Strategy 2: Use title if available and substantial
        # Check both 'title' and 'Title' (CSV has 'Title')
        title = paper.get('Title', paper.get('title', '')).strip()
        if title and len(title) > 10:  # Avoid very short titles
            # Include year if available to handle papers with same title
            year = paper.get('year', paper.get('Year', ''))
            if year:
                return f"{title}_{year}"
            return title
        
        # Strategy 3: Generate hash from paper content
        # This ensures consistency but is not human-readable
        try:
            # Create a stable hash from sorted key-value pairs
            content_items = []
            for key in sorted(paper.keys()):
                if key not in ['fusion_score', 'fusion_rank', 'relevance_score']:
                    value = paper[key]
                    if isinstance(value, (str, int, float)):
                        content_items.append(f"{key}:{value}")
            
            if content_items:
                content_str = "|".join(content_items)
                return hashlib.md5(content_str.encode('utf-8')).hexdigest()[:12]
        except Exception as e:
            logger.warning(f"Failed to generate hash for paper: {e}")
        
        # Last resort: use object id (not stable across runs)
        return f"paper_{id(paper)}"
    
    def get_explanation(self, paper: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of why a paper ranked where it did.
        
        Args:
            paper: Paper dict with fusion metadata
            
        Returns:
            Explanation string
        """
        if 'fusion_score' not in paper:
            return "Paper was not processed through fusion"
        
        score = paper.get('fusion_score', 0)
        rank = paper.get('fusion_rank', 'Unknown')
        found_in = paper.get('found_in_queries', 0)
        query_ranks = paper.get('query_ranks', [])
        
        explanation = f"Fusion rank: #{rank} (RRF score: {score:.3f})\n"
        explanation += f"Found in {found_in} sub-queries:\n"
        
        for query, original_rank in query_ranks:
            contribution = 1.0 / (original_rank + self.k)
            explanation += f"  - '{query[:50]}...' at rank #{original_rank} "
            explanation += f"(contributed {contribution:.3f} to score)\n"
        
        return explanation