"""Fusion processors for handling multi-query paper search.

This module contains processors for combining results from multiple sub-queries
using Reciprocal Rank Fusion to improve coverage for complex queries.
"""

from typing import List, Dict, Optional, Any
import logging
import time

from ..core.types import FusionResult, FusionConfig, BooleanQuery
from ..core.fusion import ReciprocalRankFusion
from ..core.exceptions import QueryTranslationError, PaperFilterError

logger = logging.getLogger(__name__)


class ManualFusionProcessor:
    """
    Process manually provided sub-queries with fusion.
    
    This processor allows testing fusion effectiveness with hand-crafted sub-queries
    before implementing automatic query decomposition. It provides detailed logging
    and error handling for each sub-query processing step.
    
    Example:
        >>> from readcube_mcp.query2label.fusion import ManualFusionProcessor
        >>> from readcube_mcp.query2label.core.types import FusionConfig
        >>> 
        >>> processor = ManualFusionProcessor(translator, paper_filter)
        >>> sub_queries = [
        ...     "docking calculations intrinsically disordered proteins",
        ...     "TDP-43 ligand binding drug discovery",
        ...     "alpha-synuclein small molecule inhibitors"
        ... ]
        >>> result = processor.process_sub_queries(sub_queries)
        >>> print(f"Found {len(result.papers)} papers with {result.improvement_ratio():.1f}x improvement")
    """
    
    def __init__(
        self, 
        translator,  # AdvancedQueryTranslator
        paper_filter,  # PaperFilter
        config: Optional[FusionConfig] = None
    ):
        """
        Initialize the manual fusion processor.
        
        Args:
            translator: Query translator (AdvancedQueryTranslator instance)
            paper_filter: Paper filter (PaperFilter instance)  
            config: Fusion configuration, uses defaults if None
        """
        self.translator = translator
        self.paper_filter = paper_filter
        self.config = config or FusionConfig()
        self.rrf = ReciprocalRankFusion(k=self.config.k)
        
        logger.info(f"Initialized ManualFusionProcessor with config: {self.config}")
    
    def process_sub_queries(
        self, 
        sub_queries: List[str], 
        max_results: Optional[int] = None
    ) -> FusionResult:
        """
        Process list of sub-queries independently and fuse results.
        
        This method:
        1. Validates input sub-queries
        2. Translates each sub-query to boolean logic
        3. Filters papers for each boolean query
        4. Applies RRF fusion if multiple sub-queries
        5. Returns comprehensive results with metadata
        
        Args:
            sub_queries: List of natural language sub-queries
            max_results: Override max results from config
            
        Returns:
            FusionResult with fused papers and detailed metadata
            
        Raises:
            ValueError: If no sub-queries provided
            QueryTranslationError: If critical translation errors occur
            PaperFilterError: If critical filtering errors occur
        """
        start_time = time.time()
        max_results = max_results or self.config.max_results
        
        # Validate input
        if not sub_queries:
            raise ValueError("No sub-queries provided")
        
        if not isinstance(sub_queries, list):
            raise ValueError("sub_queries must be a list")
        
        # Clean and validate sub-queries
        clean_queries = [q.strip() for q in sub_queries if q and q.strip()]
        if not clean_queries:
            raise ValueError("No valid sub-queries after cleaning")
        
        logger.info(f"Processing {len(clean_queries)} sub-queries with max_results={max_results}")
        
        # Handle single query case (no fusion needed)
        if len(clean_queries) == 1:
            return self._process_single_query(clean_queries[0], max_results, start_time)
        
        # Process multiple sub-queries
        individual_results = {}
        boolean_queries = []
        total_papers = 0
        successful_queries = 0
        errors = []
        
        for i, sub_query in enumerate(clean_queries):
            logger.info(f"Processing sub-query {i+1}/{len(clean_queries)}: '{sub_query[:60]}...'")
            
            try:
                # Translate to boolean query
                query_start = time.time()
                boolean_query = self.translator(sub_query)
                boolean_queries.append(boolean_query)
                
                # Validate boolean query
                if boolean_query.is_empty():
                    logger.warning(f"Sub-query '{sub_query}' produced empty boolean query")
                    individual_results[sub_query] = []
                    continue
                
                # Filter papers
                papers = self.paper_filter.filter_papers(boolean_query)
                individual_results[sub_query] = papers
                total_papers += len(papers)
                successful_queries += 1
                
                query_time = time.time() - query_start
                logger.info(
                    f"Sub-query '{sub_query[:50]}...' completed in {query_time:.2f}s: "
                    f"found {len(papers)} papers (boolean: {boolean_query})"
                )
                
            except QueryTranslationError as e:
                error_msg = f"Translation failed for sub-query '{sub_query}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Add empty results and empty boolean query to maintain alignment
                individual_results[sub_query] = []
                boolean_queries.append(BooleanQuery(set(), set(), set()))
                
            except PaperFilterError as e:
                error_msg = f"Filtering failed for sub-query '{sub_query}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # Add empty results but keep the boolean query
                individual_results[sub_query] = []
                
            except Exception as e:
                error_msg = f"Unexpected error processing sub-query '{sub_query}': {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                
                # Add empty results and empty boolean query
                individual_results[sub_query] = []
                boolean_queries.append(BooleanQuery(set(), set(), set()))
        
        # Check if we have any successful results
        if successful_queries == 0:
            logger.error("All sub-queries failed - no results to fuse")
            total_time = time.time() - start_time
            return FusionResult(
                papers=[],
                sub_queries=clean_queries,
                boolean_queries=boolean_queries,
                fusion_scores={},
                individual_results=individual_results,
                strategy_used="fusion_failed",
                total_papers_before_fusion=0
            )
        
        # Apply fusion
        try:
            # Only fuse results from successful queries
            non_empty_results = {
                query: papers for query, papers in individual_results.items() 
                if papers
            }
            
            if non_empty_results:
                fused_papers, fusion_scores = self.rrf.fuse_results(
                    non_empty_results, max_results
                )
                strategy = "fusion"
                
                logger.info(
                    f"Fusion completed: {len(non_empty_results)} non-empty results → "
                    f"{len(fused_papers)} fused papers"
                )
            else:
                # No non-empty results to fuse
                fused_papers = []
                fusion_scores = {}
                strategy = "fusion_empty"
                logger.warning("No non-empty results to fuse")
                
        except Exception as e:
            logger.error(f"Fusion failed: {e}", exc_info=True)
            # Fallback: return best individual result
            best_result = max(individual_results.values(), key=len, default=[])
            fused_papers = best_result[:max_results]
            fusion_scores = {}
            strategy = "fusion_failed_fallback"
        
        total_time = time.time() - start_time
        
        result = FusionResult(
            papers=fused_papers,
            sub_queries=clean_queries,
            boolean_queries=boolean_queries,
            fusion_scores=fusion_scores,
            individual_results=individual_results,
            strategy_used=strategy,
            total_papers_before_fusion=total_papers
        )
        
        logger.info(
            f"Manual fusion completed in {total_time:.2f}s: "
            f"{len(clean_queries)} queries → {total_papers} total papers → "
            f"{len(fused_papers)} fused results "
            f"(improvement: {result.improvement_ratio():.1f}x)"
        )
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during processing: {errors}")
        
        return result
    
    def _process_single_query(
        self, 
        query: str, 
        max_results: int,
        start_time: float
    ) -> FusionResult:
        """
        Process a single query without fusion.
        
        Args:
            query: Natural language query
            max_results: Maximum results to return
            start_time: Processing start time for timing
            
        Returns:
            FusionResult with simple strategy
        """
        logger.info(f"Processing single query (no fusion): '{query}'")
        
        try:
            # Translate and filter
            boolean_query = self.translator(query)
            papers = self.paper_filter.filter_papers(boolean_query)[:max_results]
            
            total_time = time.time() - start_time
            
            logger.info(
                f"Single query completed in {total_time:.2f}s: "
                f"found {len(papers)} papers (boolean: {boolean_query})"
            )
            
            return FusionResult(
                papers=papers,
                sub_queries=[query],
                boolean_queries=[boolean_query],
                fusion_scores={},
                individual_results={query: papers},
                strategy_used="simple",
                total_papers_before_fusion=len(papers)
            )
            
        except Exception as e:
            logger.error(f"Single query processing failed: {e}", exc_info=True)
            return FusionResult(
                papers=[],
                sub_queries=[query],
                boolean_queries=[BooleanQuery(set(), set(), set())],
                fusion_scores={},
                individual_results={query: []},
                strategy_used="simple_failed",
                total_papers_before_fusion=0
            )
    
    def analyze_coverage(self, result: FusionResult) -> Dict[str, Any]:
        """
        Analyze the coverage and effectiveness of fusion results.
        
        Args:
            result: FusionResult to analyze
            
        Returns:
            Dictionary with coverage analysis metrics
        """
        analysis = {
            "total_sub_queries": len(result.sub_queries),
            "successful_queries": len([q for q, papers in result.individual_results.items() if papers]),
            "total_unique_papers": len(set().union(*[
                {self.rrf._get_paper_id(p) for p in papers}
                for papers in result.individual_results.values()
            ])),
            "papers_in_multiple_queries": 0,
            "improvement_ratio": result.improvement_ratio(),
            "strategy_used": result.strategy_used,
            "per_query_counts": result.coverage_per_query()
        }
        
        # Count papers appearing in multiple queries
        if result.strategy_used == "fusion":
            paper_query_count = {}
            for query, papers in result.individual_results.items():
                for paper in papers:
                    paper_id = self.rrf._get_paper_id(paper)
                    paper_query_count[paper_id] = paper_query_count.get(paper_id, 0) + 1
            
            analysis["papers_in_multiple_queries"] = sum(
                1 for count in paper_query_count.values() if count > 1
            )
            analysis["max_paper_occurrences"] = max(paper_query_count.values(), default=0)
            analysis["avg_paper_occurrences"] = (
                sum(paper_query_count.values()) / len(paper_query_count)
                if paper_query_count else 0
            )
        
        return analysis
    
    def explain_result(self, result: FusionResult, paper_index: int = 0) -> str:
        """
        Generate human-readable explanation of why a specific paper ranked where it did.
        
        Args:
            result: FusionResult to explain
            paper_index: Index of paper to explain (0 = top result)
            
        Returns:
            Explanation string
        """
        if not result.papers or paper_index >= len(result.papers):
            return "No paper at the specified index"
        
        paper = result.papers[paper_index]
        explanation = f"Paper #{paper_index + 1}: {paper.get('Title', paper.get('title', 'Unknown'))}\n"
        
        if result.strategy_used == "simple":
            explanation += f"Strategy: Single query (no fusion)\n"
            explanation += f"Found in query: '{result.sub_queries[0]}'\n"
        elif result.strategy_used == "fusion":
            explanation += self.rrf.get_explanation(paper)
        else:
            explanation += f"Strategy: {result.strategy_used}\n"
        
        return explanation