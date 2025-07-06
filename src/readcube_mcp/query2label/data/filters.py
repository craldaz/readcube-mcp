"""Paper filtering logic for Query2Label system.

This module contains the PaperFilter class that applies boolean queries
to paper databases and returns filtered, ranked results based on label
matching conditions.
"""

from typing import List, Dict, Set, Optional, Any
from ..core.types import BooleanQuery


class PaperFilter:
    """Filter papers based on boolean label queries with relevance scoring.
    
    This class implements the core filtering logic that takes a boolean query
    (with MUST, SHOULD, and NOT conditions) and applies it to a paper database
    to find matching papers. It also calculates relevance scores based on
    how well each paper matches the query conditions.
    """

    def __init__(self, papers_db: List[Dict[str, Any]]):
        """Initialize the paper filter with a paper database.
        
        Args:
            papers_db: List of paper dictionaries. Each paper should have:
                - 'labels': List[str] - Required for filtering
                - 'id': Any - Unique identifier (optional)
                - 'title': str - Paper title (optional)
                - 'abstract': str - Paper abstract (optional)
                - Other fields like 'url', 'year', 'journal', 'authors' are optional
        """
        self.papers_db = papers_db
        self._validate_papers_db()

    def _validate_papers_db(self) -> None:
        """Validate that the papers database has the required structure."""
        if not isinstance(self.papers_db, list):
            raise TypeError("papers_db must be a list of dictionaries")
        
        for i, paper in enumerate(self.papers_db):
            if not isinstance(paper, dict):
                raise TypeError(f"Paper at index {i} must be a dictionary")
            
            # Check for required 'labels' field
            if 'labels' not in paper:
                raise ValueError(f"Paper at index {i} missing required 'labels' field")
            
            # Ensure labels is a list
            if not isinstance(paper['labels'], list):
                raise TypeError(f"Paper at index {i} 'labels' field must be a list")

    def filter_papers(
        self, 
        boolean_query: BooleanQuery, 
        min_should_match: int = 1
    ) -> List[Dict[str, Any]]:
        """Filter papers based on boolean query conditions.
        
        This method applies the boolean query to the paper database and returns
        a filtered list of papers that match the query conditions, sorted by
        relevance score in descending order.
        
        Args:
            boolean_query: BooleanQuery object with MUST, SHOULD, and NOT conditions
            min_should_match: Minimum number of SHOULD conditions that must match
                             (default: 1, set to 0 to disable SHOULD filtering)
        
        Returns:
            List of paper dictionaries that match the query, sorted by relevance score.
            Each paper will have an added 'relevance_score' field (0.0 to 1.0).
        
        Raises:
            TypeError: If boolean_query is not a BooleanQuery instance
            ValueError: If min_should_match is negative
        """
        if not isinstance(boolean_query, BooleanQuery):
            raise TypeError("boolean_query must be a BooleanQuery instance")
        
        if min_should_match < 0:
            raise ValueError("min_should_match must be non-negative")

        filtered_papers = []

        for paper in self.papers_db:
            # Get paper labels as a set for efficient operations
            paper_labels = set(paper.get('labels', []))
            
            # Create a copy of the paper to avoid modifying the original
            filtered_paper = paper.copy()

            # Check MUST conditions (AND logic)
            if boolean_query.must_have:
                if not boolean_query.must_have.issubset(paper_labels):
                    continue  # Skip this paper if any MUST condition is not met

            # Check NOT conditions (exclusion logic)
            if boolean_query.must_not_have:
                if boolean_query.must_not_have.intersection(paper_labels):
                    continue  # Skip this paper if any NOT condition is met

            # Check SHOULD conditions (OR logic)
            if boolean_query.should_have:
                should_matches = len(
                    boolean_query.should_have.intersection(paper_labels)
                )
                
                if should_matches < min_should_match:
                    continue  # Skip this paper if not enough SHOULD conditions match

                # Calculate relevance score based on SHOULD condition matches
                filtered_paper['relevance_score'] = should_matches / len(boolean_query.should_have)
            else:
                # No SHOULD conditions, so give perfect score for papers that pass MUST/NOT
                filtered_paper['relevance_score'] = 1.0

            filtered_papers.append(filtered_paper)

        # Sort by relevance score in descending order (highest scores first)
        filtered_papers.sort(key=lambda x: x['relevance_score'], reverse=True)

        return filtered_papers

    def get_papers_count(self) -> int:
        """Get the total number of papers in the database.
        
        Returns:
            int: Total number of papers in the database
        """
        return len(self.papers_db)

    def get_unique_labels(self) -> Set[str]:
        """Get all unique labels from the paper database.
        
        Returns:
            Set[str]: Set of all unique labels found in the database
        """
        unique_labels = set()
        
        for paper in self.papers_db:
            labels = paper.get('labels', [])
            unique_labels.update(labels)
        
        return unique_labels

    def get_label_counts(self) -> Dict[str, int]:
        """Get usage counts for all labels in the database.
        
        Returns:
            Dict[str, int]: Dictionary mapping label names to their usage counts
        """
        label_counts = {}
        
        for paper in self.papers_db:
            labels = paper.get('labels', [])
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        return label_counts

    def filter_by_label_intersection(
        self, 
        required_labels: Set[str], 
        min_overlap: int = 1
    ) -> List[Dict[str, Any]]:
        """Filter papers that have at least min_overlap labels in common with required_labels.
        
        This is a simpler filtering method that doesn't use boolean logic,
        just checks for label overlap.
        
        Args:
            required_labels: Set of labels to match against
            min_overlap: Minimum number of labels that must overlap
        
        Returns:
            List of papers that have sufficient label overlap
        """
        filtered_papers = []
        
        for paper in self.papers_db:
            paper_labels = set(paper.get('labels', []))
            overlap = len(required_labels.intersection(paper_labels))
            
            if overlap >= min_overlap:
                filtered_paper = paper.copy()
                filtered_paper['label_overlap_count'] = overlap
                filtered_paper['overlap_ratio'] = overlap / len(required_labels) if required_labels else 0.0
                filtered_papers.append(filtered_paper)
        
        # Sort by overlap count (descending)
        filtered_papers.sort(key=lambda x: x['label_overlap_count'], reverse=True)
        
        return filtered_papers

    def get_papers_with_labels(self, labels: List[str]) -> List[Dict[str, Any]]:
        """Get all papers that contain any of the specified labels.
        
        Args:
            labels: List of labels to search for
        
        Returns:
            List of papers that contain at least one of the specified labels
        """
        target_labels = set(labels)
        matching_papers = []
        
        for paper in self.papers_db:
            paper_labels = set(paper.get('labels', []))
            if target_labels.intersection(paper_labels):
                matching_papers.append(paper.copy())
        
        return matching_papers

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics for analysis and debugging.
        
        Returns:
            Dict containing various statistics about the paper database
        """
        total_papers = len(self.papers_db)
        unique_labels = self.get_unique_labels()
        label_counts = self.get_label_counts()
        
        # Calculate some statistics
        papers_per_label = {label: count for label, count in label_counts.items()}
        avg_labels_per_paper = sum(len(paper.get('labels', [])) for paper in self.papers_db) / total_papers if total_papers > 0 else 0
        
        return {
            'total_papers': total_papers,
            'unique_labels_count': len(unique_labels),
            'unique_labels': sorted(unique_labels),
            'label_usage_counts': papers_per_label,
            'avg_labels_per_paper': avg_labels_per_paper,
            'most_common_labels': sorted(papers_per_label.items(), key=lambda x: x[1], reverse=True)[:10]
        }