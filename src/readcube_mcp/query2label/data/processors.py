"""Data processing utilities for the Query2Label system.

This module contains utilities for processing and cleaning paper data,
including text normalization, label standardization, and data validation.
"""

import re
from typing import List, Dict, Set, Any, Optional, Tuple
from collections import Counter
from ..core.exceptions import DataProcessingError


class LabelProcessor:
    """Process and standardize labels from paper datasets."""
    
    def __init__(self, 
                 normalize_case: bool = True,
                 remove_special_chars: bool = True,
                 min_label_length: int = 2,
                 max_label_length: int = 100):
        """Initialize the label processor.
        
        Args:
            normalize_case: Whether to normalize labels to lowercase
            remove_special_chars: Whether to remove special characters
            min_label_length: Minimum length for valid labels
            max_label_length: Maximum length for valid labels
        """
        self.normalize_case = normalize_case
        self.remove_special_chars = remove_special_chars
        self.min_label_length = min_label_length
        self.max_label_length = max_label_length
        
    def process_labels(self, labels: List[str]) -> List[str]:
        """Process a list of labels with standardization and cleaning.
        
        Args:
            labels: List of raw labels
            
        Returns:
            List of processed and cleaned labels
        """
        processed_labels = []
        
        for label in labels:
            processed_label = self._process_single_label(label)
            if processed_label:
                processed_labels.append(processed_label)
        
        return processed_labels
    
    def _process_single_label(self, label: str) -> Optional[str]:
        """Process a single label with all cleaning steps.
        
        Args:
            label: Raw label string
            
        Returns:
            Processed label or None if invalid
        """
        if not label or not label.strip():
            return None
        
        # Start with stripped label
        processed = label.strip()
        
        # Normalize case
        if self.normalize_case:
            processed = processed.lower()
        
        # Remove special characters (keep alphanumeric, spaces, hyphens, underscores)
        if self.remove_special_chars:
            processed = re.sub(r'[^a-zA-Z0-9\s\-_]', '', processed)
        
        # Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Replace spaces with hyphens for consistency
        processed = processed.replace(' ', '-')
        
        # Check length constraints
        if len(processed) < self.min_label_length or len(processed) > self.max_label_length:
            return None
        
        return processed
    
    def deduplicate_labels(self, labels: List[str]) -> List[str]:
        """Remove duplicate labels while preserving order.
        
        Args:
            labels: List of labels that may contain duplicates
            
        Returns:
            List of unique labels in original order
        """
        seen = set()
        unique_labels = []
        
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        return unique_labels
    
    def get_label_statistics(self, all_labels: List[str]) -> Dict[str, Any]:
        """Get statistics about label usage.
        
        Args:
            all_labels: List of all labels from the dataset
            
        Returns:
            Dictionary with label statistics
        """
        label_counts = Counter(all_labels)
        
        return {
            'total_labels': len(all_labels),
            'unique_labels': len(label_counts),
            'most_common': label_counts.most_common(10),
            'avg_frequency': len(all_labels) / len(label_counts) if label_counts else 0,
            'label_distribution': dict(label_counts)
        }


class TextProcessor:
    """Process and clean text fields from papers."""
    
    def __init__(self, 
                 max_length: Optional[int] = None,
                 clean_html: bool = True,
                 normalize_whitespace: bool = True):
        """Initialize the text processor.
        
        Args:
            max_length: Maximum length for text fields (None = no limit)
            clean_html: Whether to remove HTML tags
            normalize_whitespace: Whether to normalize whitespace
        """
        self.max_length = max_length
        self.clean_html = clean_html
        self.normalize_whitespace = normalize_whitespace
        
    def process_text(self, text: str) -> str:
        """Process a text field with cleaning and normalization.
        
        Args:
            text: Raw text string
            
        Returns:
            Processed text
        """
        if not text or not text.strip():
            return ""
        
        processed = text.strip()
        
        # Remove HTML tags
        if self.clean_html:
            processed = re.sub(r'<[^>]+>', '', processed)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Truncate if needed
        if self.max_length and len(processed) > self.max_length:
            processed = processed[:self.max_length].strip()
        
        return processed
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract potential keywords from text.
        
        Args:
            text: Input text
            min_length: Minimum keyword length
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Simple keyword extraction - split on whitespace and punctuation
        words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_length, text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        return keywords


class PaperDataProcessor:
    """High-level processor for paper datasets."""
    
    def __init__(self, 
                 label_processor: Optional[LabelProcessor] = None,
                 text_processor: Optional[TextProcessor] = None):
        """Initialize the paper data processor.
        
        Args:
            label_processor: Label processor instance (creates default if None)
            text_processor: Text processor instance (creates default if None)
        """
        self.label_processor = label_processor or LabelProcessor()
        self.text_processor = text_processor or TextProcessor()
    
    def process_paper_batch(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of papers with standardization and cleaning.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of processed paper dictionaries
        """
        processed_papers = []
        
        for paper in papers:
            try:
                processed_paper = self.process_single_paper(paper)
                if processed_paper:
                    processed_papers.append(processed_paper)
            except Exception as e:
                # Log error but continue processing
                print(f"Warning: Failed to process paper {paper.get('id', 'unknown')}: {e}")
                continue
        
        return processed_papers
    
    def process_single_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single paper with all cleaning steps.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Processed paper dictionary or None if invalid
        """
        if not isinstance(paper, dict):
            return None
        
        processed_paper = paper.copy()
        
        # Process labels
        labels = paper.get('labels', [])
        if labels:
            processed_labels = self.label_processor.process_labels(labels)
            processed_labels = self.label_processor.deduplicate_labels(processed_labels)
            processed_paper['labels'] = processed_labels
        
        # Process text fields
        text_fields = ['title', 'abstract', 'journal', 'authors']
        for field in text_fields:
            if field in paper:
                processed_paper[field] = self.text_processor.process_text(str(paper[field]))
        
        # Validate that paper has required fields
        if not processed_paper.get('labels'):
            return None  # Skip papers with no valid labels
        
        return processed_paper
    
    def validate_paper_structure(self, paper: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that a paper has the required structure.
        
        Args:
            paper: Paper dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if not isinstance(paper, dict):
            errors.append("Paper must be a dictionary")
            return False, errors
        
        if 'labels' not in paper:
            errors.append("Paper missing required 'labels' field")
        elif not isinstance(paper['labels'], list):
            errors.append("Paper 'labels' field must be a list")
        elif not paper['labels']:
            errors.append("Paper must have at least one label")
        
        # Check optional but expected fields
        expected_fields = ['id', 'title', 'abstract']
        for field in expected_fields:
            if field not in paper:
                errors.append(f"Paper missing recommended field: {field}")
        
        return len(errors) == 0, errors
    
    def get_processing_statistics(self, 
                                  original_papers: List[Dict[str, Any]], 
                                  processed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the processing results.
        
        Args:
            original_papers: Original paper list
            processed_papers: Processed paper list
            
        Returns:
            Dictionary with processing statistics
        """
        original_count = len(original_papers)
        processed_count = len(processed_papers)
        
        # Count labels
        original_labels = []
        processed_labels = []
        
        for paper in original_papers:
            original_labels.extend(paper.get('labels', []))
        
        for paper in processed_papers:
            processed_labels.extend(paper.get('labels', []))
        
        return {
            'original_paper_count': original_count,
            'processed_paper_count': processed_count,
            'papers_filtered_out': original_count - processed_count,
            'retention_rate': processed_count / original_count if original_count > 0 else 0,
            'original_label_count': len(original_labels),
            'processed_label_count': len(processed_labels),
            'unique_original_labels': len(set(original_labels)),
            'unique_processed_labels': len(set(processed_labels)),
            'label_reduction_rate': (len(set(original_labels)) - len(set(processed_labels))) / len(set(original_labels)) if original_labels else 0
        }


class DataValidator:
    """Validate paper datasets for consistency and quality."""
    
    def __init__(self, 
                 min_papers: int = 1,
                 min_labels_per_paper: int = 1,
                 min_unique_labels: int = 1):
        """Initialize the data validator.
        
        Args:
            min_papers: Minimum number of papers required
            min_labels_per_paper: Minimum labels per paper
            min_unique_labels: Minimum unique labels in dataset
        """
        self.min_papers = min_papers
        self.min_labels_per_paper = min_labels_per_paper
        self.min_unique_labels = min_unique_labels
    
    def validate_dataset(self, papers: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate an entire dataset.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check minimum paper count
        if len(papers) < self.min_papers:
            errors.append(f"Dataset must have at least {self.min_papers} papers, got {len(papers)}")
        
        # Check paper structure
        for i, paper in enumerate(papers):
            is_valid, paper_errors = self._validate_paper(paper)
            if not is_valid:
                errors.extend([f"Paper {i}: {error}" for error in paper_errors])
        
        # Check label diversity
        all_labels = []
        for paper in papers:
            all_labels.extend(paper.get('labels', []))
        
        unique_labels = len(set(all_labels))
        if unique_labels < self.min_unique_labels:
            errors.append(f"Dataset must have at least {self.min_unique_labels} unique labels, got {unique_labels}")
        
        return len(errors) == 0, errors
    
    def _validate_paper(self, paper: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(paper, dict):
            errors.append("Paper must be a dictionary")
            return False, errors
        
        # Check labels
        labels = paper.get('labels', [])
        if not isinstance(labels, list):
            errors.append("Labels must be a list")
        elif len(labels) < self.min_labels_per_paper:
            errors.append(f"Paper must have at least {self.min_labels_per_paper} labels")
        
        # Check for empty labels
        if labels and any(not label or not label.strip() for label in labels):
            errors.append("Paper contains empty or whitespace-only labels")
        
        return len(errors) == 0, errors
    
    def get_quality_metrics(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get quality metrics for a dataset.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary with quality metrics
        """
        if not papers:
            return {'error': 'No papers provided'}
        
        # Basic counts
        total_papers = len(papers)
        total_labels = sum(len(paper.get('labels', [])) for paper in papers)
        unique_labels = len(set(label for paper in papers for label in paper.get('labels', [])))
        
        # Label distribution
        label_counts = Counter(label for paper in papers for label in paper.get('labels', []))
        
        # Papers with text fields
        papers_with_title = sum(1 for paper in papers if paper.get('title', '').strip())
        papers_with_abstract = sum(1 for paper in papers if paper.get('abstract', '').strip())
        
        return {
            'total_papers': total_papers,
            'total_labels': total_labels,
            'unique_labels': unique_labels,
            'avg_labels_per_paper': total_labels / total_papers if total_papers > 0 else 0,
            'papers_with_title': papers_with_title,
            'papers_with_abstract': papers_with_abstract,
            'title_coverage': papers_with_title / total_papers if total_papers > 0 else 0,
            'abstract_coverage': papers_with_abstract / total_papers if total_papers > 0 else 0,
            'most_common_labels': label_counts.most_common(10),
            'label_frequency_distribution': dict(label_counts)
        }