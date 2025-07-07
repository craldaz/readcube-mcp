"""Data loading and preprocessing utilities for Query2Label system.

This module contains the PaperDataLoader class that handles loading paper datasets
from various file formats (CSV, JSON) and converting them into the standardized
paper format used by the Query2Label system.
"""

import pandas as pd
import json
from typing import List, Dict, Set, Any, Optional, Union, Tuple
from pathlib import Path
from collections import Counter
from ..core.exceptions import DataProcessingError


class PaperDataLoader:
    """Load and preprocess paper datasets from various file formats.
    
    This class handles loading papers from CSV and JSON files, extracting labels
    with multiple delimiter support, and converting them to the standardized
    paper format used by the Query2Label system.
    """

    def __init__(self, label_delimiters: Optional[List[str]] = None):
        """Initialize the data loader with configuration.
        
        Args:
            label_delimiters: List of delimiters to use for splitting labels
                             (default: [';', ',', '|'])
        """
        self.label_delimiters = label_delimiters or [';', ',', '|']
        self.papers_db = []
        self.label_counts = {}
        self.available_labels = []
        self.data_statistics = {}

    def load_csv(
        self, 
        file_path: Union[str, Path], 
        column_mappings: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Load papers from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            column_mappings: Optional mapping of CSV columns to paper fields
                           Default mapping:
                           {'Tags': 'labels', 'Title': 'title', 'Abstract': 'abstract',
                            'Library URL': 'library_url', 'year': 'year', 
                            'Journal': 'journal', 'Author': 'authors'}
        
        Returns:
            List of paper dictionaries
            
        Raises:
            DataProcessingError: If CSV loading or processing fails
        """
        try:
            # Default column mappings based on Acelot Library CSV format
            default_mappings = {
                'Tags': 'labels',
                'Title': 'title',
                'Abstract': 'abstract',
                'Library URL': 'library_url',
                'year': 'year',
                'Journal': 'journal',
                'Author': 'authors'
            }
            
            if column_mappings:
                default_mappings.update(column_mappings)
            
            mappings = default_mappings
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Extract all tags first for counting
            all_tags = []
            for idx, row in df.iterrows():
                tags = row.get('Tags', '')
                if pd.notna(tags) and tags.strip():
                    tag_list = self._parse_labels(tags)
                    all_tags.extend(tag_list)
            
            # Calculate label statistics
            self.label_counts = dict(Counter(all_tags))
            self.available_labels = list(self.label_counts.keys())
            
            # Convert each row to paper dictionary
            papers_db = []
            for idx, row in df.iterrows():
                paper = self._convert_row_to_paper(row, idx, mappings)
                papers_db.append(paper)
            
            self.papers_db = papers_db
            self._calculate_statistics()
            
            return papers_db
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to load CSV file {file_path}: {str(e)}",
                {"file_path": str(file_path), "error_type": type(e).__name__}
            )

    def load_json(
        self, 
        file_path: Union[str, Path], 
        papers_key: str = "papers"
    ) -> List[Dict[str, Any]]:
        """Load papers from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            papers_key: Key in JSON that contains the papers array
        
        Returns:
            List of paper dictionaries
            
        Raises:
            DataProcessingError: If JSON loading or processing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract papers from JSON
            if isinstance(data, list):
                papers_data = data
            elif isinstance(data, dict) and papers_key in data:
                papers_data = data[papers_key]
            else:
                raise ValueError(f"JSON must be a list or dict with '{papers_key}' key")
            
            # Process each paper
            papers_db = []
            all_tags = []
            
            for idx, paper_data in enumerate(papers_data):
                paper = self._convert_json_to_paper(paper_data, idx)
                papers_db.append(paper)
                all_tags.extend(paper.get('labels', []))
            
            # Calculate label statistics
            self.label_counts = dict(Counter(all_tags))
            self.available_labels = list(self.label_counts.keys())
            
            self.papers_db = papers_db
            self._calculate_statistics()
            
            return papers_db
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to load JSON file {file_path}: {str(e)}",
                {"file_path": str(file_path), "error_type": type(e).__name__}
            )

    def load_papers_list(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Load papers from a pre-constructed list of dictionaries.
        
        Args:
            papers: List of paper dictionaries
        
        Returns:
            List of processed paper dictionaries
        """
        try:
            # Process each paper to ensure consistent format
            papers_db = []
            all_tags = []
            
            for idx, paper_data in enumerate(papers):
                paper = self._convert_json_to_paper(paper_data, idx)
                papers_db.append(paper)
                all_tags.extend(paper.get('labels', []))
            
            # Calculate label statistics
            self.label_counts = dict(Counter(all_tags))
            self.available_labels = list(self.label_counts.keys())
            
            self.papers_db = papers_db
            self._calculate_statistics()
            
            return papers_db
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to process papers list: {str(e)}",
                {"error_type": type(e).__name__}
            )

    def _parse_labels(self, labels_str: str) -> List[str]:
        """Parse labels from a string using multiple delimiters.
        
        Args:
            labels_str: String containing labels separated by delimiters
        
        Returns:
            List of cleaned label strings
        """
        if not labels_str or not labels_str.strip():
            return []
        
        # Try each delimiter in order
        for delimiter in self.label_delimiters:
            if delimiter in labels_str:
                labels = [label.strip() for label in labels_str.split(delimiter)]
                return [label for label in labels if label]
        
        # No delimiter found, treat as single label
        return [labels_str.strip()]

    def _convert_row_to_paper(
        self, 
        row: pd.Series, 
        idx: int, 
        mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """Convert a pandas Series row to a paper dictionary.
        
        Args:
            row: Pandas Series representing a row from the CSV
            idx: Row index to use as paper ID
            mappings: Column name mappings
        
        Returns:
            Paper dictionary
        """
        # Extract and parse labels
        tags = row.get('Tags', '')
        paper_labels = []
        if pd.notna(tags) and tags.strip():
            paper_labels = self._parse_labels(tags)
        
        paper = {
            'id': idx,
            'title': str(row.get('Title', 'No Title')),
            'abstract': str(row.get('Abstract', 'No Abstract')),
            'labels': paper_labels,
            'library_url': str(row.get('Library URL', '')),
            'year': str(row.get('year', '')),
            'journal': str(row.get('Journal', '')),
            'authors': str(row.get('Author', ''))
        }
        
        # Handle potential NaN values
        for key, value in paper.items():
            if key != 'labels' and (pd.isna(value) or str(value) == 'nan'):
                paper[key] = ''
        
        return paper

    def _convert_json_to_paper(
        self, 
        paper_data: Dict[str, Any], 
        idx: int
    ) -> Dict[str, Any]:
        """Convert a JSON paper object to standardized paper dictionary.
        
        Args:
            paper_data: Dictionary containing paper data
            idx: Index to use as paper ID if not provided
        
        Returns:
            Paper dictionary
        """
        # Handle labels field
        labels = paper_data.get('labels', [])
        if isinstance(labels, str):
            labels = self._parse_labels(labels)
        elif not isinstance(labels, list):
            labels = []
        
        paper = {
            'id': paper_data.get('id', idx),
            'title': str(paper_data.get('title', 'No Title')),
            'abstract': str(paper_data.get('abstract', 'No Abstract')),
            'labels': labels,
            'library_url': str(paper_data.get('library_url', '')),
            'year': str(paper_data.get('year', '')),
            'journal': str(paper_data.get('journal', '')),
            'authors': str(paper_data.get('authors', ''))
        }
        
        return paper

    def _calculate_statistics(self) -> None:
        """Calculate and store dataset statistics."""
        total_papers = len(self.papers_db)
        unique_labels = len(self.available_labels)
        total_label_instances = sum(len(paper.get('labels', [])) for paper in self.papers_db)
        avg_labels_per_paper = total_label_instances / total_papers if total_papers > 0 else 0
        
        # Get top labels by frequency
        top_labels = sorted(
            self.label_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        self.data_statistics = {
            'total_papers': total_papers,
            'unique_labels': unique_labels,
            'total_label_instances': total_label_instances,
            'avg_labels_per_paper': avg_labels_per_paper,
            'top_labels': top_labels
        }

    def get_papers_db(self) -> List[Dict[str, Any]]:
        """Get the loaded papers database.
        
        Returns:
            List of paper dictionaries
        """
        return self.papers_db

    def get_available_labels(self) -> List[str]:
        """Get list of all available labels.
        
        Returns:
            List of unique labels found in the dataset
        """
        return self.available_labels

    def get_label_counts(self) -> Dict[str, int]:
        """Get label usage counts.
        
        Returns:
            Dictionary mapping label names to their usage counts
        """
        return self.label_counts

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return self.data_statistics

    def filter_papers_by_label_frequency(
        self, 
        min_frequency: int = 1
    ) -> List[Dict[str, Any]]:
        """Filter papers to only include those with labels above minimum frequency.
        
        Args:
            min_frequency: Minimum label frequency to include
        
        Returns:
            Filtered list of papers
        """
        valid_labels = {
            label for label, count in self.label_counts.items() 
            if count >= min_frequency
        }
        
        filtered_papers = []
        for paper in self.papers_db:
            # Filter labels for this paper
            filtered_labels = [
                label for label in paper.get('labels', [])
                if label in valid_labels
            ]
            
            # Only include paper if it has at least one valid label
            if filtered_labels:
                filtered_paper = paper.copy()
                filtered_paper['labels'] = filtered_labels
                filtered_papers.append(filtered_paper)
        
        return filtered_papers

    def export_to_json(self, file_path: Union[str, Path]) -> None:
        """Export the loaded papers to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        try:
            export_data = {
                'papers': self.papers_db,
                'statistics': self.data_statistics,
                'label_counts': self.label_counts
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise DataProcessingError(
                f"Failed to export to JSON file {file_path}: {str(e)}",
                {"file_path": str(file_path), "error_type": type(e).__name__}
            )

    def get_sample_papers(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get a sample of papers for testing or preview.
        
        Args:
            n: Number of papers to sample
        
        Returns:
            List of sample papers
        """
        return self.papers_db[:n] if self.papers_db else []

    def load_csv_with_counts(
        self, 
        file_path: Union[str, Path], 
        column_mappings: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Load papers from CSV file and return both papers and label counts.
        
        This method is specifically designed for fusion processing where both
        the paper data and label usage statistics are needed.
        
        Args:
            file_path: Path to CSV file
            column_mappings: Optional mapping of CSV columns to paper fields
            
        Returns:
            Tuple of (papers, label_counts) where:
            - papers: List of paper dictionaries
            - label_counts: Dictionary mapping labels to occurrence counts
            
        Raises:
            DataProcessingError: If CSV loading fails
        """
        try:
            # Load papers using existing method
            papers = self.load_csv(file_path, column_mappings)
            
            # Get label counts (calculated during load_csv)
            label_counts = self.get_label_counts()
            
            return papers, label_counts
            
        except Exception as e:
            raise DataProcessingError(
                f"Failed to load CSV with counts from {file_path}: {str(e)}",
                {"file_path": str(file_path), "error_type": type(e).__name__}
            )