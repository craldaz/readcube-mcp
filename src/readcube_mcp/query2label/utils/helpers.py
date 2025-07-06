"""General utility functions and helpers for Query2Label system.

This module provides common utility functions for string processing, validation,
timing, caching, and other operations used throughout the Query2Label system.
"""

import re
import time
import hashlib
import functools
from typing import Any, Dict, List, Set, Optional, Union, Callable, TypeVar, Tuple
from pathlib import Path
import json
import csv
from collections import defaultdict, Counter
from datetime import datetime, timedelta

T = TypeVar('T')


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "operation"):
        """Initialize timer with operation name.
        
        Args:
            operation_name: Name of the operation being timed
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds.
        
        Returns:
            Elapsed time or current duration if still running
        """
        if self.duration is not None:
            return self.duration
        elif self.start_time is not None:
            return time.perf_counter() - self.start_time
        else:
            return 0.0


def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        with Timer(func.__name__) as timer:
            result = func(*args, **kwargs)
        
        # Log timing if logging is available
        try:
            from .logging_config import log_performance
            log_performance(func.__name__, timer.duration)
        except ImportError:
            pass
        
        return result
    
    return wrapper


class StringUtils:
    """String processing utilities."""
    
    @staticmethod
    def normalize_label(label: str, 
                       lowercase: bool = True, 
                       replace_spaces: bool = True,
                       remove_special: bool = True) -> str:
        """Normalize a label string.
        
        Args:
            label: Input label string
            lowercase: Convert to lowercase
            replace_spaces: Replace spaces with hyphens
            remove_special: Remove special characters
            
        Returns:
            Normalized label string
        """
        if not label:
            return ""
        
        normalized = label.strip()
        
        if lowercase:
            normalized = normalized.lower()
        
        if remove_special:
            # Keep alphanumeric, spaces, hyphens, underscores
            normalized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if replace_spaces:
            normalized = normalized.replace(' ', '-')
        
        return normalized
    
    @staticmethod
    def extract_keywords(text: str, 
                        min_length: int = 3, 
                        max_keywords: int = 50) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            min_length: Minimum keyword length
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_length, text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'they', 'them', 'their', 'there', 'where', 'when', 'what', 'which', 'who',
            'how', 'why', 'then', 'than', 'such', 'some', 'many', 'most', 'more', 'much'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency and return most common
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    @staticmethod
    def similarity_score(str1: str, str2: str) -> float:
        """Calculate simple similarity score between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 or not str2:
            return 0.0
        
        # Normalize strings
        s1 = StringUtils.normalize_label(str1)
        s2 = StringUtils.normalize_label(str2)
        
        if s1 == s2:
            return 1.0
        
        # Simple word overlap similarity
        words1 = set(s1.split('-'))
        words2 = set(s2.split('-'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Input text
            max_length: Maximum length including suffix
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix


class ValidationUtils:
    """Validation utilities."""
    
    @staticmethod
    def is_valid_label(label: str, 
                      min_length: int = 2, 
                      max_length: int = 100) -> bool:
        """Check if a label is valid.
        
        Args:
            label: Label to validate
            min_length: Minimum label length
            max_length: Maximum label length
            
        Returns:
            True if label is valid
        """
        if not label or not isinstance(label, str):
            return False
        
        normalized = StringUtils.normalize_label(label)
        return min_length <= len(normalized) <= max_length
    
    @staticmethod
    def validate_labels(labels: List[str], 
                       unique_only: bool = True) -> Tuple[List[str], List[str]]:
        """Validate a list of labels.
        
        Args:
            labels: List of labels to validate
            unique_only: Whether to require unique labels
            
        Returns:
            Tuple of (valid_labels, invalid_labels)
        """
        valid_labels = []
        invalid_labels = []
        seen = set()
        
        for label in labels:
            if ValidationUtils.is_valid_label(label):
                normalized = StringUtils.normalize_label(label)
                
                if unique_only and normalized in seen:
                    invalid_labels.append(label)
                else:
                    valid_labels.append(normalized)
                    seen.add(normalized)
            else:
                invalid_labels.append(label)
        
        return valid_labels, invalid_labels
    
    @staticmethod
    def is_valid_query(query: str, 
                      min_length: int = 3, 
                      max_length: int = 1000) -> bool:
        """Check if a query string is valid.
        
        Args:
            query: Query to validate
            min_length: Minimum query length
            max_length: Maximum query length
            
        Returns:
            True if query is valid
        """
        if not query or not isinstance(query, str):
            return False
        
        query = query.strip()
        return min_length <= len(query) <= max_length


class CacheUtils:
    """Simple caching utilities."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize cache with size and TTL limits.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        # Check TTL
        timestamp = self._timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl_seconds:
            self._remove(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest item if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = min(self._timestamps.keys(), key=self._timestamps.get)
            self._remove(oldest_key)
        
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """Remove item from cache.
        
        Args:
            key: Cache key to remove
        """
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size.
        
        Returns:
            Number of items in cache
        """
        return len(self._cache)


def memoize(ttl_seconds: int = 3600):
    """Decorator for memoizing function results.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        
    Returns:
        Decorator function
    """
    cache = CacheUtils(ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    
    return decorator


class FileUtils:
    """File and path utilities."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, creating it if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def safe_filename(filename: str, max_length: int = 255) -> str:
        """Create a safe filename by removing invalid characters.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
            
        Returns:
            Safe filename
        """
        if not filename:
            return "untitled"
        
        # Remove invalid characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove consecutive underscores
        safe = re.sub(r'_+', '_', safe)
        
        # Trim to max length
        if len(safe) > max_length:
            name, ext = Path(safe).stem, Path(safe).suffix
            max_name_length = max_length - len(ext)
            safe = name[:max_name_length] + ext
        
        return safe
    
    @staticmethod
    def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file safely.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If file cannot be read or parsed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read JSON file {file_path}: {e}")
    
    @staticmethod
    def write_json_file(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
        """Write JSON file safely.
        
        Args:
            file_path: Path to JSON file
            data: Data to write
            
        Raises:
            ValueError: If file cannot be written
        """
        try:
            FileUtils.ensure_directory(Path(file_path).parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Failed to write JSON file {file_path}: {e}")


class CollectionUtils:
    """Collection and data structure utilities."""
    
    @staticmethod
    def flatten_list(nested_list: List[List[T]]) -> List[T]:
        """Flatten a nested list.
        
        Args:
            nested_list: List of lists
            
        Returns:
            Flattened list
        """
        return [item for sublist in nested_list for item in sublist]
    
    @staticmethod
    def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
        """Split list into chunks of specified size.
        
        Args:
            items: List to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of chunks
        """
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def deduplicate_list(items: List[T], key_func: Optional[Callable[[T], Any]] = None) -> List[T]:
        """Remove duplicates from list while preserving order.
        
        Args:
            items: List to deduplicate
            key_func: Optional function to generate comparison key
            
        Returns:
            List with duplicates removed
        """
        seen = set()
        result = []
        
        for item in items:
            key = key_func(item) if key_func else item
            if key not in seen:
                seen.add(key)
                result.append(item)
        
        return result
    
    @staticmethod
    def group_by(items: List[T], key_func: Callable[[T], Any]) -> Dict[Any, List[T]]:
        """Group list items by a key function.
        
        Args:
            items: List to group
            key_func: Function to generate grouping key
            
        Returns:
            Dictionary of grouped items
        """
        groups = defaultdict(list)
        for item in items:
            key = key_func(item)
            groups[key].append(item)
        return dict(groups)


class HashUtils:
    """Hashing and checksum utilities."""
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'md5') -> str:
        """Generate hash of a string.
        
        Args:
            text: String to hash
            algorithm: Hashing algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            Hexadecimal hash string
        """
        if not text:
            return ""
        
        hash_func = getattr(hashlib, algorithm.lower())()
        hash_func.update(text.encode('utf-8'))
        return hash_func.hexdigest()
    
    @staticmethod
    def hash_object(obj: Any, algorithm: str = 'md5') -> str:
        """Generate hash of a Python object.
        
        Args:
            obj: Object to hash
            algorithm: Hashing algorithm
            
        Returns:
            Hexadecimal hash string
        """
        return HashUtils.hash_string(str(obj), algorithm)


# Global cache instance for general use
global_cache = CacheUtils(max_size=500, ttl_seconds=3600)