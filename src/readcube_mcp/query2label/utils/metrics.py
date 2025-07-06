"""Performance monitoring and metrics collection for Query2Label system.

This module provides utilities for monitoring system performance, collecting
metrics, and generating reports on system usage and efficiency.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, TypeVar
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import functools
from .logging_config import get_logger

T = TypeVar('T')


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class PerformanceStats:
    """Performance statistics for a metric."""
    count: int
    total: float
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'count': self.count,
            'total': self.total,
            'min': self.min_value,
            'max': self.max_value,
            'avg': self.avg_value,
            'median': self.median_value,
            'p95': self.p95_value,
            'p99': self.p99_value
        }


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_points_per_metric: Maximum number of data points to keep per metric
        """
        self.max_points = max_points_per_metric
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_points))
        self._lock = threading.RLock()
        self._logger = get_logger("query2label.metrics")
    
    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
            self._metrics[metric_name].append(point)
            
            self._logger.debug(
                f"Recorded metric: {metric_name}={value}",
                extra={'metric_name': metric_name, 'value': value, 'labels': labels}
            )
    
    def increment(self, metric_name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            amount: Amount to increment by
            labels: Optional labels for the metric
        """
        self.record(metric_name, amount, labels)
    
    def gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (current value).
        
        Args:
            metric_name: Name of the metric
            value: Current value
            labels: Optional labels for the metric
        """
        self.record(metric_name, value, labels)
    
    def timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.
        
        Args:
            metric_name: Name of the timing metric
            labels: Optional labels for the metric
        
        Returns:
            Timer context manager
        """
        return TimerContext(self, metric_name, labels)
    
    def get_metric(self, metric_name: str, since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metric data points.
        
        Args:
            metric_name: Name of the metric
            since: Optional timestamp to filter from
            
        Returns:
            List of metric points
        """
        with self._lock:
            points = list(self._metrics.get(metric_name, []))
            
            if since:
                points = [p for p in points if p.timestamp >= since]
            
            return points
    
    def get_stats(self, metric_name: str, since: Optional[datetime] = None) -> Optional[PerformanceStats]:
        """Get performance statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            since: Optional timestamp to filter from
            
        Returns:
            Performance statistics or None if no data
        """
        points = self.get_metric(metric_name, since)
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        return PerformanceStats(
            count=len(values),
            total=sum(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            p95_value=self._percentile(values, 0.95),
            p99_value=self._percentile(values, 0.99)
        )
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all metric names.
        
        Returns:
            List of metric names
        """
        with self._lock:
            return list(self._metrics.keys())
    
    def clear_metric(self, metric_name: str) -> None:
        """Clear all data for a metric.
        
        Args:
            metric_name: Name of the metric to clear
        """
        with self._lock:
            if metric_name in self._metrics:
                self._metrics[metric_name].clear()
    
    def clear_all(self) -> None:
        """Clear all metrics data."""
        with self._lock:
            self._metrics.clear()
    
    def export_metrics(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Export all metrics data.
        
        Args:
            since: Optional timestamp to filter from
            
        Returns:
            Dictionary containing all metrics data
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {}
            }
            
            for metric_name in self._metrics:
                points = self.get_metric(metric_name, since)
                stats = self.get_stats(metric_name, since)
                
                export_data['metrics'][metric_name] = {
                    'points': [p.to_dict() for p in points],
                    'stats': stats.to_dict() if stats else None
                }
            
            return export_data
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile value.
        
        Args:
            values: List of values
            percentile: Percentile (0.0 to 1.0)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Initialize timer context.
        
        Args:
            collector: Metrics collector instance
            metric_name: Name of the timing metric
            labels: Optional labels for the metric
        """
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metric."""
        if self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time
            self.collector.record(self.metric_name, self.duration, self.labels)


def timing_metric(metric_name: str, collector: Optional[MetricsCollector] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution and recording metrics.
    
    Args:
        metric_name: Name of the timing metric
        collector: Optional metrics collector (uses global if None)
        labels: Optional labels for the metric
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            metrics_collector = collector or global_metrics
            
            with metrics_collector.timer(metric_name, labels):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def counter_metric(metric_name: str, collector: Optional[MetricsCollector] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator for counting function calls.
    
    Args:
        metric_name: Name of the counter metric
        collector: Optional metrics collector (uses global if None)
        labels: Optional labels for the metric
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            metrics_collector = collector or global_metrics
            metrics_collector.increment(metric_name, labels=labels)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class SystemMetrics:
    """System-level metrics collection."""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        """Initialize system metrics.
        
        Args:
            collector: Optional metrics collector (uses global if None)
        """
        self.collector = collector or global_metrics
        self._logger = get_logger("query2label.system_metrics")
    
    def record_query_translation(self, query: str, result_type: str, duration: float, success: bool = True) -> None:
        """Record query translation metrics.
        
        Args:
            query: Original query string
            result_type: Type of result (e.g., 'BooleanQuery')
            duration: Processing duration in seconds
            success: Whether translation was successful
        """
        labels = {
            'result_type': result_type,
            'success': str(success).lower()
        }
        
        self.collector.record('query_translation_duration', duration, labels)
        self.collector.increment('query_translation_count', labels=labels)
        self.collector.gauge('query_length', len(query), {'result_type': result_type})
    
    def record_data_processing(self, operation: str, input_count: int, output_count: int, duration: float) -> None:
        """Record data processing metrics.
        
        Args:
            operation: Type of data processing operation
            input_count: Number of input items
            output_count: Number of output items
            duration: Processing duration in seconds
        """
        labels = {'operation': operation}
        
        self.collector.record('data_processing_duration', duration, labels)
        self.collector.gauge('data_processing_input_count', input_count, labels)
        self.collector.gauge('data_processing_output_count', output_count, labels)
        
        if input_count > 0:
            retention_rate = output_count / input_count
            self.collector.gauge('data_processing_retention_rate', retention_rate, labels)
    
    def record_paper_filtering(self, query_type: str, total_papers: int, filtered_papers: int, duration: float) -> None:
        """Record paper filtering metrics.
        
        Args:
            query_type: Type of query (e.g., 'simple', 'boolean')
            total_papers: Total number of papers in database
            filtered_papers: Number of papers after filtering
            duration: Filtering duration in seconds
        """
        labels = {'query_type': query_type}
        
        self.collector.record('paper_filtering_duration', duration, labels)
        self.collector.gauge('paper_filtering_total_papers', total_papers, labels)
        self.collector.gauge('paper_filtering_result_count', filtered_papers, labels)
        
        if total_papers > 0:
            selectivity = filtered_papers / total_papers
            self.collector.gauge('paper_filtering_selectivity', selectivity, labels)
    
    def record_label_validation(self, total_labels: int, valid_labels: int, duration: float) -> None:
        """Record label validation metrics.
        
        Args:
            total_labels: Total number of labels processed
            valid_labels: Number of valid labels
            duration: Validation duration in seconds
        """
        self.collector.record('label_validation_duration', duration)
        self.collector.gauge('label_validation_total_count', total_labels)
        self.collector.gauge('label_validation_valid_count', valid_labels)
        
        if total_labels > 0:
            validation_rate = valid_labels / total_labels
            self.collector.gauge('label_validation_success_rate', validation_rate)
    
    def record_cache_stats(self, cache_name: str, hits: int, misses: int, size: int) -> None:
        """Record cache performance metrics.
        
        Args:
            cache_name: Name of the cache
            hits: Number of cache hits
            misses: Number of cache misses
            size: Current cache size
        """
        labels = {'cache_name': cache_name}
        
        self.collector.gauge('cache_hits', hits, labels)
        self.collector.gauge('cache_misses', misses, labels)
        self.collector.gauge('cache_size', size, labels)
        
        total_requests = hits + misses
        if total_requests > 0:
            hit_rate = hits / total_requests
            self.collector.gauge('cache_hit_rate', hit_rate, labels)


class PerformanceMonitor:
    """High-level performance monitoring interface."""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        """Initialize performance monitor.
        
        Args:
            collector: Optional metrics collector (uses global if None)
        """
        self.collector = collector or global_metrics
        self.system_metrics = SystemMetrics(self.collector)
        self._logger = get_logger("query2label.performance")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self._logger.info("Performance monitoring started")
        self.collector.record('monitoring_started', 1.0)
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._logger.info("Performance monitoring stopped")
        self.collector.record('monitoring_stopped', 1.0)
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance report for the last N hours.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Performance report dictionary
        """
        since = datetime.now() - timedelta(hours=hours)
        
        report = {
            'period': f"Last {hours} hours",
            'start_time': since.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Get all metrics
        for metric_name in self.collector.get_all_metrics():
            stats = self.collector.get_stats(metric_name, since)
            if stats and stats.count > 0:
                report['metrics'][metric_name] = stats.to_dict()
        
        return report
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check.
        
        Returns:
            Health check results
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check recent query processing
        since = datetime.now() - timedelta(minutes=5)
        query_stats = self.collector.get_stats('query_translation_duration', since)
        
        if query_stats:
            health['checks']['query_processing'] = {
                'status': 'healthy',
                'recent_queries': query_stats.count,
                'avg_duration': query_stats.avg_value
            }
            
            # Flag if average duration is too high
            if query_stats.avg_value > 10.0:  # 10 seconds
                health['checks']['query_processing']['status'] = 'warning'
                health['status'] = 'warning'
        else:
            health['checks']['query_processing'] = {
                'status': 'no_data',
                'recent_queries': 0
            }
        
        # Check cache performance
        cache_hit_rate = self.collector.get_stats('cache_hit_rate', since)
        if cache_hit_rate:
            hit_rate = cache_hit_rate.avg_value
            health['checks']['cache'] = {
                'status': 'healthy' if hit_rate > 0.5 else 'warning',
                'hit_rate': hit_rate
            }
            
            if hit_rate <= 0.3:  # Very low hit rate
                health['status'] = 'warning'
        
        return health


# Global metrics collector instance
global_metrics = MetricsCollector()

# Global system metrics instance
system_metrics = SystemMetrics(global_metrics)

# Global performance monitor instance
performance_monitor = PerformanceMonitor(global_metrics)