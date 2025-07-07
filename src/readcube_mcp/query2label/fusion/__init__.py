"""Fusion processing for Query2Label system.

This package contains processors for combining results from multiple sub-queries
using Reciprocal Rank Fusion (RRF) to improve coverage for complex queries.
"""

from .processors import ManualFusionProcessor

__all__ = [
    "ManualFusionProcessor",
]