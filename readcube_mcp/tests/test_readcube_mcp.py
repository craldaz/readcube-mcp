"""
Unit and regression test for the readcube_mcp package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import readcube_mcp


def test_readcube_mcp_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "readcube_mcp" in sys.modules
