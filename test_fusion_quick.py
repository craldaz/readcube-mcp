#!/usr/bin/env python3
"""
Quick test to validate Manual Fusion Processor setup.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from readcube_mcp.query2label.data.loaders import PaperDataLoader
from readcube_mcp.query2label.fusion.processors import ManualFusionProcessor
from readcube_mcp.query2label.core.types import FusionConfig

def test_system_setup():
    """Test that all components can be initialized."""
    print("🧪 Quick System Test")
    print("=" * 25)
    
    # Test 1: Data loading
    print("1. Testing data loading...")
    data_path = Path(__file__).parent / "data" / "Acelot Library.csv"
    if not data_path.exists():
        print(f"❌ Data file not found at {data_path}")
        return False
    
    loader = PaperDataLoader()
    papers, label_counts = loader.load_csv_with_counts(str(data_path))
    print(f"✅ Loaded {len(papers)} papers with {len(label_counts)} unique labels")
    
    # Test 2: Show some sample labels and papers
    print("\n2. Sample data:")
    print(f"   Top 10 labels: {list(label_counts.keys())[:10]}")
    if papers:
        sample_paper = papers[0]
        print(f"   Sample paper: {sample_paper.get('title', 'Unknown')[:60]}...")
        print(f"   Sample labels: {sample_paper.get('labels', [])[:5]}")
    
    # Test 3: Fusion config
    print("\n3. Testing fusion configuration...")
    config = FusionConfig(k=60, max_results=10)
    print(f"✅ Fusion config: {config}")
    
    # Test 4: Manual decomposition example
    print("\n4. Example query decomposition:")
    complex_query = "docking calculations OR protein folding OR protein-ligand binding of IDPs"
    sub_queries = [
        "docking calculations intrinsically disordered proteins",
        "protein folding of intrinsically disordered proteins", 
        "protein-ligand binding interactions"
    ]
    
    print(f"Original: {complex_query}")
    print("Decomposed into:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    print("\n✅ All components initialized successfully!")
    print("\n💡 To run full fusion demo with LLM:")
    print("   export OPENAI_API_KEY='your-key'")
    print("   python manual_fusion_example.py")
    
    return True

if __name__ == "__main__":
    success = test_system_setup()
    if success:
        print("\n🎉 System ready for manual fusion!")
    else:
        print("\n❌ System setup failed")
        sys.exit(1)