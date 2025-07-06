#!/usr/bin/env python
"""
Run the full Query-›Label-›Filter pipeline on a real dataset.

Usage:
    python demo_query2label.py \
        --csv  "data/Acelot Library.csv" \
        --query "Find papers on protein folding using machine learning" \
        --top   10
"""

from pathlib import Path
import argparse
from readcube_mcp.query2label import PaperDataLoader, PaperFilter
from readcube_mcp.query2label.dspy_modules import (
    QueryToLabelsTranslator,
    AdvancedQueryTranslator,
)
from rich import print  # nice colours

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query2Label demo on real data")
    p.add_argument("--csv", required=True, help="Path to Acelot Library CSV")
    p.add_argument("--query", required=True, help="Free-text query to run")
    p.add_argument("--top", type=int, default=10, help="How many results to show")
    p.add_argument(
        "--advanced",
        action="store_true",
        help="Use AdvancedQueryTranslator (boolean/AND/OR/NOT support)",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # ------------------------------------------------------------------
    # 1. Load papers and basic stats
    # ------------------------------------------------------------------
    loader = PaperDataLoader()
    papers = loader.load_csv(csv_path)
    stats = loader.get_statistics()
    print(
        f"[bold green]Loaded[/] {stats['total_papers']} papers "
        f"({stats['unique_labels']} unique labels, "
        f"avg {stats['avg_labels_per_paper']:.2f} labels/paper)"
    )

    # ------------------------------------------------------------------
    # 2. Build label translator (LLM-backed by default)
    #    If you don't want to hit the LLM, export:
    #    Q2L_NO_DSPY=1   (library env var)  or
    #    patch in a mock translator similar to the tests.
    # ------------------------------------------------------------------
    available_labels = loader.get_available_labels()
    label_counts = loader.get_label_counts()

    if args.advanced:
        translator = AdvancedQueryTranslator(available_labels, label_counts)
    else:
        translator = QueryToLabelsTranslator(available_labels, label_counts)

    boolean_query = translator(args.query)
    print(
        f"\n[bold cyan]BooleanQuery[/]: MUST={boolean_query.must_have}, "
        f"SHOULD={boolean_query.should_have}, NOT={boolean_query.must_not_have}"
    )

    # ------------------------------------------------------------------
    # 3. Filter + rank
    # ------------------------------------------------------------------
    paper_filter = PaperFilter(papers)
    results = paper_filter.filter_papers(boolean_query)
    print(f"Returned {len(results)} papers\n")

    # ------------------------------------------------------------------
    # 4. Pretty-print top N
    # ------------------------------------------------------------------
    for i, paper in enumerate(results[: args.top], start=1):
        score = paper.get("relevance_score", 0.0)
        labels = ", ".join(paper["labels"][:5])
        print(
            f"[{i:02d}] ({score:0.2f})  {paper['title']}\n"
            f"     {paper.get('library_url', '-')}  "
            f"[dim]{labels}[/]"
        )

    if len(results) > args.top:
        print(f"... and {len(results) - args.top} more")


if __name__ == "__main__":
    import os
    import dspy

    # 🔑 LLM backend: point DSPy to your provider (OpenAI shown here)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    turbo = dspy.LM(model="gpt-4o-mini", temperature=0, max_tokens=16384, api_key=OPENAI_API_KEY)
    dspy.settings.configure(lm=turbo)
    main()
