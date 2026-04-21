#!/usr/bin/env python3
"""Orchestrator CLI: run ingestion, extraction, vector ingest, and graph ingest."""

from __future__ import annotations

import argparse
import asyncio

from src.obligation_pipeline.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate obligation extraction dataset from PDFs.")
    parser.add_argument("--input-dir", type=str, default="data/input_pdfs")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--version", type=str, required=True, help="Dataset version tag, e.g. v1.0.0")
    parser.add_argument("--skip-vector", action="store_true", help="Skip vector DB ingest stage")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph DB ingest stage")
    args = parser.parse_args()
    return asyncio.run(
        run_pipeline(
            args.input_dir,
            args.output_dir,
            args.version,
            do_vector_ingest=not args.skip_vector,
            do_graph_ingest=not args.skip_graph,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
