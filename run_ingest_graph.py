#!/usr/bin/env python3
"""Graph ingest CLI: load raw records into Neo4j."""

from __future__ import annotations

import argparse

from src.obligation_pipeline.stages import run_graph_ingest_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest obligation records to graph DB.")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()
    count = run_graph_ingest_stage(args.output_dir, args.version)
    print(f"Graph ingested records: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
