#!/usr/bin/env python3
"""Vector ingest CLI: load raw records into Chroma."""

from __future__ import annotations

import argparse

from src.obligation_pipeline.stages import run_vector_ingest_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest obligation records to vector DB.")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()
    count = run_vector_ingest_stage(args.output_dir, args.version)
    print(f"Vector ingested records: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
