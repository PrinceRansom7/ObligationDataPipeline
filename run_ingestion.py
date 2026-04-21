#!/usr/bin/env python3
"""Ingestion CLI: PDF parse + chunk manifest generation."""

from __future__ import annotations

import argparse

from src.obligation_pipeline.stages import run_ingestion_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ingestion stage for obligation pipeline.")
    parser.add_argument("--input-dir", type=str, default="data/input_pdfs")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()
    run_ingestion_stage(args.input_dir, args.output_dir, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
