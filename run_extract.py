#!/usr/bin/env python3
"""Extraction CLI: chunk manifest to structured obligation records."""

from __future__ import annotations

import argparse
import asyncio

from src.obligation_pipeline.stages import run_extraction_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction stage for obligation pipeline.")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(run_extraction_stage(args.output_dir, args.version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
