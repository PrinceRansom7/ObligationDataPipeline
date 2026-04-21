"""Orchestrator that runs all pipeline stages in sequence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .stages import run_extraction_stage, run_graph_ingest_stage, run_ingestion_stage, run_vector_ingest_stage

logger = logging.getLogger(__name__)


async def run_pipeline(
    input_dir: str,
    output_dir: str,
    dataset_version: str,
    do_vector_ingest: bool = True,
    do_graph_ingest: bool = True,
) -> int:
    load_dotenv()
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"run_report_{dataset_version}.json"
    try:
        chunks, _ = run_ingestion_stage(input_dir, output_dir, dataset_version)
        records, _ = await run_extraction_stage(output_dir, dataset_version)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed before ingest stages: %s", exc)
        return 1

    vector_count = 0
    graph_count = 0
    if do_vector_ingest and settings.enable_vector_ingest:
        try:
            vector_count = run_vector_ingest_stage(output_dir, dataset_version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector ingest skipped due to error: %s", exc)
    if do_graph_ingest and settings.enable_graph_ingest:
        try:
            graph_count = run_graph_ingest_stage(output_dir, dataset_version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph ingest skipped due to error: %s", exc)

    rejects_count = 0
    rejects_path = out_dir / f"rejects_{dataset_version}.jsonl"
    if rejects_path.exists():
        with open(rejects_path, encoding="utf-8") as f:
            rejects_count = sum(1 for _ in f if _.strip())

    report = {
        "dataset_version": dataset_version,
        "pipeline_version": settings.pipeline_version,
        "chunks_total": len(chunks),
        "records_valid": len(records),
        "records_rejected": rejects_count,
        "vector_ingested": vector_count,
        "graph_ingested": graph_count,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Done. report=%s", report_path)
    return 0
