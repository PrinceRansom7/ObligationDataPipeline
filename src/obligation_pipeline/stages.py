"""Stage-wise pipeline tasks for ingestion, extraction, and DB ingest."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .balance import balance_records
from .chunking import chunk_page_text
from .config import Settings
from .extraction import ExtractInput, extract_record
from .pdf_ingest import list_pdf_files, load_pdf_text
from .schema import ObligationRecord
from .storage_graph import ingest_records_to_neo4j
from .storage_vector import ingest_records_to_chroma
from .validate import validate_record

logger = logging.getLogger(__name__)


def _write_jsonl(path: Path, rows: list[ObligationRecord | dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, ObligationRecord):
                f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_ingestion_stage(input_dir: str, output_dir: str, dataset_version: str) -> tuple[list[ExtractInput], Path]:
    """Parse PDFs and create chunk manifest for downstream extraction."""
    load_dotenv()
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list_pdf_files(in_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {in_dir}")

    extract_inputs: list[ExtractInput] = []
    for pdf in pdfs:
        document_title = pdf.stem
        document_id = pdf.stem.lower().replace(" ", "_")
        pages = load_pdf_text(pdf)
        for page_number, page_text in pages:
            for ch in chunk_page_text(document_id, page_number, page_text):
                extract_inputs.append(
                    ExtractInput(
                        dataset_version=dataset_version,
                        document_id=document_id,
                        document_title=document_title,
                        chunk_id=ch.chunk_id,
                        chunk_text=ch.text,
                        page_number=ch.page_number,
                        char_start=ch.char_start,
                        char_end=ch.char_end,
                    )
                )
    manifest_path = out_dir / f"chunk_manifest_{dataset_version}.jsonl"
    _write_jsonl(manifest_path, [x.__dict__ for x in extract_inputs])
    logger.info("Ingestion stage wrote %s (%d chunks)", manifest_path, len(extract_inputs))
    return extract_inputs, manifest_path


async def _extract_with_limit(items: list[ExtractInput], settings: Settings) -> tuple[list[ObligationRecord], list[dict]]:
    sem = asyncio.Semaphore(settings.max_concurrency)
    client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.use_openai and settings.openai_api_key else None
    records: list[ObligationRecord] = []
    rejects: list[dict] = []

    async def _one(item: ExtractInput) -> None:
        async with sem:
            try:
                rec = await extract_record(item, settings, client=client)
                ok, err = validate_record(rec)
                if ok:
                    rec.metadata.traceability.validation_status = "passed"
                    records.append(rec)
                else:
                    rec.metadata.traceability.validation_status = "failed"
                    rejects.append({"chunk_id": item.chunk_id, "error": err, "record": rec.model_dump()})
            except Exception as exc:  # noqa: BLE001
                rejects.append({"chunk_id": item.chunk_id, "error": str(exc)})

    await asyncio.gather(*[_one(i) for i in items])
    return records, rejects


async def run_extraction_stage(output_dir: str, dataset_version: str) -> tuple[list[ObligationRecord], Path]:
    """Run obligation extraction from chunk manifest and write dataset artifacts."""
    load_dotenv()
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    out_dir = Path(output_dir)

    manifest_path = out_dir / f"chunk_manifest_{dataset_version}.jsonl"
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise FileNotFoundError(f"Chunk manifest not found or empty: {manifest_path}")
    items = [ExtractInput(**r) for r in rows]
    records, rejects = await _extract_with_limit(items, settings)
    balanced = balance_records(
        records,
        settings.target_obligation_ratio,
        settings.target_non_obligation_ratio,
        settings.target_neutral_ratio,
    )

    raw_path = out_dir / f"raw_{dataset_version}.jsonl"
    balanced_path = out_dir / f"balanced_{dataset_version}.jsonl"
    rejects_path = out_dir / f"rejects_{dataset_version}.jsonl"
    _write_jsonl(raw_path, records)
    _write_jsonl(balanced_path, balanced)
    _write_jsonl(rejects_path, rejects)
    logger.info("Extraction stage wrote raw=%s balanced=%s rejects=%s", raw_path, balanced_path, rejects_path)
    return records, raw_path


def run_vector_ingest_stage(output_dir: str, dataset_version: str) -> int:
    """Ingest extracted records into Chroma."""
    load_dotenv()
    settings = Settings()
    out_dir = Path(output_dir)
    rows = _read_jsonl(out_dir / f"raw_{dataset_version}.jsonl")
    records = [ObligationRecord.model_validate(r) for r in rows]
    return ingest_records_to_chroma(records, settings)


def run_graph_ingest_stage(output_dir: str, dataset_version: str) -> int:
    """Ingest extracted records into Neo4j."""
    load_dotenv()
    settings = Settings()
    out_dir = Path(output_dir)
    rows = _read_jsonl(out_dir / f"raw_{dataset_version}.jsonl")
    records = [ObligationRecord.model_validate(r) for r in rows]
    return ingest_records_to_neo4j(records, settings)
