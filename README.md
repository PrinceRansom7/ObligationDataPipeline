# Obligation Data Pipeline

Production-style starter pipeline for generating obligation-extraction training data from regulatory PDFs.

## What It Does

1. Ingests PDFs from `data/input_pdfs/`.
2. Extracts text and chunks into clause-like spans.
3. Uses OpenAI structured output (or rule fallback) to classify and extract JSON records.
4. Validates outputs against strict schema.
5. Balances dataset by target ratios.
6. Writes versioned JSONL outputs with logs and reject files.
7. Ingests valid records to Chroma (vector) and Neo4j (graph).

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
python run_pipeline.py --version v1.0.0
python run_pipeline.py --version v1.0.0 --skip-graph
```

## Stage-wise Runners

```bash
python run_ingestion.py --version v1.0.0
python run_extract.py --version v1.0.0
python run_ingest_vector.py --version v1.0.0
python run_ingest_graph.py --version v1.0.0
```

## Main Outputs

- `data/output/raw_<version>.jsonl` - full generated records
- `data/output/balanced_<version>.jsonl` - class-balanced records
- `data/output/rejects_<version>.jsonl` - failed validation or extraction
- `data/output/run_report_<version>.json` - run metrics
- `data/chroma/` - local Chroma collection (if enabled)

## Notes

- Set `USE_OPENAI=true` and `OPENAI_API_KEY` to enable structured extraction.
- Vector ingest uses `EMBEDDING_PROVIDER` (`sentence_transformers` or `openai`).
- Graph ingest uses Neo4j credentials from `.env`. If unavailable, run with `--skip-graph`.
