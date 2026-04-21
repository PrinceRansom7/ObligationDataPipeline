"""Chunk-level obligation extraction using OpenAI structured outputs."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import Settings
from .prompts import build_messages
from .schema import (
    Context,
    Evaluation,
    Linkages,
    Metadata,
    NonObligationOutput,
    ObligationOutput,
    ObligationRecord,
    Reasoning,
    ReasoningStep,
    SourceChunk,
    SourceDocument,
    TokenLimit,
    ToolUse,
    Traceability,
)


OBLIGATION_MARKERS = ("shall", "must", "required to", "is required to")
NON_OBLIGATION_MARKERS = ("may", "can", "at its discretion", "should")
NEUTRAL_MARKERS = ("means", "refers to", "defined as")


@dataclass
class ExtractInput:
    dataset_version: str
    document_id: str
    document_title: str
    chunk_id: str
    chunk_text: str
    page_number: int
    char_start: int
    char_end: int


def _classify(text: str) -> str:
    t = text.lower()
    if any(m in t for m in OBLIGATION_MARKERS):
        return "obligation"
    if any(m in t for m in NON_OBLIGATION_MARKERS):
        return "non_obligation"
    return "neutral"


def _build_rule_record(payload: ExtractInput, settings: Settings) -> ObligationRecord:
    classification = _classify(payload.chunk_text)
    modality_detected = "none"
    if "shall" in payload.chunk_text.lower():
        modality_detected = "shall"
    elif "must" in payload.chunk_text.lower():
        modality_detected = "must"
    elif "may" in payload.chunk_text.lower():
        modality_detected = "may"
    elif any(x in payload.chunk_text.lower() for x in NEUTRAL_MARKERS):
        modality_detected = "none"

    if classification == "obligation":
        output = [
            ObligationOutput(
                obligation_id=f"OBL-{payload.chunk_id}",
                subject="TBD_SUBJECT",
                modality="MUST" if modality_detected in {"shall", "must"} else "MUST",
                modality_detected=modality_detected,
                action=payload.chunk_text[:180],
                conditions=None,
                deadline=None,
                reference_anchor=f"page_{payload.page_number}",
            )
        ]
        final_logic = "Mandatory modal detected; treated as obligation."
        confidence = 0.80
    elif classification == "non_obligation":
        output = NonObligationOutput(
            status="rejected",
            message="Permission or non-mandatory clause",
            reason="Non-mandatory modality detected (e.g., may/can/should).",
        )
        final_logic = "Permissive modal detected; no obligation extracted."
        confidence = 0.86
    else:
        output = NonObligationOutput(
            status="no_obligation",
            message="Neutral/definition clause",
            reason="No mandatory action detected.",
        )
        final_logic = "Definition/declaration pattern detected; neutral."
        confidence = 0.90

    record = ObligationRecord(
        metadata=Metadata(
            id=f"{payload.chunk_id}_{classification}",
            source_id=payload.chunk_id,
            regulation_name=payload.document_title,
            jurisdiction="UNKNOWN",
            doc_type="Regulation",
            version=payload.dataset_version,
            last_updated=payload.dataset_version,
            regulator="UNKNOWN",
            source_document=SourceDocument(
                document_id=payload.document_id,
                document_title=payload.document_title,
            ),
            source_chunk=SourceChunk(
                chunk_id=payload.chunk_id,
                chunk_text=payload.chunk_text,
                page_number=payload.page_number,
                char_start=payload.char_start,
                char_end=payload.char_end,
                embedding_id=f"vec_{payload.chunk_id}",
            ),
            traceability=Traceability(
                model_used="rule-based",
                prompt_version=settings.prompt_version,
                pipeline_version=settings.pipeline_version,
                processing_node=settings.processing_node,
                validation_status="pending",
            ),
        ),
        classification=classification,
        linkages=Linkages(
            vector_id=f"vec_{payload.chunk_id}",
            graph_node_ids=[],
            chunk_id=payload.chunk_id,
        ),
        token_limit=TokenLimit(
            max_input_tokens=settings.max_input_tokens,
            max_output_tokens=settings.max_output_tokens,
            utilization_rate="10%",
        ),
        context=Context(raw_text=payload.chunk_text),
        tool_use=ToolUse(available_tools=[]),
        reasoning=Reasoning(
            steps=[
                ReasoningStep(step="Classify sentence modality", result=classification),
                ReasoningStep(step="Detect modal token", result=modality_detected),
            ],
            final_logic=final_logic,
        ),
        output=output,
        evaluation=Evaluation(
            confidence_score=confidence,
            risk_level="LOW" if classification != "obligation" else "MEDIUM",
            hallucination_flag=False,
        ),
    )
    return record


async def extract_record(payload: ExtractInput, settings: Settings, client: AsyncOpenAI | None = None) -> ObligationRecord:
    if not settings.use_openai or client is None:
        return _build_rule_record(payload, settings)
    base = _build_rule_record(payload, settings)
    for attempt in range(settings.max_retries):
        try:
            completion = await client.beta.chat.completions.parse(
                model=settings.openai_model,
                messages=build_messages(payload),
                max_tokens=settings.max_output_tokens,
                response_format=ObligationRecord,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("OpenAI parse returned null parsed object")
            parsed.metadata.traceability.model_used = settings.openai_model
            parsed.metadata.traceability.prompt_version = settings.prompt_version
            parsed.metadata.traceability.pipeline_version = settings.pipeline_version
            parsed.metadata.traceability.processing_node = settings.processing_node
            # Ensure source linkage cannot drift away from input payload.
            parsed.metadata.source_document.document_id = payload.document_id
            parsed.metadata.source_document.document_title = payload.document_title
            parsed.metadata.source_chunk.chunk_id = payload.chunk_id
            parsed.metadata.source_chunk.chunk_text = payload.chunk_text
            parsed.metadata.source_chunk.page_number = payload.page_number
            parsed.metadata.source_chunk.char_start = payload.char_start
            parsed.metadata.source_chunk.char_end = payload.char_end
            parsed.linkages.chunk_id = payload.chunk_id
            parsed.linkages.vector_id = parsed.linkages.vector_id or f"vec_{payload.chunk_id}"
            return parsed
        except Exception:  # noqa: BLE001
            if attempt == settings.max_retries - 1:
                break
            await asyncio.sleep(2**attempt)
    return base


def document_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
