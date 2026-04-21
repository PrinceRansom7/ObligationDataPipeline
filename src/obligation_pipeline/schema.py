"""Pydantic schema for obligation extraction dataset records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


ClassTag = Literal["obligation", "non_obligation", "neutral"]


class SourceDocument(BaseModel):
    document_id: str
    document_title: str
    document_type: str = "Regulation"
    publication_date: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    storage_path: str | None = None
    checksum: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    chunk_text: str
    page_number: int | None = None
    section: str | None = None
    clause: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    embedding_id: str | None = None
    preprocessing_version: str | None = "v1"


class Traceability(BaseModel):
    extraction_method: str = "LLM"
    model_used: str
    prompt_version: str
    pipeline_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_node: str
    validation_status: str = "pending"


class Metadata(BaseModel):
    id: str
    source_id: str
    regulation_name: str
    jurisdiction: str
    doc_type: str
    version: str
    last_updated: str
    regulator: str | None = None
    source_document: SourceDocument
    source_chunk: SourceChunk
    traceability: Traceability


class Linkages(BaseModel):
    vector_id: str
    graph_node_ids: list[str] = Field(default_factory=list)
    chunk_id: str


class TokenLimit(BaseModel):
    max_input_tokens: int
    max_output_tokens: int
    utilization_rate: str


class Context(BaseModel):
    raw_text: str


class ToolUse(BaseModel):
    available_tools: list[str] = Field(default_factory=list)
    tool_rationale: str | None = None
    tool_call: dict[str, Any] | None = None
    external_context: str | None = None


class ReasoningStep(BaseModel):
    step: str
    result: str


class Reasoning(BaseModel):
    steps: list[ReasoningStep]
    final_logic: str


class ObligationOutput(BaseModel):
    obligation_id: str
    subject: str
    modality: str
    modality_detected: str
    action: str
    conditions: str | None = None
    deadline: str | None = None
    reference_anchor: str | None = None


class NonObligationOutput(BaseModel):
    status: str
    message: str
    reason: str


class Evaluation(BaseModel):
    confidence_score: float
    risk_level: str
    hallucination_flag: bool = False


class ObligationRecord(BaseModel):
    metadata: Metadata
    classification: ClassTag
    linkages: Linkages
    token_limit: TokenLimit
    context: Context
    tool_use: ToolUse
    reasoning: Reasoning
    output: list[ObligationOutput] | NonObligationOutput
    evaluation: Evaluation
