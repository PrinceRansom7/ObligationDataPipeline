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


# ── Composite Confidence Scoring ────────────────────────────────────────
#
# A robust confidence score combining multiple signals:
#   classification, modality, completeness, reasoning, grounding.
#
# Formula:
#   confidence = w1*classification + w2*modality + w3*completeness
#                + w4*reasoning + w5*grounding
#
# Default weights: 0.25, 0.15, 0.20, 0.15, 0.25
# Tiers: high (≥0.90) → accept, medium (0.70–0.90) → review, low (<0.70) → reject
# ─────────────────────────────────────────────────────────────────────────


class ConfidenceBreakdown(BaseModel):
    """Multi-signal confidence breakdown for auditability.

    Each component is normalized to [0,1]. The composite confidence_score
    on Evaluation is a weighted sum of these.
    """

    classification_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Model's probability that the classification label is correct. "
            "Calibrate via temperature scaling on held-out data."
        ),
    )
    modality_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Strength of deontic cues: "
            "shall/must → 1.0, should → 0.6, may/can → 0.2, none → 0.1."
        ),
    )
    extraction_completeness: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Fraction of required schema fields (subject, action, conditions, "
            "deadline, etc.) that were successfully extracted."
        ),
    )
    reasoning_consistency: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Logical alignment between chain-of-thought reasoning steps "
            "and final output. Catches contradictions."
        ),
    )
    grounding_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Overlap / entailment between extracted obligation text and "
            "source chunk text. Low score flags potential hallucination."
        ),
    )


class Evaluation(BaseModel):
    """Quality evaluation with composite confidence scoring.

    confidence_score is computed as:
      w1*classification + w2*modality + w3*completeness + w4*reasoning + w5*grounding

    confidence_breakdown provides per-signal scores for auditability.
    confidence_tier provides a human-readable bucket (high/medium/low).
    hallucination_flag is set when grounding_score < 0.3.
    """

    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Composite weighted confidence score.",
    )
    confidence_breakdown: ConfidenceBreakdown = Field(
        default_factory=ConfidenceBreakdown,
        description="Per-signal confidence breakdown for auditability.",
    )
    confidence_tier: str = Field(
        default="low",
        description="High (≥0.90), medium (0.70–0.90), low (<0.70).",
    )
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
