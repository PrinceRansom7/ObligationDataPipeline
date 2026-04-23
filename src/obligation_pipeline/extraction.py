"""Chunk-level obligation extraction using OpenAI structured outputs.

Enhanced with composite confidence scoring (5 sub-signals).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher

from openai import AsyncOpenAI

from .config import Settings
from .prompts import build_messages
from .schema import (
    ConfidenceBreakdown,
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


# ── Deontic modal scoring ───────────────────────────────────────────────

MODALITY_SCORES: dict[str, float] = {
    "shall": 1.0,
    "must": 1.0,
    "is required to": 1.0,
    "is obligated to": 1.0,
    "is required": 0.9,
    "ought to": 0.8,
    "should": 0.6,
    "is expected to": 0.5,
    "may": 0.2,
    "can": 0.2,
}


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
    # Structural metadata from ontology-aware chunking (optional)
    section_number: str | None = None
    clause_number: str | None = None
    semantic_role: str | None = None
    hierarchy_path: str | None = None
    act_name: str | None = None
    chapter: str | None = None
    regulator: str | None = None
    jurisdiction: str | None = None


# ── Confidence sub-score computation ────────────────────────────────────


def _detect_modality_score(text: str) -> tuple[float, str]:
    """Detect deontic cues and return (max_modality_score, detected_modal)."""
    text_lower = text.lower()
    best_score = 0.1
    best_modal = "none"
    for keyword, score in MODALITY_SCORES.items():
        if keyword in text_lower and score > best_score:
            best_score = score
            best_modal = keyword
    return best_score, best_modal


def _compute_extraction_completeness(output, classification: str) -> float:
    """Fraction of obligation fields extracted."""
    if classification != "obligation" or not isinstance(output, list):
        return 1.0  # non-obligation doesn't need field extraction
    if not output:
        return 0.0
    obl = output[0]
    fields = ["subject", "action", "conditions", "deadline"]
    filled = sum(1 for f in fields if getattr(obl, f, None) and str(getattr(obl, f, "")) not in ("TBD_SUBJECT", "None", ""))
    return filled / len(fields)


def _compute_grounding_score(obligation_text: str | None, source_text: str) -> float:
    """Compute text overlap between extraction and source (SequenceMatcher ratio)."""
    if not obligation_text:
        return 0.0
    return SequenceMatcher(None, obligation_text.lower(), source_text.lower()).ratio()


def _compute_reasoning_consistency(reasoning: Reasoning, classification: str) -> float:
    """Check if reasoning steps support the classification."""
    reasoning_text = " ".join(s.result.lower() for s in reasoning.steps) + " " + reasoning.final_logic.lower()
    if not reasoning_text.strip():
        return 0.3
    score = 0.4
    if classification in reasoning_text:
        score = 0.8
    elif classification == "obligation" and ("shall" in reasoning_text or "must" in reasoning_text or "obligat" in reasoning_text):
        score = 0.7
    elif classification == "non_obligation" and ("may" in reasoning_text or "permis" in reasoning_text):
        score = 0.7
    elif classification == "neutral" and ("defini" in reasoning_text or "neutral" in reasoning_text):
        score = 0.7
    if len(reasoning_text) > 80:
        score = min(1.0, score + 0.1)
    return score


def _compute_composite_confidence(breakdown: ConfidenceBreakdown, settings: Settings) -> float:
    """Weighted composite confidence from sub-signals."""
    score = (
        settings.confidence_w_classification * breakdown.classification_confidence
        + settings.confidence_w_modality * breakdown.modality_confidence
        + settings.confidence_w_completeness * breakdown.extraction_completeness
        + settings.confidence_w_reasoning * breakdown.reasoning_consistency
        + settings.confidence_w_grounding * breakdown.grounding_score
    )
    return round(min(1.0, max(0.0, score)), 4)


def _confidence_tier(score: float) -> str:
    if score >= 0.90:
        return "high"
    elif score >= 0.70:
        return "medium"
    return "low"


def _classify(text: str) -> str:
    t = text.lower()
    if any(m in t for m in OBLIGATION_MARKERS):
        return "obligation"
    if any(m in t for m in NON_OBLIGATION_MARKERS):
        return "non_obligation"
    return "neutral"


def _build_rule_record(payload: ExtractInput, settings: Settings) -> ObligationRecord:
    classification = _classify(payload.chunk_text)
    modality_score, modality_detected = _detect_modality_score(payload.chunk_text)

    # Use semantic_role hint from hierarchy parsing if available
    semantic_hint = (payload.semantic_role or "").lower()
    if semantic_hint == "obligation" and classification != "obligation" and modality_score >= 0.6:
        classification = "obligation"

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
        class_conf = 0.80 + (0.10 if semantic_hint == "obligation" else 0.0)
    elif classification == "non_obligation":
        output = NonObligationOutput(
            status="rejected",
            message="Permission or non-mandatory clause",
            reason="Non-mandatory modality detected (e.g., may/can/should).",
        )
        final_logic = "Permissive modal detected; no obligation extracted."
        class_conf = 0.86
    else:
        output = NonObligationOutput(
            status="no_obligation",
            message="Neutral/definition clause",
            reason="No mandatory action detected.",
        )
        final_logic = "Definition/declaration pattern detected; neutral."
        class_conf = 0.90

    reasoning = Reasoning(
        steps=[
            ReasoningStep(step="Classify sentence modality", result=classification),
            ReasoningStep(step="Detect modal token", result=modality_detected),
        ],
        final_logic=final_logic,
    )

    # Compute composite confidence breakdown
    completeness = _compute_extraction_completeness(output, classification)
    obligation_text = payload.chunk_text[:180] if classification == "obligation" else None
    grounding = _compute_grounding_score(obligation_text, payload.chunk_text) if classification == "obligation" else 0.5
    reasoning_consistency = _compute_reasoning_consistency(reasoning, classification)

    breakdown = ConfidenceBreakdown(
        classification_confidence=round(class_conf, 4),
        modality_confidence=round(modality_score, 4),
        extraction_completeness=round(completeness, 4),
        reasoning_consistency=round(reasoning_consistency, 4),
        grounding_score=round(grounding, 4),
    )
    composite = _compute_composite_confidence(breakdown, settings)

    record = ObligationRecord(
        metadata=Metadata(
            id=f"{payload.chunk_id}_{classification}",
            source_id=payload.chunk_id,
            regulation_name=payload.document_title,
            jurisdiction=payload.jurisdiction or "UNKNOWN",
            doc_type="Regulation",
            version=payload.dataset_version,
            last_updated=payload.dataset_version,
            regulator=payload.regulator or "UNKNOWN",
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
                section=payload.section_number,
                clause=payload.clause_number,
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
        reasoning=reasoning,
        output=output,
        evaluation=Evaluation(
            confidence_score=composite,
            confidence_breakdown=breakdown,
            confidence_tier=_confidence_tier(composite),
            risk_level="LOW" if classification != "obligation" else "MEDIUM",
            hallucination_flag=grounding < 0.3 if classification == "obligation" else False,
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
            parsed.metadata.source_chunk.section = payload.section_number
            parsed.metadata.source_chunk.clause = payload.clause_number
            parsed.linkages.chunk_id = payload.chunk_id
            parsed.linkages.vector_id = parsed.linkages.vector_id or f"vec_{payload.chunk_id}"

            # Recompute composite confidence — blend LLM self-reported scores
            # with independently computed verification scores.
            modality_score, _ = _detect_modality_score(payload.chunk_text)
            completeness = _compute_extraction_completeness(parsed.output, parsed.classification)
            grounding = _compute_grounding_score(
                parsed.output[0].action if isinstance(parsed.output, list) and parsed.output else None,
                payload.chunk_text,
            )
            reasoning_cons = _compute_reasoning_consistency(parsed.reasoning, parsed.classification)

            # The LLM now returns confidence_breakdown; blend with computed values
            llm_bd = parsed.evaluation.confidence_breakdown
            def _blend(llm_val: float, computed_val: float) -> float:
                """Average LLM self-report with computed verification score."""
                if llm_val > 0:
                    return round(min(1.0, (llm_val + computed_val) / 2), 4)
                return round(computed_val, 4)

            breakdown = ConfidenceBreakdown(
                classification_confidence=_blend(llm_bd.classification_confidence, 0.85),
                modality_confidence=_blend(llm_bd.modality_confidence, modality_score),
                extraction_completeness=_blend(llm_bd.extraction_completeness, completeness),
                reasoning_consistency=_blend(llm_bd.reasoning_consistency, reasoning_cons),
                grounding_score=_blend(llm_bd.grounding_score, grounding),
            )
            composite = _compute_composite_confidence(breakdown, settings)
            parsed.evaluation.confidence_score = composite
            parsed.evaluation.confidence_breakdown = breakdown
            parsed.evaluation.confidence_tier = _confidence_tier(composite)
            parsed.evaluation.hallucination_flag = breakdown.grounding_score < 0.3
            return parsed
        except Exception:  # noqa: BLE001
            if attempt == settings.max_retries - 1:
                break
            await asyncio.sleep(2**attempt)
    return base


def document_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
