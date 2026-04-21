"""Chunking helpers for splitting legal text into clause-like units."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    char_start: int
    char_end: int


SENTENCE_SPLIT = re.compile(r"(?<=[\.\?!;])\s+")


def chunk_page_text(document_id: str, page_number: int, text: str) -> list[Chunk]:
    sentences = [x.strip() for x in SENTENCE_SPLIT.split(text) if x.strip()]
    out: list[Chunk] = []
    cursor = 0
    for idx, sentence in enumerate(sentences, start=1):
        start = text.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        cursor = end
        out.append(
            Chunk(
                chunk_id=f"{document_id}_p{page_number}_{idx}",
                text=sentence,
                page_number=page_number,
                char_start=start,
                char_end=end,
            )
        )
    return out
