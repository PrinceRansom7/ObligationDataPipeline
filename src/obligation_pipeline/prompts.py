"""Prompt construction for structured obligation extraction."""

from __future__ import annotations

def build_messages(payload: object) -> list[dict]:
    document_title = getattr(payload, "document_title")
    chunk_id = getattr(payload, "chunk_id")
    page_number = getattr(payload, "page_number")
    chunk_text = getattr(payload, "chunk_text")
    return [
        {
            "role": "system",
            "content": (
                "You are an expert legal compliance and regulatory extraction assistant.\n\n"
                "Your primary task is to read the provided regulatory chunk and accurately classify it into one of three categories: 'obligation', 'non_obligation', or 'neutral'.\n"
                "Once classified, you must return a STRICT JSON object representing the extracted data that conforms to the requested schema.\n\n"
                "SPECIAL HANDLING RULES:\n"
                "1. If the text mandates an action, requirement, or prohibition, classify it as 'obligation'.\n"
                "2. If it is explanatory, definitional, or context-setting without requiring action, it is 'non_obligation' or 'neutral'.\n"
                "3. Extract key details such as the subject (who must comply), modality (must, shall, may), action (what must be done), conditions, and deadlines when present.\n"
                "4. Your output must strictly match the JSON schema criteria evaluated downstream.\n\n"
                "VERY IMPORTANT:\n"
                "- Respond with exactly ONE JSON object.\n"
                "- Do NOT include any prose explanation, markdown blocks, or preamble before or after the JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document: {document_title}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Page: {page_number}\n"
                f"Text:\n{chunk_text}\n\n"
                "Return one JSON object only."
            ),
        },
    ]
