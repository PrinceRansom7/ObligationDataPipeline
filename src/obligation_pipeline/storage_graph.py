"""Graph database ingestion helpers for obligation records."""

from __future__ import annotations

from .config import Settings
from .schema import NonObligationOutput, ObligationRecord


def ingest_records_to_neo4j(records: list[ObligationRecord], settings: Settings) -> int:
    if not records:
        return 0
    if settings.graph_db_type.lower() != "neo4j" or not settings.neo4j_password:
        return 0
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    written = 0
    try:
        with driver.session() as session:
            for rec in records:
                doc_id = rec.metadata.source_document.document_id
                chunk_id = rec.metadata.source_chunk.chunk_id
                cls = rec.classification
                session.run(
                    """
                    MERGE (d:Document {id:$doc_id})
                    SET d.title = $doc_title
                    MERGE (c:Chunk {id:$chunk_id})
                    SET c.text = $chunk_text, c.classification = $classification
                    MERGE (d)-[:CONTAINS]->(c)
                    """,
                    doc_id=doc_id,
                    doc_title=rec.metadata.source_document.document_title,
                    chunk_id=chunk_id,
                    chunk_text=rec.context.raw_text,
                    classification=cls,
                )
                if cls == "obligation" and isinstance(rec.output, list):
                    for obl in rec.output:
                        session.run(
                            """
                            MERGE (o:Obligation {id:$obl_id})
                            SET o.subject = $subject, o.action = $action, o.modality = $modality
                            MERGE (c:Chunk {id:$chunk_id})
                            MERGE (c)-[:YIELDS]->(o)
                            """,
                            obl_id=obl.obligation_id,
                            subject=obl.subject,
                            action=obl.action,
                            modality=obl.modality,
                            chunk_id=chunk_id,
                        )
                elif isinstance(rec.output, NonObligationOutput):
                    session.run(
                        """
                        MERGE (n:NonObligation {id:$nid})
                        SET n.status = $status, n.reason = $reason
                        MERGE (c:Chunk {id:$chunk_id})
                        MERGE (c)-[:YIELDS]->(n)
                        """,
                        nid=f"NON-{chunk_id}",
                        status=rec.output.status,
                        reason=rec.output.reason,
                        chunk_id=chunk_id,
                    )
                written += 1
    finally:
        driver.close()
    return written
