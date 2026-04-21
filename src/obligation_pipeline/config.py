"""Environment-backed configuration for obligation data pipeline."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    use_openai: bool = Field(default=False, alias="USE_OPENAI")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    max_concurrency: int = Field(default=8, alias="MAX_CONCURRENCY")
    max_retries: int = Field(default=4, alias="MAX_RETRIES")
    max_input_tokens: int = Field(default=4096, alias="MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(default=1024, alias="MAX_OUTPUT_TOKENS")
    pipeline_version: str = Field(default="obl-pipeline-v1", alias="PIPELINE_VERSION")
    prompt_version: str = Field(default="v1.0", alias="PROMPT_VERSION")
    processing_node: str = Field(default="local-dev", alias="PROCESSING_NODE")

    target_obligation_ratio: float = Field(default=0.50, alias="TARGET_OBLIGATION_RATIO")
    target_non_obligation_ratio: float = Field(default=0.20, alias="TARGET_NON_OBLIGATION_RATIO")
    target_neutral_ratio: float = Field(default=0.30, alias="TARGET_NEUTRAL_RATIO")

    enable_vector_ingest: bool = Field(default=True, alias="ENABLE_VECTOR_INGEST")
    enable_graph_ingest: bool = Field(default=True, alias="ENABLE_GRAPH_INGEST")
    embedding_provider: str = Field(default="sentence_transformers", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    vector_db_type: str = Field(default="chroma", alias="VECTOR_DB_TYPE")
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="obligation_chunks", alias="CHROMA_COLLECTION")

    graph_db_type: str = Field(default="neo4j", alias="GRAPH_DB_TYPE")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
