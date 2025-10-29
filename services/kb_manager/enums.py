"""Domain enums for the knowledge base manager service."""

from __future__ import annotations

from enum import Enum


class EmbeddingBackend(str, Enum):
    """Available embedding backends for indexing."""

    DEFAULT = "default"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    YANDEX = "yandex"
    LOCAL = "local"


class KnowledgeBasePreparationMethod(str, Enum):
    """Ways to pre-process documents prior to indexing."""

    CLEAR_CHUNKING = "clear_chunking"
    RAPTOR = "raptor"
    QA_TABLE = "qa_table"


class ChunkingStrategy(str, Enum):
    """Chunking approaches supported by the service."""

    SIMPLE_LENGTH = "simple_length"
    SENTENCE = "sentence"
    TABLE_ROW = "table_row"
    PARAGRAPH_JOIN = "paragraph_join"


class ChunkSizeUnit(str, Enum):
    """Unit type for chunk sizing."""

    TOKENS = "tokens"
    CHARACTERS = "characters"
