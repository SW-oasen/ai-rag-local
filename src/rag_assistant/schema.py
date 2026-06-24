"""Shared data structures for document ingestion and retrieval."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    """Extracted text and metadata from a source document or document page."""

    text: str
    source_path: Path
    file_name: str
    document_type: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """A source-aware text chunk ready for embedding or retrieval."""

    text: str
    source_path: Path
    file_name: str
    document_type: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexedSource:
    """Summary of one source document stored in the vector index."""

    file_name: str
    source_path: Path
    document_type: str
    chunk_count: int
    page_count: int | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk and its vector-search score."""

    chunk: TextChunk
    score: float | None = None


@dataclass(frozen=True)
class SourceReference:
    """A compact source reference shown with generated answers."""

    file_name: str
    source_path: Path
    chunk_index: int
    page_number: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class RagAnswer:
    """Final RAG response with answer text and source traceability."""

    answer: str
    sources: list[SourceReference]
    retrieved_chunks: list[RetrievalResult]
    model: str
    prompt: str


@dataclass(frozen=True)
class SummaryResult:
    """Document summary result with source traceability."""

    summary: str
    sources: list[SourceReference]
    source_chunks: list[TextChunk]
    model: str
    partial_summaries: list[str]
