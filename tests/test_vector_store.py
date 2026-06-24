from pathlib import Path

import pytest

from rag_assistant.embeddings import EmbeddingProvider
from rag_assistant.schema import TextChunk
from rag_assistant.vector_store import ChromaVectorStore, VectorStoreError


class DummyEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


def test_vector_store_wraps_invalid_persist_path(tmp_path) -> None:
    invalid_path = tmp_path / "not_a_directory"
    invalid_path.write_text("file blocks directory use", encoding="utf-8")

    with pytest.raises(VectorStoreError, match="Could not open Chroma vector store"):
        ChromaVectorStore(invalid_path, DummyEmbeddingProvider(), "invalid_path")


def test_vector_store_lists_indexed_sources(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "source_listing")
    store.add_chunks(
        [
            TextChunk(
                text="First page text.",
                source_path=tmp_path / "alpha.pdf",
                file_name="alpha.pdf",
                document_type="pdf",
                chunk_index=0,
                start_char=0,
                end_char=16,
                page_number=1,
            ),
            TextChunk(
                text="Second page text.",
                source_path=tmp_path / "alpha.pdf",
                file_name="alpha.pdf",
                document_type="pdf",
                chunk_index=1,
                start_char=17,
                end_char=34,
                page_number=2,
            ),
            TextChunk(
                text="Markdown text.",
                source_path=tmp_path / "notes.md",
                file_name="notes.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=14,
            ),
        ]
    )

    sources = store.list_sources()

    assert [source.file_name for source in sources] == ["alpha.pdf", "notes.md"]
    assert sources[0].chunk_count == 2
    assert sources[0].page_count == 2
    assert sources[1].chunk_count == 1
    assert sources[1].page_count is None


def test_vector_store_similarity_search_filters_by_source(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "source_filtering")
    store.add_chunks(
        [
            TextChunk(
                text="Shared evidence from alpha.",
                source_path=Path("docs/alpha.md"),
                file_name="alpha.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=27,
            ),
            TextChunk(
                text="Shared evidence from beta.",
                source_path=Path("docs/beta.md"),
                file_name="beta.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=26,
            ),
        ]
    )

    results = store.similarity_search("shared evidence", top_k=2, source="beta.md")

    assert [result.chunk.file_name for result in results] == ["beta.md"]
