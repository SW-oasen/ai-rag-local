from pathlib import Path

import pytest

from rag_assistant.embeddings import EmbeddingProvider
from rag_assistant.schema import TextChunk
from rag_assistant.vector_store import ChromaVectorStore, VectorStoreError
from rag_assistant.vector_store import _chunk_id, _chunk_to_metadata, _metadata_to_chunk


class DummyEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


def test_chunk_metadata_preserves_profile_and_profile_specific_ids() -> None:
    general_chunk = TextChunk(
        text="General text.",
        source_path=Path("docs/shared.md"),
        file_name="shared.md",
        document_type="md",
        chunk_index=0,
        start_char=0,
        end_char=13,
        metadata={"profile": "general"},
    )
    technical_chunk = TextChunk(
        text="General text.",
        source_path=Path("docs/shared.md"),
        file_name="shared.md",
        document_type="md",
        chunk_index=0,
        start_char=0,
        end_char=13,
        metadata={"profile": "technical"},
    )

    metadata = _chunk_to_metadata(technical_chunk)
    restored = _metadata_to_chunk(technical_chunk.text, metadata)

    assert metadata["extra_profile"] == "technical"
    assert restored.metadata["profile"] == "technical"
    assert _chunk_id(general_chunk) != _chunk_id(technical_chunk)


def test_metadata_without_profile_is_restored_as_general() -> None:
    chunk = TextChunk(
        text="Legacy text.",
        source_path=Path("docs/legacy.md"),
        file_name="legacy.md",
        document_type="md",
        chunk_index=0,
        start_char=0,
        end_char=12,
    )

    metadata = _chunk_to_metadata(chunk)
    restored = _metadata_to_chunk(chunk.text, metadata)

    assert "extra_profile" not in metadata
    assert restored.metadata["profile"] == "general"


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


def test_vector_store_general_profile_includes_legacy_but_excludes_special_profiles(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "general_profile_filter")
    store.add_chunks(
        [
            TextChunk(
                text="Legacy shared evidence.",
                source_path=Path("docs/legacy.md"),
                file_name="legacy.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=23,
            ),
            TextChunk(
                text="General shared evidence.",
                source_path=Path("docs/general.md"),
                file_name="general.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=24,
                metadata={"profile": "general"},
            ),
            TextChunk(
                text="Technical shared evidence.",
                source_path=Path("docs/technical.md"),
                file_name="technical.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=26,
                metadata={"profile": "technical"},
            ),
        ]
    )

    results = store.similarity_search("shared evidence", top_k=5, profile="general")

    assert {result.chunk.file_name for result in results} == {"legacy.md", "general.md"}
    assert {result.chunk.metadata["profile"] for result in results} == {"general"}


def test_vector_store_lists_sources_by_profile(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "profile_source_listing")
    store.add_chunks(
        [
            TextChunk(
                text="Legacy shared evidence.",
                source_path=Path("docs/legacy.md"),
                file_name="legacy.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=23,
            ),
            TextChunk(
                text="General shared evidence.",
                source_path=Path("docs/general.md"),
                file_name="general.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=24,
                metadata={"profile": "general"},
            ),
            TextChunk(
                text="Technical shared evidence.",
                source_path=Path("docs/technical.md"),
                file_name="technical.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=26,
                metadata={"profile": "technical"},
            ),
        ]
    )

    general_sources = store.list_sources(profile="general")
    technical_sources = store.list_sources(profile="technical")

    assert {source.file_name for source in general_sources} == {"legacy.md", "general.md"}
    assert [source.file_name for source in technical_sources] == ["technical.md"]


def test_vector_store_special_profile_excludes_general_and_legacy_chunks(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "special_profile_filter")
    store.add_chunks(
        [
            TextChunk(
                text="Legacy shared evidence.",
                source_path=Path("docs/legacy.md"),
                file_name="legacy.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=23,
            ),
            TextChunk(
                text="Technical shared evidence.",
                source_path=Path("docs/technical.md"),
                file_name="technical.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=26,
                metadata={"profile": "technical"},
            ),
        ]
    )

    results = store.similarity_search("shared evidence", top_k=5, profile="technical")

    assert [result.chunk.file_name for result in results] == ["technical.md"]


def test_vector_store_deletes_one_source(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "source_delete")
    store.add_chunks(
        [
            TextChunk(
                text="Alpha first chunk.",
                source_path=Path("docs/alpha.md"),
                file_name="alpha.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=18,
            ),
            TextChunk(
                text="Alpha second chunk.",
                source_path=Path("docs/alpha.md"),
                file_name="alpha.md",
                document_type="md",
                chunk_index=1,
                start_char=19,
                end_char=38,
            ),
            TextChunk(
                text="Beta chunk.",
                source_path=Path("docs/beta.md"),
                file_name="beta.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=11,
            ),
        ]
    )

    deleted_count = store.delete_source("alpha.md")

    assert deleted_count == 2
    assert store.count() == 1
    assert [source.file_name for source in store.list_sources()] == ["beta.md"]


def test_vector_store_reset_deletes_all_chunks(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, DummyEmbeddingProvider(), "source_reset")
    store.add_chunks(
        [
            TextChunk(
                text="Alpha chunk.",
                source_path=Path("docs/alpha.md"),
                file_name="alpha.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=12,
            ),
            TextChunk(
                text="Beta chunk.",
                source_path=Path("docs/beta.md"),
                file_name="beta.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=11,
            ),
        ]
    )

    deleted_count = store.reset()

    assert deleted_count == 2
    assert store.count() == 0
    assert store.list_sources() == []
