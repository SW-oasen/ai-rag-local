from pathlib import Path

import pytest

from rag_assistant.embeddings import EmbeddingProvider
from rag_assistant.retriever import Retriever
from rag_assistant.schema import TextChunk
from rag_assistant.vector_store import ChromaVectorStore


class KeywordEmbeddingProvider(EmbeddingProvider):
    vocabulary = ["alpha", "beta", "gamma", "retrieval"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(term)) for term in self.vocabulary]


def test_retriever_returns_relevant_source_aware_chunks(tmp_path: Path) -> None:
    chunks = [
        TextChunk(
            text="Alpha systems explain local retrieval.",
            source_path=Path("docs/alpha.md"),
            file_name="alpha.md",
            document_type="md",
            chunk_index=0,
            start_char=0,
            end_char=38,
        ),
        TextChunk(
            text="Beta notes are about unrelated setup.",
            source_path=Path("docs/beta.txt"),
            file_name="beta.txt",
            document_type="txt",
            chunk_index=1,
            start_char=0,
            end_char=37,
        ),
    ]
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
        collection_name="test_chunks",
    )
    retriever = Retriever(vector_store, top_k=1)

    retriever.index(chunks)
    results = retriever.retrieve("alpha retrieval")

    assert len(results) == 1
    assert results[0].chunk.file_name == "alpha.md"
    assert results[0].chunk.source_path == Path("docs/alpha.md")
    assert results[0].chunk.chunk_index == 0
    assert results[0].score is not None


def test_retriever_forwards_source_filter(tmp_path: Path) -> None:
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
        collection_name="test_source_filter",
    )
    retriever = Retriever(vector_store, top_k=2)
    retriever.index(
        [
            TextChunk(
                text="Alpha systems explain local retrieval.",
                source_path=Path("docs/alpha.md"),
                file_name="alpha.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=38,
            ),
            TextChunk(
                text="Alpha systems appear in beta notes too.",
                source_path=Path("docs/beta.md"),
                file_name="beta.md",
                document_type="md",
                chunk_index=0,
                start_char=0,
                end_char=40,
            ),
        ]
    )

    results = retriever.retrieve("alpha systems", source="beta.md")

    assert [result.chunk.file_name for result in results] == ["beta.md"]


def test_retriever_rejects_invalid_top_k(tmp_path: Path) -> None:
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path,
        embedding_provider=KeywordEmbeddingProvider(),
        collection_name="test_invalid_top_k",
    )

    with pytest.raises(ValueError, match="top_k must be greater"):
        Retriever(vector_store, top_k=0)
