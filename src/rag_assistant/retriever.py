"""Retrieval interface for the RAG pipeline."""

from pathlib import Path

from rag_assistant.schema import RetrievalResult, TextChunk
from rag_assistant.vector_store import ChromaVectorStore


class Retriever:
    """Thin retrieval service over a vector store."""

    def __init__(self, vector_store: ChromaVectorStore, top_k: int = 4) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.vector_store = vector_store
        self.top_k = top_k

    def index(self, chunks: list[TextChunk]) -> None:
        """Add chunks to the underlying vector store."""

        self.vector_store.add_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source: str | Path | None = None,
        profile: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve source-aware chunks for a user query."""

        return self.vector_store.similarity_search(query, top_k=top_k or self.top_k, source=source, profile=profile)
