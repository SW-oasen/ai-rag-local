"""ChromaDB-backed vector storage for text chunks."""

from hashlib import sha1
from pathlib import Path
from typing import Any

from rag_assistant.embeddings import EmbeddingProvider
from rag_assistant.schema import IndexedSource, RetrievalResult, TextChunk


class VectorStoreError(RuntimeError):
    """Raised when the local vector database cannot be opened or queried."""


class ChromaVectorStore:
    """Persistent local vector store for source-aware chunks."""

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "rag_chunks",
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("Chroma vector storage requires the 'chromadb' package.") from exc

        self.embedding_provider = embedding_provider
        try:
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open Chroma vector store at '{persist_directory}'. "
                "If this path is on a drive where SQLite/Chroma cannot write reliably, "
                "try a different path with --vector-store, for example "
                "$env:LOCALAPPDATA\\local_rag_assistant\\vector_store."
            ) from exc

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Embed and store chunks."""

        if not chunks:
            return

        try:
            embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
            self.collection.upsert(
                ids=[_chunk_id(chunk) for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[_chunk_to_metadata(chunk) for chunk in chunks],
                embeddings=embeddings,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Could not write chunks to the Chroma vector store.") from exc

    def similarity_search(self, query: str, top_k: int = 4, source: str | Path | None = None) -> list[RetrievalResult]:
        """Return the most relevant chunks for a query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        try:
            query_embedding = self.embedding_provider.embed_query(query)
            source_filter = self._build_source_filter(source)
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=source_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Could not query the Chroma vector store.") from exc

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results: list[RetrievalResult] = []
        for document_text, metadata, distance in zip(documents, metadatas, distances):
            chunk = _metadata_to_chunk(document_text, metadata)
            results.append(RetrievalResult(chunk=chunk, score=distance))

        return results

    def count(self) -> int:
        """Return the number of stored chunks in the collection."""

        return self.collection.count()

    def delete_source(self, source: str | Path) -> int:
        """Delete all chunks for one indexed source and return the number removed."""

        try:
            source_filter = self._build_source_filter(source)
            result = self.collection.get(where=source_filter, include=["metadatas"])
            ids = result.get("ids", [])
            if not ids:
                return 0
            self.collection.delete(ids=ids)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Could not delete source from the Chroma vector store.") from exc
        return len(ids)

    def reset(self) -> int:
        """Delete all chunks from the collection and return the number removed."""

        try:
            result = self.collection.get(include=["metadatas"])
            ids = result.get("ids", [])
            if not ids:
                return 0
            self.collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError("Could not reset the Chroma vector store.") from exc
        return len(ids)

    def get_chunks_by_source(self, source: str | Path) -> list[TextChunk]:
        """Return stored chunks for a source path or file name."""

        source_text = str(source)
        try:
            result = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError("Could not read chunks from the Chroma vector store.") from exc

        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        chunks = [
            _metadata_to_chunk(document_text, metadata)
            for document_text, metadata in zip(documents, metadatas)
            if _source_matches(metadata, source_text)
        ]
        return sorted(chunks, key=lambda chunk: (str(chunk.source_path), chunk.page_number or 0, chunk.chunk_index))

    def list_sources(self) -> list[IndexedSource]:
        """Return document-level summaries for indexed sources."""

        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError("Could not read sources from the Chroma vector store.") from exc

        grouped: dict[str, dict[str, Any]] = {}
        for metadata in result.get("metadatas", []):
            source_path = str(metadata["source_path"])
            source = grouped.setdefault(
                source_path,
                {
                    "file_name": str(metadata["file_name"]),
                    "source_path": Path(source_path),
                    "document_type": str(metadata["document_type"]),
                    "chunk_count": 0,
                    "pages": set(),
                },
            )
            source["chunk_count"] += 1
            if "page_number" in metadata:
                source["pages"].add(int(metadata["page_number"]))

        sources = [
            IndexedSource(
                file_name=str(source["file_name"]),
                source_path=source["source_path"],
                document_type=str(source["document_type"]),
                chunk_count=int(source["chunk_count"]),
                page_count=len(source["pages"]) or None,
            )
            for source in grouped.values()
        ]
        return sorted(sources, key=lambda source: (source.file_name.lower(), str(source.source_path)))

    def _build_source_filter(self, source: str | Path | None) -> dict[str, str] | None:
        if source is None:
            return None

        source_text = str(source)
        matches = [indexed_source for indexed_source in self.list_sources() if _source_matches_indexed(indexed_source, source_text)]
        if not matches:
            return {"file_name": source_text}
        if len(matches) > 1:
            matched_paths = ", ".join(str(match.source_path) for match in matches)
            raise VectorStoreError(f"Source filter '{source_text}' matched multiple indexed sources: {matched_paths}")
        return {"source_path": str(matches[0].source_path)}


def _chunk_id(chunk: TextChunk) -> str:
    raw_id = f"{chunk.source_path}:{chunk.page_number}:{chunk.chunk_index}:{chunk.start_char}:{chunk.end_char}"
    return sha1(raw_id.encode("utf-8")).hexdigest()


def _chunk_to_metadata(chunk: TextChunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_path": str(chunk.source_path),
        "file_name": chunk.file_name,
        "document_type": chunk.document_type,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number

    for key, value in chunk.metadata.items():
        if isinstance(value, str | int | float | bool):
            metadata[f"extra_{key}"] = value

    return metadata


def _metadata_to_chunk(text: str, metadata: dict[str, Any]) -> TextChunk:
    extra_metadata = {
        key.removeprefix("extra_"): value
        for key, value in metadata.items()
        if key.startswith("extra_")
    }

    return TextChunk(
        text=text,
        source_path=Path(str(metadata["source_path"])),
        file_name=str(metadata["file_name"]),
        document_type=str(metadata["document_type"]),
        chunk_index=int(metadata["chunk_index"]),
        start_char=int(metadata["start_char"]),
        end_char=int(metadata["end_char"]),
        page_number=int(metadata["page_number"]) if "page_number" in metadata else None,
        metadata=extra_metadata,
    )


def _source_matches(metadata: dict[str, Any], source: str) -> bool:
    source_path = str(metadata["source_path"])
    file_name = str(metadata["file_name"])
    return source == source_path or source == file_name or source_path.endswith(source)


def _source_matches_indexed(indexed_source: IndexedSource, source: str) -> bool:
    source_path = str(indexed_source.source_path)
    return source == source_path or source == indexed_source.file_name or source_path.endswith(source)
