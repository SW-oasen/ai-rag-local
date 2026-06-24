"""Embedding providers for local retrieval."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Interface for converting text into embedding vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""


class EmbeddingError(RuntimeError):
    """Raised when local embedding generation fails."""


class OllamaEmbeddingProvider:
    """Embedding provider backed by a local Ollama model."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("Ollama embeddings require the 'ollama' package.") from exc

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model = model
        self.batch_size = batch_size
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embed(model=self.model, input=texts)
        except Exception as exc:
            raise EmbeddingError(
                "Ollama embedding generation failed. Make sure Ollama is running, "
                f"the embedding model '{self.model}' is pulled, and try a smaller "
                "batch size if the source documents are large."
            ) from exc

        embeddings = response.get("embeddings")
        if embeddings is None:
            raise EmbeddingError("Ollama embedding response did not include embeddings.")
        return embeddings
