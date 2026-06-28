import pytest

from rag_assistant.embeddings import EmbeddingError, OllamaEmbeddingProvider


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, model: str, input: list[str]) -> dict[str, list[list[float]]]:
        self.calls.append(input)
        return {"embeddings": [[float(len(text))] for text in input]}


class FailingClient:
    def embed(self, model: str, input: list[str]) -> dict[str, list[list[float]]]:
        raise RuntimeError("connection refused")


def test_ollama_embedding_provider_batches_requests() -> None:
    provider = OllamaEmbeddingProvider.__new__(OllamaEmbeddingProvider)
    provider.model = "test-embed"
    provider.batch_size = 2
    provider.client = FakeClient()

    embeddings = provider.embed_texts(["a", "bb", "ccc"])

    assert embeddings == [[1.0], [2.0], [3.0]]
    assert provider.client.calls == [["a", "bb"], ["ccc"]]


def test_ollama_embedding_provider_wraps_client_errors() -> None:
    provider = OllamaEmbeddingProvider.__new__(OllamaEmbeddingProvider)
    provider.model = "test-embed"
    provider.batch_size = 2
    provider.client = FailingClient()

    with pytest.raises(EmbeddingError, match="Ollama embedding generation failed"):
        provider.embed_texts(["a"])

