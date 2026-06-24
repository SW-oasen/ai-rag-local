"""Local LLM clients for answer generation."""

from typing import Protocol


class LlmClient(Protocol):
    """Interface for local chat-completion clients."""

    model: str

    def generate(self, prompt: str) -> str:
        """Generate an answer from a prompt."""


class OllamaLlmClient:
    """Local LLM client backed by Ollama."""

    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        host: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("Ollama LLM generation requires the 'ollama' package.") from exc

        self.model = model
        self.temperature = temperature
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        message = response.get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError("Ollama chat response did not include message content.")
        return content.strip()

