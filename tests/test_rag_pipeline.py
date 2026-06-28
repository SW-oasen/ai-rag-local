from pathlib import Path

from rag_assistant.rag_pipeline import RagPipeline
from rag_assistant.schema import RetrievalResult, TextChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None
        self.last_source: str | Path | None = None
        self.last_profile: str | None = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source: str | Path | None = None,
        profile: str | None = None,
    ) -> list[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        self.last_source = source
        self.last_profile = profile
        return self.results


class FakeLlmClient:
    model = "fake-local-model"

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "It uses ChromaDB for vector storage. [source 1]"


def test_rag_pipeline_returns_answer_sources_chunks_model_and_prompt() -> None:
    result = RetrievalResult(
        chunk=TextChunk(
            text="ChromaDB stores local document chunks.",
            source_path=Path("docs/architecture.md"),
            file_name="architecture.md",
            document_type="md",
            chunk_index=0,
            start_char=0,
            end_char=39,
            page_number=3,
        ),
        score=0.2,
    )
    retriever = FakeRetriever([result])
    llm_client = FakeLlmClient()
    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)

    answer = pipeline.answer(
        "What stores chunks?",
        top_k=1,
        source="architecture.md",
        profile="technical",
        prompt_style="technical",
    )

    assert answer.answer == "It uses ChromaDB for vector storage. [source 1]"
    assert answer.model == "fake-local-model"
    assert answer.retrieved_chunks == [result]
    assert answer.sources[0].file_name == "architecture.md"
    assert answer.sources[0].page_number == 3
    assert "What stores chunks?" in answer.prompt
    assert "For technical answers" in answer.prompt
    assert llm_client.last_prompt == answer.prompt
    assert retriever.last_query == "What stores chunks?"
    assert retriever.last_top_k == 1
    assert retriever.last_source == "architecture.md"
    assert retriever.last_profile == "technical"
