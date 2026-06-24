"""End-to-end retrieval-augmented answer pipeline."""

from pathlib import Path

from rag_assistant.llm_client import LlmClient
from rag_assistant.prompt_builder import build_rag_prompt
from rag_assistant.retriever import Retriever
from rag_assistant.schema import RagAnswer, RetrievalResult, SourceReference


class RagPipeline:
    """Retrieve relevant chunks, build a prompt, and generate an answer."""

    def __init__(self, retriever: Retriever, llm_client: LlmClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int | None = None, source: str | Path | None = None) -> RagAnswer:
        """Answer a user question with source references."""

        retrieval_results = self.retriever.retrieve(question, top_k=top_k, source=source)
        prompt = build_rag_prompt(question, retrieval_results)
        answer_text = self.llm_client.generate(prompt)

        return RagAnswer(
            answer=answer_text,
            sources=_build_sources(retrieval_results),
            retrieved_chunks=retrieval_results,
            model=self.llm_client.model,
            prompt=prompt,
        )


def _build_sources(retrieval_results: list[RetrievalResult]) -> list[SourceReference]:
    sources: list[SourceReference] = []
    seen: set[tuple[str, int, int | None]] = set()

    for result in retrieval_results:
        chunk = result.chunk
        source_key = (str(chunk.source_path), chunk.chunk_index, chunk.page_number)
        if source_key in seen:
            continue
        seen.add(source_key)
        sources.append(
            SourceReference(
                file_name=chunk.file_name,
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                score=result.score,
            )
        )

    return sources
