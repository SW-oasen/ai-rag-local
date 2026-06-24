from pathlib import Path

from rag_assistant.schema import TextChunk
from rag_assistant.summarizer import (
    DocumentSummarizer,
    build_chunk_summary_prompt,
    build_final_summary_prompt,
)


class FakeLlmClient:
    model = "fake-summary-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("Create a final concise summary"):
            return "Final summary with limitations. [example.md, chunk 0]"
        return f"Partial summary {len(self.prompts)}"


def _chunk(index: int, text: str = "Chunk text") -> TextChunk:
    return TextChunk(
        text=f"{text} {index}",
        source_path=Path("docs/example.md"),
        file_name="example.md",
        document_type="md",
        chunk_index=index,
        start_char=index * 10,
        end_char=index * 10 + 9,
    )


def test_document_summarizer_uses_all_chunks_in_groups() -> None:
    llm = FakeLlmClient()
    summarizer = DocumentSummarizer(llm_client=llm, max_chunks_per_group=2)

    result = summarizer.summarize([_chunk(0), _chunk(1), _chunk(2)], question="main idea")

    assert result.summary == "Final summary with limitations. [example.md, chunk 0]"
    assert result.model == "fake-summary-model"
    assert result.source_chunks == [_chunk(0), _chunk(1), _chunk(2)]
    assert result.partial_summaries == ["Partial summary 1", "Partial summary 2"]
    assert len(result.sources) == 3
    assert len(llm.prompts) == 3
    assert "User focus: main idea" in llm.prompts[0]


def test_document_summarizer_reports_progress() -> None:
    messages: list[str] = []
    summarizer = DocumentSummarizer(llm_client=FakeLlmClient(), max_chunks_per_group=2)

    summarizer.summarize([_chunk(0), _chunk(1), _chunk(2)], progress_callback=messages.append)

    assert messages == [
        "Summarizing 3 chunks in 2 groups.",
        "Summarizing group 1/2.",
        "Summarizing group 2/2.",
        "Merging partial summaries.",
    ]


def test_document_summarizer_handles_empty_chunks() -> None:
    result = DocumentSummarizer(FakeLlmClient()).summarize([])

    assert result.summary == "No chunks were available to summarize."
    assert result.sources == []
    assert result.partial_summaries == []


def test_summary_prompts_include_source_labels() -> None:
    chunks = [_chunk(0), _chunk(1)]

    chunk_prompt = build_chunk_summary_prompt(chunks)
    final_prompt = build_final_summary_prompt(["partial"], chunks)

    assert "[example.md, chunk 0]" in chunk_prompt
    assert "Chunk text 1" in chunk_prompt
    assert "Available source chunks:" in final_prompt
    assert "example.md, chunk 1" in final_prompt
