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

    assert messages[0] == "Summarizing 3 chunks in 2 groups (3 LLM calls total)."
    assert messages[1].startswith("Progress 0%: group 1/2 started")
    assert "2 chunks, example.md, chunks 0-1" in messages[1]
    assert messages[2].startswith("Progress 33%: group 1/2 finished in ")
    assert messages[3].startswith("Progress 33%: group 2/2 started")
    assert "1 chunks, example.md, chunk 2" in messages[3]
    assert messages[4].startswith("Progress 67%: group 2/2 finished in ")
    assert messages[5] == "Progress 67%: final merge started (2 partial summaries)."
    assert messages[6].startswith("Progress 100%: final summary finished in ")


def test_document_summarizer_handles_empty_chunks() -> None:
    result = DocumentSummarizer(FakeLlmClient()).summarize([])

    assert result.summary == "No chunks were available to summarize."
    assert result.sources == []
    assert result.partial_summaries == []


def test_summary_prompts_include_source_labels() -> None:
    chunks = [_chunk(0), _chunk(1)]

    chunk_prompt = build_chunk_summary_prompt(chunks, language="fra")
    final_prompt = build_final_summary_prompt(["partial"], chunks, language="fra")

    assert "[example.md, chunk 0]" in chunk_prompt
    assert "Chunk text 1" in chunk_prompt
    assert "Write the summary in French." in chunk_prompt
    assert "Available source chunks:" in final_prompt
    assert "example.md, chunk 1" in final_prompt
    assert "Write the summary in French." in final_prompt
