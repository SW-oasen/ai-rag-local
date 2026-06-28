from pathlib import Path

import pytest

from rag_assistant.evaluation import (
    RetrievalEvaluationError,
    RetrievalExample,
    evaluation_results_to_dicts,
    evaluate_retrieval,
    load_retrieval_examples,
)
from rag_assistant.schema import RetrievalResult, TextChunk


class FakeRetriever:
    def __init__(self) -> None:
        self.last_source: str | Path | None = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source: str | Path | None = None,
    ) -> list[RetrievalResult]:
        self.last_source = source
        return [
            RetrievalResult(
                chunk=TextChunk(
                    text="Semantic retrieval uses embeddings to match meaning.",
                    source_path=Path("docs/rag.md"),
                    file_name="rag.md",
                    document_type="md",
                    chunk_index=0,
                    start_char=0,
                    end_char=52,
                    page_number=2,
                ),
                score=0.1,
            )
        ]


class LineBreakRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source: str | Path | None = None,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=TextChunk(
                    text="Most of the stuff he has coded on his \nwebsite.",
                    source_path=Path("docs/story.pdf"),
                    file_name="story.pdf",
                    document_type="pdf",
                    chunk_index=0,
                    start_char=0,
                    end_char=48,
                ),
                score=0.1,
            )
        ]


def test_evaluate_retrieval_marks_expected_text_match_as_passed() -> None:
    examples = [
        RetrievalExample(
            question="How does retrieval match meaning?",
            expected_file_name="rag.md",
            expected_text="uses embeddings",
        )
    ]

    retriever = FakeRetriever()
    results = evaluate_retrieval(retriever, examples)

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].top_file_name == "rag.md"
    assert results[0].expected_text == "uses embeddings"
    assert results[0].top_chunk_index == 0
    assert results[0].top_page_number == 2
    assert results[0].top_score == 0.1
    assert results[0].matched_file_name == "rag.md"
    assert results[0].matched_chunk_index == 0
    assert results[0].matched_page_number == 2
    assert results[0].matched_score == 0.1
    assert retriever.last_source is None


def test_evaluate_retrieval_forwards_source_filter() -> None:
    retriever = FakeRetriever()
    examples = [
        RetrievalExample(
            question="How does retrieval match meaning?",
            expected_file_name="rag.md",
            expected_text="uses embeddings",
        )
    ]

    evaluate_retrieval(retriever, examples, source="rag.md")

    assert retriever.last_source == "rag.md"


def test_evaluate_retrieval_normalizes_pdf_line_breaks() -> None:
    examples = [
        RetrievalExample(
            question="What is on the website?",
            expected_file_name="story.pdf",
            expected_text="coded on his website",
        )
    ]

    results = evaluate_retrieval(LineBreakRetriever(), examples)

    assert results[0].passed is True


def test_evaluation_results_to_dicts_are_json_serializable() -> None:
    results = evaluate_retrieval(
        FakeRetriever(),
        [
            RetrievalExample(
                question="How does retrieval match meaning?",
                expected_file_name="rag.md",
                expected_text="uses embeddings",
            )
        ],
    )

    rows = evaluation_results_to_dicts(results)

    assert rows == [
        {
            "question": "How does retrieval match meaning?",
            "passed": True,
            "expected_file_name": "rag.md",
            "expected_text": "uses embeddings",
            "top_file_name": "rag.md",
            "top_chunk_index": 0,
            "top_page_number": 2,
            "top_score": 0.1,
            "matched_file_name": "rag.md",
            "matched_chunk_index": 0,
            "matched_page_number": 2,
            "matched_score": 0.1,
        }
    ]


def test_load_retrieval_examples_from_markdown_table(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.md"
    examples_path.write_text(
        """# Examples

| Question | Expected source | Expected evidence |
| --- | --- | --- |
| What is semantic retrieval? | README.md | semantic search |
| How do I inspect chunks? | README.md | retrieve |
""",
        encoding="utf-8",
    )

    examples = load_retrieval_examples(examples_path)

    assert examples == [
        RetrievalExample(
            question="What is semantic retrieval?",
            expected_file_name="README.md",
            expected_text="semantic search",
        ),
        RetrievalExample(
            question="How do I inspect chunks?",
            expected_file_name="README.md",
            expected_text="retrieve",
        ),
    ]


def test_load_retrieval_examples_rejects_non_markdown_file(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 binary content")

    with pytest.raises(RetrievalEvaluationError, match="expects a UTF-8 markdown examples file"):
        load_retrieval_examples(pdf_path)


def test_load_retrieval_examples_reports_missing_examples(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.md"
    examples_path.write_text("# No table here\n", encoding="utf-8")

    with pytest.raises(RetrievalEvaluationError, match="No retrieval examples found"):
        load_retrieval_examples(examples_path)
