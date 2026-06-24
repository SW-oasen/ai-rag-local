from pathlib import Path

from rag_assistant.cli import (
    build_parser,
    format_evaluation_results,
    format_indexed_sources,
    format_rag_answer,
    format_retrieval_results,
    format_source_chunks,
    format_summary_result,
    main,
    write_evaluation_json_report,
)
from rag_assistant.evaluation import RetrievalEvaluationResult
from rag_assistant.schema import IndexedSource, RagAnswer, RetrievalResult, SourceReference, SummaryResult, TextChunk


def test_build_parser_accepts_ingest_command() -> None:
    args = build_parser().parse_args(
        [
            "ingest",
            "data/raw",
            "--chunk-size",
            "500",
            "--ocr",
            "--ocr-language",
            "eng+deu",
            "--ocr-scale",
            "3.5",
            "--ocr-psm",
            "4",
        ]
    )

    assert args.path == Path("data/raw")
    assert args.chunk_size == 500
    assert args.ocr is True
    assert args.ocr_language == "eng+deu"
    assert args.ocr_scale == 3.5
    assert args.ocr_psm == 4


def test_format_retrieval_results_includes_source_and_score() -> None:
    result = RetrievalResult(
        chunk=TextChunk(
            text="Local context text.",
            source_path=Path("docs/example.md"),
            file_name="example.md",
            document_type="md",
            chunk_index=3,
            start_char=0,
            end_char=19,
        ),
        score=0.25,
    )

    output = format_retrieval_results([result])

    assert "example.md | chunk 3 | score 0.2500" in output
    assert "Path: docs\\example.md" in output or "Path: docs/example.md" in output
    assert "Local context text." in output


def test_build_parser_accepts_retrieve_source_filter() -> None:
    args = build_parser().parse_args(["retrieve", "cats", "--source", "docs/story.pdf"])

    assert args.question == "cats"
    assert args.source == Path("docs/story.pdf")


def test_build_parser_accepts_sources_command() -> None:
    args = build_parser().parse_args(["sources", "--collection", "docs"])

    assert args.collection == "docs"


def test_format_indexed_sources_includes_counts_and_paths() -> None:
    output = format_indexed_sources(
        [
            IndexedSource(
                file_name="example.pdf",
                source_path=Path("docs/example.pdf"),
                document_type="pdf",
                chunk_count=12,
                page_count=4,
            )
        ],
        total_chunks=12,
    )

    assert "Indexed sources: 1" in output
    assert "Total chunks: 12" in output
    assert "[1] example.pdf | pdf | chunks 12, pages 4" in output
    assert "Path: docs\\example.pdf" in output or "Path: docs/example.pdf" in output


def test_build_parser_accepts_chunks_command() -> None:
    args = build_parser().parse_args(["chunks", "docs/example.pdf", "--limit", "3"])

    assert args.source == Path("docs/example.pdf")
    assert args.limit == 3


def test_format_source_chunks_includes_preview_and_limit() -> None:
    chunks = [
        TextChunk(
            text="First chunk text with enough words for preview.",
            source_path=Path("docs/example.pdf"),
            file_name="example.pdf",
            document_type="pdf",
            chunk_index=0,
            start_char=0,
            end_char=45,
            page_number=1,
        ),
        TextChunk(
            text="Second chunk text.",
            source_path=Path("docs/example.pdf"),
            file_name="example.pdf",
            document_type="pdf",
            chunk_index=1,
            start_char=46,
            end_char=64,
            page_number=2,
        ),
    ]

    output = format_source_chunks(chunks, source="example.pdf", limit=1, preview_chars=18)

    assert "Chunks for source: example.pdf" in output
    assert "Total chunks: 2" in output
    assert "Showing: 1" in output
    assert "[0] example.pdf, page 1 | chars 0-45" in output
    assert "First chunk tex..." in output
    assert "Second chunk text." not in output


def test_format_rag_answer_includes_sources_and_optional_prompt() -> None:
    answer = RagAnswer(
        answer="Use the retrieved context. [source 1]",
        sources=[
            SourceReference(
                file_name="example.md",
                source_path=Path("docs/example.md"),
                chunk_index=0,
                score=0.1,
            )
        ],
        retrieved_chunks=[],
        model="fake-model",
        prompt="Question prompt",
    )

    output = format_rag_answer(answer, show_prompt=True)

    assert "Use the retrieved context. [source 1]" in output
    assert "- [1] example.md, chunk 0, score 0.1000" in output
    assert "Prompt:\nQuestion prompt" in output


def test_build_parser_accepts_ask_source_filter() -> None:
    args = build_parser().parse_args(["ask", "cats", "--source", "story.pdf"])

    assert args.question == "cats"
    assert args.source == Path("story.pdf")


def test_build_parser_accepts_summarize_command() -> None:
    args = build_parser().parse_args(["summarize", "README.md", "--question", "main idea"])

    assert args.source == Path("README.md")
    assert args.question == "main idea"
    assert args.from_index is False


def test_build_parser_accepts_eval_command() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "examples/retrieval_eval_examples.md",
            "--top-k",
            "5",
            "--source",
            "README.md",
            "--json-report",
            "reports/eval.json",
        ]
    )

    assert args.examples == Path("examples/retrieval_eval_examples.md")
    assert args.top_k == 5
    assert args.source == Path("README.md")
    assert args.json_report == Path("reports/eval.json")


def test_format_summary_result_includes_sources_and_partial_count() -> None:
    summary = SummaryResult(
        summary="Final summary.",
        sources=[
            SourceReference(
                file_name="example.md",
                source_path=Path("docs/example.md"),
                chunk_index=1,
            )
        ],
        source_chunks=[],
        model="fake-model",
        partial_summaries=["one", "two"],
    )

    output = format_summary_result(summary)

    assert "Final summary." in output
    assert "- [1] example.md, chunk 1" in output
    assert "Partial summaries: 2" in output


def test_format_evaluation_results_includes_pass_count_and_failures() -> None:
    output = format_evaluation_results(
        [
            RetrievalEvaluationResult(
                question="What is semantic retrieval?",
                passed=True,
                top_file_name="README.md",
                expected_file_name="README.md",
                expected_text="semantic search",
                matched_file_name="README.md",
                matched_chunk_index=2,
                matched_page_number=None,
                matched_score=0.12,
            ),
            RetrievalEvaluationResult(
                question="Where is the guide?",
                passed=False,
                top_file_name="notes.md",
                expected_file_name="README.md",
                expected_text="guide",
                top_chunk_index=4,
                top_page_number=7,
                top_score=0.9,
            ),
        ]
    )

    assert "Retrieval evaluation: 1/2 passed" in output
    assert "[1] PASS | expected README.md | top result README.md" in output
    assert "match: README.md, chunk 2, score 0.1200" in output
    assert "[2] FAIL | expected README.md | top result notes.md" in output
    assert "top: notes.md, chunk 4, page 7, score 0.9000" in output


def test_write_evaluation_json_report(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "eval.json"

    write_evaluation_json_report(
        [
            RetrievalEvaluationResult(
                question="What is semantic retrieval?",
                passed=True,
                top_file_name="README.md",
                expected_file_name="README.md",
                expected_text="semantic search",
                matched_file_name="README.md",
                matched_chunk_index=2,
                matched_score=0.12,
            )
        ],
        report_path,
    )

    output = report_path.read_text(encoding="utf-8")
    assert '"question": "What is semantic retrieval?"' in output
    assert '"matched_chunk_index": 2' in output


def test_eval_command_reports_document_path_error(tmp_path: Path, capsys) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 binary content")

    exit_code = main(["eval", str(pdf_path)])

    output = capsys.readouterr().out
    assert exit_code == 4
    assert "Evaluation error:" in output
    assert "expects a UTF-8 markdown examples file" in output
