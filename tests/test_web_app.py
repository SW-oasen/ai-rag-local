from pathlib import Path

from rag_assistant.schema import IndexedSource, RagAnswer, RetrievalResult, SourceReference, SummaryResult, TextChunk
from rag_assistant.schema import Document
from rag_assistant.web_app import (
    build_parser,
    render_extracted_documents,
    render_page,
    render_retrieval_results,
    render_summary,
)


def test_build_parser_accepts_ui_options() -> None:
    args = build_parser().parse_args(["--port", "9000", "--vector-store", "store", "--llm-model", "local"])

    assert args.port == 9000
    assert args.vector_store == Path("store")
    assert args.llm_model == "local"


def test_render_page_includes_sources_and_form() -> None:
    html = render_page(
        sources=[
            IndexedSource(
                file_name="example.pdf",
                source_path=Path("docs/example.pdf"),
                document_type="pdf",
                chunk_count=3,
                page_count=2,
            )
        ],
        question="What is this about?",
        selected_source="example.pdf",
    )

    assert "Local RAG Assistant" in html
    assert "What is this about?" in html
    assert "example.pdf" in html
    assert 'formaction="/retrieve"' in html
    assert 'formaction="/ask"' in html
    assert 'formaction="/summarize"' in html
    assert 'action="/extract-text"' in html
    assert 'name="ocr_language"' in html
    assert 'name="ocr_scale"' in html
    assert 'name="ocr_psm"' in html
    assert 'name="ocr_preprocess"' in html
    assert 'name="ocr_clean_text"' in html


def test_render_retrieval_results_escapes_chunk_text() -> None:
    html = render_retrieval_results(
        [
            RetrievalResult(
                chunk=TextChunk(
                    text="<script>alert('x')</script> relevant text",
                    source_path=Path("docs/example.md"),
                    file_name="example.md",
                    document_type="md",
                    chunk_index=1,
                    start_char=0,
                    end_char=40,
                ),
                score=0.25,
            )
        ]
    )

    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "score 0.2500" in html


def test_render_page_includes_answer_sources() -> None:
    html = render_page(
        answer=RagAnswer(
            answer="Use local context.",
            sources=[
                SourceReference(
                    file_name="example.md",
                    source_path=Path("docs/example.md"),
                    chunk_index=2,
                    page_number=4,
                )
            ],
            retrieved_chunks=[],
            model="fake",
            prompt="prompt",
        )
    )

    assert "Use local context." in html
    assert "example.md, chunk 2, page 4" in html


def test_render_summary_includes_sources_and_partial_count() -> None:
    html = render_summary(
        SummaryResult(
            summary="This is a safe summary.",
            sources=[
                SourceReference(
                    file_name="example.md",
                    source_path=Path("docs/example.md"),
                    chunk_index=1,
                    page_number=2,
                )
            ],
            source_chunks=[],
            model="fake",
            partial_summaries=["partial"],
        )
    )

    assert "This is a safe summary." in html
    assert "example.md, chunk 1, page 2" in html
    assert "Partial summaries: 1" in html


def test_render_extracted_documents_escapes_text_and_marks_ocr() -> None:
    html = render_extracted_documents(
        [
            Document(
                text="<script>alert('x')</script> OCR text",
                source_path=Path("docs/scan.pdf"),
                file_name="scan.pdf",
                document_type="pdf",
                page_number=1,
                metadata={"ocr_used": True},
            )
        ]
    )

    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "scan.pdf, page 1" in html
    assert "OCR" in html
