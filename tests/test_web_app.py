from pathlib import Path

from rag_assistant.schema import IndexedSource, RagAnswer, RetrievalResult, SourceReference, SummaryResult, TextChunk
from rag_assistant.schema import Document
from rag_assistant.library_store import CachedSummary
from rag_assistant.web_app import (
    build_parser,
    format_cached_summary,
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
    assert args.library_store.name == "web_library.json"


def test_render_overview_includes_navigation_and_sources() -> None:
    html = render_page(
        active_page="overview",
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
    assert "example.pdf" in html
    assert 'href="/ask"' in html
    assert 'href="/summarize"' in html
    assert 'href="/extract-text"' in html


def test_render_ask_page_includes_question_form() -> None:
    html = render_page(active_page="ask", question="What is this about?")

    assert "What is this about?" in html
    assert 'formaction="/retrieve"' in html
    assert 'formaction="/ask"' in html


def test_render_extract_page_includes_ocr_controls() -> None:
    html = render_page(active_page="extract-text")

    assert 'action="/extract-text"' in html
    assert 'formaction="/extract-text-export"' in html
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
        active_page="ask",
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


def test_format_cached_summary_as_markdown() -> None:
    text = format_cached_summary(
        CachedSummary(
            source_path="docs/example.md",
            file_name="example.md",
            summary="A useful summary.",
            model="fake",
            source_count=2,
            partial_summary_count=1,
        ),
        export_format="md",
    )

    assert "# Summary: example.md" in text
    assert "- Model: fake" in text
    assert "A useful summary." in text


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
