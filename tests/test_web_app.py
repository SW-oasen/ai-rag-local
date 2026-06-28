from pathlib import Path

from rag_assistant.schema import IndexedSource, RagAnswer, RetrievalResult, SourceReference, SummaryResult, TextChunk
from rag_assistant.schema import Document
from rag_assistant.library_store import CachedSummary, ConfiguredPath
from rag_assistant.profile_store import RagProfile
from rag_assistant.config import PROJECT_ROOT
from rag_assistant.web_app import (
    _extracted_text_file_name,
    _with_profile_metadata,
    build_parser,
    format_cached_summary,
    format_extracted_text,
    format_markdown_html,
    render_extracted_documents,
    render_page,
    render_retrieval_results,
    render_summary,
    render_summary_job_progress,
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


def test_render_storage_paths_are_relative_inside_project() -> None:
    html = render_page(
        active_page="configuration",
        vector_store_path=PROJECT_ROOT / "vector_store",
        library_store_path=PROJECT_ROOT / "data" / "processed" / "web_library.json",
    )

    assert str(PROJECT_ROOT) not in html
    assert "vector_store" in html
    assert str(Path("data") / "processed" / "web_library.json") in html


def test_render_ask_page_includes_question_form() -> None:
    html = render_page(
        active_page="ask",
        question="What is this about?",
        profiles=[
            RagProfile(name="general", prompt_style="general"),
            RagProfile(name="technical", prompt_style="technical"),
            RagProfile(name="recipes", prompt_style="recipes"),
        ],
        selected_profile="technical",
    )

    assert "What is this about?" in html
    assert 'formaction="/retrieve"' in html
    assert 'formaction="/ask"' in html
    assert '<select name="profile">' in html
    assert '<option value="general">general</option>' in html
    assert '<option value="technical" selected>technical</option>' in html
    assert '<option value="recipes">recipes</option>' in html


def test_render_extract_page_includes_ocr_controls() -> None:
    html = render_page(active_page="extract-text")

    assert 'action="/extract-text"' in html
    assert 'formaction="/extract-text-export"' in html
    assert '<select name="ocr_language">' in html
    assert '<option value="eng" selected>English (eng)</option>' in html
    assert '<option value="eng+deu">English + German (eng+deu)</option>' in html
    assert '<option value="fra">French (fra)</option>' in html
    assert '<option value="chi_sim">Chinese Simplified (chi_sim)</option>' in html
    assert '<option value="chi_tra">Chinese Traditional (chi_tra)</option>' in html
    assert 'name="use_ocr" type="checkbox" checked' in html
    assert "Enable OCR for scanned PDFs or image-only documents." in html
    assert "OCR language uses installed Tesseract codes" in html
    assert 'name="ocr_scale"' in html
    assert 'name="ocr_psm"' in html
    assert 'name="ocr_preprocess"' in html
    assert 'name="ocr_clean_text"' in html


def test_render_summary_page_includes_language_selector() -> None:
    html = render_page(active_page="summarize", summary_language="fra")

    assert '<select name="summary_language">' in html
    assert '<option value="auto">Same as source (auto)</option>' in html
    assert '<option value="fra" selected>French (fra)</option>' in html


def test_render_configuration_page_includes_index_management_actions() -> None:
    html = render_page(
        active_page="configuration",
        sources=[
            IndexedSource(
                file_name="example.pdf",
                source_path=Path("docs/example.pdf"),
                document_type="pdf",
                chunk_count=2,
                page_count=1,
            )
        ],
    )

    assert 'action="/configuration/delete-source"' in html
    assert 'action="/configuration/reset-index"' in html
    assert "Delete Source" in html
    assert "Reset Vector Index" in html


def test_render_configuration_page_includes_profile_management() -> None:
    html = render_page(
        active_page="configuration",
        profiles=[
            RagProfile(name="general", prompt_style="general"),
            RagProfile(
                name="technical",
                prompt_style="technical",
                paths=("data/raw/local-docus/tech",),
                chunk_size=800,
                chunk_overlap=100,
            ),
        ],
    )

    assert "<h2>Profiles</h2>" in html
    assert "<td>technical</td>" in html
    assert "<td>technical</td>" in html
    assert "<td>800/100</td>" in html
    assert "data/raw/local-docus/tech" in html
    assert 'action="/configuration/add-profile"' in html
    assert 'action="/configuration/add-profile-path"' in html
    assert 'action="/configuration/ingest-profile-path"' in html
    assert 'action="/configuration/remove-profile-path"' in html
    assert "Add Profile" in html
    assert "Add Profile Path" in html


def test_render_profiles_configuration_shows_profile_specific_path_status(tmp_path: Path) -> None:
    tech_folder = tmp_path / "tech"
    tech_folder.mkdir()
    (tech_folder / "api.md").write_text("api", encoding="utf-8")
    (tech_folder / "guide.md").write_text("guide", encoding="utf-8")

    html = render_page(
        active_page="configuration",
        profiles=[
            RagProfile(
                name="technical",
                prompt_style="technical",
                paths=(str(tech_folder),),
            ),
            RagProfile(
                name="recipes",
                prompt_style="recipes",
                paths=(str(tech_folder),),
            ),
        ],
        profile_sources={
            "technical": [
                IndexedSource(
                    file_name="api.md",
                    source_path=tech_folder / "api.md",
                    document_type="md",
                    chunk_count=3,
                )
            ],
            "recipes": [],
        },
    )

    assert "Partially indexed folder (1/2 sources, 3 chunks)" in html
    assert "Not indexed (2 supported files)" in html
    assert "Ingest Missing" in html


def test_with_profile_metadata_marks_chunks_for_web_ingest() -> None:
    chunks = [
        TextChunk(
            text="Tech text.",
            source_path=Path("docs/tech.md"),
            file_name="tech.md",
            document_type="md",
            chunk_index=0,
            start_char=0,
            end_char=10,
        )
    ]

    profiled_chunks = _with_profile_metadata(chunks, RagProfile(name="technical", prompt_style="technical"))

    assert profiled_chunks[0].metadata["profile"] == "technical"
    assert profiled_chunks[0].metadata["prompt_style"] == "technical"
    assert chunks[0].metadata == {}


def test_render_configuration_paths_show_actions_by_index_status() -> None:
    html = render_page(
        active_page="configuration",
        configured_paths=[
            ConfiguredPath(path=str(Path("docs") / "example.pdf")),
            ConfiguredPath(path="docs"),
            ConfiguredPath(path=str(Path("docs") / "new.pdf")),
        ],
        sources=[
            IndexedSource(
                file_name="example.pdf",
                source_path=Path("docs/example.pdf"),
                document_type="pdf",
                chunk_count=2,
                page_count=1,
            )
        ],
    )

    assert "Indexed (2 chunks, pages 1)" in html
    assert "Contains 1 indexed source (2 chunks)" in html
    assert "Not indexed" in html
    assert "Delete Index" in html
    assert "Ingest Updates" in html
    assert ">Ingest</button>" in html
    assert "Remove Path" in html


def test_render_configuration_folder_status_when_all_supported_files_are_indexed(tmp_path: Path) -> None:
    folder = tmp_path / "smard"
    folder.mkdir()
    (folder / "smard_api.md").write_text("api", encoding="utf-8")
    (folder / "smard_guide.pdf").write_bytes(b"%PDF")

    html = render_page(
        active_page="configuration",
        configured_paths=[ConfiguredPath(path=str(folder))],
        sources=[
            IndexedSource(
                file_name="smard_api.md",
                source_path=folder / "smard_api.md",
                document_type="md",
                chunk_count=4,
                page_count=None,
            ),
            IndexedSource(
                file_name="smard_guide.pdf",
                source_path=folder / "smard_guide.pdf",
                document_type="pdf",
                chunk_count=215,
                page_count=40,
            ),
        ],
    )

    assert "Indexed folder (2 sources, 219 chunks)" in html
    assert "Re-ingest Folder" in html
    assert "Contains 2 indexed sources" not in html


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
            answer="## **Result**\n\n- Use **local context**.\n- Keep sources. <script>",
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

    assert "<h3><strong>Result</strong></h3>" in html
    assert "<li>Use <strong>local context</strong>.</li>" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "example.md, chunk 2, page 4" in html


def test_format_markdown_html_renders_basic_blocks_safely() -> None:
    html = format_markdown_html(
        "#### Steps\n\n"
        "- First\n"
        "- **Second**\n\n"
        "1. Ordered\n"
        "2. Also **ordered**\n\n"
        "**Bold <script>** stays safe.\n\n"
        "```text\n<x>\n```"
    )

    assert "<h4>Steps</h4>" in html
    assert "#### Steps" not in html
    assert "<li>First</li>" in html
    assert "<li><strong>Second</strong></li>" in html
    assert "<ol><li>Ordered</li><li>Also <strong>ordered</strong></li></ol>" in html
    assert "<strong>Bold &lt;script&gt;</strong> stays safe." in html
    assert "<pre><code>&lt;x&gt;</code></pre>" in html
    assert "<script>" not in html


def test_format_markdown_html_groups_colon_sections_as_indented_items() -> None:
    html = format_markdown_html(
        "Suppen und Eintöpfe:\n"
        "Fischsuppe mit picklten Mustardgräsern [source 4].\n"
        "Li Zhuang \"Head Bowl\"-Koch mit Fleisch [source 4].\n"
        "Spicy Blood Stew [source 4]."
    )

    assert '<div class="markdown-group">' in html
    assert '<p class="markdown-group-title">Suppen und Eintöpfe:</p>' in html
    assert "<li>Fischsuppe mit picklten Mustardgräsern [source 4].</li>" in html
    assert '<li>Li Zhuang &quot;Head Bowl&quot;-Koch mit Fleisch [source 4].</li>' in html
    assert "<li>Spicy Blood Stew [source 4].</li>" in html


def test_format_markdown_html_nests_bullets_under_ordered_items() -> None:
    html = format_markdown_html(
        "1. **Energy Conversion:**\n\n"
        "- The brain uses energy from food [source 2].\n"
        "- Heat is a disordered form of energy [source 2].\n\n"
        "1. **Entropy Increase:**\n\n"
        "- This aligns with the **second law** [source 1]."
    )

    assert (
        "<ol>"
        "<li><strong>Energy Conversion:</strong>"
        "<ul>"
        "<li>The brain uses energy from food [source 2].</li>"
        "<li>Heat is a disordered form of energy [source 2].</li>"
        "</ul>"
        "</li>"
        "<li><strong>Entropy Increase:</strong>"
        "<ul><li>This aligns with the <strong>second law</strong> [source 1].</li></ul>"
        "</li>"
        "</ol>"
    ) in html


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


def test_render_summary_job_progress_polls_status_endpoint() -> None:
    html = render_summary_job_progress("job-123")

    assert 'data-summary-job="job-123"' in html
    assert "/summary-progress?job_id=" in html
    assert "Summary Progress" in html
    assert 'id="summary-job-start"' in html
    assert 'id="summary-job-current"' in html
    assert "source_name" in html


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


def test_render_extracted_documents_warns_when_all_text_is_empty() -> None:
    html = render_extracted_documents(
        [
            Document(
                text="",
                source_path=Path("docs/scan.pdf"),
                file_name="scan.pdf",
                document_type="pdf",
                page_number=1,
                metadata={"ocr_used": True},
            )
        ]
    )

    assert "No extractable text was found." in html
    assert "check the OCR language" in html


def test_format_extracted_text_includes_source_path_once() -> None:
    text = format_extracted_text(
        [
            Document(
                text="First page text.",
                source_path=Path("docs/scan.pdf"),
                file_name="scan.pdf",
                document_type="pdf",
                page_number=1,
            ),
            Document(
                text="Second page text.",
                source_path=Path("docs/scan.pdf"),
                file_name="scan.pdf",
                document_type="pdf",
                page_number=2,
            ),
        ]
    )

    assert text.count(f"Source: {Path('docs/scan.pdf')}") == 1
    assert "Path:" not in text
    assert "scan.pdf, page" not in text
    assert "# Page 1" in text
    assert "# Page 2" in text


def test_extracted_text_file_name_uses_source_stem() -> None:
    file_name = _extracted_text_file_name(
        [
            Document(
                text="Export me.",
                source_path=Path("docs/scan.pdf"),
                file_name="scan.pdf",
                document_type="pdf",
            )
        ]
    )

    assert file_name == "scan-extracted.txt"
