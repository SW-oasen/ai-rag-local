from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import rag_assistant.document_loader as document_loader
from rag_assistant.document_loader import OcrOptions, clean_ocr_text, load_document, load_documents


def test_load_text_document_preserves_metadata(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Local RAG notes", encoding="utf-8")

    documents = load_document(source)

    assert len(documents) == 1
    assert documents[0].text == "Local RAG notes"
    assert documents[0].source_path == source
    assert documents[0].file_name == "notes.txt"
    assert documents[0].document_type == "txt"
    assert documents[0].page_number is None


def test_load_documents_reads_supported_files_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "readme.md").write_text("# Project", encoding="utf-8")
    (nested / "notes.txt").write_text("Chunk me", encoding="utf-8")
    (nested / "ignored.csv").write_text("name,value", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.file_name for document in documents] == ["readme.md", "notes.txt"]


def test_load_document_rejects_unsupported_file(tmp_path: Path) -> None:
    source = tmp_path / "table.csv"
    source.write_text("name,value", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(source)


def test_load_pdf_uses_ocr_for_empty_pages_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF fake")

    class FakePage:
        def extract_text(self) -> str:
            return ""

    class FakePdfReader:
        def __init__(self, path: str) -> None:
            self.pages = [FakePage()]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakePdfReader))
    monkeypatch.setattr(document_loader, "_ocr_pdf_page", lambda path, page_index, options: "OCR text")

    documents = load_document(
        source,
        ocr_options=OcrOptions(enabled=True, language="eng+deu", scale=3.5, psm=4),
    )

    assert len(documents) == 1
    assert documents[0].text == "OCR text"
    assert documents[0].metadata["ocr_used"] is True
    assert documents[0].metadata["ocr_language"] == "eng+deu"
    assert documents[0].metadata["ocr_scale"] == 3.5
    assert documents[0].metadata["ocr_psm"] == 4


def test_clean_ocr_text_joins_hyphenated_and_wrapped_lines() -> None:
    text = "This is a hyphen-\nated word\nand wrapped line.\n\n\nNext paragraph."

    cleaned = clean_ocr_text(text)

    assert "hyphenated" in cleaned
    assert "word and wrapped line." in cleaned
    assert "\n\n\n" not in cleaned
