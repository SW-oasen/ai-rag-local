from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import zipfile

import pytest

import rag_assistant.document_loader as document_loader
from rag_assistant.document_loader import OcrOptions, clean_ocr_text, load_document, load_documents
from rag_assistant.schema import Document


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


def test_load_text_document_reads_cp1252_french_text(tmp_path: Path) -> None:
    source = tmp_path / "francais.txt"
    source.write_bytes("Le comprimé était déjà prêt.".encode("cp1252"))

    documents = load_document(source)

    assert documents[0].text == "Le comprimé était déjà prêt."


def test_load_documents_reads_supported_files_recursively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "readme.md").write_text("# Project", encoding="utf-8")
    (nested / "notes.txt").write_text("Chunk me", encoding="utf-8")
    (nested / "book.epub").write_bytes(b"not a real epub")
    (nested / "ignored.csv").write_text("name,value", encoding="utf-8")

    monkeypatch.setattr(
        document_loader,
        "load_document",
        lambda path, **kwargs: [
            Document(
                text=Path(path).name,
                source_path=Path(path),
                file_name=Path(path).name,
                document_type=Path(path).suffix.lower().lstrip("."),
            )
        ],
    )

    documents = load_documents(tmp_path)

    assert [document.file_name for document in documents] == ["readme.md", "book.epub", "notes.txt"]


def test_load_document_rejects_unsupported_file(tmp_path: Path) -> None:
    source = tmp_path / "table.csv"
    source.write_text("name,value", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(source)


def test_load_epub_document_extracts_spine_text(tmp_path: Path) -> None:
    source = tmp_path / "book.epub"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles><rootfile full-path="OEBPS/package.opf"/></rootfiles>
            </container>""",
        )
        archive.writestr(
            "OEBPS/package.opf",
            """<?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf">
              <manifest>
                <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
                <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="chapter1"/><itemref idref="chapter2"/></spine>
            </package>""",
        )
        archive.writestr("OEBPS/chapter1.xhtml", "<html><body><h1>Start</h1><p>First chapter.</p></body></html>")
        archive.writestr("OEBPS/chapter2.xhtml", "<html><body><p>Second chapter.</p></body></html>")

    documents = load_document(source)

    assert len(documents) == 1
    assert documents[0].document_type == "epub"
    assert "Start" in documents[0].text
    assert documents[0].text.index("First chapter.") < documents[0].text.index("Second chapter.")


def test_load_open_document_extracts_content_xml_text(tmp_path: Path) -> None:
    source = tmp_path / "notes.odt"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "content.xml",
            """<?xml version="1.0"?>
            <office:document-content
                xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
              <office:body><office:text>
                <text:h>Heading</text:h>
                <text:p>Paragraph <text:span>with span</text:span>.</text:p>
              </office:text></office:body>
            </office:document-content>""",
        )

    documents = load_document(source)

    assert len(documents) == 1
    assert documents[0].document_type == "odt"
    assert "Heading" in documents[0].text
    assert "Paragraph with span." in documents[0].text


def test_load_azw3_document_uses_calibre_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "book.azw3"
    source.write_bytes(b"AZW3")

    def fake_run(command, capture_output, text, check):
        output_path = Path(command[2])
        output_path.write_text("Converted AZW3 text", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(document_loader.shutil, "which", lambda name: "ebook-convert.exe")
    monkeypatch.setattr(document_loader.subprocess, "run", fake_run)

    documents = load_document(source)

    assert len(documents) == 1
    assert documents[0].document_type == "azw3"
    assert documents[0].text == "Converted AZW3 text"


def test_load_azw3_document_requires_calibre(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "book.azw3"
    source.write_bytes(b"AZW3")
    monkeypatch.setattr(document_loader.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ebook-convert"):
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


def test_load_pdf_appends_pdfplumber_tables_as_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "table.pdf"
    source.write_bytes(b"%PDF fake")

    class FakePdfPage:
        def extract_text(self) -> str:
            return "Page text"

    class FakePdfReader:
        def __init__(self, path: str) -> None:
            self.pages = [FakePdfPage()]

    class FakePlumberPage:
        def extract_tables(self):
            return [[["Name", "Value"], ["alpha", "one|two"], ["beta", None]]]

    class FakePlumberPdf:
        pages = [FakePlumberPage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakePdfReader))
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: FakePlumberPdf()))

    documents = load_document(source)

    assert len(documents) == 1
    assert "Page text" in documents[0].text
    assert "Table 1" in documents[0].text
    assert "| Name | Value |" in documents[0].text
    assert "| alpha | one\\|two |" in documents[0].text
    assert documents[0].metadata["pdf_tables_extracted"] == 1
    assert documents[0].metadata["pdf_tables_error"] == ""


def test_clean_ocr_text_joins_hyphenated_and_wrapped_lines() -> None:
    text = "This is a hyphen-\nated word\nand wrapped line.\n\n\nNext paragraph."

    cleaned = clean_ocr_text(text)

    assert "hyphenated" in cleaned
    assert "word and wrapped line." in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_ocr_text_fixes_isolated_pipe_pronouns() -> None:
    text = (
        '"No. | feel fine."\n'
        '"| don\'t know."\n'
        '"Everything happened and | didn\'t worry."\n'
        '"| hate my life. | promised Silvia."\n'
        '"Can |?"'
    )

    cleaned = clean_ocr_text(text)

    assert '"No. I feel fine."' in cleaned
    assert '"I don\'t know."' in cleaned
    assert "and I didn't worry" in cleaned
    assert '"I hate my life. I promised Silvia."' in cleaned
    assert '"Can I?"' in cleaned


def test_clean_ocr_text_fixes_common_ocr_letter_substitutions() -> None:
    text = (
        '"T think it does!"\n'
        '"] want to get tested."\n'
        '"\u201c] don\u2019t know."\n'
        '"Who 1s this? What 1f 1t works 1n time?"'
    )

    cleaned = clean_ocr_text(text)

    assert '"I think it does!"' in cleaned
    assert '"I want to get tested."' in cleaned
    assert '"\u201cI don\u2019t know."' in cleaned
    assert "Who is this? What if it works in time?" in cleaned


def test_clean_ocr_text_fixes_ocr_ill_contractions() -> None:
    text = "[Il tell my parents.\nIll probably ask later.\nIll health is different."

    cleaned = clean_ocr_text(text)

    assert "I'll tell my parents." in cleaned
    assert "I'll probably ask later." in cleaned
    assert "Ill health is different." in cleaned


def test_clean_ocr_text_keeps_unrelated_single_letters_and_numbers() -> None:
    text = "Section T starts now.\nUse channel 1 and item 1a.\nName | Value"

    cleaned = clean_ocr_text(text)

    assert "Section T starts now." in cleaned
    assert "channel 1 and item 1a" in cleaned
    assert "Name | Value" in cleaned


def test_clean_ocr_text_keeps_table_or_separator_pipes() -> None:
    text = "Name | Value\nA | B\nUse pipes | as separators."

    cleaned = clean_ocr_text(text)

    assert "Name | Value" in cleaned
    assert "A | B" in cleaned
    assert "pipes | as" in cleaned
