"""Document loading utilities for local RAG ingestion."""

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote
import zipfile
import xml.etree.ElementTree as ET

from rag_assistant.schema import Document

EPUB_EXTENSIONS = {".epub"}
OPEN_DOCUMENT_EXTENSIONS = {
    ".odt",
    ".ods",
    ".odp",
    ".odg",
    ".odf",
    ".ott",
    ".ots",
    ".otp",
    ".otg",
    ".sxw",
    ".sxc",
    ".sxi",
    ".sxd",
    ".stw",
    ".stc",
    ".sti",
    ".std",
}
FLAT_OPEN_DOCUMENT_EXTENSIONS = {".fodt", ".fods", ".fodp", ".fodg"}
SUPPORTED_EXTENSIONS = (
    {".md", ".txt", ".pdf", ".azw3"}
    | EPUB_EXTENSIONS
    | OPEN_DOCUMENT_EXTENSIONS
    | FLAT_OPEN_DOCUMENT_EXTENSIONS
)
HTML_BLOCK_TAGS = {
    "article",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "tr",
}


class OcrError(RuntimeError):
    """Raised when OCR text extraction cannot run."""


@dataclass(frozen=True)
class OcrOptions:
    """OCR extraction and cleanup options."""

    enabled: bool = False
    language: str = "eng"
    scale: float = 3.0
    psm: int = 6
    preprocess: bool = True
    clean_text: bool = True


def load_documents(
    path: str | Path,
    use_ocr: bool = False,
    ocr_language: str = "eng",
    ocr_options: OcrOptions | None = None,
) -> list[Document]:
    """Load supported documents from a file or directory path."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Document path does not exist: {source}")

    files = [source] if source.is_file() else _iter_supported_files(source)
    documents: list[Document] = []
    options = _resolve_ocr_options(use_ocr, ocr_language, ocr_options)

    for file_path in files:
        documents.extend(load_document(file_path, ocr_options=options))

    return documents


def load_document(
    path: str | Path,
    use_ocr: bool = False,
    ocr_language: str = "eng",
    ocr_options: OcrOptions | None = None,
) -> list[Document]:
    """Load one supported document file."""

    file_path = Path(path)
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {file_path.suffix}")

    if extension in {".md", ".txt"}:
        return [_load_text_document(file_path)]

    if extension in EPUB_EXTENSIONS:
        return [_load_epub_document(file_path)]

    if extension in OPEN_DOCUMENT_EXTENSIONS:
        return [_load_open_document(file_path)]

    if extension in FLAT_OPEN_DOCUMENT_EXTENSIONS:
        return [_load_flat_open_document(file_path)]

    if extension == ".azw3":
        return [_load_azw3_document(file_path)]

    options = _resolve_ocr_options(use_ocr, ocr_language, ocr_options)
    return _load_pdf_document(file_path, ocr_options=options)


def _iter_supported_files(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda path: (len(path.relative_to(directory).parts), path.name.lower()))


def _load_text_document(path: Path) -> Document:
    text = _read_plain_text(path)
    return Document(
        text=text,
        source_path=path,
        file_name=path.name,
        document_type=path.suffix.lower().lstrip("."),
    )


def _read_plain_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_epub_document(path: Path) -> Document:
    with zipfile.ZipFile(path) as archive:
        html_paths = _epub_spine_paths(archive) or _epub_html_paths(archive)
        text_parts = [_html_to_text(_read_zip_text(archive, html_path)) for html_path in html_paths]

    return _document_from_text(path, "\n\n".join(part for part in text_parts if part.strip()))


def _epub_spine_paths(archive: zipfile.ZipFile) -> list[str]:
    try:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
    except (KeyError, ET.ParseError):
        return []

    rootfile = container.find(".//{*}rootfile")
    if rootfile is None:
        return []

    opf_path = rootfile.attrib.get("full-path")
    if not opf_path:
        return []

    try:
        package = ET.fromstring(archive.read(opf_path))
    except (KeyError, ET.ParseError):
        return []

    manifest: dict[str, str] = {}
    for item in package.findall(".//{*}manifest/{*}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        media_type = item.attrib.get("media-type", "")
        if item_id and href and media_type in {"application/xhtml+xml", "text/html"}:
            manifest[item_id] = _zip_join(opf_path, href)

    paths: list[str] = []
    for itemref in package.findall(".//{*}spine/{*}itemref"):
        href = manifest.get(itemref.attrib.get("idref", ""))
        if href:
            paths.append(href)
    return paths


def _epub_html_paths(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith((".html", ".xhtml", ".htm")) and not name.endswith("/")
    )


def _zip_join(base_file: str, href: str) -> str:
    base_parts = base_file.split("/")[:-1]
    href_parts = unquote(href).split("/")
    parts: list[str] = []
    for part in base_parts + href_parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _read_zip_text(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8", errors="replace")


def _load_open_document(path: Path) -> Document:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_text = _read_zip_text(archive, "content.xml")
    except KeyError as exc:
        raise ValueError(f"OpenDocument file has no content.xml: {path}") from exc

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"OpenDocument content.xml is not readable: {path}") from exc

    return _document_from_text(path, _xml_text_to_plain_text(root))


def _load_flat_open_document(path: Path) -> Document:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError as exc:
        raise ValueError(f"Flat OpenDocument XML is not readable: {path}") from exc

    return _document_from_text(path, _xml_text_to_plain_text(root))


def _load_azw3_document(path: Path) -> Document:
    converter = shutil.which("ebook-convert")
    if converter is None:
        raise RuntimeError(
            "AZW3 loading requires Calibre's 'ebook-convert' command on PATH. "
            "Install Calibre, then retry the same ingest command."
        )

    with tempfile.TemporaryDirectory(prefix="rag-azw3-") as temp_dir:
        output_path = Path(temp_dir) / "book.txt"
        result = subprocess.run(
            [converter, str(path), str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"AZW3 conversion failed for {path}: {message}")
        text = output_path.read_text(encoding="utf-8", errors="replace")

    return _document_from_text(path, text)


def _document_from_text(path: Path, text: str) -> Document:
    return Document(
        text=_normalize_extracted_text(text),
        source_path=path,
        file_name=path.name,
        document_type=path.suffix.lower().lstrip("."),
    )


class _TextExtractingHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag == "br" or tag in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    parser = _TextExtractingHtmlParser()
    parser.feed(html)
    return _normalize_extracted_text("".join(parser.parts))


def _xml_text_to_plain_text(root: ET.Element) -> str:
    parts: list[str] = []
    block_tags = {"p", "h", "list-item", "table-row"}
    tab_tags = {"tab", "table-cell"}
    line_break_tags = {"line-break"}

    def collect(element: ET.Element) -> None:
        local_name = element.tag.rsplit("}", 1)[-1]
        if element.text:
            parts.append(element.text)
        if local_name in tab_tags:
            parts.append("\t")
        if local_name in line_break_tags:
            parts.append("\n")
        for child in element:
            collect(child)
        if element.tail:
            parts.append(element.tail)
        if local_name in block_tags:
            parts.append("\n")

    collect(root)
    return _normalize_extracted_text("".join(parts))


def _normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _load_pdf_document(path: Path, ocr_options: OcrOptions | None = None) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("PDF loading requires the 'pypdf' package.") from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    options = ocr_options or OcrOptions()
    tables_by_page, table_error = _extract_pdf_tables_by_page(path)

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_tables = tables_by_page.get(page_index, [])
        metadata = {
            "ocr_used": False,
            "ocr_language": options.language,
            "ocr_scale": options.scale,
            "ocr_psm": options.psm,
            "ocr_preprocess": options.preprocess,
            "ocr_clean_text": options.clean_text,
            "pdf_tables_extracted": len(page_tables),
            "pdf_tables_error": table_error,
        }
        if options.enabled and not text.strip():
            text = _ocr_pdf_page(path, page_index - 1, options=options)
            metadata["ocr_used"] = True
        if metadata["ocr_used"] and options.clean_text:
            text = clean_ocr_text(text)
        if page_tables:
            table_text = _format_pdf_page_tables(page_tables)
            text = "\n\n".join(part for part in (text.strip(), table_text) if part)
        documents.append(
            Document(
                text=text,
                source_path=path,
                file_name=path.name,
                document_type="pdf",
                page_number=page_index,
                metadata=metadata,
            )
        )

    return documents


def _extract_pdf_tables_by_page(path: Path) -> tuple[dict[int, list[str]], str]:
    try:
        import pdfplumber
    except ImportError:
        return {}, "pdfplumber is not installed"

    tables_by_page: dict[int, list[str]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_tables = []
                for table in page.extract_tables():
                    markdown = _pdf_table_to_markdown(table)
                    if markdown:
                        page_tables.append(markdown)
                if page_tables:
                    tables_by_page[page_index] = page_tables
    except Exception as exc:
        return {}, f"pdfplumber table extraction failed: {exc}"
    return tables_by_page, ""


def _format_pdf_page_tables(tables: list[str]) -> str:
    sections = []
    for table_index, table in enumerate(tables, start=1):
        sections.append(f"Table {table_index}\n\n{table}")
    return "\n\n".join(sections)


def _pdf_table_to_markdown(table: list[list[object]]) -> str:
    rows = [_normalize_pdf_table_row(row) for row in table]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    header = rows[0]
    body = rows[1:]
    if not any(header):
        header = [f"Column {index}" for index in range(1, width + 1)]
        body = rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _normalize_pdf_table_row(row: list[object]) -> list[str]:
    return [_normalize_pdf_table_cell(cell) for cell in row]


def _normalize_pdf_table_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = _normalize_extracted_text(str(cell))
    return text.replace("|", r"\|")


def _ocr_pdf_page(path: Path, page_index: int, options: OcrOptions | None = None) -> str:
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as exc:
        raise OcrError(
            "OCR requires optional Python packages. Install them with "
            "'.\\.venv\\Scripts\\python.exe -m pip install -e .[ocr]'. "
            "You also need the Tesseract OCR application installed and available on PATH."
        ) from exc

    try:
        resolved_options = options or OcrOptions(enabled=True)
        pdf = pdfium.PdfDocument(str(path))
        page = pdf[page_index]
        image = page.render(scale=resolved_options.scale).to_pil()
        if resolved_options.preprocess:
            image = preprocess_ocr_image(image)
        config = f"--psm {resolved_options.psm}"
        return pytesseract.image_to_string(image, lang=resolved_options.language, config=config)
    except Exception as exc:
        raise OcrError(
            f"OCR failed for {path} page {page_index + 1}. "
            "Check that Tesseract is installed and that the requested OCR language is available."
        ) from exc


def preprocess_ocr_image(image):
    """Apply lightweight OCR-friendly image preprocessing."""

    from PIL import ImageFilter, ImageOps

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    thresholded = gray.point(lambda pixel: 255 if pixel > 180 else 0)
    return thresholded.filter(ImageFilter.SHARPEN)


def clean_ocr_text(text: str) -> str:
    """Normalize common OCR whitespace and line-wrap artifacts."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)
    normalized = re.sub(r"(?<![.!?:;])\n(?=\S)", " ", normalized)
    normalized = _fix_common_ocr_substitutions(normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _fix_common_ocr_substitutions(text: str) -> str:
    """Fix conservative OCR substitutions in English prose."""

    fixed = _fix_isolated_pipe_pronouns(text)

    pronoun_followers = (
        "am",
        "can",
        "can't",
        "can\u2019t",
        "cannot",
        "could",
        "couldn't",
        "couldn\u2019t",
        "did",
        "didn't",
        "didn\u2019t",
        "do",
        "don't",
        "don\u2019t",
        "feel",
        "felt",
        "got",
        "had",
        "hadn't",
        "hadn\u2019t",
        "have",
        "haven't",
        "haven\u2019t",
        "keep",
        "kept",
        "know",
        "knew",
        "like",
        "liked",
        "love",
        "loved",
        "need",
        "needed",
        "prefer",
        "remember",
        "should",
        "shouldn't",
        "shouldn\u2019t",
        "think",
        "thought",
        "tried",
        "try",
        "want",
        "wanted",
        "was",
        "wasn't",
        "wasn\u2019t",
        "will",
        "won't",
        "won\u2019t",
        "would",
        "wouldn't",
        "wouldn\u2019t",
    )
    follower_pattern = "|".join(re.escape(word) for word in pronoun_followers)
    quote_or_space = r"[\s\"'\u201c\u2018]"

    fixed = re.sub(
        rf"(^|{quote_or_space})[\|\]T](?=\s+(?:{follower_pattern})\b)",
        r"\1I",
        fixed,
    )
    fixed = _fix_ocr_ill_contractions(fixed)
    fixed = re.sub(rf"(^|{quote_or_space})Tl(?=\s+would\b)", r"\1I", fixed)
    return re.sub(
        r"\b1([fstn])\b",
        lambda match: {"f": "if", "s": "is", "t": "it", "n": "in"}[match.group(1).lower()],
        fixed,
        flags=re.IGNORECASE,
    )


def _fix_ocr_ill_contractions(text: str) -> str:
    contraction_followers = (
        "ask",
        "be",
        "bet",
        "call",
        "come",
        "do",
        "find",
        "get",
        "give",
        "go",
        "have",
        "keep",
        "let",
        "make",
        "need",
        "never",
        "probably",
        "put",
        "see",
        "show",
        "still",
        "take",
        "tell",
        "try",
        "use",
        "wait",
        "want",
        "work",
    )
    follower_pattern = "|".join(re.escape(word) for word in contraction_followers)
    return re.sub(
        rf"(?<![A-Za-z])(?:\[Il|Ill)(?=\s+(?:{follower_pattern})\b)",
        "I'll",
        text,
        flags=re.IGNORECASE,
    )


def _fix_isolated_pipe_pronouns(text: str) -> str:
    prose_before_words = {
        "actually",
        "after",
        "although",
        "and",
        "as",
        "because",
        "before",
        "but",
        "can",
        "could",
        "did",
        "do",
        "does",
        "even",
        "guess",
        "how",
        "if",
        "like",
        "may",
        "maybe",
        "might",
        "must",
        "no",
        "once",
        "or",
        "should",
        "since",
        "so",
        "than",
        "that",
        "then",
        "though",
        "unless",
        "until",
        "well",
        "what",
        "when",
        "where",
        "while",
        "why",
        "will",
        "would",
        "yeah",
        "yes",
    }
    open_quotes = "\"'\u201c\u2018"
    close_punctuation = ".,!?;:\")]}\u201d\u2019"

    def replace_pipe(match: re.Match[str]) -> str:
        start = match.start()
        end = match.end()
        previous_char = text[start - 1] if start else ""
        next_char = text[end] if end < len(text) else ""

        if previous_char and not previous_char.isspace() and previous_char not in open_quotes:
            return "|"
        if next_char and not next_char.isspace() and next_char not in close_punctuation:
            return "|"

        before = text[:start].rstrip()
        before_context = before.rstrip(f" \t{open_quotes}")
        previous_word = re.search(r"([A-Za-z]+(?:'[A-Za-z]+)?)\W*$", before_context)
        previous_word_text = previous_word.group(1).lower() if previous_word else ""

        if not before_context:
            return "I"
        if before_context[-1] in ".!?,;:([{":
            return "I"
        if previous_word_text in prose_before_words:
            return "I"
        return "|"

    return re.sub(r"\|", replace_pipe, text)


def _resolve_ocr_options(
    use_ocr: bool,
    ocr_language: str,
    ocr_options: OcrOptions | None,
) -> OcrOptions:
    if ocr_options is not None:
        return ocr_options
    return OcrOptions(enabled=use_ocr, language=ocr_language)
