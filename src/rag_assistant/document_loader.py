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
    text = path.read_text(encoding="utf-8")
    return Document(
        text=text,
        source_path=path,
        file_name=path.name,
        document_type=path.suffix.lower().lstrip("."),
    )


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

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        metadata = {
            "ocr_used": False,
            "ocr_language": options.language,
            "ocr_scale": options.scale,
            "ocr_psm": options.psm,
            "ocr_preprocess": options.preprocess,
            "ocr_clean_text": options.clean_text,
        }
        if options.enabled and not text.strip():
            text = _ocr_pdf_page(path, page_index - 1, options=options)
            metadata["ocr_used"] = True
        if metadata["ocr_used"] and options.clean_text:
            text = clean_ocr_text(text)
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
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _resolve_ocr_options(
    use_ocr: bool,
    ocr_language: str,
    ocr_options: OcrOptions | None,
) -> OcrOptions:
    if ocr_options is not None:
        return ocr_options
    return OcrOptions(enabled=use_ocr, language=ocr_language)
