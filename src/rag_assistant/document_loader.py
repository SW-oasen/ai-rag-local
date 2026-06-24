"""Document loading utilities for local RAG ingestion."""

from dataclasses import dataclass
from pathlib import Path
import re

from rag_assistant.schema import Document

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


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
