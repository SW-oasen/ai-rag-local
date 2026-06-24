"""Text splitting utilities with source metadata preservation."""

from rag_assistant.schema import Document, TextChunk


def split_documents(
    documents: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Split documents into overlapping chunks."""

    chunks: list[TextChunk] = []

    for document in documents:
        document_chunks = split_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_index=len(chunks),
        )
        chunks.extend(document_chunks)

    return chunks


def split_document(
    document: Document,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    start_chunk_index: int = 0,
) -> list[TextChunk]:
    """Split a document into chunks while keeping source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.text
    if not text.strip():
        return []

    ranges = _chunk_ranges(text, chunk_size, chunk_overlap)
    chunks: list[TextChunk] = []

    for offset, (start_char, end_char) in enumerate(ranges):
        chunk_text = text[start_char:end_char].strip()
        if not chunk_text:
            continue

        chunks.append(
            TextChunk(
                text=chunk_text,
                source_path=document.source_path,
                file_name=document.file_name,
                document_type=document.document_type,
                page_number=document.page_number,
                chunk_index=start_chunk_index + len(chunks),
                start_char=start_char,
                end_char=end_char,
                metadata=document.metadata.copy(),
            )
        )

    return chunks


def _chunk_ranges(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = _find_chunk_end(text, start, target_end, text_length)
        ranges.append((start, end))

        if end >= text_length:
            break

        earliest_start = max(end - chunk_overlap, start + 1)
        start = _find_next_chunk_start(text, earliest_start, end)

    return ranges


def _find_chunk_end(text: str, start: int, target_end: int, text_length: int) -> int:
    if target_end >= text_length:
        return text_length

    preferred_breaks = ["\n\n", "\n", ". ", " "]
    search_window = text[start:target_end]

    for separator in preferred_breaks:
        separator_index = search_window.rfind(separator)
        if separator_index > 0:
            return start + separator_index + len(separator)

    return target_end


def _skip_leading_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _find_next_chunk_start(text: str, earliest_start: int, previous_end: int) -> int:
    """Find a readable overlap start without beginning inside a word."""

    if earliest_start >= len(text):
        return len(text)

    window = text[earliest_start:previous_end]
    preferred_breaks = ["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " "]

    for separator in preferred_breaks:
        separator_index = window.find(separator)
        if separator_index >= 0:
            return _skip_leading_whitespace(text, earliest_start + separator_index + len(separator))

    return _advance_to_word_boundary(text, earliest_start)


def _advance_to_word_boundary(text: str, start: int) -> int:
    index = _skip_leading_whitespace(text, start)
    while index < len(text) and index > 0 and text[index - 1].isalnum() and text[index].isalnum():
        index += 1
    return _skip_leading_whitespace(text, index)
