from pathlib import Path

import pytest

from rag_assistant.schema import Document
from rag_assistant.text_splitter import split_document, split_documents


def test_split_document_preserves_source_metadata() -> None:
    document = Document(
        text="Alpha beta gamma. Delta epsilon zeta.",
        source_path=Path("docs/example.md"),
        file_name="example.md",
        document_type="md",
        metadata={"collection": "demo"},
    )

    chunks = split_document(document, chunk_size=18, chunk_overlap=4)

    assert len(chunks) > 1
    assert chunks[0].file_name == "example.md"
    assert chunks[0].document_type == "md"
    assert chunks[0].source_path == Path("docs/example.md")
    assert chunks[0].metadata == {"collection": "demo"}
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_split_document_keeps_chunk_ranges_inside_text() -> None:
    text = "One two three four five six seven eight nine ten."
    document = Document(
        text=text,
        source_path=Path("sample.txt"),
        file_name="sample.txt",
        document_type="txt",
    )

    chunks = split_document(document, chunk_size=20, chunk_overlap=5)

    assert chunks
    for chunk in chunks:
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)
        assert chunk.text == text[chunk.start_char : chunk.end_char].strip()


def test_split_documents_continues_chunk_indexes_across_documents() -> None:
    documents = [
        Document("A " * 20, Path("a.txt"), "a.txt", "txt"),
        Document("B " * 20, Path("b.txt"), "b.txt", "txt"),
    ]

    chunks = split_documents(documents, chunk_size=12, chunk_overlap=2)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_split_document_rejects_invalid_overlap() -> None:
    document = Document("content", Path("sample.txt"), "sample.txt", "txt")

    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        split_document(document, chunk_size=100, chunk_overlap=100)


def test_split_document_skips_empty_text() -> None:
    document = Document("   ", Path("empty.txt"), "empty.txt", "txt")

    assert split_document(document) == []


def test_split_document_prefers_sentence_boundaries_for_overlap_starts() -> None:
    text = (
        "First sentence explains ingestion. "
        "Second sentence explains semantic retrieval. "
        "Third sentence explains answer generation."
    )
    document = Document(text, Path("sample.txt"), "sample.txt", "txt")

    chunks = split_document(document, chunk_size=58, chunk_overlap=24)

    assert len(chunks) >= 2
    assert chunks[1].text.startswith("Second sentence") or chunks[1].text.startswith("Third sentence")


def test_split_document_does_not_start_chunks_inside_words() -> None:
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo"
    document = Document(text, Path("words.txt"), "words.txt", "txt")

    chunks = split_document(document, chunk_size=28, chunk_overlap=9)

    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert chunk.start_char == 0 or not (
            text[chunk.start_char - 1].isalnum() and text[chunk.start_char].isalnum()
        )
