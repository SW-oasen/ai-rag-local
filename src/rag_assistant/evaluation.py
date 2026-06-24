"""Small retrieval evaluation helpers."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from rag_assistant.retriever import Retriever


class RetrievalEvaluationError(RuntimeError):
    """Raised when retrieval evaluation examples cannot be loaded."""


@dataclass(frozen=True)
class RetrievalExample:
    """Expected source match for a retrieval question."""

    question: str
    expected_file_name: str
    expected_text: str


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Result for one retrieval evaluation example."""

    question: str
    passed: bool
    top_file_name: str | None
    expected_file_name: str
    expected_text: str
    top_chunk_index: int | None = None
    top_page_number: int | None = None
    top_score: float | None = None
    matched_file_name: str | None = None
    matched_chunk_index: int | None = None
    matched_page_number: int | None = None
    matched_score: float | None = None


def evaluate_retrieval(
    retriever: Retriever,
    examples: list[RetrievalExample],
    top_k: int = 3,
    source: str | Path | None = None,
) -> list[RetrievalEvaluationResult]:
    """Evaluate whether retrieval returns the expected source/text."""

    results: list[RetrievalEvaluationResult] = []
    for example in examples:
        retrieved = retriever.retrieve(example.question, top_k=top_k, source=source)
        top_result = retrieved[0] if retrieved else None
        expected_text = _normalize_text(example.expected_text)
        matched_result = next(
            (
                result
                for result in retrieved
                if result.chunk.file_name == example.expected_file_name
                and expected_text in _normalize_text(result.chunk.text)
            ),
            None,
        )
        results.append(
            RetrievalEvaluationResult(
                question=example.question,
                passed=matched_result is not None,
                top_file_name=top_result.chunk.file_name if top_result else None,
                expected_file_name=example.expected_file_name,
                expected_text=example.expected_text,
                top_chunk_index=top_result.chunk.chunk_index if top_result else None,
                top_page_number=top_result.chunk.page_number if top_result else None,
                top_score=top_result.score if top_result else None,
                matched_file_name=matched_result.chunk.file_name if matched_result else None,
                matched_chunk_index=matched_result.chunk.chunk_index if matched_result else None,
                matched_page_number=matched_result.chunk.page_number if matched_result else None,
                matched_score=matched_result.score if matched_result else None,
            )
        )

    return results


def evaluation_results_to_dicts(results: list[RetrievalEvaluationResult]) -> list[dict[str, Any]]:
    """Convert retrieval evaluation results to JSON-serializable dictionaries."""

    return [
        {
            "question": result.question,
            "passed": result.passed,
            "expected_file_name": result.expected_file_name,
            "expected_text": result.expected_text,
            "top_file_name": result.top_file_name,
            "top_chunk_index": result.top_chunk_index,
            "top_page_number": result.top_page_number,
            "top_score": result.top_score,
            "matched_file_name": result.matched_file_name,
            "matched_chunk_index": result.matched_chunk_index,
            "matched_page_number": result.matched_page_number,
            "matched_score": result.matched_score,
        }
        for result in results
    ]


def load_retrieval_examples(path: str | Path) -> list[RetrievalExample]:
    """Load retrieval examples from a markdown table."""

    source = Path(path)
    if source.suffix.lower() not in {".md", ".markdown"}:
        raise RetrievalEvaluationError(
            f"Retrieval eval expects a UTF-8 markdown examples file, not '{source.suffix or 'no extension'}'. "
            "Use examples/retrieval_eval_examples.md or create a similar markdown table for this document."
        )

    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RetrievalEvaluationError(f"Could not read retrieval examples as UTF-8 markdown: {source}") from exc

    rows = _read_markdown_table_rows(text)
    examples = [
        RetrievalExample(
            question=row[0],
            expected_file_name=row[1],
            expected_text=row[2],
        )
        for row in rows
    ]
    if not examples:
        raise RetrievalEvaluationError(f"No retrieval examples found in {source}")
    return examples


def _read_markdown_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    table_started = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if table_started:
                break
            continue

        table_started = True
        cells = _split_markdown_row(stripped)
        if len(cells) < 3 or _is_separator_row(cells) or _is_header_row(cells):
            continue
        rows.append(cells[:3])

    return rows


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(cell.replace("-", "").replace(":", "").strip() == "" for cell in cells)


def _is_header_row(cells: list[str]) -> bool:
    normalized = [cell.lower() for cell in cells[:3]]
    return normalized == ["question", "expected source", "expected evidence"]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
