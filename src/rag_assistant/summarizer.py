"""Dedicated document summarization pipeline."""

from collections.abc import Callable
import time

from rag_assistant.llm_client import LlmClient
from rag_assistant.schema import SourceReference, SummaryResult, TextChunk


class DocumentSummarizer:
    """Summarize full documents from their chunks without top-k retrieval."""

    def __init__(self, llm_client: LlmClient, max_chunks_per_group: int = 4) -> None:
        if max_chunks_per_group <= 0:
            raise ValueError("max_chunks_per_group must be greater than zero")
        self.llm_client = llm_client
        self.max_chunks_per_group = max_chunks_per_group

    def summarize(
        self,
        chunks: list[TextChunk],
        question: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> SummaryResult:
        """Create a map-reduce summary over all provided chunks."""

        ordered_chunks = sorted(chunks, key=lambda chunk: (str(chunk.source_path), chunk.page_number or 0, chunk.chunk_index))
        if not ordered_chunks:
            return SummaryResult(
                summary="No chunks were available to summarize.",
                sources=[],
                source_chunks=[],
                model=self.llm_client.model,
                partial_summaries=[],
            )

        groups = _group_chunks(ordered_chunks, self.max_chunks_per_group)
        total_steps = len(groups) + 1
        started_at = time.monotonic()
        if progress_callback:
            progress_callback(
                f"Summarizing {len(ordered_chunks)} chunks in {len(groups)} groups "
                f"({total_steps} LLM calls total)."
            )

        partial_summaries = [
            self._summarize_group(group, index, len(groups), total_steps, question, progress_callback)
            for index, group in enumerate(groups, start=1)
        ]
        if progress_callback:
            progress_callback(
                f"Progress {_progress_percent(len(groups), total_steps)}%: final merge started "
                f"({len(partial_summaries)} partial summaries)."
            )
        merge_started_at = time.monotonic()
        final_summary = self.llm_client.generate(
            build_final_summary_prompt(partial_summaries, ordered_chunks, question=question)
        )
        if progress_callback:
            progress_callback(
                "Progress 100%: final summary finished "
                f"in {_format_duration(time.monotonic() - merge_started_at)} "
                f"(total {_format_duration(time.monotonic() - started_at)})."
            )

        return SummaryResult(
            summary=final_summary,
            sources=_build_sources(ordered_chunks),
            source_chunks=ordered_chunks,
            model=self.llm_client.model,
            partial_summaries=partial_summaries,
        )

    def _summarize_group(
        self,
        group: list[TextChunk],
        group_index: int,
        group_count: int,
        total_steps: int,
        question: str | None,
        progress_callback: Callable[[str], None] | None,
    ) -> str:
        if progress_callback:
            completed_steps = group_index - 1
            progress_callback(
                f"Progress {_progress_percent(completed_steps, total_steps)}%: "
                f"group {group_index}/{group_count} started ({_format_group_scope(group)})."
            )
        group_started_at = time.monotonic()
        summary = self.llm_client.generate(build_chunk_summary_prompt(group, question=question))
        if progress_callback:
            progress_callback(
                f"Progress {_progress_percent(group_index, total_steps)}%: "
                f"group {group_index}/{group_count} finished in "
                f"{_format_duration(time.monotonic() - group_started_at)}."
            )
        return summary


def build_chunk_summary_prompt(chunks: list[TextChunk], question: str | None = None) -> str:
    """Build a prompt for summarizing a group of source chunks."""

    focus = f"\nUser focus: {question}\n" if question else ""
    return f"""Summarize the following document chunks.

Rules:
- Preserve important facts, entities, methods, decisions, and limitations.
- Do not add information that is not present in the chunks.
- Mention uncertainty when the chunks are incomplete.
- Keep source labels in the summary when useful.
{focus}
Chunks:
{_format_chunks(chunks)}

Partial summary:
"""


def build_final_summary_prompt(
    partial_summaries: list[str],
    source_chunks: list[TextChunk],
    question: str | None = None,
) -> str:
    """Build a prompt for merging partial summaries into a final summary."""

    focus = f"\nUser focus: {question}\n" if question else ""
    partial_text = "\n\n".join(
        f"[partial summary {index}]\n{summary}" for index, summary in enumerate(partial_summaries, start=1)
    )
    source_text = ", ".join(_source_label(chunk) for chunk in source_chunks)
    return f"""Create a final concise summary from the partial summaries.

Rules:
- Use only the partial summaries.
- Keep the final answer coherent and non-repetitive.
- Include key limitations or missing context if visible.
- Cite source chunk labels where appropriate.
{focus}
Available source chunks:
{source_text}

Partial summaries:
{partial_text}

Final summary:
"""


def _group_chunks(chunks: list[TextChunk], group_size: int) -> list[list[TextChunk]]:
    return [chunks[index : index + group_size] for index in range(0, len(chunks), group_size)]


def _progress_percent(completed_steps: int, total_steps: int) -> int:
    if total_steps <= 0:
        return 100
    return round((completed_steps / total_steps) * 100)


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _format_group_scope(chunks: list[TextChunk]) -> str:
    if not chunks:
        return "0 chunks"

    first = chunks[0]
    last = chunks[-1]
    source = first.file_name if first.file_name == last.file_name else "multiple sources"
    page_range = _format_page_range(first.page_number, last.page_number)
    chunk_range = (
        f"chunk {first.chunk_index}"
        if first.chunk_index == last.chunk_index
        else f"chunks {first.chunk_index}-{last.chunk_index}"
    )
    return f"{len(chunks)} chunks, {source}, {chunk_range}{page_range}"


def _format_page_range(first_page: int | None, last_page: int | None) -> str:
    if first_page is None and last_page is None:
        return ""
    if first_page == last_page:
        return f", page {first_page}"
    if first_page is None or last_page is None:
        return ""
    return f", pages {first_page}-{last_page}"


def _format_chunks(chunks: list[TextChunk]) -> str:
    return "\n\n".join(f"[{_source_label(chunk)}]\n{chunk.text}" for chunk in chunks)


def _source_label(chunk: TextChunk) -> str:
    page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
    return f"{chunk.file_name}, chunk {chunk.chunk_index}{page}"


def _build_sources(chunks: list[TextChunk]) -> list[SourceReference]:
    sources: list[SourceReference] = []
    seen: set[tuple[str, int, int | None]] = set()
    for chunk in chunks:
        key = (str(chunk.source_path), chunk.chunk_index, chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceReference(
                file_name=chunk.file_name,
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
            )
        )
    return sources
