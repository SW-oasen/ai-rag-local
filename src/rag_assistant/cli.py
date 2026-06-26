"""Command-line interface for local RAG ingestion and question answering."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys
from typing import Sequence

from rag_assistant.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    VECTOR_STORE_DIR,
)
from rag_assistant.document_loader import OcrOptions, load_documents
from rag_assistant.embeddings import EmbeddingError, OllamaEmbeddingProvider
from rag_assistant.evaluation import (
    RetrievalEvaluationError,
    RetrievalEvaluationResult,
    evaluation_results_to_dicts,
    evaluate_retrieval,
    load_retrieval_examples,
)
from rag_assistant.llm_client import OllamaLlmClient
from rag_assistant.rag_pipeline import RagPipeline
from rag_assistant.retriever import Retriever
from rag_assistant.schema import IndexedSource, RagAnswer, RetrievalResult, TextChunk
from rag_assistant.summarizer import DocumentSummarizer
from rag_assistant.text_splitter import split_documents
from rag_assistant.vector_store import ChromaVectorStore, VectorStoreError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local RAG CLI."""

    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    try:
        return args.handler(args)
    except EmbeddingError as exc:
        print(f"Embedding error: {exc}")
        return 2
    except VectorStoreError as exc:
        print(f"Vector store error: {exc}")
        return 3
    except RetrievalEvaluationError as exc:
        print(f"Evaluation error: {exc}")
        return 4


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="rag-assistant",
        description="Local-first RAG assistant for document ingestion and Q&A.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Load documents, chunk them, and store embeddings.")
    ingest.add_argument("path", type=Path, help="File or directory to ingest.")
    _add_storage_args(ingest)
    ingest.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ingest.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ingest.add_argument("--ocr", action="store_true", help="Run OCR on PDF pages with no selectable text.")
    ingest.add_argument("--ocr-language", default="eng", help="Tesseract OCR language code.")
    ingest.add_argument("--ocr-scale", type=float, default=3.0, help="PDF render scale for OCR.")
    ingest.add_argument("--ocr-psm", type=int, default=6, help="Tesseract page segmentation mode.")
    ingest.add_argument("--no-ocr-preprocess", action="store_true", help="Disable OCR image preprocessing.")
    ingest.add_argument("--no-ocr-clean", action="store_true", help="Disable OCR text cleanup.")
    ingest.set_defaults(handler=_handle_ingest)

    retrieve = subparsers.add_parser("retrieve", help="Retrieve relevant chunks for a question.")
    retrieve.add_argument("question", help="Question or search query.")
    _add_storage_args(retrieve)
    retrieve.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    retrieve.add_argument("--source", type=Path, default=None, help="Limit retrieval to one indexed source path/file name.")
    retrieve.set_defaults(handler=_handle_retrieve)

    sources = subparsers.add_parser("sources", help="List documents stored in the vector index.")
    _add_storage_args(sources)
    sources.set_defaults(handler=_handle_sources)

    delete_source = subparsers.add_parser("delete-source", help="Delete one indexed source from the vector index.")
    delete_source.add_argument("source", type=Path, help="Indexed source path or file name.")
    _add_storage_args(delete_source)
    delete_source.set_defaults(handler=_handle_delete_source)

    reset_index = subparsers.add_parser("reset-index", help="Delete all chunks from the vector index.")
    _add_storage_args(reset_index)
    reset_index.add_argument("--yes", action="store_true", help="Confirm deleting all indexed chunks.")
    reset_index.set_defaults(handler=_handle_reset_index)

    chunks = subparsers.add_parser("chunks", help="Inspect stored chunks for one indexed source.")
    chunks.add_argument("source", type=Path, help="Indexed source path or file name.")
    _add_storage_args(chunks)
    chunks.add_argument("--limit", type=int, default=20, help="Maximum number of chunks to print.")
    chunks.add_argument("--preview-chars", type=int, default=220, help="Maximum preview characters per chunk.")
    chunks.set_defaults(handler=_handle_chunks)

    ask = subparsers.add_parser("ask", help="Retrieve context and answer with a local LLM.")
    ask.add_argument("question", help="Question to answer.")
    _add_storage_args(ask)
    ask.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ask.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ask.add_argument("--temperature", type=float, default=0.1)
    ask.add_argument("--show-prompt", action="store_true")
    ask.add_argument("--source", type=Path, default=None, help="Limit retrieval to one indexed source path/file name.")
    ask.set_defaults(handler=_handle_ask)

    summarize = subparsers.add_parser("summarize", help="Summarize a full document without top-k retrieval.")
    summarize.add_argument("source", type=Path, help="File to load directly, or indexed source path/file name.")
    _add_storage_args(summarize)
    summarize.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    summarize.add_argument("--temperature", type=float, default=0.1)
    summarize.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    summarize.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    summarize.add_argument("--max-chunks-per-group", type=int, default=4)
    summarize.add_argument("--question", default=None, help="Optional focus question for the summary.")
    summarize.add_argument(
        "--from-index",
        action="store_true",
        help="Read chunks from the vector store instead of loading the source file directly.",
    )
    summarize.set_defaults(handler=_handle_summarize)

    evaluate = subparsers.add_parser("eval", help="Run retrieval evaluation examples.")
    evaluate.add_argument("examples", type=Path, help="Markdown table with retrieval evaluation examples.")
    _add_storage_args(evaluate)
    evaluate.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    evaluate.add_argument("--source", type=Path, default=None, help="Limit evaluation retrieval to one indexed source.")
    evaluate.add_argument("--json-report", type=Path, default=None, help="Write detailed results to a JSON file.")
    evaluate.set_defaults(handler=_handle_eval)

    return parser


def _add_storage_args(parser: ArgumentParser) -> None:
    parser.add_argument("--vector-store", type=Path, default=VECTOR_STORE_DIR)
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--ollama-host", default=None)


def _handle_ingest(args: Namespace) -> int:
    documents = load_documents(args.path, ocr_options=_ocr_options_from_args(args))
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    vector_store = _create_vector_store(args)
    vector_store.add_chunks(chunks)

    print(f"Ingested documents: {len(documents)}")
    print(f"Stored chunks: {len(chunks)}")
    print(f"Vector store: {args.vector_store}")
    print(f"Collection size: {vector_store.count()}")
    return 0


def _ocr_options_from_args(args: Namespace) -> OcrOptions:
    return OcrOptions(
        enabled=args.ocr,
        language=args.ocr_language,
        scale=args.ocr_scale,
        psm=args.ocr_psm,
        preprocess=not args.no_ocr_preprocess,
        clean_text=not args.no_ocr_clean,
    )


def _handle_retrieve(args: Namespace) -> int:
    retriever = _create_retriever(args)
    results = retriever.retrieve(args.question, top_k=args.top_k, source=args.source)
    print(format_retrieval_results(results))
    return 0


def _handle_sources(args: Namespace) -> int:
    vector_store = _create_vector_store(args)
    print(format_indexed_sources(vector_store.list_sources(), total_chunks=vector_store.count()))
    return 0


def _handle_delete_source(args: Namespace) -> int:
    deleted_count = _create_vector_store(args).delete_source(args.source)
    print(f"Deleted chunks: {deleted_count}")
    return 0


def _handle_reset_index(args: Namespace) -> int:
    if not args.yes:
        print("Reset aborted. Pass --yes to delete all indexed chunks.")
        return 1
    deleted_count = _create_vector_store(args).reset()
    print(f"Deleted chunks: {deleted_count}")
    return 0


def _handle_chunks(args: Namespace) -> int:
    chunks = _create_vector_store(args).get_chunks_by_source(args.source)
    print(format_source_chunks(chunks, source=args.source, limit=args.limit, preview_chars=args.preview_chars))
    return 0


def _handle_ask(args: Namespace) -> int:
    retriever = _create_retriever(args)
    llm_client = OllamaLlmClient(
        model=args.llm_model,
        host=args.ollama_host,
        temperature=args.temperature,
    )
    answer = RagPipeline(retriever=retriever, llm_client=llm_client).answer(
        args.question,
        top_k=args.top_k,
        source=args.source,
    )
    print(format_rag_answer(answer, show_prompt=args.show_prompt))
    return 0


def _handle_summarize(args: Namespace) -> int:
    if args.from_index:
        chunks = _create_vector_store(args).get_chunks_by_source(args.source)
    else:
        documents = load_documents(args.source)
        chunks = split_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print(f"Loaded chunks for summary: {len(chunks)}", file=sys.stderr, flush=True)
    llm_client = OllamaLlmClient(
        model=args.llm_model,
        host=args.ollama_host,
        temperature=args.temperature,
    )
    summary = DocumentSummarizer(
        llm_client=llm_client,
        max_chunks_per_group=args.max_chunks_per_group,
    ).summarize(chunks, question=args.question, progress_callback=_print_progress)
    print(format_summary_result(summary))
    return 0


def _handle_eval(args: Namespace) -> int:
    examples = load_retrieval_examples(args.examples)
    retriever = _create_retriever(args)
    results = evaluate_retrieval(retriever, examples, top_k=args.top_k, source=args.source)
    if args.json_report:
        write_evaluation_json_report(results, args.json_report)
    print(format_evaluation_results(results))
    return 0 if all(result.passed for result in results) else 1


def _create_retriever(args: Namespace) -> Retriever:
    return Retriever(_create_vector_store(args), top_k=args.top_k)


def _create_vector_store(args: Namespace) -> ChromaVectorStore:
    args.vector_store.mkdir(parents=True, exist_ok=True)
    embedding_provider = OllamaEmbeddingProvider(
        model=args.embedding_model,
        host=args.ollama_host,
        batch_size=args.embedding_batch_size,
    )
    return ChromaVectorStore(
        persist_directory=args.vector_store,
        embedding_provider=embedding_provider,
        collection_name=args.collection,
    )


def format_retrieval_results(results: list[RetrievalResult]) -> str:
    """Format retrieval results for terminal output."""

    if not results:
        return "No chunks retrieved."

    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        score = f"{result.score:.4f}" if result.score is not None else "n/a"
        page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        sections.append(
            "\n".join(
                [
                    f"[{index}] {chunk.file_name} | chunk {chunk.chunk_index}{page} | score {score}",
                    f"Path: {chunk.source_path}",
                    chunk.text,
                ]
            )
        )

    return "\n\n".join(sections)


def format_indexed_sources(sources: list[IndexedSource], total_chunks: int) -> str:
    """Format indexed source summaries for terminal output."""

    lines = [f"Indexed sources: {len(sources)}", f"Total chunks: {total_chunks}"]
    if not sources:
        lines.append("No sources indexed.")
        return "\n".join(lines)

    lines.append("")
    for index, source in enumerate(sources, start=1):
        pages = f", pages {source.page_count}" if source.page_count is not None else ""
        lines.append(
            f"[{index}] {source.file_name} | {source.document_type} | chunks {source.chunk_count}{pages}"
        )
        lines.append(f"Path: {source.source_path}")
    return "\n".join(lines)


def format_source_chunks(
    chunks: list[TextChunk],
    source: str | Path,
    limit: int = 20,
    preview_chars: int = 220,
) -> str:
    """Format stored chunks for one indexed source."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if preview_chars <= 0:
        raise ValueError("preview_chars must be greater than zero")

    lines = [f"Chunks for source: {source}", f"Total chunks: {len(chunks)}"]
    if not chunks:
        lines.append("No chunks found for this source.")
        return "\n".join(lines)

    shown_chunks = chunks[:limit]
    lines.append(f"Showing: {len(shown_chunks)}")
    lines.append("")
    for chunk in shown_chunks:
        page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        lines.append(
            f"[{chunk.chunk_index}] {chunk.file_name}{page} | chars {chunk.start_char}-{chunk.end_char}"
        )
        lines.append(_preview_text(chunk.text, preview_chars))
    return "\n".join(lines)


def format_rag_answer(answer: RagAnswer, show_prompt: bool = False) -> str:
    """Format a RAG answer for terminal output."""

    lines = [answer.answer, "", "Sources:"]
    if answer.sources:
        for index, source in enumerate(answer.sources, start=1):
            page = f", page {source.page_number}" if source.page_number is not None else ""
            score = f", score {source.score:.4f}" if source.score is not None else ""
            lines.append(f"- [{index}] {source.file_name}, chunk {source.chunk_index}{page}{score}")
    else:
        lines.append("- No sources returned.")

    if show_prompt:
        lines.extend(["", "Prompt:", answer.prompt])

    return "\n".join(lines)


def format_summary_result(summary) -> str:
    """Format a document summary for terminal output."""

    lines = [summary.summary, "", "Sources:"]
    if summary.sources:
        for index, source in enumerate(summary.sources, start=1):
            page = f", page {source.page_number}" if source.page_number is not None else ""
            lines.append(f"- [{index}] {source.file_name}, chunk {source.chunk_index}{page}")
    else:
        lines.append("- No sources returned.")

    lines.append("")
    lines.append(f"Partial summaries: {len(summary.partial_summaries)}")
    return "\n".join(lines)


def format_evaluation_results(results: list[RetrievalEvaluationResult]) -> str:
    """Format retrieval evaluation results for terminal output."""

    if not results:
        return "No evaluation results."

    passed_count = sum(1 for result in results if result.passed)
    lines = [f"Retrieval evaluation: {passed_count}/{len(results)} passed", ""]
    for index, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        top_file = result.top_file_name or "none"
        lines.append(
            f"[{index}] {status} | expected {result.expected_file_name} | top result {top_file} | {result.question}"
        )
        if result.matched_file_name is not None:
            page = f", page {result.matched_page_number}" if result.matched_page_number is not None else ""
            score = f", score {result.matched_score:.4f}" if result.matched_score is not None else ""
            lines.append(f"    match: {result.matched_file_name}, chunk {result.matched_chunk_index}{page}{score}")
        elif result.top_chunk_index is not None:
            page = f", page {result.top_page_number}" if result.top_page_number is not None else ""
            score = f", score {result.top_score:.4f}" if result.top_score is not None else ""
            lines.append(f"    top: {top_file}, chunk {result.top_chunk_index}{page}{score}")
    return "\n".join(lines)


def write_evaluation_json_report(results: list[RetrievalEvaluationResult], path: str | Path) -> None:
    """Write retrieval evaluation details to a JSON file."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(evaluation_results_to_dicts(results), indent=2),
        encoding="utf-8",
    )


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _preview_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
