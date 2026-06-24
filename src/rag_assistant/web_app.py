"""Dependency-free local web UI for the RAG assistant."""

from argparse import ArgumentParser, Namespace
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs

from rag_assistant.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    VECTOR_STORE_DIR,
)
from rag_assistant.document_loader import OcrOptions, load_documents
from rag_assistant.embeddings import OllamaEmbeddingProvider
from rag_assistant.llm_client import OllamaLlmClient
from rag_assistant.rag_pipeline import RagPipeline
from rag_assistant.retriever import Retriever
from rag_assistant.schema import Document, IndexedSource, RagAnswer, RetrievalResult, SummaryResult
from rag_assistant.summarizer import DocumentSummarizer
from rag_assistant.vector_store import ChromaVectorStore


def main(argv: list[str] | None = None) -> int:
    """Run the local browser UI."""

    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(args))
    print(f"RAG assistant UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RAG assistant UI.")
    finally:
        server.server_close()
    return 0


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="rag-assistant-ui",
        description="Local browser UI for indexed document retrieval and Q&A.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vector-store", type=Path, default=VECTOR_STORE_DIR)
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--ocr-language", default="eng")
    parser.add_argument("--ocr-scale", type=float, default=3.0)
    parser.add_argument("--ocr-psm", type=int, default=6)
    return parser


def create_handler(args: Namespace) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to CLI configuration."""

    class RagUiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self._send_html(render_page(error="Unknown page."))
                return
            self._send_html(render_page(sources=self._load_sources()))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            question = _first(form, "question").strip()
            source = _first(form, "source").strip() or None
            extract_path = _first(form, "extract_path").strip()
            use_ocr = _first(form, "use_ocr") == "on"
            ocr_language = _first(form, "ocr_language").strip() or args.ocr_language
            ocr_scale = _parse_positive_float(_first(form, "ocr_scale"), args.ocr_scale)
            ocr_psm = _parse_positive_int(_first(form, "ocr_psm"), args.ocr_psm)
            ocr_preprocess = _first(form, "ocr_preprocess") == "on"
            ocr_clean_text = _first(form, "ocr_clean_text") == "on"
            top_k = _parse_positive_int(_first(form, "top_k"), DEFAULT_TOP_K)

            try:
                if self.path == "/retrieve":
                    results = self._retrieve(question, top_k=top_k, source=source)
                    self._send_html(
                        render_page(
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            retrieval_results=results,
                        )
                    )
                    return
                if self.path == "/ask":
                    answer = self._answer(question, top_k=top_k, source=source)
                    self._send_html(
                        render_page(
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            answer=answer,
                        )
                    )
                    return
                if self.path == "/summarize":
                    summary = self._summarize(source)
                    self._send_html(
                        render_page(
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            summary=summary,
                        )
                    )
                    return
                if self.path == "/extract-text":
                    ocr_options = OcrOptions(
                        enabled=use_ocr,
                        language=ocr_language,
                        scale=ocr_scale,
                        psm=ocr_psm,
                        preprocess=ocr_preprocess,
                        clean_text=ocr_clean_text,
                    )
                    documents = self._extract_text(extract_path, ocr_options=ocr_options)
                    self._send_html(
                        render_page(
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            extract_path=extract_path,
                            ocr_options=ocr_options,
                            extracted_documents=documents,
                        )
                    )
                    return
                self._send_html(render_page(error="Unknown action."))
            except Exception as exc:
                self._send_html(render_page(sources=self._load_sources_safe(), error=str(exc)))

        def log_message(self, format: str, *args) -> None:
            return

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _vector_store(self) -> ChromaVectorStore:
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

        def _retriever(self) -> Retriever:
            return Retriever(self._vector_store(), top_k=DEFAULT_TOP_K)

        def _load_sources(self) -> list[IndexedSource]:
            return self._vector_store().list_sources()

        def _load_sources_safe(self) -> list[IndexedSource]:
            try:
                return self._load_sources()
            except Exception:
                return []

        def _retrieve(self, question: str, top_k: int, source: str | None) -> list[RetrievalResult]:
            if not question:
                return []
            return self._retriever().retrieve(question, top_k=top_k, source=source)

        def _answer(self, question: str, top_k: int, source: str | None) -> RagAnswer | None:
            if not question:
                return None
            llm_client = OllamaLlmClient(
                model=args.llm_model,
                host=args.ollama_host,
                temperature=args.temperature,
            )
            return RagPipeline(retriever=self._retriever(), llm_client=llm_client).answer(
                question,
                top_k=top_k,
                source=source,
            )

        def _summarize(self, source: str | None) -> SummaryResult:
            if not source:
                raise ValueError("Select one indexed source before summarizing.")
            llm_client = OllamaLlmClient(
                model=args.llm_model,
                host=args.ollama_host,
                temperature=args.temperature,
            )
            chunks = self._vector_store().get_chunks_by_source(source)
            return DocumentSummarizer(llm_client=llm_client).summarize(chunks)

        def _extract_text(self, path: str, ocr_options: OcrOptions) -> list[Document]:
            if not path:
                raise ValueError("Enter a local file or folder path before extracting text.")
            return load_documents(path, ocr_options=ocr_options)

    return RagUiHandler


def render_page(
    sources: list[IndexedSource] | None = None,
    question: str = "",
    selected_source: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    retrieval_results: list[RetrievalResult] | None = None,
    answer: RagAnswer | None = None,
    summary: SummaryResult | None = None,
    extract_path: str = "",
    ocr_options: OcrOptions | None = None,
    extracted_documents: list[Document] | None = None,
    error: str | None = None,
) -> str:
    """Render the single-page local UI."""

    sources = sources or []
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local RAG Assistant</title>
  <style>{_styles()}</style>
</head>
<body>
  <main>
    <header>
      <h1>Local RAG Assistant</h1>
      <p>{len(sources)} indexed source{'' if len(sources) == 1 else 's'}</p>
    </header>
    {render_query_form(sources, question, selected_source, top_k)}
    {render_extract_form(extract_path, ocr_options)}
    {render_error(error)}
    {render_answer(answer)}
    {render_summary(summary)}
    {render_extracted_documents(extracted_documents)}
    {render_retrieval_results(retrieval_results)}
    {render_sources(sources)}
  </main>
</body>
</html>"""


def render_query_form(
    sources: list[IndexedSource],
    question: str,
    selected_source: str | None,
    top_k: int,
) -> str:
    source_options = ['<option value="">All sources</option>']
    for source in sources:
        value = escape(source.file_name)
        selected = " selected" if selected_source == source.file_name else ""
        source_options.append(f'<option value="{value}"{selected}>{value}</option>')
    return f"""
    <section class="query">
      <form method="post">
        <label>
          Question
          <input name="question" value="{escape(question)}" placeholder="Ask about your indexed documents" autofocus>
        </label>
        <div class="controls">
          <label>
            Source
            <select name="source">{''.join(source_options)}</select>
          </label>
          <label>
            Top K
            <input class="small" name="top_k" type="number" min="1" max="20" value="{top_k}">
          </label>
        </div>
        <div class="actions">
          <button formaction="/retrieve" type="submit">Retrieve</button>
          <button formaction="/ask" type="submit">Ask</button>
          <button formaction="/summarize" type="submit">Summarize Source</button>
        </div>
      </form>
    </section>"""


def render_extract_form(extract_path: str = "", ocr_options: OcrOptions | None = None) -> str:
    options = ocr_options or OcrOptions()
    checked = " checked" if options.enabled else ""
    preprocess_checked = " checked" if options.preprocess else ""
    clean_checked = " checked" if options.clean_text else ""
    return f"""
    <section class="query">
      <form method="post" action="/extract-text">
        <label>
          Extract Text Path
          <input name="extract_path" value="{escape(extract_path)}" placeholder="data/raw/example.pdf">
        </label>
        <div class="controls ocr-controls">
          <label>
            OCR Language
            <input name="ocr_language" value="{escape(options.language)}" placeholder="eng or eng+deu">
          </label>
          <label>
            Scale
            <input class="small" name="ocr_scale" type="number" min="1" step="0.5" value="{options.scale}">
          </label>
          <label>
            PSM
            <input class="small" name="ocr_psm" type="number" min="1" max="13" value="{options.psm}">
          </label>
        </div>
        <div class="actions">
          <label class="checkbox">
            <input name="use_ocr" type="checkbox"{checked}>
            OCR
          </label>
          <label class="checkbox">
            <input name="ocr_preprocess" type="checkbox"{preprocess_checked}>
            Preprocess
          </label>
          <label class="checkbox">
            <input name="ocr_clean_text" type="checkbox"{clean_checked}>
            Clean Text
          </label>
          <button type="submit">Extract Text</button>
        </div>
      </form>
    </section>"""


def render_sources(sources: list[IndexedSource]) -> str:
    if not sources:
        return '<section><h2>Sources</h2><p>No indexed sources found.</p></section>'
    rows = []
    for source in sources:
        pages = f"{source.page_count}" if source.page_count is not None else "n/a"
        rows.append(
            "<tr>"
            f"<td>{escape(source.file_name)}</td>"
            f"<td>{escape(source.document_type)}</td>"
            f"<td>{source.chunk_count}</td>"
            f"<td>{pages}</td>"
            f"<td>{escape(str(source.source_path))}</td>"
            "</tr>"
        )
    return f"""
    <section>
      <h2>Sources</h2>
      <table>
        <thead><tr><th>Name</th><th>Type</th><th>Chunks</th><th>Pages</th><th>Path</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def render_retrieval_results(results: list[RetrievalResult] | None) -> str:
    if results is None:
        return ""
    if not results:
        return "<section><h2>Retrieved Chunks</h2><p>No chunks retrieved.</p></section>"
    items = []
    for result in results:
        chunk = result.chunk
        score = f"{result.score:.4f}" if result.score is not None else "n/a"
        page = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        items.append(
            "<article>"
            f"<h3>{escape(chunk.file_name)}, chunk {chunk.chunk_index}{page}</h3>"
            f"<p class=\"meta\">score {score} | {escape(str(chunk.source_path))}</p>"
            f"<p>{escape(_preview(chunk.text, 700))}</p>"
            "</article>"
        )
    return f"<section><h2>Retrieved Chunks</h2>{''.join(items)}</section>"


def render_answer(answer: RagAnswer | None) -> str:
    if answer is None:
        return ""
    sources = "".join(
        f"<li>{escape(source.file_name)}, chunk {source.chunk_index}"
        f"{', page ' + str(source.page_number) if source.page_number is not None else ''}</li>"
        for source in answer.sources
    )
    return f"""
    <section>
      <h2>Answer</h2>
      <p>{escape(answer.answer)}</p>
      <h3>Sources</h3>
      <ul>{sources or '<li>No sources returned.</li>'}</ul>
    </section>"""


def render_summary(summary: SummaryResult | None) -> str:
    if summary is None:
        return ""
    sources = "".join(
        f"<li>{escape(source.file_name)}, chunk {source.chunk_index}"
        f"{', page ' + str(source.page_number) if source.page_number is not None else ''}</li>"
        for source in summary.sources
    )
    return f"""
    <section>
      <h2>Summary</h2>
      <p>{escape(summary.summary)}</p>
      <h3>Sources</h3>
      <ul>{sources or '<li>No sources returned.</li>'}</ul>
      <p class="meta">Partial summaries: {len(summary.partial_summaries)}</p>
    </section>"""


def render_extracted_documents(documents: list[Document] | None) -> str:
    if documents is None:
        return ""
    if not documents:
        return "<section><h2>Extracted Text</h2><p>No documents loaded.</p></section>"

    items = []
    for document in documents:
        page = f", page {document.page_number}" if document.page_number is not None else ""
        ocr = " | OCR" if document.metadata.get("ocr_used") else ""
        items.append(
            "<article>"
            f"<h3>{escape(document.file_name)}{page}</h3>"
            f"<p class=\"meta\">{escape(document.document_type)} | {escape(str(document.source_path))}{ocr}</p>"
            f"<pre>{escape(document.text.strip() or '[no text extracted]')}</pre>"
            "</article>"
        )
    return f"<section><h2>Extracted Text</h2>{''.join(items)}</section>"


def render_error(error: str | None) -> str:
    if not error:
        return ""
    return f'<section class="error"><h2>Error</h2><p>{escape(error)}</p></section>'


def _first(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key, [""])
    return values[0] if values else ""


def _parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _parse_positive_float(value: str, default: float) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _preview(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _styles() -> str:
    return """
:root { color-scheme: light; font-family: Arial, sans-serif; }
body { margin: 0; background: #f7f7f4; color: #202124; }
main { max-width: 1120px; margin: 0 auto; padding: 28px; }
header { display: flex; justify-content: space-between; align-items: end; gap: 16px; border-bottom: 1px solid #d8d8d2; padding-bottom: 16px; }
h1 { font-size: 28px; margin: 0; font-weight: 700; }
h2 { font-size: 18px; margin: 0 0 14px; }
h3 { font-size: 15px; margin: 0 0 6px; }
p { line-height: 1.5; }
section { margin-top: 22px; }
.query { background: #ffffff; border: 1px solid #d8d8d2; padding: 18px; }
label { display: grid; gap: 6px; font-size: 13px; font-weight: 700; }
input, select { box-sizing: border-box; width: 100%; border: 1px solid #b9bbb5; padding: 10px; font: inherit; background: #ffffff; }
input.small { max-width: 110px; }
.controls { display: grid; grid-template-columns: minmax(220px, 1fr) 120px; gap: 14px; margin-top: 14px; }
.ocr-controls { grid-template-columns: minmax(220px, 1fr) 120px 120px; }
.actions { display: flex; gap: 10px; margin-top: 16px; }
button { border: 1px solid #202124; background: #202124; color: #ffffff; padding: 10px 14px; font: inherit; cursor: pointer; }
button:first-child { background: #ffffff; color: #202124; }
.checkbox { display: flex; align-items: center; gap: 8px; font-weight: 700; }
.checkbox input { width: auto; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #fafafa; border: 1px solid #e7e7e2; padding: 12px; max-height: 420px; overflow: auto; }
table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d8d8d2; }
th, td { text-align: left; border-bottom: 1px solid #e7e7e2; padding: 9px; vertical-align: top; font-size: 14px; }
article { background: #ffffff; border: 1px solid #d8d8d2; padding: 14px; margin-top: 10px; }
.meta { color: #60635d; font-size: 13px; margin: 0 0 8px; }
.error { border-left: 4px solid #a6342e; background: #fff4f2; padding: 14px; }
@media (max-width: 760px) {
  main { padding: 18px; }
  header, .controls { display: grid; grid-template-columns: 1fr; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
