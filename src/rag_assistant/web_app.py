"""Dependency-free local web UI for the RAG assistant."""

from argparse import ArgumentParser, Namespace
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from rag_assistant.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    VECTOR_STORE_DIR,
)
from rag_assistant.document_loader import SUPPORTED_EXTENSIONS, OcrOptions, load_documents
from rag_assistant.embeddings import OllamaEmbeddingProvider
from rag_assistant.library_store import CachedSummary, ConfiguredPath, LibraryStore
from rag_assistant.llm_client import OllamaLlmClient
from rag_assistant.rag_pipeline import RagPipeline
from rag_assistant.retriever import Retriever
from rag_assistant.schema import Document, IndexedSource, RagAnswer, RetrievalResult, SummaryResult
from rag_assistant.summarizer import DocumentSummarizer
from rag_assistant.text_splitter import split_documents
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
    parser.add_argument("--library-store", type=Path, default=PROCESSED_DATA_DIR / "web_library.json")
    parser.add_argument("--collection", default="rag_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
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
            parsed = urlparse(self.path)
            route = _normalize_route(parsed.path)
            query = parse_qs(parsed.query)
            sources = self._load_sources_safe()

            try:
                if route == "/summary-export":
                    self._send_summary_export(query)
                    return
                if route == "/summarize":
                    selected_source = _first(query, "source").strip() or None
                    self._send_html(
                        render_page(
                            active_page="summarize",
                            sources=sources,
                            selected_source=selected_source,
                            cached_summary=self._library_store().get_summary(selected_source) if selected_source else None,
                        )
                    )
                    return
                if route == "/configuration":
                    self._send_html(
                        render_page(
                            active_page="configuration",
                            sources=sources,
                            configured_paths=self._library_store().list_paths(),
                            vector_store_path=args.vector_store,
                            library_store_path=args.library_store,
                        )
                    )
                    return
                if route == "/ask":
                    self._send_html(render_page(active_page="ask", sources=sources))
                    return
                if route == "/extract-text":
                    self._send_html(render_page(active_page="extract-text", sources=sources))
                    return
                if route == "/overview":
                    self._send_html(
                        render_page(
                            active_page="overview",
                            sources=sources,
                            configured_paths=self._library_store().list_paths(),
                            vector_store_path=args.vector_store,
                            library_store_path=args.library_store,
                        )
                    )
                    return
                self._send_html(render_page(active_page="overview", sources=sources, error="Unknown page."))
            except Exception as exc:
                self._send_html(render_page(active_page="overview", sources=sources, error=str(exc)))

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
            route = _normalize_route(urlparse(self.path).path)

            try:
                if route == "/retrieve":
                    results = self._retrieve(question, top_k=top_k, source=source)
                    self._send_html(
                        render_page(
                            active_page="ask",
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            retrieval_results=results,
                        )
                    )
                    return
                if route == "/ask":
                    answer = self._answer(question, top_k=top_k, source=source)
                    self._send_html(
                        render_page(
                            active_page="ask",
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            answer=answer,
                        )
                    )
                    return
                if route == "/summarize":
                    progress: list[str] = []
                    summary = self._summarize(source, progress=progress)
                    cached_summary = self._cache_summary(summary)
                    self._send_html(
                        render_page(
                            active_page="summarize",
                            sources=self._load_sources(),
                            question=question,
                            selected_source=source,
                            top_k=top_k,
                            summary=summary,
                            cached_summary=cached_summary,
                            progress_messages=progress,
                        )
                    )
                    return
                if route in {"/extract-text", "/extract-text-export"}:
                    ocr_options = OcrOptions(
                        enabled=use_ocr,
                        language=ocr_language,
                        scale=ocr_scale,
                        psm=ocr_psm,
                        preprocess=ocr_preprocess,
                        clean_text=ocr_clean_text,
                    )
                    documents = self._extract_text(extract_path, ocr_options=ocr_options)
                    if route == "/extract-text-export":
                        self._send_text_download("extracted-text.txt", format_extracted_text(documents))
                        return
                    self._send_html(
                        render_page(
                            active_page="extract-text",
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
                if route == "/configuration/add-path":
                    self._library_store().add_path(_first(form, "document_path").strip())
                    self._send_configuration(message="Document path added.")
                    return
                if route == "/configuration/remove-path":
                    self._library_store().remove_path(_first(form, "document_path").strip())
                    self._send_configuration(message="Document path removed.")
                    return
                if route == "/configuration/ingest-path":
                    progress = self._ingest_path(_first(form, "document_path").strip())
                    self._send_configuration(progress=progress)
                    return
                if route == "/configuration/summarize-source":
                    progress: list[str] = []
                    summary = self._summarize(source, progress=progress)
                    self._cache_summary(summary)
                    self._send_configuration(
                        selected_source=source,
                        message="Summary cached.",
                        progress=progress,
                    )
                    return
                if route == "/configuration/remove-summary":
                    self._library_store().remove_summary(source or "")
                    self._send_configuration(selected_source=source, message="Cached summary removed.")
                    return
                self._send_html(render_page(active_page="overview", error="Unknown action."))
            except Exception as exc:
                active_page = _page_for_route(route)
                self._send_html(render_page(active_page=active_page, sources=self._load_sources_safe(), error=str(exc)))

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

        def _library_store(self) -> LibraryStore:
            return LibraryStore(args.library_store)

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

        def _summarize(self, source: str | None, progress: list[str] | None = None) -> SummaryResult:
            if not source:
                raise ValueError("Select one indexed source before summarizing.")
            llm_client = OllamaLlmClient(
                model=args.llm_model,
                host=args.ollama_host,
                temperature=args.temperature,
            )
            chunks = self._vector_store().get_chunks_by_source(source)
            return DocumentSummarizer(llm_client=llm_client).summarize(chunks, progress_callback=progress.append if progress is not None else None)

        def _extract_text(self, path: str, ocr_options: OcrOptions) -> list[Document]:
            if not path:
                raise ValueError("Enter a local file or folder path before extracting text.")
            return load_documents(path, ocr_options=ocr_options)

        def _ingest_path(self, path: str) -> list[str]:
            if not path:
                raise ValueError("Enter a local file or folder path before ingestion.")
            progress = [f"Loading documents from {path}."]
            documents = load_documents(path)
            progress.append(f"Loaded {len(documents)} document item{'s' if len(documents) != 1 else ''}.")
            chunks = split_documents(
                documents,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            progress.append(f"Created {len(chunks)} chunk{'s' if len(chunks) != 1 else ''}.")
            self._vector_store().add_chunks(chunks)
            progress.append(f"Stored {len(chunks)} chunk{'s' if len(chunks) != 1 else ''} in the vector DB.")
            return progress

        def _cache_summary(self, summary: SummaryResult) -> CachedSummary:
            if not summary.source_chunks:
                raise ValueError("No source chunks were available to cache.")
            first_chunk = summary.source_chunks[0]
            cached = CachedSummary(
                source_path=str(first_chunk.source_path),
                file_name=first_chunk.file_name,
                summary=summary.summary,
                model=summary.model,
                source_count=len(summary.sources),
                partial_summary_count=len(summary.partial_summaries),
            )
            self._library_store().save_summary(cached)
            return cached

        def _send_configuration(
            self,
            selected_source: str | None = None,
            message: str | None = None,
            progress: list[str] | None = None,
        ) -> None:
            self._send_html(
                render_page(
                    active_page="configuration",
                    sources=self._load_sources_safe(),
                    selected_source=selected_source,
                    configured_paths=self._library_store().list_paths(),
                    vector_store_path=args.vector_store,
                    library_store_path=args.library_store,
                    message=message,
                    progress_messages=progress or [],
                )
            )

        def _send_summary_export(self, query: dict[str, list[str]]) -> None:
            source = _first(query, "source").strip()
            export_format = _first(query, "format").strip().lower() or "md"
            cached = self._library_store().get_summary(source)
            if cached is None:
                raise ValueError("No cached summary found for export.")
            if export_format not in {"md", "txt"}:
                raise ValueError("Summary export format must be md or txt.")
            suffix = "md" if export_format == "md" else "txt"
            file_name = f"{Path(cached.file_name).stem}-summary.{suffix}"
            content = format_cached_summary(cached, export_format=export_format)
            self._send_text_download(file_name, content)

        def _send_text_download(self, file_name: str, content: str) -> None:
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{file_name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RagUiHandler


def render_page(
    active_page: str = "overview",
    sources: list[IndexedSource] | None = None,
    question: str = "",
    selected_source: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    retrieval_results: list[RetrievalResult] | None = None,
    answer: RagAnswer | None = None,
    summary: SummaryResult | None = None,
    cached_summary: CachedSummary | None = None,
    extract_path: str = "",
    ocr_options: OcrOptions | None = None,
    extracted_documents: list[Document] | None = None,
    configured_paths: list[ConfiguredPath] | None = None,
    vector_store_path: Path | None = None,
    library_store_path: Path | None = None,
    message: str | None = None,
    progress_messages: list[str] | None = None,
    error: str | None = None,
) -> str:
    """Render the local UI shell and selected page."""

    sources = sources or []
    configured_paths = configured_paths or []
    progress_messages = progress_messages or []
    page_content = render_page_content(
        active_page=active_page,
        sources=sources,
        question=question,
        selected_source=selected_source,
        top_k=top_k,
        retrieval_results=retrieval_results,
        answer=answer,
        summary=summary,
        cached_summary=cached_summary,
        extract_path=extract_path,
        ocr_options=ocr_options,
        extracted_documents=extracted_documents,
        configured_paths=configured_paths,
        vector_store_path=vector_store_path,
        library_store_path=library_store_path,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local RAG Assistant</title>
  <style>{_styles()}</style>
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <h1>Local RAG Assistant</h1>
        <p>{len(sources)} indexed source{'' if len(sources) == 1 else 's'} | {sum(source.chunk_count for source in sources)} chunks</p>
      </div>
      {render_nav(active_page)}
    </header>
    {render_message(message)}
    {render_error(error)}
    {render_progress(progress_messages)}
    {page_content}
  </main>
</body>
</html>"""


def render_page_content(
    active_page: str,
    sources: list[IndexedSource],
    question: str,
    selected_source: str | None,
    top_k: int,
    retrieval_results: list[RetrievalResult] | None,
    answer: RagAnswer | None,
    summary: SummaryResult | None,
    cached_summary: CachedSummary | None,
    extract_path: str,
    ocr_options: OcrOptions | None,
    extracted_documents: list[Document] | None,
    configured_paths: list[ConfiguredPath],
    vector_store_path: Path | None,
    library_store_path: Path | None,
) -> str:
    if active_page == "ask":
        return (
            render_query_form(sources, question, selected_source, top_k)
            + render_answer(answer)
            + render_retrieval_results(retrieval_results)
        )
    if active_page == "summarize":
        return render_summary_page(sources, selected_source, cached_summary, summary)
    if active_page == "extract-text":
        return render_extract_form(extract_path, ocr_options) + render_extracted_documents(extracted_documents)
    if active_page == "configuration":
        return render_configuration_page(
            sources=sources,
            configured_paths=configured_paths,
            selected_source=selected_source,
            vector_store_path=vector_store_path,
            library_store_path=library_store_path,
        )
    return render_overview(sources, configured_paths, vector_store_path, library_store_path)


def render_nav(active_page: str) -> str:
    items = [
        ("overview", "/overview", "Overview"),
        ("ask", "/ask", "Ask"),
        ("summarize", "/summarize", "Summarize"),
        ("extract-text", "/extract-text", "Extract Text"),
        ("configuration", "/configuration", "Configuration"),
    ]
    links = []
    for page, href, label in items:
        current = " class=\"active\"" if page == active_page else ""
        links.append(f'<a href="{href}"{current}>{label}</a>')
    return f"<nav>{''.join(links)}</nav>"


def render_overview(
    sources: list[IndexedSource],
    configured_paths: list[ConfiguredPath],
    vector_store_path: Path | None,
    library_store_path: Path | None,
) -> str:
    total_chunks = sum(source.chunk_count for source in sources)
    total_pages = sum(source.page_count or 0 for source in sources)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return f"""
    <section class="overview-grid">
      <article class="metric"><h2>Documents</h2><p>{len(sources)}</p></article>
      <article class="metric"><h2>Chunks</h2><p>{total_chunks}</p></article>
      <article class="metric"><h2>Pages</h2><p>{total_pages or 'n/a'}</p></article>
      <article class="metric"><h2>Paths</h2><p>{len(configured_paths)}</p></article>
    </section>
    <section class="quick-links">
      <a href="/ask">Ask questions</a>
      <a href="/summarize">Review summaries</a>
      <a href="/extract-text">Extract text</a>
      <a href="/configuration">Manage library</a>
    </section>
    <section>
      <h2>Library</h2>
      <dl class="details">
        <dt>Supported files</dt><dd>{escape(supported)}</dd>
        <dt>Vector store</dt><dd>{escape(_display_path(vector_store_path))}</dd>
        <dt>Library store</dt><dd>{escape(_display_path(library_store_path))}</dd>
      </dl>
    </section>
    {render_sources(sources)}"""


def render_summary_page(
    sources: list[IndexedSource],
    selected_source: str | None,
    cached_summary: CachedSummary | None,
    generated_summary: SummaryResult | None,
) -> str:
    current_summary = cached_summary
    if generated_summary is not None and generated_summary.source_chunks:
        first_chunk = generated_summary.source_chunks[0]
        current_summary = CachedSummary(
            source_path=str(first_chunk.source_path),
            file_name=first_chunk.file_name,
            summary=generated_summary.summary,
            model=generated_summary.model,
            source_count=len(generated_summary.sources),
            partial_summary_count=len(generated_summary.partial_summaries),
        )

    selected = selected_source or (current_summary.source_path if current_summary else None)
    export_links = ""
    if current_summary is not None:
        params = urlencode({"source": current_summary.source_path, "format": "md"})
        txt_params = urlencode({"source": current_summary.source_path, "format": "txt"})
        export_links = (
            '<div class="actions compact">'
            f'<a class="button" href="/summary-export?{params}">Export .md</a>'
            f'<a class="button secondary" href="/summary-export?{txt_params}">Export .txt</a>'
            "</div>"
        )

    return f"""
    <section class="query">
      <form method="post" action="/summarize">
        {render_source_selector(sources, selected, label="Document", required=True)}
        <div class="actions">
          <button class="secondary" formaction="/summarize" formmethod="get" type="submit">View Cached Summary</button>
          <button type="submit">Generate / Update Summary</button>
        </div>
      </form>
    </section>
    {render_cached_summary(current_summary)}
    {export_links}"""


def render_configuration_page(
    sources: list[IndexedSource],
    configured_paths: list[ConfiguredPath],
    selected_source: str | None,
    vector_store_path: Path | None,
    library_store_path: Path | None,
) -> str:
    path_rows = []
    for configured_path in configured_paths:
        value = escape(configured_path.path)
        path_rows.append(
            "<tr>"
            f"<td>{value}</td>"
            "<td class=\"row-actions\">"
            '<form method="post" action="/configuration/ingest-path">'
            f'<input type="hidden" name="document_path" value="{value}">'
            '<button type="submit">Ingest</button>'
            "</form>"
            '<form method="post" action="/configuration/remove-path">'
            f'<input type="hidden" name="document_path" value="{value}">'
            '<button class="secondary" type="submit">Remove</button>'
            "</form>"
            "</td>"
            "</tr>"
        )
    paths_table = (
        "<table><thead><tr><th>Path</th><th>Actions</th></tr></thead>"
        f"<tbody>{''.join(path_rows)}</tbody></table>"
        if path_rows
        else "<p>No configured document paths yet.</p>"
    )
    return f"""
    <section class="query">
      <form method="post" action="/configuration/add-path">
        <label>
          Document Path
          <input name="document_path" placeholder="data/raw or D:\\Docs\\paper.pdf">
        </label>
        <div class="actions">
          <button type="submit">Add Path</button>
        </div>
      </form>
    </section>
    <section>
      <h2>Configured Paths</h2>
      {paths_table}
    </section>
    <section class="query">
      <form method="post" action="/configuration/summarize-source">
        {render_source_selector(sources, selected_source, label="Summary Cache", required=True)}
        <div class="actions">
          <button type="submit">Create / Update Cached Summary</button>
          <button class="secondary" formaction="/configuration/remove-summary" type="submit">Remove Cached Summary</button>
        </div>
      </form>
    </section>
    <section>
      <h2>Storage</h2>
      <dl class="details">
        <dt>Vector store</dt><dd>{escape(_display_path(vector_store_path))}</dd>
        <dt>Library store</dt><dd>{escape(_display_path(library_store_path))}</dd>
      </dl>
    </section>"""


def render_source_selector(
    sources: list[IndexedSource],
    selected_source: str | None,
    label: str = "Source",
    required: bool = False,
) -> str:
    source_options = ['<option value="">All sources</option>' if not required else '<option value="">Select source</option>']
    for source in sources:
        value = escape(str(source.source_path))
        selected = " selected" if selected_source in {str(source.source_path), source.file_name} else ""
        source_options.append(f'<option value="{value}"{selected}>{escape(source.file_name)}</option>')
    return f"""
        <label>
          {escape(label)}
          <select name="source">{''.join(source_options)}</select>
        </label>"""


def render_query_form(
    sources: list[IndexedSource],
    question: str,
    selected_source: str | None,
    top_k: int,
) -> str:
    return f"""
    <section class="query">
      <form method="post">
        <label>
          Question
          <input name="question" value="{escape(question)}" placeholder="Ask about your indexed documents" autofocus>
        </label>
        <div class="controls">
          {render_source_selector(sources, selected_source)}
          <label>
            Top K
            <input class="small" name="top_k" type="number" min="1" max="20" value="{top_k}">
          </label>
        </div>
        <div class="actions">
          <button formaction="/retrieve" type="submit">Retrieve</button>
          <button formaction="/ask" type="submit">Ask</button>
        </div>
      </form>
    </section>"""


def render_extract_form(extract_path: str = "", ocr_options: OcrOptions | None = None) -> str:
    options = ocr_options or OcrOptions(enabled=True)
    checked = " checked" if options.enabled else ""
    preprocess_checked = " checked" if options.preprocess else ""
    clean_checked = " checked" if options.clean_text else ""
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return f"""
    <section class="query">
      <form method="post" action="/extract-text">
        <label>
          Extract Text Path
          <input name="extract_path" value="{escape(extract_path)}" placeholder="data/raw/example.pdf">
        </label>
        <p class="meta">Supported file types: {escape(supported)}</p>
        <p class="meta">Enable OCR for scanned PDFs or image-only documents.</p>
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
          <button class="secondary" formaction="/extract-text-export" type="submit">Export .txt</button>
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


def render_cached_summary(summary: CachedSummary | None) -> str:
    if summary is None:
        return "<section><h2>Cached Summary</h2><p>No cached summary selected.</p></section>"
    return f"""
    <section>
      <h2>Cached Summary</h2>
      <p class="meta">{escape(summary.file_name)} | {escape(summary.model)} | partial summaries {summary.partial_summary_count}</p>
      <div class="summary-text">{format_summary_html(summary.summary)}</div>
    </section>"""


def render_message(message: str | None) -> str:
    if not message:
        return ""
    return f'<section class="message"><p>{escape(message)}</p></section>'


def render_progress(messages: list[str]) -> str:
    if not messages:
        return ""
    items = "".join(f"<li>{escape(message)}</li>" for message in messages)
    return f"<section class=\"progress\"><h2>Progress</h2><ol>{items}</ol></section>"


def render_error(error: str | None) -> str:
    if not error:
        return ""
    return f'<section class="error"><h2>Error</h2><p>{escape(error)}</p></section>'


def format_summary_html(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        return "<p>[empty summary]</p>"
    return "".join(f"<p>{escape(block).replace(chr(10), '<br>')}</p>" for block in blocks)


def format_cached_summary(summary: CachedSummary, export_format: str = "md") -> str:
    if export_format == "txt":
        return "\n".join(
            [
                f"Summary: {summary.file_name}",
                f"Model: {summary.model}",
                "",
                summary.summary.strip(),
                "",
            ]
        )
    return "\n".join(
        [
            f"# Summary: {summary.file_name}",
            "",
            f"- Model: {summary.model}",
            f"- Partial summaries: {summary.partial_summary_count}",
            "",
            summary.summary.strip(),
            "",
        ]
    )


def format_extracted_text(documents: list[Document]) -> str:
    if not documents:
        return ""

    source_paths = []
    for document in documents:
        source_path = str(document.source_path)
        if source_path not in source_paths:
            source_paths.append(source_path)

    sections = []
    for document in documents:
        title = f"# Page {document.page_number}" if document.page_number is not None else "# Text"
        sections.append(
            "\n".join(
                [
                    title,
                    "",
                    document.text.strip() or "[no text extracted]",
                ]
            )
        )

    header = "\n".join(f"Source: {source_path}" for source_path in source_paths)
    return "\n\n".join([header, *sections])


def _normalize_route(path: str) -> str:
    if path == "/":
        return "/overview"
    return path.rstrip("/") or "/overview"


def _page_for_route(route: str) -> str:
    if route in {"/ask", "/retrieve"}:
        return "ask"
    if route.startswith("/summarize") or route == "/summary-export":
        return "summarize"
    if route.startswith("/extract-text"):
        return "extract-text"
    if route.startswith("/configuration"):
        return "configuration"
    return "overview"


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


def _display_path(path: Path | None) -> str:
    if path is None:
        return "default"
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _styles() -> str:
    return """
:root { color-scheme: light; font-family: Arial, sans-serif; }
body { margin: 0; background: #f4f6f8; color: #202124; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
header { display: flex; justify-content: space-between; align-items: end; gap: 16px; border-bottom: 1px solid #d5dbe3; padding-bottom: 16px; }
h1 { font-size: 28px; margin: 0; font-weight: 700; }
h2 { font-size: 18px; margin: 0 0 14px; }
h3 { font-size: 15px; margin: 0 0 6px; }
p { line-height: 1.5; }
nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
nav a, .quick-links a, .button { border: 1px solid #2f3a45; color: #202124; background: #ffffff; padding: 9px 12px; text-decoration: none; font-size: 14px; }
nav a.active { background: #2f3a45; color: #ffffff; }
section { margin-top: 22px; }
.query { background: #ffffff; border: 1px solid #d5dbe3; padding: 18px; }
label { display: grid; gap: 6px; font-size: 13px; font-weight: 700; }
input, select { box-sizing: border-box; width: 100%; border: 1px solid #b9bbb5; padding: 10px; font: inherit; background: #ffffff; }
input.small { max-width: 110px; }
.controls { display: grid; grid-template-columns: minmax(220px, 1fr) 120px; gap: 14px; margin-top: 14px; }
.ocr-controls { grid-template-columns: minmax(220px, 1fr) 120px 120px; }
.actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.actions.compact { margin-top: 10px; }
button { border: 1px solid #202124; background: #202124; color: #ffffff; padding: 10px 14px; font: inherit; cursor: pointer; }
button.secondary, .button.secondary { background: #ffffff; color: #202124; }
.checkbox { display: flex; align-items: center; gap: 8px; font-weight: 700; }
.checkbox input { width: auto; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #fafafa; border: 1px solid #e7e7e2; padding: 12px; max-height: 420px; overflow: auto; }
table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d5dbe3; }
th, td { text-align: left; border-bottom: 1px solid #e7e7e2; padding: 9px; vertical-align: top; font-size: 14px; }
article { background: #ffffff; border: 1px solid #d5dbe3; padding: 14px; margin-top: 10px; }
.overview-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.metric p { font-size: 28px; font-weight: 700; margin: 0; }
.quick-links { display: flex; flex-wrap: wrap; gap: 10px; }
.details { display: grid; grid-template-columns: 150px 1fr; gap: 8px 14px; background: #ffffff; border: 1px solid #d5dbe3; padding: 14px; }
.details dt { font-weight: 700; }
.details dd { margin: 0; overflow-wrap: anywhere; }
.row-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.row-actions form { margin: 0; }
.summary-text { background: #ffffff; border: 1px solid #d5dbe3; padding: 16px; }
.meta { color: #60635d; font-size: 13px; margin: 0 0 8px; }
.error { border-left: 4px solid #a6342e; background: #fff4f2; padding: 14px; }
.message { border-left: 4px solid #2d6a4f; background: #eff8f3; padding: 12px 14px; }
.message p { margin: 0; }
.progress { border-left: 4px solid #3b5b92; background: #eef4ff; padding: 14px; }
.progress ol { margin: 0; padding-left: 22px; }
@media (max-width: 760px) {
  main { padding: 18px; }
  header, .controls, .ocr-controls, .overview-grid, .details { display: grid; grid-template-columns: 1fr; }
  nav { justify-content: flex-start; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
