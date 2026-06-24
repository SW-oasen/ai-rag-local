# Local RAG Assistant with Local LLM

A practical Retrieval-Augmented Generation (RAG) project that runs locally on an AI PC.  
The goal is to build a document-based assistant that can ingest local files, create embeddings, store searchable chunks in a vector database, retrieve relevant context, and answer questions using a local open-source LLM.

## Current Status

Implemented:

- Source-aware loading for Markdown, text, and simple PDF files
- Overlapping text chunking with metadata preservation
- Ollama embedding provider wrapper
- ChromaDB-backed local vector store
- Retriever service for top-k semantic search
- Source-aware RAG prompt builder
- Ollama local LLM client wrapper
- RAG pipeline returning answer, sources, retrieved chunks, model, and prompt
- CLI commands for ingestion, retrieval, and local question answering
- Dedicated map-reduce document summarization pipeline
- Sentence/word-aware chunk overlap starts to avoid mid-word fragments
- Starter retrieval evaluation examples
- Unit tests for loading, chunking, retrieval, prompt construction, pipeline behavior, summarization, and CLI formatting

Recommended local models:

- Generation: `qwen3-coder:30b` for quality, `qwen2.5-coder:7b` for faster iteration
- Embeddings: `nomic-embed-text`

Install the embedding model with Ollama:

```powershell
ollama pull nomic-embed-text
```

Install project dependencies into the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

If Windows certificate verification blocks PyPI, retry with:

```powershell
.\.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .[dev]
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

Start the local browser UI:

```powershell
.\.venv\Scripts\rag-assistant-ui.exe
```

Then open:

```text
http://127.0.0.1:8765
```

## CLI Usage

Ingest documents from a file or directory:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw
```

For larger document folders, use a smaller embedding batch size:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw --embedding-batch-size 8
```

For scanned PDFs, install OCR extras and the Tesseract application, then enable OCR:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[ocr]
winget install UB-Mannheim.TesseractOCR
```

Open a new terminal and verify:

```powershell
tesseract --version
```

OCR ingestion example:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw --ocr --ocr-language eng+deu --ocr-scale 3 --ocr-psm 6 --embedding-batch-size 4
```

OCR options:

- `--ocr-language`: Tesseract language, for example `eng`, `deu`, or `eng+deu`
- `--ocr-scale`: PDF render scale before OCR; try `3` or `4` for small/blurred text
- `--ocr-psm`: Tesseract page segmentation mode; try `6` for text blocks, `4` for columns, `11` for sparse text
- `--no-ocr-preprocess`: disable grayscale/contrast/threshold/sharpen preprocessing
- `--no-ocr-clean`: disable text cleanup for line wraps and hyphenated words

If ChromaDB reports a SQLite disk I/O error on the project drive, use a vector-store path on another local drive:

```powershell
$env:RAG_VECTOR_STORE_DIR = "$env:TEMP\local_rag_vector_store"
.\.venv\Scripts\rag-assistant.exe ingest data/raw --vector-store $env:RAG_VECTOR_STORE_DIR --embedding-batch-size 8
```

By default, the CLI stores ChromaDB data under:

```powershell
$env:TEMP\local_rag_assistant\vector_store
```

Retrieve relevant chunks without calling the LLM:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "What is this project about?"
```

Limit retrieval to one indexed source:

```powershell
.\.venv\Scripts\rag-assistant.exe retrieve "What evidence is in the office?" --source "Lions and Tigers and Snares.pdf"
```

List documents currently stored in the vector index:

```powershell
.\.venv\Scripts\rag-assistant.exe sources
```

Inspect stored chunks for one indexed source:

```powershell
.\.venv\Scripts\rag-assistant.exe chunks "Lions and Tigers and Snares.pdf" --limit 5 --preview-chars 180
```

Ask a question with retrieved context and the local LLM:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "What is this project about?"
```

Ask against one indexed source:

```powershell
.\.venv\Scripts\rag-assistant.exe ask "What evidence is in the office?" --source "Lions and Tigers and Snares.pdf"
```

Run retrieval evaluation examples against the current index:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/retrieval_eval_examples.md --top-k 5
```

The `eval` command takes a UTF-8 markdown file with questions and expected evidence, not the PDF or source document itself.

The starter evaluation file expects `README.md` to be indexed. If your vector store currently contains a different document set, create matching examples or index the project docs first:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest README.md
```

For the sample short story document:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/lions_tigers_retrieval_eval_examples.md --top-k 5
```

Limit retrieval evaluation to one indexed source:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/lions_tigers_retrieval_eval_examples.md --source "Lions and Tigers and Snares.pdf" --top-k 5
```

Write a detailed JSON evaluation report:

```powershell
.\.venv\Scripts\rag-assistant.exe eval examples/lions_tigers_retrieval_eval_examples.md --source "Lions and Tigers and Snares.pdf" --top-k 5 --json-report data/processed/lions_tigers_eval_report.json
```

Summarize a whole document without using top-k retrieval:

```powershell
.\.venv\Scripts\rag-assistant.exe summarize README.md --llm-model qwen2.5-coder:7b
```

Summarize chunks for a document that has already been indexed:

cpowershell
.\.venv\Scripts\rag-assistant.exe summarize README.md --from-index --vector-store $env:RAG_VECTOR_STORE_DIR --llm-model qwen2.5-coder:7b
```


---
start web app
```
.\.venv\Scripts\rag-assistant-ui.exe
```
call web site
http://127.0.0.1:8765

ingest files from folder
```
.\.venv\Scripts\rag-assistant.exe ingest data/raw --embedding-batch-size 4
```

ingest single file
```
.\.venv\Scripts\rag-assistant.exe ingest path\to\file.pdf --embedding-batch-size 4
```

check what is indexed
```
.\.venv\Scripts\rag-assistant.exe sources
```

check which model is in use on ollama
```bash
ollama ps
```


Command roles:

- `sources`: index inspection; shows indexed documents, chunk counts, and page counts
- `chunks`: source inspection; shows stored chunk/page boundaries and text previews
- `retrieve`: semantic search only; shows chunks and scores
- `ask`: top-k retrieval plus answer generation
- `summarize`: full-document summarization over all chunks for a selected source
- `eval`: retrieval quality check against expected source/evidence examples
- `rag-assistant-ui`: local browser UI for source inspection, retrieval, Q&A, and selected-source summarization

Retrieval quality notes:

- Chunking prefers paragraph, sentence, and word boundaries.
- Retrieval should be inspected with `retrieve` before tuning prompts.
- Starter evaluation examples live in `examples/retrieval_eval_examples.md`.

Useful options:

```powershell
.\.venv\Scripts\rag-assistant.exe ingest data/raw --chunk-size 900 --chunk-overlap 150
.\.venv\Scripts\rag-assistant.exe retrieve "retrieval quality" --top-k 5
.\.venv\Scripts\rag-assistant.exe retrieve "retrieval quality" --source README.md --top-k 5
.\.venv\Scripts\rag-assistant.exe ask "Summarize the documents" --llm-model qwen3-coder:30b --show-prompt
.\.venv\Scripts\rag-assistant.exe summarize README.md --question "implementation plan" --max-chunks-per-group 4
```



## 1. Project Goal

This project demonstrates a realistic local AI workflow:

- Ingest local documents
- Split documents into meaningful chunks
- Create embeddings locally
- Store embeddings in a vector database
- Retrieve relevant chunks for a user question
- Generate answers with a local LLM
- Show sources for every answer
- Provide a clean architecture that can later be extended into an AI agent

The project should be portfolio-ready and understandable for recruiters, technical reviewers, and future development.

## 2. Why This Project Matters

Many AI demos rely on external APIs. This project focuses on a local-first setup:

- Lower running cost
- More privacy for personal documents
- Better understanding of the full RAG pipeline
- Hands-on experience with embeddings, chunking, vector search, prompt design, and evaluation
- Foundation for future AI agent projects

## 3. Target Use Case

The assistant should answer questions over local documents such as:

- PDF files
- Markdown files
- Text files
- Project documentation
- Notes
- README files
- Technical reports

Example questions:

- "What is the main idea of this document?"
- "Which methods are mentioned?"
- "Summarize the implementation plan."
- "Which limitations are described?"
- "Where in the source files is this topic discussed?"

## 4. Planned Tech Stack

Initial recommended stack:

- Python
- Ollama or another local LLM runtime
- Local embedding model
- ChromaDB or Qdrant as vector database
- LangChain or LlamaIndex only if useful, not mandatory
- Streamlit or FastAPI for a simple interface
- pytest for testing
- uv or venv for environment management

The implementation should stay modular so individual parts can be replaced later.

## 5. Core Features

### Phase 1: Minimal RAG Pipeline

- Load local documents
- Extract text
- Split text into chunks
- Generate embeddings
- Store chunks and metadata in a vector database
- Retrieve top-k relevant chunks
- Generate answer with local LLM
- Display answer with source references

### Phase 2: Better Retrieval Quality

- Add metadata filtering
- Improve chunking strategy
- Add reranking if needed
- Show retrieved context before generation
- Add simple retrieval evaluation examples

### Phase 3: User Interface

- Build a simple local UI
- Upload or select documents
- Ask questions
- Display answer and sources
- Show retrieved chunks for debugging

### Phase 4: Evaluation and Portfolio Polish

- Add test documents
- Add example questions and expected answer notes
- Measure retrieval quality
- Document limitations
- Add screenshots
- Add architecture diagram
- Prepare project for GitHub portfolio

## 6. Suggested Project Structure

```text
local-rag-assistant/
├── README.md
├── project_context.md
├── codex_system_prompt.md
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── vector_store/
├── src/
│   └── rag_assistant/
│       ├── __init__.py
│       ├── config.py
│       ├── document_loader.py
│       ├── text_splitter.py
│       ├── embeddings.py
│       ├── vector_store.py
│       ├── retriever.py
│       ├── llm_client.py
│       ├── rag_pipeline.py
│       └── app.py
├── tests/
│   ├── test_text_splitter.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
└── notebooks/
```

## 7. Development Workflow

The project should be developed step by step:

1. Create project structure
2. Implement document loading
3. Implement text splitting
4. Implement embeddings
5. Implement vector storage
6. Implement retrieval
7. Connect local LLM
8. Add source-aware answer generation
9. Add tests
10. Add UI
11. Improve retrieval quality
12. Polish documentation

Each step should be tested before moving to the next step.

## 8. Quality Requirements

- Keep the code readable and modular
- Avoid unnecessary complexity
- Prefer small functions with clear responsibility
- Add type hints where useful
- Add docstrings for important modules and functions
- Write tests for core logic
- Do not commit large raw data, vector databases, model files, or local caches
- Always show source references in generated answers
- Be transparent when the retrieved context is insufficient

## 9. Git Ignore Recommendations

The following should not be committed:

```text
.venv/
__pycache__/
*.pyc
.env
data/raw/
data/processed/
vector_store/
models/
*.sqlite
*.db
.DS_Store
```

## 10. Future Extensions

Possible extensions:

- Multi-document comparison
- Table extraction
- PDF layout-aware parsing
- Local reranker
- Conversation memory
- Agent tools for file search and project navigation
- Evaluation dashboard
- Hybrid search with BM25 + vector search
- Support for German and English documents
- Integration into personal local AI organizer
