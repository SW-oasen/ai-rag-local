# Project Context: Local RAG Assistant with Local LLM

## 1. Background

The developer wants to build a realistic local RAG project using a local open-source LLM on an AI PC. The project should be implemented in VS Code with Codex as the main coding agent.

The developer has experience with Python, Java, machine learning projects, portfolio projects, GitHub, VS Code, and practical step-by-step learning. The implementation should therefore focus on clarity, real-world usefulness, and portfolio value.

This project should become a foundation for future AI agent projects.

## 2. Main Objective

Build a local document question-answering assistant.

The assistant should:

- Read local documents
- Convert them into searchable chunks
- Store embeddings locally
- Retrieve relevant context
- Answer questions using a local LLM
- Show source references
- Run locally without depending on paid external LLM APIs

## 3. Important Design Principles

### Local-first

The project should run on the user's AI PC. Avoid mandatory cloud dependencies.

### Step-by-step implementation

Do not build everything at once. Plan first, then implement and test one part at a time.

### Portfolio-ready

The project should be understandable from the README, screenshots, code structure, and examples.

### Modular architecture

Each part should be replaceable:

- LLM runtime
- embedding model
- vector database
- document parser
- UI framework

### Debuggability

The system should make retrieval visible. It should be possible to inspect:

- loaded documents
- generated chunks
- metadata
- retrieved chunks
- final prompt
- final answer

## 4. Initial Scope

The first version should be simple and reliable.

### In scope

- Markdown, text, and simple PDF support
- Local embedding generation
- Local vector database
- Basic semantic retrieval
- Local LLM answer generation
- Source display
- Basic tests
- Simple local UI or CLI

### Out of scope for first version

- Complex agent behavior
- Long-term memory
- OCR-heavy document parsing
- Full table understanding
- Complex PDF layout reconstruction
- Multi-user deployment
- Cloud hosting
- Advanced access control

These can be added later.

## 5. Suggested Implementation Phases

### Phase 0: Planning

Codex should inspect the repository, propose a concrete implementation plan, and wait for approval before making large changes.

Deliverables:

- Project structure
- Dependency proposal
- First milestone plan

### Phase 1: Core Data Pipeline

Implement:

- document loading
- text extraction
- chunking
- metadata creation
- basic tests

Expected result:

- Documents can be loaded and split into chunks with source metadata.

### Phase 2: Embeddings and Vector Store

Implement:

- local embedding model wrapper
- vector database initialization
- add documents to vector store
- similarity search
- retrieval tests

Expected result:

- User questions return relevant chunks.

### Phase 3: Local LLM Integration

Implement:

- local LLM client
- prompt template
- answer generation
- source-aware response format

Expected result:

- The assistant answers based on retrieved context and cites sources.

### Phase 4: Interface

Implement one of:

- CLI first
- Streamlit UI
- FastAPI backend with simple frontend later

Recommendation: start with CLI or Streamlit for faster validation.

### Phase 5: Evaluation and Improvement

Add:

- example documents
- example questions
- retrieval inspection
- simple evaluation notes
- limitations section
- screenshots for portfolio

## 6. Recommended Architecture

```text
User Question
     |
     v
Retriever
     |
     v
Top-k Relevant Chunks
     |
     v
Prompt Builder
     |
     v
Local LLM
     |
     v
Answer + Sources
```

Document ingestion:

```text
Documents
     |
     v
Loader
     |
     v
Text Splitter
     |
     v
Embedding Model
     |
     v
Vector Store
```

## 7. Coding Expectations for Codex

Codex should:

- Explain the planned change before implementing
- Prefer small commits or small logical changes
- Modify only necessary files
- Keep code simple and readable
- Add tests for core logic
- Run tests after implementation
- Report what changed, what was tested, and what remains open
- Avoid large rewrites unless explicitly requested
- Preserve working code
- Ask before changing architecture significantly

## 8. Retrieval Requirements

Every retrieved chunk should keep metadata:

- source file path
- file name
- chunk index
- page number if available
- character range if available
- document type

The final answer should include source references.

If the retrieved context is weak or insufficient, the assistant should say so instead of guessing.

## 9. Prompting Requirements

The RAG prompt should instruct the local LLM to:

- answer only from the provided context when possible
- say when the context is insufficient
- cite sources
- avoid hallucinating details
- answer in the user's language if possible

## 10. Testing Strategy

Start with small tests:

- text splitting does not lose content
- chunks include metadata
- vector store returns expected chunks for simple queries
- prompt builder includes context and question
- RAG pipeline returns an answer object with sources

Tests do not need to evaluate model intelligence at first. They should verify pipeline behavior.

## 11. Known Challenges

The project should explicitly acknowledge common RAG limitations:

- context length limits
- imperfect chunking
- weak retrieval for tables and figures
- PDF extraction problems
- hallucination risk
- local model quality differences
- slow inference on large models
- need for evaluation

## 12. Success Criteria

The first successful version is reached when:

- local documents can be indexed
- a user can ask a question
- relevant chunks are retrieved
- the local LLM generates an answer
- source references are shown
- the project can be run from clear README instructions
- core logic has basic tests


## 13. RAG Quality Requirements

The system must not work like a simple keyword search.

It should support:
- semantic retrieval, not only exact word matching
- paragraph-level and section-level understanding
- summarization of full documents
- summarization of selected sections
- question answering over multiple chunks
- source references for every answer
- transparent retrieval debugging

The system should distinguish between:
- keyword search
- semantic retrieval
- answer generation
- document summarization
- multi-step synthesis

For summarization tasks, the system should not rely only on top-k retrieval.
It should use a dedicated summarization flow, for example:
1. load all chunks of the selected document
2. summarize chunks or sections
3. merge partial summaries
4. create a final concise summary with limitations