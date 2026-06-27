"""Prompt construction for source-aware RAG answers."""

from rag_assistant.schema import RetrievalResult


def build_rag_prompt(question: str, retrieval_results: list[RetrievalResult]) -> str:
    """Build a prompt that asks the LLM to answer from retrieved context."""

    context = _format_context(retrieval_results)
    return f"""You are a local RAG assistant answering questions from provided context.

Rules:
- Answer using only the context when possible.
- If the context is insufficient, say that the available context is insufficient.
- Cite sources using the source labels from the context, such as [source 1].
- Answer in the same language as the user question when possible.
- Keep the answer concise and technically accurate.
- Format the answer as Markdown. Use short headings, bullet lists, numbered steps, or tables when they make the answer easier to read.
- Do not wrap the whole answer in a code block.

Context:
{context}

Question:
{question}

Answer:
"""


def _format_context(retrieval_results: list[RetrievalResult]) -> str:
    if not retrieval_results:
        return "No relevant context was retrieved."

    blocks: list[str] = []
    for index, result in enumerate(retrieval_results, start=1):
        chunk = result.chunk
        location_parts = [chunk.file_name, f"chunk {chunk.chunk_index}"]
        if chunk.page_number is not None:
            location_parts.append(f"page {chunk.page_number}")
        score = f", score {result.score:.4f}" if result.score is not None else ""
        label = f"[source {index}: {', '.join(location_parts)}{score}]"
        blocks.append(f"{label}\n{chunk.text}")

    return "\n\n".join(blocks)
