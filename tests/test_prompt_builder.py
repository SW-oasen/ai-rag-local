from pathlib import Path

from rag_assistant.prompt_builder import build_rag_prompt
from rag_assistant.schema import RetrievalResult, TextChunk


def test_build_rag_prompt_includes_question_context_and_sources() -> None:
    chunk = TextChunk(
        text="The project uses local embeddings and ChromaDB.",
        source_path=Path("docs/readme.md"),
        file_name="readme.md",
        document_type="md",
        chunk_index=2,
        start_char=10,
        end_char=55,
        page_number=None,
    )

    prompt = build_rag_prompt(
        "Which vector database is used?",
        [RetrievalResult(chunk=chunk, score=0.12)],
    )

    assert "Which vector database is used?" in prompt
    assert "The project uses local embeddings and ChromaDB." in prompt
    assert "[source 1: readme.md, chunk 2, score 0.1200]" in prompt
    assert "If the context is insufficient" in prompt
    assert "Format the answer as Markdown" in prompt


def test_build_rag_prompt_handles_empty_context() -> None:
    prompt = build_rag_prompt("What is missing?", [])

    assert "No relevant context was retrieved." in prompt
    assert "What is missing?" in prompt


def test_build_rag_prompt_includes_technical_style_rules() -> None:
    prompt = build_rag_prompt("How do I configure it?", [], prompt_style="technical")

    assert "For technical answers" in prompt
    assert "prerequisites, procedure, verification, and troubleshooting" in prompt


def test_build_rag_prompt_includes_recipe_style_rules() -> None:
    prompt = build_rag_prompt("Welche Suppen gibt es?", [], prompt_style="recipes")

    assert "For recipe or cooking answers" in prompt
    assert "Use nested lists for groups" in prompt
