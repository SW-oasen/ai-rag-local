from pathlib import Path

import pytest

from rag_assistant.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from rag_assistant.profile_store import ProfileStore, RagProfile, default_prompt_style, normalize_profile_name


def test_profile_store_returns_default_profile_when_file_is_missing(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    profiles = store.list_profiles()

    assert [profile.name for profile in profiles] == ["general"]
    assert profiles[0].chunk_size == DEFAULT_CHUNK_SIZE
    assert profiles[0].chunk_overlap == DEFAULT_CHUNK_OVERLAP


def test_profile_store_saves_and_loads_profile(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")
    store.save_profile(
        RagProfile(
            name="technical",
            description="Technical documents",
            paths=("data/raw/tech",),
            prompt_style="technical",
            chunk_size=800,
            chunk_overlap=100,
        )
    )

    profile = store.get_profile("technical")

    assert profile.description == "Technical documents"
    assert profile.paths == ("data/raw/tech",)
    assert profile.prompt_style == "technical"
    assert profile.chunk_size == 800
    assert profile.chunk_overlap == 100
    assert [item.name for item in store.list_profiles()] == ["general", "technical"]


def test_profile_store_ensures_new_profile(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    profile = store.ensure_profile("recipes")

    assert profile.name == "recipes"
    assert profile.prompt_style == "recipes"
    assert store.get_profile("recipes").description == "recipes RAG profile."


def test_profile_store_adds_and_removes_profile_paths(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    updated = store.add_path("technical", "data/raw/tech")
    expected_path = str(Path("data/raw/tech"))

    assert updated.name == "technical"
    assert updated.paths == (expected_path,)
    assert store.get_profile("technical").paths == (expected_path,)

    updated = store.remove_path("technical", "data/raw/tech")

    assert updated.paths == ()


def test_default_prompt_style_uses_known_profile_names() -> None:
    assert default_prompt_style("technical") == "technical"
    assert default_prompt_style("recipes") == "recipes"
    assert default_prompt_style("unknown") == "general"


def test_profile_store_rejects_invalid_profile_names() -> None:
    with pytest.raises(ValueError, match="Profile names"):
        normalize_profile_name("bad profile")


def test_profile_store_validates_chunk_overlap(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    with pytest.raises(ValueError, match="chunk_overlap"):
        store.save_profile(RagProfile(name="bad", chunk_size=100, chunk_overlap=100))
