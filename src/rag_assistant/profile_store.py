"""JSON-backed profile configuration for scenario-specific RAG modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from rag_assistant.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_PROFILE, PROFILE_STORE_PATH


PROFILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
KNOWN_PROMPT_STYLES = {"general", "technical", "recipes", "research", "legal"}


@dataclass(frozen=True)
class RagProfile:
    """A named RAG behavior and source grouping."""

    name: str
    description: str = ""
    paths: tuple[str, ...] = ()
    prompt_style: str = "general"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class ProfileStore:
    """Persist and retrieve local RAG profiles."""

    def __init__(self, path: str | Path = PROFILE_STORE_PATH) -> None:
        self.path = Path(path)

    def list_profiles(self) -> list[RagProfile]:
        profiles = [_profile_from_dict(item) for item in self._read().get("profiles", [])]
        if not any(profile.name == DEFAULT_PROFILE for profile in profiles):
            profiles.append(default_profile())
        return sorted(profiles, key=lambda profile: profile.name.lower())

    def get_profile(self, name: str | None) -> RagProfile:
        profile_name = normalize_profile_name(name)
        for profile in self.list_profiles():
            if profile.name == profile_name:
                return profile
        raise ValueError(f"Unknown profile: {profile_name}")

    def ensure_profile(self, name: str | None) -> RagProfile:
        profile_name = normalize_profile_name(name)
        for profile in self.list_profiles():
            if profile.name == profile_name:
                return profile
        profile = RagProfile(
            name=profile_name,
            description=f"{profile_name} RAG profile.",
            prompt_style=default_prompt_style(profile_name),
        )
        self.save_profile(profile)
        return profile

    def save_profile(self, profile: RagProfile) -> None:
        validate_profile(profile)
        data = self._read()
        profiles = data.setdefault("profiles", [])
        profile_dict = asdict(profile)
        profile_dict["paths"] = list(profile.paths)
        for index, item in enumerate(profiles):
            if str(item.get("name")) == profile.name:
                profiles[index] = profile_dict
                break
        else:
            profiles.append(profile_dict)
        profiles.sort(key=lambda item: str(item.get("name", "")).lower())
        self._write(data)

    def add_path(self, profile_name: str | None, path: str | Path) -> RagProfile:
        profile = self.ensure_profile(profile_name)
        normalized_path = str(Path(path))
        if not normalized_path.strip():
            return profile
        paths = tuple(sorted({*profile.paths, normalized_path}, key=str.lower))
        updated = RagProfile(
            name=profile.name,
            description=profile.description,
            paths=paths,
            prompt_style=profile.prompt_style,
            chunk_size=profile.chunk_size,
            chunk_overlap=profile.chunk_overlap,
        )
        self.save_profile(updated)
        return updated

    def remove_path(self, profile_name: str | None, path: str | Path) -> RagProfile:
        profile = self.get_profile(profile_name)
        normalized_path = str(Path(path))
        paths = tuple(path_item for path_item in profile.paths if path_item != normalized_path)
        updated = RagProfile(
            name=profile.name,
            description=profile.description,
            paths=paths,
            prompt_style=profile.prompt_style,
            chunk_size=profile.chunk_size,
            chunk_overlap=profile.chunk_overlap,
        )
        self.save_profile(updated)
        return updated

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"profiles": [asdict(default_profile()) | {"paths": []}]}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"profiles": [asdict(default_profile()) | {"paths": []}]}
        if not isinstance(data, dict):
            return {"profiles": [asdict(default_profile()) | {"paths": []}]}
        data.setdefault("profiles", [])
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def default_profile() -> RagProfile:
    return RagProfile(
        name=DEFAULT_PROFILE,
        description="General-purpose RAG profile.",
        prompt_style="general",
    )


def default_prompt_style(profile_name: str) -> str:
    normalized_name = profile_name.strip().lower()
    return normalized_name if normalized_name in KNOWN_PROMPT_STYLES else "general"


def normalize_profile_name(name: str | None) -> str:
    profile_name = (name or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    if not PROFILE_NAME_PATTERN.match(profile_name):
        raise ValueError(
            "Profile names must start with a letter or number and contain only letters, "
            "numbers, underscores, or hyphens."
        )
    return profile_name


def validate_profile(profile: RagProfile) -> None:
    normalize_profile_name(profile.name)
    if profile.chunk_size <= 0:
        raise ValueError("profile chunk_size must be greater than zero")
    if profile.chunk_overlap < 0:
        raise ValueError("profile chunk_overlap must be zero or greater")
    if profile.chunk_overlap >= profile.chunk_size:
        raise ValueError("profile chunk_overlap must be smaller than chunk_size")


def _profile_from_dict(item: dict[str, Any]) -> RagProfile:
    profile = RagProfile(
        name=normalize_profile_name(str(item.get("name", DEFAULT_PROFILE))),
        description=str(item.get("description", "")),
        paths=tuple(str(path) for path in item.get("paths", []) if str(path).strip()),
        prompt_style=str(item.get("prompt_style", "general")),
        chunk_size=int(item.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(item.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
    )
    validate_profile(profile)
    return profile
