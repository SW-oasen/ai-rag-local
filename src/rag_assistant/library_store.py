"""Small JSON-backed store for web UI document library state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfiguredPath:
    """A file or folder path managed from the web UI."""

    path: str


@dataclass(frozen=True)
class CachedSummary:
    """A generated document summary cached outside the vector index."""

    source_path: str
    file_name: str
    summary: str
    model: str
    source_count: int
    partial_summary_count: int


class LibraryStore:
    """Persist web UI paths and cached summaries in a local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list_paths(self) -> list[ConfiguredPath]:
        data = self._read()
        return [ConfiguredPath(path=str(item["path"])) for item in data.get("paths", [])]

    def add_path(self, path: str | Path) -> None:
        normalized = str(Path(path))
        if not normalized.strip():
            return
        data = self._read()
        paths = data.setdefault("paths", [])
        if all(str(item.get("path")) != normalized for item in paths):
            paths.append({"path": normalized})
            paths.sort(key=lambda item: str(item["path"]).lower())
            self._write(data)

    def remove_path(self, path: str | Path) -> None:
        normalized = str(Path(path))
        data = self._read()
        original_paths = data.get("paths", [])
        data["paths"] = [item for item in original_paths if str(item.get("path")) != normalized]
        if data["paths"] != original_paths:
            self._write(data)

    def get_summary(self, source: str | Path) -> CachedSummary | None:
        source_text = str(source)
        summaries = self._read().get("summaries", {})
        item = summaries.get(source_text)
        if item is None:
            for candidate in summaries.values():
                if source_text in {str(candidate.get("file_name")), str(candidate.get("source_path"))}:
                    item = candidate
                    break
        if item is None:
            return None
        return CachedSummary(
            source_path=str(item["source_path"]),
            file_name=str(item["file_name"]),
            summary=str(item["summary"]),
            model=str(item["model"]),
            source_count=int(item["source_count"]),
            partial_summary_count=int(item["partial_summary_count"]),
        )

    def save_summary(self, summary: CachedSummary) -> None:
        data = self._read()
        summaries = data.setdefault("summaries", {})
        summaries[summary.source_path] = asdict(summary)
        self._write(data)

    def remove_summary(self, source: str | Path) -> None:
        source_text = str(source)
        data = self._read()
        summaries = data.setdefault("summaries", {})
        keys_to_remove = [
            key
            for key, item in summaries.items()
            if source_text in {key, str(item.get("source_path")), str(item.get("file_name"))}
        ]
        for key in keys_to_remove:
            summaries.pop(key, None)
        if keys_to_remove:
            self._write(data)

    def clear_summaries(self) -> None:
        data = self._read()
        if data.get("summaries"):
            data["summaries"] = {}
            self._write(data)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"paths": [], "summaries": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"paths": [], "summaries": {}}
        if not isinstance(data, dict):
            return {"paths": [], "summaries": {}}
        data.setdefault("paths", [])
        data.setdefault("summaries", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
