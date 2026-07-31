"""Shared data models — the currency passed between acquire, transcribe, and output.

Every record persisted to disk carries a `_v` version field so readers dispatch on
version, not field inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

RECORD_VERSION = 1


@dataclass
class Word:
    """A single token with its timing, in seconds."""
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    backend: str = ""              # "parakeet" | "whisper.cpp"
    model: str = ""                # "v2" | "ggml-large-v3-turbo" | ...
    duration: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_v"] = RECORD_VERSION
        return d


@dataclass
class MediaItem:
    source: str                    # "youtube" | "instagram" | "twitch" | "x"
    url: str
    id: str
    title: str | None = None
    author: str | None = None
    published: str | None = None   # ISO 8601
    duration: float | None = None
    extra: dict = field(default_factory=dict)  # platform metadata: plays, likes, view_count, ...

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_v"] = RECORD_VERSION
        return d


@dataclass
class Result:
    item: MediaItem
    transcript: Transcript | None
    out_dir: str
    files: dict = field(default_factory=dict)  # {"metadata":.., "txt":.., "json":.., "srt":..}

    def to_dict(self) -> dict:
        return {
            "_v": RECORD_VERSION,
            "item": self.item.to_dict(),
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "out_dir": self.out_dir,
            "files": self.files,
        }
