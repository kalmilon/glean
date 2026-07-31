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
class Profile:
    """A social account — the unit returned by profile lookup and account search."""
    source: str                    # "instagram" | ...
    username: str
    id: str | None = None
    full_name: str | None = None
    followers: int | None = None
    following: int | None = None
    posts: int | None = None
    verified: bool = False
    private: bool = False
    bio: str | None = None
    external_url: str | None = None
    url: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_v"] = RECORD_VERSION
        return d


@dataclass
class Sound:
    """An audio track reels are built on, aggregated over every reel seen using it.

    Instagram splits these in two and names them differently: licensed music carries a title and a
    display artist, while an original sound is titled after whoever posted it first. `kind` keeps
    that distinction rather than flattening it — a trending licensed track and a creator's own audio
    are not the same thing to go and reuse.

    `uses` and the play figures count only the reels this run actually saw, never a platform total.
    An account sampled 24 reels deep cannot say how many reels exist worldwide on a sound, so these
    numbers rank a set against itself and mean nothing outside it.
    """
    source: str                    # "instagram" | ...
    id: str                        # audio_cluster_id (licensed) or audio_asset_id (original)
    kind: str                      # "licensed_music" | "original_sounds"
    title: str | None = None
    artist: str | None = None
    uses: int = 0                  # reels seen using it
    total_plays: int = 0           # summed over those reels
    median_plays: float | None = None
    accounts: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)  # reel URLs
    url: str = ""                  # the platform's page for this sound
    extra: dict = field(default_factory=dict)

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
