"""Source contract — a platform glean knows how to fetch from.

A Source recognises its own URLs and can describe a single item's metadata.
Discovery verbs (channel/search/list/scout) are module-level functions, not part of
this Protocol, because they differ per platform.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from glean.models import MediaItem


@runtime_checkable
class Source(Protocol):
    name: str

    def matches(self, url: str) -> bool:
        """True if this URL belongs to this platform."""
        ...

    def fetch_metadata(self, url: str, cfg) -> MediaItem:
        """Return metadata for a single item (no transcription, no heavy download)."""
        ...
