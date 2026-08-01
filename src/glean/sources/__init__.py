"""Source registry + URL detection. Imports are lazy and tolerant so a single broken
source module never takes down the whole CLI.
"""

from __future__ import annotations

_SOURCES = ("youtube", "instagram", "tiktok", "twitch", "x")


def get_source(name: str):
    """Construct a Source by name."""
    if name == "youtube":
        from glean.sources.youtube import YouTubeSource
        return YouTubeSource()
    if name == "instagram":
        from glean.sources.instagram import InstagramSource
        return InstagramSource()
    if name == "tiktok":
        from glean.sources.tiktok import TikTokSource
        return TikTokSource()
    if name == "twitch":
        from glean.sources.twitch import TwitchSource
        return TwitchSource()
    if name == "x":
        from glean.sources.x import XSource
        return XSource()
    raise ValueError(f"unknown source {name!r}")


def detect_source(url: str) -> str:
    """Return the name of the first source whose matches() accepts this URL."""
    for name in _SOURCES:
        try:
            if get_source(name).matches(url):
                return name
        except Exception:
            continue
    raise ValueError(f"no source recognises URL: {url}")
