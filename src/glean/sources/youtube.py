"""YouTube source — recognise YouTube URLs/IDs and enumerate channels & searches.

Metadata comes from the `yt_dlp` python module in extract-only mode (no download).
Transcription happens elsewhere in the pipeline; this module never touches audio.
Pure-Python: the only heavy lifter is yt-dlp itself.
"""

from __future__ import annotations

import re

from glean.models import MediaItem

_VIDEO_ID = r"[A-Za-z0-9_-]{11}"
_URL_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/)(" + _VIDEO_ID + r")"),
    re.compile(r"^(" + _VIDEO_ID + r")$"),
]
_YOUTUBE_HOST = re.compile(r"(?://|\.|^)(?:www\.|m\.)?(youtube\.com|youtu\.be)\b", re.IGNORECASE)


def _extract_video_id(url_or_id: str) -> str | None:
    """Pull an 11-char video ID out of a URL, or return a bare ID unchanged."""
    candidate = url_or_id.strip()
    for pattern in _URL_ID_PATTERNS:
        m = pattern.search(candidate)
        if m:
            return m.group(1)
    return None


def _iso_published(upload_date: str | None) -> str | None:
    """Convert yt-dlp's YYYYMMDD upload_date into an ISO 8601 date string."""
    if not upload_date or not re.fullmatch(r"\d{8}", str(upload_date)):
        return None
    s = str(upload_date)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _entry_to_item(entry: dict) -> MediaItem | None:
    """Map a yt-dlp info/entry dict to a MediaItem, or None if it has no usable ID."""
    video_id = entry.get("id") or ""
    if not video_id:
        return None
    duration = entry.get("duration")
    extra = {
        "view_count": entry.get("view_count"),
        "like_count": entry.get("like_count"),
        "channel": entry.get("channel") or entry.get("uploader"),
        "channel_id": entry.get("channel_id"),
        "webpage_url": entry.get("webpage_url"),
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    return MediaItem(
        source="youtube",
        url=entry.get("webpage_url") or _watch_url(video_id),
        id=video_id,
        title=entry.get("title"),
        author=entry.get("uploader") or entry.get("channel"),
        published=_iso_published(entry.get("upload_date")),
        duration=float(duration) if duration is not None else None,
        extra=extra,
    )


class YouTubeSource:
    """Source implementation for YouTube videos, shorts, and live URLs."""

    name = "youtube"

    def matches(self, url: str) -> bool:
        """True for youtube.com / youtu.be URLs and bare 11-char video IDs."""
        candidate = url.strip()
        if _YOUTUBE_HOST.search(candidate):
            return True
        if re.fullmatch(_VIDEO_ID, candidate):
            return True
        return False

    def fetch_metadata(self, url: str, cfg) -> MediaItem:
        """Extract a single video's metadata via yt-dlp (download=False)."""
        import yt_dlp

        video_id = _extract_video_id(url)
        target = _watch_url(video_id) if video_id else url
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(target, download=False)
        item = _entry_to_item(info)
        if item is None:
            raise ValueError(f"could not extract YouTube metadata for {url!r}")
        return item


def _flat_entries(target: str, limit: int) -> list[dict]:
    """Run yt-dlp flat extraction against a target and return its entries."""
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    if limit:
        ydl_opts["playlistend"] = limit
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=False)
    return list(info.get("entries") or [])


def _collect(entries: list[dict], limit: int) -> list[MediaItem]:
    """Dedupe entries by video ID and map them to MediaItems, capped at limit."""
    items: list[MediaItem] = []
    seen: set[str] = set()
    for entry in entries:
        item = _entry_to_item(entry)
        if item is None or item.id in seen:
            continue
        seen.add(item.id)
        items.append(item)
        if limit and len(items) >= limit:
            break
    return items


def channel(handle: str, cfg, limit: int = 30) -> list[MediaItem]:
    """List a channel's uploads (newest first) via yt-dlp's flat playlist extraction.

    Accepts an @handle, a bare handle, or a full channel URL.
    """
    handle = handle.strip()
    if _YOUTUBE_HOST.search(handle):
        target = handle if "/videos" in handle else handle.rstrip("/") + "/videos"
    else:
        normalised = handle if handle.startswith("@") else "@" + handle.lstrip("@")
        target = f"https://www.youtube.com/{normalised}/videos"
    return _collect(_flat_entries(target, limit), limit)


def search(query: str, cfg, limit: int = 30) -> list[MediaItem]:
    """Search YouTube via yt-dlp's `ytsearchN:` provider and return metadata items."""
    n = limit or 30
    return _collect(_flat_entries(f"ytsearch{n}:{query}", n), n)
