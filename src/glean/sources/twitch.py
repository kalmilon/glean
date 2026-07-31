"""Twitch source — VODs and clips, both natively supported by yt-dlp.

Recognises `twitch.tv` URLs (videos/VODs and clips) and describes a single item's
metadata via a yt-dlp extract-only pass. Downloading is handled by the shared yt-dlp
acquire path, not here.
"""

from __future__ import annotations

from urllib.parse import urlparse

from glean.config import Config
from glean.models import MediaItem

# Platform-metadata keys copied into MediaItem.extra when yt-dlp reports them.
_EXTRA_KEYS = ("view_count", "like_count", "comment_count", "repost_count", "channel", "channel_id", "uploader_id", "uploader_url", "thumbnail", "ext", "live_status", "was_live", "extractor_key")


def _split(url: str) -> tuple[str, str]:
    """Return (lowercased host, path) tolerant of scheme-less URLs."""
    normalized = url if "://" in url else "//" + url.lstrip("/")
    parsed = urlparse(normalized)
    return (parsed.hostname or "").lower(), parsed.path or ""


def _published(info: dict) -> str | None:
    """Best ISO 8601 timestamp yt-dlp offers: unix timestamp, else upload date."""
    from datetime import datetime, timezone
    ts = info.get("timestamp") or info.get("release_timestamp")
    if ts:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    day = str(info.get("upload_date") or "")
    return f"{day[:4]}-{day[4:6]}-{day[6:]}" if len(day) == 8 else None


def _id_from_path(path: str) -> str:
    """Fallback id when yt-dlp omits one: the VOD/clip slug in the URL path."""
    parts = [p for p in path.split("/") if p]
    for anchor in ("videos", "clip"):
        if anchor in parts and parts.index(anchor) + 1 < len(parts):
            return parts[parts.index(anchor) + 1]
    return parts[-1] if parts else ""


def _to_media_item(info: dict, fallback_url: str) -> MediaItem:
    """Map a yt-dlp info dict (full or flat entry) onto a MediaItem(source='twitch')."""
    duration = info.get("duration")
    extra = {k: info[k] for k in _EXTRA_KEYS if info.get(k) is not None}
    return MediaItem(
        source="twitch",
        url=info.get("webpage_url") or fallback_url,
        id=info.get("id") or _id_from_path(_split(fallback_url)[1]),
        title=info.get("title"),
        author=info.get("uploader") or info.get("channel") or info.get("creator") or info.get("uploader_id"),
        published=_published(info),
        duration=float(duration) if duration is not None else None,
        extra=extra,
    )


def _extract(url: str, *, flat: bool = False) -> dict:
    """Run yt-dlp in extract-only mode and return a JSON-safe info dict."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    if flat:
        opts["extract_flat"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=False))


class TwitchSource:
    name = "twitch"

    def matches(self, url: str) -> bool:
        host, _ = _split(url)
        return host == "twitch.tv" or host.endswith(".twitch.tv")

    def fetch_metadata(self, url: str, cfg: Config) -> MediaItem:
        return _to_media_item(_extract(url), url)


def get_channel(channel: str, cfg: Config, limit: int = 30) -> list[MediaItem]:
    """List a channel's VODs (newest first) via yt-dlp flat extraction. Optional helper."""
    name = channel.rstrip("/").split("/")[-1].lstrip("@")
    info = _extract(f"https://www.twitch.tv/{name}/videos", flat=True)
    items: list[MediaItem] = []
    for entry in info.get("entries") or []:
        if not (entry.get("id") or entry.get("url")):
            continue
        items.append(_to_media_item(entry, entry.get("url") or entry.get("webpage_url") or ""))
        if limit and len(items) >= limit:
            break
    return items
