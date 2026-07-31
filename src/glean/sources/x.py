"""X (Twitter) source — tweet videos, natively supported by yt-dlp.

Recognises `x.com` / `twitter.com` status URLs and describes a single tweet's video
metadata via a yt-dlp extract-only pass. Downloading is handled by the shared yt-dlp
acquire path, not here.
"""

from __future__ import annotations

from urllib.parse import urlparse

from glean.config import Config
from glean.models import MediaItem

# Platform-metadata keys copied into MediaItem.extra when yt-dlp reports them.
_EXTRA_KEYS = ("view_count", "like_count", "comment_count", "repost_count", "favorite_count", "uploader_id", "uploader_url", "channel", "thumbnail", "ext", "extractor_key")


def _split(url: str) -> tuple[str, str]:
    """Return (lowercased host, path) tolerant of scheme-less URLs."""
    normalized = url if "://" in url else "//" + url.lstrip("/")
    parsed = urlparse(normalized)
    return (parsed.hostname or "").lower(), parsed.path or ""


def _is_x_host(host: str) -> bool:
    return host in ("x.com", "twitter.com") or host.endswith((".x.com", ".twitter.com"))


def _published(info: dict) -> str | None:
    """Best ISO 8601 timestamp yt-dlp offers: unix timestamp, else upload date."""
    from datetime import datetime, timezone
    ts = info.get("timestamp") or info.get("release_timestamp")
    if ts:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    day = str(info.get("upload_date") or "")
    return f"{day[:4]}-{day[4:6]}-{day[6:]}" if len(day) == 8 else None


def _id_from_path(path: str) -> str:
    """Fallback id when yt-dlp omits one: the numeric tweet id after /status/."""
    parts = [p for p in path.split("/") if p]
    if "status" in parts and parts.index("status") + 1 < len(parts):
        return parts[parts.index("status") + 1]
    return parts[-1] if parts else ""


def _to_media_item(info: dict, fallback_url: str) -> MediaItem:
    """Map a yt-dlp info dict onto a MediaItem(source='x')."""
    duration = info.get("duration")
    extra = {k: info[k] for k in _EXTRA_KEYS if info.get(k) is not None}
    return MediaItem(
        source="x",
        url=info.get("webpage_url") or fallback_url,
        id=info.get("id") or _id_from_path(_split(fallback_url)[1]),
        title=info.get("title") or info.get("description"),
        author=info.get("uploader") or info.get("uploader_id") or info.get("creator"),
        published=_published(info),
        duration=float(duration) if duration is not None else None,
        extra=extra,
    )


def _extract(url: str) -> dict:
    """Run yt-dlp in extract-only mode and return a JSON-safe info dict."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=False))


class XSource:
    name = "x"

    def matches(self, url: str) -> bool:
        host, path = _split(url)
        return _is_x_host(host) and "/status/" in path

    def fetch_metadata(self, url: str, cfg: Config) -> MediaItem:
        return _to_media_item(_extract(url), url)
