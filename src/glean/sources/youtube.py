"""YouTube source — recognise YouTube URLs/IDs and enumerate channels & searches.

Metadata comes from the `yt_dlp` python module in extract-only mode (no download).
Transcription happens elsewhere in the pipeline; this module never touches audio.
Pure-Python: the only heavy lifter is yt-dlp itself.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from glean.models import MediaItem, Transcript, Word

_VIDEO_ID = r"[A-Za-z0-9_-]{11}"
_URL_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/)(" + _VIDEO_ID + r")"),
    re.compile(r"^(" + _VIDEO_ID + r")$"),
]
_YOUTUBE_HOST = re.compile(r"(?://|\.|^)(?:www\.|m\.)?(youtube\.com|youtu\.be)\b", re.IGNORECASE)
_YOUTUBE_HOSTS = ("youtube.com", "youtu.be")


def _is_youtube_host(url: str) -> bool:
    """True only when the URL's actual host is youtube.com/youtu.be or a subdomain of them."""
    host = (urlparse(url if "://" in url else "//" + url).hostname or "").lower()
    return host in _YOUTUBE_HOSTS or any(host.endswith("." + h) for h in _YOUTUBE_HOSTS)


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
        if _is_youtube_host(candidate):
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


def _pick_caption_track(tracks: dict, want: str) -> tuple[str | None, list | None]:
    """Choose a caption track from a {lang: [formats]} map, preferring `want`'s language."""
    if not tracks:
        return None, None
    base = want.split("-")[0].lower()
    if want in tracks:
        return want, tracks[want]
    for lang, fmts in tracks.items():
        if lang.split("-")[0].lower() == base:
            return lang, fmts
    lang, fmts = next(iter(tracks.items()))
    return lang, fmts


def _json3_transcript(events: list, lang: str | None, model: str) -> Transcript:
    """Parse a YouTube json3 caption payload into a Transcript with word-level timings."""
    words: list[Word] = []
    lines: list[str] = []
    for event in events:
        segs = event.get("segs")
        if not segs:
            continue
        ev_start = event.get("tStartMs") or 0
        ev_end = ev_start + (event.get("dDurationMs") or 0)
        line_parts: list[str] = []
        for i, seg in enumerate(segs):
            raw = seg.get("utf8", "")
            line_parts.append(raw)
            token = raw.strip()
            if not token:
                continue
            start = (ev_start + (seg.get("tOffsetMs") or 0)) / 1000.0
            nxt = next((s["tOffsetMs"] for s in segs[i + 1:] if "tOffsetMs" in s), None)
            end = (ev_start + nxt) / 1000.0 if nxt is not None else ev_end / 1000.0
            words.append(Word(text=token, start=start, end=max(start, end)))
        line = "".join(line_parts).strip()
        if line:
            lines.append(line)
    duration = words[-1].end if words else None
    return Transcript(text=" ".join(lines), words=words, language=lang, backend="youtube-captions", model=model, duration=duration)


def captions(url: str, cfg, language: str | None = None) -> Transcript:
    """Fetch YouTube's own captions (manual preferred, else auto) as a Transcript.

    Reads the json3 caption track via yt-dlp metadata + requests. No audio download.
    Raises RuntimeError when the video exposes no usable caption track.
    """
    import requests
    import yt_dlp

    video_id = _extract_video_id(url)
    target = _watch_url(video_id) if video_id else url
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=False)

    want = (language or "en").strip() or "en"
    lang, fmts = _pick_caption_track(info.get("subtitles") or {}, want)
    model = "manual"
    if fmts is None:
        lang, fmts = _pick_caption_track(info.get("automatic_captions") or {}, want)
        model = "auto"
    json3 = next((f for f in (fmts or []) if f.get("ext") == "json3" and f.get("url")), None)
    if json3 is None:
        raise RuntimeError(f"no captions available for {url}")

    resp = requests.get(json3["url"], timeout=30)
    resp.raise_for_status()
    events = (resp.json() or {}).get("events") or []
    tx = _json3_transcript(events, lang, model)
    if not tx.text:
        raise RuntimeError(f"no captions available for {url}")
    return tx
