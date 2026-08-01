"""TikTok source — videos, account listings, and the sounds an account builds on.

yt-dlp carries the audio on every entry as `track` + `artist`, and a single flat extraction of an
account returns those alongside plays, likes, reposts and timestamps. That is the whole reason
TikTok is worth having here: one call per account answers both "what did well" and "what was it
built on", where YouTube exposes no sound attribution at all.

What is missing is TikTok's numeric music id, which lives in the video page's embedded JSON rather
than in yt-dlp's output. Sounds are therefore identified by their name and artist. Two genuinely
different tracks sharing both would merge — rare, and the alternative is a page fetch per video.
The same gap is why a Sound from here has no `url`: the platform's music page is keyed by that id.

Downloading is not done here. TikTok goes through the shared yt-dlp acquire path.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from glean.config import Config, ytdlp_cookie_opts
from glean.models import MediaItem, Sound

# TikTok's own name for a creator's own audio. Matching Instagram's vocabulary here is deliberate:
# --music-only then means the same thing on both platforms.
ORIGINAL_SOUND_TITLE = "original sound"

_EXTRA_KEYS = (
    "view_count", "like_count", "comment_count", "repost_count", "save_count",
    "channel", "channel_id", "uploader_id", "uploader_url", "thumbnail", "ext",
    "track", "artist",
)


def _split(url: str) -> tuple[str, str]:
    """Return (lowercased host, path) tolerant of scheme-less URLs."""
    normalized = url if "://" in url else "//" + url.lstrip("/")
    parsed = urlparse(normalized)
    return (parsed.hostname or "").lower(), parsed.path or ""


def _published(info: dict) -> str | None:
    """Best ISO 8601 timestamp yt-dlp offers: unix timestamp, else upload date."""
    ts = info.get("timestamp")
    if ts:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    day = str(info.get("upload_date") or "")
    return f"{day[:4]}-{day[4:6]}-{day[6:]}" if len(day) == 8 else None


def _normalize_username(value: str) -> str:
    """Strip a leading @ or profile URL down to a bare username."""
    name = value.strip().lstrip("@")
    if "tiktok.com" in name.lower():
        name = _split(name)[1].lstrip("/").split("/", 1)[0].lstrip("@")
    return name.split("?", 1)[0]


def _to_media_item(info: dict, fallback_url: str) -> MediaItem:
    """Map a yt-dlp info dict (full or flat entry) onto a MediaItem(source='tiktok')."""
    duration = info.get("duration")
    extra = {k: info[k] for k in _EXTRA_KEYS if info.get(k) is not None}
    return MediaItem(
        source="tiktok",
        url=info.get("webpage_url") or info.get("url") or fallback_url,
        id=str(info.get("id") or ""),
        # TikTok has no titles — the caption is what yt-dlp puts in `title`.
        title=info.get("title") or info.get("description"),
        author=info.get("uploader") or info.get("channel") or info.get("uploader_id"),
        published=_published(info),
        duration=float(duration) if duration is not None else None,
        extra=extra,
    )


def _extract(url: str, *, flat: bool = False, limit: int = 0, cfg=None) -> dict:
    """Run yt-dlp in extract-only mode and return a JSON-safe info dict."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, **ytdlp_cookie_opts(cfg)}
    if flat:
        opts["extract_flat"] = True
    if limit:
        opts["playlistend"] = limit
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=False))


class TikTokSource:
    """Recognises TikTok video URLs and describes a single item."""

    name = "tiktok"

    def matches(self, url: str) -> bool:
        host, _ = _split(url)
        return host == "tiktok.com" or host.endswith(".tiktok.com")

    def fetch_metadata(self, url: str, cfg: Config) -> MediaItem:
        return _to_media_item(_extract(url, cfg=cfg), url)


def list_account(username: str, cfg, limit: int = 30) -> list[MediaItem]:
    """An account's recent videos, newest first."""
    name = _normalize_username(username)
    info = _extract(f"https://www.tiktok.com/@{name}", flat=True, limit=limit, cfg=cfg)
    items: list[MediaItem] = []
    for entry in info.get("entries") or []:
        if not (entry.get("id") or entry.get("url")):
            continue
        items.append(_to_media_item(entry, entry.get("url") or ""))
        if limit and len(items) >= limit:
            break
    return items


def _sound_of(entry: dict) -> tuple[str, str, str | None, str | None] | None:
    """Identify the audio behind one video as (id, kind, title, artist), or None if unnamed.

    The id is a name pair, not a platform id — TikTok does not expose its numeric music id here —
    lowercased so the same track credited with different capitalisation still groups.

    Original audio needs the creator in that key. Every creator's own audio is called "original
    sound", sometimes with their name appended and sometimes with the artist field left empty, so
    keying on the title alone merged every original sound on the platform into one row: 39 videos
    across two unrelated accounts reported as a single shared sound. The owner is whoever the
    artist field names, falling back to the uploader when it is blank.
    """
    track = (entry.get("track") or "").strip()
    if not track:
        return None
    artist = (entry.get("artist") or "").strip() or None

    # "original sound", "original sound - name" and "originalsound" are all the same thing to
    # TikTok, so match the prefix rather than the exact string.
    if track.lower().startswith(ORIGINAL_SOUND_TITLE):
        owner = artist or entry.get("uploader") or entry.get("channel") or entry.get("uploader_id")
        return f"original::{(owner or '').lower()}", "original_sounds", track, owner

    return f"{track.lower()}::{(artist or '').lower()}", "licensed_music", track, artist


def _median(values: list[float]) -> float | None:
    positives = sorted(v for v in values if v is not None and v > 0)
    n = len(positives)
    if n == 0:
        return None
    mid = n // 2
    return positives[mid] if n % 2 else (positives[mid - 1] + positives[mid]) / 2


def sounds(usernames: list[str], cfg, top: int = 30, limit: int = 30, since_days: int = 90, music_only: bool = False) -> list[Sound]:
    """Rank the audio a set of accounts is building on, most-used first.

    A survey of the accounts you name, not a global chart — TikTok's trending-sounds endpoint
    (Creative Center) refuses anonymous callers, so there is nothing public to read instead.

    Ranked by how many sampled videos use a sound, then by total plays, so a track several
    accounts reached for outranks one video that went far on its own audio.
    """
    cutoff = int(time.time()) - since_days * 86400
    found: dict[str, dict] = {}

    for raw in usernames:
        name = _normalize_username(raw)
        info = _extract(f"https://www.tiktok.com/@{name}", flat=True, limit=limit, cfg=cfg)
        for entry in info.get("entries") or []:
            if (entry.get("timestamp") or 0) < cutoff:
                continue
            ident = _sound_of(entry)
            if ident is None:
                continue
            sid, kind, title, artist = ident
            # An original sound belongs to whoever posted it; it is not the interchangeable thing
            # "what sound should I use" is asking about.
            if music_only and kind != "licensed_music":
                continue
            plays = entry.get("view_count") or 0
            bucket = found.setdefault(sid, {
                "kind": kind, "title": title, "artist": artist,
                "plays": [], "accounts": [], "examples": [],
            })
            bucket["plays"].append(plays)
            if name not in bucket["accounts"]:
                bucket["accounts"].append(name)
            url = entry.get("url") or entry.get("webpage_url")
            if url and len(bucket["examples"]) < 3:
                bucket["examples"].append(url)

    out = [
        Sound(
            source="tiktok",
            id=sid,
            kind=b["kind"],
            title=b["title"],
            artist=b["artist"],
            uses=len(b["plays"]),
            total_plays=sum(b["plays"]),
            median_plays=_median([float(p) for p in b["plays"]]),
            accounts=b["accounts"],
            examples=b["examples"],
            # The music page is keyed by a numeric id yt-dlp does not report, so there is no link
            # to give. Better empty than a guessed URL.
            url="",
        )
        for sid, b in found.items()
    ]
    out.sort(key=lambda s: (s.uses, s.total_plays), reverse=True)
    return out[:top]
