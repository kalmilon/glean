"""Instagram source — recognise Reel/post URLs and enumerate an account's Reels.

Metadata comes from Instagram's public JSON, reached with the web App-ID header and
no cookies (the same anonymous path the `ig` bash tool uses). Media *downloading*
is not done here: the pipeline's `acquire.download_audio` handles the Cobalt route.
This module owns metadata and discovery only.

Pure-Python: `requests` + stdlib. The heavy lifters live in the pipeline.

  * `InstagramSource` — matches reel/p/tv URLs, fetches one item's metadata.
  * `list_account(username, cfg, limit)` — an account's most recent Reels.
  * `scout(usernames, cfg, top, since_days)` — Reels ranked by visible-evidence score.

The scout score ranks *candidates* from public counters; it does not prove why a
Reel performed. Formula (ported verbatim from the `ig` tool):

    ln(plays + 1) + 2·ln(outlier_ratio + 1) + 10·engagement_rate − age_days / 30
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone

import requests

from glean.models import MediaItem

INSTAGRAM_APP_ID = "936619743392459"
INSTAGRAM_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/126 Safari/537.36"
)

# Instagram encodes a media's numeric primary key as a URL-safe base64 shortcode.
_SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

_URL_SHORTCODE = re.compile(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", re.IGNORECASE)
_HOST = re.compile(r"(?:^|[./])instagram\.com/", re.IGNORECASE)
_USERNAME_OK = re.compile(r"^[A-Za-z0-9._]+$")

_RETRIES = 4
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30


def _headers() -> dict:
    return {
        "x-ig-app-id": INSTAGRAM_APP_ID,
        "user-agent": INSTAGRAM_USER_AGENT,
        "accept": "*/*",
    }


def _get_json(url: str) -> dict:
    """GET a public Instagram JSON endpoint with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < _RETRIES:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Instagram request failed after {_RETRIES} attempts: {url}") from last_error


def _normalize_username(value: str) -> str:
    """Strip a leading @ or profile URL down to a bare username, validated."""
    name = value.strip()
    name = name.lstrip("@")
    name = re.sub(r"^https?://(www\.)?instagram\.com/", "", name, flags=re.IGNORECASE)
    name = name.split("/", 1)[0].split("?", 1)[0]
    if not _USERNAME_OK.match(name):
        raise ValueError(f"invalid Instagram username: {value!r}")
    return name


def _shortcode(url: str) -> str | None:
    m = _URL_SHORTCODE.search(url)
    return m.group(1) if m else None


def _media_id(shortcode: str) -> int:
    """Decode a Reel/post shortcode into its numeric media primary key."""
    pk = 0
    for ch in shortcode:
        pk = pk * 64 + _SHORTCODE_ALPHABET.index(ch)
    return pk


def _reel_url(code: str) -> str:
    return f"https://www.instagram.com/reel/{code}/"


def _iso_published(taken_at) -> str | None:
    """Convert a unix `taken_at` timestamp into an ISO 8601 UTC string."""
    if not taken_at:
        return None
    try:
        return datetime.fromtimestamp(int(taken_at), tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _item_from_media(item: dict, username: str | None = None) -> MediaItem:
    """Map a public Instagram feed/media item dict to a MediaItem."""
    code = item.get("code") or ""
    author = username or (item.get("user") or {}).get("username")
    caption = (item.get("caption") or {}).get("text") if isinstance(item.get("caption"), dict) else None
    plays = item.get("play_count")
    view_count = item.get("view_count")
    extra = {
        "plays": plays if plays is not None else view_count,
        "view_count": view_count,
        "likes": item.get("like_count"),
        "comments": item.get("comment_count"),
        "product_type": item.get("product_type"),
        "width": item.get("original_width"),
        "height": item.get("original_height"),
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    return MediaItem(
        source="instagram",
        url=_reel_url(code) if code else _reel_url(item.get("pk", "")),
        id=code or str(item.get("pk", "")),
        title=caption or None,
        author=author,
        published=_iso_published(item.get("taken_at")),
        duration=item.get("video_duration"),
        extra=extra,
    )


class InstagramSource:
    """Recognises Instagram Reel/post/IGTV URLs and describes a single item."""

    name = "instagram"

    def matches(self, url: str) -> bool:
        return bool(_HOST.search(url) and _URL_SHORTCODE.search(url))

    def fetch_metadata(self, url: str, cfg) -> MediaItem:
        code = _shortcode(url)
        if not code:
            raise ValueError(f"not an Instagram Reel/post URL: {url!r}")
        info_url = f"https://www.instagram.com/api/v1/media/{_media_id(code)}/info/"
        data = _get_json(info_url)
        items = data.get("items") or []
        if not items:
            raise RuntimeError(f"Instagram returned no media for {url!r}")
        item = _item_from_media(items[0])
        # Preserve the caller's exact URL and the canonical shortcode as the id.
        item.url = url
        item.id = code
        return item


def _fetch_reels(username: str, limit: int) -> list[dict]:
    """Page the public user feed, keeping only Reels (product_type == 'clips')."""
    reels: dict[str, dict] = {}
    max_id: str | None = None
    page = 0
    # Fetch beyond the requested count: Instagram pins up to three (possibly old)
    # posts at the front of the first page.
    while len(reels) < limit + 6 and page < 20:
        page += 1
        feed_url = f"https://www.instagram.com/api/v1/feed/user/{username}/username/?count=12"
        if max_id:
            feed_url += f"&max_id={max_id}"
        data = _get_json(feed_url)
        for item in data.get("items") or []:
            if item.get("product_type") != "clips":
                continue
            code = item.get("code")
            if code:
                reels[code] = item
        max_id = data.get("next_max_id") or None
        if not max_id:
            break
        # Keep anonymous profile discovery deliberately low-rate.
        time.sleep(2)
    ordered = sorted(reels.values(), key=lambda it: it.get("taken_at") or 0, reverse=True)
    return ordered[:limit]


def list_account(username: str, cfg, limit: int = 24) -> list[MediaItem]:
    """Return an account's most recent public Reels as MediaItems, newest first."""
    name = _normalize_username(username)
    return [_item_from_media(item, name) for item in _fetch_reels(name, limit)]


def _median(values: list[float]) -> float | None:
    positives = sorted(v for v in values if v is not None and v > 0)
    n = len(positives)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return positives[mid]
    return (positives[mid - 1] + positives[mid]) / 2


def _score_reel(item: dict, account_median: float | None, now: int, cutoff: int) -> dict | None:
    """Compute the scout evidence dict for one reel, or None if it drops out.

    Drops a reel that is older than the cutoff or has no positive play count —
    those cannot be ranked on visible evidence.
    """
    plays = item.get("play_count") or item.get("view_count")
    taken_at = item.get("taken_at") or 0
    if taken_at < cutoff or not plays or plays <= 0:
        return None

    likes = item.get("like_count") or 0
    comments = item.get("comment_count") or 0
    age_days = round((now - taken_at) / 86400 * 10) / 10
    outlier_ratio = round(plays / account_median * 100) / 100 if account_median and account_median > 0 else None
    engagement_rate = round((likes + comments) / plays * 10000) / 10000

    scout_score = (
        math.log(plays + 1)
        + 2 * math.log((outlier_ratio or 0) + 1)
        + 10 * engagement_rate
        - age_days / 30
    )
    return {
        "plays": plays,
        "account_median_plays": account_median,
        "age_days": age_days,
        "outlier_ratio": outlier_ratio,
        "engagement_rate": engagement_rate,
        "scout_score": round(scout_score, 6),
    }


def scout(usernames: list[str], cfg, top: int = 30, since_days: int = 90) -> list[MediaItem]:
    """Rank recent Reels across accounts by visible-evidence score, best first.

    For each account, fetch its recent Reels, compute the fetched-sample play
    median, then score every reel by plays, performance vs that median,
    likes/comments, and age. Returns the top-N MediaItems, each carrying the
    scoring evidence in `extra`.
    """
    now = int(time.time())
    cutoff = now - since_days * 86400
    limit = 24  # per-account fetch depth; mirrors the ig tool's default

    scored: list[tuple[float, MediaItem]] = []
    for raw in usernames:
        name = _normalize_username(raw)
        reels = _fetch_reels(name, limit)
        account_median = _median([r.get("play_count") or r.get("view_count") for r in reels])
        for reel in reels:
            evidence = _score_reel(reel, account_median, now, cutoff)
            if evidence is None:
                continue
            media = _item_from_media(reel, name)
            media.extra.update(evidence)
            scored.append((evidence["scout_score"], media))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [media for _, media in scored[:top]]
