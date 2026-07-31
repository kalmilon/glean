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
import os
import re
import time
from datetime import datetime, timezone

import requests

from glean.models import MediaItem, Profile

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


def _cookie(cfg) -> str | None:
    """The Instagram sessionid, if the caller supplied one (cfg or $GLEAN_IG_SESSIONID)."""
    value = getattr(cfg, "ig_cookie", None) if cfg is not None else None
    return value or os.environ.get("GLEAN_IG_SESSIONID") or None


def _headers(cfg=None) -> dict:
    headers = {
        "x-ig-app-id": INSTAGRAM_APP_ID,
        "user-agent": INSTAGRAM_USER_AGENT,
        "accept": "*/*",
    }
    sessionid = _cookie(cfg)
    if sessionid:
        headers["cookie"] = f"sessionid={sessionid}"
    return headers


def _get_json(url: str, cfg=None) -> dict:
    """GET a public Instagram JSON endpoint with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(cfg), timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
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
        data = _get_json(info_url, cfg)
        items = data.get("items") or []
        if not items:
            raise RuntimeError(f"Instagram returned no media for {url!r}")
        item = _item_from_media(items[0])
        # Preserve the caller's exact URL and the canonical shortcode as the id.
        item.url = url
        item.id = code
        return item


def _fetch_reels(username: str, limit: int, cfg=None) -> list[dict]:
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
        data = _get_json(feed_url, cfg)
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
    return [_item_from_media(item, name) for item in _fetch_reels(name, limit, cfg)]


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
        reels = _fetch_reels(name, limit, cfg)
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


# --- Profiles: lookup (cookieless) and search (needs a session cookie) ---

def _profile_from_web_info(user: dict) -> Profile:
    """Map the web_profile_info user blob to a Profile."""
    handle = user.get("username", "")
    return Profile(
        source="instagram",
        username=handle,
        id=str(user.get("id")) if user.get("id") is not None else None,
        full_name=user.get("full_name") or None,
        followers=(user.get("edge_followed_by") or {}).get("count"),
        following=(user.get("edge_follow") or {}).get("count"),
        posts=(user.get("edge_owner_to_timeline_media") or {}).get("count"),
        verified=bool(user.get("is_verified")),
        private=bool(user.get("is_private")),
        bio=user.get("biography") or None,
        external_url=user.get("external_url") or None,
        url=f"https://www.instagram.com/{handle}/" if handle else "",
        extra={"category": user.get("category_name") or user.get("category")},
    )


def _profile_from_search_user(user: dict) -> Profile:
    """Map a topsearch/users-search result user to a Profile (fewer fields available)."""
    handle = user.get("username", "")
    return Profile(
        source="instagram",
        username=handle,
        id=str(user.get("pk") or user.get("id") or "") or None,
        full_name=user.get("full_name") or None,
        followers=user.get("follower_count"),
        verified=bool(user.get("is_verified")),
        private=bool(user.get("is_private")),
        url=f"https://www.instagram.com/{handle}/" if handle else "",
        extra={k: user.get(k) for k in ("mutual_followers_count",) if user.get(k) is not None},
    )


def profile(username: str, cfg=None) -> Profile:
    """Fetch one account's public profile by exact handle. Works without a cookie."""
    name = _normalize_username(username)
    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={name}"
    data = _get_json(url, cfg)
    user = (data.get("data") or {}).get("user")
    if not user:
        raise RuntimeError(f"Instagram returned no profile for @{name}")
    return _profile_from_web_info(user)


def search(query: str, cfg=None, limit: int = 20) -> list[Profile]:
    """Search Instagram for accounts matching a free-text query.

    Instagram blocks anonymous search, so this needs a logged-in `sessionid`
    (set $GLEAN_IG_SESSIONID or `ig_sessionid` in config). `profile()` is not a
    cookie-free alternative — it reads the same gated endpoint — but `list()`
    is, so a known handle's reels can still be reached without one.
    """
    if not _cookie(cfg):
        raise RuntimeError(
            "Instagram profile search requires a session cookie — anonymous search "
            "is blocked by Instagram. Copy the `sessionid` cookie from your logged-in "
            "instagram.com and set GLEAN_IG_SESSIONID (or ig_sessionid in config). "
            "`glean ig profile` needs the same cookie; to reach a known handle's reels "
            "without one, use `glean ig list <@handle>`."
        )
    q = requests.utils.quote(query.strip())
    url = f"https://www.instagram.com/web/search/topsearch/?context=blended&query={q}&count={limit}"
    data = _get_json(url, cfg)
    users = [entry.get("user") or {} for entry in (data.get("users") or [])]
    profiles = [_profile_from_search_user(u) for u in users if u.get("username")]
    return profiles[:limit]
