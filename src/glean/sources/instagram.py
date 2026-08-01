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

from glean.models import MediaItem, Profile, Sound

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
# Rate limiting clears on a slower clock than a generic blip; 1-2-4s was not enough to ride out
# a burst of feed paging, which then surfaced as a permanent-looking failure on the next call.
_RATE_LIMIT_BACKOFF = 15


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


def _raise_if_terminal(resp) -> None:
    """Raise for the statuses no amount of retrying will change.

    Retrying these was actively misleading: a cookie-gated endpoint answered 401 four times over
    seven seconds and reported "request failed after 4 attempts", which reads as a flaky network
    and sent the reader looking for one. Naming the wall is the whole point.
    """
    if resp.status_code in (401, 403):
        raise _Unauthorized(
            f"Instagram refused this anonymously (HTTP {resp.status_code}). It needs a session "
            "cookie: copy `sessionid` from your logged-in instagram.com and set GLEAN_IG_SESSIONID "
            "(or ig_sessionid in config)."
        )


class _Unauthorized(RuntimeError):
    """Instagram requires a session cookie for this endpoint. Never retried."""


def _request_json(method: str, url: str, cfg=None, data: dict | None = None) -> dict:
    """Call an Instagram JSON endpoint, retrying only what retrying can fix.

    Rate limiting is the failure worth waiting out, and it is slower to clear than the old
    1-2-4-second ladder allowed — a burst of feed paging could leave a later profile lookup looking
    permanently broken when it was only throttled. 429 therefore backs off harder than a generic
    error, while 401/403 stop immediately.
    """
    headers = _headers(cfg)
    if method == "POST":
        headers = {**headers, "content-type": "application/x-www-form-urlencoded", "x-requested-with": "XMLHttpRequest"}

    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            if method == "POST":
                resp = requests.post(url, headers=headers, data=data or {}, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            else:
                resp = requests.get(url, headers=headers, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            _raise_if_terminal(resp)
            if resp.status_code == 429:
                last_error = RuntimeError("HTTP 429 rate limited")
                if attempt < _RETRIES:
                    time.sleep(_RATE_LIMIT_BACKOFF * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except _Unauthorized:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < _RETRIES:
                time.sleep(2 ** (attempt - 1))

    hint = " — Instagram rate limits anonymous callers, so this often clears on its own" if "429" in str(last_error) else ""
    raise RuntimeError(f"Instagram request failed after {_RETRIES} attempts{hint}: {url}") from last_error


def _get_json(url: str, cfg=None) -> dict:
    """GET a public Instagram JSON endpoint."""
    return _request_json("GET", url, cfg)


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


def _post_json(url: str, data: dict, cfg=None) -> dict:
    """POST a form-encoded Instagram endpoint."""
    return _request_json("POST", url, cfg, data=data)


def cluster_id_of(target: str) -> str:
    """Accept a bare audio cluster id or a /reels/audio/<id>/ URL and return the id."""
    text = target.strip()
    m = re.search(r"/reels?/audio/(\d+)", text)
    if m:
        return m.group(1)
    if text.isdigit():
        return text
    raise ValueError(f"not an audio cluster id or /reels/audio/ URL: {target!r}")


def reels_for_sound(audio_cluster_id: str, cfg, limit: int = 24) -> list[MediaItem]:
    """Every reel Instagram will show for one sound — real usage, not a sample of accounts.

    This is the number `sounds()` cannot give: it counts what a run happened to fetch, while this
    asks Instagram directly who is using a track. The endpoint is the one the /reels/audio/ page
    calls, and it is cookie-only — the page itself server-renders nothing, so there is no anonymous
    path to the same answer.
    """
    if not _cookie(cfg):
        raise RuntimeError(
            "Listing the reels on a sound requires a session cookie — Instagram serves this only to "
            "a logged-in caller. Copy the `sessionid` cookie from your logged-in instagram.com and "
            "set GLEAN_IG_SESSIONID (or ig_sessionid in config)."
        )

    items: list[MediaItem] = []
    seen: set[str] = set()
    max_id = ""
    while len(items) < limit:
        payload = {"audio_cluster_id": audio_cluster_id, "original_sound_audio_asset_id": ""}
        if max_id:
            payload["max_id"] = max_id
        data = _post_json("https://www.instagram.com/api/v1/clips/music/", payload, cfg)
        batch = data.get("items") or []
        if not batch:
            break
        for row in batch:
            # The endpoint wraps each reel in {"media": {...}}; older shapes hand back the media flat.
            media = row.get("media") if isinstance(row, dict) and "media" in row else row
            if not isinstance(media, dict):
                continue
            code = media.get("code")
            if not code or code in seen:
                continue
            seen.add(code)
            items.append(_item_from_media(media))
            if len(items) >= limit:
                break
        max_id = (data.get("paging_info") or {}).get("max_id") or ""
        if not max_id:
            break
        # Same deliberate low rate as the anonymous feed paging above.
        time.sleep(2)
    return items


# --- Sounds: which audio a set of accounts is actually building on ---


def _sound_of(item: dict) -> tuple[str, str, str | None, str | None] | None:
    """Identify the audio behind one reel as (id, kind, title, artist), or None if it has none.

    The two kinds live in different sub-objects and are keyed differently. Licensed music is keyed
    by audio_cluster_id, which is what Instagram's own /reels/audio/ page takes and what makes two
    accounts using the same track group together. An original sound has no cluster, so it is keyed
    by its asset id and is by nature used mostly by one account.
    """
    cm = item.get("clips_metadata") or {}
    kind = cm.get("audio_type") or ""

    asset = (cm.get("music_info") or {}).get("music_asset_info") or {}
    if asset:
        sid = asset.get("audio_cluster_id") or asset.get("id")
        if sid:
            return str(sid), kind or "licensed_music", asset.get("title"), asset.get("display_artist")

    original = cm.get("original_sound_info") or {}
    if original:
        sid = original.get("audio_asset_id")
        if sid:
            artist = (original.get("ig_artist") or {}).get("username")
            return str(sid), kind or "original_sounds", original.get("original_audio_title"), artist

    return None


def sounds(usernames: list[str], cfg, top: int = 30, limit: int = 24, since_days: int = 90, music_only: bool = False) -> list[Sound]:
    """Rank the audio a set of accounts is building on, most-used first.

    This is a survey of what the accounts you name are using, not a global chart — Instagram
    publishes no trending-sounds endpoint that answers without a login. Sampling accounts you
    already care about answers the more useful question anyway: what is working in this niche.

    Ranked by how many of the sampled reels use a sound, then by total plays, so a track three
    accounts reached for outranks one reel that happened to go far on its own audio.
    """
    now = int(time.time())
    cutoff = now - since_days * 86400
    found: dict[str, dict] = {}

    for raw in usernames:
        name = _normalize_username(raw)
        for reel in _fetch_reels(name, limit, cfg):
            if (reel.get("taken_at") or 0) < cutoff:
                continue
            ident = _sound_of(reel)
            if ident is None:
                continue
            sid, kind, title, artist = ident
            # An original sound belongs to whoever posted it; it can be remixed but it is not the
            # interchangeable thing "what sound should I use" is asking about. --music-only drops it.
            if music_only and kind != "licensed_music":
                continue
            plays = reel.get("play_count") or reel.get("view_count") or 0
            entry = found.setdefault(sid, {
                "kind": kind, "title": title, "artist": artist,
                "plays": [], "accounts": [], "examples": [],
            })
            entry["plays"].append(plays)
            if name not in entry["accounts"]:
                entry["accounts"].append(name)
            if reel.get("code") and len(entry["examples"]) < 3:
                entry["examples"].append(f"https://www.instagram.com/reel/{reel['code']}/")

    out = [
        Sound(
            source="instagram",
            id=sid,
            kind=e["kind"],
            title=e["title"],
            artist=e["artist"],
            uses=len(e["plays"]),
            total_plays=sum(e["plays"]),
            median_plays=_median([float(p) for p in e["plays"]]),
            accounts=e["accounts"],
            examples=e["examples"],
            # Only a licensed track has a browsable page; an original sound's asset id is not a
            # cluster id and this URL 404s for it, so it is left empty rather than made up.
            url=f"https://www.instagram.com/reels/audio/{sid}/" if e["kind"] == "licensed_music" else "",
        )
        for sid, e in found.items()
    ]
    out.sort(key=lambda s: (s.uses, s.total_plays), reverse=True)
    return out[:top]


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
    (set $GLEAN_IG_SESSIONID or `ig_sessionid` in config). Only *search* is gated:
    `profile()` and `list()` both answer anonymously, so a known handle is still
    reachable without one.
    """
    if not _cookie(cfg):
        raise RuntimeError(
            "Instagram profile search requires a session cookie — anonymous search "
            "is blocked by Instagram. Copy the `sessionid` cookie from your logged-in "
            "instagram.com and set GLEAN_IG_SESSIONID (or ig_sessionid in config). "
            "`glean ig profile` and `glean ig list` both work without one, so a handle you "
            "already know is still reachable."
        )
    q = requests.utils.quote(query.strip())
    url = f"https://www.instagram.com/web/search/topsearch/?context=blended&query={q}&count={limit}"
    data = _get_json(url, cfg)
    users = [entry.get("user") or {} for entry in (data.get("users") or [])]
    profiles = [_profile_from_search_user(u) for u in users if u.get("username")]
    return profiles[:limit]
