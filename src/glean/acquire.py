"""Media acquisition — turn a URL into (metadata, 16 kHz mono WAV).

glean owns all downloading here. Three paths:

  * Instagram → a local Cobalt resolver running in Podman. Cobalt hands back a
    direct media URL (no Instagram cookies); we fetch it and downmix to WAV.
    Metadata comes from Instagram's public oEmbed endpoint (App-ID header).
  * TikTok → yt-dlp for its metadata, falling back to Cobalt when the extractor
    is broken, which for TikTok is a recurring rather than exceptional state.
  * Everything else (YouTube / Twitch / X / generic) → yt-dlp for both the
    bestaudio download and the `-J` metadata, then ffmpeg to WAV.

Transcription backends never touch the network — they receive the WAV this
module guarantees: 16 kHz, mono, PCM.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from glean.config import Config, ytdlp_cookie_opts
from glean.models import MediaItem
from glean.sources import detect_source

INSTAGRAM_APP_ID = "936619743392459"
INSTAGRAM_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

COBALT_CONTAINER = "clips-cobalt"
COBALT_IMAGE = "ghcr.io/imputnet/cobalt:11"
DEFAULT_COBALT_URL = "http://127.0.0.1:9000"

_IG_SHORTCODE = re.compile(r"instagram\.com/(?:reel|reels|p|tv)/([^/?#]+)")
_TIKTOK_ID = re.compile(r"tiktok\.com/(?:@[^/]+/)?(?:video|photo)/(\d+)")


def download_audio(url: str, cfg: Config) -> tuple[MediaItem, str]:
    """Return (metadata, path_to_16k_mono_wav). Dispatch by platform."""
    source = _classify(url)
    if source == "instagram":
        return _acquire_instagram(url, cfg)
    if source == "tiktok":
        return _acquire_tiktok(url, cfg)
    return _acquire_ytdlp(url, cfg, source)


def ffmpeg_to_wav(src: str, dst: str) -> None:
    """Downmix any media to 16 kHz mono WAV — the format every backend expects."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg to acquire audio") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed converting {src} to WAV: {e.stderr.strip()[-300:]}") from e


def _classify(url: str) -> str:
    """Name the platform for this URL, or 'generic' if no source claims it."""
    try:
        return detect_source(url)
    except Exception:
        return "generic"


# --- TikTok path (yt-dlp, then Cobalt) --------------------------------------


def _acquire_tiktok(url: str, cfg: Config) -> tuple[MediaItem, str]:
    """yt-dlp first for the metadata it carries, then Cobalt, which resolves media independently.

    TikTok reshapes its page often enough that yt-dlp's extractor breaks for stretches at a time —
    on 2026.7.4 it cannot extract a single video at all ("unable to extract universal data for
    rehydration"), while account listings still work. Cobalt does not read that page, so falling
    through to it costs the rich metadata rather than the transcript, which is the part worth
    protecting. If Cobalt is unreachable the yt-dlp failure is what surfaces: it is the more
    informative of the two, and the one that names the actual breakage.
    """
    try:
        return _acquire_ytdlp(url, cfg, "tiktok")
    except RuntimeError as ytdlp_error:
        item = MediaItem(source="tiktok", url=url, id=_tiktok_id(url) or url)
        job = cfg.job_dir(item)
        wav = job / "audio.wav"
        try:
            _cobalt_download_to_wav(url, wav, cfg, job)
        except Exception:
            raise ytdlp_error from None
        return item, str(wav)


def _tiktok_id(url: str) -> str | None:
    """The numeric video id in a TikTok URL, when it carries one."""
    m = _TIKTOK_ID.search(url)
    return m.group(1) if m else None


# --- yt-dlp path (youtube / twitch / x / generic) ---------------------------


def _acquire_ytdlp(url: str, cfg: Config, source: str) -> tuple[MediaItem, str]:
    """Download bestaudio + metadata via yt-dlp, then ffmpeg to a 16 kHz mono WAV."""
    import yt_dlp

    work = Path(tempfile.mkdtemp(prefix="glean-dl-", dir=_scratch(cfg)))
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        **ytdlp_cookie_opts(cfg),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"yt-dlp failed for {url}: {type(e).__name__}: {e}") from e

    if info.get("entries"):
        info = next((entry for entry in info["entries"] if entry), info)

    item = _item_from_ytdlp(info, url, source)
    downloaded = _largest_media(work)
    if downloaded is None:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"yt-dlp produced no audio file for {url}")

    job = cfg.job_dir(item)
    wav = job / "audio.wav"
    try:
        ffmpeg_to_wav(str(downloaded), str(wav))
        if cfg.keep_audio:
            shutil.move(str(downloaded), str(job / f"source{downloaded.suffix}"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return item, str(wav)


def _item_from_ytdlp(info: dict, url: str, source: str) -> MediaItem:
    """Map a yt-dlp info dict onto a MediaItem, keeping platform counters in `extra`."""
    video_id = str(info.get("id") or "") or _fallback_id(url)
    duration = info.get("duration")
    extra_keys = (
        "view_count", "like_count", "comment_count", "repost_count",
        "channel", "channel_id", "uploader_id", "extractor_key", "webpage_url", "ext",
    )
    extra = {k: info.get(k) for k in extra_keys if info.get(k) is not None}
    return MediaItem(
        source=source,
        url=info.get("webpage_url") or url,
        id=video_id,
        title=info.get("title"),
        author=info.get("uploader") or info.get("channel") or info.get("uploader_id"),
        published=_iso_from_ytdlp(info),
        duration=float(duration) if duration is not None else None,
        extra=extra,
    )


def _iso_from_ytdlp(info: dict) -> str | None:
    """Prefer a precise epoch timestamp; fall back to yt-dlp's YYYYMMDD upload_date."""
    ts = info.get("timestamp")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    upload_date = info.get("upload_date")
    if upload_date and re.fullmatch(r"\d{8}", str(upload_date)):
        s = str(upload_date)
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return None


def _largest_media(work: Path) -> Path | None:
    """The downloaded media file in `work` — the largest non-sidecar file present."""
    candidates = [p for p in work.iterdir() if p.is_file() and p.suffix.lower() not in (".json", ".part", ".ytdl")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# --- Instagram path (Cobalt in Podman) --------------------------------------


def _acquire_instagram(url: str, cfg: Config) -> tuple[MediaItem, str]:
    """Resolve a public Reel via local Cobalt, download it, and downmix to WAV."""
    reel_url = url.replace("/reels/", "/reel/")
    shortcode = _ig_shortcode(reel_url)
    item = _ig_metadata(reel_url, shortcode)
    job = cfg.job_dir(item)
    wav = job / "audio.wav"
    _cobalt_download_to_wav(reel_url, wav, cfg, job)
    return item, str(wav)


def _ig_shortcode(url: str) -> str:
    """Pull the Reel shortcode out of an Instagram URL (its stable item id)."""
    m = _IG_SHORTCODE.search(url)
    if not m:
        raise ValueError(f"not a recognisable Instagram Reel/post URL: {url!r}")
    return m.group(1)


def _ig_metadata(url: str, shortcode: str) -> MediaItem:
    """Best-effort public metadata via Instagram's oEmbed endpoint (no cookies)."""
    data = _ig_oembed(url)
    author = title = None
    extra: dict = {}
    if data:
        author = data.get("author_name")
        title = data.get("title")
        extra = {
            k: data.get(k)
            for k in ("author_name", "provider_name", "thumbnail_url", "thumbnail_width", "thumbnail_height")
            if data.get(k) is not None
        }
    return MediaItem(source="instagram", url=url, id=shortcode, title=title, author=author, extra=extra)


def _ig_oembed(url: str) -> dict | None:
    """Fetch Instagram's public oEmbed JSON with the App-ID header, or None."""
    headers = {
        "x-ig-app-id": INSTAGRAM_APP_ID,
        "user-agent": INSTAGRAM_USER_AGENT,
        "accept": "*/*",
    }
    endpoint = "https://i.instagram.com/api/v1/oembed/"
    for attempt in range(3):
        try:
            resp = requests.get(endpoint, params={"url": url}, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def _cobalt_download_to_wav(reel_url: str, wav: Path, cfg: Config, job: Path) -> None:
    """Get a media URL from Cobalt, download it, and produce the 16 kHz mono WAV.

    Starts a local Cobalt container only when the configured endpoint is the
    default loopback one and nothing is already listening. Falls back from the
    audio-only stream to the full video when the audio response is unusable.
    """
    base = (cfg.cobalt_url or DEFAULT_COBALT_URL).rstrip("/")
    started_here = False
    started_machine = False
    if not _cobalt_ready(base):
        if base != DEFAULT_COBALT_URL:
            raise RuntimeError(f"Cobalt is unavailable at {base}")
        started_here, started_machine = _start_local_cobalt(base)

    source = job / "ig-source"
    try:
        try:
            media_url = _cobalt_resolve(base, reel_url, "audio")
            _download_file(media_url, source)
            try:
                ffmpeg_to_wav(str(source), str(wav))
            except RuntimeError:
                media_url = _cobalt_resolve(base, reel_url, "auto")
                _download_file(media_url, source)
                ffmpeg_to_wav(str(source), str(wav))
            if cfg.keep_audio:
                shutil.move(str(source), str(job / "source"))
        finally:
            if not cfg.keep_audio:
                source.unlink(missing_ok=True)
    finally:
        if started_here:
            _stop_local_cobalt()
        if started_machine:
            subprocess.run(["podman", "machine", "stop"], capture_output=True)


def _cobalt_resolve(base: str, reel_url: str, mode: str) -> str:
    """POST to Cobalt and return the direct media URL it tunnels/redirects to."""
    payload: dict = {"url": reel_url, "downloadMode": mode, "filenameStyle": "basic"}
    if mode == "audio":
        payload["audioFormat"] = "wav"
    try:
        resp = requests.post(
            base + "/", json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Cobalt request failed: {e}") from e

    status = data.get("status", "error")
    if status in ("tunnel", "redirect"):
        media_url = data.get("url")
        if not media_url or not str(media_url).startswith(("http://", "https://")):
            raise RuntimeError("Cobalt returned an unsafe media URL")
        return media_url
    if status == "error":
        raise RuntimeError(f"Cobalt error: {data.get('error', {}).get('code', 'unknown')}")
    raise RuntimeError(f"unsupported Cobalt response: {status}")


def _download_file(media_url: str, dst: Path) -> None:
    """Stream a remote media URL to disk."""
    try:
        with requests.get(media_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dst, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"failed to download media from Cobalt tunnel: {e}") from e


def _cobalt_ready(base: str) -> bool:
    """True if a Cobalt instance answers at `base`."""
    try:
        return requests.get(base + "/", timeout=2).ok
    except requests.RequestException:
        return False


def _start_local_cobalt(base: str) -> tuple[bool, bool]:
    """Bring up the local Cobalt container in Podman, waiting until it answers.

    Returns (started_container, started_machine) so the caller knows what this
    invocation must tear down afterwards. Raises with an actionable message if
    Podman is missing or Cobalt never becomes ready — on the timeout path the
    container (and machine) this invocation started are stopped before raising.
    """
    if not shutil.which("podman"):
        # Pointing at an already-running Cobalt is the other way out of this, and on a machine
        # without podman it is usually the only one — a container sharing its host's network
        # namespace reaches the host's Cobalt at this very address. Naming only podman sends
        # people off to install a container runtime when nothing here needs to run a container.
        raise RuntimeError(
            f"Instagram downloads need Cobalt, and nothing is listening at {base}. "
            "Either install podman so one can be started here, or run Cobalt elsewhere "
            "and set GLEAN_COBALT_URL (or cobalt_url in config) to its address."
        )

    started_machine = False
    if subprocess.run(["podman", "info"], capture_output=True).returncode != 0:
        subprocess.run(["podman", "machine", "start"], capture_output=True)
        started_machine = True

    if subprocess.run(["podman", "container", "exists", COBALT_CONTAINER]).returncode == 0:
        if subprocess.run(["podman", "start", COBALT_CONTAINER], capture_output=True).returncode != 0:
            subprocess.run(["podman", "rm", "-f", COBALT_CONTAINER], capture_output=True)
            subprocess.run(["podman", "image", "rm", "-f", COBALT_IMAGE], capture_output=True)
            _podman_run_cobalt(base)
    else:
        _podman_run_cobalt(base)

    for _ in range(30):
        if _cobalt_ready(base):
            return True, started_machine
        time.sleep(1)
    _stop_local_cobalt()
    if started_machine:
        subprocess.run(["podman", "machine", "stop"], capture_output=True)
    raise RuntimeError(f"local Cobalt did not become ready; inspect with: podman logs {COBALT_CONTAINER}")


def _podman_run_cobalt(base: str) -> None:
    """Launch the Cobalt image bound to loopback:9000."""
    subprocess.run(
        ["podman", "run", "-d", "--name", COBALT_CONTAINER,
         "-p", "127.0.0.1:9000:9000", "-e", f"API_URL={base}", COBALT_IMAGE],
        capture_output=True, check=True,
    )


def _stop_local_cobalt() -> None:
    """Release the Cobalt container once the media is local."""
    subprocess.run(["podman", "stop", "--time", "5", COBALT_CONTAINER], capture_output=True)


def _scratch(cfg: Config) -> str:
    """A per-run scratch parent under the cache dir for staging downloads."""
    base = cfg.cache_dir / "work"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _fallback_id(url: str) -> str:
    """A stable, filesystem-safe id derived from the URL when the extractor gives none."""
    import hashlib

    return "url-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
