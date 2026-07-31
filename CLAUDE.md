# glean — agent guide

Local-first research tool. URL (or account/query) → transcript + metadata, written to
disk and emitted as JSON. Transcription is local and free: Parakeet/ANE on macOS,
whisper.cpp on Linux (Alpine/musl target).

## Golden rules
- **musl-safe, pure-Python deps only.** No numpy/torch/onnxruntime/ctranslate2/lxml.
  Heavy work is done by binaries (ffmpeg, yt-dlp, whisper.cpp, fluidaudiocli).
- **No cloud in the default path.** No API keys required to transcribe.
- Architecture and interfaces live in `docs/BUILD_SPEC.md` — that file is authoritative.
- Prefer `uv` over pip. Function signatures stay on one line. Name things for their role.
- Persisted records carry `_v` (see `models.RECORD_VERSION`).

## Shape
```
src/glean/
  cli.py          entry point (`glean` command)
  config.py       out-dir resolution: --out > $GLEAN_OUT > config.toml > ./glean-out
  pipeline.py     acquire → transcribe → write
  acquire.py      yt-dlp (youtube/twitch/x) + Cobalt/Podman (instagram)
  output.py       metadata.json + transcript.{txt,json,srt}
  transcribe/     parakeet.py (mac), whisper_cpp.py (linux); auto-detected
  sources/        youtube.py, instagram.py, twitch.py, x.py
```

## Run
```
uv sync
uv run glean doctor
uv run glean yt <url>
uv run glean ig <reel-url>
uv run glean url <any-url> --out ~/research --json
```

## Discovery flags (channel/search/list/scout)
- `--transcribe` — run every discovered item through `batch.transcribe_items` and emit Results
  instead of bare metadata. Per-item failures are logged to stderr and skipped, never fatal.
- `--run-dir DIR [--run-note TEXT]` — set `cfg.out_dir = DIR` so per-item job dirs collect under
  it, then write `DIR/run.json` via `output.write_run_manifest(kind, params, records, note,
  transcribed)`. The manifest carries `_v` and a per-record summary; readers dispatch on `_v`.
- `yt --captions` (YouTube only) — use YouTube's own caption track instead of local
  transcription. Single video → `pipeline.transcribe_url(url, cfg, captions=True)`; with
  `--transcribe` on channel/search → the whole batch uses captions. Errors clearly on non-YouTube.

## Cookies (optional)
- `GLEAN_YT_COOKIES=<cookies.txt>` or `GLEAN_YT_COOKIES_FROM_BROWSER=<chrome|safari|firefox>` —
  passed to every yt-dlp call (download, channel, search, captions) via `config.ytdlp_cookie_opts`.
  Fixes YouTube's "confirm you're not a bot" blocks.
- `GLEAN_IG_SESSIONID=<sessionid>` — enables `ig search` and lifts `ig list`/`scout` rate limits.
- Both also settable in `~/.config/glean/config.toml` (`yt_cookies` / `yt_cookies_from_browser` / `ig_sessionid`).

## Adding a source
Implement the `Source` protocol in `sources/<name>.py`, register it in `sources/__init__.py`,
add a subparser in `cli.py`. If the platform is yt-dlp-supported, reuse the yt-dlp acquire path.
