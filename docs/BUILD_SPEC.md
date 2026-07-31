# glean — Build Spec (authoritative)

`glean` is a **local-first research tool**: give it a URL (or an account/query) from
YouTube, Instagram, Twitch, or X, and it returns the **transcript + metadata** on disk
and as JSON. Transcription runs **locally and free** — Parakeet on the Apple Neural
Engine on macOS, whisper.cpp on Linux. No cloud, no API key required for the default path.

This file is the single source of truth. Every build agent MUST read it fully and
implement to these interfaces exactly. Do not invent alternate signatures. Do not modify
files you do not own.

Target machines: a macOS dev box (Apple Silicon) and an **Alpine Linux (musl) x86_64,
CPU-only** remote. **Every dependency must be musl-safe** — pure-Python only. No numpy,
no torch, no onnxruntime, no ctranslate2, no lxml. Parse HTML with `bs4` + the stdlib
`html.parser`. External heavy work is done by *binaries* (ffmpeg, yt-dlp, whisper.cpp,
fluidaudiocli), never Python C-extensions.

---

## Package layout & file ownership

```
glean/
  pyproject.toml                 [CONTRACT — already written]
  src/glean/
    __init__.py                  [CONTRACT]
    models.py                    [CONTRACT]  Word, Transcript, MediaItem, Result
    config.py                    [CONTRACT]  Config + resolution
    transcribe/
      base.py                    [CONTRACT]  TranscribeBackend Protocol, exceptions
      __init__.py                [CONTRACT]  get_backend() + auto-detect
      parakeet.py                [AGENT: backends]
      whisper_cpp.py             [AGENT: backends]
    acquire.py                   [AGENT: pipeline]
    pipeline.py                  [AGENT: pipeline]
    output.py                    [AGENT: pipeline]
    cli.py                       [AGENT: cli]
    setup_cmd.py                 [AGENT: setup]
    sources/
      base.py                    [CONTRACT]  Source Protocol
      __init__.py                [CONTRACT]  detect_source(), get_source()
      youtube.py                 [AGENT: youtube]
      instagram.py               [AGENT: instagram]
      twitch.py                  [AGENT: twitch-x]
      x.py                       [AGENT: twitch-x]
  tests/                         [AGENT: setup]
```

CONTRACT files are written before the workflow and are frozen — import from them, never edit.

---

## Data models (`glean/models.py`) — the shared currency

```python
RECORD_VERSION = 1

@dataclass
class Word:      text: str; start: float; end: float          # seconds

@dataclass
class Transcript:
    text: str
    words: list[Word]            # may be empty if a backend gives no word timings
    language: str | None
    backend: str                 # "parakeet" | "whisper.cpp"
    model: str                   # e.g. "v2" or "ggml-large-v3-turbo"
    duration: float | None

@dataclass
class MediaItem:
    source: str                  # "youtube" | "instagram" | "twitch" | "x"
    url: str
    id: str
    title: str | None = None
    author: str | None = None
    published: str | None = None # ISO 8601
    duration: float | None = None
    extra: dict = {}             # platform metadata: plays, likes, view_count, ...

@dataclass
class Result:
    item: MediaItem
    transcript: Transcript | None
    out_dir: str                 # the per-job directory
    files: dict                  # {"metadata": path, "txt": path, "json": path, "srt": path}
```

`.to_dict()` on Transcript/MediaItem/Result returns a JSON-safe dict and injects `"_v":
RECORD_VERSION` at the top level of every persisted record. Readers dispatch on `_v`.

---

## Transcription backend interface (`transcribe/base.py`)

```python
class TranscribeBackend(Protocol):
    name: str
    def available(self) -> bool: ...
    def transcribe(self, wav_path: str, *, language: str | None = None,
                   word_timestamps: bool = True) -> Transcript: ...
```

Input is ALWAYS a 16 kHz mono WAV (acquire guarantees this). Backends never download media.

### `parakeet.py` (macOS / Apple Neural Engine) — the current engine
Invoke the `fluidaudiocli` binary (env `GLEAN_FLUIDAUDIO_BIN`, default `fluidaudiocli`):
```
fluidaudiocli transcribe <wav> --model-version v2 --word-timestamps \
    --output-json <tmp.json> [--language <code>]
```
Run it once on a small WAV to learn the exact JSON shape, then map it → `Transcript`
(text, words with start/end, language, backend="parakeet", model="v2"). `available()`:
`shutil.which` the binary AND `sys.platform == "darwin"`. Default model version "v2"
(overridable via env `GLEAN_PARAKEET_MODEL`).

### `whisper_cpp.py` (Linux / macOS fallback) — free, musl-native
Locate the binary: env `GLEAN_WHISPER_BIN`, else first of `whisper-cli`, `whisper-cpp`,
`main` on PATH. Model: env `GLEAN_WHISPER_MODEL`, else `<cache>/models/ggml-large-v3-turbo.bin`.
Invoke:
```
whisper-cli -m <model> -f <wav> -oj -ojf -otxt -osrt -of <out_prefix> \
    -l <lang or "auto"> -t <threads>
```
`-oj -ojf` writes `<out_prefix>.json` (full: token-level offsets). Parse it → `Transcript`.
Derive `words` from token/segment offsets (offsets are ms). If only segment-level offsets
are available, emit segment-level "words" rather than nothing. `model` = the ggml basename.
`available()`: binary present AND model file exists (setup fetches it). `threads`: default
`os.cpu_count()`.

---

## Source interface (`sources/base.py`)

```python
class Source(Protocol):
    name: str
    def matches(self, url: str) -> bool: ...          # does this URL belong to me?
    def fetch_metadata(self, url: str, cfg: "Config") -> MediaItem: ...
```

`sources/__init__.py` (CONTRACT) exposes:
- `detect_source(url) -> str` — return the name of the first source whose `matches()` is true.
- `get_source(name) -> Source` — construct a source by name (lazy import, tolerant).

Discovery verbs (channel/search/list/scout) are **module-level functions**, not part of the
Protocol (they differ per platform). The CLI imports and calls them directly. Signatures the
CLI relies on (implement EXACTLY):

```python
# sources/youtube.py
class YouTubeSource: ...                                   # implements Source
def channel(handle: str, cfg: Config, limit: int = 30) -> list[MediaItem]: ...
def search(query: str, cfg: Config, limit: int = 30) -> list[MediaItem]: ...

# sources/instagram.py
class InstagramSource: ...
def list_account(username: str, cfg: Config, limit: int = 24) -> list[MediaItem]: ...
def scout(usernames: list[str], cfg: Config, top: int = 30, since_days: int = 90) -> list[MediaItem]: ...

# sources/twitch.py  → class TwitchSource
# sources/x.py       → class XSource
```

---

## Acquire / pipeline / output (`acquire.py`, `pipeline.py`, `output.py`)

### `acquire.py`
```python
def download_audio(url: str, cfg: Config) -> tuple[MediaItem, str]:
    """Return (metadata, path_to_16k_mono_wav). Dispatch by platform."""
```
- **Instagram** → Cobalt-in-Podman path (port from the ig bash below). Get the media URL
  from a local Cobalt container, download it, `ffmpeg` → 16 kHz mono WAV. Metadata from
  Instagram's public JSON (App-ID header, no cookies).
- **Everything else (youtube/twitch/x/generic)** → `yt-dlp`: fetch bestaudio + `-J` metadata,
  then `ffmpeg -ac 1 -ar 16000` → WAV. yt-dlp natively supports YouTube, Twitch, and X.
- Provide `ffmpeg_to_wav(src, dst)` helper: `ffmpeg -y -i src -vn -ac 1 -ar 16000 dst`.
- Respect `cfg.keep_audio` (keep the source download) else clean up.

### `pipeline.py`
```python
def transcribe_url(url: str, cfg: Config, backend: str | None = None) -> Result:
    item, wav = acquire.download_audio(url, cfg)
    tx = get_backend(backend, cfg).transcribe(wav, language=cfg.language)
    return output.write_result(item, tx, cfg)
```

### `output.py`
```python
def write_result(item: MediaItem, tx: Transcript | None, cfg: Config) -> Result:
    """Create cfg.job_dir(item), write metadata.json + transcript.{txt,json,srt}, return Result."""
def to_srt(tx: Transcript) -> str: ...
```
Every JSON file carries `_v`. `transcript.txt` = plain text. `transcript.srt` = cues from
`words` (or segment fallback). Print nothing here; the CLI decides human vs `--json` output.

---

## Config (`glean/config.py`) — agent-friendly by design

```python
@dataclass
class Config:
    out_dir: Path
    cache_dir: Path
    backend: str | None
    language: str | None
    keep_audio: bool
    json_output: bool
    cobalt_url: str | None
    parakeet_model: str
    whisper_model: Path | None

    @classmethod
    def resolve(cls, args, env=os.environ) -> "Config": ...
    def job_dir(self, item: MediaItem) -> Path:   # <out_dir>/<source>/<id>/
```

**out_dir resolution order (this is the feature agents care about):**
`--out FLAG` → `$GLEAN_OUT` → `[out] dir` in `$GLEAN_CONFIG` or `~/.config/glean/config.toml`
→ default `./glean-out`. cache_dir: `$GLEAN_CACHE` → `$XDG_CACHE_HOME/glean` → `~/.cache/glean`.
Directories are created on demand. Config is read with the stdlib `tomllib`.

---

## CLI grammar (`glean/cli.py`) — `main()` is the entry point

```
glean url  <URL>                         # auto-detect platform → transcribe
glean yt   <URL|ID>                      # transcribe one video
glean yt   channel <@handle|URL> [--limit N]
glean yt   search  "<query>" [--limit N]
glean ig   <URL>                         # transcribe one reel
glean ig   list  <@account> [--limit N]
glean ig   scout <@a> <@b>... [--top N] [--since-days N]
glean twitch <URL>                       # VOD / clip → transcribe
glean x    <URL>                         # tweet video → transcribe
glean setup                              # bootstrap deps (build whisper.cpp + fetch model on Linux)
glean doctor                             # print environment diagnosis

Global flags: --out DIR  --backend {parakeet,whisper.cpp}  --language CODE
              --json  --keep-audio
```
`--json` prints `Result.to_dict()` (or a list) to stdout for agent consumption; otherwise
print a short human summary (title, out_dir, first lines of transcript). Use argparse
subparsers. Exit non-zero with a clear message on failure. `setup`/`doctor` live in
`setup_cmd.py`; `cli.py` calls into them.

---

## Porting sources (read these when implementing)

- YouTube backend + Gemini/captions + yt-dlp download + channel logic:
  `/Users/kalmilon/work/personal-projects/yt-transcripts/yt_transcripts.py`
- Instagram list/scout/transcribe/research + Cobalt/Podman orchestration:
  `/Users/kalmilon/work/mindfront-social-media/tools/instagram/ig`
  (and `ig-transcribe`, `ig-download` in the same dir)
- Parakeet/fluidaudiocli invocation reference + ffmpeg WAV normalization:
  `/Users/kalmilon/work/mindfront-social-media/clips.py` (transcribe section)

Port the *logic*, not the structure — target the interfaces above. Keep it clean and typed.
Function signatures stay on one line. Name things for their role. No cloud backend in the
default path (Gemini may be added later as an opt-in, not now).

## Validation each agent runs before returning
- `python -c "import ast; ast.parse(open('<your file>').read())"` for each file you wrote.
- Backends: actually run the binary on a tiny WAV and confirm your parser maps the real JSON.
- Do not leave `TODO`/`pass`-only bodies in the core path. Twitch/X may reuse the yt-dlp path.
