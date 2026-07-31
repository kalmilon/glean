# glean

A local-first research tool. Point it at a YouTube video, Instagram reel, Twitch VOD, or
X post and it hands back the **transcript + metadata** — on disk and as JSON. Transcription
runs **locally and free**: Parakeet on the Apple Neural Engine on macOS, whisper.cpp on
Linux. No cloud, no API key.

Built for both a Mac dev box and a headless Alpine (musl) Linux server, and for agents that
want to drop research artifacts into a folder of their choosing.

## Install

```bash
uv tool install git+ssh://git@github.com/<you>/glean.git   # or: uv sync in a clone
glean setup      # on Linux: builds whisper.cpp + fetches the model; checks ffmpeg/yt-dlp
glean doctor     # confirm the environment is ready
```

## Use

```bash
glean yt   https://youtu.be/VIDEO                 # one video → transcript
glean yt   channel @handle --limit 20             # batch a channel feed
glean yt   search "topic" --limit 20
glean ig   https://instagram.com/reel/XXXX/       # one reel → transcript
glean ig   list  @account --limit 24              # recent reels + metadata
glean ig   scout @a @b --top 30 --since-days 90   # rank an account's reels
glean ig   profile @account                       # full profile (followers, bio, verified) — cookieless
glean ig   search "query" --limit 20              # find accounts — needs a session cookie (see below)
glean twitch https://twitch.tv/videos/123         # VOD / clip
glean x      https://x.com/user/status/123        # tweet video
glean url    <any-url>                             # auto-detect the platform
```

## Where things land

Each job writes `<out>/<source>/<id>/` containing `metadata.json`,
`transcript.txt`, `transcript.json` (word timings), and `transcript.srt`.

Output directory precedence — pick whichever fits your workflow:

```
--out DIR   >   $GLEAN_OUT   >   [out] in ~/.config/glean/config.toml   >   ./glean-out
```

Add `--json` to print machine-readable results to stdout (for agents/pipelines),
`--backend {parakeet,whisper.cpp}` to force an engine, `--language CODE` to hint,
`--keep-audio` to retain the downloaded media.

## Batch transcription & run manifests

Discovery verbs (`yt channel`, `yt search`, `ig list`, `ig scout`) emit **metadata only**
by default. Add `--transcribe` to run every result through the pipeline and get transcripts
back instead. Per-item failures are logged to stderr and skipped — one bad video never
aborts the batch.

```bash
glean yt channel @handle --limit 20 --transcribe          # transcribe a whole channel feed
glean ig list @account --limit 24 --transcribe --json      # transcripts as a JSON array
```

`--run-dir DIR` bundles a whole run into one place: every per-item job dir lands under `DIR`,
and glean writes a `DIR/run.json` manifest — a versioned index of the run (`_v`, `tool`,
`version`, `kind`, `created`, `note`, `params`, `count`, `transcribed`, and a per-record
summary). Pair it with `--run-note TEXT` to stamp a description into the manifest. It works on
every discovery verb, transcribing or not (including `ig profile`/`ig search`, which record the
returned profiles).

```bash
glean yt search "climbing drills" --limit 15 --transcribe \
    --run-dir ~/research/climbing --run-note "drill breakdowns for the coaching deck"
# → ~/research/climbing/youtube/<id>/... plus ~/research/climbing/run.json
```

## YouTube captions (skip transcription)

`--captions` is **YouTube only**: pull YouTube's own caption track instead of downloading audio
and transcribing locally. Fast and free when a video is already captioned; glean errors clearly
if the URL isn't YouTube or the video has no captions.

```bash
glean yt https://youtu.be/VIDEO --captions                 # one video, via its captions
glean yt channel @handle --transcribe --captions           # caption the whole feed
```

## Instagram search & cookies

`ig profile <@handle>` works anonymously. Free-text `ig search <query>` does **not** —
Instagram blocks anonymous search, so it needs a logged-in session:

```bash
export GLEAN_IG_SESSIONID="<the 'sessionid' cookie from your instagram.com session>"
glean ig search "climbing coach"
```

(or `ig_sessionid = "..."` in `~/.config/glean/config.toml`). A cookie also raises the
rate limits on `list`/`scout`.

## How it works

- **Acquire**: `yt-dlp` for YouTube / Twitch / X; a local **Cobalt** container (via Podman)
  for cookieless Instagram. Audio is normalized to 16 kHz mono WAV with `ffmpeg`.
- **Transcribe**: whichever local backend fits the machine (auto-detected; override with `--backend`).
- **Output**: versioned JSON + text + SRT.

Nothing leaves the machine.
