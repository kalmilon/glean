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

## How it works

- **Acquire**: `yt-dlp` for YouTube / Twitch / X; a local **Cobalt** container (via Podman)
  for cookieless Instagram. Audio is normalized to 16 kHz mono WAV with `ffmpeg`.
- **Transcribe**: whichever local backend fits the machine (auto-detected; override with `--backend`).
- **Output**: versioned JSON + text + SRT.

Nothing leaves the machine.
