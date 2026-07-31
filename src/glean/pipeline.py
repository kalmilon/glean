"""The transcription pipeline: URL in, Result out.

Three stages, each owned by a sibling module:
  acquire  → (metadata, 16 kHz mono WAV)
  backend  → Transcript
  output   → files on disk + a Result to hand back
"""

from __future__ import annotations

from glean import acquire, output
from glean.config import Config
from glean.models import Result
from glean.sources import detect_source
from glean.transcribe import get_backend


def transcribe_url(url: str, cfg: Config, backend: str | None = None, captions: bool = False) -> Result:
    """Acquire audio for `url`, transcribe it locally, and write the result to disk.

    With `captions=True` the transcript comes from YouTube's own caption track (no
    audio download, no ASR backend); this path is YouTube-only.
    """
    if captions:
        try:
            is_youtube = detect_source(url) == "youtube"
        except ValueError:
            is_youtube = False
        if not is_youtube:
            raise RuntimeError("--captions is only available for YouTube")
        from glean.sources import youtube

        item = youtube.YouTubeSource().fetch_metadata(url, cfg)
        tx = youtube.captions(url, cfg, language=cfg.language)
        return output.write_result(item, tx, cfg)
    item, wav = acquire.download_audio(url, cfg)
    engine = get_backend(backend or cfg.backend, cfg)
    tx = engine.transcribe(wav, language=cfg.language)
    return output.write_result(item, tx, cfg)
