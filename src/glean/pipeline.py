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
from glean.transcribe import get_backend


def transcribe_url(url: str, cfg: Config, backend: str | None = None) -> Result:
    """Acquire audio for `url`, transcribe it locally, and write the result to disk."""
    item, wav = acquire.download_audio(url, cfg)
    engine = get_backend(backend or cfg.backend, cfg)
    tx = engine.transcribe(wav, language=cfg.language)
    return output.write_result(item, tx, cfg)
