"""Batch transcription — turn a list of discovered items into Results.

Discovery verbs (yt channel/search, ig list/scout) yield MediaItems. With
`--transcribe`, the CLI hands that list here and we transcribe each one through the
normal pipeline, one at a time. A single item's failure is logged to stderr and
skipped — it never aborts the batch. Only the successful Results come back.
"""

from __future__ import annotations

import sys

from glean.config import Config
from glean.models import MediaItem, Result


def transcribe_items(items: list[MediaItem], cfg: Config, captions: bool = False) -> list[Result]:
    """Transcribe each MediaItem via the pipeline; log and skip per-item failures."""
    from glean import pipeline

    results: list[Result] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{total}] {item.source} {item.id}", file=sys.stderr, flush=True)
        try:
            results.append(pipeline.transcribe_url(item.url, cfg, captions=captions))
        except Exception as err:
            print(f"glean: skipped {item.url}: {err}", file=sys.stderr, flush=True)
    return results
