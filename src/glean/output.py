"""Result persistence — write metadata + transcript to the per-job directory.

Every job lands as four files under `cfg.job_dir(item)`:
  metadata.json    item.to_dict()      (carries _v)
  transcript.txt   plain text
  transcript.json  tx.to_dict()        (carries _v)
  transcript.srt   timed cues from word timings (segment fallback if none)

This module prints nothing; the CLI decides human vs `--json` presentation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import glean
from glean.config import Config
from glean.models import RECORD_VERSION, MediaItem, Result, Transcript, Word

_MAX_CUE_WORDS = 10
_MAX_CUE_GAP = 1.0     # seconds of silence that forces a new cue
_MAX_CUE_SPAN = 6.0    # seconds a single cue may cover
_FALLBACK_WORDS_PER_CUE = 12


def write_result(item: MediaItem, tx: Transcript | None, cfg: Config) -> Result:
    """Create the job directory, write every artefact, and return a Result."""
    job = cfg.job_dir(item)
    files: dict = {}

    metadata_path = job / "metadata.json"
    _write_json(metadata_path, item.to_dict())
    files["metadata"] = str(metadata_path)

    if tx is not None:
        txt_path = job / "transcript.txt"
        txt_path.write_text(tx.text or "", encoding="utf-8")
        files["txt"] = str(txt_path)

        json_path = job / "transcript.json"
        _write_json(json_path, tx.to_dict())
        files["json"] = str(json_path)

        srt_path = job / "transcript.srt"
        srt_path.write_text(to_srt(tx), encoding="utf-8")
        files["srt"] = str(srt_path)

    return Result(item=item, transcript=tx, out_dir=str(job), files=files)


def write_run_manifest(run_dir, kind: str, params: dict, records: list, note=None, transcribed: bool = False) -> str:
    """Write <run_dir>/run.json — a versioned index of one discovery/batch run — and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "_v": RECORD_VERSION,
        "tool": "glean",
        "version": glean.__version__,
        "kind": kind,
        "created": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "params": params,
        "count": len(records),
        "transcribed": transcribed,
        "results": [r.to_dict() if hasattr(r, "to_dict") else str(r) for r in records],
    }
    path = run_dir / "run.json"
    _write_json(path, manifest)
    return str(path)


def to_srt(tx: Transcript) -> str:
    """Render a Transcript as SubRip cues — grouped word timings, or a text fallback."""
    cues = _cues_from_words(tx.words) if tx.words else _cues_from_text(tx)
    return _render(cues)


def _cues_from_words(words: list[Word]) -> list[tuple[float, float, str]]:
    """Group consecutive words into readable cues by count, span, and silence gap."""
    cues: list[tuple[float, float, str]] = []
    bucket: list[Word] = []
    for word in words:
        if bucket:
            span = word.end - bucket[0].start
            gap = word.start - bucket[-1].end
            if len(bucket) >= _MAX_CUE_WORDS or span > _MAX_CUE_SPAN or gap > _MAX_CUE_GAP:
                cues.append(_flush(bucket))
                bucket = []
        bucket.append(word)
    if bucket:
        cues.append(_flush(bucket))
    return cues


def _flush(bucket: list[Word]) -> tuple[float, float, str]:
    text = " ".join(w.text.strip() for w in bucket).strip()
    return (bucket[0].start, bucket[-1].end, text)


def _cues_from_text(tx: Transcript) -> list[tuple[float, float, str]]:
    """Fallback: chunk plain text into cues spread evenly across the duration."""
    tokens = (tx.text or "").split()
    if not tokens:
        return []
    total = tx.duration if tx.duration and tx.duration > 0 else max(1.0, len(tokens) / 2.5)
    chunks = [tokens[i:i + _FALLBACK_WORDS_PER_CUE] for i in range(0, len(tokens), _FALLBACK_WORDS_PER_CUE)]
    count = len(chunks)
    return [(total * i / count, total * (i + 1) / count, " ".join(chunk)) for i, chunk in enumerate(chunks)]


def _render(cues: list[tuple[float, float, str]]) -> str:
    """Serialise (start, end, text) cues into SubRip format."""
    lines: list[str] = []
    for index, (start, end, text) in enumerate(cues, start=1):
        if end <= start:
            end = start + 0.5
        lines.append(str(index))
        lines.append(f"{_timestamp(start)} --> {_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp: HH:MM:SS,mmm."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
