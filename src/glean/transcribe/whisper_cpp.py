"""whisper.cpp backend — the free, musl-native engine (Linux default, macOS fallback).

Runs the `whisper-cli` binary with `-oj -ojf`, which writes `<prefix>.json` containing
segment- and token-level offsets in MILLISECONDS. We fold whisper's subword tokens back
into whole words (a token beginning with a space opens a new word), converting ms → s. If
tokens are absent we fall back to one "word" per segment so downstream SRT still has cues.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from glean.models import Transcript, Word
from glean.transcribe.base import BackendUnavailable

BINARY_CANDIDATES = ("whisper-cli", "whisper-cpp", "main")


def _binary(cfg=None) -> str | None:
    override = os.environ.get("GLEAN_WHISPER_BIN")
    if override:
        return override if (shutil.which(override) or os.path.exists(override)) else None
    for candidate in BINARY_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    build_bin = _built_binary(cfg)
    if build_bin:
        return build_bin
    return None


def _built_binary(cfg) -> str | None:
    """Probe the well-known location `glean setup` builds whisper.cpp into."""
    cache_dir = getattr(cfg, "cache_dir", None)
    if cache_dir is None:
        return None
    build_bin = Path(cache_dir) / "whisper.cpp" / "build" / "bin"
    for candidate in BINARY_CANDIDATES:
        found = build_bin / candidate
        if found.exists():
            return str(found)
    return None


def _is_special_token(text: str) -> bool:
    """whisper markers like [_BEG_], [_EOT_], [_TT_123] — never part of the spoken word."""
    return text.startswith("[_") and text.endswith("]")


class WhisperCppBackend:
    """Transcribe with a local whisper.cpp build against a ggml model file."""

    name = "whisper.cpp"

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.binary = _binary(cfg)
        self.model_path = self._resolve_model(cfg)

    def _resolve_model(self, cfg) -> Path | None:
        if cfg is not None and hasattr(cfg, "default_whisper_model"):
            return cfg.default_whisper_model()
        env_model = os.environ.get("GLEAN_WHISPER_MODEL")
        return Path(env_model).expanduser() if env_model else None

    def available(self) -> bool:
        return self.binary is not None and self.model_path is not None and self.model_path.exists()

    def transcribe(self, wav_path: str, *, language: str | None = None, word_timestamps: bool = True) -> Transcript:
        if not self.available():
            raise BackendUnavailable("whisper.cpp backend unavailable (needs binary + ggml model) — run `glean setup`")

        threads = os.cpu_count() or 4
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "out")
            cmd = [
                self.binary, "-m", str(self.model_path), "-f", wav_path,
                "-oj", "-ojf", "-otxt", "-osrt", "-of", prefix,
                "-l", language or "auto", "-t", str(threads),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            data = self._load_json(prefix + ".json", cmd, proc)

        return self._to_transcript(data, word_timestamps)

    def _load_json(self, path: str, cmd: list[str], proc) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"whisper-cli produced no usable JSON (rc={proc.returncode}): {proc.stderr.strip()[:400]}") from exc

    def _to_transcript(self, data: dict, word_timestamps: bool) -> Transcript:
        segments = data.get("transcription") or []
        text = "".join(seg.get("text", "") for seg in segments).strip()
        language = (data.get("result") or {}).get("language")
        words = self._words_from_tokens(segments) if word_timestamps else []
        if not words:
            words = self._words_from_segments(segments)
        duration = self._duration(segments)
        return Transcript(
            text=text,
            words=words,
            language=language,
            backend=self.name,
            model=Path(str(data.get("params", {}).get("model") or self.model_path or "")).stem,
            duration=duration,
        )

    def _words_from_tokens(self, segments: list) -> list[Word]:
        words: list[Word] = []
        buf, start, end = "", None, None
        for seg in segments:
            for tok in seg.get("tokens") or []:
                piece = tok.get("text", "")
                if _is_special_token(piece):
                    continue
                offsets = tok.get("offsets") or {}
                if "from" not in offsets or "to" not in offsets:
                    continue
                if piece.startswith(" ") and buf.strip():
                    words.append(Word(text=buf.strip(), start=start or 0.0, end=end or 0.0))
                    buf, start = "", None
                if start is None:
                    start = offsets["from"] / 1000.0
                buf += piece
                end = offsets["to"] / 1000.0
        if buf.strip():
            words.append(Word(text=buf.strip(), start=start or 0.0, end=end or 0.0))
        return words

    def _words_from_segments(self, segments: list) -> list[Word]:
        words: list[Word] = []
        for seg in segments:
            offsets = seg.get("offsets") or {}
            piece = seg.get("text", "").strip()
            if not piece or "from" not in offsets or "to" not in offsets:
                continue
            words.append(Word(text=piece, start=offsets["from"] / 1000.0, end=offsets["to"] / 1000.0))
        return words

    def _duration(self, segments: list) -> float | None:
        last = None
        for seg in segments:
            to = (seg.get("offsets") or {}).get("to")
            if to is not None:
                last = to / 1000.0
        return last
