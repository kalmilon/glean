"""Parakeet backend — Apple Neural Engine transcription via the `fluidaudiocli` binary.

macOS only. `fluidaudiocli transcribe <wav> --model-version v2 --word-timestamps
--output-json <tmp>` writes a JSON blob whose `wordTimings` carry start/end in SECONDS.
This is the default engine on the Apple Silicon dev box; free, offline, on-device.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from glean.models import Transcript, Word
from glean.transcribe.base import BackendUnavailable

DEFAULT_MODEL = "v2"


def _binary() -> str:
    return os.environ.get("GLEAN_FLUIDAUDIO_BIN") or "fluidaudiocli"


class ParakeetBackend:
    """Transcribe with FluidAudio's Parakeet CoreML model on the Neural Engine."""

    name = "parakeet"

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.binary = _binary()
        self.model = getattr(cfg, "parakeet_model", None) or os.environ.get("GLEAN_PARAKEET_MODEL") or DEFAULT_MODEL

    def available(self) -> bool:
        return sys.platform == "darwin" and shutil.which(self.binary) is not None

    def transcribe(self, wav_path: str, *, language: str | None = None, word_timestamps: bool = True) -> Transcript:
        if not self.available():
            raise BackendUnavailable("parakeet backend unavailable (needs macOS + fluidaudiocli) — run `glean setup`")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_json = tmp.name
        try:
            cmd = [self.binary, "transcribe", wav_path, "--model-version", self.model, "--output-json", out_json]
            if word_timestamps:
                cmd.append("--word-timestamps")
            if language:
                cmd += ["--language", language]
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            data = self._load_json(out_json, cmd)
        finally:
            try:
                os.unlink(out_json)
            except OSError:
                pass

        return self._to_transcript(data, language)

    def _load_json(self, path: str, cmd: list[str]) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"fluidaudiocli produced no usable JSON ({' '.join(cmd)}): {exc}") from exc

    def _to_transcript(self, data: dict, language: str | None) -> Transcript:
        words = [
            Word(text=w.get("word", ""), start=float(w.get("startTime", 0.0)), end=float(w.get("endTime", 0.0)))
            for w in data.get("wordTimings") or []
        ]
        ds = data.get("durationSeconds")
        duration = float(ds) if ds else (words[-1].end if words else None)
        return Transcript(
            text=(data.get("text") or "").strip(),
            words=words,
            language=language,
            backend=self.name,
            model=str(data.get("modelVersion") or self.model),
            duration=duration,
        )
