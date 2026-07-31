"""Transcription backend contract.

A backend takes a 16 kHz mono WAV and returns a Transcript. Backends never download
media — acquire guarantees the WAV. All backends are local and free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from glean.models import Transcript


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend's binary or model is not present."""


@runtime_checkable
class TranscribeBackend(Protocol):
    name: str

    def available(self) -> bool:
        """True if this backend can run on the current machine (binary + model present)."""
        ...

    def transcribe(self, wav_path: str, *, language: str | None = None, word_timestamps: bool = True) -> Transcript:
        ...
