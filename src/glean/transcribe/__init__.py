"""Backend registry and platform auto-detection.

Selection order when no backend is named:
    macOS + Parakeet available  →  parakeet
    whisper.cpp available        →  whisper.cpp
    otherwise                    →  BackendUnavailable (points at `glean setup`)
"""

from __future__ import annotations

import sys

from glean.transcribe.base import BackendUnavailable, TranscribeBackend


def _construct(name: str, cfg) -> TranscribeBackend:
    if name in ("parakeet", "ane"):
        from glean.transcribe.parakeet import ParakeetBackend
        return ParakeetBackend(cfg)
    if name in ("whisper.cpp", "whisper", "whispercpp"):
        from glean.transcribe.whisper_cpp import WhisperCppBackend
        return WhisperCppBackend(cfg)
    raise BackendUnavailable(f"unknown backend {name!r} (expected 'parakeet' or 'whisper.cpp')")


def get_backend(name: str | None, cfg=None) -> TranscribeBackend:
    """Return an explicit backend by name, or auto-detect the best local one."""
    if name:
        backend = _construct(name, cfg)
        if not backend.available():
            raise BackendUnavailable(f"backend {name!r} is not available on this machine — run `glean setup`")
        return backend

    order = ["parakeet", "whisper.cpp"] if sys.platform == "darwin" else ["whisper.cpp", "parakeet"]
    for candidate in order:
        try:
            backend = _construct(candidate, cfg)
        except BackendUnavailable:
            continue
        if backend.available():
            return backend

    raise BackendUnavailable(
        "no local transcription backend available. On macOS install fluidaudiocli; "
        "on Linux run `glean setup` to build whisper.cpp and fetch a model."
    )
