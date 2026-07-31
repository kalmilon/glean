"""Smoke tests — cheap, offline guards on the contracts every other module leans on.

No network, no binaries. Covers: the contract modules import; Config.resolve()'s
out-dir precedence; detect_source() URL mapping; and the `_v` version stamp on every
persisted record.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from glean.config import DEFAULT_OUT, Config
from glean.models import RECORD_VERSION, MediaItem, Result, Transcript, Word
from glean.sources import detect_source


def test_contract_modules_import():
    import glean.config  # noqa: F401
    import glean.models  # noqa: F401
    import glean.setup_cmd  # noqa: F401
    import glean.sources  # noqa: F401
    import glean.transcribe  # noqa: F401


def test_out_dir_precedence(monkeypatch, tmp_path):
    # Isolate from the developer's real ~/.config/glean/config.toml.
    monkeypatch.setenv("GLEAN_CONFIG", str(tmp_path / "no-such-config.toml"))
    monkeypatch.delenv("GLEAN_OUT", raising=False)

    # Nothing set → the built-in default.
    assert Config.resolve().out_dir == Path(DEFAULT_OUT)

    # $GLEAN_OUT wins over the default.
    monkeypatch.setenv("GLEAN_OUT", str(tmp_path / "from_env"))
    assert Config.resolve().out_dir == (tmp_path / "from_env")

    # --out flag wins over $GLEAN_OUT.
    args = SimpleNamespace(out=str(tmp_path / "from_flag"))
    assert Config.resolve(args).out_dir == (tmp_path / "from_flag")


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://www.instagram.com/reel/CxAbc123DEf/", "instagram"),
        ("https://www.twitch.tv/videos/1234567890", "twitch"),
        ("https://x.com/jack/status/20", "x"),
        ("https://twitter.com/jack/status/20", "x"),
    ],
)
def test_detect_source(url, expected):
    # The source modules are owned by other agents; skip cleanly until one lands,
    # then assert the real mapping once it's importable.
    pytest.importorskip(f"glean.sources.{expected}")
    assert detect_source(url) == expected


def test_to_dict_injects_version():
    tx = Transcript(text="hi", words=[Word("hi", 0.0, 0.5)], language="en", backend="whisper.cpp", model="ggml-large-v3-turbo", duration=0.5)
    assert tx.to_dict()["_v"] == RECORD_VERSION

    item = MediaItem(source="youtube", url="https://youtu.be/dQw4w9WgXcQ", id="dQw4w9WgXcQ")
    assert item.to_dict()["_v"] == RECORD_VERSION

    result = Result(item=item, transcript=tx, out_dir=str(item.id), files={})
    stamped = result.to_dict()
    assert stamped["_v"] == RECORD_VERSION
    assert stamped["item"]["_v"] == RECORD_VERSION
    assert stamped["transcript"]["_v"] == RECORD_VERSION


def test_doctor_runs_and_exits_zero(monkeypatch, tmp_path):
    from glean.setup_cmd import run_doctor

    monkeypatch.setenv("GLEAN_CONFIG", str(tmp_path / "no-such-config.toml"))
    monkeypatch.setenv("GLEAN_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("GLEAN_CACHE", str(tmp_path / "cache"))
    assert run_doctor(Config.resolve()) == 0
