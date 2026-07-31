"""Smoke tests — cheap, offline guards on the contracts every other module leans on.

No network, no binaries. Covers: the contract modules import; Config.resolve()'s
out-dir precedence; detect_source() URL mapping; and the `_v` version stamp on every
persisted record.
"""

from __future__ import annotations

import json
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


def test_profile_record_versioned():
    from glean.models import Profile, RECORD_VERSION
    p = Profile(source="instagram", username="x", followers=10)
    assert p.to_dict()["_v"] == RECORD_VERSION
    assert p.to_dict()["username"] == "x"


def test_ig_search_requires_cookie(monkeypatch):
    import pytest
    monkeypatch.delenv("GLEAN_IG_SESSIONID", raising=False)
    from glean.sources import instagram
    with pytest.raises(RuntimeError, match="session cookie"):
        instagram.search("anything", cfg=None)


def test_write_run_manifest(tmp_path):
    from glean.output import write_run_manifest

    items = [
        MediaItem(source="youtube", url="https://youtu.be/aaaaaaaaaaa", id="aaaaaaaaaaa", title="One"),
        MediaItem(source="youtube", url="https://youtu.be/bbbbbbbbbbb", id="bbbbbbbbbbb", title="Two"),
    ]
    path = write_run_manifest(tmp_path, "yt-channel", {"handle": "@x", "limit": 30}, items, note="a note", transcribed=False)

    assert path == str(tmp_path / "run.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["_v"] == RECORD_VERSION
    assert data["tool"] == "glean"
    assert data["kind"] == "yt-channel"
    assert data["count"] == 2
    assert data["transcribed"] is False
    assert data["note"] == "a note"
    assert data["params"] == {"handle": "@x", "limit": 30}
    assert "created" in data and "version" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["_v"] == RECORD_VERSION


def test_captions_on_non_youtube_raises(monkeypatch, tmp_path):
    # The captions path is YouTube-only; on any other platform the pipeline must
    # reject it up front (a RuntimeError) before it ever reaches the network.
    monkeypatch.setenv("GLEAN_CONFIG", str(tmp_path / "no-such-config.toml"))
    monkeypatch.setenv("GLEAN_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("GLEAN_CACHE", str(tmp_path / "cache"))
    from glean import acquire, pipeline

    def _no_network(*args, **kwargs):
        raise AssertionError("captions guard must fire before any download")

    monkeypatch.setattr(acquire, "download_audio", _no_network)
    with pytest.raises(RuntimeError):
        pipeline.transcribe_url("https://x.com/jack/status/20", Config.resolve(), captions=True)


def test_ytdlp_cookie_opts():
    from glean.config import Config, ytdlp_cookie_opts
    assert ytdlp_cookie_opts(Config.resolve(env={"GLEAN_YT_COOKIES_FROM_BROWSER": "chrome"})) == {"cookiesfrombrowser": ("chrome",)}
    assert ytdlp_cookie_opts(Config.resolve(env={"GLEAN_YT_COOKIES": "/tmp/c.txt"})) == {"cookiefile": "/tmp/c.txt"}
    assert ytdlp_cookie_opts(Config.resolve(env={})) == {}
    assert ytdlp_cookie_opts(None) == {}


def test_doctor_resolvers(tmp_path, monkeypatch):
    """yt-dlp is detected as a module (not a PATH binary), and whisper resolves
    from the setup-built cache location even when nothing is on PATH."""
    from glean import setup_cmd
    from glean.config import Config
    assert setup_cmd._ytdlp_version()  # ships as a dependency
    built = tmp_path / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    built.parent.mkdir(parents=True)
    built.write_text("#!/bin/sh\n")
    built.chmod(0o755)
    cfg = Config.resolve(env={"GLEAN_CACHE": str(tmp_path)})
    monkeypatch.setenv("PATH", "")
    assert setup_cmd._whisper_bin({}, cfg) == str(built)
    assert setup_cmd._whisper_bin({}) is None
