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


def test_single_target_honours_run_dir(monkeypatch, tmp_path):
    """`--run-dir` on one video must land its job dir and manifest under DIR.

    It used to be wired into the discovery verbs only, so passing it with a single video parsed,
    exited 0 and wrote nothing — a silent no-op that reads as success.
    """
    from glean import cli, pipeline

    captured = {}

    def fake_transcribe_url(url, cfg, backend=None, captions=False):
        captured["out_dir"] = cfg.out_dir
        captured["captions"] = captions
        return Result(
            item=MediaItem(source="youtube", url=url, id="aaaaaaaaaaa", title="One"),
            transcript=Transcript(text="hi", words=[Word(text="hi", start=0.0, end=1.0)], language="en", backend="test", model="test"),
            out_dir=str(tmp_path / "job"),
            files={},
        )

    monkeypatch.setattr(pipeline, "transcribe_url", fake_transcribe_url)

    run_dir = tmp_path / "run1"
    args = SimpleNamespace(run_dir=str(run_dir), run_note="a note", captions=True)
    rc = cli._transcribe_one("https://youtu.be/aaaaaaaaaaa", args, Config.resolve())

    assert rc == 0
    assert captured["out_dir"] == run_dir, "cfg.out_dir must point at the run dir before transcription"
    assert captured["captions"] is True, "--captions must reach every single-target path"

    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert data["kind"] == "transcribe"
    assert data["note"] == "a note"
    assert data["transcribed"] is True
    assert data["count"] == 1


def test_run_flags_accepted_on_every_transcribing_command():
    """url/twitch/x gained the run-manifest flags that previously lived on yt and ig alone."""
    from glean.cli import build_parser

    parser = build_parser()
    for cmd in ("url", "yt", "ig", "twitch", "x"):
        args = parser.parse_args([cmd, "https://example.com/v", "--run-dir", "d", "--run-note", "n"])
        assert args.run_dir == "d" and args.run_note == "n", cmd

    # `url` delegates to the YouTube path, so it must accept YouTube's own flag too.
    assert parser.parse_args(["url", "https://youtu.be/x", "--captions"]).captions is True


def _reel(code, plays, *, cluster=None, title=None, artist=None, asset=None, orig_title=None, orig_artist=None):
    """A minimal feed item shaped like the two audio layouts Instagram actually returns."""
    cm = {"audio_type": "licensed_music" if cluster else ("original_sounds" if asset else "")}
    if cluster:
        cm["music_info"] = {"music_asset_info": {"audio_cluster_id": cluster, "title": title, "display_artist": artist}}
    if asset:
        cm["original_sound_info"] = {"audio_asset_id": asset, "original_audio_title": orig_title, "ig_artist": {"username": orig_artist}}
    return {"code": code, "play_count": plays, "taken_at": 9_999_999_999, "clips_metadata": cm}


def test_sound_of_reads_both_audio_layouts():
    from glean.sources.instagram import _sound_of

    assert _sound_of(_reel("a", 1, cluster="c1", title="Track", artist="Band")) == ("c1", "licensed_music", "Track", "Band")
    assert _sound_of(_reel("b", 1, asset="a1", orig_title="Original audio", orig_artist="someone")) == ("a1", "original_sounds", "Original audio", "someone")
    assert _sound_of({"code": "c", "clips_metadata": {}}) is None


def test_sounds_group_by_track_and_rank_by_reuse(monkeypatch):
    """Reuse must outrank raw reach: a track two accounts reached for beats one bigger lone reel."""
    from glean.sources import instagram

    feeds = {
        "one": [_reel("r1", 100, cluster="c1", title="Shared", artist="Band"),
                _reel("r2", 900, asset="a1", orig_title="Original audio", orig_artist="one")],
        "two": [_reel("r3", 200, cluster="c1", title="Shared", artist="Band")],
    }
    monkeypatch.setattr(instagram, "_fetch_reels", lambda name, limit, cfg=None: feeds[name])

    found = instagram.sounds(["@one", "@two"], cfg=None)
    assert [s.id for s in found] == ["c1", "a1"], "two uses must outrank a single louder reel"

    shared = found[0]
    assert shared.uses == 2 and shared.total_plays == 300
    assert shared.accounts == ["one", "two"]
    assert shared.url == "https://www.instagram.com/reels/audio/c1/"
    assert len(shared.examples) == 2

    # An original sound has no cluster, so it has no browsable page — better empty than a 404.
    assert found[1].url == ""
    assert found[1].to_dict()["_v"] == RECORD_VERSION


def test_sounds_music_only_drops_original_audio(monkeypatch):
    from glean.sources import instagram

    feed = [_reel("r1", 100, cluster="c1", title="Track", artist="Band"),
            _reel("r2", 900, asset="a1", orig_title="Original audio", orig_artist="one")]
    monkeypatch.setattr(instagram, "_fetch_reels", lambda name, limit, cfg=None: feed)

    assert [s.id for s in instagram.sounds(["@one"], cfg=None, music_only=True)] == ["c1"]
    assert len(instagram.sounds(["@one"], cfg=None)) == 2


def _tt(vid, views, track, artist=None, uploader=None, ts=9_999_999_999):
    return {"id": vid, "url": f"https://www.tiktok.com/@{uploader or 'x'}/video/{vid}",
            "view_count": views, "track": track, "artist": artist, "uploader": uploader, "timestamp": ts}


def test_tiktok_original_sound_is_keyed_per_creator():
    """Every creator's own audio is called "original sound"; keying on the title alone merged the lot.

    Before this, 39 videos across two unrelated accounts reported as one shared sound.
    """
    from glean.sources.tiktok import _sound_of

    a = _sound_of(_tt("1", 10, "original sound", artist=None, uploader="alice"))
    b = _sound_of(_tt("2", 10, "original sound", artist=None, uploader="bob"))
    assert a[0] != b[0], "two creators' original sounds must not share an id"
    assert a[1] == b[1] == "original_sounds"
    assert a[3] == "alice", "a blank artist field falls back to the uploader"

    # The suffixed form is the same thing and must not be filed as licensed music.
    suffixed = _sound_of(_tt("3", 10, "original sound - alice", artist="alice", uploader="alice"))
    assert suffixed[1] == "original_sounds"

    # A real track groups on name + artist, across accounts.
    t1 = _sound_of(_tt("4", 10, "Some Track", artist="A Band", uploader="alice"))
    t2 = _sound_of(_tt("5", 10, "some track", artist="A Band", uploader="bob"))
    assert t1[0] == t2[0] and t1[1] == "licensed_music"

    assert _sound_of(_tt("6", 10, "")) is None


def test_tiktok_sounds_group_and_rank(monkeypatch):
    from glean.sources import tiktok

    feeds = {
        "alice": [_tt("1", 100, "Shared", "Band", "alice"), _tt("2", 900, "original sound", None, "alice")],
        "bob": [_tt("3", 200, "Shared", "Band", "bob")],
    }
    monkeypatch.setattr(tiktok, "_extract", lambda url, **kw: {"entries": feeds[url.rsplit("@", 1)[1]]})

    found = tiktok.sounds(["@alice", "@bob"], cfg=None)
    assert [s.title for s in found] == ["Shared", "original sound"], "reuse outranks a louder lone video"
    assert found[0].uses == 2 and found[0].total_plays == 300
    assert found[0].accounts == ["alice", "bob"]
    assert found[0].source == "tiktok"
    # TikTok's music page needs a numeric id yt-dlp does not report — better empty than guessed.
    assert found[0].url == ""

    assert [s.title for s in tiktok.sounds(["@alice"], cfg=None, music_only=True)] == ["Shared"]


def test_tiktok_falls_back_to_cobalt_when_ytdlp_breaks(monkeypatch, tmp_path):
    """TikTok's extractor breaks for stretches; a broken extractor must not cost the transcript."""
    from glean import acquire

    cfg = Config.resolve()
    cfg.out_dir = tmp_path

    def broken_ytdlp(url, c, source):
        raise RuntimeError("yt-dlp failed: unable to extract universal data for rehydration")

    calls = {}

    def fake_cobalt(url, wav, c, job):
        calls["url"] = url
        Path(wav).parent.mkdir(parents=True, exist_ok=True)
        Path(wav).write_bytes(b"RIFF")

    monkeypatch.setattr(acquire, "_acquire_ytdlp", broken_ytdlp)
    monkeypatch.setattr(acquire, "_cobalt_download_to_wav", fake_cobalt)

    url = "https://www.tiktok.com/@someone/video/7668726289376808222"
    item, wav = acquire.download_audio(url, cfg)
    assert item.source == "tiktok"
    assert item.id == "7668726289376808222", "the id comes off the URL when yt-dlp gave nothing"
    assert calls["url"] == url and Path(wav).exists()


def test_tiktok_surfaces_the_ytdlp_error_when_cobalt_also_fails(monkeypatch, tmp_path):
    """Cobalt being absent must not mask the real reason — the extractor is what broke."""
    from glean import acquire

    cfg = Config.resolve()
    cfg.out_dir = tmp_path

    def broken_ytdlp(url, c, source):
        raise RuntimeError("yt-dlp failed: unable to extract universal data")

    def no_cobalt(url, wav, c, job):
        raise RuntimeError("Instagram downloads need Cobalt, and nothing is listening")

    monkeypatch.setattr(acquire, "_acquire_ytdlp", broken_ytdlp)
    monkeypatch.setattr(acquire, "_cobalt_download_to_wav", no_cobalt)

    with pytest.raises(RuntimeError, match="unable to extract universal data"):
        acquire.download_audio("https://www.tiktok.com/@a/video/123", cfg)


def test_cluster_id_accepts_id_or_url():
    from glean.sources.instagram import cluster_id_of

    assert cluster_id_of("1939359193596829") == "1939359193596829"
    assert cluster_id_of("https://www.instagram.com/reels/audio/1939359193596829/") == "1939359193596829"
    with pytest.raises(ValueError, match="audio cluster id"):
        cluster_id_of("@someone")


def test_reels_for_sound_requires_a_cookie(monkeypatch):
    from glean.sources import instagram

    monkeypatch.delenv("GLEAN_IG_SESSIONID", raising=False)
    with pytest.raises(RuntimeError, match="session cookie"):
        instagram.reels_for_sound("123", cfg=None)


def test_reels_for_sound_pages_and_dedupes(monkeypatch):
    """The endpoint wraps reels in {"media": ...}, pages by max_id, and can repeat across pages."""
    from glean.sources import instagram

    monkeypatch.setenv("GLEAN_IG_SESSIONID", "fake")
    monkeypatch.setattr(instagram.time, "sleep", lambda *_: None)

    pages = [
        {"items": [{"media": {"code": "A", "play_count": 10}}, {"media": {"code": "B", "play_count": 20}}],
         "paging_info": {"max_id": "cursor1"}},
        # "B" repeats, and this page hands the media back flat rather than wrapped.
        {"items": [{"code": "B", "play_count": 20}, {"code": "C", "play_count": 30}], "paging_info": {}},
    ]
    calls = []

    def fake_post(url, data, cfg=None):
        calls.append(data)
        return pages[len(calls) - 1]

    monkeypatch.setattr(instagram, "_post_json", fake_post)

    items = instagram.reels_for_sound("999", cfg=None, limit=24)
    assert [i.id for i in items] == ["A", "B", "C"], "a reel seen twice must appear once"
    assert calls[0]["audio_cluster_id"] == "999" and "max_id" not in calls[0]
    assert calls[1]["max_id"] == "cursor1", "the second page must carry the cursor"
    assert all(i.source == "instagram" for i in items)


def test_reels_for_sound_stops_at_limit(monkeypatch):
    from glean.sources import instagram

    monkeypatch.setenv("GLEAN_IG_SESSIONID", "fake")
    monkeypatch.setattr(instagram.time, "sleep", lambda *_: None)
    monkeypatch.setattr(instagram, "_post_json", lambda url, data, cfg=None: {
        "items": [{"media": {"code": f"c{n}", "play_count": n}} for n in range(10)],
        "paging_info": {"max_id": "more"},
    })

    assert len(instagram.reels_for_sound("999", cfg=None, limit=3)) == 3


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")
    def json(self):
        return self._payload


def test_auth_failure_is_not_retried(monkeypatch):
    """401 four times over seven seconds reported "failed after 4 attempts" — it reads as a flaky
    network and sends you looking for one. A wall must be named on the first hit."""
    from glean.sources import instagram

    calls = []
    monkeypatch.setattr(instagram.requests, "get", lambda *a, **k: calls.append(1) or _Resp(401))
    monkeypatch.setattr(instagram.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="session cookie"):
        instagram._get_json("https://example.test/x")
    assert len(calls) == 1, "an auth wall must not be retried"


def test_rate_limit_is_retried_and_named(monkeypatch):
    """429 is the failure worth waiting out, and the message should say it usually clears."""
    from glean.sources import instagram

    slept = []
    monkeypatch.setattr(instagram.requests, "get", lambda *a, **k: _Resp(429))
    monkeypatch.setattr(instagram.time, "sleep", lambda s: slept.append(s))

    with pytest.raises(RuntimeError, match="clears on its own"):
        instagram._get_json("https://example.test/x")
    assert len(slept) == instagram._RETRIES - 1
    assert slept[0] >= instagram._RATE_LIMIT_BACKOFF, "429 backs off harder than a generic blip"


def test_successful_request_returns_payload(monkeypatch):
    from glean.sources import instagram

    monkeypatch.setattr(instagram.requests, "get", lambda *a, **k: _Resp(200, {"ok": True}))
    assert instagram._get_json("https://example.test/x") == {"ok": True}
