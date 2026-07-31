"""Command-line entry point. Parses the `glean` grammar, resolves a Config, and
dispatches to the pipeline (single-URL transcription) or to a source's discovery
verbs (channel/search/list/scout, which emit metadata only).

Sibling modules that do heavy work (pipeline, sources.*, setup_cmd) are imported
lazily inside each handler so the parser can be built even before they exist and a
single broken module never blocks an unrelated command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glean.config import Config
from glean.sources import detect_source
from glean.transcribe.base import BackendUnavailable

PREVIEW_CHARS = 400


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argparse grammar. Global flags live on every subcommand
    (via a shared parent) so they may follow the subcommand, e.g. `glean yt URL --json`."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", metavar="DIR", help="output directory (overrides $GLEAN_OUT and config)")
    common.add_argument("--backend", choices=["parakeet", "whisper.cpp"], help="force a transcription backend")
    common.add_argument("--language", metavar="CODE", help="language hint for transcription (e.g. en)")
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON to stdout")
    common.add_argument("--keep-audio", action="store_true", dest="keep_audio", help="keep the downloaded source audio")

    # Run-manifest flags belong to every command that produces job dirs, which is all of them except
    # setup and doctor. They lived on yt and ig alone, so the same run collected through `url`,
    # `twitch` or `x` had nowhere to go — and the shorthand was less capable than what it delegates to.
    runflags = argparse.ArgumentParser(add_help=False)
    runflags.add_argument("--run-dir", metavar="DIR", dest="run_dir", help="collect this run's job dirs under DIR and write a run.json manifest")
    runflags.add_argument("--run-note", metavar="TEXT", dest="run_note", help="a note recorded in the run.json manifest")

    parser = argparse.ArgumentParser(prog="glean", description="Local-first transcript + metadata research tool.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_url = sub.add_parser("url", parents=[common, runflags], help="auto-detect the platform then transcribe")
    p_url.add_argument("target", help="any supported media URL")
    # `url` dispatches to the YouTube path for a YouTube link, so rejecting YouTube's own flag here
    # made the shorthand quietly less capable than the command it delegates to. It is inert for the
    # other platforms, which is already what --captions means for a video that has no captions.
    p_url.add_argument("--captions", action="store_true", help="use the platform's own captions instead of local transcription (YouTube only)")

    p_yt = sub.add_parser("yt", parents=[common, runflags], help="YouTube: a video URL/ID, or `channel <@handle>` / `search <query>`")
    p_yt.add_argument("target", nargs="+", metavar="TARGET", help="video URL/ID, or `channel <@handle>` / `search <query>`")
    p_yt.add_argument("--limit", type=int, default=30, help="max items for channel/search (default 30)")
    p_yt.add_argument("--transcribe", action="store_true", help="transcribe every channel/search result (default: metadata only)")
    p_yt.add_argument("--captions", action="store_true", help="use YouTube's own captions instead of local transcription (YouTube only)")

    p_ig = sub.add_parser("ig", parents=[common, runflags], help="Instagram: reel URL, `list`/`scout <@acct>`, `profile <@handle>`, `search <query>`")
    p_ig.add_argument("target", nargs="+", metavar="TARGET", help="reel URL, or `list`/`scout <@acct>`, `profile <@handle>`, `search <query>`")
    p_ig.add_argument("--limit", type=int, default=24, help="max results for `list`/`search` (default 24)")
    p_ig.add_argument("--top", type=int, default=30, help="max results for `scout` (default 30)")
    p_ig.add_argument("--since-days", type=int, default=90, dest="since_days", help="`scout` recency window in days (default 90)")
    p_ig.add_argument("--transcribe", action="store_true", help="transcribe every list/scout result (default: metadata only)")

    p_tw = sub.add_parser("twitch", parents=[common, runflags], help="Twitch VOD or clip → transcribe")
    p_tw.add_argument("target", help="a Twitch VOD or clip URL")

    p_x = sub.add_parser("x", parents=[common, runflags], help="X (Twitter) video → transcribe")
    p_x.add_argument("target", help="a tweet URL containing video")

    sub.add_parser("setup", parents=[common], help="bootstrap local dependencies (whisper.cpp + model on Linux)")
    sub.add_parser("doctor", parents=[common], help="print an environment diagnosis")

    return parser


def _transcribe_one(url: str, args, cfg: Config) -> int:
    """Transcribe a single target, honouring `--run-dir` the way a discovery run does.

    One video is still a run: setting cfg.out_dir first puts its job dir inside the run directory,
    and the manifest records it. Without this the flag parsed, exited 0 and wrote nothing, which is
    the one failure that reads as success. `--captions` is read here too so that every single-target
    path shares it rather than only the one command that branched early for it.
    """
    from glean import pipeline
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        cfg.out_dir = Path(run_dir)
    result = pipeline.transcribe_url(
        url, cfg, backend=cfg.backend, captions=bool(getattr(args, "captions", False))
    )
    _emit_result(result, cfg)
    if run_dir:
        from glean.output import write_run_manifest
        path = write_run_manifest(
            run_dir, "transcribe", {"target": url}, [result],
            note=getattr(args, "run_note", None), transcribed=True,
        )
        print(f"run: {path}", file=sys.stderr)
    return 0


def _cmd_url(args, cfg: Config) -> int:
    detect_source(args.target)  # raises ValueError with a clear message if unrecognised
    return _transcribe_one(args.target, args, cfg)


def _cmd_yt(args, parser: argparse.ArgumentParser, cfg: Config) -> int:
    tokens = args.target
    verb = tokens[0].lower()
    if verb == "channel":
        if len(tokens) < 2:
            parser.error("yt channel needs a <@handle|URL>")
        from glean.sources import youtube
        items = youtube.channel(tokens[1], cfg, limit=args.limit)
        return _run_discovery(items, "yt-channel", {"handle": tokens[1], "limit": args.limit}, args, cfg)
    if verb == "search":
        query = " ".join(tokens[1:]).strip()
        if not query:
            parser.error("yt search needs a query")
        from glean.sources import youtube
        items = youtube.search(query, cfg, limit=args.limit)
        return _run_discovery(items, "yt-search", {"query": query, "limit": args.limit}, args, cfg)
    return _transcribe_one(tokens[0], args, cfg)


def _cmd_ig(args, parser: argparse.ArgumentParser, cfg: Config) -> int:
    tokens = args.target
    verb = tokens[0].lower()
    if verb == "list":
        if len(tokens) < 2:
            parser.error("ig list needs an <@account>")
        from glean.sources import instagram
        items = instagram.list_account(tokens[1], cfg, limit=args.limit)
        return _run_discovery(items, "ig-list", {"account": tokens[1], "limit": args.limit}, args, cfg)
    if verb == "scout":
        usernames = tokens[1:]
        if not usernames:
            parser.error("ig scout needs one or more <@account> arguments")
        from glean.sources import instagram
        items = instagram.scout(usernames, cfg, top=args.top, since_days=args.since_days)
        return _run_discovery(items, "ig-scout", {"accounts": usernames, "top": args.top, "since_days": args.since_days}, args, cfg)
    if verb == "profile":
        if len(tokens) < 2:
            parser.error("ig profile needs an <@handle>")
        from glean.sources import instagram
        return _run_profiles([instagram.profile(tokens[1], cfg)], "ig-profile", {"handle": tokens[1]}, args, cfg)
    if verb == "search":
        query = " ".join(tokens[1:]).strip()
        if not query:
            parser.error("ig search needs a <query>")
        from glean.sources import instagram
        return _run_profiles(instagram.search(query, cfg, limit=args.limit), "ig-search", {"query": query, "limit": args.limit}, args, cfg)
    return _transcribe_one(tokens[0], args, cfg)


def _cmd_setup(cfg: Config) -> int:
    from glean import setup_cmd
    return setup_cmd.run_setup(cfg)


def _cmd_doctor(cfg: Config) -> int:
    from glean import setup_cmd
    setup_cmd.run_doctor(cfg)
    return 0


def _run_discovery(items, kind: str, params: dict, args, cfg: Config) -> int:
    """Emit discovered MediaItems — transcribing them first when `--transcribe` is set — and write a run.json manifest when `--run-dir` is given.

    Records land in the manifest as Results (transcribed) or MediaItems (metadata only). Setting cfg.out_dir before transcription pulls every per-item job dir inside the run directory.
    """
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        cfg.out_dir = Path(run_dir)
    transcribe = bool(getattr(args, "transcribe", False))
    if transcribe:
        from glean import batch
        records = batch.transcribe_items(list(items), cfg, captions=bool(getattr(args, "captions", False)))
        _emit_results(records, cfg)
    else:
        records = list(items)
        _emit_items(records, cfg)
    if run_dir:
        from glean.output import write_run_manifest
        path = write_run_manifest(run_dir, kind, params, records, note=getattr(args, "run_note", None), transcribed=transcribe)
        print(f"run: {path}", file=sys.stderr)
    return 0


def _run_profiles(profiles, kind: str, params: dict, args, cfg: Config) -> int:
    """Emit Profiles and, when `--run-dir` is given, write a run.json manifest of them (profiles are never transcribed)."""
    profiles = list(profiles)
    _emit_profiles(profiles, cfg)
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        from glean.output import write_run_manifest
        path = write_run_manifest(run_dir, kind, params, profiles, note=getattr(args, "run_note", None), transcribed=False)
        print(f"run: {path}", file=sys.stderr)
    return 0


def _emit_results(results, cfg: Config) -> None:
    """Emit a batch of Results — a JSON array under `--json`, else a per-item human summary."""
    results = list(results)
    if cfg.json_output:
        print(json.dumps([r.to_dict() for r in results]))
        return
    print(f"{len(results)} transcript(s)")
    for result in results:
        _emit_result(result, cfg)


def _emit_result(result, cfg: Config) -> None:
    if cfg.json_output:
        print(json.dumps(result.to_dict()))
        return
    item = result.item
    print(item.title or item.id or item.url)
    print(f"  out: {result.out_dir}")
    tx = result.transcript
    if tx is None or not (tx.text or "").strip():
        print("  (no transcript text)")
        return
    lines = [ln for ln in (tx.text or "").strip().splitlines() if ln.strip()]
    preview = "\n".join(lines[:2]) if lines else (tx.text or "").strip()
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS].rstrip() + "…"
    for ln in preview.splitlines():
        print(f"  {ln}")


def _emit_profiles(profiles, cfg: Config) -> None:
    profiles = list(profiles)
    if cfg.json_output:
        print(json.dumps([p.to_dict() for p in profiles]))
        return
    print(f"{len(profiles)} profile(s)")
    for p in profiles:
        badge = " ✓" if p.verified else ""
        followers = f"{p.followers:,}" if isinstance(p.followers, int) else "?"
        print(f"- @{p.username}{badge}  ({followers} followers)")
        if p.full_name:
            print(f"    {p.full_name}")
        if p.bio:
            print(f"    {p.bio.splitlines()[0][:100]}")
        print(f"    {p.url}")


def _emit_items(items, cfg: Config) -> None:
    items = list(items)
    if cfg.json_output:
        print(json.dumps([it.to_dict() for it in items]))
        return
    print(f"{len(items)} item(s)")
    for it in items:
        head = it.title or it.id
        print(f"- {head}")
        if it.author:
            print(f"    by {it.author}")
        print(f"    {it.url}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return 2

    cfg = Config.resolve(args)

    try:
        if args.command == "url":
            return _cmd_url(args, cfg)
        if args.command == "yt":
            return _cmd_yt(args, parser, cfg)
        if args.command == "ig":
            return _cmd_ig(args, parser, cfg)
        if args.command in ("twitch", "x"):
            return _transcribe_one(args.target, args, cfg)
        if args.command == "setup":
            return _cmd_setup(cfg)
        if args.command == "doctor":
            return _cmd_doctor(cfg)
    except BackendUnavailable as exc:
        print(f"glean: {exc}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"glean: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command {args.command!r}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    sys.exit(main())
