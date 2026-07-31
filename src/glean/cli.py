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

    parser = argparse.ArgumentParser(prog="glean", description="Local-first transcript + metadata research tool.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_url = sub.add_parser("url", parents=[common], help="auto-detect the platform then transcribe")
    p_url.add_argument("target", help="any supported media URL")

    p_yt = sub.add_parser("yt", parents=[common], help="YouTube: a video URL/ID, or `channel <@handle>` / `search <query>`")
    p_yt.add_argument("target", nargs="+", metavar="TARGET", help="video URL/ID, or `channel <@handle>` / `search <query>`")
    p_yt.add_argument("--limit", type=int, default=30, help="max items for channel/search (default 30)")

    p_ig = sub.add_parser("ig", parents=[common], help="Instagram: a reel URL, or `list <@account>` / `scout <@a> <@b>...`")
    p_ig.add_argument("target", nargs="+", metavar="TARGET", help="reel URL, or `list <@account>` / `scout <@a> <@b>...`")
    p_ig.add_argument("--limit", type=int, default=24, help="max reels for `list` (default 24)")
    p_ig.add_argument("--top", type=int, default=30, help="max results for `scout` (default 30)")
    p_ig.add_argument("--since-days", type=int, default=90, dest="since_days", help="`scout` recency window in days (default 90)")

    p_tw = sub.add_parser("twitch", parents=[common], help="Twitch VOD or clip → transcribe")
    p_tw.add_argument("target", help="a Twitch VOD or clip URL")

    p_x = sub.add_parser("x", parents=[common], help="X (Twitter) video → transcribe")
    p_x.add_argument("target", help="a tweet URL containing video")

    sub.add_parser("setup", parents=[common], help="bootstrap local dependencies (whisper.cpp + model on Linux)")
    sub.add_parser("doctor", parents=[common], help="print an environment diagnosis")

    return parser


def _transcribe_one(url: str, cfg: Config) -> int:
    from glean import pipeline
    result = pipeline.transcribe_url(url, cfg, backend=cfg.backend)
    _emit_result(result, cfg)
    return 0


def _cmd_url(args, cfg: Config) -> int:
    detect_source(args.target)  # raises ValueError with a clear message if unrecognised
    return _transcribe_one(args.target, cfg)


def _cmd_yt(args, parser: argparse.ArgumentParser, cfg: Config) -> int:
    tokens = args.target
    verb = tokens[0].lower()
    if verb == "channel":
        if len(tokens) < 2:
            parser.error("yt channel needs a <@handle|URL>")
        from glean.sources import youtube
        _emit_items(youtube.channel(tokens[1], cfg, limit=args.limit), cfg)
        return 0
    if verb == "search":
        query = " ".join(tokens[1:]).strip()
        if not query:
            parser.error("yt search needs a query")
        from glean.sources import youtube
        _emit_items(youtube.search(query, cfg, limit=args.limit), cfg)
        return 0
    return _transcribe_one(tokens[0], cfg)


def _cmd_ig(args, parser: argparse.ArgumentParser, cfg: Config) -> int:
    tokens = args.target
    verb = tokens[0].lower()
    if verb == "list":
        if len(tokens) < 2:
            parser.error("ig list needs an <@account>")
        from glean.sources import instagram
        _emit_items(instagram.list_account(tokens[1], cfg, limit=args.limit), cfg)
        return 0
    if verb == "scout":
        usernames = tokens[1:]
        if not usernames:
            parser.error("ig scout needs one or more <@account> arguments")
        from glean.sources import instagram
        _emit_items(instagram.scout(usernames, cfg, top=args.top, since_days=args.since_days), cfg)
        return 0
    return _transcribe_one(tokens[0], cfg)


def _cmd_setup(cfg: Config) -> int:
    from glean import setup_cmd
    setup_cmd.run_setup(cfg)
    return 0


def _cmd_doctor(cfg: Config) -> int:
    from glean import setup_cmd
    setup_cmd.run_doctor(cfg)
    return 0


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
            return _transcribe_one(args.target, cfg)
        if args.command == "setup":
            return _cmd_setup(cfg)
        if args.command == "doctor":
            return _cmd_doctor(cfg)
    except BackendUnavailable as exc:
        print(f"glean: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"glean: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command {args.command!r}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    sys.exit(main())
