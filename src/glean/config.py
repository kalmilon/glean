"""Runtime configuration, resolved once per invocation.

Agent-friendly by design: the output directory is resolvable four ways, in
precedence order, so an orchestrating agent can drop transcripts exactly where it
wants without editing anything:

    --out FLAG  >  $GLEAN_OUT  >  [out] in config.toml  >  ./glean-out
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from glean.models import MediaItem

DEFAULT_OUT = "glean-out"
DEFAULT_PARAKEET_MODEL = "v2"


def _config_file(env) -> Path:
    return Path(env.get("GLEAN_CONFIG", os.path.expanduser("~/.config/glean/config.toml")))


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


@dataclass
class Config:
    out_dir: Path
    cache_dir: Path
    backend: str | None = None
    language: str | None = None
    keep_audio: bool = False
    json_output: bool = False
    cobalt_url: str | None = None
    parakeet_model: str = DEFAULT_PARAKEET_MODEL
    whisper_model: Path | None = None
    ig_cookie: str | None = None    # Instagram sessionid, for authenticated search

    @classmethod
    def resolve(cls, args=None, env=None) -> "Config":
        """Build a Config from parsed CLI args + environment, applying precedence."""
        env = os.environ if env is None else env
        conf = _load_toml(_config_file(env))
        out_conf = conf.get("out")

        flag_out = getattr(args, "out", None) if args else None
        out_dir = Path(flag_out or env.get("GLEAN_OUT") or out_conf or DEFAULT_OUT).expanduser()

        cache_dir = Path(
            env.get("GLEAN_CACHE")
            or (Path(env["XDG_CACHE_HOME"]) / "glean" if env.get("XDG_CACHE_HOME") else "")
            or os.path.expanduser("~/.cache/glean")
        ).expanduser()

        whisper_model_env = env.get("GLEAN_WHISPER_MODEL") or conf.get("whisper_model")
        whisper_model = Path(whisper_model_env).expanduser() if whisper_model_env else None

        return cls(
            out_dir=out_dir,
            cache_dir=cache_dir,
            backend=(getattr(args, "backend", None) if args else None) or conf.get("backend"),
            language=(getattr(args, "language", None) if args else None) or conf.get("language"),
            keep_audio=bool(getattr(args, "keep_audio", False) if args else False),
            json_output=bool(getattr(args, "json", False) if args else False),
            cobalt_url=env.get("GLEAN_COBALT_URL") or conf.get("cobalt_url"),
            parakeet_model=env.get("GLEAN_PARAKEET_MODEL") or conf.get("parakeet_model") or DEFAULT_PARAKEET_MODEL,
            whisper_model=whisper_model,
            ig_cookie=env.get("GLEAN_IG_SESSIONID") or conf.get("ig_sessionid"),
        )

    def default_whisper_model(self) -> Path:
        """Where the whisper.cpp model lives when not overridden."""
        return self.whisper_model or (self.cache_dir / "models" / "ggml-large-v3-turbo.bin")

    def job_dir(self, item: MediaItem) -> Path:
        """Per-job output directory: <out_dir>/<source>/<id>/ (created on demand)."""
        d = self.out_dir / item.source / _safe(item.id)
        d.mkdir(parents=True, exist_ok=True)
        return d


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "item"
