"""`glean setup` and `glean doctor` — environment bootstrap and diagnosis.

`doctor` reports what's present without changing anything and always exits 0.
`setup` builds/fetches the Linux transcription backend (whisper.cpp + model) and
verifies the macOS one (fluidaudiocli). Both are idempotent: they skip work already
done. Only stdlib + subprocess is used here — no C-extensions, no cloud, no API key.
Heavy lifting is delegated to binaries (git, cmake, ffmpeg, whisper.cpp).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from glean.config import Config

FLUIDAUDIO_BIN = "fluidaudiocli"
WHISPER_BIN_CANDIDATES = ("whisper-cli", "whisper-cpp", "main")
WHISPER_REPO = "https://github.com/ggml-org/whisper.cpp"
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
FLUIDAUDIO_REPO = "https://github.com/FluidInference/FluidAudio"


# --- binary resolution (mirrors the backend modules, kept independent on purpose) ---

def _fluidaudio_bin(env=os.environ) -> str | None:
    """Resolve the fluidaudiocli binary the same way the parakeet backend does."""
    return shutil.which(env.get("GLEAN_FLUIDAUDIO_BIN") or FLUIDAUDIO_BIN)


def _whisper_bin(env=os.environ, cfg=None) -> str | None:
    """Resolve the whisper.cpp binary the same way the backend does: $GLEAN_WHISPER_BIN,
    then PATH, then the location `glean setup` builds it into (under cfg.cache_dir)."""
    override = env.get("GLEAN_WHISPER_BIN")
    if override:
        return shutil.which(override) or (override if Path(override).exists() else None)
    for candidate in WHISPER_BIN_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    if cfg is not None:
        built = _find_built_whisper(cfg.cache_dir / "whisper.cpp")
        if built:
            return str(built)
    return None


def _ytdlp_version() -> str | None:
    """glean shells out to the yt_dlp *module*, not the binary, so check the import
    (a `uv tool install` never links the yt-dlp entry point onto PATH)."""
    try:
        import yt_dlp
        return getattr(getattr(yt_dlp, "version", None), "__version__", "installed")
    except Exception:
        return None


# --- doctor ---

def _line(present: bool, label: str, detail: str) -> str:
    return f"  [{'ok' if present else '  '}] {label:<30} {detail}"


def run_doctor(cfg: Config, env=os.environ) -> int:
    """Print a human-readable environment diagnosis. Never mutates anything; always exits 0."""
    mac = sys.platform == "darwin"

    ffmpeg = shutil.which("ffmpeg")
    ytdlp = _ytdlp_version()
    podman = shutil.which("podman")
    fluid = _fluidaudio_bin(env)
    whisper = _whisper_bin(env, cfg)
    model = cfg.default_whisper_model()
    model_present = model.exists()

    parakeet_ready = bool(fluid) and mac
    whisper_ready = bool(whisper) and model_present
    transcription_ready = parakeet_ready or whisper_ready

    report = ["glean doctor", ""]

    report.append("Media tools")
    report.append(_line(bool(ffmpeg), "ffmpeg", ffmpeg or "not found — install ffmpeg"))
    report.append(_line(bool(ytdlp), "yt-dlp", f"{ytdlp} (python module)" if ytdlp else "not found — reinstall glean (yt_dlp ships as a dependency)"))
    report.append(_line(bool(podman), "podman (Instagram/Cobalt)", podman or "not found — optional, only needed for Instagram"))
    report.append("")

    report.append("Transcription backends")
    fluid_detail = fluid or ("not found — run `glean setup`" if mac else "not found (macOS-only backend)")
    report.append(_line(parakeet_ready, "parakeet (fluidaudiocli)", fluid_detail))
    whisper_detail = whisper or ("not found — run `glean setup`" if not mac else "not found (Linux fallback)")
    report.append(_line(bool(whisper), "whisper.cpp binary", whisper_detail))
    report.append(_line(model_present, "whisper.cpp model", f"{model} {'(present)' if model_present else '(missing — run `glean setup`)'}"))
    report.append("")

    verdict = "ready" if transcription_ready else "NOT ready — run `glean setup`"
    report.append(f"Transcription: {verdict}")
    report.append("")

    report.append("Cookies (optional — for anti-bot / private search)")
    if cfg.yt_cookies:
        report.append(f"  [ok] youtube/twitch/x   cookies file: {cfg.yt_cookies}")
    elif cfg.yt_cookies_from_browser:
        report.append(f"  [ok] youtube/twitch/x   from browser: {cfg.yt_cookies_from_browser}")
    else:
        report.append("  [  ] youtube/twitch/x   none (set GLEAN_YT_COOKIES or GLEAN_YT_COOKIES_FROM_BROWSER if YouTube blocks you)")
    report.append(f"  [{'ok' if cfg.ig_cookie else '  '}] instagram search    {'sessionid set' if cfg.ig_cookie else 'none (GLEAN_IG_SESSIONID enables ig search)'}")
    report.append("")

    report.append("Paths")
    report.append(f"  out_dir    {cfg.out_dir}")
    report.append(f"  cache_dir  {cfg.cache_dir}")
    report.append("")

    print("\n".join(report))
    return 0


# --- setup ---

def run_setup(cfg: Config, env=os.environ) -> int:
    """Bootstrap the local transcription backend for this platform. Idempotent."""
    if sys.platform == "darwin":
        return _setup_macos(cfg, env)
    return _setup_linux(cfg, env)


def _setup_macos(cfg: Config, env) -> int:
    fluid = _fluidaudio_bin(env)
    if fluid:
        print(f"fluidaudiocli present: {fluid}")
        print("macOS transcription (Parakeet on the Apple Neural Engine) is ready.")
        return 0
    print("fluidaudiocli not found — Parakeet transcription needs it.")
    print("Build FluidAudio's CLI and put it on your PATH:")
    print(f"  git clone {FLUIDAUDIO_REPO}")
    print("  cd FluidAudio && swift build -c release --product fluidaudiocli")
    print("  cp .build/release/fluidaudiocli ~/bin/     # or anywhere on PATH")
    print("Then re-run `glean doctor`. (Or point GLEAN_FLUIDAUDIO_BIN at the binary.)")
    return 1


def _setup_linux(cfg: Config, env) -> int:
    rc = 0

    whisper = _whisper_bin(env, cfg)
    if whisper:
        print(f"whisper.cpp binary present: {whisper}")
    else:
        built = _build_whisper_cpp(cfg)
        if built is None:
            rc = 1
        else:
            print(f"whisper.cpp built: {built}")
            print(f"Add it to PATH, or set GLEAN_WHISPER_BIN={built}")

    model = cfg.default_whisper_model()
    if model.exists():
        print(f"whisper.cpp model present: {model}")
    elif not _download(WHISPER_MODEL_URL, model, "ggml-large-v3-turbo"):
        rc = 1

    if rc == 0:
        print("Linux transcription (whisper.cpp) is ready.")
    return rc


def _build_whisper_cpp(cfg: Config) -> Path | None:
    """Clone + cmake-build whisper.cpp under the cache dir. Returns the built binary, or None."""
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            print(f"'{tool}' is required to build whisper.cpp but was not found.")
            print("On Alpine, install the build toolchain first:  apk add build-base cmake git")
            return None

    repo = cfg.cache_dir / "whisper.cpp"
    if repo.exists():
        print(f"Reusing existing whisper.cpp checkout at {repo}")
    else:
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning whisper.cpp into {repo} ...")
        if _run(["git", "clone", "--depth", "1", WHISPER_REPO, str(repo)]) != 0:
            print("git clone failed.")
            return None

    print("Building whisper.cpp with cmake ...")
    if _run(["cmake", "-B", "build"], cwd=repo) != 0:
        print("cmake configure failed.")
        return None
    if _run(["cmake", "--build", "build", "-j", str(os.cpu_count() or 4)], cwd=repo) != 0:
        print("cmake build failed.")
        return None

    binary = _find_built_whisper(repo)
    if binary is None:
        print("Build finished but no whisper binary was found under build/.")
        return None
    return binary


def _find_built_whisper(repo: Path) -> Path | None:
    for rel in ("build/bin/whisper-cli", "build/bin/main", "build/bin/whisper-cpp", "build/whisper-cli", "build/main"):
        candidate = repo / rel
        if candidate.exists():
            return candidate
    build = repo / "build"
    if build.exists():
        for name in ("whisper-cli", "main", "whisper-cpp"):
            for found in build.rglob(name):
                if found.is_file() and os.access(found, os.X_OK):
                    return found
    return None


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode
    except FileNotFoundError:
        print(f"command not found: {cmd[0]}")
        return 127


def _download(url: str, dest: Path, label: str) -> bool:
    """Stream a file to dest via a .part temp, then atomically rename. Returns success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {label} -> {dest}")
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  {pct:3d}%  {done >> 20} / {total >> 20} MiB", end="", file=sys.stderr)
            if total:
                print("", file=sys.stderr)
        tmp.replace(dest)
        return True
    except Exception as exc:
        print(f"download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False
