"""What produced a record: code version, environment, and fingerprints of the data and weights.

Standard library only, so records can be checked without MLX.

A world's fingerprint hashes the files the build writes deterministically (world, splits,
vocabulary, edit sets), so a world rebuilt from its recorded build config can be checked
against the one a run used: `python -m hemispheres.records verify-data data/world-a`.
"""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def world_files(world_dir: str | Path) -> dict[str, str]:
    """sha256 of every deterministically built file in a world directory, by relative path."""
    d = Path(world_dir)
    files = [d / n for n in ("world.json", "splits.json", "vocab.json")] + sorted((d / "edits").glob("k*.json"))
    return {str(f.relative_to(d)): sha256(f) for f in files if f.exists()}


def world_fingerprint(world_dir: str | Path) -> str:
    files = world_files(world_dir)
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def git(*args: str, repo: Path = REPO) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version(repo: Path = REPO) -> dict:
    """The commit, and every file that differs from it: modified tracked files and untracked files
    that aren't git-ignored (a new module a run imports makes the tree dirty too)."""
    modified = git("diff", "--name-only", "HEAD", repo=repo)
    untracked = git("ls-files", "--others", "--exclude-standard", repo=repo)
    if modified is None or untracked is None:
        return {"commit": git("rev-parse", "HEAD", repo=repo), "dirty": None}
    changed = sorted({*modified.splitlines(), *untracked.splitlines()})
    return {"commit": git("rev-parse", "HEAD", repo=repo), "dirty": bool(changed), "changed": changed}


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    if "mlx.core" in sys.modules:
        import mlx.core as mx
        env["mlx"] = mx.__version__
        env["device"] = str(mx.default_device())
    return env


def record(data_dirs: list[str] = ()) -> dict:
    """Provenance for a run or evaluation that read `data_dirs`."""
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": code_version(),
            "env": environment(), "data": {d: world_fingerprint(d) for d in dict.fromkeys(data_dirs)}}
