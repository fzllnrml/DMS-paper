"""Atomic per-job writes; only results/ is writable by execution commands."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ALLOWED = REPO / "research_experiments" / "results"


def clean(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value):
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    p = Path(path).resolve()
    st = p.stat()
    return _file_hash(str(p), st.st_size, st.st_mtime_ns, st.st_ino)


@lru_cache(maxsize=20000)
def _file_hash(path, size, mtime, inode):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def directory_hash(path):
    root = Path(path)
    return fingerprint({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()})


def safe_path(path):
    p, allowed = Path(path).resolve(), ALLOWED.resolve()
    # Refuse a symlinked results root too.
    if ALLOWED.is_symlink() or not allowed.is_relative_to(REPO.resolve()):
        raise ValueError("results/ must be inside this checkout")
    if p != allowed and not p.is_relative_to(allowed):
        raise ValueError(f"Output must be under {ALLOWED}")
    return p


def write_json(path, value, exclusive=False):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and p.exists():
        raise FileExistsError(p)
    fd, name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False))
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            os.link(tmp, p)
        else:
            os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def write_arrays(path, **arrays):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def source_version():
    import importlib.metadata
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    try:
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    paths = []
    for source_root in [REPO / "research_experiments", REPO / "src"]:
        for directory, subdirs, names in os.walk(source_root):
            subdirs[:] = [d for d in subdirs if d not in {"results", "__pycache__", ".venv", "venv", ".git"} and not d.startswith(".venv-")]
            paths.extend(Path(directory) / name for name in names if name.endswith(".py"))
    paths.sort()
    hashes = {str(p.relative_to(REPO)): file_hash(p) for p in paths}
    dependencies = {}
    for name in ["numpy", "scipy", "torch", "torchvision", "scikit-learn", "matplotlib"]:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return dict(commit=commit, tracked_dirty=dirty, source_sha256=fingerprint(hashes), files=hashes,
                python=sys.version, platform=platform.platform(), dependencies=dependencies)


class JobStore:
    def __init__(self, root, job_id, resume=False):
        self.path = safe_path(Path(root) / job_id)
        self.done = (self.path / "DONE.json").exists()
        if self.path.exists() and not resume:
            raise FileExistsError(f"Use --resume for {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = self.path / "RUNNING"
        self._fd = None

    def __enter__(self):
        if self.done:
            return self
        # Atomic lock prevents two concurrent writers. A killed process leaves a
        # lock: CLI unlock verifies its PID before removing it.
        self._fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self._fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self.lock.unlink(missing_ok=True)

    def json(self, name, value):
        write_json(self.path / name, value)

    def arrays(self, name, **value):
        write_arrays(self.path / name, **value)

    def complete(self):
        files = {p.name: file_hash(p) for p in sorted(self.path.iterdir())
                 if p.is_file() and p.name not in {"RUNNING", "DONE.json"} and not p.name.endswith(".tmp")}
        self.json("DONE.json", dict(files=files))

    def verify(self):
        expected = json.loads((self.path / "DONE.json").read_text())["files"]
        if any(file_hash(self.path / k) != v for k, v in expected.items()):
            raise ValueError("Completed job checksum mismatch")
