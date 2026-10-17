"""V2-only persistence, atomic completion markers and reproducible identities."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

# Only these PURE helpers are imported. Never import/use v1 persistence or runner.
from research_experiments.common.storage import clean, canonical, fingerprint, array_hash

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
FORMAT = "dms_research_v2"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_path(path):
    """Resolve symlinks as well as '..'; even a symlinked results root is refused."""
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = REPO / raw
    allowed = RESULTS.absolute()
    if RESULTS.resolve() != allowed:
        raise ValueError("V2 results root must not be a symlink")
    resolved = raw.resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"All outputs must be inside {allowed}: {path}")
    return resolved


def input_path(path, overrides=None):
    """Explicit external inputs are read-only; no download or implicit training."""
    if path is None:
        return None
    mapped = (overrides or {}).get("model_paths", {}).get(str(path))
    if mapped:
        p = Path(mapped).expanduser()
        return (p if p.is_absolute() else REPO / p).resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        for prefix, key in [("runs/", "mnist_root"), ("checkpoints/", "checkpoint_root"),
                            ("data/mnist", "data_root")]:
            if str(p).startswith(prefix) and (overrides or {}).get(key):
                return (Path(overrides[key]).expanduser() / str(p)[len(prefix):].lstrip("/")).resolve()
        p = REPO / p
    return p.resolve()


def atomic_bytes(path, data, exclusive=False):
    p = output_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(tmp, p)  # atomic create-or-fail, no preregistration overwrite race
        else:
            os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, data, exclusive=False):
    atomic_bytes(path, (json.dumps(clean(data), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n").encode(), exclusive=exclusive)


def write_arrays(path, **arrays):
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def write_csv(path, rows, fields=None):
    import io
    rows = list(rows)
    fields = fields or sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list))
                         else clean(v) for k, v in row.items()})
    atomic_bytes(path, stream.getvalue().encode())


def source_identity():
    # Include ALL frozen imports, so editing v1/src after a lock invalidates it.
    files = {}
    for base in [ROOT, REPO / "research_experiments", REPO / "src"]:
        for directory, subdirs, names in os.walk(base):
            subdirs[:] = [d for d in subdirs if d not in {"results", "__pycache__", ".venv", "venv", "env",
                                                               "cache", "data", "datasets", "checkpoints", ".git"}
                          and not d.startswith(".venv-")]
            for name in sorted(names):
                p = Path(directory) / name
                if p.suffix in {".py", ".json", ".md", ".txt"}:
                    files[str(p.relative_to(REPO))] = file_hash(p)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(base_commit=commit, files_sha256=files, digest=fingerprint(files))


def runtime_identity():
    versions = {}
    for name in ["numpy", "scipy", "mpmath", "torch", "torchvision", "scikit-learn", "matplotlib"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


@contextmanager
def job_claim(directory):
    """One writer per job. An interrupted job is never mistaken for a finished one."""
    p = output_path(directory)
    p.mkdir(parents=True, exist_ok=True)
    lock = p / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Job is claimed: {lock}; inspect PID before --recover-stale") from exc
    with os.fdopen(fd, "w") as stream:
        json.dump(dict(pid=os.getpid(), host=platform.node(), started=time.time()), stream)
    try:
        yield p
    finally:
        lock.unlink(missing_ok=True)


def completion(directory, identity):
    p = output_path(directory)
    artifacts = {str(f.relative_to(p)): file_hash(f) for f in sorted(p.rglob("*"))
                 if f.is_file() and f.name not in {"DONE.json", "RUNNING.lock", "ERROR.json"}}
    write_json(p / "DONE.json", dict(format=FORMAT, identity=identity, artifacts=artifacts))


def completed(directory, identity=None):
    p = output_path(directory)
    marker = p / "DONE.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    if data.get("format") != FORMAT or identity is not None and data.get("identity") != identity:
        return False
    return bool(data.get("artifacts")) and all(
        (p / rel).is_file() and file_hash(p / rel) == sha for rel, sha in data["artifacts"].items())
