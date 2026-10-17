"""Byte-level verification of the frozen repository; V2 is the only write root."""
import json
import os
from pathlib import Path
import subprocess

from .storage import REPO, ROOT, file_hash, write_json

BASE_COMMIT = "10454bee79847a1246662fa5c280aba8b71b6801"


def verify_git(commit=BASE_COMMIT):
    records = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", commit], cwd=REPO).split(b"\0")
    changed, count = [], 0
    for record in records:
        if not record:
            continue
        header, name = record.split(b"\t", 1)
        mode, kind, expected = header.split()
        if kind != b"blob" or name.startswith(b"research_experiments_v2/"):
            continue
        p = REPO / name.decode()
        count += 1
        if mode == b"120000":
            # Git stores the literal link target, not the checkpoint's bytes.
            # The tracked target itself is verified separately below.
            if not p.is_symlink():
                changed.append(str(p.relative_to(REPO)))
                continue
            actual = subprocess.check_output(["git", "hash-object", "--stdin"],
                                             input=os.fsencode(os.readlink(p)), cwd=REPO).strip()
            if actual != expected:
                changed.append(str(p.relative_to(REPO)))
            continue
        if not p.is_file():
            changed.append(str(p.relative_to(REPO)))
            continue
        if p.is_symlink():
            changed.append(str(p.relative_to(REPO)))
            continue
        # hash-object does not modify the object store without -w; --no-filters
        # compares literal bytes, including newline conventions.
        actual = subprocess.check_output(["git", "hash-object", "--no-filters", str(p)], cwd=REPO).strip()
        if actual != expected:
            changed.append(str(p.relative_to(REPO)))
    return dict(base_commit=commit, files_checked=count, changed=changed, byte_identical=not changed)


def snapshot(destination):
    paths = {}
    for p in REPO.rglob("*"):
        rel = p.relative_to(REPO)
        if ".git" in rel.parts or p.is_relative_to(ROOT) or not p.is_file():
            continue
        if any(part in {".venv", "venv", "__pycache__"} for part in rel.parts):
            continue
        paths[str(rel)] = file_hash(p)
    write_json(destination, dict(format="dms_v2_isolation_baseline", sha256=paths))
    return dict(files_snapshotted=len(paths), destination=str(destination))


def verify_snapshot(path):
    data = json.loads(Path(path).read_text())
    if data.get("format") != "dms_v2_isolation_baseline":
        raise ValueError("Invalid isolation baseline")
    changed = [rel for rel, sha in data["sha256"].items() if not (REPO / rel).is_file() or file_hash(REPO / rel) != sha]
    return dict(files_checked=len(data["sha256"]), changed=changed, byte_identical=not changed)
