"""Read-only V1 evidence audit/export. V1 rows never enter V2 result analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from .storage import file_hash, output_path, write_json, write_csv


def audit(archive, destination):
    archive = Path(archive).expanduser().resolve()
    with zipfile.ZipFile(archive) as z:
        runs_names = [n for n in z.namelist() if n.endswith("/runs.csv")]
        if len(runs_names) != 1:
            raise ValueError("Expected exactly one original runs.csv")
        rows = list(csv.DictReader(io.StringIO(z.read(runs_names[0]).decode())))
        info_name = next(n for n in z.namelist() if n.endswith("/analysis.json"))
        old_info = json.loads(z.read(info_name))
        archived_name = next((n for n in z.namelist() if n.endswith("/archived_per_run.csv")), None)
        archived_rows = list(csv.DictReader(io.StringIO(z.read(archived_name).decode()))) if archived_name else []
        paired_names = [n for n in z.namelist() if n.endswith("/paired_contrasts.csv")]
        report = dict(origin="v1_archive_only", archive_sha256=file_hash(archive),
                      files=[dict(name=i.filename, bytes=i.file_size) for i in z.infolist()],
                      n_rows=len(rows), n_jobs=len({r["job_id"] for r in rows}), n_historical_rows=len(archived_rows),
                      crc_error=z.testzip(),
                      jobs_by_suite=dict(Counter(r["suite"] for r in {r["job_id"]: r for r in rows}.values())),
                      dimensions=dict(Counter(r["dimension"] for r in {r["job_id"]: r for r in rows}.values())),
                      paired_file_bytes=[z.getinfo(n).file_size for n in paired_names],
                      original_analysis=old_info,
                      sufficient_for=["descriptive table audit", "next-stage design and implementation"],
                      insufficient_for=["recompute matching from coordinates", "false Merge/basin reconstruction", "per-particle Split benefit", "checkpoint/schedule attribution", "per-image MNIST training-proximity audit"])
    destination = output_path(destination)
    write_json(destination / "archive_audit.json", report)
    # Exact fixed settings within E3 only. These are descriptive old deltas,
    # not a new experiment, a matched-NFE claim, or an uncertainty estimate.
    contrasts = [("independent_final_cluster", "historical_v3f2"),
                 ("single_level", "independent_final_cluster"),
                 ("independent_final_cluster", "ddim_cluster")]
    deltas = []
    for left, right in contrasts:
        by_label = {}
        for label in [left, right]:
            selected = [r for r in rows if r["suite"] == "03_baselines" and r["label"] == label]
            if len({r["setting"] for r in selected}) != 1:
                raise ValueError("V1 retrospective label contains multiple settings")
            by_label[label] = {(r["distribution_seed"], r["seed"], r["radius"]): r for r in selected}
        for key in by_label[left].keys() & by_label[right].keys():
            a, b = by_label[left][key], by_label[right][key]
            if not a["initial_hash"] or a["initial_hash"] != b["initial_hash"]:
                continue
            ca = float(a["neural_nfe"] or 0) + float(a["analytic_evaluations"] or 0)
            cb = float(b["neural_nfe"] or 0) + float(b["analytic_evaluations"] or 0)
            for metric in ["recall", "one_to_one_f1", "soft_error", "n_candidates"]:
                if a.get(metric) and b.get(metric):
                    deltas.append(dict(origin="v1_archive_only", left_job=a["job_id"], right_job=b["job_id"],
                                       left_label=left, right_label=right, distribution_seed=key[0], search_seed=key[1],
                                       radius=key[2], metric=metric, delta=float(a[metric])-float(b[metric]),
                                       left_queries=ca, right_queries=cb, equal_query_count=ca == cb,
                                       initial_hash=a["initial_hash"], interval="not estimated; one distribution, two search seeds"))
    write_csv(destination / "v1_descriptive_paired_deltas.csv", deltas,
              None if deltas else ["origin", "left_job", "right_job", "delta"])
    return report


def export_evidence(old_root, request_path, destination, dry_run=False, suites=None, tier=None):
    root = Path(old_root).expanduser().resolve()
    request = json.loads(Path(request_path).read_text())
    if request.get("format") != "dms_v1_evidence_request":
        raise ValueError("Wrong evidence request format")
    found, missing = [], []
    for job in request["jobs"]:
        if suites and job["suite"] not in suites or tier and job["tier"] != tier:
            continue
        directory = root / job["relative_directory"]
        for name in job["files"]:
            if Path(name).name != name:
                raise ValueError("Evidence file must be a basename")
            p = (directory / name).resolve()
            if not p.is_relative_to(root):
                raise ValueError("Evidence path escapes the original results root")
            if p.is_file():
                found.append(p)
            else:
                missing.append(str(p.relative_to(root)))
    found = sorted(set(found))
    report = dict(origin="read_only_v1_export", archive_sha256=request["source_archive_sha256"],
                  found_files=len(found), estimated_uncompressed_bytes=sum(p.stat().st_size for p in found),
                  missing=missing, old_results_modified=False)
    if dry_run:
        return report
    if not found:
        raise FileNotFoundError("No requested V1 evidence found; inspect the dry-run missing list")
    output = output_path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".v1-export-", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED) as z:
            hashes = {}
            for p in found:
                rel = str(p.relative_to(root))
                z.write(p, "v1_evidence/" + rel)
                hashes[rel] = file_hash(p)
            z.writestr("v1_evidence_manifest.json", json.dumps(dict(report, sha256=hashes), ensure_ascii=False, indent=2))
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return dict(report, output=str(output), sha256=file_hash(output))
