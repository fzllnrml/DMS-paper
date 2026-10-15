"""Export archived numeric summaries without claiming a fresh recomputation."""
from __future__ import annotations

import json
from pathlib import Path


def archived_tables(root):
    aggregates, per_run, provenance = [], [], []
    seen = {}
    for p in sorted(Path(root).rglob("retrospective.json")):
        if not (p.parent / "DONE.json").exists():
            continue
        report = json.loads(p.read_text())
        for item in report.get("saved_records", {}).values():
            source, obj = item["path"], item["content"]
            if isinstance(obj, list):
                for row in obj:
                    if not isinstance(row, dict):
                        continue
                    if "metric" in row and "mean" in row:
                        aggregates.append(dict(row, archive_source=source, status="archived_aggregate_not_recomputed"))
                    elif "search_seed" in row and "method" in row:
                        key = tuple(str(row.get(k)) for k in ["suite_id", "method", "model_seed", "search_seed", "population_size", "protocol"])
                        if key in seen:
                            provenance.append(dict(source=source, duplicate_of=seen[key], key=key))
                        else:
                            seen[key] = source
                            per_run.append(dict(row, archive_source=source, status="archived_per_run_not_recomputed"))
            elif isinstance(obj, dict) and source.endswith("summary_search_metrics.json"):
                for dim, methods in obj.items():
                    for method, radii in methods.items():
                        for radius, row in radii.items():
                            aggregates.append(dict(row, dimension=int(dim), method=method, radius=float(radius),
                                archive_source=source, status="archived_aggregate_not_recomputed"))
    return aggregates, per_run, provenance
