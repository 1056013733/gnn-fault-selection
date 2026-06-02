from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def detected_instance_counts(fi: dict[str, Any], node_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in node_names:
        entry = fi.get(name)
        if not isinstance(entry, dict):
            continue
        hit = 0
        for value in entry.values():
            if isinstance(value, dict) and bool(value.get("detected", False)):
                hit += 1
        if hit:
            counts[name] = hit
    return counts


def oracle_fault_instances(counts: dict[str, int], k: int) -> int:
    return sum(counts[n] for n in sorted(counts, key=lambda n: (counts[n], n), reverse=True)[:k])


def eval_selection(selected: list[str], counts: dict[str, int], oracle: int) -> dict[str, float | int]:
    total = sum(counts.values())
    hit = sum(counts.get(n, 0) for n in selected)
    return {
        "fault_instance_total": total,
        "fault_instance_selected": hit,
        "fault_instance_snc": hit / total if total else 0.0,
        "fault_instance_nsde": hit / oracle if oracle else 0.0,
        "fault_instance_oracle": oracle,
    }


def topk(ranked: list[str], k: int) -> list[str]:
    return list(ranked[: max(0, min(int(k), len(ranked)))])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

