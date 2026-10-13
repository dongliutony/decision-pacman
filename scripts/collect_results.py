"""Summarize one or more scripts/eval_models.sh runs as one table.

  python3 scripts/collect_results.py runs/eval_p1-students_20261012-120000 [more eval dirs...]

Reads each directory's agreement_<model>.json and games_<clock>_r<run>_<model>.log, writes
results.md and results.json into the first directory, and prints the table. Realtime pellets
are the mean over runs, with the lowest and highest run in brackets. Standard library only.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys


def last_summary(path: str) -> dict | None:
    lines = [line for line in open(path) if line.startswith("{")]
    return json.loads(lines[-1]) if lines else None


def collect(dirs: list[str]) -> list[dict]:
    rows: dict[str, dict] = {}
    for d in dirs:
        encoders = {}
        if os.path.exists(os.path.join(d, "models.txt")):
            for spec in open(os.path.join(d, "models.txt")).read().split():
                model, _, enc = spec.partition(":")
                encoders[model] = enc
        for path in glob.glob(os.path.join(d, "agreement_*.json")):
            rep = json.load(open(path))["report"]
            row = rows.setdefault(rep["model"], {"model": rep["model"], "encoder": encoders.get(rep["model"], rep.get("encoder", "features"))})
            row.update({"acc": rep["acc"], "acc_decisive": rep["acc_decisive"], "ce": rep["ce"], "agreement_p50_ms": rep["latency_p50_ms"]})
        for path in glob.glob(os.path.join(d, "games_*.log")):
            m = re.match(r"games_(lockstep|realtime)_r(\d+)_(.+)\.log$", os.path.basename(path))
            s = last_summary(path)
            if not m or not s:
                continue
            clock, model = m.group(1), m.group(3)
            row = rows.setdefault(model, {"model": model, "encoder": encoders.get(model, s["encoder"])})
            row.setdefault(clock, []).append({"pellets": s["meanPellets"], "seconds": s["meanSeconds"], "p50_ms": s["meanLatencyP50"], "stale": s["staleRate"]})
    return sorted(rows.values(), key=lambda r: r["model"])


def fmt(row: dict, clock: str) -> str:
    runs = row.get(clock) or []
    if not runs:
        return "-"
    p = [r["pellets"] for r in runs]
    s = f"{statistics.mean(p):.1f}"
    return s + (f" [{min(p):.0f}-{max(p):.0f}]" if len(p) > 1 else "")


def table(rows: list[dict]) -> str:
    out = ["| Model | Encoder | Agreement | Decisive | CE | Lockstep pellets | Realtime pellets | Realtime p50 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rt = r.get("realtime") or []
        p50 = f"{statistics.mean(x['p50_ms'] for x in rt):.0f} ms" if rt else "-"
        ag = lambda k, f: f.format(r[k]) if k in r else "-"  # noqa: E731
        out.append(f"| {r['model']} | {r['encoder']} | {ag('acc', '{:.3f}')} | {ag('acc_decisive', '{:.3f}')} | {ag('ce', '{:.3f}')} | "
                   f"{fmt(r, 'lockstep')} | {fmt(r, 'realtime')} | {p50} |")
    return "\n".join(out)


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    rows = collect(dirs)
    text = table(rows)
    with open(os.path.join(dirs[0], "results.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(dirs[0], "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(text)


if __name__ == "__main__":
    main()
