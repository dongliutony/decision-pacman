"""Turn labeled game states into training rows.

  python training/build_sft.py --data data/v1 --out data/sft --temperature 50
  python training/build_sft.py --data data/q1 --out data/sft_q1 --target label --smoothing 0.1
  python training/build_sft.py --data data/q1 --out data/sft_q1_peek --target label --smoothing 0.1 --encoder features-peek5s

Each row holds the chat messages Ollama would score, the option keys in
letter order, a target distribution, and the label. With --target values the
target is a softmax of the labeler's per-option values (the oracle's) divided
by the temperature. With --target label it is the record's label (for example
the move a chat-model teacher answered), with --smoothing spread evenly over
the other options. --encoder picks an encoding stored under "alt" by
gen_teacher_data.ts --also-encode instead of the record's own. Exact duplicate
states (in the chosen encoding) are dropped within each split.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import Counter

from prompt import messages


def soft_target(values: dict, keys: list[str], temperature: float) -> list[float]:
    vs = [values[k] for k in keys]
    top = max(vs)
    ws = [math.exp((v - top) / temperature) for v in vs]
    total = sum(ws)
    return [round(w / total, 6) for w in ws]


def label_target(label: str, keys: list[str], smoothing: float) -> list[float]:
    """The label's option gets 1 - smoothing; the rest is spread evenly over the other options."""
    if len(keys) == 1:
        return [1.0]
    rest = smoothing / (len(keys) - 1)
    return [round(1.0 - smoothing if k == label else rest, 6) for k in keys]


def encoding(r: dict, encoder: str | None) -> dict:
    """The record's state, instructions, and criteria under `encoder` (its own encoding if None or the same)."""
    if not encoder or encoder == r.get("encoder"):
        return {"state": r["state"], "instructions": r["instructions"], "criteria": r["criteria"]}
    alt = (r.get("alt") or {}).get(encoder)
    if alt is None:
        raise KeyError(f"record has no {encoder} encoding (generate with --also-encode {encoder})")
    return alt


def make_target(r: dict, keys: list[str], kind: str, temperature: float, smoothing: float) -> list[float] | None:
    if kind == "values":
        return soft_target(r["values"], keys, temperature) if r.get("values") else None
    if kind == "label":
        return label_target(r["label"], keys, smoothing) if r.get("label") in keys else None
    raise ValueError(f"unknown target kind {kind}")


def run_order(path: str) -> str:
    """Sort key for a run folder that ignores its leading start time, so duplicate states across runs
    resolve the same way however the runs' start times fell (gen_teacher_data names folders
    <timestamp>_<player>_<labeler>_<encoder>_s<seed>_p<pid>)."""
    name = os.path.basename(os.path.dirname(path))
    return name.split("_", 1)[1] if "_" in name else name


def load_records(split_dir: str, kind: str = "values", encoder: str | None = None,
                 temperature: float = 50.0, smoothing: float = 0.0) -> list[dict]:
    """Labeled records of a split, deduplicated on the chosen encoding, with `target` and the chosen encoding set."""
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(split_dir, "*", "states.jsonl")), key=run_order):
        for line in open(path):
            r = json.loads(line)
            e = encoding(r, encoder)
            keys = list(e["criteria"].keys())
            t = make_target(r, keys, kind, temperature, smoothing)
            if t is None:
                continue
            sig = json.dumps([e["state"], keys], sort_keys=False)
            if sig in seen:
                continue
            seen.add(sig)
            rows.append({**r, **e, "target": t})
    return rows


def build(split_dir: str, kind: str, encoder: str | None, temperature: float, smoothing: float) -> list[dict]:
    return [{
        "messages": messages(r["state"], r["instructions"], r["criteria"]),
        "keys": list(r["criteria"].keys()),
        "target": r["target"],
        "label": r["label"],
        "player": r["player"],
        "seed": r["seed"],
    } for r in load_records(split_dir, kind, encoder, temperature, smoothing)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--target", choices=["values", "label"], default="values")
    ap.add_argument("--encoder", default=None, help="use this encoding from the record's alt field")
    ap.add_argument("--temperature", type=float, default=50.0, help="--target values: value scale of the softmax")
    ap.add_argument("--smoothing", type=float, default=0.0, help="--target label: probability spread over the other options")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val"):
        rows = build(os.path.join(args.data, split), args.target, args.encoder, args.temperature, args.smoothing)
        random.Random(0).shuffle(rows)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        peaked = sum(max(r["target"]) > 0.9 for r in rows)
        tied = sum(len(r["target"]) > 1 and sorted(r["target"])[-2] > 0.4 for r in rows)
        print(f"{split}: {len(rows)} rows, players {dict(Counter(r['player'] for r in rows))}, "
              f"labels {dict(Counter(r['label'] for r in rows))}, target max>0.9: {peaked}, near-ties: {tied}")


if __name__ == "__main__":
    main()
