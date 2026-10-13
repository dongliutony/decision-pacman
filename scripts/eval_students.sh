#!/usr/bin/env bash
# Compare the two students on this machine: agreement with the oracle, then games.
#
#   scripts/eval_students.sh [tag] [games]     # defaults: p1 10
#
# Needs the models from scripts/import_students.sh and the raw validation states in
# data/<tag>/val (scripts/make_oracle_dataset.sh rebuilds them). Each student reads its
# own encoding: features for pacman-0.8b-<tag>-features, features-ghosts for -ghosts.
# Games use the ladder protocol (seeds 100-109, 5-minute cap), first in lockstep (the
# game waits for every answer, so it measures decisions alone), then in realtime (a slow
# answer costs game time; the ghost student's longer prompt is slower). The two students
# alternate so they face the same machine load. On an Apple M4 a full run takes hours;
# pass a smaller game count (and N=200 MAX_SECONDS=60) for a first look.
set -euo pipefail
cd "$(dirname "$0")/.."
tag=${1:-p1}
games=${2:-10}
N=${N:-1000}                 # validation states for the agreement check
MAX_SECONDS=${MAX_SECONDS:-300}
curl -sf http://localhost:11434/api/version >/dev/null || { echo "Ollama is not running: brew services start ollama" >&2; exit 1; }
[ -d "data/$tag/val" ] || { echo "data/$tag/val missing: run scripts/make_oracle_dataset.sh $tag" >&2; exit 1; }
out="runs/eval_${tag}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
models=("pacman-0.8b-$tag-features" "pacman-0.8b-$tag-ghosts")
encoders=(features features-ghosts)

for i in 0 1; do
  enc_flag=()
  [ "${encoders[$i]}" = features ] || enc_flag=(--encoder "${encoders[$i]}")
  python3 training/score_ollama.py --model "${models[$i]}" --data "data/$tag/val" --n "$N" ${enc_flag[@]+"${enc_flag[@]}"} \
    --out "$out/agreement_${encoders[$i]}.json" | tee -a "$out/agreement.jsonl"
done

for clock in lockstep realtime; do
  for i in 0 1; do
    npm run --silent headless -- --policy "${models[$i]}" --encoder "${encoders[$i]}" --clock "$clock" \
      --games "$games" --seed 100 --max-seconds "$MAX_SECONDS" --base-url http://localhost:11434 --out "$out/games" \
      | tee "$out/${clock}_${encoders[$i]}.log" | grep -E '^(seed|\{)'
  done
done

echo
echo "== summary ($out)"
python3 - "$out" <<'EOF'
import json, sys, glob, os
out = sys.argv[1]
for line in open(os.path.join(out, "agreement.jsonl")):
    r = json.loads(line)
    print(f"agreement {r['model']:28s} acc {r['acc']:.3f}  decisive {r['acc_decisive']:.3f}  ce {r['ce']:.3f}  p50 {r['latency_p50_ms']} ms")
for path in sorted(glob.glob(os.path.join(out, "*_features*.log"))):
    last = [l for l in open(path) if l.startswith("{")][-1]
    r = json.loads(last)
    print(f"{r['clock']:9s} {r['policy']:28s} pellets {r['meanPellets']:7.1f}  seconds {r['meanSeconds']:4d}  p50 {r['meanLatencyP50']} ms")
EOF
