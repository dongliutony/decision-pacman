#!/usr/bin/env bash
# Build the oracle-labeled dataset for the two students, from scratch.
#
#   scripts/make_oracle_dataset.sh [name]        # default name: p1
#
# Writes data/<name>/{train,val}/ (raw states, every state in `features` and,
# under `alt`, `features-ghosts`, labeled by oracle-pellets-5s) and
# data/sft_<name>/{sft_features,sft_ghosts}/{train,val}.jsonl, plus
# data/sft_<name>.zip. The players are deterministic for a seed, so any machine
# rebuilds the same rows. CPU only: about 3 minutes on an Apple M4.
# Needs Node 20+ (npm ci done) and python3 (stdlib only).
set -euo pipefail
name=${1:-p1}
cd "$(dirname "$0")/.."

major=$(node --version | sed -E 's/^v([0-9]+).*/\1/')
if [ "$major" -lt 20 ]; then
  echo "Node 20 or later is needed (found $(node --version))" >&2
  exit 1
fi
[ -d node_modules ] || npm ci

raw="data/$name"
sft="data/sft_$name"
if [ -e "$raw" ] || [ -e "$sft" ]; then
  echo "$raw or $sft already exists; remove it or pick another name" >&2
  exit 1
fi

gen() { npm run --silent gen-data -- --labeler oracle-pellets-5s --encoder features --also-encode features-ghosts "$@"; }
mkdir -p "$raw/logs"
# The same mix of players and seed ranges as the original q1/v1 data. Evaluation seeds 100-199 are never used.
gen --player oracle-pellets-5s --games 20 --seed 1000 --out "$raw/train" > "$raw/logs/train_oracle.log" 2>&1 &
gen --player greedy --games 30 --seed 2000 --out "$raw/train" > "$raw/logs/train_greedy.log" 2>&1 &
gen --player random --games 30 --seed 3000 --out "$raw/train" > "$raw/logs/train_random.log" 2>&1 &
gen --player oracle-pellets-5s --games 3 --seed 900 --out "$raw/val" > "$raw/logs/val_oracle.log" 2>&1 &
gen --player greedy --games 5 --seed 920 --out "$raw/val" > "$raw/logs/val_greedy.log" 2>&1 &
gen --player random --games 5 --seed 940 --out "$raw/val" > "$raw/logs/val_random.log" 2>&1 &
fail=0
for pid in $(jobs -p); do wait "$pid" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "data generation failed; see $raw/logs/" >&2
  exit 1
fi

# Soft targets from the oracle's per-option values (temperature 50 = about 5 pellets).
python3 training/build_sft.py --data "$raw" --out "$sft/sft_features"
python3 training/build_sft.py --data "$raw" --out "$sft/sft_ghosts" --encoder features-ghosts
(cd data && zip -qr "sft_$name.zip" "sft_$name")
wc -l "$sft"/*/*.jsonl
ls -la "data/sft_$name.zip"
