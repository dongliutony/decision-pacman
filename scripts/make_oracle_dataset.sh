#!/usr/bin/env bash
# Build an oracle-labeled dataset at one of the sizes in docs/plan_ceiling.md, from scratch.
#
#   scripts/make_oracle_dataset.sh [p1|p2|p3|p4]        # default p1
#   JOBS=64 scripts/make_oracle_dataset.sh p4             # parallel processes (default: all cores)
#
# Writes data/<preset>/{train,val}/ (raw states, every state in `features` and, under `alt`,
# `features-ghosts`, labeled by oracle-pellets-5s with per-option values) and
# data/sft_<preset>/{sft_features,sft_ghosts}/{train,val}.jsonl, plus data/sft_<preset>.zip.
#
#   preset  scale  games (oracle / greedy / random)   training seeds
#   p1      1x     20 / 30 / 30                       1000-1019, 2000-2029, 3000-3029
#   p2      3.3x   66 / 100 / 100                     1000-1065, 2000-2099, 3000-3099
#   p3      10x    200 / 300 / 300                    1000-1199, 2000-2299, 3000-3299
#   p4      33x    660 / 1000 / 1000                  1000-1659, 2000-2999, 3000-3999
#
# The validation states are p1's for every preset (seeds 900-902, 920-924, 940-944), so all
# experiments share one validation set. Evaluation seeds 100-199 are never used. Every game
# makes its own players, so splitting games across processes (CHUNK games each) gives the same
# states as one process, and build_sft.py orders run folders by player and seed, so any machine
# builds the same rows. CPU only: p1 takes about 3 minutes on an Apple M4.
# Needs Node 20+ and python3 (stdlib only).
set -euo pipefail
preset=${1:-p1}
cd "$(dirname "$0")/.."
JOBS=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
CHUNK=${CHUNK:-10}
case "$preset" in
  p1) oracle=20 greedy=30 random=30 ;;
  p2) oracle=66 greedy=100 random=100 ;;
  p3) oracle=200 greedy=300 random=300 ;;
  p4) oracle=660 greedy=1000 random=1000 ;;
  *) echo "unknown preset $preset (p1, p2, p3, p4)" >&2; exit 1 ;;
esac

major=$(node --version | sed -E 's/^v([0-9]+).*/\1/')
[ "$major" -ge 20 ] || { echo "Node 20 or later is needed (found $(node --version))" >&2; exit 1; }
[ -d node_modules ] || npm ci

raw="data/$preset"
sft="data/sft_$preset"
if [ -e "$raw" ] || [ -e "$sft" ]; then
  echo "$raw or $sft already exists; remove it first" >&2
  exit 1
fi
mkdir -p "$raw/logs"

# One line per task: player, first seed, games, split.
tasks() {
  local player=$1 seed0=$2 n=$3 split=$4 s k
  for ((s = 0; s < n; s += CHUNK)); do
    k=$((n - s < CHUNK ? n - s : CHUNK))
    echo "$player $((seed0 + s)) $k $split"
  done
}
{
  tasks oracle-pellets-5s 1000 "$oracle" train
  tasks greedy 2000 "$greedy" train
  tasks random 3000 "$random" train
  tasks oracle-pellets-5s 900 3 val
  tasks greedy 920 5 val
  tasks random 940 5 val
} > "$raw/tasks.txt"
echo "$(wc -l < "$raw/tasks.txt") tasks on $JOBS processes"

export raw
# shellcheck disable=SC2016
if ! xargs -P "$JOBS" -L 1 bash -c '
  node_modules/.bin/tsx scripts/gen_teacher_data.ts --player "$1" --labeler oracle-pellets-5s \
    --encoder features --also-encode features-ghosts --seed "$2" --games "$3" --out "$raw/$4" \
    > "$raw/logs/$4_$1_$2.log" 2>&1 || { echo "task failed: $*" >&2; exit 255; }
' _ < "$raw/tasks.txt"; then
  echo "data generation failed; see $raw/logs/" >&2
  exit 1
fi
for split in train val; do echo "$split: $(cat "$raw/$split"/*/states.jsonl | wc -l) states"; done

# Soft targets from the oracle's per-option values (temperature 50 = about 5 pellets).
python3 training/build_sft.py --data "$raw" --out "$sft/sft_features"
python3 training/build_sft.py --data "$raw" --out "$sft/sft_ghosts" --encoder features-ghosts
(cd data && zip -qr "sft_$preset.zip" "sft_$preset")
wc -l "$sft"/*/*.jsonl
