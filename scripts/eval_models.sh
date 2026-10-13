#!/usr/bin/env bash
# Evaluate decision models the same way for every experiment (docs/plan_ceiling.md).
#
#   MODELS="pacman-0.8b-p1-features:features pacman-0.8b-p1-ghosts:features-ghosts" \
#     BASE_URL=http://127.0.0.1:11500 scripts/eval_models.sh p1-students
#
# Each entry in MODELS is <model>:<encoder>, the encoding that model was trained on. For each:
# 1. agreement with oracle-pellets-5s on the fixed validation states (training/score_ollama.py);
# 2. GAMES lockstep games on seeds 100-109 (the game waits for every answer: decisions alone);
# 3. RUNS rounds of GAMES realtime games (a slow answer costs game time). Realtime varies by up
#    to about 150 pellets between runs, hence the repeats.
# Models alternate within each round, so they face the same load. BASE_URL is any /v1/systemone
# endpoint: the vLLM adapter from scripts/serve_vllm.sh, or Ollama (http://localhost:11434).
# Results: runs/eval_<label>_<time>/results.md (and results.json), from scripts/collect_results.py.
# Environment: BASE_URL (default http://127.0.0.1:11500), VAL (default data/p1/val), N (1000),
# GAMES (10), RUNS (3), MAX_SECONDS (300), CLOCKS ("lockstep realtime").
set -euo pipefail
cd "$(dirname "$0")/.."
label=${1:?usage: MODELS="model:encoder ..." scripts/eval_models.sh <label>}
read -r -a models <<< "${MODELS:?set MODELS to \"model:encoder ...\"}"
BASE_URL=${BASE_URL:-http://127.0.0.1:11500}
VAL=${VAL:-data/p1/val}
N=${N:-1000}
GAMES=${GAMES:-10}
RUNS=${RUNS:-3}
MAX_SECONDS=${MAX_SECONDS:-300}
read -r -a clocks <<< "${CLOCKS:-lockstep realtime}"
curl -sf "$BASE_URL/api/version" >/dev/null || { echo "no /v1/systemone endpoint at $BASE_URL (scripts/serve_vllm.sh starts one)" >&2; exit 1; }
[ -d "$VAL" ] || { echo "$VAL missing: run scripts/make_oracle_dataset.sh p1" >&2; exit 1; }
out="runs/eval_${label}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
printf '%s\n' "${models[@]}" > "$out/models.txt"

for spec in "${models[@]}"; do
  model=${spec%%:*}; enc=${spec#*:}
  enc_flag=()
  [ "$enc" = features ] || enc_flag=(--encoder "$enc")
  python3 training/score_ollama.py --base-url "$BASE_URL" --model "$model" --data "$VAL" --n "$N" \
    ${enc_flag[@]+"${enc_flag[@]}"} --out "$out/agreement_$model.json"
done

for clock in "${clocks[@]}"; do
  rounds=$RUNS
  [ "$clock" = lockstep ] && rounds=1
  for run in $(seq 1 "$rounds"); do
    for spec in "${models[@]}"; do
      model=${spec%%:*}; enc=${spec#*:}
      echo "== $clock run $run: $model ($enc)"
      npm run --silent headless -- --policy "$model" --encoder "$enc" --clock "$clock" --games "$GAMES" --seed 100 \
        --max-seconds "$MAX_SECONDS" --base-url "$BASE_URL" --out "$out/games" \
        > "$out/games_${clock}_r${run}_$model.log" 2>&1 || { echo "games failed; see $out/games_${clock}_r${run}_$model.log" >&2; exit 1; }
      grep -E '^\{' "$out/games_${clock}_r${run}_$model.log" | tail -n 1
    done
  done
done

python3 scripts/collect_results.py "$out"
