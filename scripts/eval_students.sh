#!/usr/bin/env bash
# Compare the two students of one dataset: a shortcut for scripts/eval_models.sh.
#
#   scripts/eval_students.sh [tag] [games]                                   # defaults: p1 10
#   BASE_URL=http://127.0.0.1:11500 SIZE=4b scripts/eval_students.sh p2      # vLLM adapter on a server
#
# Evaluates pacman-<size>-<tag>-features (features) and pacman-<size>-<tag>-ghosts
# (features-ghosts). BASE_URL defaults to the local Ollama (models from
# scripts/import_students.sh); on a GPU server use the adapter from scripts/serve_vllm.sh.
# Other settings (N, RUNS, MAX_SECONDS, VAL) pass through to eval_models.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
tag=${1:-p1}
SIZE=${SIZE:-0.8b}
export GAMES=${2:-${GAMES:-10}}
export BASE_URL=${BASE_URL:-http://localhost:11434}
export MODELS="pacman-$SIZE-$tag-features:features pacman-$SIZE-$tag-ghosts:features-ghosts"
exec scripts/eval_models.sh "$tag-$SIZE"
