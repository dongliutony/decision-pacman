#!/usr/bin/env bash
# Serve trained students with vLLM behind the /v1/systemone adapter, for evaluation on a GPU server.
#
# LoRA mode (default): one vLLM process holds the base model and every student's adapter.
#   GPU=0 scripts/serve_vllm.sh Qwen/Qwen3.5-0.8B \
#     pacman-0.8b-p1-features=runs/students_p1/sft_features/adapter \
#     pacman-0.8b-p1-ghosts=runs/students_p1/sft_ghosts/adapter
#
# Merged mode, if vLLM cannot attach LoRA to this base: one vLLM process per merged model
# (training/merge.py output), one GPU each.
#   MERGED=1 GPUS="0 1" scripts/serve_vllm.sh - \
#     pacman-0.8b-p1-features=runs/students_p1/sft_features/merged \
#     pacman-0.8b-p1-ghosts=runs/students_p1/sft_ghosts/merged
#
# The adapter listens on http://127.0.0.1:$ADAPTER_PORT (default 11500); point the game, the
# scorers, and scripts/eval_models.sh there with BASE_URL. Runs in the foreground: Ctrl+C stops
# vLLM and the adapter together. Logs go to runs/serve_<time>/.
# Environment: VLLM (the vllm command, default: vllm), GPU / GPUS, VLLM_PORT (default 8000),
# ADAPTER_PORT, GPU_UTIL (default 0.85), MAX_MODEL_LEN (default 4096), VLLM_ARGS (more flags).
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -ge 2 ] || { echo "usage: scripts/serve_vllm.sh <base_model or -> <name=path>..." >&2; exit 1; }
base=$1; shift
VLLM=${VLLM:-vllm}
VLLM_PORT=${VLLM_PORT:-8000}
ADAPTER_PORT=${ADAPTER_PORT:-11500}
GPU_UTIL=${GPU_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
VLLM_ARGS=${VLLM_ARGS:-}
command -v "$VLLM" >/dev/null || { echo "$VLLM not found: install vLLM or set VLLM=/path/to/vllm" >&2; exit 1; }
logs="runs/serve_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$logs"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait_ready() {  # url log
  for _ in $(seq 1 900); do
    curl -sf "$1/v1/models" >/dev/null && return 0
    sleep 1
  done
  echo "vLLM at $1 did not come up in 15 minutes; see $2" >&2
  exit 1
}

urls=()
if [ "${MERGED:-0}" = 1 ]; then
  read -r -a gpus <<< "${GPUS:-0}"
  [ "${#gpus[@]}" -ge "$#" ] || { echo "MERGED=1 needs one GPU per model in GPUS" >&2; exit 1; }
  i=0
  for spec in "$@"; do
    name=${spec%%=*}; path=${spec#*=}
    port=$((VLLM_PORT + i))
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpus[$i]} "$VLLM" serve "$path" --served-model-name "$name" --port "$port" \
      --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" --enable-prefix-caching $VLLM_ARGS \
      > "$logs/vllm_$name.log" 2>&1 &
    pids+=($!)
    urls+=("http://127.0.0.1:$port")
    echo "vLLM: $name on GPU ${gpus[$i]}, port $port, log $logs/vllm_$name.log"
    i=$((i + 1))
  done
else
  [ "$base" != - ] || { echo "LoRA mode needs the base model as the first argument" >&2; exit 1; }
  modules=()
  for spec in "$@"; do modules+=("$spec"); done
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${GPU:-0} "$VLLM" serve "$base" --port "$VLLM_PORT" \
    --enable-lora --max-lora-rank 16 --max-loras "$#" --lora-modules "${modules[@]}" \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" --enable-prefix-caching $VLLM_ARGS \
    > "$logs/vllm.log" 2>&1 &
  pids+=($!)
  urls+=("http://127.0.0.1:$VLLM_PORT")
  echo "vLLM: $base with $# LoRA students on GPU ${GPU:-0}, port $VLLM_PORT, log $logs/vllm.log"
fi
for u in "${urls[@]}"; do wait_ready "$u" "$logs"; done
echo "vLLM ready: $(for u in "${urls[@]}"; do curl -s "$u/v1/models" | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))'; done | tr '\n' ' ')"

vllm_flags=()
for u in "${urls[@]}"; do vllm_flags+=(--vllm "$u"); done
python3 training/systemone_vllm.py "${vllm_flags[@]}" --port "$ADAPTER_PORT" > "$logs/adapter.log" 2>&1 &
pids+=($!)
for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$ADAPTER_PORT/api/version" >/dev/null && break; sleep 0.5; done
echo "adapter: http://127.0.0.1:$ADAPTER_PORT/v1/systemone (Ctrl+C to stop everything)"
wait "${pids[0]}"
