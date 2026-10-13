#!/usr/bin/env bash
# Train the two students of one dataset at the same time, one per GPU.
#
#   training/train_students.sh [sft_dir] [out_dir]          # defaults: data/sft_p1 runs/students_p1_<size>
#   BASE=Qwen/Qwen3.5-4B SIZE=4b GPUS="2 3" training/train_students.sh data/sft_p2
#
# Run it inside tmux (or with nohup) so a dropped SSH session does not stop it.
# Environment:
#   BASE=Qwen/Qwen3.5-0.8B  the Hugging Face base model; SIZE (default 0.8b) names the students
#                           pacman-<size>-<tag>-features and pacman-<size>-<tag>-ghosts
#   STUDENTS="sft_features sft_ghosts"   which students to train here
#   GPUS="0 1"              one GPU per student, in the same order
#   STEPS=2000              the same number of steps for both, so both see the same number of rows
#   GRAD_CKPT=false         false, true, or unsloth (true for a 27B base, to fit activations)
#   MICRO_BATCH=            rows per pass, e.g. 8; gradients add up to one 32-row step
#   EXTRA_ARGS=""           more train.py flags, for example "--lr 2e-4"
#   EXPORT=none             none: vLLM serves the LoRA adapters (scripts/serve_vllm.sh);
#                           gguf: also merge and export Q8_0 GGUFs for Ollama, bundled in a tar
#   LLAMA_CPP_DIR           the llama.cpp clone from training/setup_cuda.sh (EXPORT=gguf only)
# GPUs of different speed change wall time, not the result: both students get the same data
# order, steps, and seed.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-Qwen/Qwen3.5-0.8B}
SIZE=${SIZE:-0.8b}
sft=${1:-data/sft_p1}
tag=$(basename "$sft"); tag=${tag#sft_}
out=${2:-runs/students_${tag}_$SIZE}
read -r -a gpus <<< "${GPUS:-0 1}"
read -r -a students <<< "${STUDENTS:-sft_features sft_ghosts}"
STEPS=${STEPS:-2000}
GRAD_CKPT=${GRAD_CKPT:-false}
MICRO_BATCH=${MICRO_BATCH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
EXPORT=${EXPORT:-none}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/llama.cpp}

[ -x training/.venv/bin/python ] || { echo "training/.venv missing: run training/setup_cuda.sh first" >&2; exit 1; }
[ "$EXPORT" != gguf ] || [ -x "$LLAMA_CPP_DIR/.venv/bin/python" ] || { echo "$LLAMA_CPP_DIR/.venv missing: run WITH_GGUF=1 training/setup_cuda.sh" >&2; exit 1; }
[ "${#gpus[@]}" -ge "${#students[@]}" ] || { echo "GPUS needs one GPU index per student (${students[*]})" >&2; exit 1; }
for name in "${students[@]}"; do
  [ -f "$sft/$name/train.jsonl" ] && [ -f "$sft/$name/val.jsonl" ] || { echo "$sft/$name/{train,val}.jsonl missing" >&2; exit 1; }
done
[ ! -e "$out" ] || { echo "$out already exists; remove it or pass another out_dir" >&2; exit 1; }
mkdir -p "$out"
source training/.venv/bin/activate
echo "base $BASE, $STEPS steps, grad checkpoint $GRAD_CKPT${MICRO_BATCH:+, micro-batch $MICRO_BATCH}, out $out" | tee "$out/settings.txt"

pids=()
for i in "${!students[@]}"; do
  name=${students[$i]}
  mb=()
  [ -z "$MICRO_BATCH" ] || mb=(--micro-batch "$MICRO_BATCH")
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${gpus[$i]} python training/train.py --base "$BASE" --data "$sft/$name" --out "$out/$name" \
    --steps "$STEPS" --grad-checkpoint "$GRAD_CKPT" ${mb[@]+"${mb[@]}"} --eval-every 200 --val-limit 2000 --log-every 20 $EXTRA_ARGS \
    > "$out/$name.log" 2>&1 &
  pids+=($!)
  echo "training $name on GPU ${gpus[$i]} (pid $!), log: $out/$name.log"
done
fail=0
for i in "${!students[@]}"; do
  if wait "${pids[$i]}"; then echo "${students[$i]}: done"; else echo "${students[$i]}: FAILED, see $out/${students[$i]}.log" >&2; fail=1; fi
done
[ "$fail" -eq 0 ] || exit 1

echo
for name in "${students[@]}"; do
  echo "== $name"
  grep -E "^(before training|eval step)" "$out/$name.log" || true
  grep -E "^step " "$out/$name.log" | tail -n 1 || true
done

# How to serve these students for scripts/eval_models.sh.
serve="scripts/serve_vllm.sh $BASE"
for name in "${students[@]}"; do serve+=" pacman-$SIZE-$tag-${name#sft_}=$out/$name/adapter"; done
echo "$serve" > "$out/serve.txt"
echo
echo "serve them with: $serve"

[ "$EXPORT" = gguf ] || exit 0
mkdir -p "$out/gguf"
for name in "${students[@]}"; do
  gguf="$out/gguf/pacman-$SIZE-$tag-${name#sft_}-q8_0.gguf"
  CUDA_VISIBLE_DEVICES=${gpus[0]} training/export_gguf.sh "$out/$name/adapter" "$gguf" "$LLAMA_CPP_DIR" > "$out/$name.export.log" 2>&1 \
    || { echo "export of $name FAILED, see $out/$name.export.log" >&2; exit 1; }
  echo "exported $gguf"
done
(cd "$out/gguf" && { sha256sum ./*.gguf 2>/dev/null || shasum -a 256 ./*.gguf; } > SHA256SUMS)
files=(gguf)
for name in "${students[@]}"; do files+=("$name.log" "$name/log.jsonl" "$name/adapter"); done
bundle="students_${tag}_${SIZE}_$(IFS=-; echo "${students[*]}" | sed 's/sft_//g').tar"
tar -C "$out" -cf "$out/$bundle" "${files[@]}"
ls -la "$out/$bundle"
