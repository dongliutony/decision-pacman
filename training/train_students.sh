#!/usr/bin/env bash
# Train both students at the same time, one per GPU, then merge and export each to a Q8_0 GGUF.
#
#   training/train_students.sh [sft_dir] [out_dir]     # defaults: data/sft_p1 runs/students_p1
#
# Run it inside tmux (or with nohup) so a dropped SSH session does not stop it.
# Environment:
#   STUDENTS="sft_features sft_ghosts"   which students to train here; one each on two servers works too
#   GPUS="0 1"     one GPU per student, in the same order (on a one-GPU server: STUDENTS=sft_ghosts GPUS=0)
#   STEPS=2000     the same number of steps for both, so both see the same number of rows
#   EXTRA_ARGS=""  more train.py flags, for example "--lr 2e-4"
#   LLAMA_CPP_DIR  the llama.cpp clone from training/setup_cuda.sh (default ~/llama.cpp)
# The GPUs may differ in speed (H200 and RTX Pro 6000); that changes wall time, not the result:
# both students get the same data order, steps, and seed.
set -euo pipefail
cd "$(dirname "$0")/.."
sft=${1:-data/sft_p1}
out=${2:-runs/students_p1}
read -r -a gpus <<< "${GPUS:-0 1}"
STEPS=${STEPS:-2000}
EXTRA_ARGS=${EXTRA_ARGS:-}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/llama.cpp}
read -r -a students <<< "${STUDENTS:-sft_features sft_ghosts}"
tag=$(basename "$sft"); tag=${tag#sft_}

[ -x training/.venv/bin/python ] || { echo "training/.venv missing: run training/setup_cuda.sh first" >&2; exit 1; }
[ -x "$LLAMA_CPP_DIR/.venv/bin/python" ] || { echo "$LLAMA_CPP_DIR/.venv missing: run training/setup_cuda.sh first" >&2; exit 1; }
[ "${#gpus[@]}" -ge "${#students[@]}" ] || { echo "GPUS needs one GPU index per student (${students[*]})" >&2; exit 1; }
for name in "${students[@]}"; do
  [ -f "$sft/$name/train.jsonl" ] && [ -f "$sft/$name/val.jsonl" ] || { echo "$sft/$name/{train,val}.jsonl missing" >&2; exit 1; }
done
[ ! -e "$out" ] || { echo "$out already exists; remove it or pass another out_dir" >&2; exit 1; }
mkdir -p "$out"
source training/.venv/bin/activate

# Train. --grad-checkpoint false: these GPUs have the memory, and recomputing costs time.
pids=()
for i in "${!students[@]}"; do
  name=${students[$i]}
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${gpus[$i]} python training/train.py --data "$sft/$name" --out "$out/$name" \
    --steps "$STEPS" --grad-checkpoint false --eval-every 200 --val-limit 2000 --log-every 20 $EXTRA_ARGS \
    > "$out/$name.log" 2>&1 &
  pids+=($!)
  echo "training $name on GPU ${gpus[$i]} (pid $!), log: $out/$name.log"
done
fail=0
for i in "${!students[@]}"; do
  if wait "${pids[$i]}"; then echo "${students[$i]}: done"; else echo "${students[$i]}: FAILED, see $out/${students[$i]}.log" >&2; fail=1; fi
done
[ "$fail" -eq 0 ] || exit 1

# Merge each adapter into the base and convert it to Q8_0.
mkdir -p "$out/gguf"
for name in "${students[@]}"; do
  gguf="$out/gguf/pacman-0.8b-$tag-${name#sft_}-q8_0.gguf"
  CUDA_VISIBLE_DEVICES=${gpus[0]} training/export_gguf.sh "$out/$name/adapter" "$gguf" "$LLAMA_CPP_DIR" > "$out/$name.export.log" 2>&1 \
    || { echo "export of $name FAILED, see $out/$name.export.log" >&2; exit 1; }
  echo "exported $gguf"
done

# What to bring back to the Mac: the GGUFs, the logs, and the adapters (small).
echo
for name in "${students[@]}"; do
  echo "== $name"
  grep -E "^(before training|eval step)" "$out/$name.log" || true
  grep -E "^step " "$out/$name.log" | tail -n 1 || true
done
(cd "$out/gguf" && { sha256sum ./*.gguf 2>/dev/null || shasum -a 256 ./*.gguf; } > SHA256SUMS)
files=(gguf)
for name in "${students[@]}"; do files+=("$name.log" "$name/log.jsonl" "$name/adapter"); done
bundle="students_${tag}_$(IFS=-; echo "${students[*]}" | sed 's/sft_//g').tar"
tar -C "$out" -cf "$out/$bundle" "${files[@]}"
ls -la "$out/$bundle"
