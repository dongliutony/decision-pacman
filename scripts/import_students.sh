#!/usr/bin/env bash
# Import the students exported by training/train_students.sh into the local Ollama.
#
#   scripts/import_students.sh <students.tar or gguf dir>...       # TAG=p1 SIZE=0.8b by default
#
# Takes one or more bundles from training/train_students.sh (one per server is fine) and creates
# pacman-<size>-<tag>-features and/or pacman-<size>-<tag>-ghosts, whichever GGUFs it finds. Each Modelfile takes
# its system prompt from training/prompt.py, the one the students were trained under
# (byte for byte, including the newlines around it), uses TEMPLATE {{ .Prompt }} as the
# published model does, and declares the decision capability so Ollama serves /v1/systemone.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] || { echo "usage: scripts/import_students.sh <students.tar or gguf dir>..." >&2; exit 1; }
tag=${TAG:-p1}
size=${SIZE:-0.8b}
command -v ollama >/dev/null || { echo "ollama not found" >&2; exit 1; }
curl -sf http://localhost:11434/api/version >/dev/null || { echo "Ollama is not running: brew services start ollama" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dirs=()
for src in "$@"; do
  if [ -f "$src" ]; then
    d="$work/$(basename "$src")"
    mkdir -p "$d" && tar -C "$d" -xf "$src"
    dirs+=("$d/gguf")
  else
    dirs+=("$src")
  fi
done
for d in "${dirs[@]}"; do
  if [ -f "$d/SHA256SUMS" ]; then (cd "$d" && shasum -a 256 -c SHA256SUMS); fi
done

found=0
for kind in features ghosts; do
  gguf=""
  for d in "${dirs[@]}"; do [ -f "$d/pacman-$size-$tag-$kind-q8_0.gguf" ] && gguf="$d/pacman-$size-$tag-$kind-q8_0.gguf"; done
  [ -n "$gguf" ] || { echo "no pacman-$size-$tag-$kind-q8_0.gguf in $*; skipping $kind"; continue; }
  found=$((found + 1))
  modelfile="$work/Modelfile.$kind"
  python3 - "$gguf" > "$modelfile" <<'EOF'
import os, sys
sys.path.insert(0, "training")
from prompt import SYSTEM
assert '"""' not in SYSTEM
print(f"FROM {os.path.abspath(sys.argv[1])}")
print("TEMPLATE {{ .Prompt }}")
print(f'SYSTEM """{SYSTEM}"""')
print("PARAMETER num_ctx 2050")
print("CAPABILITY decision")
EOF
  ollama create "pacman-$size-$tag-$kind" -f "$modelfile"
done
[ "$found" -gt 0 ] || { echo "no student GGUFs found" >&2; exit 1; }
ollama list | grep "pacman-$size-$tag-"
