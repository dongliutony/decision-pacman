#!/usr/bin/env bash
# One-time setup on a CUDA server: the training environment and the GGUF converter.
#
#   training/setup_cuda.sh
#
# Creates training/.venv (uv) with a CUDA 12.8 PyTorch build and the packages in
# training/pyproject.toml, then clones llama.cpp at a pinned tag into
# $LLAMA_CPP_DIR (default ~/llama.cpp) with its own .venv for convert_hf_to_gguf.py,
# which training/export_gguf.sh expects. Prints each GPU's compute capability:
# H200 is 9.0 (Hopper), RTX Pro 6000 and RTX 5090 are 12.0 (Blackwell). Blackwell
# needs a cu128 (or newer) build of PyTorch, which this installs.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/llama.cpp}
LLAMA_CPP_TAG=${LLAMA_CPP_TAG:-b11298}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver on this machine" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv not found: install it with 'curl -LsSf https://astral.sh/uv/install.sh | sh'" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Training environment.
[ -d training/.venv ] || uv venv training/.venv --python 3.12
uv pip install --python training/.venv/bin/python torch --index-url "$TORCH_INDEX"
uv pip install --python training/.venv/bin/python -r training/pyproject.toml
training/.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, compute capability {major}.{minor}, bf16 {torch.cuda.is_bf16_supported()}")
    # A real kernel on every GPU: fails here, not an hour into training, if this build lacks the architecture.
    x = torch.randn(1024, 1024, device=f"cuda:{i}", dtype=torch.bfloat16)
    (x @ x).sum().item()
print("kernel check passed on every GPU")
EOF
training/.venv/bin/python -c "import unsloth, transformers, peft; print('unsloth', unsloth.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"

# GGUF converter, with its own environment as training/export_gguf.sh expects.
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  git clone --depth 1 -b "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi
[ -d "$LLAMA_CPP_DIR/.venv" ] || uv venv "$LLAMA_CPP_DIR/.venv" --python 3.12
req="$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
[ -f "$req" ] || req="$LLAMA_CPP_DIR/requirements.txt"
uv pip install --python "$LLAMA_CPP_DIR/.venv/bin/python" -r "$req" -e "$LLAMA_CPP_DIR/gguf-py"
# llama.cpp pins an older transformers; Qwen3.5 tokenizers need 5.5 or later.
uv pip install --python "$LLAMA_CPP_DIR/.venv/bin/python" "transformers>=5.5"
"$LLAMA_CPP_DIR/.venv/bin/python" -c "import gguf, transformers; print('converter ready: transformers', transformers.__version__)"
echo "setup done. LLAMA_CPP_DIR=$LLAMA_CPP_DIR"
