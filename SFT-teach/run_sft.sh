#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# SFT Teacher — chạy trên cloud GPU (RTX Pro 6000 96GB)
#
# Usage:
#   bash SFT-teach/run_sft.sh SFT-teach/gpt2_teacher.yaml
#   bash SFT-teach/run_sft.sh SFT-teach/opt_teacher.yaml
#   bash SFT-teach/run_sft.sh SFT-teach/qwen_teacher.yaml [--push REPO_ID]
#
# Tự động: setup venv → install deps → download data → prepare → SFT teacher
# Kết quả: runs/{pair_name}/sft_teacher/best/
# ===========================================================================

CONFIG="${1:?Usage: bash SFT-teach/run_sft.sh <config.yaml> [--push REPO_ID]}"
PUSH_REPO=""

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH_REPO="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

WORKDIR="${WORKDIR:-$(pwd)}"
VENV_DIR="${WORKDIR}/.venv"

# --- 1. Setup venv + deps (skip if already done) ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "=== Creating venv ==="
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

if ! python -c "import torch; import transformers" 2>/dev/null; then
    echo "=== Installing dependencies ==="
    pip install --upgrade pip
    pip install -r requirements-server.txt
    pip install -r requirements.txt
    pip install -e .
fi

# --- 2. PYTHONPATH ---
export PYTHONPATH="${WORKDIR}/src:${PYTHONPATH:-}"
export HF_HOME="${WORKDIR}/hf_cache"

# --- 3. GPU check ---
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"

# --- 4. Parse pair_name from config ---
PAIR=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('${CONFIG}'); print(c.pair_name)")
echo "=== SFT Teacher: ${PAIR} (${CONFIG}) ==="

# --- 5. Data download + prepare + SFT teacher ---
mkdir -p "${WORKDIR}/logs"
LOGFILE="${WORKDIR}/logs/sft_teacher_${PAIR}_$(date +%Y%m%d_%H%M%S).log"
echo "Log: ${LOGFILE}"

python -m mta.pipeline --config "${CONFIG}" --only data,sft_teacher 2>&1 | tee "${LOGFILE}"

# --- 6. Verify output ---
CKPT_DIR="${WORKDIR}/runs/${PAIR}/sft_teacher/best"
if [ -d "${CKPT_DIR}" ]; then
    echo ""
    echo "=== SFT Teacher complete ==="
    echo "Checkpoint: ${CKPT_DIR}"
    ls -lh "${CKPT_DIR}/"
else
    echo "ERROR: Checkpoint not found at ${CKPT_DIR}"
    exit 1
fi

# --- 7. Optional: push to HuggingFace Hub ---
if [ -n "${PUSH_REPO}" ]; then
    echo "=== Pushing to HuggingFace Hub: ${PUSH_REPO} ==="
    python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
api.upload_folder(
    folder_path='${CKPT_DIR}',
    repo_id='${PUSH_REPO}',
    token=os.environ.get('HF_TOKEN'),
)
print('Pushed to ${PUSH_REPO}')
"
fi

echo ""
echo "=== Done ==="
echo "Để dùng checkpoint này ở local, set trong config:"
echo "  teacher:"
echo "    path: \"${CKPT_DIR}\""
if [ -n "${PUSH_REPO}" ]; then
    echo "  hoặc:"
    echo "    path: \"${PUSH_REPO}\""
fi
