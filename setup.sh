#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
VENV_DIR="${WORKDIR}/.venv"
ENV_FILE="${WORKDIR}/.env"

echo "=== MTA setup ==="

# 1. Check prerequisites
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. GPU training will not work."
fi
python3 --version || { echo "Python3 not found"; exit 1; }

# 2. Create venv if needed
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# 3. Install dependencies (vLLM first for correct torch/CUDA)
echo "Installing server requirements..."
pip install --upgrade pip
pip install -r requirements-server.txt
pip install -r requirements.txt
pip install -e .

# 4. spaCy model
python -m spacy download en_core_web_sm

# 5. Environment variables
export HF_HOME="${WORKDIR}/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

cat > "${ENV_FILE}" <<EOF
HF_HOME=${WORKDIR}/hf_cache
HF_HUB_ENABLE_HF_TRANSFER=1
EOF
echo "Environment written to ${ENV_FILE}"

# 6. Download data and precompute spans
export PYTHONPATH="${WORKDIR}/src:${PYTHONPATH:-}"
SETUP_CONFIG="${1:-configs/gpt2.yaml}"
echo "Running data + spans stages (${SETUP_CONFIG})..."
python -m mta.pipeline --config "${SETUP_CONFIG}" --only data,spans

echo "=== Setup complete ==="
