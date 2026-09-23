#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: bash run_all.sh CONFIG [--shutdown]}"
SHUTDOWN=false
for arg in "$@"; do
    [ "$arg" = "--shutdown" ] && SHUTDOWN=true
done

WORKDIR="${WORKDIR:-$(pwd)}"

# Source environment
if [ -f "${WORKDIR}/.env" ]; then
    set -a; source "${WORKDIR}/.env"; set +a
fi

# Activate venv
source "${WORKDIR}/.venv/bin/activate"
export PYTHONPATH="${WORKDIR}/src:${PYTHONPATH:-}"

# Create log directory
mkdir -p "${WORKDIR}/logs"
LOGFILE="${WORKDIR}/logs/run_$(date +%Y%m%d_%H%M%S).log"

echo "=== MTA full pipeline ==="
echo "Config: ${CONFIG}"
echo "Log: ${LOGFILE}"

# Run pipeline
python -m mta.pipeline --config "${CONFIG}" 2>&1 | tee "${LOGFILE}"

# Show results
PAIR=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('${CONFIG}'); print(c.get('pair_name','unknown'))")
SUMMARY="results/${PAIR}/summary.md"
if [ -f "${SUMMARY}" ]; then
    echo ""
    echo "=== Results ==="
    cat "${SUMMARY}"
fi

# Optional: push results to HF Hub
if [ -n "${HF_TOKEN:-}" ]; then
    PUSH_REPO=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('${CONFIG}'); print(c.get('results',{}).get('push_repo',''))" 2>/dev/null || echo "")
    if [ -n "${PUSH_REPO}" ]; then
        echo "Pushing results to ${PUSH_REPO}..."
        python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='results/${PAIR}', repo_id='${PUSH_REPO}', repo_type='dataset')
"
    fi
fi

if $SHUTDOWN; then
    echo "Shutting down in 60 seconds..."
    sleep 60
    sudo shutdown -h now
fi
