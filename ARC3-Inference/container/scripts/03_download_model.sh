#!/usr/bin/env bash
# Download the model weights into the shared HF cache used by the container.
#
# Run this on a node with internet access. If the model is gated, export HF_TOKEN
# before running this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/config.env"
mkdir -p "$HF_HOME"
export HF_HOME

# The xet backend can buffer aggressively on login nodes; plain HTTP is lower
# memory and streams directly to disk.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"

echo "[03] Downloading $MODEL_ID into $HF_HOME ..."
uvx --from huggingface_hub hf download "$MODEL_ID" \
  --max-workers "${HF_DOWNLOAD_MAX_WORKERS:-4}" \
  ${HF_TOKEN:+--token "$HF_TOKEN"}

echo "[03] Done. Cached under $HF_HOME/hub"
du -sh "$HF_HOME"/hub 2>/dev/null || true
