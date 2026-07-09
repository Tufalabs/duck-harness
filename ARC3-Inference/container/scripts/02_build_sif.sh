#!/usr/bin/env bash
# Build the Apptainer .sif from the docker archive produced by 01_pull_image.sh.
#
# Run this on any node with Apptainer available. SSH is only a convenience if
# your current node lacks Apptainer, e.g.:
#   ssh coe-hpc3 'cd ~/arc-agi-3/duck-harness/ARC3-Inference && bash container/scripts/02_build_sif.sh'
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/config.env"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[02] ERROR: apptainer not found." >&2
  echo "[02] Run this script on a node with Apptainer, or use ssh from here to such a node." >&2
  exit 1
fi

[[ -f "$IMAGE_TAR" ]] || {
  echo "[02] ERROR: $IMAGE_TAR missing; run container/scripts/01_pull_image.sh first." >&2
  exit 1
}

echo "[02] Building $SIF_PATH from $IMAGE_TAR ..."
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$VLLM_CONTAINER_STORE/.apptainer-tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$VLLM_CONTAINER_STORE/.apptainer-cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

apptainer build --force "$SIF_PATH" "docker-archive://$IMAGE_TAR"
echo "[02] Done:"
ls -lh "$SIF_PATH"
echo "[02] Quick sanity check:"
apptainer exec "$SIF_PATH" python3 -c "import vllm; print('vllm', vllm.__version__)" || true
