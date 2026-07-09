#!/usr/bin/env bash
# Pull the vLLM Docker image into a docker-archive tarball.
#
# Run this on a node with internet access. It does not need Apptainer.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/config.env"
mkdir -p "$VLLM_CONTAINER_STORE"

CRANE="${CRANE:-$HOME/bin/crane}"
if [[ ! -x "$CRANE" ]]; then
  echo "[01] crane not found at $CRANE; downloading static binary ..."
  CRANE_VER="${CRANE_VER:-v0.20.2}"
  mkdir -p "$HOME/bin"
  curl -sSL -o /tmp/crane.tgz \
    "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz"
  tar xzf /tmp/crane.tgz -C "$HOME/bin" crane
  chmod +x "$HOME/bin/crane"
fi

echo "[01] Pulling $VLLM_IMAGE (linux/amd64) -> $IMAGE_TAR"
echo "[01] This is large but safe on the login node; it only needs internet and disk I/O."
"$CRANE" pull --platform linux/amd64 "$VLLM_IMAGE" "$IMAGE_TAR"
echo "[01] Done:"
ls -lh "$IMAGE_TAR"
echo "[01] Next: run container/scripts/02_build_sif.sh on any node with Apptainer."
