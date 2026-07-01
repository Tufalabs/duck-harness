#!/bin/bash
#SBATCH --job-name=0-history-turns
#SBATCH --time=13:00:00
#SBATCH --output=/shared/runs/20260602_112245_0-history-turns/stdout.log
#SBATCH --error=/shared/runs/20260602_112245_0-history-turns/stderr.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32768M
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --export=ALL,UV_OVERRIDE=/shared/runs/20260602_112245_0-history-turns/deployment-overrides.txt

set -euo pipefail
cd /shared/runs/20260602_112245_0-history-turns

# Slurm propagates the submitter's env; a launcher in
# Jupyter / VS Code ships
# MPLBACKEND=module://matplotlib_inline.backend_inline into
# the worker, which breaks matplotlib import. Pin to Agg.
export MPLBACKEND=Agg

# Slurm propagates only the PATH the launcher had. If that
# was a non-login shell, ~/.local/bin is missing here.
export PATH="$HOME/.local/bin:$PATH"

# Bootstrap uv inside a pyxis container (R2.38) where the
# host's ~/.local/bin isn't mounted and the base image
# (e.g. nvcr.io/nvidia/pytorch) doesn't ship uv.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# R2.37: per-run venv from the bundled sources.
UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p "$(dirname "${UV_CACHE_DIR}")"
export UV_PROJECT_ENVIRONMENT="/shared/runs/20260602_112245_0-history-turns/.venv"
(
    flock 9
    cd /shared/runs/20260602_112245_0-history-turns/src/ARC3-Inference
    uv sync
    if [ -n "${UV_OVERRIDE:-}" ]; then
        if [ ! -f "${UV_OVERRIDE}" ]; then
            echo "UV_OVERRIDE points to a missing file: ${UV_OVERRIDE}" >&2
            exit 1
        fi
        uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -r "$UV_OVERRIDE"
    fi
) 9>"${UV_CACHE_DIR}.lock"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# ``-u`` for unbuffered stdio (otherwise the solver loop
# looks wedged from outside).
#
# ``exec`` is critical for R2.33 graceful stop: without it
# the shell stays as python's parent and ``scancel
# --signal=USR1 --full`` delivers SIGUSR1 to the shell
# (default action: Terminate). Shell exits, python gets
# reparented to slurmstepd and never sees the signal. With
# exec, python takes over the shell's PID so its SIGUSR1
# handler is the one slurm signals.
exec python -u run_in_worker.py
