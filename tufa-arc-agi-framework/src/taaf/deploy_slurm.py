"""``TufaSlurmTarget`` — deploy a benchmark as a Slurm job on the Tufa
cluster (R2.23 + R2.37 + R2.38 + cluster slice of R2.31–R2.36/R2.39).

Lifecycle:

1. ``deploy()`` snapshots sources (R2.35), pickles the benchmark to
   ``job_dir/benchmark_initial.pkl``, renders ``run_in_worker.py`` and
   the sbatch script, submits via ``sbatch --parsable``, returns
   immediately with a ``TufaSlurmHandle``.
2. The sbatch script creates a per-run ``.venv`` from the bundled
   sources (R2.37), then runs ``run_in_worker.py`` (in the optional
   container, R2.38).
3. The worker derives ``soft_end_time`` from ``SLURM_JOB_END_TIME`` and
   ``max_runtime_s`` (R2.32) and awaits ``Benchmark.run``.
4. ``handle.wait()`` polls ``squeue``; once the job leaves the queue,
   loads ``benchmark.json``. ``handle.stop()`` is
   ``scancel --signal=USR1 --full`` (R2.33). ``_attach()`` rebuilds a
   handle from the ``job_id`` in ``deploy_meta.json``.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import taaf.benchmark
import taaf.deploy
import taaf.support

GPU = Literal["B200", "B300", "RTX", "CPU"]

# Hardware tier → ``--gres=gpu:<name>:<count>`` (e.g. ``gpu:b200:8``).
# ``B300``/``RTX`` follow the same naming convention forward-compat.
# ``CPU`` → no ``--gres`` and switches the rendering to
# ``--cpus-per-task``/``--mem``.
_GPU_GRES_NAMES: dict[str, str | None] = {
    "B200": "b200",
    "B300": "b300",
    "RTX": "rtx",
    "CPU": None,
}

_DEFAULT_POLL_INTERVAL_S = 30.0
_SLURM_TEARDOWN_BUFFER = timedelta(minutes=10)


@dataclass
class TufaSlurmHandle(taaf.deploy.DeploymentHandle):
    """Handle to a Slurm job. ``deploy()`` returns one of these *before*
    the job has started running.

    Fields:

    - ``job_id``: slurm job id from ``sbatch --parsable``.
    - ``poll_interval_s``: how often ``wait()`` polls ``squeue``.
    """

    # Empty default so the ABC's job_dir (no default) stays first.
    job_id: str = ""
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S
    _cached_benchmark: taaf.benchmark.Benchmark | None = field(default=None, repr=False)

    def wait(self) -> taaf.benchmark.Benchmark:
        """Block until the job leaves the queue, then load
        ``benchmark.pkl``."""
        while not self.is_done:
            time.sleep(self.poll_interval_s)
        return self._load_benchmark()

    def stop(self) -> None:
        """Request graceful shutdown via ``scancel --signal=USR1 --full``
        (R2.33). The worker's asyncio handler cancels the solver task,
        routing through ``Benchmark.run``'s teardown.

        ``--full`` matters: bare ``scancel --signal=X`` only signals
        running srun steps, not the batch script itself; the non-image
        path runs work directly in the batch shell with no srun wrapper,
        so without ``--full`` the signal is dropped entirely there.
        The worker also has to ``exec python -u`` so the shell hands
        the signal to python rather than terminating itself (see
        :func:`_render_sbatch_script`).

        Idempotent. For a hard kill bypassing teardown, use bare
        ``scancel <job_id>`` (SIGTERM, which the worker doesn't handle).
        """
        subprocess.run(
            ["scancel", "--signal=USR1", "--full", self.job_id],
            check=False,
            capture_output=True,
        )

    @property
    def is_done(self) -> bool:
        """True iff ``squeue`` no longer lists the job. We deliberately
        don't read sacct — the final state is already recoverable from
        ``benchmark.json`` + the stdout log, and sacct has reporting lag.
        """
        result = subprocess.run(
            ["squeue", "-j", self.job_id, "--noheader", "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == ""

    @classmethod
    def _attach(cls, job_dir: Path, meta: dict[str, Any]) -> TufaSlurmHandle:
        job_id = meta.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"deploy_meta.json in {job_dir} has no job_id; cannot reattach to Slurm job.")
        return cls(job_dir=job_dir, job_id=job_id)

    def _load_benchmark(self) -> taaf.benchmark.Benchmark:
        if self._cached_benchmark is not None:
            return self._cached_benchmark
        json_path = self.job_dir / "benchmark.json"
        if not json_path.is_file():
            raise RuntimeError(
                f"Slurm job {self.job_id} no longer in queue, but no benchmark.json in {self.job_dir}. "
                "The worker may have crashed before Benchmark.run's teardown wrote it; "
                "check stdout.log / stderr.log in the job dir."
            )
        self._cached_benchmark = taaf.benchmark.Benchmark.from_json(json_path)
        return self._cached_benchmark


@dataclass
class TufaSlurmTarget(taaf.deploy.DeploymentTarget):
    """Deploy as a Slurm job on the Tufa cluster (R2.23).

    Fields:

    - ``gpu``: hardware tier — ``B200`` / ``B300`` / ``RTX`` or ``CPU``
      (no GPU, useful for fast smoke tests). Mapped via the
      module-level ``_GPU_GRES_NAMES`` table.
    - ``time``: sbatch walltime, e.g. ``"04:00:00"``. Hard upper bound;
      the worker derives ``soft_end_time = end − 10min`` (R2.32).
    - ``gpu_count``: number of GPUs (must be 0 when ``gpu == "CPU"``).
    - ``max_runtime_s``: optional experiment budget; the worker stops
      at the earlier of this and the walltime-derived soft deadline.
    - ``cpus_per_gpu`` / ``mem_per_gpu_bytes`` (default 8 + 32 GiB) or
      ``total_cpus`` / ``total_mem_bytes``: pick one coordinate system.
      The per-gpu form renders ``--cpus-per-gpu`` / ``--mem-per-gpu``
      directly; totals are divided by ``gpu_count`` (ceiling) before
      rendering. Mixing them raises at deploy time. The CPU branch
      accepts per-gpu as a convenience.
    - ``image`` (R2.38): docker URI or ``.sqsh`` path. ``None`` ⇒ bare
      cluster python.
    - ``partition`` / ``nodelist``: passed through to sbatch verbatim.
    - ``extra_mounts``: extra ``host:container`` bind mounts (only used
      with ``image``). ``job_dir`` is always mounted.
    - ``extra_sbatch_flags``: free-form additional sbatch directives.
    - ``extra_source_repos``: additional local repos copied into
      ``job_dir/src`` alongside auto-discovered editables.
    - ``main_project``: name of the bundled repo the worker
      ``uv sync``s from. Required — there's exactly one right answer
      per downstream project and ``uv sync`` is the only command that
      honors ``[tool.uv.sources]`` path rewrites.
    """

    gpu: GPU = "B200"
    time: str = "04:00:00"
    gpu_count: int = field(default=1, kw_only=True)
    max_runtime_s: float | None = field(default=None, kw_only=True)
    cpus_per_gpu: int | None = field(default=None, kw_only=True)
    mem_per_gpu_bytes: int | None = field(default=None, kw_only=True)
    total_cpus: int | None = field(default=None, kw_only=True)
    total_mem_bytes: int | None = field(default=None, kw_only=True)
    image: str | None = None
    partition: str | None = None
    nodelist: str | None = None
    extra_mounts: list[tuple[str, str]] = field(default_factory=lambda: list[tuple[str, str]]())
    extra_sbatch_flags: list[str] = field(default_factory=lambda: list[str]())
    extra_source_repos: list[Path] = field(default_factory=lambda: list[Path](), kw_only=True)
    main_project: str = field(kw_only=True)

    async def deploy(self, benchmark: taaf.benchmark.Benchmark) -> TufaSlurmHandle:
        if benchmark.job_dir is None:
            raise ValueError("TufaSlurmTarget requires benchmark.job_dir to be set (R2.34).")
        job_dir = benchmark.job_dir.resolve()
        taaf.deploy.check_job_dir_unused(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        taaf.deploy.snapshot_editable_sources(job_dir / "src", extra_repos=self.extra_source_repos)
        # Persist launcher-side git overview. The bundled snapshot
        # excludes .git, so this file is the only durable record of
        # the real per-repo state at run start.
        taaf.deploy.write_git_status(job_dir)

        # Pre-run benchmark; teardown writes the post-run JSON.
        taaf.support.atomic_pickle_dump(benchmark, job_dir / "benchmark_initial.pkl")
        # Pickle the target so the worker hands it back as
        # ``runtime_environment=`` (R12.01) — solvers introspect the
        # object instead of parsing a tag.
        taaf.support.atomic_pickle_dump(self, job_dir / "deploy_target.pkl")

        # R2.31.2 — capture preamble on the launcher. The bundled
        # snapshot isn't a git repo so the worker can't produce it.
        preamble = taaf.deploy.format_preamble(benchmark)
        (job_dir / "run_in_worker.py").write_text(_render_worker_script(job_dir, preamble))
        sbatch_script = _render_sbatch_script(self, benchmark, job_dir)
        (job_dir / "sbatch_script.sh").write_text(sbatch_script)

        # ``--parsable`` makes sbatch print just ``<job_id>`` (or
        # ``<job_id>;<cluster>``); strip any trailing cluster qualifier.
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=sbatch_script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            # Surface sbatch's stderr in the message — ``check=True``
            # would hide it inside ``e.stderr`` where it usually goes
            # unread.
            raise RuntimeError(
                f"sbatch failed (rc={result.returncode}). "
                f"stderr:\n{result.stderr.strip()}\n"
                f"Rendered script saved at {job_dir / 'sbatch_script.sh'}."
            )
        job_id = result.stdout.strip().split(";")[0]
        if not job_id:
            raise RuntimeError(f"sbatch returned no job id; stdout={result.stdout!r} stderr={result.stderr!r}")

        taaf.deploy.write_deploy_meta(
            job_dir=job_dir,
            target=self,
            handle_class=TufaSlurmHandle,
            benchmark=benchmark,
            job_id=job_id,
        )

        return TufaSlurmHandle(job_dir=job_dir, job_id=job_id)


# --- Script rendering -----------------------------------------------------


def _render_worker_script(job_dir: Path, preamble: str) -> str:
    """The Python entry-point ``sbatch`` invokes.

    ``preamble`` is captured on the launcher (where the real git
    checkout lives) and embedded as a literal so the worker can print
    it before the run — the bundled source snapshot isn't a git repo.

    Soft deadline is derived from ``SLURM_JOB_END_TIME`` rather than
    deploy()'s wall clock so queue waits don't eat into it.

    The body sits inside ``if __name__ == "__main__":`` so spawned
    subprocesses (``multiprocessing.get_context('spawn')`` — vllm
    engine-core, BFS controllers) don't re-run the benchmark when
    Python's spawn machinery rebuilds the child's ``__main__`` via
    ``runpy.run_path()``.
    """
    return f'''"""TAAF worker entry — generated by TufaSlurmTarget.deploy(). Do not edit."""

import asyncio
import os
import pickle
import signal
from pathlib import Path

import taaf.benchmark  # noqa: F401
import taaf.deploy_slurm

JOB_DIR = Path({str(job_dir)!r})


async def _amain(bm, soft_end, target):
    # R2.33: route SIGUSR1 (from scancel --signal=USR1) into
    # Benchmark.run's teardown by cancelling the live solver task.
    # SIGTERM (default scancel) stays as a hard kill via the default
    # python handler.
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, bm.request_stop)
    try:
        await bm.run(soft_end_time=soft_end, runtime_environment=target)
    except asyncio.CancelledError:
        # Teardown already wrote artifacts; swallow so the worker
        # exits 0 and sacct shows COMPLETED rather than FAILED for a
        # graceful stop.
        pass


if __name__ == "__main__":
    with open(JOB_DIR / "deploy_target.pkl", "rb") as f:
        target = pickle.load(f)

    SOFT_END = taaf.deploy_slurm.worker_soft_end_time(
        slurm_end_epoch=os.environ.get("SLURM_JOB_END_TIME"),
        max_runtime_s=getattr(target, "max_runtime_s", None),
    )

    print({preamble!r})
    print(f"deploy.slurm: job_dir       = {{JOB_DIR}}")
    print(f"deploy.slurm: SLURM_JOB_ID  = {{os.environ.get('SLURM_JOB_ID', '?')}}")
    print(f"deploy.slurm: soft_end_time = {{SOFT_END}}")
    print("---")

    with open(JOB_DIR / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = JOB_DIR
    asyncio.run(_amain(bm, SOFT_END, target))

    # Bypass interpreter shutdown's threadpool-join: any stuck
    # background thread (vLLM NCCL, C-level recv, ...) would hang the
    # slurm job in RUNNING long after the run finished. Benchmark.run's
    # finally already wrote everything; os._exit lets the OS reclaim.
    os._exit(0)
'''


def worker_soft_end_time(
    *,
    slurm_end_epoch: str | None,
    max_runtime_s: float | None,
    now: datetime | None = None,
) -> datetime | None:
    now = now or datetime.now()

    experiment_deadline = (
        now + timedelta(seconds=float(max_runtime_s)) if max_runtime_s is not None and max_runtime_s > 0 else None
    )

    slurm_deadline = None
    if slurm_end_epoch:
        slurm_end = datetime.fromtimestamp(int(slurm_end_epoch))
        remaining = slurm_end - now
        buffer = min(_SLURM_TEARDOWN_BUFFER, remaining / 2)
        slurm_deadline = slurm_end - buffer

    deadlines = [deadline for deadline in (experiment_deadline, slurm_deadline) if deadline is not None]
    return min(deadlines, default=None)


def _resolve_resources(target: TufaSlurmTarget) -> tuple[int, int]:
    """Resolve target's CPU/memory fields into sbatch values, returning
    ``(cpus, mem_mib)``. GPU jobs get the per-gpu pair; CPU jobs get
    the totals. Raises ``ValueError`` on mixed / partial coordinate
    systems. Called at deploy time (not construction) so callers can
    mutate the target between ``__init__`` and ``deploy()``.
    """
    per_gpu_set = (target.cpus_per_gpu is not None) or (target.mem_per_gpu_bytes is not None)
    totals_set = (target.total_cpus is not None) or (target.total_mem_bytes is not None)
    if per_gpu_set and totals_set:
        raise ValueError(
            "TufaSlurmTarget: cannot mix per-gpu (cpus_per_gpu / mem_per_gpu_bytes) "
            "with totals (total_cpus / total_mem_bytes) — pick one coordinate system."
        )
    if per_gpu_set and (target.cpus_per_gpu is None or target.mem_per_gpu_bytes is None):
        raise ValueError(
            f"TufaSlurmTarget: cpus_per_gpu and mem_per_gpu_bytes must be set together "
            f"(got cpus_per_gpu={target.cpus_per_gpu!r}, mem_per_gpu_bytes={target.mem_per_gpu_bytes!r})."
        )
    if totals_set and (target.total_cpus is None or target.total_mem_bytes is None):
        raise ValueError(
            f"TufaSlurmTarget: total_cpus and total_mem_bytes must be set together "
            f"(got total_cpus={target.total_cpus!r}, total_mem_bytes={target.total_mem_bytes!r})."
        )

    # Normalize to a per-gpu pair. Defaults (8 CPUs + 32 GiB) match the
    # single-GPU case and scale to multi-GPU. ``None`` here means "use
    # totals instead" — handled below.
    per_gpu_cpus: int | None
    per_gpu_mem_bytes: int | None
    if per_gpu_set:
        assert target.cpus_per_gpu is not None and target.mem_per_gpu_bytes is not None
        per_gpu_cpus = target.cpus_per_gpu
        per_gpu_mem_bytes = target.mem_per_gpu_bytes
    elif not totals_set:
        per_gpu_cpus = 8
        per_gpu_mem_bytes = 32 * 1024**3
    else:
        per_gpu_cpus = None
        per_gpu_mem_bytes = None

    if target.gpu == "CPU":
        # Accept per-gpu as totals here so callers don't have to switch
        # coordinate systems just to smoke-test on CPU.
        if per_gpu_cpus is not None and per_gpu_mem_bytes is not None:
            cpus_total = per_gpu_cpus
            mem_bytes_total = per_gpu_mem_bytes
        else:
            assert target.total_cpus is not None and target.total_mem_bytes is not None
            cpus_total = target.total_cpus
            mem_bytes_total = target.total_mem_bytes
        return cpus_total, max(1, mem_bytes_total // (1024 * 1024))

    # GPU branch.
    if target.gpu_count < 1:
        raise ValueError("gpu_count must be at least 1 for GPU Slurm jobs.")
    if per_gpu_cpus is not None and per_gpu_mem_bytes is not None:
        return per_gpu_cpus, max(1, per_gpu_mem_bytes // (1024 * 1024))
    # Totals → divide by gpu_count (ceiling) so the allocation ≥ ask.
    assert target.total_cpus is not None and target.total_mem_bytes is not None
    cpus_pg = -(-target.total_cpus // target.gpu_count)
    mem_pg_mib = max(1, -(-target.total_mem_bytes // (target.gpu_count * 1024 * 1024)))
    return cpus_pg, mem_pg_mib


def _render_sbatch_script(
    target: TufaSlurmTarget,
    benchmark: taaf.benchmark.Benchmark,
    job_dir: Path,
) -> str:
    """Build the sbatch script that runs the benchmark on a worker node.

    Layout:
    1. ``#SBATCH`` directives (walltime / CPU / mem / partition / nodelist
       / gres — all from the target).
    2. A bash work block that creates the per-run ``.venv`` (R2.37),
       installs the bundled tufalabs repos editable, then invokes
       ``run_in_worker.py``.
    3. The work block is wrapped in ``srun --container-image=... bash -c
       '...'`` when ``image`` is set (R2.38), otherwise run directly on
       the allocation.
    """
    sb: list[str] = ["#!/bin/bash"]
    sb.append(f"#SBATCH --job-name={shlex.quote(benchmark.label or 'taaf')}")
    sb.append(f"#SBATCH --time={target.time}")
    sb.append(f"#SBATCH --output={job_dir}/stdout.log")
    sb.append(f"#SBATCH --error={job_dir}/stderr.log")
    # The Tufa cluster's submit plugin requires --ntasks=1 and the
    # per-gpu CPU/memory forms for GPU jobs (it rejects
    # --cpus-per-task / --mem); CPU-only jobs use the standard forms.
    sb.append("#SBATCH --ntasks=1")
    cpus_val, mem_mib = _resolve_resources(target)
    if target.gpu == "CPU":
        if target.gpu_count != 0:
            raise ValueError("gpu_count must be 0 for CPU Slurm jobs.")
        sb.append(f"#SBATCH --cpus-per-task={cpus_val}")
        sb.append(f"#SBATCH --mem={mem_mib}M")
    else:
        sb.append(f"#SBATCH --cpus-per-gpu={cpus_val}")
        sb.append(f"#SBATCH --mem-per-gpu={mem_mib}M")
    if target.partition:
        sb.append(f"#SBATCH --partition={target.partition}")
    if target.nodelist:
        sb.append(f"#SBATCH --nodelist={target.nodelist}")
    gres_name = _GPU_GRES_NAMES[target.gpu]
    if gres_name is not None:
        sb.append(f"#SBATCH --gres=gpu:{gres_name}:{target.gpu_count}")
    for flag in target.extra_sbatch_flags:
        sb.append(f"#SBATCH {flag}")

    install_block = _render_install_block(target, job_dir)

    work = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(job_dir))}",
            "",
            "# Slurm propagates the submitter's env; a launcher in",
            "# Jupyter / VS Code ships",
            "# MPLBACKEND=module://matplotlib_inline.backend_inline into",
            "# the worker, which breaks matplotlib import. Pin to Agg.",
            "export MPLBACKEND=Agg",
            "",
            "# Slurm propagates only the PATH the launcher had. If that",
            "# was a non-login shell, ~/.local/bin is missing here.",
            'export PATH="$HOME/.local/bin:$PATH"',
            "",
            "# Bootstrap uv inside a pyxis container (R2.38) where the",
            "# host's ~/.local/bin isn't mounted and the base image",
            "# (e.g. nvcr.io/nvidia/pytorch) doesn't ship uv.",
            "if ! command -v uv >/dev/null 2>&1; then",
            "    curl -LsSf https://astral.sh/uv/install.sh | sh",
            '    export PATH="$HOME/.local/bin:$PATH"',
            "fi",
            "",
            "# R2.37: per-run venv from the bundled sources.",
            *install_block,
            "",
            "# ``-u`` for unbuffered stdio (otherwise the solver loop",
            "# looks wedged from outside).",
            "#",
            "# ``exec`` is critical for R2.33 graceful stop: without it",
            "# the shell stays as python's parent and ``scancel",
            "# --signal=USR1 --full`` delivers SIGUSR1 to the shell",
            "# (default action: Terminate). Shell exits, python gets",
            "# reparented to slurmstepd and never sees the signal. With",
            "# exec, python takes over the shell's PID so its SIGUSR1",
            "# handler is the one slurm signals.",
            "exec python -u run_in_worker.py",
        ]
    )

    sb.append("")
    if target.image:
        mounts = [f"{job_dir}:{job_dir}"] + [f"{s}:{d}" for s, d in target.extra_mounts]
        sb.append(
            "srun "
            f"--container-image={shlex.quote(target.image)} "
            f"--container-mounts={','.join(shlex.quote(m) for m in mounts)} "
            f"--container-workdir={shlex.quote(str(job_dir))} "
            f"bash -c {shlex.quote(work)}"
        )
    else:
        sb.append(work)

    return "\n".join(sb) + "\n"


def _render_install_block(target: TufaSlurmTarget, job_dir: Path) -> list[str]:
    """Bash lines building the worker's per-run venv (R2.37) from the
    bundled sources under ``$JOB_DIR/src/``.

    Always ``uv sync`` from ``src/<target.main_project>``'s pyproject —
    it's the only uv command that honours ``[tool.uv.sources]`` path
    rewrites, and degenerates to a plain locked install when none are
    declared. ``UV_OVERRIDE`` lets a launcher point at a requirements
    file whose entries are then installed ``--no-deps`` afterward.

    The sync runs inside a ``flock``ed subshell on ``$UV_CACHE_DIR``:
    in no-image mode the cache lives on (often-NFS) ``$HOME/.cache/uv``
    where concurrent jobs would race; in image mode the cache is
    container-local and the lock is a no-op.

    Factored out so tests can bash-execute it against fixture sources
    rather than only string-asserting the rendered output.
    """
    return [
        'UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"',
        'mkdir -p "$(dirname "${UV_CACHE_DIR}")"',
        f'export UV_PROJECT_ENVIRONMENT="{job_dir}/.venv"',
        "(",
        "    flock 9",
        f"    cd {shlex.quote(str(job_dir / 'src' / target.main_project))}",
        "    uv sync --locked",
        '    if [ -n "${UV_OVERRIDE:-}" ]; then',
        '        if [ ! -f "${UV_OVERRIDE}" ]; then',
        '            echo "UV_OVERRIDE points to a missing file: ${UV_OVERRIDE}" >&2',
        "            exit 1",
        "        fi",
        '        uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -r "$UV_OVERRIDE"',
        "    fi",
        ') 9>"${UV_CACHE_DIR}.lock"',
        'source "$UV_PROJECT_ENVIRONMENT/bin/activate"',
    ]
