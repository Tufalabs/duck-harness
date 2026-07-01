"""Command-line launcher for ``SolverRandom`` on Kaggle."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import taaf.benchmark
import taaf.deploy_kaggle
import taaf.game_api
import taaf.solver_examples
import taaf.standard_benchmarks

DEFAULT_RUN_NAME = "taaf-random-kaggle"
DEFAULT_MAX_RUNTIME_MINUTES = 10.0
DEFAULT_GAME_LIMIT = 25
DEFAULT_MAX_ACTIONS_PER_GAME: int | None = None


def make_random_kaggle_benchmark(
    *,
    job_dir: Path,
    label: str = DEFAULT_RUN_NAME,
    game_ids: Sequence[str] = (),
    game_limit: int | None = DEFAULT_GAME_LIMIT,
    seed: int = 0,
    max_actions_per_game: int | None = DEFAULT_MAX_ACTIONS_PER_GAME,
    delay_move: float = 0.0,
) -> taaf.benchmark.Benchmark:
    """Build a Kaggle-ready benchmark using TAAF's built-in random solver."""
    if game_limit is not None and game_limit <= 0:
        raise ValueError("game_limit must be positive when set.")
    # The official games are CLICK-heavy; color-targeted clicks give the
    # random smoke run real signal instead of near-always missing blobs.
    solver = taaf.solver_examples.SolverRandom(
        seed=seed,
        max_actions_per_game=max_actions_per_game,
        delay_move=delay_move,
        click_on_color=True,
    )
    normalized_game_ids = _normalize_game_ids(game_ids)
    if normalized_game_ids:
        benchmark = taaf.benchmark.Benchmark(
            label=label,
            games=[taaf.game_api.GameAPI(env_name=game_id) for game_id in normalized_game_ids],
            solver=solver,
            n_passes=1,
        )
    else:
        benchmark = taaf.standard_benchmarks.make_benchmark_kaggle_official_110(solver=solver)
        benchmark.label = label
    if game_limit is not None:
        benchmark.games = benchmark.games[:game_limit]
    benchmark.job_dir = job_dir
    return benchmark


def make_random_kaggle_target(
    *,
    run_name: str = DEFAULT_RUN_NAME,
    username: str | None = None,
    kernel_slug: str | None = None,
    kernel_title: str | None = None,
    dataset_ref: str | None = None,
    max_runtime_minutes: float = DEFAULT_MAX_RUNTIME_MINUTES,
    cpu_only: bool = True,
    public: bool = False,
    enable_internet: bool = False,
    run_as_submission: bool = False,
    dry_run: bool = False,
    accelerator: str | None = taaf.deploy_kaggle.DEFAULT_ACCELERATOR,
) -> taaf.deploy_kaggle.KaggleTarget:
    """Build the Kaggle target for a random-solver run.

    The random solver needs no extra model datasets or setup commands, so
    this target is intentionally thin. CPU-only is the default because random
    is primarily useful as a cheap Kaggle integration smoke test.
    """
    if max_runtime_minutes <= 0:
        raise ValueError("max_runtime_minutes must be positive.")
    resolved_slug = taaf.deploy_kaggle.slugify(kernel_slug or run_name or DEFAULT_RUN_NAME)
    return taaf.deploy_kaggle.KaggleTarget(
        username=username or None,
        kernel_slug=resolved_slug,
        kernel_title=kernel_title or resolved_slug,
        dataset_ref=dataset_ref or None,
        max_runtime_s=max_runtime_minutes * 60.0,
        cpu_only=cpu_only,
        public=public,
        enable_internet=enable_internet,
        run_as_submission=run_as_submission,
        dry_run=dry_run,
        accelerator=accelerator,
        dataset_version_message=f"Update {resolved_slug} random TAAF source bundle.",
    )


async def deploy_random_kaggle(args: argparse.Namespace) -> taaf.deploy_kaggle.KaggleHandle:
    """Build and deploy a random-solver Kaggle benchmark from parsed args."""
    job_dir = _resolve_job_dir(args)
    benchmark = make_random_kaggle_benchmark(
        job_dir=job_dir,
        label=str(args.run_name),
        game_ids=parse_game_ids(str(args.game or "")),
        game_limit=cast(int | None, args.game_limit),
        seed=int(args.seed),
        max_actions_per_game=cast(int | None, args.max_actions_per_game),
        delay_move=float(args.delay_move),
    )
    target = make_random_kaggle_target(
        run_name=str(args.run_name),
        username=_none_if_blank(args.username),
        kernel_slug=_none_if_blank(args.kernel_slug),
        kernel_title=_none_if_blank(args.kernel_title),
        dataset_ref=_none_if_blank(args.dataset_ref),
        max_runtime_minutes=float(args.max_runtime_minutes),
        cpu_only=bool(args.cpu_only),
        public=bool(args.public),
        enable_internet=bool(args.enable_internet),
        run_as_submission=bool(args.run_as_submission),
        dry_run=bool(args.dry_run),
        accelerator=_none_if_blank(args.accelerator),
    )
    handle = await benchmark.deploy(target)
    return cast(taaf.deploy_kaggle.KaggleHandle, handle)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch TAAF's SolverRandom on Kaggle.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--job-dir", default="")
    parser.add_argument(
        "--game", default="", help="Game id, comma-separated ids, or a JSON list. Defaults to 25 official games."
    )
    parser.add_argument("--game-limit", type=_positive_int, default=DEFAULT_GAME_LIMIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions-per-game", type=_positive_int, default=DEFAULT_MAX_ACTIONS_PER_GAME)
    parser.add_argument("--delay-move", type=_nonnegative_float, default=0.0)
    parser.add_argument("--username", default="")
    parser.add_argument("--kernel-slug", default="")
    parser.add_argument("--kernel-title", default="")
    parser.add_argument("--dataset-ref", default="")
    parser.add_argument("--max-runtime-minutes", type=_positive_float, default=DEFAULT_MAX_RUNTIME_MINUTES)
    parser.add_argument("--cpu-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--public", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-internet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-as-submission", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accelerator", default=taaf.deploy_kaggle.DEFAULT_ACCELERATOR)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def parse_game_ids(raw_value: str) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        parsed: object = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("--game JSON value must be a list.")
        return _normalize_game_ids(cast(list[object], parsed))
    return _normalize_game_ids(text.split(","))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _prepend_current_python_bin_to_path()
    handle = asyncio.run(deploy_random_kaggle(args))
    job_dir = handle.job_dir
    benchmark = _load_staged_benchmark(job_dir)

    print(f"Run directory: {job_dir.absolute()}")
    print(f"Games: {len(benchmark.games)}")
    max_actions_text = "none" if args.max_actions_per_game is None else str(args.max_actions_per_game)
    print(f"Max actions per game: {max_actions_text}")
    print(f"Kaggle kernel: {handle.kernel_id}")
    print(f"Kaggle notebook: https://www.kaggle.com/code/{handle.kernel_id}")
    print(f"Kaggle source dataset: {handle.dataset_ref}")
    if not handle.uploaded:
        print(f"Kaggle dry-run bundles: {job_dir / 'kaggle'}")
    elif args.wait:
        handle.wait()
        print(f"Kaggle output: {job_dir / 'kaggle-output'}")
    else:
        print("Kaggle deployment submitted; not waiting for completion.")


def _normalize_game_ids(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in out:
            out.append(item)
    return out


def _resolve_job_dir(args: argparse.Namespace) -> Path:
    raw_job_dir = str(args.job_dir or "").strip()
    if raw_job_dir:
        return Path(raw_job_dir)
    slug = taaf.deploy_kaggle.slugify(str(args.run_name or DEFAULT_RUN_NAME))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("/tmp") / f"{slug}-{timestamp}"


def _load_staged_benchmark(job_dir: Path) -> taaf.benchmark.Benchmark:
    bundle_benchmark = job_dir / "kaggle" / "source-dataset" / "benchmark_initial.pkl"
    if bundle_benchmark.is_file():
        import pickle

        with bundle_benchmark.open("rb") as file:
            return cast(taaf.benchmark.Benchmark, pickle.load(file))
    return taaf.benchmark.Benchmark(job_dir=job_dir)


def _prepend_current_python_bin_to_path() -> None:
    """Make console scripts installed next to the active Python discoverable."""
    script_dir = "Scripts" if os.name == "nt" else "bin"
    candidates = [
        Path(sys.prefix) / script_dir,
        Path(sys.executable).parent,
        Path(sys.executable).resolve().parent,
    ]
    python_bins = list(dict.fromkeys(str(path) for path in candidates if path.is_dir()))
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    missing_bins = [path for path in python_bins if path not in path_entries]
    if missing_bins:
        os.environ["PATH"] = os.pathsep.join([*missing_bins, *path_entries])


def _none_if_blank(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


if __name__ == "__main__":
    main()
