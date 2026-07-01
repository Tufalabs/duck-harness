"""Export scores from TAAF benchmark artifacts.

`make eval` intentionally does not re-score runs from action histories. It
loads the saved TAAF `Benchmark` and asks the framework `GameRun` objects for
their score; this module only formats and aggregates those framework scores
into the lightweight score file used by significance checks.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import taaf.benchmark
import taaf.game

from inference.utils.run_artifacts import is_selectable_run_dir_name, run_dir_sort_key

DEFAULT_CONFIG_PATH = "configs/eval.json"
RUN_CONFIG_FILENAME = "run_config.json"
BENCHMARK_FILE_NAME = "benchmark.json"
EVALUATION_FILE_NAME = "evaluation.json"
LEGACY_EVALUATION_FILE_NAME = "eval_official.json"
SCORE_FILE_NAME = "score.json"
SCORING_VERSION = "taaf-framework-score-v1"
FINAL_SCORE_SOURCE = "taaf.final_score"
LIVE_SCORE_SOURCE = "taaf.GameRun._compute_final_score"
COMPLETED_GAME_RUN_STATES = {"gave_up", "won"}
FINALIZED_SCORING_STATES = {"gave_up", "won", "cancelled"}


@dataclass(frozen=True)
class GameTrialResult:
    game_id: str
    run_name: str
    score: float
    score_source: str
    levels_completed: float = 0.0
    total_levels: int = 0
    state: str = ""
    action_count: int = 0
    generated_tokens: int = 0


@dataclass(frozen=True)
class RunEvaluation:
    run_name: str
    score: float
    games: list[GameTrialResult]


@dataclass(frozen=True)
class GameAggregateResult:
    game_id: str
    average_score: float
    run_scores: list[tuple[str, float]]
    average_levels_completed: float = 0.0
    total_levels: int = 0
    trial_count: int = 0


@dataclass(frozen=True)
class EvaluationSummary:
    overall_score: float
    overall_score_std: float
    runs: list[RunEvaluation]
    games: list[GameAggregateResult]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a top-level JSON object.")
    return payload


def _load_eval_config(config_path: Path) -> dict[str, Any]:
    return _load_json(config_path)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    return _load_optional_json(run_dir / RUN_CONFIG_FILENAME)


def _normalize_run_names(raw_runs: Any) -> list[str]:
    if not isinstance(raw_runs, list):
        raise ValueError("`runs` must be a JSON array of run directory names.")
    return [str(item).strip() for item in raw_runs if str(item).strip()]


def _discover_run_names(runs_dir: Path) -> list[str]:
    if not runs_dir.exists():
        return []
    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir() and is_selectable_run_dir_name(path.name)],
        key=run_dir_sort_key,
        reverse=True,
    )
    return [path.name for path in run_dirs]


def _split_trial_sort_key(path: Path) -> tuple[int, int | str, str]:
    try:
        return (0, int(path.name), path.name)
    except ValueError:
        return (1, path.name, path.name)


def _split_trial_dirs(run_dir: Path) -> list[Path]:
    split_dirs = [path for name in ("passes", "seeds") for path in [(run_dir / name)] if path.exists()]
    return sorted(
        [
            path
            for split_dir in split_dirs
            for path in split_dir.iterdir()
            if path.is_dir() and (path / BENCHMARK_FILE_NAME).exists()
        ],
        key=_split_trial_sort_key,
    )


def _expand_run_dirs(run_dirs: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for run_dir in run_dirs:
        if (run_dir / "passes").exists() or (run_dir / "seeds").exists():
            expanded.extend(_split_trial_dirs(run_dir))
            continue
        split_trial_dirs = _split_trial_dirs(run_dir)
        if split_trial_dirs:
            expanded.extend(split_trial_dirs)
        elif (run_dir / BENCHMARK_FILE_NAME).exists():
            expanded.append(run_dir)
    return expanded


def _completed_trial_dirs(run_dirs: list[Path]) -> list[Path]:
    completed: list[Path] = []
    for run_dir in _expand_run_dirs(run_dirs):
        if _run_evaluations_from_benchmark(run_dir):
            completed.append(run_dir)
    return completed


def _experiment_dir_for_trial(run_dir: Path) -> Path:
    if run_dir.parent.name in {"passes", "seeds"}:
        return run_dir.parent.parent
    return run_dir


def _trial_run_name(run_dir: Path) -> str:
    if run_dir.parent.name not in {"passes", "seeds"}:
        return run_dir.name
    config = _load_run_config(run_dir)
    if config.get("pass_offset") is not None:
        return run_dir.parent.parent.name
    if run_dir.parent.name == "passes":
        return f"{run_dir.parent.parent.name}/pass-{run_dir.name}"
    raw_seed = config.get("seed")
    seed_label = str(raw_seed) if raw_seed is not None else run_dir.name
    return f"{run_dir.parent.parent.name}/seed-{seed_label}"


def _read_git_commit(run_dir: Path) -> str | None:
    candidates = [run_dir / "git_info.txt"]
    experiment_dir = _experiment_dir_for_trial(run_dir)
    if experiment_dir != run_dir:
        candidates.append(experiment_dir / "git_info.txt")
    git_info_path = next((path for path in candidates if path.exists()), None)
    if git_info_path is None:
        return None
    for line in git_info_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("commit:"):
            commit = line.removeprefix("commit:").strip()
            return commit or None
    return None


def _common_metadata_value(configs: list[dict[str, Any]], key: str) -> Any:
    values = [config.get(key) for config in configs if config.get(key) not in (None, "")]
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return values


def _nested_common_metadata_value(configs: list[dict[str, Any]], *keys: str) -> Any:
    values: list[Any] = []
    for config in configs:
        current: Any = config
        for key in keys:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current not in (None, ""):
            values.append(current)
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return values


def _known_trials(configs: list[dict[str, Any]]) -> list[int]:
    trials: list[int] = []
    for config in configs:
        raw_schedule = config.get("pass_schedule", config.get("seed_schedule"))
        raw_trials = raw_schedule if isinstance(raw_schedule, list) else [config.get("seed")]
        for raw_trial in raw_trials:
            if raw_trial is None:
                continue
            try:
                trial = int(raw_trial)
            except (TypeError, ValueError):
                continue
            if trial not in trials:
                trials.append(trial)
    return sorted(trials)


def _default_dataset_description(*, dataset: Any, include_tags: Any, game_count: int) -> str:
    normalized_tags = {str(tag).strip() for tag in include_tags or [] if str(tag).strip()}
    if (
        str(dataset or "").strip() in {"official", "eval"}
        or "official" in normalized_tags
    ) and game_count == 25:
        return "25 official ARC3 games"
    return ""


def _default_score_output_path(run_dirs: list[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0] / SCORE_FILE_NAME
    experiment_dirs = [_experiment_dir_for_trial(run_dir) for run_dir in run_dirs]
    first_experiment_dir = experiment_dirs[0] if experiment_dirs else None
    if first_experiment_dir is not None and all(path == first_experiment_dir for path in experiment_dirs):
        return first_experiment_dir / SCORE_FILE_NAME
    root = run_dirs[0].parent if run_dirs else Path("runs")
    return root / SCORE_FILE_NAME


def _score_source_label(sources: list[str]) -> str:
    unique = sorted({source for source in sources if source})
    return ", ".join(unique) if unique else LIVE_SCORE_SOURCE


def _summary_score_source(summary: EvaluationSummary) -> str:
    return _score_source_label(
        [
            game.score_source
            for run in summary.runs
            for game in run.games
        ]
    )


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_score(value: Any, *, path: Path, game_id: str, run_name: str, source: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: {run_name}/{game_id} {source} must be numeric, got {value!r}."
        ) from exc


def _framework_score(
    run: taaf.game.GameRun,
    *,
    path: Path,
    game_id: str,
    run_name: str,
) -> tuple[float, str]:
    if run.final_score is not None:
        return (
            _coerce_score(
                run.final_score,
                path=path,
                game_id=game_id,
                run_name=run_name,
                source=FINAL_SCORE_SOURCE,
            ),
            FINAL_SCORE_SOURCE,
        )
    return (
        _coerce_score(
            run._compute_final_score(),
            path=path,
            game_id=game_id,
            run_name=run_name,
            source=LIVE_SCORE_SOURCE,
        ),
        LIVE_SCORE_SOURCE,
    )


def _history_tokens(run: taaf.game.GameRun) -> int:
    return sum(_as_int(record.generated_tokens, default=0) for record in run.history)


def _history_action_count(run: taaf.game.GameRun) -> int:
    return len(run.history)


def _benchmark_game_runs(benchmark: taaf.benchmark.Benchmark, *, path: Path) -> list[taaf.game.GameRun]:
    if not benchmark.game_runs:
        raise ValueError(f"{path}: expected non-empty TAAF `game_runs`.")
    return list(benchmark.game_runs)


def _benchmark_n_passes(benchmark: taaf.benchmark.Benchmark, *, path: Path) -> int:
    n_passes = _as_int(benchmark.n_passes, default=1)
    if n_passes <= 0:
        raise ValueError(f"{path}: n_passes must be positive, got {benchmark.n_passes!r}.")
    return n_passes


def _benchmark_runs_by_pass(
    benchmark: taaf.benchmark.Benchmark,
    *,
    path: Path,
) -> list[list[taaf.game.GameRun]]:
    game_runs = _benchmark_game_runs(benchmark, path=path)
    n_passes = _benchmark_n_passes(benchmark, path=path)
    if len(game_runs) % n_passes != 0:
        raise ValueError(
            f"{path}: game_run count {len(game_runs)} is not divisible by n_passes={n_passes}."
        )
    games_per_pass = len(game_runs) // n_passes
    if games_per_pass <= 0:
        raise ValueError(f"{path}: no TAAF game runs found.")
    return [
        game_runs[pass_idx * games_per_pass : (pass_idx + 1) * games_per_pass]
        for pass_idx in range(n_passes)
    ]


def _pass_is_cleanly_completed(pass_runs: list[taaf.game.GameRun]) -> bool:
    if not pass_runs:
        return False
    return all(
        str(run.state or "") in COMPLETED_GAME_RUN_STATES
        and run.final_score is not None
        for run in pass_runs
    )


def _pass_is_finalized_for_scoring(pass_runs: list[taaf.game.GameRun]) -> bool:
    if not pass_runs:
        return False
    return all(
        str(run.state or "") in FINALIZED_SCORING_STATES
        and run.final_score is not None
        for run in pass_runs
    )


def _benchmark_has_ended(benchmark: taaf.benchmark.Benchmark) -> bool:
    return benchmark.end_time is not None


def _load_benchmark(run_dir: Path) -> tuple[Path, taaf.benchmark.Benchmark]:
    path = run_dir / BENCHMARK_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{run_dir}: missing {BENCHMARK_FILE_NAME}. "
            "make eval now reads framework scores from TAAF benchmark output."
        )
    try:
        return path, taaf.benchmark.Benchmark.from_json(path)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid TAAF benchmark JSON ({exc})") from exc


def _game_result_from_framework_run(
    raw_run: taaf.game.GameRun,
    *,
    path: Path,
    run_name: str,
) -> GameTrialResult:
    game_id = str(raw_run.game_id or "").strip()
    if not game_id:
        raise ValueError(f"{path}: TAAF game run is missing game_id.")
    score, score_source = _framework_score(raw_run, path=path, game_id=game_id, run_name=run_name)
    total_levels = _as_int(raw_run.number_of_levels, default=0)
    levels_completed = float(_as_int(raw_run.levels_completed, default=0))
    return GameTrialResult(
        game_id=game_id,
        run_name=run_name,
        score=score,
        score_source=score_source,
        levels_completed=levels_completed,
        total_levels=total_levels,
        state=str(raw_run.state or ""),
        action_count=_history_action_count(raw_run),
        generated_tokens=_history_tokens(raw_run),
    )


def _run_evaluations_from_benchmark(run_dir: Path) -> list[RunEvaluation]:
    path, benchmark = _load_benchmark(run_dir)
    runs_by_pass = _benchmark_runs_by_pass(benchmark, path=path)
    config = _load_run_config(run_dir)
    raw_pass_offset = config.get("pass_offset")
    pass_offset = int(raw_pass_offset or 0)
    uses_pass_offset = raw_pass_offset is not None
    base_run_name = _trial_run_name(run_dir)
    evaluations: list[RunEvaluation] = []

    include_finalized_cancelled = _benchmark_has_ended(benchmark)
    for pass_idx, pass_runs in enumerate(runs_by_pass):
        pass_is_evaluable = (
            _pass_is_finalized_for_scoring(pass_runs)
            if include_finalized_cancelled
            else _pass_is_cleanly_completed(pass_runs)
        )
        if not pass_is_evaluable:
            continue
        display_pass_idx = pass_offset + pass_idx
        run_name = (
            base_run_name
            if len(runs_by_pass) == 1 and not uses_pass_offset
            else f"{base_run_name}/pass-{display_pass_idx}"
        )
        seen_game_ids: set[str] = set()
        games: list[GameTrialResult] = []
        for raw_run in pass_runs:
            game = _game_result_from_framework_run(raw_run, path=path, run_name=run_name)
            if game.game_id in seen_game_ids:
                raise ValueError(f"{path}: duplicate game_id {game.game_id!r} in {run_name}.")
            seen_game_ids.add(game.game_id)
            games.append(game)
        evaluations.append(
            RunEvaluation(
                run_name=run_name,
                score=_mean([game.score for game in games]),
                games=games,
            )
        )
    return evaluations


def evaluate_runs(run_dirs: list[Path], *, environments_dir: str | None = None) -> EvaluationSummary:
    del environments_dir
    trial_dirs = _expand_run_dirs(run_dirs)
    run_results: list[RunEvaluation] = []
    for run_dir in trial_dirs:
        run_results.extend(_run_evaluations_from_benchmark(run_dir))
    if not run_results:
        raise ValueError(
            "No cleanly completed pass/trial results found. "
            "If the run is still live, wait for TAAF's next periodic benchmark.json save."
        )

    game_ids: list[str] = []
    for run in run_results:
        for game in run.games:
            if game.game_id not in game_ids:
                game_ids.append(game.game_id)
    if not game_ids:
        raise ValueError("No TAAF game scores found.")

    game_results: list[GameAggregateResult] = []
    for game_id in game_ids:
        trial_games = [
            game
            for run in run_results
            for game in run.games
            if game.game_id == game_id
        ]
        scores = [game.score for game in trial_games]
        game_results.append(
            GameAggregateResult(
                game_id=game_id,
                average_score=_mean(scores),
                run_scores=[(game.run_name, game.score) for game in trial_games],
                average_levels_completed=_mean([game.levels_completed for game in trial_games]),
                total_levels=max((game.total_levels for game in trial_games), default=0),
                trial_count=len(trial_games),
            )
        )

    run_scores = [run.score for run in run_results]
    overall_score = _mean([game.average_score for game in game_results])
    return EvaluationSummary(
        overall_score=overall_score,
        overall_score_std=_stddev(run_scores),
        runs=run_results,
        games=game_results,
    )


def _aggregate_runs_for_output(run_name: str, runs: list[RunEvaluation]) -> dict[str, Any]:
    game_ids: list[str] = []
    for run in runs:
        for game in run.games:
            if game.game_id not in game_ids:
                game_ids.append(game.game_id)

    games: list[dict[str, Any]] = []
    for game_id in game_ids:
        trial_games = [
            game
            for run in runs
            for game in run.games
            if game.game_id == game_id
        ]
        total_levels = max((game.total_levels for game in trial_games), default=0)
        levels_completed_mean = _mean([game.levels_completed for game in trial_games])
        completion_rate = (
            _mean([
                game.levels_completed / game.total_levels
                for game in trial_games
                if game.total_levels > 0
            ])
            if trial_games
            else 0.0
        )
        games.append(
            {
                "game_id": game_id,
                "score": _mean([game.score for game in trial_games]),
                "levels_completed": levels_completed_mean,
                "levels_completed_mean": levels_completed_mean,
                "total_levels": total_levels,
                "completion_rate": completion_rate,
                "trial_count": len(trial_games),
                "score_source": _score_source_label([game.score_source for game in trial_games]),
            }
        )

    return {
        "run_name": run_name,
        "score": _mean([game["score"] for game in games]) if games else _mean([run.score for run in runs]),
        "score_source": _score_source_label(
            [
                game.score_source
                for run in runs
                for game in run.games
            ]
        ),
        "games": games,
    }


def save_run_evaluations(summary: EvaluationSummary, *, run_dirs: list[Path]) -> list[Path]:
    saved_paths: list[Path] = []
    trial_dirs = _expand_run_dirs(run_dirs) or run_dirs
    for run_dir in trial_dirs:
        base_run_name = _trial_run_name(run_dir)
        relevant_runs = [
            run
            for run in summary.runs
            if run.run_name == base_run_name or run.run_name.startswith(f"{base_run_name}/pass-")
        ]
        if not relevant_runs:
            continue
        output_path = run_dir / EVALUATION_FILE_NAME
        output_path.write_text(
            json.dumps(_aggregate_runs_for_output(base_run_name, relevant_runs), indent=2),
            encoding="utf-8",
        )
        saved_paths.append(output_path)
    return saved_paths


def _hardware_metadata(configs: list[dict[str, Any]]) -> dict[str, Any]:
    raw_hardware = _common_metadata_value(configs, "hardware")
    if isinstance(raw_hardware, dict):
        return raw_hardware
    gpu = _nested_common_metadata_value(configs, "deployment", "slurm", "gpu")
    if gpu in (None, ""):
        return {}
    gpu_count = _nested_common_metadata_value(configs, "deployment", "slurm", "gpu_count")
    return {"gpu_type": str(gpu).lower(), "gpu_count": int(gpu_count or 1)}


def build_score_payload(summary: EvaluationSummary, *, run_dirs: list[Path]) -> dict[str, Any]:
    trial_dirs = _completed_trial_dirs(run_dirs)
    configs = [_load_run_config(run_dir) for run_dir in trial_dirs]
    game_ids = [game.game_id for game in summary.games]
    known_trials = _known_trials(configs)
    dataset = _common_metadata_value(configs, "dataset")
    include_tags = _common_metadata_value(configs, "include_tags")
    exclude_tags = _common_metadata_value(configs, "exclude_tags")
    dataset_description = _common_metadata_value(configs, "dataset_description")
    if dataset_description is None:
        dataset_description = _default_dataset_description(
            dataset=dataset,
            include_tags=include_tags,
            game_count=len(game_ids),
        )

    games: dict[str, dict[str, Any]] = {}
    trial_count = 0
    for game in summary.games:
        trial_scores = {run_name: score for run_name, score in game.run_scores}
        trial_count += len(trial_scores)
        games[game.game_id] = {
            "score": game.average_score,
            "trial_scores": trial_scores,
            "trial_count": len(trial_scores),
            "seed_scores": trial_scores,
            "seed_count": len(trial_scores),
        }

    git_commits = [
        commit
        for run_dir in trial_dirs
        for commit in [_read_git_commit(run_dir)]
        if commit is not None
    ]
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "version": 1,
        "score": summary.overall_score,
        "games": games,
        "metadata": {
            "created_at": created_at,
            "scoring_version": SCORING_VERSION,
            "score_source": _summary_score_source(summary),
            "dataset": dataset,
            "include_tags": include_tags or [],
            "exclude_tags": exclude_tags or [],
            "dataset_description": dataset_description,
            "game_ids": game_ids,
            "trials": known_trials,
            "trial_labels": [run.run_name for run in summary.runs],
            "trials_available": bool(summary.runs),
            "seeds": known_trials,
            "seed_labels": [run.run_name for run in summary.runs],
            "seeds_available": bool(known_trials),
            "game_count": len(game_ids),
            "trial_count": trial_count,
            "model": _common_metadata_value(configs, "model"),
            "setup_id": _common_metadata_value(configs, "model"),
            "git_commit": git_commits[0] if git_commits else None,
            "hardware": _hardware_metadata(configs),
            "runtime_budget": {
                "max_experiment_runtime_minutes": _common_metadata_value(
                    configs,
                    "max_experiment_runtime_minutes",
                ),
                "max_experiment_runtime_hours": _common_metadata_value(
                    configs,
                    "max_experiment_runtime_hours",
                ),
                "max_runtime_minutes_per_game": _common_metadata_value(
                    configs,
                    "max_runtime_minutes_per_game",
                ),
                "concurrent_jobs": _common_metadata_value(configs, "concurrent_jobs"),
            },
            "run_dirs": [str(run_dir) for run_dir in trial_dirs],
            "experiment_dirs": sorted({str(_experiment_dir_for_trial(run_dir)) for run_dir in trial_dirs}),
            "run_names": [run.run_name for run in summary.runs],
        },
    }


def save_score_file(
    summary: EvaluationSummary,
    *,
    run_dirs: list[Path],
    output_path: str | Path | None = None,
) -> Path:
    trial_dirs = _expand_run_dirs(run_dirs)
    path = Path(output_path) if output_path else _default_score_output_path(run_dirs if len(run_dirs) == 1 else trial_dirs)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_score_payload(summary, run_dirs=run_dirs)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def render_evaluation(summary: EvaluationSummary) -> str:
    lines = [
        "Evaluation Result",
        f"Score Source: {_summary_score_source(summary)}",
        "Per-Game Breakdown",
    ]

    run_game_lookup = {
        run.run_name: {game.game_id: game for game in run.games}
        for run in summary.runs
    }

    for game in summary.games:
        run_games = [
            run_game_lookup[run.run_name][game.game_id]
            for run in summary.runs
            if game.game_id in run_game_lookup.get(run.run_name, {})
        ]
        max_score = max((run_game.score for run_game in run_games), default=0.0)
        lines.append(
            f"  {game.game_id}: avg_score={game.average_score:.6f} "
            f"max_score={max_score:.6f} "
            f"avg_levels_completed={game.average_levels_completed:.2f}/{game.total_levels} "
            f"trials={game.trial_count}"
        )
        for run_game in run_games:
            lines.append(
                f"    {run_game.run_name}: score={run_game.score:.6f} "
                f"levels_completed={run_game.levels_completed:.0f}/{run_game.total_levels} "
                f"state={run_game.state or 'unknown'}"
            )
        lines.append("")

    if lines[-1] == "":
        lines.pop()

    cancelled_runs = [
        (
            run,
            sum(1 for game in run.games if game.state == "cancelled"),
            len(run.games),
        )
        for run in summary.runs
        if any(game.state == "cancelled" for game in run.games)
    ]
    if cancelled_runs:
        lines.append("")
        lines.append("Cancelled/Timed-Out Passes")
        for run, cancelled_count, game_count in cancelled_runs:
            lines.append(
                f"  {run.run_name}: {cancelled_count}/{game_count} games cancelled; "
                f"score={run.score:.6f}"
            )

    lines.append("")
    lines.append("Per-Trial Scores")
    for run in summary.runs:
        lines.append(f"  {run.run_name}: {run.score:.6f}")
    lines.append("")
    lines.append(f"Overall Score: {summary.overall_score:.4f} +/- {summary.overall_score_std:.4f}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TAAF framework-computed scores from saved benchmark runs.")
    parser.add_argument("run_dirs", nargs="*", help="Optional run directory or directories to evaluate directly.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"Path to eval JSON config (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument(
        "--score-output",
        default="",
        help="Optional path for the lightweight aggregate score JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config)
    config = _load_eval_config(config_path)

    if args.run_dirs:
        run_dirs = [Path(run_dir) for run_dir in args.run_dirs]
    else:
        runs_dir = Path(str(config.get("runs_dir") or "runs"))
        run_names = _normalize_run_names(config.get("runs"))
        if not run_names:
            run_names = _discover_run_names(runs_dir)
        if not run_names:
            raise FileNotFoundError(f"No run directories found under {runs_dir}")

        run_dirs = [runs_dir / run_name for run_name in run_names]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Configured run directory does not exist: {run_dir}")

    summary = evaluate_runs(run_dirs)
    save_run_evaluations(summary, run_dirs=run_dirs)
    score_output = args.score_output or str(config.get("score_output") or "")
    score_path = save_score_file(summary, run_dirs=run_dirs, output_path=score_output or None)
    print(render_evaluation(summary))
    print(f"\nScore file: {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
