"""Compare lightweight ARC3 score files with paired per-game statistics."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import taaf.benchmark
import taaf.game
from taaf.diagnostics import _paired_permutation_test, _paired_score_test


DEFAULT_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CONFIG_PATH = "configs/significance.json"
RUNTIME_TOLERANCE_MINUTES = 1e-6
SCORE_FILE_NAME = "score.json"
LEGACY_SCORE_FILE_NAME = "score_official.json"
EVAL_FILE_NAME = "evaluation.json"
LEGACY_EVAL_FILE_NAME = "eval_official.json"


@dataclass(frozen=True)
class GameScore:
    game_id: str
    score: float
    trial_scores: dict[str, float]

    @property
    def seed_scores(self) -> dict[str, float]:
        return self.trial_scores


@dataclass(frozen=True)
class ScoreFile:
    path: Path
    score: float
    games: dict[str, GameScore]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FrameworkPValues:
    paired_t_p_value: float
    paired_t_statistic: float
    paired_t_df: float
    paired_t_games: int
    paired_t_zero_variance: bool
    paired_permutation_p_value: float
    paired_permutation_games: int
    paired_permutation_exact: bool
    paired_permutation_count: int
    paired_permutation_zero_variance: bool


@dataclass(frozen=True)
class ComparisonResult:
    baseline_score: float
    candidate_score: float
    delta: float
    paired_games: int
    trials_per_game: str
    total_trials: str
    probability_true_delta_gt_zero: float
    posterior_mean_delta: float
    posterior_90_ci: tuple[float, float]
    win_count: int
    win_rate: float
    bootstrap_90_ci: tuple[float, float]
    framework_p_values: FrameworkPValues
    passed_acceptance_threshold: bool
    accept_as_new_best: bool
    reason: str
    game_alignment: str
    compatibility_checks: tuple[str, ...]

    @property
    def seeds_per_game(self) -> str:
        return self.trials_per_game


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a top-level JSON object.")
    return payload


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return _load_json(config_path)


def _first_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_score_arg(args: argparse.Namespace, config: dict[str, Any], *, kind: str) -> str:
    raw_value = getattr(args, kind)
    if raw_value:
        return str(raw_value)
    if kind == "baseline":
        raw_value = _first_config_value(
            config,
            "baseline_score_file",
            "baseline_score",
            "baseline",
            "current_best_score_file",
            "current_best_score",
        )
    else:
        raw_value = _first_config_value(
            config,
            "candidate_score_file",
            "candidate_score",
            "candidate",
            "new_score_file",
            "new_score",
        )
    if raw_value:
        return str(raw_value)
    raise ValueError(
        f"Missing {kind} score file. Set `{kind}_score_file` in {args.config} "
        f"or pass --{kind}."
    )


def _resolve_float_setting(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    attr: str,
    default: float,
) -> float:
    raw_value = getattr(args, attr)
    if raw_value is None:
        raw_value = config.get(attr)
    if raw_value is None:
        return default
    return _as_float(raw_value, label=attr)


def _resolve_int_setting(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    attr: str,
    default: int,
    min_value: int | None = None,
) -> int:
    raw_value = getattr(args, attr)
    if raw_value is None:
        raw_value = config.get(attr)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{attr} must be an integer, got {raw_value!r}.") from exc
    if min_value is not None and value < min_value:
        label = "positive integer" if min_value == 1 else f"integer >= {min_value}"
        raise ValueError(f"{attr} must be a {label}, got {value}.")
    return value


def _resolve_score_path(path: Path) -> Path:
    if path.is_dir():
        score_path = path / SCORE_FILE_NAME
        if score_path.exists():
            return score_path
        legacy_score_path = path / LEGACY_SCORE_FILE_NAME
        if legacy_score_path.exists():
            return legacy_score_path
        eval_path = path / EVAL_FILE_NAME
        if eval_path.exists():
            return eval_path
        legacy_eval_path = path / LEGACY_EVAL_FILE_NAME
        if legacy_eval_path.exists():
            return legacy_eval_path
    return path


def _as_float(value: Any, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}.") from exc


def _load_games_from_mapping(raw_games: dict[str, Any], *, path: Path) -> dict[str, GameScore]:
    games: dict[str, GameScore] = {}
    for game_id, raw_game in raw_games.items():
        if not isinstance(raw_game, dict):
            raise ValueError(f"{path}: game {game_id!r} must be an object.")
        trial_scores_raw = raw_game.get("trial_scores") or raw_game.get("seed_scores") or {}
        if not isinstance(trial_scores_raw, dict):
            raise ValueError(f"{path}: game {game_id!r} trial_scores must be an object.")
        trial_scores = {
            str(trial): _as_float(score, label=f"{path} {game_id} trial {trial}")
            for trial, score in trial_scores_raw.items()
        }
        score = _as_float(raw_game.get("score"), label=f"{path} {game_id} score")
        games[str(game_id)] = GameScore(game_id=str(game_id), score=score, trial_scores=trial_scores)
    return games


def _load_games_from_evaluation(raw_games: list[Any], *, path: Path, run_name: str) -> dict[str, GameScore]:
    games: dict[str, GameScore] = {}
    for raw_game in raw_games:
        if not isinstance(raw_game, dict):
            continue
        game_id = str(raw_game.get("game_id") or "").strip()
        if not game_id:
            continue
        score = _as_float(raw_game.get("score", 0.0), label=f"{path} {game_id} score")
        games[game_id] = GameScore(game_id=game_id, score=score, trial_scores={run_name: score})
    return games


def load_score_file(path: str | Path) -> ScoreFile:
    resolved_path = _resolve_score_path(Path(path))
    payload = _load_json(resolved_path)
    raw_games = payload.get("games")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if isinstance(raw_games, dict):
        games = _load_games_from_mapping(raw_games, path=resolved_path)
    elif isinstance(raw_games, list):
        run_name = str(payload.get("run_name") or resolved_path.parent.name)
        games = _load_games_from_evaluation(raw_games, path=resolved_path, run_name=run_name)
    else:
        raise ValueError(f"{resolved_path}: expected `games` to be an object or list.")
    if not games:
        raise ValueError(f"{resolved_path}: no game scores found.")
    score = _as_float(payload.get("score", _mean([game.score for game in games.values()])), label=f"{resolved_path} score")
    return ScoreFile(path=resolved_path, score=score, games=games, metadata=metadata)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _probability_true_delta_gt_zero(mean_delta: float, std_delta: float, n_games: int) -> tuple[float, float]:
    if n_games <= 0:
        raise ValueError("At least one paired game is required.")
    if std_delta == 0.0:
        if mean_delta > 0.0:
            return 1.0, 0.0
        if mean_delta == 0.0:
            return 0.5, 0.0
        return 0.0, 0.0
    se_delta = std_delta / math.sqrt(n_games)
    probability = statistics.NormalDist().cdf(mean_delta / se_delta)
    return probability, se_delta


def _normal_ci(mean_delta: float, se_delta: float, *, mass: float = 0.90) -> tuple[float, float]:
    if se_delta == 0.0:
        return mean_delta, mean_delta
    alpha = 1.0 - mass
    distribution = statistics.NormalDist(mu=mean_delta, sigma=se_delta)
    return distribution.inv_cdf(alpha / 2.0), distribution.inv_cdf(1.0 - alpha / 2.0)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    deltas: list[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    mass: float = 0.90,
) -> tuple[float, float]:
    if not deltas:
        raise ValueError("At least one paired delta is required.")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer.")
    rng = random.Random(seed)
    n_games = len(deltas)
    means = [
        sum(rng.choice(deltas) for _ in range(n_games)) / n_games
        for _ in range(samples)
    ]
    means.sort()
    alpha = 1.0 - mass
    return _percentile(means, alpha / 2.0), _percentile(means, 1.0 - alpha / 2.0)


def _aligned_game_ids(
    baseline: ScoreFile,
    candidate: ScoreFile,
    *,
    allow_intersection: bool,
) -> tuple[list[str], str]:
    baseline_ids = set(baseline.games)
    candidate_ids = set(candidate.games)
    if baseline_ids == candidate_ids:
        return sorted(baseline_ids), "exact"
    missing_from_candidate = sorted(baseline_ids - candidate_ids)
    missing_from_baseline = sorted(candidate_ids - baseline_ids)
    if not allow_intersection:
        details = []
        if missing_from_candidate:
            details.append(f"candidate missing {len(missing_from_candidate)} game(s): {', '.join(missing_from_candidate)}")
        if missing_from_baseline:
            details.append(f"current_best missing {len(missing_from_baseline)} game(s): {', '.join(missing_from_baseline)}")
        raise ValueError("Score files do not contain the same game_ids; " + "; ".join(details))
    intersection = sorted(baseline_ids & candidate_ids)
    if not intersection:
        raise ValueError("Score files have no game_ids in common.")
    alignment = (
        f"intersection ({len(intersection)} paired of "
        f"current_best={len(baseline_ids)} candidate={len(candidate_ids)})"
    )
    return intersection, alignment


def _trial_counts(score_file: ScoreFile, game_ids: list[str]) -> list[int]:
    counts: list[int] = []
    for game_id in game_ids:
        trial_count = len(score_file.games[game_id].trial_scores)
        counts.append(trial_count if trial_count > 0 else 1)
    return counts


def _format_trial_count_summary(baseline_counts: list[int], candidate_counts: list[int]) -> str:
    all_counts = baseline_counts + candidate_counts
    if all_counts and len(set(all_counts)) == 1:
        return str(all_counts[0])
    baseline_range = f"{min(baseline_counts)}-{max(baseline_counts)}" if baseline_counts else "0"
    candidate_range = f"{min(candidate_counts)}-{max(candidate_counts)}" if candidate_counts else "0"
    return f"mixed (current_best {baseline_range}, candidate {candidate_range})"


def _format_trial_count(baseline_counts: list[int], candidate_counts: list[int]) -> str:
    baseline_total = sum(baseline_counts)
    candidate_total = sum(candidate_counts)
    if baseline_total == candidate_total:
        return f"{candidate_total} per setup"
    return f"current_best={baseline_total} candidate={candidate_total}"


def _coerce_float_metadata(value: Any, *, label: str, path: Path) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        raise ValueError(f"{path}: {label} must be a single value, got {value!r}.")
    return _as_float(value, label=f"{path} {label}")


def _runtime_budget(score_file: ScoreFile) -> dict[str, Any]:
    runtime_budget = score_file.metadata.get("runtime_budget")
    return runtime_budget if isinstance(runtime_budget, dict) else {}


def _per_game_runtime_minutes(score_file: ScoreFile) -> float | None:
    runtime_budget = _runtime_budget(score_file)
    raw_minutes = runtime_budget.get(
        "max_runtime_minutes_per_game",
        score_file.metadata.get("max_runtime_minutes_per_game"),
    )
    return _coerce_float_metadata(
        raw_minutes,
        label="max_runtime_minutes_per_game",
        path=score_file.path,
    )


def _runtime_concurrent_jobs(score_file: ScoreFile) -> int | None:
    runtime_budget = _runtime_budget(score_file)
    return _coerce_int_metadata(
        runtime_budget.get("concurrent_jobs", score_file.metadata.get("concurrent_jobs")),
        label="concurrent_jobs",
        path=score_file.path,
    )


def _format_minutes(value: float | None) -> str:
    if value is None:
        return "missing"
    if math.isclose(value, round(value), abs_tol=RUNTIME_TOLERANCE_MINUTES):
        return f"{round(value):.0f}"
    return f"{value:.3f}"


def _assert_runtime_budget_compatible(
    baseline: ScoreFile,
    candidate: ScoreFile,
) -> str:
    baseline_minutes = _per_game_runtime_minutes(baseline)
    candidate_minutes = _per_game_runtime_minutes(candidate)
    if baseline_minutes is None or candidate_minutes is None:
        raise ValueError(
            "Compatibility check failed: both score files must record "
            "runtime_budget.max_runtime_minutes_per_game."
        )
    if not math.isclose(
        baseline_minutes,
        candidate_minutes,
        abs_tol=RUNTIME_TOLERANCE_MINUTES,
    ):
        raise ValueError(
            "Compatibility check failed: per-game runtime limits differ "
            f"(current_best={_format_minutes(baseline_minutes)} minutes, "
            f"candidate={_format_minutes(candidate_minutes)} minutes)."
        )
    baseline_concurrency = _runtime_concurrent_jobs(baseline)
    candidate_concurrency = _runtime_concurrent_jobs(candidate)
    if baseline_concurrency is not None and candidate_concurrency is not None:
        if baseline_concurrency != candidate_concurrency:
            raise ValueError(
                "Compatibility check failed: concurrent worker counts differ "
                f"(current_best={baseline_concurrency}, candidate={candidate_concurrency})."
            )
        return (
            "runtime_budget: matched "
            f"({_format_minutes(baseline_minutes)} minutes/game; concurrent_jobs={baseline_concurrency})"
        )
    return (
        "runtime_budget: matched "
        f"({_format_minutes(baseline_minutes)} minutes/game; concurrent_jobs=not recorded)"
    )


def _coerce_int_metadata(value: Any, *, label: str, path: Path) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        raise ValueError(f"{path}: {label} must be a single value, got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {label} must be an integer, got {value!r}.") from exc


def _hardware_identity(score_file: ScoreFile) -> tuple[str, int]:
    hardware = score_file.metadata.get("hardware")
    if hardware not in (None, "") and not isinstance(hardware, dict):
        raise ValueError(
            f"{score_file.path}: metadata.hardware must be a single object, got {hardware!r}."
        )
    hardware = hardware if isinstance(hardware, dict) else {}
    raw_gpu_type = hardware.get("gpu_type", score_file.metadata.get("gpu_type"))
    raw_gpu_count = hardware.get("gpu_count", score_file.metadata.get("gpu_count"))
    gpu_type = str(raw_gpu_type or "").strip().lower()
    gpu_count = _coerce_int_metadata(
        raw_gpu_count,
        label="gpu_count",
        path=score_file.path,
    )
    if not gpu_type or gpu_count is None:
        raise ValueError(
            "Compatibility check failed: both score files must record "
            "metadata.hardware.gpu_type and metadata.hardware.gpu_count."
        )
    return gpu_type, gpu_count


def _format_hardware(identity: tuple[str, int]) -> str:
    gpu_type, gpu_count = identity
    return f"{gpu_type} x{gpu_count}"


def _assert_hardware_compatible(baseline: ScoreFile, candidate: ScoreFile) -> str:
    baseline_hardware = _hardware_identity(baseline)
    candidate_hardware = _hardware_identity(candidate)
    if baseline_hardware != candidate_hardware:
        raise ValueError(
            "Compatibility check failed: GPU profiles differ "
            f"(current_best={_format_hardware(baseline_hardware)}, "
            f"candidate={_format_hardware(candidate_hardware)})."
        )
    return f"gpu: matched ({_format_hardware(baseline_hardware)})"


def _metadata_sequence(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, list):
        return tuple(sorted(str(item).strip() for item in value if str(item).strip()))
    return (str(value).strip(),)


def _metadata_int(value: Any, *, fallback: int, label: str, path: Path) -> int:
    coerced = _coerce_int_metadata(value, label=label, path=path)
    return fallback if coerced is None else coerced


def _dataset_identity(score_file: ScoreFile) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]:
    return (
        _metadata_sequence(score_file.metadata.get("dataset")),
        _metadata_sequence(score_file.metadata.get("include_tags")),
        _metadata_sequence(score_file.metadata.get("exclude_tags")),
        _metadata_int(
            score_file.metadata.get("game_count"),
            fallback=len(score_file.games),
            label="game_count",
            path=score_file.path,
        ),
    )


def _format_sequence(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(values) + "]" if values else "[]"


def _format_dataset(identity: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]) -> str:
    datasets, include_tags, exclude_tags, game_count = identity
    dataset_label = _format_sequence(datasets)
    include_label = _format_sequence(include_tags)
    exclude_label = _format_sequence(exclude_tags)
    return (
        f"dataset={dataset_label}, include_tags={include_label}, "
        f"exclude_tags={exclude_label}, game_count={game_count}"
    )


def _assert_dataset_compatible(baseline: ScoreFile, candidate: ScoreFile) -> str:
    baseline_dataset = _dataset_identity(baseline)
    candidate_dataset = _dataset_identity(candidate)
    if baseline_dataset != candidate_dataset:
        raise ValueError(
            "Compatibility check failed: dataset metadata differs "
            f"(current_best={_format_dataset(baseline_dataset)}, "
            f"candidate={_format_dataset(candidate_dataset)})."
        )
    return f"dataset: matched ({_format_dataset(baseline_dataset)})"


def _assert_trial_counts_compatible(
    baseline_counts: list[int],
    candidate_counts: list[int],
) -> str:
    if baseline_counts != candidate_counts:
        raise ValueError(
            "Compatibility check failed: trial counts differ by paired game "
            f"(current_best={_format_trial_count_summary(baseline_counts, baseline_counts)}, "
            f"candidate={_format_trial_count_summary(candidate_counts, candidate_counts)})."
        )
    if baseline_counts and len(set(baseline_counts)) == 1:
        trial_label = f"{baseline_counts[0]} per game"
    else:
        trial_label = _format_trial_count_summary(baseline_counts, candidate_counts)
    return f"trials: matched ({trial_label}; {sum(baseline_counts)} trials per setup)"


def _model_metadata(score_file: ScoreFile) -> str:
    model = score_file.metadata.get("model") or score_file.metadata.get("setup_id")
    if model in (None, ""):
        return "unknown"
    if isinstance(model, list):
        return "[" + ", ".join(str(item) for item in model) + "]"
    return str(model)


def _model_compatibility_line(baseline: ScoreFile, candidate: ScoreFile) -> str:
    baseline_model = _model_metadata(baseline)
    candidate_model = _model_metadata(candidate)
    if baseline_model == candidate_model:
        return f"model: matched ({baseline_model})"
    return (
        "model: different "
        f"(current_best={baseline_model}, candidate={candidate_model}; non-blocking)"
    )


def _score_compatibility_checks(
    baseline: ScoreFile,
    candidate: ScoreFile,
    *,
    baseline_counts: list[int],
    candidate_counts: list[int],
) -> tuple[str, ...]:
    return (
        _assert_runtime_budget_compatible(baseline, candidate),
        _assert_hardware_compatible(baseline, candidate),
        _assert_dataset_compatible(baseline, candidate),
        _assert_trial_counts_compatible(baseline_counts, candidate_counts),
        _model_compatibility_line(baseline, candidate),
    )


def _as_framework_benchmark(label: str, score_file: ScoreFile, game_ids: list[str]) -> taaf.benchmark.Benchmark:
    benchmark = taaf.benchmark.Benchmark(label=label, games=[], solver=None, n_passes=1)
    benchmark.game_weights = [1.0] * len(game_ids)
    benchmark.game_runs = [
        taaf.game.GameRun(
            game_id=game_id,
            number_of_levels=1,
            base_actions_per_level=[1],
            final_score=score_file.games[game_id].score,
        )
        for game_id in game_ids
    ]
    return benchmark


def _framework_p_values(
    baseline: ScoreFile,
    candidate: ScoreFile,
    game_ids: list[str],
) -> FrameworkPValues:
    baseline_benchmark = _as_framework_benchmark("current_best", baseline, game_ids)
    candidate_benchmark = _as_framework_benchmark("candidate", candidate, game_ids)
    paired_t = _paired_score_test(baseline_benchmark, candidate_benchmark)
    paired_permutation = _paired_permutation_test(baseline_benchmark, candidate_benchmark)
    return FrameworkPValues(
        paired_t_p_value=float(paired_t["p"]),
        paired_t_statistic=float(paired_t["t"]),
        paired_t_df=float(paired_t["df"]),
        paired_t_games=int(paired_t["n_games"]),
        paired_t_zero_variance=bool(paired_t["zero_variance"]),
        paired_permutation_p_value=float(paired_permutation["p"]),
        paired_permutation_games=int(paired_permutation["n_games"]),
        paired_permutation_exact=bool(paired_permutation["exact"]),
        paired_permutation_count=int(paired_permutation["n_permutations"]),
        paired_permutation_zero_variance=bool(paired_permutation["zero_variance"]),
    )


def compare_scores(
    baseline: ScoreFile,
    candidate: ScoreFile,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    allow_intersection: bool = False,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ComparisonResult:
    game_ids, game_alignment = _aligned_game_ids(
        baseline,
        candidate,
        allow_intersection=allow_intersection,
    )
    baseline_counts = _trial_counts(baseline, game_ids)
    candidate_counts = _trial_counts(candidate, game_ids)
    compatibility_checks = _score_compatibility_checks(
        baseline,
        candidate,
        baseline_counts=baseline_counts,
        candidate_counts=candidate_counts,
    )
    baseline_game_scores = [baseline.games[game_id].score for game_id in game_ids]
    candidate_game_scores = [candidate.games[game_id].score for game_id in game_ids]
    deltas = [
        candidate_score - baseline_score
        for baseline_score, candidate_score in zip(baseline_game_scores, candidate_game_scores, strict=True)
    ]
    mean_delta = _mean(deltas)
    std_delta = _sample_std(deltas)
    probability, se_delta = _probability_true_delta_gt_zero(mean_delta, std_delta, len(deltas))
    posterior_ci = _normal_ci(mean_delta, se_delta)
    bootstrap_ci = bootstrap_mean_ci(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    framework_p_values = _framework_p_values(baseline, candidate, game_ids)
    win_count = sum(1 for delta in deltas if delta > 0.0)
    passed = probability >= confidence_threshold
    comparator = ">=" if passed else "<"
    reason = f"P(true_delta > 0 | results) {comparator} {confidence_threshold:.2f}"
    return ComparisonResult(
        baseline_score=_mean(baseline_game_scores),
        candidate_score=_mean(candidate_game_scores),
        delta=mean_delta,
        paired_games=len(game_ids),
        trials_per_game=_format_trial_count_summary(baseline_counts, candidate_counts),
        total_trials=_format_trial_count(baseline_counts, candidate_counts),
        probability_true_delta_gt_zero=probability,
        posterior_mean_delta=mean_delta,
        posterior_90_ci=posterior_ci,
        win_count=win_count,
        win_rate=win_count / len(game_ids),
        bootstrap_90_ci=bootstrap_ci,
        framework_p_values=framework_p_values,
        passed_acceptance_threshold=passed,
        accept_as_new_best=passed,
        reason=reason,
        game_alignment=game_alignment,
        compatibility_checks=compatibility_checks,
    )


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def render_comparison(
    result: ComparisonResult,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    compatibility_lines = ["Compatibility checks:"]
    compatibility_lines.extend(f"  {line}" for line in result.compatibility_checks)
    framework = result.framework_p_values
    permutation_mode = "exact" if framework.paired_permutation_exact else "monte_carlo"
    return "\n".join(
        [
            "Comparison: candidate vs current_best",
            "",
            "Scores:",
            f"  current_best_score: {result.baseline_score:.6f}",
            f"  candidate_score:    {result.candidate_score:.6f}",
            f"  delta:              {result.delta:+.6f}",
            f"  paired_games:       {result.paired_games}",
            f"  game_alignment:     {result.game_alignment}",
            f"  trials_per_game:    {result.trials_per_game}",
            f"  total_trials:       {result.total_trials}",
            "",
            *compatibility_lines,
            "",
            "Acceptance threshold:",
            "  accept if:",
            f"    P(true_delta > 0 | results) >= {confidence_threshold:.2f}",
            "  minimum_delta_threshold: none",
            "  acceptance_policy: internal_highscore",
            f"  confidence_threshold: {confidence_threshold:.2f}",
            '  interpretation: "Candidate is probably better; not a final external benchmark claim."',
            "",
            "Bayesian estimate:",
            f"  P(true_delta > 0 | results): {result.probability_true_delta_gt_zero:.6f}",
            f"  posterior_mean_delta:        {result.posterior_mean_delta:+.6f}",
            f"  posterior_90_ci:             [{result.posterior_90_ci[0]:+.6f}, {result.posterior_90_ci[1]:+.6f}]",
            "",
            "Robustness checks:",
            f"  win_rate:                    {result.win_rate * 100:.1f}% ({result.win_count}/{result.paired_games} games)",
            f"  bootstrap_90_ci:             [{result.bootstrap_90_ci[0]:+.6f}, {result.bootstrap_90_ci[1]:+.6f}]",
            f"  framework_paired_t_p_value:  {_format_float(framework.paired_t_p_value)}",
            f"  framework_paired_t_stat:     t={_format_float(framework.paired_t_statistic)} "
            f"df={_format_float(framework.paired_t_df)} n_games={framework.paired_t_games} "
            f"zero_variance={_format_bool(framework.paired_t_zero_variance)}",
            f"  framework_permutation_p:     {_format_float(framework.paired_permutation_p_value)}",
            f"  framework_permutation_mode:  {permutation_mode} "
            f"n={framework.paired_permutation_count} n_games={framework.paired_permutation_games} "
            f"zero_variance={_format_bool(framework.paired_permutation_zero_variance)}",
            "",
            "Decision:",
            f"  passed_acceptance_threshold: {_format_bool(result.passed_acceptance_threshold)}",
            f"  accept_as_new_best:          {_format_bool(result.accept_as_new_best)}",
            f'  reason: "{result.reason}"',
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ARC3 lightweight score files.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to significance JSON config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--baseline", default="", help="Path to the current best score JSON or run directory.")
    parser.add_argument("--candidate", default="", help="Path to the candidate score JSON or run directory.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Bayesian probability threshold for accepting the candidate.",
    )
    parser.add_argument("--allow-intersection", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args.config)
    baseline_path = _resolve_score_arg(args, config, kind="baseline")
    candidate_path = _resolve_score_arg(args, config, kind="candidate")
    confidence_threshold = _resolve_float_setting(
        args,
        config,
        attr="confidence_threshold",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    bootstrap_samples = _resolve_int_setting(
        args,
        config,
        attr="bootstrap_samples",
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        min_value=1,
    )
    bootstrap_seed = _resolve_int_setting(
        args,
        config,
        attr="bootstrap_seed",
        default=DEFAULT_BOOTSTRAP_SEED,
        min_value=0,
    )
    baseline = load_score_file(baseline_path)
    candidate = load_score_file(candidate_path)
    result = compare_scores(
        baseline,
        candidate,
        confidence_threshold=confidence_threshold,
        allow_intersection=args.allow_intersection or bool(config.get("allow_intersection", False)),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    print(render_comparison(result, confidence_threshold=confidence_threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
