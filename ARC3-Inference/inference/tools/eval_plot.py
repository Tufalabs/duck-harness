"""Plot aggregate official eval metrics across a group of runs."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from inference.tools import eval as eval_tool


DEFAULT_OUTPUT_ROOT = "runs/aggregates"


@dataclass(frozen=True)
class GamePlotStats:
    game_id: str
    score_mean: float
    score_std: float
    completion_rate_mean: float
    completion_rate_std: float
    run_count: int


@dataclass(frozen=True)
class PlotSummary:
    runs: list[str]
    games: list[GamePlotStats]
    score_overall_mean: float
    score_overall_std: float
    completion_overall_mean: float
    completion_overall_std: float


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _evaluation_path(run_dir: Path) -> Path | None:
    for filename in (eval_tool.EVALUATION_FILE_NAME, eval_tool.LEGACY_EVALUATION_FILE_NAME):
        path = run_dir / filename
        if path.exists():
            return path
    return None


def _load_evaluation(run_dir: Path) -> dict[str, Any]:
    path = _evaluation_path(run_dir)
    if path is None:
        raise FileNotFoundError(f"{run_dir}: missing {eval_tool.EVALUATION_FILE_NAME}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_evaluation_files(run_dirs: list[Path]) -> list[Path]:
    missing_run_dirs = [run_dir for run_dir in run_dirs if _evaluation_path(run_dir) is None]
    if not missing_run_dirs:
        return []

    generated_paths: list[Path] = []
    for run_dir in missing_run_dirs:
        summary = eval_tool.evaluate_runs([run_dir])
        generated_paths.extend(eval_tool.save_run_evaluations(summary, run_dirs=[run_dir]))
    return generated_paths


ensure_eval_official = ensure_evaluation_files


def _game_completion_rate(game_payload: dict[str, Any]) -> float:
    if "completion_rate" in game_payload:
        try:
            completion_rate = float(game_payload.get("completion_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            completion_rate = 0.0
        return max(0.0, min(1.0, completion_rate))

    try:
        levels_completed = float(
            game_payload.get(
                "levels_completed_mean",
                game_payload.get("levels_completed", 0.0),
            )
            or 0.0
        )
    except (TypeError, ValueError):
        levels_completed = 0.0
    try:
        total_levels = float(game_payload.get("total_levels", 0) or 0)
    except (TypeError, ValueError):
        total_levels = 0.0
    if total_levels <= 0:
        return 0.0
    return max(0.0, min(1.0, levels_completed / total_levels))


def summarize_runs(run_dirs: list[Path]) -> PlotSummary:
    per_game_scores: dict[str, list[float]] = {}
    per_game_completion_rates: dict[str, list[float]] = {}
    run_names: list[str] = []

    for run_dir in run_dirs:
        payload = _load_evaluation(run_dir)
        run_name = str(payload.get("run_name") or run_dir.name)
        run_names.append(run_name)
        for game_payload in payload.get("games", []):
            if not isinstance(game_payload, dict):
                continue
            game_id = str(game_payload.get("game_id") or "").strip()
            if not game_id:
                continue
            try:
                score = float(game_payload.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            per_game_scores.setdefault(game_id, []).append(score)
            per_game_completion_rates.setdefault(game_id, []).append(_game_completion_rate(game_payload))

    game_ids = sorted(per_game_scores)
    game_stats = [
        GamePlotStats(
            game_id=game_id,
            score_mean=_mean(per_game_scores[game_id]),
            score_std=_std(per_game_scores[game_id]),
            completion_rate_mean=_mean(per_game_completion_rates.get(game_id, [])),
            completion_rate_std=_std(per_game_completion_rates.get(game_id, [])),
            run_count=len(per_game_scores[game_id]),
        )
        for game_id in game_ids
    ]

    score_game_means = [game.score_mean for game in game_stats]
    completion_game_means = [game.completion_rate_mean for game in game_stats]
    return PlotSummary(
        runs=run_names,
        games=game_stats,
        score_overall_mean=_mean(score_game_means),
        score_overall_std=_std(score_game_means),
        completion_overall_mean=_mean(completion_game_means),
        completion_overall_std=_std(completion_game_means),
    )


def _plot_metric(
    *,
    labels: list[str],
    means: list[float],
    stds: list[float],
    ylabel: str,
    title: str,
    overall_mean: float,
    overall_std: float,
    output_path: Path,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.2), 5.0))
    positions = list(range(len(labels)))
    ax.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        color="#4C78A8",
        edgecolor="#1f1f1f",
        linewidth=0.8,
    )
    ax.axhline(overall_mean, color="#D62728", linestyle="--", linewidth=1.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nOverall mean={overall_mean:.4f}, std={overall_std:.4f}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_plots(summary: PlotSummary, *, output_dir: Path) -> dict[str, Path]:
    labels = [game.game_id for game in summary.games]
    score_plot_path = output_dir / "official_score_by_game.png"
    completion_plot_path = output_dir / "completion_rate_by_game.png"
    summary_json_path = output_dir / "summary.json"
    score_y_max = max(
        100.0,
        max((game.score_mean + game.score_std for game in summary.games), default=0.0) * 1.05,
    )

    _plot_metric(
        labels=labels,
        means=[game.score_mean for game in summary.games],
        stds=[game.score_std for game in summary.games],
        ylabel="Official ARC Score (0-100)",
        title="Official ARC Score by Game",
        overall_mean=summary.score_overall_mean,
        overall_std=summary.score_overall_std,
        output_path=score_plot_path,
        y_min=0.0,
        y_max=score_y_max,
    )
    _plot_metric(
        labels=labels,
        means=[game.completion_rate_mean for game in summary.games],
        stds=[game.completion_rate_std for game in summary.games],
        ylabel="Completion Rate",
        title="Completion Rate by Game",
        overall_mean=summary.completion_overall_mean,
        overall_std=summary.completion_overall_std,
        output_path=completion_plot_path,
        y_min=0.0,
        y_max=1.0,
    )

    summary_json_path.write_text(
        json.dumps(
            {
                "runs": summary.runs,
                "score_overall_mean": summary.score_overall_mean,
                "score_overall_std": summary.score_overall_std,
                "completion_overall_mean": summary.completion_overall_mean,
                "completion_overall_std": summary.completion_overall_std,
                "games": [
                    {
                        "game_id": game.game_id,
                        "score_mean": game.score_mean,
                        "score_std": game.score_std,
                        "completion_rate_mean": game.completion_rate_mean,
                        "completion_rate_std": game.completion_rate_std,
                        "run_count": game.run_count,
                    }
                    for game in summary.games
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "score_plot": score_plot_path,
        "completion_plot": completion_plot_path,
        "summary_json": summary_json_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate official eval metrics for a group of runs.")
    parser.add_argument("--runs-dir", default="runs", help="Parent directory containing run folders.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directory names to include.")
    parser.add_argument("--name", default="group", help="Output subdirectory name under the output root.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory for plots (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_dir = Path(args.runs_dir)
    run_dirs = [runs_dir / str(name) for name in args.runs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    generated_paths = ensure_evaluation_files(run_dirs)
    for path in generated_paths:
        print(f"Computed missing eval: {path}")

    summary = summarize_runs(run_dirs)
    output_dir = Path(args.output_dir) / str(args.name).strip()
    outputs = save_summary_plots(summary, output_dir=output_dir)
    print(f"Saved score plot: {outputs['score_plot']}")
    print(f"Saved completion plot: {outputs['completion_plot']}")
    print(f"Saved summary JSON: {outputs['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
