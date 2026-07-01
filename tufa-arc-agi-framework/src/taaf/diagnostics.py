"""Per-benchmark and cross-benchmark diagnostics (R4.*).

Public surface:

- ``generate_run_html`` — R4.01 per-run HTML
- ``generate_run_summary_txt`` — R4.02 text summary
- ``generate_comparison_html`` — R4.11/R4.12 cross-run comparison
- ``regenerate_run_diagnostics`` / ``regenerate_comparison_diagnostics``
  — load + regen from a finished run dir

PNGs are embedded inline as base64; MP4s are written to disk and linked.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import re
import statistics
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, cast

import imageio.v3 as iio
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats

import taaf.benchmark
import taaf.game

matplotlib.use("Agg")  # headless rendering

# scipy.stats.ttest_rel emits "Precision loss occurred" when paired
# differences have near-zero variance (common with small n_games). The
# p-value is still meaningful for our use; the warning is noise. Inner
# ``catch_warnings`` blocks don't suppress reliably under ipykernel /
# pytest, so install a global filter at import time.
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred",
    category=RuntimeWarning,
    module=r"scipy\.stats.*",
)


# --- Style constants --------------------------------------------------------

BG_COLOR = "#1e1e1e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#3a3a3a"
ACCENT_COLORS = [
    "#9bd1ff",
    "#ffb47a",
    "#a8e6a1",
    "#e6a1d3",
    "#f0d878",
    "#ff9b9b",
    "#b39bff",
    "#d4e67a",
    "#9be6e6",
    "#ff9bc8",
    "#9bb4ff",
    "#9be6cb",
]
# Distinct palette for the comparison page so per-run vs cross-run
# plots are visually distinguishable side-by-side.
RUN_COLORS = [
    "#ff6b9d",
    "#7dd3fc",
    "#fbbf24",
    "#a78bfa",
    "#34d399",
    "#f87171",
    "#fb923c",
    "#84cc16",
    "#2dd4bf",
    "#818cf8",
    "#e879f9",
    "#4ade80",
]
# Muted colors for ``GameRun.state`` cells — low saturation so the
# column reads as metadata, not competing with the curve palette.
_STATE_COLORS: dict[str, str] = {
    "won": "#6db04e",  # muted green
    "gave_up": "#8a8a8a",  # gray
    "cancelled": "#d09c4d",  # muted amber
    "crashed": "#d05050",  # muted red
    "playing": "#5a92c8",  # muted blue (shouldn't appear after teardown)
    "not_started": "#666",
}


def _state_cell_html(state: str) -> str:
    return f'<td style="color: {_STATE_COLORS.get(state, "#888")}">{escape(state)}</td>'


THUMB_DEFAULT_PX = 32
THUMB_FINAL_PX = 64
MOVIE_FPS = 8
# Cap frame count so a runaway-long game can't produce a multi-MB MP4.
# Over the cap, frames are sampled at equal index spacing (first and
# last frame preserved). Subsamples are flagged at row-level and
# page-level in the HTML.
MAX_MOVIE_FRAMES = 1000
# Nearest-neighbor upscale applied per-frame before x264 encoding. At
# native 64×64 H.264 has only 4×4 macroblocks to work with, then the
# browser bilinear-upscales another ~11×, so pixel art ends up blurry.
# Upscaling here gives x264 real room without changing the art (kron
# with a constant block preserves color exactly).
MP4_UPSCALE = 2


# --- Score variants ---------------------------------------------------------
# Three scorers feed the per-run and comparison HTML's tab widget. The
# per-game cards / drill-downs / per-pass-per-game table always render
# under ARC regardless of which tab is active.


@dataclass(frozen=True)
class Scorer:
    """One scoring scheme. ``partial`` evaluates the score on a
    ``(actions_per_level, base_actions_per_level, levels_completed,
    n_levels)`` snapshot — same shape as
    ``GameRun._compute_final_score``, generalised to different weight /
    aggregation rules. Used for both the per-action level-win events
    on a curve and for the terminal score per run.

    Fields:

    - ``key``: url-safe id used in filenames / cache keys.
    - ``label``: display name in nav links and plot titles.
    - ``description``: short HTML paragraph under the page's score nav.
    - ``y_axis_label``: y-axis label on the cross-game pooled curves.
    - ``path_suffix``: filename suffix for this variant's HTML
      (``""`` for ARC).
    - ``partial``: the partial-score function.
    """

    key: str
    label: str
    description: str
    y_axis_label: str
    path_suffix: str
    partial: Callable[[list[int], list[int] | None, int, int], float]


def _arc_partial_score(
    actions_per_level: list[int],
    base_actions_per_level: list[int] | None,
    levels_completed: int,
    n_levels: int,
) -> float:
    """Canonical ARC-AGI3 score on a snapshot. Mirror of
    ``GameRun._compute_final_score``; see there for the formula."""
    if base_actions_per_level is None or n_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(n_levels):
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < levels_completed
        actions = actions_per_level[level_idx] if level_idx < len(actions_per_level) else 0
        baseline = base_actions_per_level[level_idx]
        if completed and actions > 0:
            level_score = min(115.0, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    if total_weights == 0:
        return 0.0
    score = total_score / total_weights
    max_score = max_weights / total_weights * 100
    return min(score, max_score)


def _weighted_partial_score(
    actions_per_level: list[int],
    base_actions_per_level: list[int] | None,
    levels_completed: int,
    n_levels: int,
) -> float:
    """ARC per-level formula with weights ``[1, 2, 2, …]`` — level 1
    contributes 1×, every later level 2×. Values reaching deeper levels
    more uniformly than the official 1, 2, 3, … ramp.
    """
    if base_actions_per_level is None or n_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(n_levels):
        weight = 1 if level_idx == 0 else 2
        total_weights += weight
        completed = level_idx < levels_completed
        actions = actions_per_level[level_idx] if level_idx < len(actions_per_level) else 0
        baseline = base_actions_per_level[level_idx]
        if completed and actions > 0:
            level_score = min(115.0, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    if total_weights == 0:
        return 0.0
    score = total_score / total_weights
    max_score = max_weights / total_weights * 100
    return min(score, max_score)


def _levels_partial_score(
    actions_per_level: list[int],
    base_actions_per_level: list[int] | None,
    levels_completed: int,
    n_levels: int,
) -> float:
    """Raw completed-level count. No baseline, no normalisation — so
    comparable across runs of the same game set, not across different
    sets."""
    del actions_per_level, base_actions_per_level, n_levels
    return float(levels_completed)


ARC_SCORER = Scorer(
    key="arc",
    label="Official ARC",
    description=(
        "Standard ARC-AGI3 score. Each completed level contributes "
        "<code>min(115, (baseline / actions)² × 100)</code>, weighted by level "
        "index (level 1 → weight 1, level 2 → 2, …) and normalised by the "
        "total weight. Mirrors the engine scorecard."
    ),
    y_axis_label="average score per game",
    path_suffix="",
    partial=_arc_partial_score,
)
WEIGHTED_SCORER = Scorer(
    key="weighted",
    label="Weighted (1, 2, 2, …)",
    description=(
        "Same per-level score formula as the official ARC scorer, but with "
        "flat weights <code>[1, 2, 2, …]</code> — level 1 contributes 1×, "
        "every later level 2×. Values reaching deeper levels more uniformly "
        "than the official 1, 2, 3, … ramp."
    ),
    y_axis_label="average score per game",
    path_suffix="_weighted",
    partial=_weighted_partial_score,
)
LEVELS_SCORER = Scorer(
    key="levels",
    label="Levels beaten",
    description=(
        "Raw count of levels completed during the run — no baseline ratio, "
        "no partial credit, no normalisation. Games with more levels have a "
        "higher possible maximum, so this score is comparable across runs of "
        "the same games but <em>not</em> across different game sets."
    ),
    y_axis_label="average levels beaten per game",
    path_suffix="_levels",
    partial=_levels_partial_score,
)

SCORERS: list[Scorer] = [ARC_SCORER, WEIGHTED_SCORER, LEVELS_SCORER]


def _score_for(run: taaf.game.GameRun, scorer: Scorer) -> float:
    """Terminal score for ``run`` under ``scorer``. For ARC, prefers the
    engine-reconciled ``run.final_score`` (set by ``Game.finish_game()``)
    when available; for variants, always recomputes from the snapshot.
    """
    if scorer is ARC_SCORER and run.final_score is not None:
        return run.final_score
    return scorer.partial(
        run.actions_per_level,
        run.base_actions_per_level,
        run.levels_completed,
        run.number_of_levels,
    )


def _resolve_score_fn(
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None,
) -> Callable[[taaf.game.GameRun], float]:
    """Per-run score callable used by the stats helpers. ``None`` ⇒ the
    terminal score (run evaluated where it ended). The common-budget
    variant passes a function that reads each run's score-vs-tokens curve
    at a shared per-game token budget instead."""
    if score_fn is not None:
        return score_fn

    def _terminal(run: taaf.game.GameRun) -> float:
        return _score_for(run, scorer)

    return _terminal


def _scorer_tabs_html(panels: list[tuple[Scorer, str]]) -> str:
    """CSS-only folder-tab widget. ``panels`` is a list of
    ``(scorer, panel_inner_html)`` tuples; the first is the default-active
    tab. With a single panel (in-flight: ARC only), the tab chrome is
    skipped and the panel's HTML is returned bare so the page lays out as
    if there were no tab UI.
    """
    if not panels:
        return ""
    if len(panels) == 1:
        return panels[0][1]
    inputs: list[str] = []
    labels: list[str] = []
    panel_divs: list[str] = []
    for i, (scorer, inner_html) in enumerate(panels):
        checked = " checked" if i == 0 else ""
        rid = f"scorer-{scorer.key}"
        inputs.append(f'<input type="radio" name="scorer" id="{rid}"{checked}>')
        labels.append(f'<label for="{rid}" class="tab-label">{escape(scorer.label)}</label>')
        panel_divs.append(
            f'<div class="tab-panel" id="panel-{scorer.key}">'
            f'<p class="scorer-desc">{scorer.description}</p>'
            f"{inner_html}"
            "</div>"
        )
    return (
        '<div class="scorer-tabs">'
        + "".join(inputs)
        + '<div class="tab-row">'
        + "".join(labels)
        + "</div>"
        + "".join(panel_divs)
        + "</div>"
    )


def _budget_scorer_tabs_html(
    scorer_panels: list[tuple[Scorer, dict[str, str]]],
    *,
    budget_labels: list[tuple[str, str]],
) -> str:
    """Two-level CSS-only tab widget (R4.12): an outer row of budget tabs
    above an inner row of score-variant tabs. A panel is visible only
    when its budget tab and its scorer tab are both selected — so the
    statistics are shown at each run's endpoint and at a shared per-game
    token budget, for each scorer.

    ``budget_labels`` is ``[(key, label), …]`` (keys ``full`` / ``capped``,
    matching the hardcoded visibility CSS); ``scorer_panels`` maps each
    scorer to its ``{budget_key: inner_html}``. With a single budget mode
    the budget dimension is dropped — the scorer-only widget is used.
    """
    if len(budget_labels) < 2:
        only = budget_labels[0][0]
        return _scorer_tabs_html([(s, by[only]) for s, by in scorer_panels])
    inputs: list[str] = []
    for i, (bk, _) in enumerate(budget_labels):
        inputs.append(f'<input type="radio" name="bs-budget" id="bs-b-{bk}"{" checked" if i == 0 else ""}>')
    for i, (scorer, _) in enumerate(scorer_panels):
        inputs.append(f'<input type="radio" name="bs-scorer" id="bs-s-{scorer.key}"{" checked" if i == 0 else ""}>')
    budget_row = "".join(f'<label for="bs-b-{bk}" class="tab-label">{escape(bl)}</label>' for bk, bl in budget_labels)
    scorer_row = "".join(
        f'<label for="bs-s-{s.key}" class="tab-label">{escape(s.label)}</label>' for s, _ in scorer_panels
    )
    panel_divs: list[str] = []
    for scorer, by_budget in scorer_panels:
        for bk, _ in budget_labels:
            panel_divs.append(
                f'<div class="tab-panel" id="bs-p-{bk}-{scorer.key}">'
                f'<p class="scorer-desc">{scorer.description}</p>'
                f"{by_budget[bk]}"
                "</div>"
            )
    return (
        '<div class="bs-tabs">'
        + "".join(inputs)
        + f'<div class="tab-row budget-row">{budget_row}</div>'
        + f'<div class="tab-row scorer-row">{scorer_row}</div>'
        + "".join(panel_divs)
        + "</div>"
    )


_VARIANT_REMAINDER_NOTE = (
    '<p class="design-note">Sections below this point — per-game cards, '
    "per-game drill-downs, and the per-pass-per-game table — always render "
    "under the official ARC-AGI score, independent of the variant chosen above."
    "</p>"
)


def _weights_banner_html(benchmark: taaf.benchmark.Benchmark) -> str:
    """Yellow banner shown when ``Benchmark.game_weights`` is set.
    Summarises with totals + weight-0 exclusion count — a full per-game
    list would dominate the page on large benchmarks.
    """
    if benchmark.game_weights is None:
        return ""
    weights = _weights_by_game_id(benchmark)
    if not weights:
        return ""
    n_total = len(weights)
    n_excluded = sum(1 for w in weights.values() if w == 0)
    excl_note = f" {n_excluded} game{'s' if n_excluded != 1 else ''} excluded (weight 0)." if n_excluded else ""
    return (
        '<div class="weights-banner">'
        "<p><strong>Game weights applied.</strong> The pooled curves, "
        "per-pass mean, head-to-head means, and t-tests below weight each "
        "game per <code>Benchmark.game_weights</code> "
        f"({n_total} game{'s' if n_total != 1 else ''} total.{excl_note}) "
        "Per-game cards and the per-pass-per-game table at the bottom "
        "remain unweighted.</p>"
        "</div>"
    )


def _comparison_weights_banner_html(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
) -> str:
    """Yellow banner for the comparison page. ``generate_comparison_html``
    has harmonized all benchmarks to share weights, so the banner
    describes the single shared weighting.
    """
    del labels
    if not any(b.game_weights is not None for b in benchmarks):
        return ""
    weights = _weights_by_game_id(benchmarks[0])
    if not weights:
        return ""
    n_total = len(weights)
    n_excluded = sum(1 for w in weights.values() if w == 0)
    excl_note = f" {n_excluded} game{'s' if n_excluded != 1 else ''} excluded (weight 0)." if n_excluded else ""
    return (
        '<div class="weights-banner">'
        "<p><strong>Game weights applied.</strong> Cross-game aggregation "
        "(pooled curves, per-run mean, t-tests, head-to-head means) uses "
        f"the shared <code>Benchmark.game_weights</code> ({n_total} game"
        f"{'s' if n_total != 1 else ''} total.{excl_note}) "
        "Per-game grid stays unweighted.</p>"
        "</div>"
    )


# --- Data extraction --------------------------------------------------------


def _live_score(run: taaf.game.GameRun) -> float:
    """Score reflecting the run's current state, finalized or in-flight.
    ``run.final_score`` is None until ``finish_game()`` runs, so we
    compute on the fly from the snapshot. Pure function of local fields
    — never touches the engine scorecard, so safe to call from the
    periodic save loop.
    """
    if run.final_score is not None:
        return run.final_score
    return run._compute_final_score()


def _total_tokens(run: taaf.game.GameRun) -> int:
    # Folding the no-move give-up/cancel tokens (R11.03) in at this single
    # primitive means every grand-total consumer and every score-vs-tokens
    # curve (whose terminal x is this total) picks them up unchanged.
    return sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens


def _total_wallclock(run: taaf.game.GameRun) -> float:
    # Fold in the no-move give-up/cancel turn (mirrors _total_tokens).
    # final_wallclock_seconds and ActionRecord.wallclock_seconds share the
    # same monotonic-since-start_game reference; finish_game() stamps final
    # after the last action, so max() picks the finish time once finalized
    # and the last action while in-flight (final defaults to 0.0).
    last = run.history[-1].wallclock_seconds if run.history else 0.0
    return max(last, run.final_wallclock_seconds)


def _benchmark_elapsed_seconds(benchmark: taaf.benchmark.Benchmark) -> float | None:
    if benchmark.start_time is None:
        return None
    end_time = benchmark.end_time
    if end_time is None:
        if benchmark.start_time.tzinfo is None:
            end_time = datetime.now()
        else:
            end_time = datetime.now(benchmark.start_time.tzinfo)
    return max((end_time - benchmark.start_time).total_seconds(), 0.0)


def _count_actions(run: taaf.game.GameRun) -> int:
    """Total counted actions — same denominator the score formula uses.
    Equal to ``len(run.history)`` since no-ops aren't admitted."""
    return sum(run.actions_per_level)


def _total_movie_frames(run: taaf.game.GameRun) -> int:
    """Uncapped frame count the movie would render. One per state
    (animation frames are skipped). Used to detect cap-induced truncation.
    """
    return len(run.intermediate_states)


def _game_ids(benchmark: taaf.benchmark.Benchmark) -> list[str]:
    """Unique game_ids in alphabetical order — stable display order
    independent of the solver's dispatch order. Use ``benchmark.game_runs``
    directly if you need play order.
    """
    return sorted({r.game_id for r in benchmark.game_runs})


def _pass_runs(benchmark: taaf.benchmark.Benchmark, pass_idx: int) -> list[taaf.game.GameRun]:
    """All runs in the given pass, one per game (passes-major layout in
    ``benchmark.game_runs``: pass ``p`` of game ``g`` lives at
    ``p * n_games + g``).
    """
    n_games = len(benchmark.game_runs) // benchmark.n_passes if benchmark.n_passes else 0
    return list(benchmark.game_runs[pass_idx * n_games : (pass_idx + 1) * n_games])


def _game_runs_by_id(benchmark: taaf.benchmark.Benchmark, game_id: str) -> list[taaf.game.GameRun]:
    return [r for r in benchmark.game_runs if r.game_id == game_id]


def _weights_by_game_id(benchmark: taaf.benchmark.Benchmark) -> dict[str, float]:
    """Resolve ``Benchmark.game_weights`` (list parallel to ``games``)
    into a ``{game_id: weight}`` map using the first pass to learn the
    canonical game order. Only resolvable post-``Benchmark.run`` because
    ``Game.game_id`` is only populated after ``start_game``.
    """
    if benchmark.n_passes == 0 or not benchmark.game_runs:
        return {}
    n_games = len(benchmark.game_runs) // benchmark.n_passes
    if n_games == 0:
        return {}
    weights = benchmark.game_weights if benchmark.game_weights is not None else [1.0] * n_games
    result: dict[str, float] = {}
    for i in range(n_games):
        run = benchmark.game_runs[i]  # pass 0, game i — canonical order
        if i < len(weights):
            result[run.game_id] = float(weights[i])
    return result


def _format_duration(td: timedelta) -> str:
    """Compact human-readable duration (h/m/s, integer seconds)."""
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _timestamps_block_html(start: datetime | None, end: datetime | None, *, with_end: bool) -> str:
    """Inline HTML block for the start/end/duration block. ``with_end`` controls
    whether to show end-time + duration (used on the per-run page) or just
    start-time (comparison index, where each run is summarized briefly)."""
    if start is None:
        return ""
    parts: list[str] = [f"<p>Started: <code>{_format_timestamp(start)}</code></p>"]
    if with_end:
        if end is not None:
            parts.append(f"<p>Ended: <code>{_format_timestamp(end)}</code></p>")
            parts.append(f"<p>Duration: <code>{_format_duration(end - start)}</code></p>")
        else:
            parts.append("<p>Status: <em>in progress</em></p>")
    return "".join(parts)


def _git_status_block_html(benchmark: taaf.benchmark.Benchmark) -> str:
    """Embed ``job_dir/git_status.txt`` (written by the launcher) as a
    collapsible ``<details>`` block. Empty string if the file is
    missing — most often because ``job_dir`` is unset entirely.
    """
    if benchmark.job_dir is None:
        return ""
    path = benchmark.job_dir / "git_status.txt"
    if not path.exists():
        return ""
    return f"<details><summary>git status</summary><pre>{escape(path.read_text())}</pre></details>"


def _per_game_mean_scores(
    benchmark: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, float]:
    """Per-game mean score across passes. Weight 0 games are excluded
    (they should not appear in cross-game scatters / aggregates).
    ``score_fn`` overrides the per-run score (see ``_resolve_score_fn``)."""
    sf = _resolve_score_fn(scorer, score_fn)
    weights = _weights_by_game_id(benchmark)
    out: dict[str, float] = {}
    for g in _game_ids(benchmark):
        if weights.get(g, 1.0) == 0.0:
            continue
        scores = [sf(r) for r in _game_runs_by_id(benchmark, g)]
        if scores:
            out[g] = statistics.mean(scores)
    return out


def _run_pass_stats(
    benchmark: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, float | None]:
    """Per-run mean / σ / SEM across passes.

    For each pass we take the weighted-mean per-game score (across all
    games in that pass, using ``Benchmark.game_weights``), then aggregate
    those per-pass means. ``sigma`` is the sample-stdev (ddof=1) of the
    per-pass means; ``sem`` = σ / √n_passes. Both are ``None`` when
    ``n_passes < 2`` — sample stdev is undefined for a single point.
    Weight-0 games are excluded from each pass's mean. ``score_fn``
    overrides the per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    weights = _weights_by_game_id(benchmark)
    pass_means: list[float] = []
    for p in range(benchmark.n_passes):
        runs = _pass_runs(benchmark, p)
        kept = [(r, weights.get(r.game_id, 1.0)) for r in runs]
        kept = [(r, w) for r, w in kept if w > 0]
        if not kept:
            continue
        total_w = sum(w for _, w in kept)
        weighted_sum = sum(sf(r) * w for r, w in kept)
        pass_means.append(weighted_sum / total_w)
    if not pass_means:
        return {"mean": None, "sigma": None, "sem": None, "n_passes": 0}
    mean = float(np.mean(pass_means))
    if len(pass_means) < 2:
        return {"mean": mean, "sigma": None, "sem": None, "n_passes": len(pass_means)}
    sigma = float(np.std(pass_means, ddof=1))
    sem = sigma / (len(pass_means) ** 0.5)
    return {"mean": mean, "sigma": sigma, "sem": sem, "n_passes": len(pass_means)}


# --- Curve construction (R4.01 plot semantics) ------------------------------


def _per_game_partial_curve(
    run: taaf.game.GameRun,
    x_fn: Callable[[taaf.game.GameRun], float],
    level_cap: float = float("inf"),
    *,
    scorer: Scorer = ARC_SCORER,
) -> list[tuple[float, float]]:
    """Per-game step events that credit score at each level-completing
    action. Returns ``[(0.0, 0.0), ..., (final_x, final_score)]``.

    Off-by-one: ``intermediate_states[i + 1]`` is the state after
    ``history[i]``; we attribute action ``i`` to the level current
    before the action (running ``levels_completed`` at entry).
    ``actions_per_level`` is incremented *before* the level-completion
    check, so the level that just won has its full final action count
    when ``scorer.partial`` runs.

    ``level_cap`` (per-run page's what-if): truncates the lane when the
    in-level cumulative cost crosses the cap before the level is won.
    Default ``inf`` reproduces the unrestricted curve exactly.
    """
    total_x = float(x_fn(run))
    if not run.history or run.base_actions_per_level is None:
        return [(0.0, 0.0), (total_x, _score_for(run, scorer))]

    # Cumulative x at the end of each action. Hard-code the two known
    # x_fns; anything else falls back to uniform interpolation (keeps
    # the curve well-defined but doesn't exercise partial credit).
    n_actions = len(run.history)
    if x_fn is _total_tokens:
        cum_x_at_action: list[float] = []
        s = 0
        for rec in run.history:
            s += rec.generated_tokens
            cum_x_at_action.append(float(s))
    elif x_fn is _total_wallclock:
        cum_x_at_action = [float(rec.wallclock_seconds) for rec in run.history]
    else:
        per_action = total_x / n_actions if n_actions else 0.0
        cum_x_at_action = [(i + 1) * per_action for i in range(n_actions)]

    events: list[tuple[float, float]] = [(0.0, 0.0)]
    actions_per_level = [0] * run.number_of_levels
    levels_completed = 0  # framework-monotonic counter
    level_start_cum_x = 0.0
    capped_total_x = 0.0
    truncated = False

    for i, _rec in enumerate(run.history):
        if i + 1 >= len(run.intermediate_states):
            break
        post_state = run.intermediate_states[i + 1]
        cum_x_after = cum_x_at_action[i]
        in_level_cost = cum_x_after - level_start_cum_x
        if in_level_cost > level_cap:
            # Cap exceeded mid-level: no score credit for this level.
            capped_total_x = level_start_cum_x + level_cap
            truncated = True
            break
        if levels_completed < run.number_of_levels:
            actions_per_level[levels_completed] += 1
        # ``GameRun.levels_completed`` is monotonic; max() prevents a
        # full-reset (which makes arcengine's counter decrease) from
        # clawing back previously-credited levels.
        new_completed = max(int(post_state.raw.levels_completed), levels_completed)
        if new_completed > levels_completed:
            levels_completed = new_completed
            partial = scorer.partial(
                actions_per_level,
                run.base_actions_per_level,
                levels_completed,
                run.number_of_levels,
            )
            events.append((cum_x_after, partial))
            level_start_cum_x = cum_x_after
        capped_total_x = cum_x_after

    final_x = capped_total_x if truncated else total_x
    if events[-1][0] < final_x - 1e-9:
        events.append((final_x, events[-1][1]))
    return events


def _pass_curve(
    pass_runs: list[taaf.game.GameRun],
    x_fn: Callable[[taaf.game.GameRun], float],
    *,
    scorer: Scorer = ARC_SCORER,
    weights_by_game_id: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean-cost-per-lane vs mean-score-per-lane step curve.

    Each input ``GameRun`` is one independent lane. All lanes run in
    parallel at equal speed; at parameter ``t`` lane ``i`` has spent
    ``min(t, T_i)`` cost. Both quantities are averaged (or weighted-
    averaged) across lanes. Knots are emitted wherever any lane steps.

    Decomposability: equal-weight pooling all ``(game, pass)`` runs in
    one call equals averaging passes within game then games. The
    function is agnostic to what the lanes mean — pass lanes within one
    pass, all-passes-of-one-game lanes, and everything-pooled lanes all
    use the same code path.

    Reading the curve: at ``x = X``, the value is the expected score
    when the method runs at an average per-game budget of ``X``.
    """
    if not pass_runs:
        return np.array([0.0]), np.array([0.0])
    lane_weights = (
        [weights_by_game_id.get(r.game_id, 1.0) for r in pass_runs] if weights_by_game_id is not None else None
    )
    return _pool_lane_events(
        [_per_game_partial_curve(r, x_fn, scorer=scorer) for r in pass_runs],
        weights=lane_weights,
    )


def _pool_lane_events(
    per_lane_events: list[list[tuple[float, float]]],
    weights: list[float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Step-curve combiner shared by ``_pass_curve`` and the per-level
    cap what-if. Returns the weighted-mean-cost vs weighted-mean-score
    step curve. ``weights`` defaults to equal-1.0. Lanes with weight 0
    are filtered out entirely (matches ``Benchmark.game_weights``'s
    "weight 0 = exclude from cross-game aggregation" semantics).
    """
    n = len(per_lane_events)
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != lanes length {n}")
    kept: list[tuple[list[tuple[float, float]], float]] = [
        (evts, w) for evts, w in zip(per_lane_events, weights) if w > 0
    ]
    if not kept:
        return np.array([0.0]), np.array([0.0])
    kept_events = [e for e, _ in kept]
    kept_weights = [w for _, w in kept]
    total_weight = sum(kept_weights)
    per_lane_total = [evts[-1][0] for evts in kept_events]
    knot_set: set[float] = {0.0}
    for evts in kept_events:
        for cx, _ in evts:
            knot_set.add(cx)
    knots = sorted(knot_set)
    xs: list[float] = []
    ys: list[float] = []
    for t in knots:
        combined_cost = 0.0
        score_sum = 0.0
        for evts, total, w in zip(kept_events, per_lane_total, kept_weights):
            t_in_g = min(t, total)
            combined_cost += t_in_g * w
            cur = 0.0
            for cx, cy in evts:
                if cx <= t_in_g + 1e-9:
                    cur = cy
                else:
                    break
            score_sum += cur * w
        xs.append(combined_cost / total_weight)
        ys.append(score_sum / total_weight)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _max_tokens_per_level(benchmark: taaf.benchmark.Benchmark) -> int:
    """Largest token total any single level consumed across all
    ``(game, pass)`` lanes, won or not. Returns 0 if no lane has run yet.
    """
    best = 0
    for run in benchmark.game_runs:
        if not run.history:
            continue
        apl = run.actions_per_level
        cum = 0
        cum_at_action: list[int] = []
        for rec in run.history:
            cum += rec.generated_tokens
            cum_at_action.append(cum)
        start = 0
        for n_in_level in apl:
            if n_in_level == 0:
                continue
            end = start + n_in_level - 1
            level_total = cum_at_action[end] - (cum_at_action[start - 1] if start > 0 else 0)
            if level_total > best:
                best = level_total
            start += n_in_level
    return best


def _tokens_vs_wallclock_curve(
    benchmark: taaf.benchmark.Benchmark,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cumulative generated tokens across every action in every run as
    a function of job wallclock seconds (measured from
    ``benchmark.start_time``). Per-action x is
    ``(run.started_at - benchmark.start_time) + rec.wallclock_seconds``
    so games that queued behind a pool semaphore don't all collapse to
    benchmark t=0. Old JSONs that predate ``GameRun.started_at`` fall
    back to a zero offset (game-relative wallclock only).
    """
    bench_start = benchmark.start_time
    events: list[tuple[float, int]] = []
    for run in benchmark.game_runs:
        if run.started_at is not None and bench_start is not None:
            offset = (run.started_at - bench_start).total_seconds()
        else:
            offset = 0.0
        for rec in run.history:
            events.append((offset + float(rec.wallclock_seconds), int(rec.generated_tokens)))
        # No-move give-up / cancel tokens (R11.03): charged at the finish
        # time (final_wallclock_seconds — same axis as per-action
        # wallclock_seconds), so they land after the last move.
        if run.final_generated_tokens:
            events.append((offset + float(run.final_wallclock_seconds), int(run.final_generated_tokens)))
    if not events:
        return np.array([0.0]), np.array([0.0])
    events.sort(key=lambda e: e[0])
    xs = np.empty(len(events) + 1, dtype=np.float64)
    ys = np.empty(len(events) + 1, dtype=np.float64)
    xs[0] = 0.0
    ys[0] = 0.0
    cum = 0
    for i, (t, n) in enumerate(events, start=1):
        cum += n
        xs[i] = t
        ys[i] = float(cum)
    return xs, ys


def _render_tokens_vs_wallclock_png(
    benchmarks: list[taaf.benchmark.Benchmark],
    title: str,
    *,
    labels: list[str] | None = None,
    palette: list[str] | None = None,
) -> str:
    """One curve per benchmark — single-curve on per-run pages, multi-
    curve on comparison pages (pass ``labels`` + ``RUN_COLORS``)."""
    return _render_curves_png(
        [_tokens_vs_wallclock_curve(b) for b in benchmarks],
        x_label="job wallclock (s)",
        y_label="cumulative generated tokens",
        title=title,
        labels=labels,
        palette=palette,
    )


def _capped_pass_curve(
    pass_runs: list[taaf.game.GameRun],
    x_fn: Callable[[taaf.game.GameRun], float],
    level_cap: float,
    *,
    scorer: Scorer = ARC_SCORER,
    weights_by_game_id: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``_pass_curve`` with each lane truncated by a per-level cap.
    ``cap = max-observed-per-level`` reproduces the unrestricted curve
    exactly.
    """
    if not pass_runs:
        return np.array([0.0]), np.array([0.0])
    capped = [_per_game_partial_curve(r, x_fn, level_cap, scorer=scorer) for r in pass_runs]
    lane_weights = (
        [weights_by_game_id.get(r.game_id, 1.0) for r in pass_runs] if weights_by_game_id is not None else None
    )
    return _pool_lane_events(capped, weights=lane_weights)


def _lane_score_at_budget(run: taaf.game.GameRun, budget: float, scorer: Scorer) -> float:
    """Score this single ``(game, pass)`` lane reached after spending
    ``budget`` generated tokens, read off its score-vs-tokens step
    curve. A lane that finished under budget keeps its terminal score.
    """
    events = _per_game_partial_curve(run, _total_tokens, scorer=scorer)
    total = events[-1][0]
    if budget >= total - 1e-9:
        return _score_for(run, scorer)
    cur = 0.0
    for cx, cy in events:
        if cx <= budget + 1e-9:
            cur = cy
        else:
            break
    return cur


def _capped_score_fn(scorer: Scorer, ceiling: float) -> Callable[[taaf.game.GameRun], float]:
    """Per-run score callable that caps every lane at a shared per-game
    token ``ceiling`` (see ``_lane_score_at_budget``). ``ceiling`` is the
    parallel-cutoff point from ``_global_ceiling_for_budget``, not the
    displayed budget itself."""

    def _f(run: taaf.game.GameRun) -> float:
        return _lane_score_at_budget(run, ceiling, scorer)

    return _f


def _min_tokens_per_game_budget(benchmarks: list[taaf.benchmark.Benchmark]) -> float:
    """Common per-game token budget for the capped statistics: the
    smallest across runs of each run's weighted-mean generated tokens
    per ``(game, pass)`` lane. This is the rightmost x at which every
    run's score-vs-tokens curve still has data — past it, the shortest
    run can't be read, so it's the fairest shared budget."""
    per_run: list[float] = []
    for b in benchmarks:
        weights = _weights_by_game_id(b)
        kept = [(r, weights.get(r.game_id, 1.0)) for r in b.game_runs if weights.get(r.game_id, 1.0) > 0]
        if not kept:
            continue
        total_w = sum(w for _, w in kept)
        per_run.append(sum(_total_tokens(r) * w for r, w in kept) / total_w)
    return min(per_run) if per_run else 0.0


def _global_ceiling_for_budget(benchmarks: list[taaf.benchmark.Benchmark], budget: float) -> float:
    """Per-game token *ceiling* that realizes the common ``budget`` as a
    shared cutoff. Mental model: every ``(game, pass)`` lane across all
    runs spends tokens in parallel, and we freeze the instant the
    weighted-mean spend reaches ``budget``. The freeze point is this
    ceiling — capping every lane at it makes the average lane spend exactly
    ``budget`` while leaving lanes that finished cheaper untouched and
    chopping only the expensive ones.

    Capping directly at ``budget`` would undershoot: cheap lanes spend less
    than ``budget`` with nothing to offset them, so the average falls below
    it and heterogeneous runs get understated. Returns ``inf`` when
    ``budget`` already meets or exceeds the grand mean (nothing to cap).
    """
    lanes: list[tuple[float, float]] = []
    for b in benchmarks:
        weights = _weights_by_game_id(b)
        for r in b.game_runs:
            w = weights.get(r.game_id, 1.0)
            if w > 0:
                lanes.append((float(_total_tokens(r)), w))
    if not lanes:
        return 0.0
    total_w = sum(w for _, w in lanes)
    target_sum = budget * total_w  # solve Σ w·min(C, T) == target_sum for C
    if target_sum >= sum(t * w for t, w in lanes) - 1e-9:
        return float("inf")
    lanes.sort()
    sum_fixed = 0.0  # Σ w·T over lanes already below the ceiling
    w_tail = total_w  # weight of lanes still at or above the ceiling
    for t, w in lanes:
        # For C in [prev t, this t], Σ w·min(C, T) = sum_fixed + C·w_tail,
        # which peaks at C = t; the target lands in the first such interval.
        if sum_fixed + t * w_tail >= target_sum:
            return (target_sum - sum_fixed) / w_tail
        sum_fixed += w * t
        w_tail -= w
    return float("inf")  # unreachable given the grand-mean guard above


# --- PNG rendering ----------------------------------------------------------


def _fig_to_png_b64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frame_png_b64(frame: taaf.game.Frame) -> str:
    """Native-resolution PNG of an ARC frame; CSS handles zoom."""
    h, w = frame.data.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG_COLOR)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(frame.data, cmap=taaf.game.ARC_CMAP, vmin=0, vmax=15, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return _fig_to_png_b64(fig)


def _thumbnail_html(run: taaf.game.GameRun | None, kind: str, width_px: int = THUMB_DEFAULT_PX) -> str:
    """Inline thumbnail. ``kind`` is 'initial' or 'final'. Empty string if unavailable.

    Real ARC-AGI3 games typically render a meaningful WIN frame (the
    final board position), so we just take ``intermediate_states[-1]``
    for the final thumbnail. Example games that emit a flat
    celebration-color frame on win are expected to render their natural
    final gameplay position instead.
    """
    if run is None or not run.intermediate_states:
        return ""
    state = run.intermediate_states[0] if kind == "initial" else run.intermediate_states[-1]
    b64 = _frame_png_b64(state.frame)
    return f'<img class="pixelart" width="{width_px}" src="data:image/png;base64,{b64}" alt="{kind}">'


def _render_curves_png(
    curves: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    labels: list[str] | None = None,
    palette: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
    overlay_curves: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None,
) -> str:
    """Multi-curve step plot. ``palette`` selects the color cycle
    (default ``ACCENT_COLORS``; comparison plots use ``RUN_COLORS``).
    ``overlay_curves`` are drawn faint under the main curves — used to
    show per-pass variance under a pooled curve without losing the
    pooled curve's budget interpretation. No legend entry for overlays.
    """
    pal = palette if palette is not None else ACCENT_COLORS
    fig = plt.figure(figsize=figsize, dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    if overlay_curves:
        for xs, ys in overlay_curves:
            ax.step(xs, ys, where="post", color=pal[0], alpha=0.45, linewidth=0.7, zorder=1)
    for i, (xs, ys) in enumerate(curves):
        label = labels[i] if labels else None
        ax.step(xs, ys, where="post", color=pal[i % len(pal)], label=label, linewidth=1.8, zorder=3)
    _style_dark(ax, x_label, y_label, title)
    return _fig_to_png_b64(fig)


def _draw_labelled_points(
    ax: matplotlib.axes.Axes,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    color: str = ACCENT_COLORS[0],
) -> None:
    """Scatter labelled points in the dark-theme style. Shared by per-game
    score-vs-tokens (per-run) and head-to-head per-game (comparison)."""
    ax.scatter(xs, ys, color=color, s=70, edgecolors=TEXT_COLOR, linewidths=0.5, zorder=3)
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", color=TEXT_COLOR, fontsize=8)


def _render_scatter_png(
    per_game_a: dict[str, float],
    per_game_b: dict[str, float],
    a_label: str,
    b_label: str,
    title: str,
) -> str:
    fig = plt.figure(figsize=(6, 6), dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    games = sorted(set(per_game_a) & set(per_game_b))
    xs = [per_game_a[g] for g in games]
    ys = [per_game_b[g] for g in games]
    _draw_labelled_points(ax, xs, ys, games)
    bound = max(max(xs, default=0.0), max(ys, default=0.0), 1.0) * 1.05
    ax.plot([0, bound], [0, bound], color=GRID_COLOR, linestyle="--", linewidth=1, label="y = x", zorder=1)
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_aspect("equal")
    _style_dark(ax, a_label, b_label, title)
    return _fig_to_png_b64(fig)


def _render_per_game_score_vs_tokens_scatter_png(benchmark: taaf.benchmark.Benchmark, title: str) -> str:
    """One labelled point per game: x = mean tokens across passes, y = mean
    score across passes. Complements the pooled-lane curve by exposing which
    games are cheap-and-easy vs. expensive-and-hard."""
    fig = plt.figure(figsize=(7, 5), dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    games = _game_ids(benchmark)
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for g in games:
        runs = _game_runs_by_id(benchmark, g)
        if not runs:
            continue
        xs.append(float(statistics.mean(_total_tokens(r) for r in runs)))
        ys.append(float(statistics.mean(_live_score(r) for r in runs)))
        labels.append(g)
    _draw_labelled_points(ax, xs, ys, labels)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    _style_dark(ax, "mean tokens per pass", "mean score per pass", title)
    return _fig_to_png_b64(fig)


def _style_dark(ax: matplotlib.axes.Axes, x_label: str, y_label: str, title: str) -> None:
    ax.set_xlabel(x_label, color=TEXT_COLOR)
    ax.set_ylabel(y_label, color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Anchor legend outside the axes to avoid overlapping curves;
        # bbox_inches="tight" extends the figure bbox to include it.
        legend = ax.legend(
            handles,
            labels,
            facecolor=BG_COLOR,
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
        )
        legend.get_frame().set_alpha(0.9)


# --- Movie rendering --------------------------------------------------------


def _arc_to_rgb(data: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    palette = np.array([taaf.game.ARC_COLORS[i] for i in range(16)], dtype=np.float32)
    rgb = (palette[data] * 255).astype(np.uint8)
    if MP4_UPSCALE > 1:
        # Nearest-neighbor block-tile; preserves color exactly.
        rgb = rgb.repeat(MP4_UPSCALE, axis=0).repeat(MP4_UPSCALE, axis=1)
    return rgb


def _render_run_mp4(run: taaf.game.GameRun, out_path: Path, fps: int = MOVIE_FPS) -> bool:
    """Write an MP4 of the run's gameplay. Returns False (no file
    written) when ``intermediate_states`` has < 2 frames (e.g. after a
    from_json without the sidecar). One frame per state — animation
    frames between states are skipped (they multiply length without
    adding analytical value). Over ``MAX_MOVIE_FRAMES`` total states,
    states are subsampled at equal index spacing (first and last
    preserved, with integer-rounding discretisation in between).
    """
    n = len(run.intermediate_states)
    if n < 2:
        return False
    states: list[taaf.game.GameState]
    if n <= MAX_MOVIE_FRAMES:
        states = list(run.intermediate_states)
    else:
        idx: list[int] = np.linspace(0, n - 1, MAX_MOVIE_FRAMES).round().astype(int).tolist()
        states = [run.intermediate_states[i] for i in idx]
    frames: list[npt.NDArray[np.uint8]] = [_arc_to_rgb(state.frame.data) for state in states]
    if len(frames) < 2:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1: encode at native dims, no 16-pixel padding
    # (H.264 still requires even dims — MP4_UPSCALE handles that).
    # +faststart moves the moov atom to file start so browsers' seek
    # bars work (ffmpeg writes it at the end by default).
    iio.imwrite(
        out_path,
        np.stack(frames),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    return True


# --- Statistical test (R4.12) -----------------------------------------------


def _pass_level_score_test(
    bm_a: taaf.benchmark.Benchmark,
    bm_b: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, Any]:
    """Welch's (unpaired) t-test on per-pass mean scores. Each pass is
    one observation (mean per-game score in that pass).

    Pass-level rather than per-game pairing because solvers tend to
    excel on different games — there's no shared game ranking, so
    per-game pairing wouldn't cancel noise.

    Returns NaN ``t`` / ``p`` / ``df`` when either side has < 2
    passes. At n = 2 per side Welch df ≈ 1 — essentially powerless;
    bump n_passes to 3+ for a usable p. ``score_fn`` overrides the
    per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    weights_a = _weights_by_game_id(bm_a)
    weights_b = _weights_by_game_id(bm_b)
    pass_means_a: list[float] = []
    pass_means_b: list[float] = []
    for p in range(bm_a.n_passes):
        kept = [(r, weights_a.get(r.game_id, 1.0)) for r in _pass_runs(bm_a, p) if weights_a.get(r.game_id, 1.0) > 0]
        if kept:
            total_w = sum(w for _, w in kept)
            pass_means_a.append(sum(sf(r) * w for r, w in kept) / total_w)
    for p in range(bm_b.n_passes):
        kept = [(r, weights_b.get(r.game_id, 1.0)) for r in _pass_runs(bm_b, p) if weights_b.get(r.game_id, 1.0) > 0]
        if kept:
            total_w = sum(w for _, w in kept)
            pass_means_b.append(sum(sf(r) * w for r, w in kept) / total_w)
    overall_a = float(np.mean(pass_means_a)) if pass_means_a else 0.0
    overall_b = float(np.mean(pass_means_b)) if pass_means_b else 0.0
    if len(pass_means_a) < 2 or len(pass_means_b) < 2:
        return {
            "mean_a": overall_a,
            "mean_b": overall_b,
            "mean_diff": overall_b - overall_a,
            "t": float("nan"),
            "p": float("nan"),
            "df": float("nan"),
            "n_passes_a": len(pass_means_a),
            "n_passes_b": len(pass_means_b),
            "zero_variance": False,
        }
    # Zero sample variance on either side: Welch's can't estimate that
    # side's noise. scipy returns t = ±inf / p ∈ {0, NaN} which reads
    # like a real significance estimate. Flag so the renderer can say
    # "not meaningful" instead of showing the bogus number.
    zero_variance = float(np.var(pass_means_a, ddof=1)) == 0.0 or float(np.var(pass_means_b, ddof=1)) == 0.0
    res: Any = scipy.stats.ttest_ind(pass_means_b, pass_means_a, equal_var=False)
    return {
        "mean_a": overall_a,
        "mean_b": overall_b,
        "mean_diff": overall_b - overall_a,
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "df": float(res.df) if hasattr(res, "df") else float("nan"),
        "n_passes_a": len(pass_means_a),
        "n_passes_b": len(pass_means_b),
        "zero_variance": zero_variance,
    }


def _weighted_paired_t_test(diffs: list[float], weights: list[float]) -> dict[str, Any]:
    """Weighted paired t-test on ``diffs`` with parallel per-pair
    ``weights``. Weights are treated as importance / reliability, not
    frequencies — each pair is one independent observation; weights
    rescale contribution but don't multiply sample size. Using
    ``sum(w) - 1`` as df would overstate certainty and produce
    p-values that disagree with the paired permutation test.

    Formulae::

        mean_w = sum(w·d) / sum(w)
        var_w  = sum(w·(d − mean_w)²) / (sum(w) − sum(w²)/sum(w))
                                          # analytic-weight Bessel
                                          # analogue, scale-invariant
                                          # under uniform rescaling
        n_eff  = (sum w)² / sum(w²)       # Kish effective N
        SE     = √(var_w / n_eff)
        t      = mean_w / SE
        df     = n_eff − 1
        p      = 2·P(T > |t|),  T ~ Student-t(df)

    With all weights 1.0, ``n_eff = n`` and this reduces to
    ``scipy.stats.ttest_rel`` (asserted by
    ``test_weighted_paired_t_matches_scipy_when_uniform``).
    """
    if len(diffs) != len(weights):
        raise ValueError(f"diffs length {len(diffs)} != weights length {len(weights)}")
    n = len(diffs)
    out_nan = {
        "t": float("nan"),
        "p": float("nan"),
        "df": float("nan"),
        "n_pairs": n,
        "zero_variance": False,
    }
    if n < 2:
        return out_nan
    d = np.asarray(diffs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    sw = float(np.sum(w))
    sw2 = float(np.sum(w * w))
    if sw <= 0.0 or sw2 <= 0.0:
        return out_nan
    n_eff = sw * sw / sw2
    if n_eff <= 1.0:  # df = n_eff - 1 ≤ 0 — undefined
        return out_nan
    mean_w = float(np.sum(w * d) / sw)
    # ``sw - sw2/sw = (sw² - sw2)/sw`` equals n-1 when w_i = 1 and is
    # invariant under uniform rescaling (doubling every weight leaves t).
    denom = sw - sw2 / sw
    if denom <= 0.0:  # only one nonzero weight — undefined
        return out_nan
    var_w = float(np.sum(w * (d - mean_w) ** 2) / denom)
    if var_w == 0.0:
        return {**out_nan, "zero_variance": True}
    se = (var_w / n_eff) ** 0.5
    t = mean_w / se
    df = n_eff - 1.0
    p = 2.0 * float(scipy.stats.t.sf(abs(t), df))  # pyright: ignore[reportUnknownArgumentType]
    return {"t": t, "p": p, "df": df, "n_pairs": n, "zero_variance": False}


# 2^20 ≈ 1M sign-flips runs in a vectorised numpy step in well under
# a second; beyond that we switch to Monte Carlo.
_PERMUTATION_EXACT_MAX_N = 20
_PERMUTATION_MC_SAMPLES = 10000


def _weighted_paired_permutation_test(diffs: list[float], weights: list[float]) -> dict[str, Any]:
    """Weighted paired sign-flip permutation test on ``diffs`` with
    parallel per-pair ``weights``. Statistic
    ``T = sum(w·d) / sum(w)``; under the exchangeability null each
    sign is equally likely. Exact enumeration of all 2^n sign-flips
    for ``n ≤ _PERMUTATION_EXACT_MAX_N``, else Monte Carlo with the
    ``(count+1)/(N+1)`` correction. Two-sided
    ``p = P(|T_permuted| ≥ |T_observed|)``.

    Reduces to the standard sign-flip test when weights are uniform
    (asserted by ``test_weighted_permutation_matches_explicit_when_uniform``).
    """
    if len(diffs) != len(weights):
        raise ValueError(f"diffs length {len(diffs)} != weights length {len(weights)}")
    n = len(diffs)
    if n < 2:
        return {
            "p": float("nan"),
            "n_pairs": n,
            "exact": False,
            "n_permutations": 0,
            "zero_variance": False,
        }
    d = np.asarray(diffs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return {
            "p": float("nan"),
            "n_pairs": n,
            "exact": False,
            "n_permutations": 0,
            "zero_variance": False,
        }
    zero_variance = bool(np.all(d == 0.0))
    observed = float(np.sum(w * d) / sw)
    abs_obs = abs(observed)
    if zero_variance:
        return {
            "p": 1.0,
            "n_pairs": n,
            "exact": True,
            "n_permutations": 1 << n,
            "zero_variance": True,
        }
    if n <= _PERMUTATION_EXACT_MAX_N:
        masks = np.arange(1 << n, dtype=np.int64)[:, None]
        bits = (masks >> np.arange(n)) & 1
        signs = 1 - 2 * bits  # (2^n, n)
        permuted_means = (signs * w * d).sum(axis=1) / sw
        count_as_extreme = int(np.sum(np.abs(permuted_means) >= abs_obs - 1e-12))
        n_perms = 1 << n
        p = count_as_extreme / n_perms
        return {
            "p": p,
            "n_pairs": n,
            "exact": True,
            "n_permutations": n_perms,
            "zero_variance": False,
        }
    # Fixed seed derived from the observed statistic so the reported p
    # is reproducible across regens, with distinct seeds across datasets.
    rng = np.random.default_rng(abs(hash(observed)) & 0xFFFFFFFF)
    signs = rng.choice(np.array([-1, 1], dtype=np.float64), size=(_PERMUTATION_MC_SAMPLES, n))
    permuted_means = (signs * w * d).sum(axis=1) / sw
    count_as_extreme = int(np.sum(np.abs(permuted_means) >= abs_obs - 1e-12))
    p = (count_as_extreme + 1) / (_PERMUTATION_MC_SAMPLES + 1)
    return {
        "p": p,
        "n_pairs": n,
        "exact": False,
        "n_permutations": _PERMUTATION_MC_SAMPLES,
        "zero_variance": False,
    }


def _paired_score_test(
    bm_a: taaf.benchmark.Benchmark,
    bm_b: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, Any]:
    """Weighted paired t-test on per-game mean scores. One pair per
    common game (mean across passes under each solver), per-pair weight
    averaged from the two benchmarks' ``game_weights``. Weight-0 games
    are excluded. Same null hypothesis as ``_pass_level_score_test``;
    per-game pairing wins when game difficulty correlates across
    solvers, pass-level wins otherwise. ``score_fn`` overrides the
    per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    common = sorted(set(_game_ids(bm_a)) & set(_game_ids(bm_b)))
    weights_a = _weights_by_game_id(bm_a)
    weights_b = _weights_by_game_id(bm_b)
    diffs: list[float] = []
    pair_weights: list[float] = []
    for g in common:
        wa = weights_a.get(g, 1.0)
        wb = weights_b.get(g, 1.0)
        if wa == 0.0 or wb == 0.0:
            continue
        a_scores = [sf(r) for r in _game_runs_by_id(bm_a, g)]
        b_scores = [sf(r) for r in _game_runs_by_id(bm_b, g)]
        if not a_scores or not b_scores:
            continue
        diffs.append(statistics.mean(b_scores) - statistics.mean(a_scores))
        # Per-pair weight averages the two sides; identical when both
        # benchmarks carry the same game_weights (the common case).
        pair_weights.append((wa + wb) / 2.0)
    result = _weighted_paired_t_test(diffs, pair_weights)
    return {
        "t": result["t"],
        "p": result["p"],
        "df": result["df"],
        "n_games": result["n_pairs"],
        "zero_variance": result["zero_variance"],
    }


def _paired_permutation_test(
    bm_a: taaf.benchmark.Benchmark,
    bm_b: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, Any]:
    """Weighted paired sign-flip permutation test on per-game mean
    differences. Same per-game pairing as ``_paired_score_test``;
    robust to non-Gaussian / heavy-tailed / skewed differences where
    the paired t-test's normality assumption is suspect. ``score_fn``
    overrides the per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    common = sorted(set(_game_ids(bm_a)) & set(_game_ids(bm_b)))
    weights_a = _weights_by_game_id(bm_a)
    weights_b = _weights_by_game_id(bm_b)
    diffs: list[float] = []
    pair_weights: list[float] = []
    for g in common:
        wa = weights_a.get(g, 1.0)
        wb = weights_b.get(g, 1.0)
        if wa == 0.0 or wb == 0.0:
            continue
        a_scores = [sf(r) for r in _game_runs_by_id(bm_a, g)]
        b_scores = [sf(r) for r in _game_runs_by_id(bm_b, g)]
        if not a_scores or not b_scores:
            continue
        diffs.append(statistics.mean(b_scores) - statistics.mean(a_scores))
        pair_weights.append((wa + wb) / 2.0)
    result = _weighted_paired_permutation_test(diffs, pair_weights)
    return {
        "p": result["p"],
        "n_games": result["n_pairs"],
        "exact": result["exact"],
        "n_permutations": result["n_permutations"],
        "zero_variance": result["zero_variance"],
    }


def _welch_ok(res: dict[str, Any]) -> bool:
    """Welch's t-test is reportable only with ≥ 2 passes on each side."""
    return res["n_passes_a"] >= 2 and res["n_passes_b"] >= 2


def _paired_ok(res: dict[str, Any]) -> bool:
    """The paired tests need ≥ 2 common games to have any signal."""
    return res["n_games"] >= 2


def _direction_arrow(
    mean_row: float | None,
    mean_col: float | None,
    row_label: str,
    col_label: str,
) -> tuple[str, str]:
    """``(arrow, winner_label)`` pointing toward the higher-scoring run:
    ``←`` when this cell's row run scores higher, ``↑`` when its column
    run does. Empty when tied or either mean is unknown."""
    if mean_row is None or mean_col is None or mean_row == mean_col:
        return "", ""
    return ("←", row_label) if mean_row > mean_col else ("↑", col_label)


def _pvalue_cell_html(res: dict[str, Any], ok: bool, *, arrow: str = "", winner: str = "") -> str:
    """One ``<td>`` for a pairwise p-value matrix: ``n/a`` when the test
    is unreportable (too few passes/games), ``n.m.`` when a zero-variance
    side makes the p meaningless, else the two-sided p (bold if < 0.05).
    An optional direction ``arrow`` (← row run higher, ↑ column run
    higher) points toward the better-scoring run, ``winner`` labels it."""
    prefix = f'<span class="pval-arrow" title="{escape(winner)} scores higher">{arrow}</span> ' if arrow else ""
    if not ok:
        return f'<td class="pval-na" title="too few passes / games for this test">{prefix}n/a</td>'
    if res.get("zero_variance"):
        return f'<td class="pval-na" title="zero sample variance — p not meaningful">{prefix}n.m.</td>'
    p = res["p"]
    if np.isnan(p):
        return f'<td class="pval-na" title="undefined">{prefix}n/a</td>'
    cls = "pval-sig" if p < 0.05 else "pval"
    return f'<td class="{cls}">{prefix}{p:.4f}</td>'


def _render_pvalue_matrix(
    title: str,
    labels: list[str],
    results: dict[tuple[int, int], dict[str, Any]],
    ok_fn: Callable[[dict[str, Any]], bool],
    means: list[float | None],
) -> str:
    """One symmetric NxN p-value matrix. ``results`` is keyed by the
    upper-triangle pair ``(i, j)`` with ``i < j``; the two-sided p is
    symmetric so cell ``(j, i)`` reuses it and the diagonal is blank.
    ``means`` (per-run mean score, parallel to ``labels``) drives the
    per-cell direction arrow toward the higher-scoring run."""
    header_cells = "".join(f"<th><code>{escape(lbl)}</code></th>" for lbl in labels)
    rows = [f"<tr><th></th>{header_cells}</tr>"]
    for i, li in enumerate(labels):
        cells = [f"<th><code>{escape(li)}</code></th>"]
        for j in range(len(labels)):
            if i == j:
                cells.append('<td class="pval-diag">—</td>')
                continue
            res = results[(i, j) if i < j else (j, i)]
            arrow, winner = _direction_arrow(means[i], means[j], labels[i], labels[j])
            cells.append(_pvalue_cell_html(res, ok_fn(res), arrow=arrow, winner=winner))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<h4>{title}</h4><table class="pmatrix">{"".join(rows)}</table>'


def _pvalue_matrices_inner(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> str:
    """R4.12 (>2 runs): the same three tests as the 2-run head-to-head,
    each rendered as a symmetric NxN matrix of pairwise two-sided
    p-values. Returns the legend + matrices; the caller wraps it in a
    box. ``score_fn`` selects endpoint vs common-budget scoring."""
    n = len(benchmarks)
    # Tests are two-sided ⇒ p is symmetric in the pair; compute the
    # upper triangle once and mirror it in the renderer.
    welch: dict[tuple[int, int], dict[str, Any]] = {}
    paired: dict[tuple[int, int], dict[str, Any]] = {}
    perm: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            welch[(i, j)] = _pass_level_score_test(benchmarks[i], benchmarks[j], scorer=scorer, score_fn=score_fn)
            paired[(i, j)] = _paired_score_test(benchmarks[i], benchmarks[j], scorer=scorer, score_fn=score_fn)
            perm[(i, j)] = _paired_permutation_test(benchmarks[i], benchmarks[j], scorer=scorer, score_fn=score_fn)
    # Per-run mean score (same scoring as the cells) drives the direction
    # arrow toward the higher-scoring run in each cell.
    means = [_run_pass_stats(b, scorer=scorer, score_fn=score_fn)["mean"] for b in benchmarks]
    matrices = (
        _render_pvalue_matrix("Welch's t-test — per-pass mean scores", labels, welch, _welch_ok, means)
        + _render_pvalue_matrix("Paired t-test — per-game mean scores", labels, paired, _paired_ok, means)
        + _render_pvalue_matrix("Paired permutation test — per-game differences", labels, perm, _paired_ok, means)
    )
    return (
        "<p>Each test asks whether two runs differ in mean per-game score — same null "
        "hypothesis, different variance / null structure (see the 2-run page for the full "
        "explanation). Cells are two-sided p-values; each matrix is symmetric and the "
        "diagonal is blank. The arrow points to the higher-scoring run "
        '(<span class="pval-arrow">←</span> this row, <span class="pval-arrow">↑</span> this '
        "column). <strong>Bold</strong> marks p &lt; 0.05. "
        "<code>n.m.</code> = not meaningful (zero variance on a side); "
        "<code>n/a</code> = too few passes / games.</p>"
        f"{matrices}"
    )


def _fmt_stat(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def _two_run_stats_inner(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> str:
    """Head-to-head inner content for exactly two runs: per-run mean
    table (winner bolded with 🏆), the winner line, and the three
    p-value test lines. Returns inner HTML; the caller wraps it in a
    box. ``score_fn`` selects endpoint vs common-budget scoring."""
    result = _pass_level_score_test(benchmarks[0], benchmarks[1], scorer=scorer, score_fn=score_fn)
    stats_a = _run_pass_stats(benchmarks[0], scorer=scorer, score_fn=score_fn)
    stats_b = _run_pass_stats(benchmarks[1], scorer=scorer, score_fn=score_fn)
    a_label, b_label = labels[0], labels[1]
    a_mean, b_mean = result["mean_a"], result["mean_b"]

    # Highlight the winner with bold + 🏆; tie if exactly equal.
    if a_mean > b_mean:
        winner_text = f"<strong>{escape(a_label)}</strong> ahead by {a_mean - b_mean:.2f} points"
        row_a_score = f"<strong>{a_mean:.2f}</strong> 🏆"
        row_b_score = f"{b_mean:.2f}"
    elif b_mean > a_mean:
        winner_text = f"<strong>{escape(b_label)}</strong> ahead by {b_mean - a_mean:.2f} points"
        row_a_score = f"{a_mean:.2f}"
        row_b_score = f"<strong>{b_mean:.2f}</strong> 🏆"
    else:
        winner_text = "Tied."
        row_a_score = f"{a_mean:.2f}"
        row_b_score = f"{b_mean:.2f}"

    # Welch's needs n ≥ 2 per side; n = 2 gives df ≈ 1 and is
    # effectively powerless. Zero variance on either side makes p
    # meaningless — we report that distinctly rather than letting
    # p = 0.0000 look real.
    n_a, n_b = result["n_passes_a"], result["n_passes_b"]
    if n_a >= 2 and n_b >= 2:
        if result.get("zero_variance"):
            welch_line = (
                f"<li><strong>Welch's t-test on per-pass mean scores</strong> "
                f"(n_passes = {n_a} vs {n_b}): "
                f"<strong>not meaningful</strong> — at least one side has zero "
                f"sample variance across passes (deterministic solver, or all "
                f"passes produced identical mean scores).</li>"
            )
        else:
            welch_caveat = (
                " <em>With only 2 passes per side, df ≈ 1 — essentially powerless. "
                "Re-run with more passes for a usable p.</em>"
                if n_a == 2 or n_b == 2
                else ""
            )
            welch_line = (
                f"<li><strong>Welch's t-test on per-pass mean scores</strong> "
                f"(each pass treated as one independent observation; "
                f"n_passes = {n_a} vs {n_b}, df = {result['df']:.2f}, t = {result['t']:.3f}): "
                f"<strong>p = {result['p']:.4f}</strong>.{welch_caveat}</li>"
            )
    else:
        welch_line = (
            f"<li><strong>Welch's t-test on per-pass mean scores</strong>: "
            f"too few passes ({n_a} vs {n_b}) — need ≥ 2 per side.</li>"
        )

    # Same null hypothesis as Welch's but per-game pairing — wins when
    # the two solvers correlate across games (similar relative difficulty).
    paired = _paired_score_test(benchmarks[0], benchmarks[1], scorer=scorer, score_fn=score_fn)
    if paired["n_games"] >= 2:
        if paired.get("zero_variance"):
            paired_line = (
                f"<li><strong>Paired t-test on per-game mean scores</strong> "
                f"(n_games = {paired['n_games']}): "
                f"<strong>not meaningful</strong> — the per-game differences "
                f"have zero variance across games (identical mean scores per "
                f"game between the two solvers, so the paired test has no "
                f"signal to estimate).</li>"
            )
        else:
            paired_line = (
                f"<li><strong>Paired t-test on per-game mean scores</strong> "
                f"(pairs each game's mean under A with its mean under B; "
                f"n_games = {paired['n_games']}, df = {paired['df']:.0f}, t = {paired['t']:.3f}): "
                f"<strong>p = {paired['p']:.4f}</strong>.</li>"
            )
    else:
        paired_line = (
            f"<li><strong>Paired t-test on per-game mean scores</strong>: "
            f"too few games ({paired['n_games']}) — need ≥ 2.</li>"
        )

    # Robust to non-Gaussian / heavy-tailed per-game differences where
    # the paired t-test's normality assumption is suspect.
    permutation = _paired_permutation_test(benchmarks[0], benchmarks[1], scorer=scorer, score_fn=score_fn)
    if permutation["n_games"] >= 2:
        if permutation.get("zero_variance"):
            permutation_line = (
                f"<li><strong>Paired permutation test on per-game mean scores</strong> "
                f"(n_games = {permutation['n_games']}): "
                f"<strong>not meaningful</strong> — every per-game difference is zero, "
                f"so every sign-flip yields the same statistic as the observed.</li>"
            )
        else:
            method = (
                f"exact, {permutation['n_permutations']} sign-flips"
                if permutation["exact"]
                else f"Monte Carlo, {permutation['n_permutations']} samples"
            )
            permutation_line = (
                f"<li><strong>Paired permutation test on per-game mean scores</strong> "
                f"(sign-flip null on per-game differences; {method}; "
                f"n_games = {permutation['n_games']}): "
                f"<strong>p = {permutation['p']:.4f}</strong>.</li>"
            )
    else:
        permutation_line = (
            f"<li><strong>Paired permutation test on per-game mean scores</strong>: "
            f"too few games ({permutation['n_games']}) — need ≥ 2.</li>"
        )

    test_line = (
        "<p>Three tests below ask the same question — does the mean per-game score differ "
        "between the two runs? They differ in the variance / null structure they assume:</p>"
        f"<ul>{welch_line}{paired_line}{permutation_line}</ul>"
    )
    sigma_caveat = (
        ""
        if min(stats_a["n_passes"] or 0, stats_b["n_passes"] or 0) >= 2
        else "<p><em>σ across passes is undefined for runs with only one pass.</em></p>"
    )
    return (
        "<table>"
        "<tr><th>run</th><th>mean per-game score</th>"
        "<th>σ across passes</th><th>σ/√N (SEM)</th></tr>"
        f"<tr><td><code>{escape(a_label)}</code></td><td>{row_a_score}</td>"
        f"<td>{_fmt_stat(stats_a['sigma'])}</td><td>{_fmt_stat(stats_a['sem'])}</td></tr>"
        f"<tr><td><code>{escape(b_label)}</code></td><td>{row_b_score}</td>"
        f"<td>{_fmt_stat(stats_b['sigma'])}</td><td>{_fmt_stat(stats_b['sem'])}</td></tr>"
        "</table>"
        f"<p>{winner_text}.</p>"
        f"{sigma_caveat}"
        f"{test_line}"
    )


def _comparison_stats_box(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None,
    heading: str,
    note: str,
) -> str:
    """One score-comparison summary box for a given per-run score
    function. Two runs → head-to-head; more than two → pairwise p-value
    matrices. ``heading`` / ``note`` distinguish the endpoint and
    common-budget variants rendered side by side per scorer."""
    if len(benchmarks) == 2:
        inner = _two_run_stats_inner(benchmarks, labels, scorer=scorer, score_fn=score_fn)
    else:
        inner = _pvalue_matrices_inner(benchmarks, labels, scorer=scorer, score_fn=score_fn)
    note_html = f"<p>{note}</p>" if note else ""
    return f'<div class="summary-box"><h3>{escape(heading)}</h3>{note_html}{inner}</div>'


def _per_run_stats_box(
    benchmarks: list[taaf.benchmark.Benchmark],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> str:
    """At-a-glance per-run mean / σ across passes / SEM box (all runs).
    ``score_fn`` selects endpoint vs common-budget scoring."""
    rows = ["<tr><th>run</th><th>passes</th><th>mean</th><th>σ across passes</th><th>SEM</th></tr>"]
    for b in benchmarks:
        s = _run_pass_stats(b, scorer=scorer, score_fn=score_fn)
        rows.append(
            f"<tr><td><code>{escape(b.label)}</code></td>"
            f"<td>{s['n_passes']}</td>"
            f"<td>{_fmt_stat(s['mean'])}</td>"
            f"<td>{_fmt_stat(s['sigma'])}</td>"
            f"<td>{_fmt_stat(s['sem'])}</td></tr>"
        )
    return '<div class="summary-box"><h3>Per-run stats</h3><table>' + "".join(rows) + "</table></div>"


# --- HTML helpers -----------------------------------------------------------


_CSS = """
body { background: #1e1e1e; color: #e0e0e0; font-family: -apple-system, system-ui, sans-serif;
       padding: 20px; max-width: 1400px; margin: 0 auto; line-height: 1.4; }
h1, h2, h3 { color: #ffffff; }
table { border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #3a3a3a; padding: 6px 10px; text-align: left; vertical-align: middle; }
th { background: #2a2a2a; }
.pixelart { image-rendering: pixelated; image-rendering: -moz-crisp-edges; image-rendering: crisp-edges;
            vertical-align: middle; }
a { color: #9bd1ff; }
a:visited { color: #c9b1ff; }
pre { background: #2a2a2a; padding: 10px; border-radius: 4px; overflow-x: auto; }
.game-name { display: inline-flex; align-items: center; gap: 6px; }
.summary-box { background: #2a2a2a; padding: 12px 16px; border-radius: 4px; margin: 12px 0; }
.game-link { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
             margin: 4px; background: #2a2a2a; border-radius: 4px; text-decoration: none; }
.intro { color: #b8b8b8; font-style: italic; margin: 4px 0 16px 0; max-width: 70ch; }
.design-note { background: #3a2a1e; border-left: 4px solid #ffb47a; padding: 10px 14px;
               margin: 12px 0; border-radius: 4px; }
.curve-note { background: #1e2e2a; border-left: 4px solid #74a98c; padding: 10px 14px;
              margin: 12px 0; border-radius: 4px; max-width: 80ch; }
.curve-note p { margin: 6px 0; }
.curve-note code { background: #2a3a36; padding: 1px 4px; border-radius: 2px; }
.in-flight-banner { background: #1e2e3a; border-left: 4px solid #5a92c8; padding: 12px 16px;
                    margin: 12px 0; border-radius: 4px; font-size: 1.05em; color: #cfe2f3; }
.weights-banner { background: #2a2a1e; border-left: 4px solid #f0d878; padding: 12px 16px;
                  margin: 12px 0; border-radius: 4px; color: #f4e8c0; }
.weights-banner p { margin: 4px 0; }
.weights-banner ul { margin: 4px 0 0 24px; padding: 0; }
/* Folder-tab score selector: pure-CSS radio + sibling-selector. The
   active tab visually merges into the panel below by matching the
   panel's background and hiding its bottom border. */
.scorer-tabs > input[type="radio"] { display: none; }
.scorer-tabs .tab-row { display: flex; border-bottom: 2px solid #5a92c8; margin: 20px 0 0 0; }
.scorer-tabs .tab-label { padding: 8px 18px; background: #2a2a2a; color: #b8b8b8;
                          border: 2px solid #3a3a3a; border-bottom: none;
                          border-radius: 6px 6px 0 0; margin-right: 4px;
                          margin-bottom: -2px; cursor: pointer; user-select: none;
                          font-weight: 500; }
.scorer-tabs .tab-label:hover { color: #cfe2f3; }
.scorer-tabs .tab-panel { display: none; padding: 16px 4px; }
.scorer-tabs > input#scorer-arc:checked ~ .tab-row label[for="scorer-arc"],
.scorer-tabs > input#scorer-weighted:checked ~ .tab-row label[for="scorer-weighted"],
.scorer-tabs > input#scorer-levels:checked ~ .tab-row label[for="scorer-levels"] {
    background: #1e1e1e; color: #ffffff; border-color: #5a92c8; border-bottom-color: #1e1e1e;
}
.scorer-tabs > input#scorer-arc:checked ~ #panel-arc,
.scorer-tabs > input#scorer-weighted:checked ~ #panel-weighted,
.scorer-tabs > input#scorer-levels:checked ~ #panel-levels { display: block; }
.scorer-tabs .scorer-desc { color: #b8b8b8; font-style: italic; margin: 4px 0 12px 0; max-width: 80ch; }
/* Two-level budget × scorer tab widget (R4.12 comparison statistics).
   A panel shows only when its budget tab AND scorer tab are both
   checked — the standard two-radio-group CSS-only selection. The budget
   row is amber, the scorer row blue, to read as nested. */
.bs-tabs > input[type="radio"] { display: none; }
.bs-tabs .tab-row { display: flex; flex-wrap: wrap; }
.bs-tabs .budget-row { border-bottom: 2px solid #e0a060; margin: 20px 0 0 0; }
.bs-tabs .scorer-row { border-bottom: 2px solid #5a92c8; margin: 14px 0 0 0; }
.bs-tabs .tab-label { padding: 8px 18px; background: #2a2a2a; color: #b8b8b8;
                      border: 2px solid #3a3a3a; border-bottom: none;
                      border-radius: 6px 6px 0 0; margin-right: 4px;
                      margin-bottom: -2px; cursor: pointer; user-select: none; font-weight: 500; }
.bs-tabs .tab-label:hover { color: #cfe2f3; }
.bs-tabs .tab-panel { display: none; padding: 16px 4px; }
.bs-tabs > #bs-b-full:checked ~ .budget-row label[for="bs-b-full"],
.bs-tabs > #bs-b-capped:checked ~ .budget-row label[for="bs-b-capped"] {
    background: #1e1e1e; color: #ffffff; border-color: #e0a060; border-bottom-color: #1e1e1e;
}
.bs-tabs > #bs-s-arc:checked ~ .scorer-row label[for="bs-s-arc"],
.bs-tabs > #bs-s-weighted:checked ~ .scorer-row label[for="bs-s-weighted"],
.bs-tabs > #bs-s-levels:checked ~ .scorer-row label[for="bs-s-levels"] {
    background: #1e1e1e; color: #ffffff; border-color: #5a92c8; border-bottom-color: #1e1e1e;
}
.bs-tabs > #bs-b-full:checked ~ #bs-s-arc:checked ~ #bs-p-full-arc,
.bs-tabs > #bs-b-full:checked ~ #bs-s-weighted:checked ~ #bs-p-full-weighted,
.bs-tabs > #bs-b-full:checked ~ #bs-s-levels:checked ~ #bs-p-full-levels,
.bs-tabs > #bs-b-capped:checked ~ #bs-s-arc:checked ~ #bs-p-capped-arc,
.bs-tabs > #bs-b-capped:checked ~ #bs-s-weighted:checked ~ #bs-p-capped-weighted,
.bs-tabs > #bs-b-capped:checked ~ #bs-s-levels:checked ~ #bs-p-capped-levels { display: block; }
.bs-tabs .scorer-desc { color: #b8b8b8; font-style: italic; margin: 4px 0 12px 0; max-width: 80ch; }
.per-game-grid { display: flex; flex-direction: column; gap: 12px; margin: 12px 0; }
.per-game-card { display: flex; align-items: center; gap: 12px; padding: 12px;
                 background: #2a2a2a; border-radius: 4px; flex-wrap: wrap; }
.per-game-card img { max-width: 360px; height: auto; border-radius: 4px; }
.per-game-header { min-width: 160px; }
/* Pairwise p-value matrices (>2-run comparison). */
.summary-box h4 { margin: 14px 0 4px 0; }
table.pmatrix td { text-align: center; }
td.pval-sig { font-weight: bold; color: #9be8a8; }
td.pval-na { color: #888; }
td.pval-diag { color: #666; }
.pval-arrow { color: #e0a060; font-weight: bold; }
"""

# Stamped on the wallclock plot only — the tokens plot has a clean
# budget interpretation (see ``_TOKENS_CURVE_NOTE``) but wallclock is
# trickier because games run in parallel during the benchmark, so the
# realised x-axis "wallclock per game" is not just ``total / n_games``.
_PLOT_DESIGN_NOTE = (
    '<div class="design-note">'
    "<strong>Design note:</strong> we still have to think about how to "
    "plot this exactly when we have some real benchmark data — games "
    "run in parallel during the benchmark, so wallclock per game is "
    "not just total wallclock / n_games."
    "</div>"
)

# Pinned next to every *cross-game* score-vs-tokens plot. Skipped on
# per-game plots, where the curve already means "this single game's
# expected score at budget X" without further explanation.
_TOKENS_CURVE_NOTE = (
    '<div class="curve-note">'
    "<p><strong>What this curve shows.</strong> Reading the curve at <em>x = X</em> "
    "tells you the expected score per "
    "game when the method is run at an average per-game budget of <em>X</em> "
    "tokens. Slow games may use more, fast games less; the average works "
    "out to <em>X</em>. </p>"
    "</div>"
)


def _is_in_flight(benchmark: taaf.benchmark.Benchmark) -> bool:
    """A benchmark is in-flight if ``run()`` started but hasn't finished yet
    (the periodic save loop generates diagnostics mid-run, R2.13). After
    teardown ``end_time`` is set in ``Benchmark.run``'s ``finally`` block, so
    ``end_time is None`` while ``start_time is not None`` is the unambiguous
    in-flight signal.
    """
    return benchmark.start_time is not None and benchmark.end_time is None


def _in_flight_banner_html(benchmark: taaf.benchmark.Benchmark) -> str:
    if not _is_in_flight(benchmark):
        return ""
    started = _format_timestamp(benchmark.start_time) if benchmark.start_time else "?"
    written = _format_timestamp(datetime.now())
    return (
        '<div class="in-flight-banner">'
        "⏳ <strong>Run in progress.</strong> This page was generated by the "
        f"periodic save loop while the benchmark is still executing (started <code>{started}</code>, "
        f"this snapshot written <code>{written}</code>). "
        "Numbers, plots, and movies reflect a partial snapshot — they will be "
        "rewritten when the run finishes."
        "</div>"
    )


def _html_page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        f"<style>{_CSS}</style></head>\n<body>\n{body}\n</body></html>\n"
    )


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def _game_name_html(run: taaf.game.GameRun, thumb_px: int = THUMB_DEFAULT_PX) -> str:
    thumb = _thumbnail_html(run, "initial", width_px=thumb_px)
    return f'<span class="game-name">{thumb}<code>{escape(run.game_id)}</code></span>'


def _write_movie_player_html(
    out_path: Path,
    mp4_name: str,
    title: str,
    solver_label: str,
    solver_analysis_html: str | None = None,
) -> None:
    """Write a small HTML wrapper that embeds an MP4 with playback controls
    and CSS-zoom rendering. The MP4 stays at native size on disk; the
    browser zooms it for display via ``image-rendering: pixelated``.

    ``solver_analysis_html`` (when set) is a path **relative to job_dir**
    (i.e. relative to the directory containing this wrapper's parent).
    The wrapper lives at ``{job_dir}/movies/{stem}.html``, so we render
    the link as ``../{solver_analysis_html}``.
    """
    analysis_html = ""
    if solver_analysis_html:
        # Wrapper is at {job_dir}/movies/{stem}.html; analysis path is
        # relative to job_dir → prepend "../".
        analysis_href = f"../{solver_analysis_html}"
        analysis_html = f'<p><a href="{analysis_href}">Detailed analysis by solver →</a></p>'
    body = (
        '<p><a href="../diagnostics.html">← back to diagnostics</a></p>'
        f"<h1>{escape(title)}</h1>"
        f"<p>Solver: <code>{escape(solver_label)}</code></p>"
        f"{analysis_html}"
        '<p class="intro">Recorded play-through of one (game, pass) cell. Encoded at native '
        "frame size (typically 64×64); your browser zooms it up here.</p>"
        f'<video class="pixelart" controls loop autoplay '
        f'style="width: min(720px, 90vw); height: auto;" src="{mp4_name}"></video>'
    )
    out_path.write_text(_html_page(title, body))


# --- Public: per-run HTML (R4.01) -------------------------------------------


def generate_run_html(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> None:
    """R4.01: per-run HTML at ``out_path``; MP4s under
    ``{out_path.parent}/movies/``.

    Movie generation is idempotent on disk (re-encoding only happens
    when the MP4 is missing) so the periodic save loop accretes movies
    as game-passes finish. The page's tab widget selects between
    Official ARC / Weighted / Levels for the top-of-page plots; per-game
    cards and the per-pass-per-game table always render under ARC. In
    in-flight saves only the ARC panel is built.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_games = len(_game_ids(benchmark))

    # ---- Fixed parts (rendered once, always under ARC) --------------------

    per_game_scatter_b64 = _render_per_game_score_vs_tokens_scatter_png(
        benchmark, title=f"{benchmark.label} — per-game score vs tokens"
    )

    tokens_vs_wall_b64 = _render_tokens_vs_wallclock_png(
        [benchmark],
        title=f"{benchmark.label} — total generated tokens vs job wallclock",
    )

    # Each MP4 gets a tiny HTML wrapper (``<video controls>`` + CSS
    # zoom); table cells link to the wrapper, not the raw MP4 (the
    # browser would otherwise play it at native 64×64).
    movies_dir = out_path.parent / "movies"
    movie_rels: dict[tuple[str, int], str | None] = {}
    movie_truncated: dict[tuple[str, int], bool] = {}
    for p in range(benchmark.n_passes):
        for run in _pass_runs(benchmark, p):
            stem = f"g{_safe(run.game_id)}_p{p}"
            mp4_path = movies_dir / f"{stem}.mp4"
            wrapper_path = movies_dir / f"{stem}.html"
            mp4_existed = mp4_path.exists()
            if not mp4_existed:
                # Wait for a terminal state — a mid-play MP4 would be
                # frozen in place by the idempotent-on-disk rule.
                if run.state in ("not_started", "playing"):
                    movie_rels[(run.game_id, p)] = None
                    movie_truncated[(run.game_id, p)] = False
                    continue
                if not _render_run_mp4(run, mp4_path):
                    movie_rels[(run.game_id, p)] = None
                    movie_truncated[(run.game_id, p)] = False
                    continue
            # Re-write the wrapper every save so it picks up the latest
            # ``solver_note`` / ``solver_analysis_html``; the MP4 stays.
            _write_movie_player_html(
                wrapper_path,
                f"{stem}.mp4",
                title=f"{run.game_id} — pass {p}",
                solver_label=benchmark.solver_label,
                solver_analysis_html=run.solver_analysis_html,
            )
            movie_rels[(run.game_id, p)] = f"movies/{stem}.html"
            # Best-effort: after a from_json reload intermediate_states
            # is empty so the flag silently goes False (cosmetic only).
            movie_truncated[(run.game_id, p)] = _total_movie_frames(run) > MAX_MOVIE_FRAMES

    any_truncated = any(movie_truncated.values())

    # Game-major rows so single-game scanning stays local. Score column
    # is always ARC, independent of which scorer tab is active.
    rows = [
        "<tr><th>game</th><th>pass</th><th>state</th><th>score</th>"
        "<th>level</th><th>actions</th><th>tokens</th><th>movie</th><th>note</th></tr>"
    ]
    for game_id in _game_ids(benchmark):
        for p, run in enumerate(_game_runs_by_id(benchmark, game_id)):
            score = _live_score(run)
            tokens = _total_tokens(run)
            mp4_rel = movie_rels.get((run.game_id, p))
            if mp4_rel:
                final_thumb = _thumbnail_html(run, "final", width_px=THUMB_FINAL_PX)
                trunc_tag = (
                    '<br><em style="font-size: 0.85em">subsampled</em>' if movie_truncated.get((run.game_id, p)) else ""
                )
                cell = f'<a href="{mp4_rel}">{final_thumb or "▶"}{trunc_tag}</a>'
            else:
                cell = "—"
            note_html = escape(run.solver_note) if run.solver_note else ""
            rows.append(
                f"<tr><td>{_game_name_html(run)}</td>"
                f"<td>{p}</td>"
                f"{_state_cell_html(run.state)}"
                f"<td>{score:.2f}</td>"
                f"<td>{run.levels_completed}/{run.number_of_levels}</td>"
                f"<td>{_count_actions(run)}</td>"
                f"<td>{tokens}</td>"
                f"<td>{cell}</td>"
                f"<td>{note_html}</td></tr>"
            )
    table_html = "<table>" + "".join(rows) + "</table>"

    git_status_html = _git_status_block_html(benchmark)

    # Per-page banner if any movie hit the frame cap.
    truncation_banner = (
        f"<p>⚠ Some movies exceeded the {MAX_MOVIE_FRAMES}-frame cap and were "
        "subsampled at equal index spacing (first and last frames preserved). "
        "Affected rows are flagged in the table below; the action history in "
        "the JSON / table reflects the full run.</p>"
        if any_truncated
        else ""
    )

    per_game_grid_html = _per_game_grid_for_run_html(benchmark)

    # Per-level token caps (derived from run data, shared across scorers).
    max_per_level = _max_tokens_per_level(benchmark)
    whatif_fractions = [1.0, 0.8, 0.6, 0.4, 0.2]
    whatif_caps = [int(max_per_level * f) for f in whatif_fractions]
    whatif_labels = [
        f"{int(f * 100)}%  ({c // 1000}k tokens/level)" if c >= 1000 else f"{int(f * 100)}%  ({c} tokens/level)"
        for f, c in zip(whatif_fractions, whatif_caps)
    ]

    # ---- Per-scorer tab panels --------------------------------------------
    # In-flight: only ARC, to keep the periodic save loop cheap (variant
    # scores recompute from scratch each save).
    scorers_to_render = [ARC_SCORER] if _is_in_flight(benchmark) else SCORERS
    weights_by_id = _weights_by_game_id(benchmark)

    panels: list[tuple[Scorer, str]] = []
    for scorer in scorers_to_render:
        tokens_curve = _pass_curve(benchmark.game_runs, _total_tokens, scorer=scorer, weights_by_game_id=weights_by_id)
        wall_curve = _pass_curve(benchmark.game_runs, _total_wallclock, scorer=scorer, weights_by_game_id=weights_by_id)
        # Per-pass overlays under the pooled curve so cross-pass variance
        # is visible without losing the pooled budget interpretation.
        # Skipped at n_passes == 1.
        per_pass_tokens: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        per_pass_wall: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        if benchmark.n_passes > 1:
            for p in range(benchmark.n_passes):
                pass_runs = _pass_runs(benchmark, p)
                per_pass_tokens.append(
                    _pass_curve(pass_runs, _total_tokens, scorer=scorer, weights_by_game_id=weights_by_id)
                )
                per_pass_wall.append(
                    _pass_curve(pass_runs, _total_wallclock, scorer=scorer, weights_by_game_id=weights_by_id)
                )
        tokens_b64 = _render_curves_png(
            [tokens_curve],
            x_label="average tokens per game",
            y_label=scorer.y_axis_label,
            title=f"{benchmark.label} — {scorer.label} vs total tokens",
            overlay_curves=per_pass_tokens or None,
        )
        wall_b64 = _render_curves_png(
            [wall_curve],
            x_label="average wallclock per game (s)",
            y_label=scorer.y_axis_label,
            title=f"{benchmark.label} — {scorer.label} vs wallclock",
            overlay_curves=per_pass_wall or None,
        )
        whatif_curves = [
            _capped_pass_curve(benchmark.game_runs, _total_tokens, c, scorer=scorer, weights_by_game_id=weights_by_id)
            for c in whatif_caps
        ]
        whatif_b64 = _render_curves_png(
            whatif_curves,
            x_label="average tokens per game",
            y_label=scorer.y_axis_label,
            title=(
                f"{benchmark.label} — {scorer.label} what-if: per-level token caps  "
                f"(100% = {max_per_level} tokens, the worst observed level)"
            ),
            labels=whatif_labels,
        )
        panel_inner = (
            f"{_per_run_stats_box([benchmark], scorer=scorer)}"
            "<h2>Score vs cumulative tokens</h2>"
            f"{_TOKENS_CURVE_NOTE}"
            f'<img src="data:image/png;base64,{tokens_b64}" alt="score vs tokens">'
            "<h2>What-if: per-level token caps</h2>"
            '<p class="intro">Each curve shows what the score-vs-tokens curve '
            "would have been if every level had been stopped at the listed "
            "per-level token cap. The cap is on cumulative tokens spent "
            "<em>since the most recent level win</em>: cross the cap without "
            "winning the level and the lane truncates there with no further "
            "credit (full credit is kept for previously-won levels). "
            "<strong>100%</strong> is the largest token total any single level "
            "actually consumed in this run (won or not); the other curves are "
            "fractional cuts of that.</p>"
            f'<img src="data:image/png;base64,{whatif_b64}" alt="score vs tokens, per-level caps">'
            "<h2>Score vs wallclock</h2>"
            f"{_PLOT_DESIGN_NOTE}"
            f'<img src="data:image/png;base64,{wall_b64}" alt="score vs wallclock">'
        )
        panels.append((scorer, panel_inner))

    tabs_html = _scorer_tabs_html(panels)
    # Skip the boundary callout when there's nothing to contrast (in-flight
    # or single-scorer page) — it would be misleading.
    remainder_note = _VARIANT_REMAINDER_NOTE if len(scorers_to_render) > 1 else ""

    body = (
        f"<h1>{escape(benchmark.label or 'benchmark')}</h1>"
        f"{_in_flight_banner_html(benchmark)}"
        '<p class="intro">Diagnostics for one TAAF benchmark run — a single solver played each '
        "game over a fixed number of passes. Plots track score-vs-cost across the run; the "
        "table lists every (game, pass) cell, with the movie cell linking to a video of that "
        "play-through.</p>"
        '<div class="summary-box">'
        f"<p>Solver: <code>{escape(benchmark.solver_label)}</code></p>"
        f"<p>{n_games} game{'s' if n_games != 1 else ''} ×"
        f" {benchmark.n_passes} pass{'es' if benchmark.n_passes != 1 else ''}"
        f" = {len(benchmark.game_runs)} runs</p>"
        f"{_timestamps_block_html(benchmark.start_time, benchmark.end_time, with_end=True)}"
        f"{truncation_banner}"
        f"{git_status_html}"
        "</div>"
        f"{_weights_banner_html(benchmark)}"
        f"{tabs_html}"
        f"{remainder_note}"
        "<h2>Total generated tokens vs job wallclock</h2>"
        f'<img src="data:image/png;base64,{tokens_vs_wall_b64}" alt="total tokens vs job wallclock">'
        "<h2>Per-game score vs tokens</h2>"
        f'<img src="data:image/png;base64,{per_game_scatter_b64}" alt="per-game score vs tokens scatter">'
        "<h2>Per-game</h2>"
        f"{per_game_grid_html}"
        "<h2>Per-pass per-game results</h2>"
        f"{table_html}"
    )
    out_path.write_text(_html_page(benchmark.label or "benchmark", body))


# --- Public: per-run summary (R4.02) ----------------------------------------


def run_summary_text(benchmark: taaf.benchmark.Benchmark) -> str:
    """Build the Slack-friendly text summary (R4.02). Also sourced by
    ``Benchmark.run``'s end-of-run stdout print.
    """
    scores = [_live_score(r) for r in benchmark.game_runs]
    won = sum(1 for r in benchmark.game_runs if r.state == "won")
    total_actions = sum(_count_actions(r) for r in benchmark.game_runs)
    total_tokens = sum(_total_tokens(r) for r in benchmark.game_runs)
    total_wall = sum(_total_wallclock(r) for r in benchmark.game_runs)
    elapsed_seconds = _benchmark_elapsed_seconds(benchmark)
    generated_tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds and elapsed_seconds > 0 else None

    lines: list[str] = [
        f"benchmark: {benchmark.label}",
        f"solver:    {benchmark.solver_label}",
        f"games:     {len(_game_ids(benchmark))}",
        f"passes:    {benchmark.n_passes}",
        f"runs:      {len(benchmark.game_runs)} (won: {won})",
    ]
    if benchmark.start_time is not None:
        lines.append(f"started:   {_format_timestamp(benchmark.start_time)}")
    if benchmark.end_time is not None and benchmark.start_time is not None:
        lines.append(f"ended:     {_format_timestamp(benchmark.end_time)}")
        lines.append(f"duration:  {_format_duration(benchmark.end_time - benchmark.start_time)}")
    elif benchmark.start_time is not None:
        lines.append("ended:     in progress")
    lines.extend(
        [
            f"mean score:    {statistics.mean(scores):.2f}" if scores else "mean score:    —",
            f"median score:  {statistics.median(scores):.2f}" if scores else "median score:  —",
            f"total actions: {total_actions}",
            f"total tokens:  {total_tokens}",
            (
                f"generated tokens/sec: {generated_tokens_per_second:.2f} (job wallclock)"
                if generated_tokens_per_second is not None
                else "generated tokens/sec: —"
            ),
            f"total wallclock: {total_wall:.1f}s",
            "",
            "per-game (mean across passes):",
        ]
    )
    for g in _game_ids(benchmark):
        runs = _game_runs_by_id(benchmark, g)
        gs = [_live_score(r) for r in runs]
        actions = sum(_count_actions(r) for r in runs) // max(len(runs), 1)
        tokens = sum(_total_tokens(r) for r in runs) // max(len(runs), 1)
        levels_avg = sum(r.levels_completed for r in runs) / max(len(runs), 1)
        total_levels = runs[0].number_of_levels if runs else 0
        score_str = f"{statistics.mean(gs):.2f}" if gs else "—"
        lines.append(
            f"  {g}: score={score_str}, levels={levels_avg:.1f}/{total_levels}, actions={actions}, tokens={tokens}"
        )
    return "\n".join(lines) + "\n"


def generate_run_summary_txt(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> None:
    """R4.02: write the brief text summary to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(run_summary_text(benchmark))


# --- Public: comparison HTML (R4.11/R4.12) ----------------------------------


def _harmonize_for_comparison(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
) -> tuple[list[str], set[str]]:
    """Mutate ``benchmarks`` in place so a comparison page can be rendered.

    - If game-id sets differ, restrict each benchmark to the intersection
      (drop game_runs / game_weights / games slots accordingly, preserving
      the pass-major layout).
    - If resolved ``{game_id: weight}`` maps differ, override every
      non-first benchmark's ``game_weights`` with the first benchmark's
      values (mapped by game_id, so slot-order differences are fine).

    Returns ``(notices, intersection)``. ``notices`` is a list of HTML
    snippets describing each mutation applied (empty when nothing
    changed). ``intersection`` is the resolved set of common game ids;
    an empty set signals the caller to skip rendering entirely.
    """
    notices: list[str] = []
    game_id_sets: list[set[str]] = [set(_game_ids(b)) for b in benchmarks]
    # ``set.intersection(*…)`` unpacked-call loses element type through pyright's
    # variadic stub; iterate so the result stays typed as ``set[str]``.
    intersection: set[str] = set[str]()
    if game_id_sets:
        intersection = set(game_id_sets[0])
        for s in game_id_sets[1:]:
            intersection &= s
    if any(s != intersection for s in game_id_sets):
        parts: list[str] = []
        for lbl, s in zip(labels, game_id_sets):
            dropped = sorted(s - intersection)
            if not dropped:
                continue
            shown = ", ".join(escape(g) for g in dropped[:5])
            if len(dropped) > 5:
                shown += f", … ({len(dropped) - 5} more)"
            parts.append(f"<code>{escape(lbl)}</code> dropped {len(dropped)} ({shown})")
        notices.append(
            "<strong>Game sets differ across runs.</strong> Restricted to the intersection "
            f"({len(intersection)} game{'s' if len(intersection) != 1 else ''}). " + "; ".join(parts) + "."
        )
        for b in benchmarks:
            if b.n_passes == 0 or not b.game_runs:
                continue
            orig_n_games = len(b.game_runs) // b.n_passes
            canonical = [b.game_runs[i].game_id for i in range(orig_n_games)]
            keep = [i for i, gid in enumerate(canonical) if gid in intersection]
            new_runs: list[taaf.game.GameRun] = []
            for p in range(b.n_passes):
                base = p * orig_n_games
                for i in keep:
                    new_runs.append(b.game_runs[base + i])
            b.game_runs = new_runs
            if b.game_weights is not None:
                b.game_weights = [b.game_weights[i] for i in keep]
            if len(b.games) == orig_n_games:
                b.games = [b.games[i] for i in keep]
    if not intersection:
        return notices, intersection
    # Weights are resolved via ``{game_id: weight}`` so different
    # ``Benchmark.games`` orderings compare semantically. ``None``
    # resolves to all-1.0 — equal to an explicit all-1.0 list.
    weights_maps = [_weights_by_game_id(b) for b in benchmarks]
    if any(w != weights_maps[0] for w in weights_maps[1:]):
        notices.append(
            f"<strong>Game weights differ across runs.</strong> Applied "
            f"<code>{escape(labels[0])}</code>'s <code>game_weights</code> to every run."
        )
        first_weights = weights_maps[0]
        for b in benchmarks[1:]:
            if b.n_passes == 0 or not b.game_runs:
                continue
            n_games = len(b.game_runs) // b.n_passes
            slot_gids = [b.game_runs[i].game_id for i in range(n_games)]
            b.game_weights = [first_weights.get(g, 1.0) for g in slot_gids]
    return notices, intersection


def _harmonization_banner_html(notices: list[str]) -> str:
    """Yellow banner above the comparison body listing any in-place
    harmonization the renderer applied to the input benchmarks."""
    if not notices:
        return ""
    items = "".join(f"<li>{n}</li>" for n in notices)
    return (
        f'<div class="weights-banner"><p><strong>Inputs harmonized for comparison.</strong></p><ul>{items}</ul></div>'
    )


def generate_comparison_html(benchmarks: list[taaf.benchmark.Benchmark], out_dir: Path) -> None:
    """R4.11 / R4.12: write the comparison tree at ``out_dir``
    (``index.html`` plus ``per_game/<game>.html`` drill-downs).

    Refuses in-flight inputs — cross-run statistics are meaningless on
    partial snapshots. Inputs are harmonized in place via
    ``_harmonize_for_comparison`` (intersection of game-id sets;
    weights unified to the first benchmark). Both surface in a banner.
    Empty intersection raises rather than leaving a stale index.html.
    """
    if len(benchmarks) < 2:
        raise ValueError("generate_comparison_html requires at least 2 benchmarks")
    in_flight = [b.label or f"run {i}" for i, b in enumerate(benchmarks) if _is_in_flight(b)]
    if in_flight:
        raise ValueError(
            f"cannot generate comparison: {len(in_flight)} benchmark(s) still in-flight: "
            f"{in_flight}. Wait for run() to finish (end_time gets set in the teardown)."
        )
    labels = [b.label or f"run {i}" for i, b in enumerate(benchmarks)]
    harmonize_notices, intersection = _harmonize_for_comparison(benchmarks, labels)
    if not intersection:
        # Raise rather than no-op: a quiet return would leave a stale
        # index.html from a previous compare reading as fresh output.
        raise ValueError(
            f"generate_comparison_html: empty intersection of game-id sets across "
            f"{len(benchmarks)} benchmarks — nothing to compare. Adjust the input "
            f"benchmarks so at least one game_id is common to all of them."
        )
    common_games = sorted(intersection)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_game_dir = out_dir / "per_game"
    per_game_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fixed parts (rendered once, always under ARC) --------------------

    # R4.12: per-game cards (initial-state thumbnail + tokens / wallclock
    # plots) linking to the per-game drill-down.
    per_game_section = _per_game_grid_html(benchmarks, common_games, labels)

    # Top-of-page links to each run's per-run diagnostics. Skipped when
    # a benchmark has no job_dir or its diagnostics.html isn't on disk
    # yet (comparison generated standalone before the per-run save).
    per_run_links: list[str] = []
    for b, lbl in zip(benchmarks, labels):
        if b.job_dir is None:
            continue
        target = b.job_dir / "diagnostics.html"
        if not target.exists():
            continue
        rel = os.path.relpath(target, start=out_dir)
        per_run_links.append(f'<a href="{rel}"><code>{escape(lbl)}</code></a>')
    per_run_links_html = f"<p>Per-run details: {' · '.join(per_run_links)}</p>" if per_run_links else ""

    runs_lines: list[str] = []
    for b in benchmarks:
        passes = f"{b.n_passes} pass{'es' if b.n_passes != 1 else ''}"
        when = f" — started {_format_timestamp(b.start_time)}" if b.start_time else ""
        runs_lines.append(f"<li><code>{escape(b.label)}</code> ({passes}){when}</li>")
    runs_summary = "<ul>" + "".join(runs_lines) + "</ul>"

    # Per-game pages — always rendered under ARC, regardless of which
    # comparison index variant the reader is on.
    for g in common_games:
        _write_per_game_page(g, benchmarks, per_game_dir / f"{_safe(g)}.html")

    # ---- Per-scorer tab panels --------------------------------------------
    # All three scorers always render — in-flight inputs are refused
    # above.

    # Two statistics variants (R4.12): runs scored at their endpoints, and
    # re-scored at a shared per-game token budget so a run that merely spent
    # more tokens doesn't look better for it. The budget is
    # scorer-independent, so it's computed once. Rendered as a budget ×
    # scorer tab grid (budget tabs above scorer tabs).
    budget = _min_tokens_per_game_budget(benchmarks)
    ceiling = _global_ceiling_for_budget(benchmarks, budget) if budget > 0 else 0.0
    budget_modes: list[tuple[str, str, bool, str]] = [
        (
            "full",
            "Full runs (max tokens)",
            False,
            "Each run scored where it actually ended, at its own per-game token budget.",
        )
    ]
    if budget > 0:
        budget_modes.append(
            (
                "capped",
                f"Common budget ({budget:,.0f} tok/game)",
                True,
                (
                    f"Every (game, pass) re-scored at a shared per-game budget of {budget:,.0f} generated "
                    "tokens — the smallest run's mean tokens per game, the rightmost point all runs' "
                    "score-vs-tokens curves share. Modeled as a parallel cutoff: all games spend tokens "
                    "together and freeze when the average game has spent the budget, so cheaper games keep "
                    "their final score while more expensive ones are capped at the shared cutoff."
                ),
            )
        )

    scorer_panels: list[tuple[Scorer, dict[str, str]]] = []
    for scorer in SCORERS:
        # Pooled curves are budget-independent (they span the full range),
        # so render once per scorer and reuse in every budget panel.
        token_curves = [
            _pass_curve(b.game_runs, _total_tokens, scorer=scorer, weights_by_game_id=_weights_by_game_id(b))
            for b in benchmarks
        ]
        wall_curves = [
            _pass_curve(b.game_runs, _total_wallclock, scorer=scorer, weights_by_game_id=_weights_by_game_id(b))
            for b in benchmarks
        ]
        tokens_b64 = _render_curves_png(
            token_curves,
            x_label="average tokens per game",
            y_label=scorer.y_axis_label,
            title=f"comparison — {scorer.label} vs total tokens",
            labels=labels,
            palette=RUN_COLORS,
        )
        wall_b64 = _render_curves_png(
            wall_curves,
            x_label="average wallclock per game (s)",
            y_label=scorer.y_axis_label,
            title=f"comparison — {scorer.label} vs wallclock",
            labels=labels,
            palette=RUN_COLORS,
        )
        plots_html = (
            "<h2>Score vs cumulative tokens</h2>"
            f"{_TOKENS_CURVE_NOTE}"
            f'<img src="data:image/png;base64,{tokens_b64}" alt="comparison tokens">'
            "<h2>Score vs wallclock</h2>"
            f"{_PLOT_DESIGN_NOTE}"
            f'<img src="data:image/png;base64,{wall_b64}" alt="comparison wallclock">'
        )

        by_budget: dict[str, str] = {}
        for mode_key, _mode_label, is_capped, note in budget_modes:
            score_fn = _capped_score_fn(scorer, ceiling) if is_capped else None
            per_run_html = _per_run_stats_box(benchmarks, scorer=scorer, score_fn=score_fn)
            stats_box = _comparison_stats_box(
                benchmarks,
                labels,
                scorer=scorer,
                score_fn=score_fn,
                heading="Score comparison",
                note=note,
            )
            scatter_html = ""
            if len(benchmarks) == 2:
                per_game_a = _per_game_mean_scores(benchmarks[0], scorer=scorer, score_fn=score_fn)
                per_game_b = _per_game_mean_scores(benchmarks[1], scorer=scorer, score_fn=score_fn)
                scatter_b64 = _render_scatter_png(
                    per_game_a,
                    per_game_b,
                    labels[0],
                    labels[1],
                    f"head-to-head per-game mean — {scorer.label}",
                )
                scatter_html = (
                    f'<h2>Head-to-head</h2><img src="data:image/png;base64,{scatter_b64}" alt="head-to-head scatter">'
                )
            by_budget[mode_key] = f"{per_run_html}{stats_box}{plots_html}{scatter_html}"
        scorer_panels.append((scorer, by_budget))

    tabs_html = _budget_scorer_tabs_html(
        scorer_panels,
        budget_labels=[(k, lbl) for k, lbl, _, _ in budget_modes],
    )

    tokens_vs_wall_b64 = _render_tokens_vs_wallclock_png(
        benchmarks,
        title="comparison — total generated tokens vs job wallclock",
        labels=labels,
        palette=RUN_COLORS,
    )

    body = (
        f"<h1>Comparison: {' vs '.join(escape(label) for label in labels)}</h1>"
        '<p class="intro">Side-by-side comparison of multiple TAAF benchmark runs on the same set '
        "of games. The three score tests appear under budget tabs (full runs vs a shared per-game "
        "token budget — the smallest run's) above the score-variant tabs, as a head-to-head box for "
        "two runs or NxN matrices of pairwise p-values for more. With exactly two runs you also get a "
        "head-to-head scatter. Click a per-game thumbnail to drill into how every run did on that "
        "single game.</p>"
        f'<div class="summary-box"><p>Runs:</p>{runs_summary}'
        f"<p>Games: {len(common_games)}</p></div>"
        f"{per_run_links_html}"
        f"{_harmonization_banner_html(harmonize_notices)}"
        f"{_comparison_weights_banner_html(benchmarks, labels)}"
        f"{tabs_html}"
        f"{_VARIANT_REMAINDER_NOTE}"
        "<h2>Total generated tokens vs job wallclock</h2>"
        f'<img src="data:image/png;base64,{tokens_vs_wall_b64}" alt="total tokens vs job wallclock">'
        f"{per_game_section}"
    )
    (out_dir / "index.html").write_text(_html_page("Comparison", body))


def _per_game_run_curve(
    bm: taaf.benchmark.Benchmark,
    game_id: str,
    x_fn: Callable[[taaf.game.GameRun], float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Single-game pooled-pass curve for one benchmark.

    Treats every pass of ``game_id`` as one independent lane and runs them
    through ``_pass_curve``. The result is "expected (cost, score) for this
    game at the given budget", which is the per-game-flavoured analogue of
    the cross-game pooled curves on the run / comparison pages.
    """
    runs = _game_runs_by_id(bm, game_id)
    if not runs:
        return np.array([0.0]), np.array([0.0])
    return _pass_curve(runs, x_fn)


def _per_game_plots_b64(
    benchmarks: list[taaf.benchmark.Benchmark],
    game_id: str,
    labels: list[str],
    *,
    figsize: tuple[float, float] = (4.5, 3),
) -> tuple[str, str]:
    """Per-game (tokens, wallclock) plots — one pooled curve per run."""
    token_curves = [_per_game_run_curve(b, game_id, _total_tokens) for b in benchmarks]
    wall_curves = [_per_game_run_curve(b, game_id, _total_wallclock) for b in benchmarks]
    tokens = _render_curves_png(
        token_curves,
        x_label="average tokens for this game",
        y_label="average score for this game",
        title=f"{game_id} — score vs tokens",
        labels=labels,
        palette=RUN_COLORS,
        figsize=figsize,
    )
    wall = _render_curves_png(
        wall_curves,
        x_label="average wallclock for this game (s)",
        y_label="average score for this game",
        title=f"{game_id} — score vs wallclock",
        labels=labels,
        palette=RUN_COLORS,
        figsize=figsize,
    )
    return tokens, wall


_PER_GAME_CARD_CACHE_VERSION = 2
"""Stamped into the cache key; bump when the rendering of a per-game
card changes so old caches are invalidated rather than served stale."""


def _per_game_card_key(g: str, runs: list[taaf.game.GameRun], in_flight: bool) -> str:
    """Stable cache key — the card output is a pure function of these.
    The in-flight flag is part of the key because in-flight cards skip
    the per-pass overlay; without it, the post-run regen would re-use
    the overlay-free version.
    """
    parts = [str(_PER_GAME_CARD_CACHE_VERSION), g, "if" if in_flight else "done"]
    for r in runs:
        parts.append(f"{r.state}:{len(r.history)}:{r.levels_completed}:{r.final_score}")
    return "|".join(parts)


def _load_per_game_card_cache(job_dir: Path | None) -> dict[str, tuple[str, str]]:
    """Load `{game_id: (key, html)}` from disk; empty dict on miss."""
    if job_dir is None:
        return {}
    path = job_dir / ".per_game_cards.pkl"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cached_obj: Any = pickle.load(f)
    except Exception:
        # Corrupted / incompatible — caller will overwrite.
        return {}
    if not isinstance(cached_obj, dict):
        return {}
    return cast("dict[str, tuple[str, str]]", cached_obj)


def _save_per_game_card_cache(job_dir: Path | None, cache: dict[str, tuple[str, str]]) -> None:
    if job_dir is None:
        return
    path = job_dir / ".per_game_cards.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f)
    tmp.replace(path)


def _per_game_grid_for_run_html(
    benchmark: taaf.benchmark.Benchmark,
    *,
    figsize: tuple[float, float] = (4.5, 3),
) -> str:
    """Per-game cards for the per-run page: thumbnail + (tokens,
    wallclock) plots. One pooled curve per game.

    Cached on disk at ``{job_dir}/.per_game_cards.pkl``, keyed on
    per-pass ``(state, len(history), levels_completed, final_score)``.
    A finished pass freezes its key so each card renders once and is
    reused on every subsequent regen.
    """
    in_flight = _is_in_flight(benchmark)
    cache = _load_per_game_card_cache(benchmark.job_dir)
    new_cache: dict[str, tuple[str, str]] = {}
    cards: list[str] = []
    for g in _game_ids(benchmark):
        runs = _game_runs_by_id(benchmark, g)
        if not runs:
            continue
        key = _per_game_card_key(g, runs, in_flight)
        cached = cache.get(g)
        if cached is not None and cached[0] == key:
            cards.append(cached[1])
            new_cache[g] = cached
            continue
        thumb = _thumbnail_html(runs[0], "initial", width_px=THUMB_FINAL_PX)
        # Per-pass overlays: skipped in-flight (cheap periodic loop)
        # and when n_passes == 1 (would duplicate the pooled curve).
        per_pass_tokens: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        per_pass_wall: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        if not in_flight and len(runs) > 1:
            for r in runs:
                per_pass_tokens.append(_pass_curve([r], _total_tokens))
                per_pass_wall.append(_pass_curve([r], _total_wallclock))
        tokens_b64 = _render_curves_png(
            [_pass_curve(runs, _total_tokens)],
            x_label="average tokens for this game",
            y_label="average score for this game",
            title=f"{g} — score vs tokens",
            figsize=figsize,
            overlay_curves=per_pass_tokens or None,
        )
        wall_b64 = _render_curves_png(
            [_pass_curve(runs, _total_wallclock)],
            x_label="average wallclock for this game (s)",
            y_label="average score for this game",
            title=f"{g} — score vs wallclock",
            figsize=figsize,
            overlay_curves=per_pass_wall or None,
        )
        html = (
            '<div class="per-game-card">'
            f'<div class="per-game-header"><span class="game-name">{thumb}<code>{escape(g)}</code></span></div>'
            f'<img src="data:image/png;base64,{tokens_b64}" alt="{escape(g)} tokens">'
            f'<img src="data:image/png;base64,{wall_b64}" alt="{escape(g)} wallclock">'
            "</div>"
        )
        cards.append(html)
        new_cache[g] = (key, html)
    _save_per_game_card_cache(benchmark.job_dir, new_cache)
    return '<div class="per-game-grid">' + "".join(cards) + "</div>"


def _per_game_grid_html(
    benchmarks: list[taaf.benchmark.Benchmark],
    common_games: list[str],
    labels: list[str],
) -> str:
    """Per-game cards on the comparison index. Each card holds the
    initial-state thumbnail + game id + two small clickable plots
    (tokens, wallclock) wrapped in a link to ``per_game/<id>.html`` —
    the R4.12 "two plots again, but per game; clickable to drill-down".
    """
    cards: list[str] = []
    for g in common_games:
        g_runs = _game_runs_by_id(benchmarks[0], g)
        thumb = _thumbnail_html(g_runs[0], "initial", width_px=THUMB_FINAL_PX) if g_runs else ""
        tokens_b64, wall_b64 = _per_game_plots_b64(benchmarks, g, labels)
        href = f"per_game/{_safe(g)}.html"
        cards.append(
            '<div class="per-game-card">'
            f'<div class="per-game-header"><a href="{href}" class="game-link">{thumb}<code>{escape(g)}</code></a></div>'
            f'<a href="{href}"><img src="data:image/png;base64,{tokens_b64}" alt="{escape(g)} tokens"></a>'
            f'<a href="{href}"><img src="data:image/png;base64,{wall_b64}" alt="{escape(g)} wallclock"></a>'
            "</div>"
        )
    return '<h2>Per-game</h2><div class="per-game-grid">' + "".join(cards) + "</div>"


def _write_per_game_page(game_id: str, benchmarks: list[taaf.benchmark.Benchmark], out_path: Path) -> None:
    """Per-game drill-down page in the comparison tree.

    Same column shape as the per-run table, just sliced by game instead of
    by pass. Movie cells link across to each run's `<job_dir>/movies/...`
    via a path relative to this per-game page.
    """
    rows = [
        "<tr><th>run</th><th>pass</th><th>state</th><th>score</th>"
        "<th>level</th><th>actions</th><th>tokens</th><th>movie</th><th>note</th></tr>"
    ]
    representative_run: taaf.game.GameRun | None = None
    page_dir = out_path.parent  # so we can resolve movie paths relative to here
    for bm in benchmarks:
        for p in range(bm.n_passes):
            for r in _pass_runs(bm, p):
                if r.game_id != game_id:
                    continue
                if representative_run is None:
                    representative_run = r
                score = _live_score(r)
                tokens = _total_tokens(r)
                # Movie link — only if the benchmark wrote a job_dir AND the
                # MP4 wrapper actually exists on disk. The wrapper is named
                # `movies/g{game_id}_p{pass}.html` inside the run's job_dir.
                movie_cell = "—"
                if bm.job_dir is not None:
                    stem = f"g{_safe(r.game_id)}_p{p}"
                    wrapper_path = bm.job_dir / "movies" / f"{stem}.html"
                    if wrapper_path.exists():
                        rel = os.path.relpath(wrapper_path, start=page_dir)
                        thumb = _thumbnail_html(r, "final", width_px=THUMB_FINAL_PX)
                        movie_cell = f'<a href="{rel}">{thumb or "▶"}</a>'
                note_html = escape(r.solver_note) if r.solver_note else ""
                rows.append(
                    f"<tr><td><code>{escape(bm.label)}</code></td>"
                    f"<td>{p}</td>"
                    f"{_state_cell_html(r.state)}"
                    f"<td>{score:.2f}</td>"
                    f"<td>{r.levels_completed}/{r.number_of_levels}</td>"
                    f"<td>{_count_actions(r)}</td>"
                    f"<td>{tokens}</td>"
                    f"<td>{movie_cell}</td>"
                    f"<td>{note_html}</td></tr>"
                )
    table_html = "<table>" + "".join(rows) + "</table>"
    thumb = _thumbnail_html(representative_run, "initial", width_px=128)
    # Full-size per-game plots above the table — same data as the small
    # ones on the comparison index, just larger.
    labels = [b.label or f"run {i}" for i, b in enumerate(benchmarks)]
    tokens_b64, wall_b64 = _per_game_plots_b64(benchmarks, game_id, labels, figsize=(8, 4))
    body = (
        '<p><a href="../index.html">← back to comparison</a></p>'
        f'<h1><span class="game-name">{thumb}<code>{escape(game_id)}</code></span></h1>'
        '<p class="intro">All runs of this single game across the benchmarks being compared. '
        "Plots show how each run progressed for THIS game in particular; the table at the "
        "bottom lists every (run, pass) cell with score and cost.</p>"
        "<h2>Score vs cumulative tokens</h2>"
        f'<img src="data:image/png;base64,{tokens_b64}" alt="{escape(game_id)} tokens">'
        "<h2>Score vs wallclock</h2>"
        f"{_PLOT_DESIGN_NOTE}"
        f'<img src="data:image/png;base64,{wall_b64}" alt="{escape(game_id)} wallclock">'
        "<h2>All (run, pass) cells</h2>"
        f"{table_html}"
    )
    out_path.write_text(_html_page(f"Comparison — {game_id}", body))


# --- Public: regenerate-from-disk helpers ----------------------------------


def _load_benchmark_from_dir(run_dir: Path) -> taaf.benchmark.Benchmark:
    """Load a finished benchmark from its ``job_dir``.

    Prefers ``benchmark.json`` (and the ``intermediate_states.pkl``
    sidecar that :meth:`Benchmark.from_json` attaches by default): that
    pair round-trips everything the diagnostics renderers (movies,
    level-step curves, per-level cap what-if) need, without requiring
    the original solver / game classes to be importable. Falls back to
    the legacy monolithic ``benchmark.pkl`` for runs that predate the
    split layout.
    """
    json_path = run_dir / "benchmark.json"
    pkl_path = run_dir / "benchmark.pkl"
    if json_path.exists():
        # ``from_json`` auto-attaches intermediate_states from its sidecar
        # and sets job_dir to the JSON's parent; no further wiring needed.
        return taaf.benchmark.Benchmark.from_json(json_path)
    if pkl_path.exists():
        bm = taaf.benchmark.Benchmark.from_pickle(pkl_path)
        bm.job_dir = run_dir
        return bm
    raise FileNotFoundError(f"No benchmark.json and no benchmark.pkl in {run_dir}")


def regenerate_run_diagnostics(
    benchmark: str | Path | taaf.benchmark.Benchmark,
    out_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Regenerate per-run diagnostics for a finished benchmark.
    ``benchmark`` is a run dir path or a ``Benchmark`` instance. When
    a path is passed, ``out_dir`` defaults to it (in-place regen).
    When a ``Benchmark`` instance is passed, ``out_dir`` is mandatory.

    Refuses in-flight inputs. If ``out_dir`` already holds a different
    ``benchmark.json``, refuses unless ``overwrite=True``. When the
    output dir is fresh, writes ``benchmark.json`` +
    ``intermediate_states.pkl`` there so it's self-contained for future
    regens. ``games.pkl`` / ``solver.pkl`` are not re-written.
    """
    if isinstance(benchmark, (str, Path)):
        input_dir = Path(benchmark)
        bm = _load_benchmark_from_dir(input_dir)
        if out_dir is None:
            out_dir = input_dir
    else:
        bm = benchmark
        if out_dir is None:
            raise ValueError("out_dir is required when passing a Benchmark instance (no implicit path to default to)")
    out_dir = Path(out_dir)

    if _is_in_flight(bm):
        raise ValueError(
            f"benchmark is in-flight (start_time={bm.start_time}, end_time=None). "
            f"regenerate_run_diagnostics only operates on finished runs; "
            f"re-run after teardown."
        )

    # JSON equality check against any existing snapshot at out_dir.
    # ``to_json_dict`` captures all diagnostically-relevant state and
    # dodges Python class-identity quirks.
    out_json = out_dir / "benchmark.json"
    must_write = True
    if out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
        if bm.to_json_dict() == existing:
            must_write = False
        elif not overwrite:
            raise ValueError(
                f"{out_json} already exists and does not match the provided "
                f"benchmark. Pass overwrite=True to replace it, choose a "
                f"different out_dir, or delete the existing snapshot first."
            )

    # Temporarily point ``job_dir`` at out_dir so the card cache and
    # save methods write there. Restored on exit.
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_job_dir = bm.job_dir
    bm.job_dir = out_dir
    try:
        if must_write:
            bm._save_json()
            bm._save_intermediate_states()
        generate_run_html(bm, out_dir / "diagnostics.html")
        generate_run_summary_txt(bm, out_dir / "summary.txt")
    finally:
        bm.job_dir = saved_job_dir


def regenerate_comparison_diagnostics(
    benchmarks: list[str | Path | taaf.benchmark.Benchmark],
    out_dir: str | Path,
) -> None:
    """Write the comparison HTML tree to ``out_dir`` from a mix of run
    directories and/or in-memory ``Benchmark`` instances. Refuses
    in-flight inputs.

    Only HTML lands at ``out_dir`` — per-run artifacts stay in their
    input dirs and per-game movie links use ``os.path.relpath``, so
    the input dirs must stay reachable from ``out_dir`` for the links
    to work.
    """
    bms: list[taaf.benchmark.Benchmark] = []
    for b in benchmarks:
        if isinstance(b, (str, Path)):
            bms.append(_load_benchmark_from_dir(Path(b)))
        else:
            bms.append(b)
    generate_comparison_html(bms, Path(out_dir))
