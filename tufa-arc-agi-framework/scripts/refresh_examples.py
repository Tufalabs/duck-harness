"""Regenerate the committed example diagnostics under ``examples/diagnostics/``.

Run via ``make refresh-examples``. Outputs:

- ``examples/diagnostics/run_a/`` — per-run HTML for ``SolverSequence``.
- ``examples/diagnostics/run_b/`` — per-run HTML for ``SolverRandom(seed=42)``.
- ``examples/diagnostics/comparison/`` — multi-run HTML.

The committed HTMLs double as living documentation: a new contributor can
open them in a browser to see what the diagnostics look like without
running anything. They are NOT byte-compared against in tests
(matplotlib + ffmpeg outputs aren't deterministic enough for that to be
useful) — re-running this script after intentional diagnostics changes
and committing the diff is the workflow.

Game lineup: ``ExampleGame`` + ``GameAPI(env_name="ls20")``. The
comparison features a real arc_agi env next to a synthetic baseline.
Three passes per benchmark so the per-pass spread is meaningful
(Welch's df ≥ 1 needs n ≥ 2; n = 3 starts giving a usable p, see the
comparison page caveat).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import arcengine

import taaf

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "diagnostics"


def _act(action_id: int) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data={})


# 6-move solution to ExampleGame + RESET separator + 13-move solution to ls20
# level 0. SolverSequence plays the entire list on every game; ExampleGame
# finishes after move 6 and ignores the rest, while ls20 absorbs the prefix,
# RESETs, then clears level 0.
LS20_LEVEL0_WINNING_MOVES = [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1]
COMBINED_WINNING_SEQUENCE: list[arcengine.ActionInput] = (
    [_act(1)] * 3 + [_act(2)] * 3
    + [_act(0)]  # RESET — bring ls20 back to fresh state before its sequence
    + [_act(i) for i in LS20_LEVEL0_WINNING_MOVES]
)


async def _main() -> None:
    if EXAMPLES_DIR.exists():
        shutil.rmtree(EXAMPLES_DIR)
    EXAMPLES_DIR.mkdir(parents=True)

    games: list[taaf.game.Game] = [
        taaf.game_examples.ExampleGame(),
        taaf.game_api.GameAPI(env_name="ls20"),
    ]

    bm_a = taaf.benchmark.Benchmark(
        label="seq_winner",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=COMBINED_WINNING_SEQUENCE),
        n_passes=3,
        job_dir=EXAMPLES_DIR / "run_a",
    )
    bm_b = taaf.benchmark.Benchmark(
        label="random_seed42",
        games=games,
        solver=taaf.solver_examples.SolverRandom(seed=42, max_actions_per_game=200),
        n_passes=3,
        job_dir=EXAMPLES_DIR / "run_b",
    )
    await bm_a.run()
    await bm_b.run()
    taaf.diagnostics.generate_comparison_html([bm_a, bm_b], EXAMPLES_DIR / "comparison")

    # ``benchmark.json`` and ``benchmark.pkl`` are large and not part of the
    # human-facing example surface; drop them so the committed directory
    # stays small. (The pickle is ~120 KB per run; JSON ~85 KB.)
    for run_dir in (EXAMPLES_DIR / "run_a", EXAMPLES_DIR / "run_b"):
        for name in ("benchmark.json", "benchmark.pkl"):
            p = run_dir / name
            if p.exists():
                p.unlink()

    written = sorted(EXAMPLES_DIR.rglob("*"))
    print(f"\nWrote {sum(1 for p in written if p.is_file())} files under {EXAMPLES_DIR}/:")
    for path in written:
        if path.is_file():
            print(f"  {path.relative_to(EXAMPLES_DIR)}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(_main())
