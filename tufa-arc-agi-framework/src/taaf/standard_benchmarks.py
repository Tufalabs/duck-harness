"""Team-agreed standard benchmarks: shared game lists + weights so
results are directly comparable across runs. Each benchmark is frozen
once published; new ones land as new functions.
"""

from __future__ import annotations

import collections

import taaf.benchmark
import taaf.competition_arcade
import taaf.game
import taaf.game_api
import taaf.solver


def make_benchmark_all_games(solver: taaf.solver.Solver | None = None) -> taaf.benchmark.Benchmark:
    """All-game benchmark over the full local game catalog.

    This benchmark enumerated every game and probed its level count from the
    offline env files, neither of which is bundled in this build, so the
    all-game benchmark is unavailable. Use
    :func:`make_benchmark_kaggle_official_110` (the 25 official games), or run
    against the live competition Arcade.
    """
    del solver
    raise RuntimeError(
        "make_benchmark_all_games requires the full local game catalog, which is "
        "not bundled in this build. Use make_benchmark_kaggle_official_110 or the "
        "live competition Arcade."
    )


def make_benchmark_kaggle_official_110(
    solver: taaf.solver.Solver | None = None,
    *,
    arcade_spec: taaf.game_api.ArcadeSpec | None = None,
    competition_clone_ids: bool = False,
    competition_sim: bool = False,
) -> taaf.benchmark.Benchmark:
    """Kaggle submission-shaped public benchmark (R2.57).

    The 25 official games are repeated round-robin to 110 independent
    ``GameAPI`` entries, with ``n_passes=1``. When
    ``competition_clone_ids=True``, the repeated entries use the same
    deterministic clone IDs exposed by
    ``CompetitionArcadeServer.official_110()``, making the benchmark
    compatible with the competition-style Arcade that does not support
    multiple runs for the same game ID.

    ``competition_sim=True`` targets a local competition-arcade simulator
    (R11.13) started lazily on whichever process runs the benchmark: it
    sets a ``competition_sim`` ``ArcadeSpec`` and forces
    ``competition_clone_ids`` so the game IDs line up with the server's
    official-110 clone set. This lets a non-submission run exercise the
    submission-shaped arcade without an actual leaderboard submission.
    """
    if competition_sim:
        if arcade_spec is not None:
            raise ValueError("Pass either competition_sim=True or an explicit arcade_spec, not both.")
        arcade_spec = taaf.game_api.ArcadeSpec(competition_sim=True)
        competition_clone_ids = True
    official_ids = list(taaf.competition_arcade.official_game_ids())
    if len(official_ids) != 25:
        raise RuntimeError(f"Expected 25 official games for Kaggle benchmark, got {len(official_ids)}.")
    if competition_clone_ids:
        # Competition arcade registers each clone as a distinct EnvironmentInfo,
        # so the natural ``info.game_id`` (``p005`` etc.) is already unique — no
        # external override needed.
        game_ids = taaf.competition_arcade.clone_game_ids(official_ids, total_runs=110)
        external_ids: list[str | None] = [None] * len(game_ids)
    else:
        # No competition arcade to synthesize unique env IDs — only the 25 base
        # IDs are known. Override each instance's public ``game_id`` with
        # ``{base}_{i}`` where ``i`` is the per-base occurrence index, so
        # ``Benchmark.game_runs`` has unique IDs (Benchmark.run asserts this)
        # while each instance still plays its base env underneath.
        counts: collections.Counter[str] = collections.Counter()
        game_ids = []
        external_ids = []
        for i in range(110):
            base = official_ids[i % len(official_ids)]
            game_ids.append(base)
            external_ids.append(f"{base}_{counts[base]}")
            counts[base] += 1
    games: list[taaf.game.Game]
    if arcade_spec is None:
        games = [taaf.game_api.GameAPI(env_name=g, external_game_id=ext) for g, ext in zip(game_ids, external_ids)]
    else:
        games = [
            taaf.game_api.GameAPI(env_name=g, external_game_id=ext, arcade_spec=arcade_spec)
            for g, ext in zip(game_ids, external_ids)
        ]
    return taaf.benchmark.Benchmark(
        label="kaggle_official_110",
        games=games,
        solver=solver,
        n_passes=1,
    )
