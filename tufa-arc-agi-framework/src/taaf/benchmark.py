"""``Benchmark`` orchestration (R2.1–R2.15). Deployment (R2.21–R2.39)
lives in ``taaf.deploy*``."""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import taaf.game
import taaf.solver
import taaf.support

if TYPE_CHECKING:
    import taaf.deploy


@dataclass
class Benchmark:
    """Benchmark configuration + run state (R2.1).

    Fields:

    - ``label``: short identifier used in plots and filenames.
    - ``games``: the games to play.
    - ``solver``: the solver instance. Must be set before ``run()``.
    - ``n_passes``: number of independent copies of each game played by
      the solver (R2.2).
    - ``job_dir``: output directory. ``None`` disables JSON save and
      diagnostics generation entirely — useful for in-process tests.
    - ``game_weights``: optional per-game weights parallel to ``games``.
      ``None`` means equal weights. Weight 0 excludes a game from
      cross-game diagnostics aggregation but keeps it in per-game
      drill-downs.
    - ``game_runs``: populated by ``run()``. Ordered passes-major:
      ``game_runs[pass_idx * n_games + g]`` is pass ``pass_idx`` of game
      ``g``.
    - ``solver_label``: copied from the played solver after ``setup()``.
    - ``start_time`` / ``end_time``: local-time naive ``datetime`` from
      the most recent ``run()``. Round-trip through JSON as ISO-8601.
    - ``periodic_save_interval_s``: how often the periodic save loop
      fires. Default 600 s per R2.13; tunable for tests.
    """

    label: str = ""
    games: list[taaf.game.Game] = field(default_factory=lambda: list[taaf.game.Game]())
    solver: taaf.solver.Solver | None = None
    n_passes: int = 1
    job_dir: Path | None = None
    game_weights: list[float] | None = None

    game_runs: list[taaf.game.GameRun] = field(default_factory=lambda: list[taaf.game.GameRun](), init=False)
    solver_label: str = field(default="", init=False)
    start_time: datetime | None = field(default=None, init=False)
    end_time: datetime | None = field(default=None, init=False)
    periodic_save_interval_s: float = field(default=600.0, init=True, repr=False)

    # Internal cancellation flag — lets run() distinguish deadline-fired
    # CancelledError (swallow) from caller-cancellation (re-raise).
    _deadline_fired: bool = field(default=False, init=False, repr=False)
    # Live solver task while run() is awaiting it; lets request_stop()
    # cancel cooperatively from a signal handler. ``compare=False`` to
    # preserve dataclass equality; always None at pickle time.
    _solver_task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False, compare=False)

    async def run(
        self,
        soft_end_time: datetime | None = None,
        runtime_environment: taaf.deploy.DeploymentTarget | None = None,
        minimal_diagnostics: bool = False,
    ) -> None:
        """Run the solver on ``n_passes`` deepcopies of each game.

        ``soft_end_time`` (R2.12): when given, the solver task is
        cancelled at that moment; solvers respond by calling
        ``finish_game()`` (R12.02) which marks each game ``cancelled``.
        Timezone-aware or naive (local clock).

        ``runtime_environment`` (R12.01): the ``DeploymentTarget`` this
        benchmark is running under, stamped on the solver before
        ``setup()``. ``None`` for direct calls outside a target.

        On solver exception (R2.15): swallowed; teardown continues. On
        caller-cancellation: teardown then re-raises ``CancelledError``.
        At teardown any game still in ``playing`` is marked ``crashed``,
        JSON + sidecars + diagnostics are written when ``job_dir`` is set.
        ``minimal_diagnostics`` is for Kaggle submission: keep stdout
        status, but skip JSON / sidecar / HTML writes and frame logging.
        Also stamped on the solver so it can trim its own diagnostics.
        """
        assert self.solver is not None, "Benchmark.solver must be set before run()"
        assert self.games, "Benchmark.games must be non-empty before run()"
        if self.game_weights is not None:
            if len(self.game_weights) != len(self.games):
                raise ValueError(
                    f"game_weights length {len(self.game_weights)} must equal games length {len(self.games)}"
                )
            for i, w in enumerate(self.game_weights):
                if w < 0:
                    raise ValueError(f"game_weights[{i}] = {w} must be >= 0")

        self.start_time = datetime.now()
        self.end_time = None

        # Capture launcher-side git overview for diagnostics. Skip if a
        # deploy target already wrote it: on a Slurm worker the
        # snapshot venv has no ``.git`` and overwriting would clobber
        # the launcher's good capture.
        if self.job_dir is not None and not (self.job_dir / "git_status.txt").exists():
            from taaf import deploy  # noqa: PLC0415

            deploy.write_git_status(self.job_dir)

        # R12.02 / R11.05: solvers and games are picklable only until
        # setup()/start_game(). Snapshot pristine copies onto ``self``
        # before setup runs; the played copies live in this frame and
        # never enter the pickle. Reassignment doesn't affect the
        # caller's references — they kept the originals.
        self.solver = copy.deepcopy(self.solver)
        self.solver.runtime_environment = runtime_environment
        self.solver.job_dir = self.job_dir
        self.solver.soft_end_time = soft_end_time
        self.solver.minimal_diagnostics = minimal_diagnostics
        played_solver = copy.deepcopy(self.solver)
        played_solver.setup()
        self.solver_label = played_solver.label

        self.games = [copy.deepcopy(g) for g in self.games]

        # R2.2: pass copies of each game to the single solver. The
        # passes-major layout described in the class docstring is set here.
        session = taaf.game.RunSession(record_intermediate_states=not minimal_diagnostics)
        to_play: list[taaf.game.Game] = []
        caller_cancelled = False
        solver_task: asyncio.Task[None] | None = None
        periodic_task: asyncio.Task[None] | None = None
        deadline_task: asyncio.Task[None] | None = None
        # Setup runs inside this try so the finally below tears down every
        # game/resource opened here — including when the duplicate-id
        # validation (or an interrupt) fires mid-setup, before the solver runs.
        try:
            for pass_idx in range(self.n_passes):
                for game in self.games:
                    game_copy = copy.deepcopy(game)
                    game_copy.start_game(session)
                    assert game_copy.game_run is not None
                    self.game_runs.append(game_copy.game_run)
                    to_play.append(game_copy)
                # game_id is populated only by start_game(), so uniqueness
                # can't be checked until the first pass has opened its games;
                # do it before opening any more (later passes legitimately
                # repeat the same ids).
                if pass_idx == 0 and len({r.game_id for r in self.game_runs}) != len(self.games):
                    raise ValueError("duplicate game_ids in benchmark.games are not allowed")

            solver_task = asyncio.create_task(played_solver.run_games(to_play))
            self._solver_task = solver_task
            if self.job_dir is not None:
                periodic_task = asyncio.create_task(
                    self._minimal_status_loop() if minimal_diagnostics else self._periodic_save_loop()
                )
            if soft_end_time is not None:
                deadline_task = asyncio.create_task(self._cancel_at(soft_end_time, solver_task))

            try:
                await solver_task
            except asyncio.CancelledError:
                if not self._deadline_fired:
                    caller_cancelled = True
            except Exception as e:
                # R2.15: solver errors are not fatal. Narrowed from
                # BaseException so KeyboardInterrupt / SystemExit /
                # GeneratorExit still propagate after teardown.
                print(
                    f"\n[Benchmark.run] solver task raised "
                    f"{type(e).__name__}; teardown will mark surviving "
                    f"games as crashed. Traceback:",
                    flush=True,
                )
                traceback.print_exc()
        finally:
            # Clear before teardown so a stray request_stop() can't
            # cancel a stale solver_task. Teardown is uncancellable.
            self._solver_task = None
            if periodic_task is not None:
                periodic_task.cancel()
            if deadline_task is not None:
                deadline_task.cancel()
            if periodic_task is not None:
                await asyncio.gather(periodic_task, return_exceptions=True)
            if deadline_task is not None:
                await asyncio.gather(deadline_task, return_exceptions=True)

            try:
                played_solver.teardown()
            except Exception:
                traceback.print_exc()

            # R11.02 ``crashed``: any game still in ``playing`` at
            # teardown. Set ``crashed`` *before* finish_game so the
            # cancelled/gave_up branch is skipped, ``_finish_game`` runs
            # (R11.12 reconciliation in GameAPI), and the finish-line
            # prints the correct state. The finish-line print must not
            # raise — we're already swallowing the original solver
            # exception here, and an unhandled raise would mask it.
            for game in to_play:
                run = game.game_run
                if run is not None and run.state == "playing":
                    run.state = "crashed"
                    try:
                        game.finish_game()
                    except Exception:
                        traceback.print_exc()
                    if run.final_score is None:
                        run.final_score = run._compute_final_score()

            # Stop any per-run resources the session started (e.g. a local
            # competition-arcade server) now that every game has finished.
            session.close()

            # Stamp end_time before the final save so it lands in JSON.
            self.end_time = datetime.now()
            if not minimal_diagnostics:
                self._save_json()
                self._save_intermediate_states()
                self._save_games()
                self._save_solver()
                self._generate_diagnostics()

            # ``from taaf import diagnostics`` (not ``import
            # taaf.diagnostics``) avoids binding ``taaf`` as a
            # function-local name, which would shadow the module-level
            # ``taaf`` for the rest of run().
            from taaf import diagnostics  # noqa: PLC0415

            print(diagnostics.run_summary_text(self))
            if self.job_dir is not None and not minimal_diagnostics:
                # Absolute path so the link is clickable in VSCode.
                print(f"diagnostics: {(self.job_dir / 'diagnostics.html').absolute()}")

        if caller_cancelled:
            raise asyncio.CancelledError()

    def request_stop(self) -> None:
        """Cancel the in-flight solver task, triggering ``run()``'s
        teardown. Idempotent; no-op outside ``run()``'s solver-await.

        Designed for an asyncio signal handler so external graceful-stop
        paths (R2.33, e.g. SIGUSR1) reuse the soft-deadline teardown.
        Unlike the deadline path, leaves ``_deadline_fired`` False so
        ``run()`` re-raises ``CancelledError`` after teardown.
        """
        task = self._solver_task
        if task is not None and not task.done():
            task.cancel()

    async def deploy(self, target: taaf.deploy.DeploymentTarget) -> taaf.deploy.DeploymentHandle:
        """Set up and run this benchmark in ``target`` (R2.21). Delegates
        to ``target.deploy(self)``. ``async`` so notebook callers can
        ``await bm.deploy(target)`` directly.
        """
        return await target.deploy(self)

    async def _periodic_save_loop(self) -> None:
        """Save JSON + sidecar(s) + diagnostics every
        ``periodic_save_interval_s`` (R2.13). Movie rendering is
        idempotent on disk, so newly-finished games accumulate movies
        sweep by sweep. Sidecar pickles other than intermediate_states
        are written only at final teardown. Also prints the same
        summary text the teardown emits, so long runs leave a periodic
        progress heartbeat in stdout/stderr."""
        from taaf import diagnostics  # noqa: PLC0415

        while True:
            await asyncio.sleep(self.periodic_save_interval_s)
            try:
                self._save_json()
                self._generate_diagnostics()
                print(diagnostics.run_summary_text(self))
            except Exception:
                traceback.print_exc()

    async def _minimal_status_loop(self) -> None:
        """Kaggle submission heartbeat: no writes, just stdout status."""
        from taaf import diagnostics  # noqa: PLC0415

        while True:
            await asyncio.sleep(self.periodic_save_interval_s)
            try:
                print(diagnostics.run_summary_text(self))
            except Exception:
                traceback.print_exc()

    async def _cancel_at(self, soft_end_time: datetime, target: asyncio.Task[Any]) -> None:
        # Match local-naive vs UTC against ``soft_end_time``'s tzinfo.
        if soft_end_time.tzinfo is None:
            now = datetime.now()
        else:
            now = datetime.now(timezone.utc)
        delay = (soft_end_time - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        self._deadline_fired = True
        target.cancel()

    def _save_json(self) -> None:
        if self.job_dir is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_json_dump(self.to_json_dict(), self.job_dir / "benchmark.json")

    # Per-payload sidecar filenames. The split keeps diagnostics-only
    # loads cheap and free of user `Game` / `Solver` imports — see
    # `from_json` for how each one is attached.
    INTERMEDIATE_STATES_PKL = "intermediate_states.pkl"
    GAMES_PKL = "games.pkl"
    SOLVER_PKL = "solver.pkl"

    def _save_intermediate_states(self) -> None:
        """Save ``intermediate_states`` to its sidecar. Called from
        teardown only — the frames are large and the periodic loop
        doesn't need them."""
        if self.job_dir is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(
            [r.intermediate_states for r in self.game_runs],
            self.job_dir / self.INTERMEDIATE_STATES_PKL,
        )

    def _load_intermediate_states(self) -> None:
        """Attach ``intermediate_states`` from the sidecar. Silent no-op
        when the sidecar is missing (in-flight runs, older runs)."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.INTERMEDIATE_STATES_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            states_by_run: list[list[taaf.game.GameState]] = pickle.load(f)
        if len(states_by_run) != len(self.game_runs):
            raise ValueError(
                f"{path}: runs length {len(states_by_run)} != benchmark.game_runs "
                f"length {len(self.game_runs)}; sidecar is stale"
            )
        for run, states in zip(self.game_runs, states_by_run, strict=True):
            run.intermediate_states = list(states)

    def _save_games(self) -> None:
        """Save the unstarted-original ``games`` list to its sidecar."""
        if self.job_dir is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(self.games, self.job_dir / self.GAMES_PKL)

    def _load_games(self) -> None:
        """Attach ``games`` from the sidecar. Silent no-op when missing.
        Requires the user's ``Game`` subclass to be importable."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.GAMES_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            self.games = pickle.load(f)

    def _save_solver(self) -> None:
        """Save the pristine pre-``setup()`` ``solver`` snapshot."""
        if self.job_dir is None or self.solver is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(self.solver, self.job_dir / self.SOLVER_PKL)

    def _load_solver(self) -> None:
        """Attach ``solver`` from the sidecar. Silent no-op when missing.
        Requires the user's ``Solver`` subclass to be importable."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.SOLVER_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            self.solver = pickle.load(f)

    def _generate_diagnostics(self) -> None:
        """R2.13: writes per-run HTML + .txt to ``job_dir``. Lazy import
        of ``taaf.diagnostics`` breaks the import cycle (diagnostics
        imports benchmark for type annotations). Prints elapsed time —
        useful for spotting when in-flight diagnostics start dominating
        the periodic-save interval on a large benchmark.
        """
        if self.job_dir is None:
            return
        from taaf import diagnostics  # noqa: PLC0415

        t0 = time.monotonic()
        diagnostics.generate_run_html(self, self.job_dir / "diagnostics.html")
        diagnostics.generate_run_summary_txt(self, self.job_dir / "summary.txt")
        print(f"benchmark: regenerated diagnostics in {self.job_dir} in {time.monotonic() - t0:.2f}s")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the benchmark snapshot. ``games`` is intentionally
        absent per R2.14 ("reconstructs the full class except the game
        list")."""
        return {
            "label": self.label,
            "n_passes": self.n_passes,
            "solver_label": self.solver_label,
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "game_weights": list(self.game_weights) if self.game_weights is not None else None,
            "periodic_save_interval_s": self.periodic_save_interval_s,
            "game_runs": [r.to_json_dict() for r in self.game_runs],
        }

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        with_intermediate_states: bool = True,
        with_games: bool = False,
        with_solver: bool = False,
    ) -> Benchmark:
        """Reconstruct a Benchmark from its JSON snapshot (R2.14).

        Each ``with_*`` flag attaches a sibling sidecar pickle:

        - ``with_intermediate_states`` (default True): the per-step
          frames sidecar. Safe to leave on (taaf / arcengine / numpy
          only) and needed for movies + level-step curves.
        - ``with_games`` / ``with_solver``: re-run payloads. Require
          the user's ``Game`` / ``Solver`` subclasses to be importable.

        Each load is a silent no-op when the sidecar is missing.
        ``job_dir`` is set to ``path.parent`` so subsequent regen and
        sidecar loaders know where to look.
        """
        with open(path) as f:
            d: dict[str, Any] = json.load(f)
        bm = cls(
            label=d["label"],
            n_passes=d["n_passes"],
            game_weights=list(d["game_weights"]) if d.get("game_weights") is not None else None,
            periodic_save_interval_s=d.get("periodic_save_interval_s", 600.0),
        )
        bm.solver_label = d.get("solver_label", "")
        if d.get("start_time"):
            bm.start_time = datetime.fromisoformat(d["start_time"])
        if d.get("end_time"):
            bm.end_time = datetime.fromisoformat(d["end_time"])
        bm.game_runs = [taaf.game.GameRun.from_json_dict(r) for r in d["game_runs"]]
        bm.job_dir = Path(path).parent
        if with_intermediate_states:
            bm._load_intermediate_states()
        if with_games:
            bm._load_games()
        if with_solver:
            bm._load_solver()
        return bm

    @classmethod
    def from_pickle(cls, path: Path) -> Benchmark:
        """Load a legacy monolithic ``benchmark.pkl`` (pre-sidecar runs).
        Back-compat shim only — ``from_json`` is the canonical entry
        point. Requires the user's ``Solver`` / ``Game`` subclasses to
        be importable.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected pickled {cls.__name__}, got {type(obj).__name__}")
        return obj
