"""arcengine-backed ``Game`` subclass (R11.11) with R11.12 score
reconciliation. ``ArcadeSpec`` is the picklable description of an
``arc_agi.Arcade``; ``RunSession`` caches one Arcade per unique spec.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
import threading
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import arc_agi
import arcengine

from taaf.game import Game, GameState, RunSession

# ``environments_dir = _AUTO_ENV_DIR`` was the sentinel for offline env
# files bundled with a worker deployment. Those files are not shipped in
# this build, so ``__auto__`` is rejected; pass an explicit path or run
# against the live competition Arcade.
_AUTO_ENV_DIR = "__auto__"


def _resolve_environments_dir(value: str) -> str:
    """Resolve an explicit env-files dir, or reject the ``__auto__`` sentinel.

    The ``__auto__`` mode used to locate offline env files bundled with a
    worker deployment, which are not shipped in this build. Live competition
    Arcades pass ``environments_dir=""`` and never reach this branch; offline
    play now requires an explicit ``environments_dir``.
    """
    if value == _AUTO_ENV_DIR:
        raise RuntimeError(
            "Offline env files (the '__auto__' mode) are not bundled in this build. "
            "Run against the live competition Arcade, or pass an explicit "
            "environments_dir."
        )
    return value


# ``arc_agi.Arcade``: without a ``logger=`` kwarg it installs its own
# stdout INFO handler that clobbers any post-hoc ``setLevel``. Pass a
# private logger to take its "user provided" branch — INFO drops at the
# level check; WARNING+ still propagate.
_ARCADE_LOGGER = logging.getLogger("taaf.game_api.arcade")
_ARCADE_LOGGER.setLevel(logging.WARNING)

# ``arc_agi.scorecard`` installs its own INFO handler at import time and
# disables propagation — raising the level here is the only handle.
logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)


@dataclass
class _CompetitionScorecard:
    """Shared one-scorecard lifecycle for competition-mode Arcades."""

    arcade: arc_agi.Arcade
    scorecard_id: str | None = None
    active_runs: int = 0
    closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def open_run(self) -> str:
        with self._lock:
            if self.closed:
                raise RuntimeError("competition scorecard is already closed")
            if self.scorecard_id is None:
                self.scorecard_id = self.arcade.create_scorecard()
            assert self.scorecard_id is not None
            self.active_runs += 1
            return self.scorecard_id

    def finish_run(self) -> arc_agi.EnvironmentScorecard | None:
        with self._lock:
            if self.scorecard_id is None or self.closed:
                return None
            self.active_runs -= 1
            if self.active_runs > 0:
                return None
            self.closed = True
            return self.arcade.close_scorecard(self.scorecard_id)


@dataclass(frozen=True)
class ArcadeSpec:
    """Picklable, hashable description of an ``arc_agi.Arcade``. Two specs
    that compare equal share one Arcade per ``RunSession``. New fields
    must remain hashable.

    Fields:

    - ``operation_mode``: ``OperationMode.OFFLINE`` (default) hits the
      offline bundled env files — no network, no API key. Override with
      ``ONLINE`` for ARC-AGI3-API runs.
    - ``arc_base_url``: ARC-AGI3 API URL (only used in ONLINE mode).
    - ``environments_dir``: path to env JSON files, or ``"__auto__"``
      (default) to resolve lazily on whichever process opens the game
      (necessary for worker containers — see module-level comment).
    - ``competition_sim``: when True, the lazy build starts a local
      ``CompetitionArcadeServer`` (official-110 clone set) and connects in
      ``COMPETITION`` mode instead of an offline Arcade, so a non-submission
      run exercises the submission-shaped arcade (shared scorecard, hidden
      baselines, cloned IDs) without a real submission. One server, shared
      across games via the ``RunSession`` and stopped by ``RunSession.close``.
      Stays picklable — it stores only the intent, not the live server.

    Notes:

    - ``arc_api_key`` is **not** carried — ``arc_agi.Arcade`` reads
      ``ARC_API_KEY`` from env or fetches anonymous. This keeps secrets
      out of pickled benchmarks.
    - ``recordings_dir`` is not carried because we never call
      ``arcade.make(save_recording=True)``.
    """

    operation_mode: arc_agi.OperationMode = arc_agi.OperationMode.OFFLINE
    arc_base_url: str = "https://three.arcprize.org"
    environments_dir: str = _AUTO_ENV_DIR
    competition_sim: bool = False


@dataclass
class GameAPI(Game):
    """``Game`` wrapping an ``arc_agi`` environment (R11.11). The Arcade
    is built lazily on first ``_start_game`` via the ``RunSession``
    cache, so unstarted instances are cheap and pickle-safe.

    R11.12 reconciliation: each instance gets its own scorecard so
    ``_finish_game`` can pull *this run's* engine score in isolation.
    Mismatches fire ``warnings.warn`` + stderr print; the whole
    reconciliation is try/except-wrapped since it runs inside
    ``Benchmark.run``'s teardown ``finally`` and must not raise.

    Fields:

    - ``env_name``: arcengine env id, e.g. ``"ls20"``.
    - ``arcade_spec``: how to build the underlying Arcade.
    - ``env``: live ``arc_agi.EnvironmentWrapper``, populated by
      ``_start_game``. Available to callers who want to introspect the
      engine directly.
    """

    env_name: str = field(kw_only=True)
    arcade_spec: ArcadeSpec = field(default_factory=ArcadeSpec, kw_only=True)
    # If set, overrides the engine's natural ``game_id`` after ``start_game``.
    # Lets a benchmark factory give every entry a distinct ``game_id`` when the
    # underlying ``env_name`` repeats (e.g. round-robin 110-game runs); the
    # engine scorecard reconciliation still looks up runs by the engine's own
    # id, not this one.
    external_game_id: str | None = field(default=None, kw_only=True)
    env: arc_agi.EnvironmentWrapper | None = field(default=None, init=False, repr=False)
    _arcade: arc_agi.Arcade | None = field(default=None, init=False, repr=False)
    _scorecard_id: str | None = field(default=None, init=False, repr=False)
    _competition_scorecard: _CompetitionScorecard | None = field(default=None, init=False, repr=False)

    def _start_game(self, session: RunSession) -> GameState:
        spec = self.arcade_spec

        def _build_arcade() -> arc_agi.Arcade:
            if spec.competition_sim:
                # Lazy import avoids a game_api <-> competition_arcade cycle.
                from taaf import competition_arcade  # noqa: PLC0415

                server = competition_arcade.CompetitionArcadeServer.official_110().start()
                session.register_closeable(server)
                return arc_agi.Arcade(
                    operation_mode=arc_agi.OperationMode.COMPETITION,
                    arc_base_url=server.base_url,
                    environments_dir="",
                    logger=_ARCADE_LOGGER,
                )
            kwargs = asdict(spec)
            kwargs.pop("competition_sim", None)
            kwargs["environments_dir"] = _resolve_environments_dir(kwargs["environments_dir"])
            return arc_agi.Arcade(**kwargs, logger=_ARCADE_LOGGER)

        arcade = session.get_or_make(spec, _build_arcade)
        self._arcade = arcade
        competition_scorecard: _CompetitionScorecard | None = None
        scorecard_opened = False
        if spec.competition_sim or spec.operation_mode == arc_agi.OperationMode.COMPETITION:
            # Submission-style Arcades allow only one scorecard. Share it
            # across every GameAPI in this RunSession, then close it after
            # the final game finishes.
            competition_scorecard = session.get_or_make(
                ("taaf.game_api.competition_scorecard", spec),
                lambda: _CompetitionScorecard(arcade),
            )
            self._competition_scorecard = competition_scorecard
            self._scorecard_id = competition_scorecard.open_run()
            scorecard_opened = True
        else:
            # Per-game scorecard so R11.12 can fetch this run's score in
            # isolation rather than disambiguating across passes/games
            # sharing the Arcade's default scorecard.
            self._scorecard_id = arcade.create_scorecard()

        try:
            env = arcade.make(self.env_name, scorecard_id=self._scorecard_id)
        except Exception:
            if competition_scorecard is not None and scorecard_opened:
                competition_scorecard.finish_run()
            raise
        if env is None:
            if competition_scorecard is not None and scorecard_opened:
                competition_scorecard.finish_run()
            raise RuntimeError(f"arc_agi.Arcade.make({self.env_name!r}) returned None")
        self.env = env
        # Without ONLY_RESET_LEVELS, arcengine's handle_reset full-resets to
        # level 0 whenever _action_count == 0 — and set_level zeros that
        # counter after every transition. So a mid-play RESET on level
        # N > 0 with no actions taken on the new level snaps the player
        # back to level 1. Set AFTER arcade.make so the make-time RESET
        # still full-resets and registers via new_play (which R11.12
        # reconciliation needs). Process-wide and idempotent.
        os.environ["ONLY_RESET_LEVELS"] = "true"
        info = env.environment_info
        initial = env.observation_space
        if initial is None:
            if competition_scorecard is not None and scorecard_opened:
                competition_scorecard.finish_run()
            raise RuntimeError(f"env {self.env_name!r} observation_space is None after make()")
        self.game_id = info.game_id if self.external_game_id is None else self.external_game_id
        # win_levels lives on FrameDataRaw, not EnvironmentInfo.
        self.number_of_levels = initial.win_levels
        if info.baseline_actions:
            self.base_actions_per_level = list(info.baseline_actions)
        else:
            # R11.02 permits None (submission mode hides baselines). Until
            # submission lands, an empty response is an engine bug — warn
            # loudly so the run doesn't silently degrade to zero partial credit.
            warnings.warn(
                f"arcengine returned no baseline_actions for env {self.env_name!r}; "
                f"falling back to base_actions_per_level=None (zero partial credit). "
                f"This is expected only in submission mode.",
                stacklevel=2,
            )
            self.base_actions_per_level = None
        self.grid_size = (64, 64)
        return GameState(raw=initial)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        assert self.env is not None, "_execute_action before _start_game"
        resp = self.env.step(action.id, data=dict(action.data))
        if resp is None:
            raise RuntimeError(f"env.step returned None for action {action.id.name}")
        # Empty frame = engine refused to advance (typically a non-RESET
        # action after GAME_OVER). Solvers must check the engine state
        # and issue RESET instead.
        if not resp.frame:
            raise RuntimeError(
                f"arcengine returned an empty frame for action {action.id.name} "
                f"(env state {resp.state.name if hasattr(resp, 'state') else '?'}). "
                f"This typically means a non-RESET action was issued after GAME_OVER. "
                f"Solvers must check ``state.raw.state == arcengine.GameState.GAME_OVER`` "
                f"and issue RESET when it fires."
            )
        return GameState(raw=resp)

    def _finish_game(self) -> None:
        """R11.12 reconciliation. Defensive — never raises."""
        try:
            if self._arcade is None or self._scorecard_id is None or self.game_run is None:
                return
            if self._competition_scorecard is not None:
                self._competition_scorecard.finish_run()
                return
            framework_score = self.game_run._compute_final_score()
            engine_scorecard = self._arcade.close_scorecard(self._scorecard_id)
            if engine_scorecard is None:
                return
            # find_environment wants the engine's own game_id, not our
            # possibly-overridden ``external_game_id`` view.
            engine_game_id = self.env.environment_info.game_id if self.env is not None else self.game_id
            env_list = engine_scorecard.find_environment(engine_game_id)
            if env_list is None or not env_list.runs:
                return
            engine_score = env_list.runs[0].score
            if not math.isclose(framework_score, engine_score, abs_tol=1e-6):
                msg = (
                    f"R11.12: score mismatch for {self.game_id}: "
                    f"framework={framework_score:.4f}, engine={engine_score:.4f}"
                )
                warnings.warn(msg, stacklevel=2)
                print(f"[GameAPI] {msg}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — must not raise out of teardown finally
            warnings.warn(
                f"R11.12 reconciliation failed for {self.game_id}: {e!r}",
                stacklevel=2,
            )

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()  # ABC's pre-start check fires first
        # Drop live values; keep keys = None so attribute access still works.
        state["_arcade"] = None
        state["_scorecard_id"] = None
        state["_competition_scorecard"] = None
        state["env"] = None
        return state

    def __deepcopy__(self, memo: dict[int, Any]) -> GameAPI:
        """Deepcopy with ``_arcade`` shared by reference — ``arc_agi.Arcade``
        holds a ``requests.Session()`` and ``threading.Lock()``, neither
        deepcopy-safe."""
        if self.game_run is not None and not self.allow_deepcopy:
            raise RuntimeError(
                f"Cannot deepcopy {type(self).__name__} after start_game() unless allow_deepcopy=True. R11.05 contract."
            )
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in {"_arcade", "_competition_scorecard"}:
                object.__setattr__(new, k, v)
            else:
                object.__setattr__(new, k, copy.deepcopy(v, memo))
        return new
