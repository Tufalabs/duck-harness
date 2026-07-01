"""Core game abstractions (R11.01–R11.05). The R11.06 example subclass
lives in ``taaf.game_examples``.

This module also owns the arcengine enum pickle fix (collocated with the
only module that imports arcengine) and the ARC color palette.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import warnings
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeVar

import arcengine
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

T = TypeVar("T")

# arcengine's GameAction / GameState enums use composite keys in
# ``_value2member_map_``, so the default ``Enum(value)`` reconstruction
# fails on unpickle. Reduce by name instead.


def _reconstruct_enum(cls: type, name: str) -> Any:
    return cls[name]  # type: ignore[index]


def _enum_reduce(self: Any, protocol: int = 0) -> Any:
    return (_reconstruct_enum, (self.__class__, self._name_))


arcengine.enums.GameAction.__reduce_ex__ = _enum_reduce  # type: ignore[attr-defined]
arcengine.enums.GameState.__reduce_ex__ = _enum_reduce  # type: ignore[attr-defined]


# --- ARC palette -------------------------------------------------------------

ARC_COLORS: dict[int, tuple[float, float, float]] = {
    0: (1.0, 1.0, 1.0),
    1: (0.8, 0.8, 0.8),
    2: (0.6, 0.6, 0.6),
    3: (0.4, 0.4, 0.4),
    4: (0.2, 0.2, 0.2),
    5: (0.0, 0.0, 0.0),
    6: (0.898, 0.227, 0.639),
    7: (1.0, 0.482, 0.8),
    8: (0.976, 0.235, 0.192),
    9: (0.118, 0.576, 1.0),
    10: (0.533, 0.847, 0.945),
    11: (1.0, 0.863, 0.0),
    12: (1.0, 0.522, 0.106),
    13: (0.573, 0.071, 0.192),
    14: (0.310, 0.800, 0.188),
    15: (0.639, 0.337, 0.839),
}
ARC_CMAP = mcolors.ListedColormap([ARC_COLORS[i] for i in range(16)])


# --- Frame -------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A single visible frame as a 2D int8 numpy array, values 0-15 (ARC palette)."""

    data: npt.NDArray[np.int8]

    def __post_init__(self) -> None:
        assert self.data.dtype == np.int8, f"Frame data must be int8, got {self.data.dtype}"
        assert self.data.ndim == 2, f"Frame data must be 2D, got shape {self.data.shape}"
        assert np.all((self.data >= 0) & (self.data <= 15)), "Frame data must contain values 0-15 only"

    def draw(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Draw this frame on ``ax``, or create a fresh axes if None.

        Returns the axes for chaining/inspection.
        """
        if ax is None:
            h, w = self.data.shape
            scale = 4.0 / max(h, w)
            _fig, ax = plt.subplots(figsize=(w * scale, h * scale))
        ax.imshow(self.data, cmap=ARC_CMAP, vmin=0, vmax=15, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax


# --- RunSession --------------------------------------------------------------


@dataclass
class RunSession:
    """Per-``Benchmark.run()`` resource cache. ``Game`` subclasses with
    shared resources (notably ``GameAPI``'s ``arc_agi.Arcade``) call
    ``get_or_make`` from ``_start_game``. Owned by ``Benchmark.run``'s
    frame, never pickled. Standalone callers (tests, notebooks) get a
    fresh session per ``Game.start_game()`` call.

    The lock is forward-compat insurance — ``Benchmark.run`` calls
    ``start_game`` serially today.
    """

    record_intermediate_states: bool = True

    _resources: dict[Hashable, Any] = field(default_factory=lambda: dict[Hashable, Any](), repr=False)
    _closeables: list[Any] = field(default_factory=lambda: list[Any](), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Separate from ``_lock`` so a factory running inside ``get_or_make``
    # (which holds ``_lock``) can call ``register_closeable`` without
    # deadlocking — GameAPI's competition-sim build does exactly that.
    _closeables_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_or_make(self, key: Hashable, factory: Callable[[], T]) -> T:
        with self._lock:
            if key not in self._resources:
                self._resources[key] = factory()
            return self._resources[key]  # type: ignore[no-any-return]

    def register_closeable(self, resource: Any) -> None:
        """Register a resource exposing ``stop()`` to be torn down by
        ``close()`` at end of run (e.g. a local competition-arcade server)."""
        with self._closeables_lock:
            self._closeables.append(resource)

    def close(self) -> None:
        """Stop every registered closeable, most-recent first. Called by
        ``Benchmark.run`` at teardown, so it must not raise."""
        with self._closeables_lock:
            closeables = self._closeables[::-1]
            self._closeables = []
        for resource in closeables:
            try:
                resource.stop()
            except Exception as exc:  # noqa: BLE001 — teardown must not raise
                warnings.warn(f"RunSession.close: failed to stop {resource!r}: {exc!r}", stacklevel=2)


# --- GameState (R11.01) ------------------------------------------------------


@dataclass(frozen=True)
class GameState:
    """Wraps ``arcengine.FrameDataRaw`` with helper properties (R11.01).

    Fields:

    - ``raw``: the underlying arcengine frame data.
    - ``previous_action``: the ``ActionInput`` that led here, or ``None``
      for the initial state.
    - ``just_won_level``: True if this state crossed a level boundary.
    """

    raw: arcengine.FrameDataRaw
    previous_action: arcengine.ActionInput | None = None
    just_won_level: bool = False

    @property
    def frame(self) -> Frame:
        """The final visible frame (last in the raw frame list)."""
        return Frame(data=self.raw.frame[-1])

    @property
    def animation_frames(self) -> list[Frame]:
        """All frames except the final visible one."""
        return [Frame(data=f) for f in self.raw.frame[:-1]]

    @property
    def all_frames(self) -> list[Frame]:
        """All frames (animation + final)."""
        return [Frame(data=f) for f in self.raw.frame]

    @property
    def available_actions(self) -> list[int]:
        """Legal action ids, with RESET (0) always present."""
        raw = list(self.raw.available_actions)
        if 0 in raw:
            return raw
        return [0, *raw]

    @property
    def levels_completed(self) -> int:
        return self.raw.levels_completed

    @property
    def game_over(self) -> bool:
        """True if the game has actually ended.

        Synonymous with ``won`` in this framework: arcengine's ``GAME_OVER``
        is a recoverable dead-end-of-attempt (``RESET`` recovers), so only
        ``WIN`` is a real end state. R11.01 keeps both as separate
        properties for the conceptual distinction.
        """
        return self.raw.state == arcengine.GameState.WIN

    @property
    def won(self) -> bool:
        """True if the game ended in a win. Synonym for ``game_over``."""
        return self.raw.state == arcengine.GameState.WIN


# --- ActionRecord ------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One executed action with token + wallclock cost (R11.02).

    Fields:

    - ``action``: the executed ``ActionInput``.
    - ``generated_tokens`` / ``uncached_input_tokens``: solver-reported
      token counts, or 0 if the solver doesn't track them (R11.03).
    - ``wallclock_seconds``: monotonic-clock seconds since
      ``Game.start_game()``.
    """

    action: arcengine.ActionInput
    generated_tokens: int
    uncached_input_tokens: int
    wallclock_seconds: float


# --- GameRun (R11.02) --------------------------------------------------------

GameRunState = Literal["not_started", "playing", "won", "gave_up", "cancelled", "crashed"]


@dataclass
class GameRun:
    """Per-game run state (R11.02).

    State machine::

        not_started → playing               on Game.start_game()
        playing     → won                   on execute_action returning won=True
        playing     → cancelled             on finish_game with asyncio cancel outstanding
        playing     → gave_up               on finish_game otherwise
        playing     → crashed               Benchmark teardown, finish_game never called

    Fields:

    - ``game_id`` / ``number_of_levels`` / ``base_actions_per_level`` /
      ``hint``: basic info populated by ``Game.start_game()``.
      ``base_actions_per_level`` is ``None`` when the engine hides
      baselines (e.g. submission mode).
    - ``state`` / ``levels_completed`` / ``final_score``: progress
      summary. ``final_score`` is set by ``finish_game()``.
    - ``history`` / ``actions_per_level``: per-action record and
      per-level counter. Invariant: ``sum(actions_per_level) == len(history)``.
    - ``final_generated_tokens`` / ``final_uncached_input_tokens``: tokens
      the solver spent on its final turn *without* producing a move — the
      give-up / cancellation decision — reported via ``finish_game(...)``
      (R11.03). 0 when the solver doesn't report them. Counted in token
      totals but not as an action (``actions_per_level`` is unchanged).
    - ``final_wallclock_seconds``: monotonic-clock seconds since
      ``start_game()`` at the moment ``finish_game()`` ran — the same
      reference and duration semantics as ``ActionRecord.wallclock_seconds``
      (so it sits on the diagnostics wallclock axis, and survives JSON as a
      duration). Lets the no-move ``final_*_tokens`` be placed at the time
      they were actually spent, after the last move.
    - ``intermediate_states``: full per-step ``GameState`` list. Kept
      out of JSON per R11.02; persisted to ``intermediate_states.pkl``
      sidecar.
    - ``record_intermediate_states``: when false, only history and score
      bookkeeping are kept. Kaggle submission uses this to avoid storing
      frames in memory.
    - ``solver_note`` / ``solver_analysis_html``: solver-provided
      diagnostics. Either can be re-set during play; the periodic save
      loop picks up the latest values. ``solver_analysis_html`` is a
      path **relative to job_dir**, linked from the movie wrapper page.
    - ``started_at`` / ``started_at_monotonic``: wall-clock start stamps,
      both populated at ``Game.start_game()``. ``started_at`` is a
      local-naive ``datetime`` that survives JSON — diagnostics offsets
      it against ``Benchmark.start_time`` to place each action on a
      true job-wallclock axis (games queued behind a pool semaphore
      don't collapse to benchmark t=0). ``started_at_monotonic`` is a
      ``time.monotonic()`` reading used purely as the reference for
      ``ActionRecord.wallclock_seconds``; immune to NTP / DST jumps
      mid-run, but not persisted (monotonic epochs don't survive a
      process boundary).
    """

    game_id: str
    number_of_levels: int
    base_actions_per_level: list[int] | None
    hint: str | None = None
    state: GameRunState = "not_started"
    history: list[ActionRecord] = field(default_factory=lambda: list[ActionRecord]())
    intermediate_states: list[GameState] = field(default_factory=lambda: list[GameState](), repr=False)
    record_intermediate_states: bool = field(default=True, repr=False)
    actions_per_level: list[int] = field(default_factory=lambda: list[int]())
    levels_completed: int = 0
    started_at: datetime | None = None
    started_at_monotonic: float | None = field(default=None, repr=False)
    final_score: float | None = None
    solver_note: str | None = None
    solver_analysis_html: str | None = None
    final_generated_tokens: int = 0
    final_uncached_input_tokens: int = 0
    final_wallclock_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.actions_per_level:
            self.actions_per_level = [0] * self.number_of_levels

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict. ``intermediate_states`` is
        dropped per R11.02 (persisted to a sidecar pickle instead)."""
        return {
            "game_id": self.game_id,
            "number_of_levels": self.number_of_levels,
            "base_actions_per_level": self.base_actions_per_level,
            "hint": self.hint,
            "state": self.state,
            "history": [
                {
                    "action": {"id": rec.action.id.name, "data": dict(rec.action.data)},
                    "generated_tokens": rec.generated_tokens,
                    "uncached_input_tokens": rec.uncached_input_tokens,
                    "wallclock_seconds": rec.wallclock_seconds,
                }
                for rec in self.history
            ],
            "record_intermediate_states": self.record_intermediate_states,
            "actions_per_level": list(self.actions_per_level),
            "levels_completed": self.levels_completed,
            "final_score": self.final_score,
            "solver_note": self.solver_note,
            "solver_analysis_html": self.solver_analysis_html,
            "final_generated_tokens": self.final_generated_tokens,
            "final_uncached_input_tokens": self.final_uncached_input_tokens,
            "final_wallclock_seconds": self.final_wallclock_seconds,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> GameRun:
        started_at_raw = d.get("started_at")
        run = cls(
            game_id=d["game_id"],
            number_of_levels=d["number_of_levels"],
            base_actions_per_level=d["base_actions_per_level"],
            hint=d.get("hint"),
            state=d["state"],
            actions_per_level=list(d.get("actions_per_level") or []),
            record_intermediate_states=bool(d.get("record_intermediate_states", True)),
            levels_completed=d.get("levels_completed", 0),
            final_score=d.get("final_score"),
            solver_note=d.get("solver_note"),
            solver_analysis_html=d.get("solver_analysis_html"),
            final_generated_tokens=d.get("final_generated_tokens", 0),
            final_uncached_input_tokens=d.get("final_uncached_input_tokens", 0),
            final_wallclock_seconds=d.get("final_wallclock_seconds", 0.0),
            started_at=datetime.fromisoformat(started_at_raw) if started_at_raw else None,
        )
        for entry in d["history"]:
            action = arcengine.ActionInput(
                id=arcengine.GameAction[entry["action"]["id"]],
                data=dict(entry["action"]["data"]),
            )
            run.history.append(
                ActionRecord(
                    action=action,
                    generated_tokens=entry["generated_tokens"],
                    uncached_input_tokens=entry["uncached_input_tokens"],
                    wallclock_seconds=entry["wallclock_seconds"],
                )
            )
        return run

    def _compute_final_score(self) -> float:
        """ARC-AGI3 score formula. Mirrors
        ``arc_agi.scorecard.EnvironmentScoreCalculator`` (v0.9.8).

        Per level: ``min(115, (baseline / actions)² × 100)`` if completed
        with ≥ 1 action, else 0. Weights are 1-indexed (level 0 → 1).
        Final score is ``total_score / total_weights`` capped at
        ``max_weights / total_weights × 100``, where ``max_weights``
        sums only levels that scored > 0.
        """
        if self.base_actions_per_level is None or self.number_of_levels == 0:
            return 0.0
        total_score = 0.0
        total_weights = 0
        max_weights = 0
        for level_idx in range(self.number_of_levels):
            weight = level_idx + 1
            total_weights += weight
            completed = level_idx < self.levels_completed
            actions = self.actions_per_level[level_idx] if level_idx < len(self.actions_per_level) else 0
            baseline = self.base_actions_per_level[level_idx]
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


# --- Game (R11.03 / R11.04 / R11.05) -----------------------------------------


@dataclass
class Game:
    """Abstract game interface (R11.03 / R11.04). Subclasses implement
    ``_start_game`` / ``_execute_action`` / ``_finish_game``; the public
    methods on this class handle ``GameRun`` bookkeeping, action
    validation, token logging, and the state machine.

    R11.05 pickle / deepcopy contract: pre-``start_game`` the default
    walk over ``__dict__`` runs as usual. Post-``start_game``,
    ``__getstate__`` raises and ``__deepcopy__`` raises unless
    ``allow_deepcopy=True`` is set on the instance.

    Fields:

    - ``allow_deepcopy``: opt-in for post-start deepcopy. Default
      ``False`` so subclasses that *can* support it still require an
      explicit instance-level toggle.
    - ``game_id`` / ``number_of_levels`` / ``base_actions_per_level`` /
      ``hint`` / ``grid_size``: populated by ``_start_game()``.
      ``grid_size`` is ``(width, height)`` to match the (x, y) order
      of CLICK / ACTION6 data. ``base_actions_per_level`` is ``None``
      when the engine hides baselines (submission mode).
    - ``game_run``: the live ``GameRun``, or ``None`` before
      ``start_game``.
    """

    allow_deepcopy: bool = field(default=False, kw_only=True)
    game_id: str = field(default="", init=False)
    number_of_levels: int = field(default=0, init=False)
    base_actions_per_level: list[int] | None = field(default=None, init=False)
    hint: str | None = field(default=None, init=False)
    grid_size: tuple[int, int] = field(default=(64, 64), init=False)
    game_run: GameRun | None = field(default=None, init=False, repr=False)
    _current_state: GameState | None = field(default=None, init=False, repr=False)

    @property
    def current_state(self) -> GameState:
        assert self._current_state is not None, "current_state requires start_game()"
        return self._current_state

    def start_game(self, session: RunSession | None = None) -> GameState:
        """Start the game, populate basic info, return initial state (R11.03).

        ``session`` carries shared per-run resources for subclasses that
        need them. Standalone callers can omit it; a fresh session is
        constructed on the spot.
        """
        assert self.game_run is None, "start_game() already called"
        if session is None:
            session = RunSession()
        initial = self._start_game(session)
        # Verify the subclass populated the R11.03 basic info.
        # ``base_actions_per_level`` may legitimately be None (R11.02:
        # engine hides baselines in submission mode).
        assert self.game_id, f"_start_game() must populate game_id (got {self.game_id!r})"
        assert self.number_of_levels > 0, (
            f"_start_game() must populate number_of_levels > 0 (got {self.number_of_levels})"
        )
        if self.base_actions_per_level is not None:
            assert len(self.base_actions_per_level) == self.number_of_levels, (
                f"base_actions_per_level has {len(self.base_actions_per_level)} entries; "
                f"number_of_levels is {self.number_of_levels}"
            )
        self.game_run = GameRun(
            game_id=self.game_id,
            number_of_levels=self.number_of_levels,
            base_actions_per_level=self.base_actions_per_level,
            hint=self.hint,
            state="playing",
            record_intermediate_states=session.record_intermediate_states,
            started_at=datetime.now(),
            started_at_monotonic=time.monotonic(),
        )
        if self.game_run.record_intermediate_states:
            self.game_run.intermediate_states.append(initial)
        self._current_state = initial
        return initial

    def execute_action(
        self,
        action: arcengine.ActionInput,
        generated_tokens: int = 0,
        uncached_input_tokens: int = 0,
    ) -> GameState:
        """Execute a move (R11.03). Validates the action, logs an
        ``ActionRecord``, then delegates to ``_execute_action``. Raises
        ``ValueError`` for disallowed actions; ``_execute_action`` may
        raise its own exceptions (e.g. ``GameAPI`` raises on
        engine-rejected moves).
        """
        assert self.game_run is not None, "Call start_game() first"
        assert self.game_run.state == "playing", f"Cannot execute_action in state {self.game_run.state!r}"
        valid = self.current_state.available_actions
        if action.id.value not in valid:
            raise ValueError(f"Action {action.id.name} (id={action.id.value}) not in available_actions {valid}")
        action.id.validate_data(action.data)  # arcengine's own pydantic shape check
        if action.id.is_simple() and action.data:
            raise ValueError(f"Simple action {action.id.name} must not carry data; got {action.data}")
        if action.id.is_complex():
            x = action.data["x"]
            y = action.data["y"]
            w, h = self.grid_size
            if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < w and 0 <= y < h):
                raise ValueError(
                    f"Action {action.id.name} coordinates ({x}, {y}) out of bounds for grid_size {self.grid_size}"
                )

        assert self.game_run.started_at_monotonic is not None
        elapsed = time.monotonic() - self.game_run.started_at_monotonic
        level_before = self.current_state.levels_completed

        # Run the subclass hook *before* any bookkeeping so a raise leaves
        # the run in a consistent state (no orphan history entry, no partial
        # actions_per_level increment, no missing intermediate_states).
        new_state_inner = self._execute_action(action)

        just_won_level = new_state_inner.levels_completed > level_before

        # Manual green frame inserted if a level transition produced only a
        # single frame — arcengine normally emits at least two on
        # transitions; this is a safety net for less-conformant games.
        if just_won_level and not new_state_inner.won and len(new_state_inner.raw.frame) <= 1:
            w, h = self.grid_size
            green_frame = np.full((h, w), 14, dtype=np.int8)
            new_state_inner.raw.frame = [green_frame, *new_state_inner.raw.frame]

        # Re-wrap to stamp previous_action / just_won_level — don't trust
        # the subclass to set them.
        new_state = GameState(
            raw=new_state_inner.raw,
            previous_action=action,
            just_won_level=just_won_level,
        )

        assert level_before < self.number_of_levels, (
            f"level_before={level_before} >= number_of_levels={self.number_of_levels}: "
            "subclass returned a state with levels_completed past number_of_levels "
            "without also setting arcengine.GameState.WIN"
        )

        # Commit all bookkeeping atomically.
        self.game_run.history.append(
            ActionRecord(
                action=action,
                generated_tokens=generated_tokens,
                uncached_input_tokens=uncached_input_tokens,
                wallclock_seconds=elapsed,
            )
        )
        if self.game_run.record_intermediate_states:
            self.game_run.intermediate_states.append(new_state)
        self._current_state = new_state
        self.game_run.actions_per_level[level_before] += 1

        if new_state.levels_completed > self.game_run.levels_completed:
            self.game_run.levels_completed = new_state.levels_completed

        if new_state.won:
            self.game_run.state = "won"

        # Bookkeeping invariants — every action produces exactly one
        # history entry, one increment to actions_per_level, and one
        # intermediate state. The score formula depends on these agreeing.
        assert sum(self.game_run.actions_per_level) == len(self.game_run.history), (
            f"sum(actions_per_level)={sum(self.game_run.actions_per_level)} "
            f"!= len(history)={len(self.game_run.history)}"
        )
        if self.game_run.record_intermediate_states:
            assert len(self.game_run.intermediate_states) == len(self.game_run.history) + 1, (
                f"len(intermediate_states)={len(self.game_run.intermediate_states)} "
                f"!= len(history)+1={len(self.game_run.history) + 1}"
            )
        assert self.game_run.levels_completed <= self.number_of_levels, (
            f"levels_completed={self.game_run.levels_completed} > number_of_levels={self.number_of_levels}"
        )
        return new_state

    def finish_game(self, generated_tokens: int = 0, uncached_input_tokens: int = 0) -> None:
        """Mark the game as finished (R11.03 / R11.02 state machine).

        If the game is still ``playing``, transitions to ``cancelled``
        when an asyncio cancel is outstanding on the current task, else
        ``gave_up``. ``Benchmark.run`` sets the state to ``crashed``
        before calling this in its teardown path. Idempotent.

        ``generated_tokens`` / ``uncached_input_tokens`` (R11.03) let the
        solver report tokens it spent on this final turn *without* making a
        move — the give-up / cancellation decision. They land on the run's
        ``final_*_tokens`` fields and count toward token totals, but not as
        an action. The finish time is stamped into ``final_wallclock_seconds``
        (monotonic elapsed since ``start_game``, matching
        ``ActionRecord.wallclock_seconds``) so those tokens land at the time
        they were spent on the wallclock axis. Recorded on the first
        (non-idempotent) call only.
        """
        assert self.game_run is not None, "finish_game() before start_game()"
        if self.game_run.final_score is not None:
            return
        self.game_run.final_generated_tokens = generated_tokens
        self.game_run.final_uncached_input_tokens = uncached_input_tokens
        if self.game_run.started_at_monotonic is not None:
            self.game_run.final_wallclock_seconds = time.monotonic() - self.game_run.started_at_monotonic
        self._finish_game()
        if self.game_run.state == "playing":
            cancelling = False
            try:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    cancelling = True
            except RuntimeError:
                # No running loop — sync caller, treat as gave_up.
                cancelling = False
            self.game_run.state = "cancelled" if cancelling else "gave_up"
        self.game_run.final_score = self.game_run._compute_final_score()
        # One-line per-game finish note to stdout — same fields as the
        # per-pass row in the diagnostics HTML, plus per-level
        # actions/baseline so the score is auditable from the log alone.
        run = self.game_run
        actions = sum(run.actions_per_level)
        tokens = sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
        per_level = ",".join(
            f"{run.actions_per_level[i]}/"
            f"{run.base_actions_per_level[i] if run.base_actions_per_level is not None else '?'}"
            for i in range(run.number_of_levels)
        )
        note_suffix = f' note="{run.solver_note}"' if run.solver_note else ""
        print(
            f"[finished] {run.game_id} state={run.state} "
            f"level={run.levels_completed}/{run.number_of_levels} "
            f"score={run.final_score:.2f} actions={actions} tokens={tokens} "
            f"per-level={per_level}{note_suffix}"
        )

    # --- pickle / deepcopy contract (R11.05) --------------------------------

    def __getstate__(self) -> dict[str, Any]:
        if self.game_run is not None:
            raise RuntimeError(
                f"Cannot pickle {type(self).__name__} after start_game(). "
                "R11.05 contract: pickle is forbidden post-start."
            )
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __deepcopy__(self, memo: dict[int, Any]) -> Game:
        if self.game_run is not None and not self.allow_deepcopy:
            raise RuntimeError(
                f"Cannot deepcopy {type(self).__name__} after start_game() unless "
                "allow_deepcopy=True. R11.05: deepcopy is the only post-start "
                "lifecycle step that subclasses can opt in to."
            )
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            object.__setattr__(new, k, copy.deepcopy(v, memo))
        return new

    # --- subclass hooks (R11.04) --------------------------------------------

    def _start_game(self, session: RunSession) -> GameState:
        raise NotImplementedError("Subclasses must implement _start_game")

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        raise NotImplementedError("Subclasses must implement _execute_action")

    def _finish_game(self) -> None:
        """Optional teardown. Default: no-op."""
