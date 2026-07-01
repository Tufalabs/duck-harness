"""Example ``Game`` subclasses (R11.06). Production subclasses live in
``taaf.game_api``."""

from __future__ import annotations

from dataclasses import dataclass, field

import arcengine
import numpy as np

from taaf.game import Game, GameState, RunSession


def _default_target_sequence() -> list[list[arcengine.GameAction]]:
    return [
        [arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION1],
        [arcengine.GameAction.ACTION2, arcengine.GameAction.ACTION2, arcengine.GameAction.ACTION2],
    ]


@dataclass
class ExampleGame(Game):
    """Trivial deterministic multi-level game (R11.06).

    Each level requires pressing a fixed target sequence in order; a wrong
    action or RESET zeros the in-level progress (but not the level
    counter). Defaults to two levels (``[ACTION1×3]``, ``[ACTION2×3]``)
    with ``RESET`` / ``ACTION1`` / ``ACTION2`` available — narrow enough
    that ``SolverRandom`` finishes in bounded time. Frame is a 64×64 int8
    grid uniformly filled with ``(level * 4 + progress) % 16``, or 14
    (green) on win.
    """

    target_sequence_per_level: list[list[arcengine.GameAction]] = field(default_factory=_default_target_sequence)
    label: str = field(default="example_game", init=True)

    _current_level: int = field(default=0, init=False, repr=False)
    _progress: int = field(default=0, init=False, repr=False)

    def _start_game(self, session: RunSession) -> GameState:
        del session  # unused; ExampleGame holds no shared per-run resources
        self.game_id = self.label
        self.number_of_levels = len(self.target_sequence_per_level)
        self.base_actions_per_level = [len(seq) for seq in self.target_sequence_per_level]
        self.hint = "press the target sequence per level"
        self.grid_size = (64, 64)
        self._current_level = 0
        self._progress = 0
        return self._build_state(state=arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            self._progress = 0
            return self._build_state(state=arcengine.GameState.NOT_FINISHED)
        target = self.target_sequence_per_level[self._current_level]
        if action.id == target[self._progress]:
            self._progress += 1
            if self._progress >= len(target):
                self._current_level += 1
                self._progress = 0
                if self._current_level >= len(self.target_sequence_per_level):
                    return self._build_state(state=arcengine.GameState.WIN)
        else:
            self._progress = 0
        return self._build_state(state=arcengine.GameState.NOT_FINISHED)

    def _build_state(self, state: arcengine.GameState) -> GameState:
        if state == arcengine.GameState.WIN:
            fill = 14  # ARC green
            levels_completed = len(self.target_sequence_per_level)
        else:
            fill = (self._current_level * 4 + self._progress) % 16
            levels_completed = self._current_level
        frame_arr = np.full((64, 64), fill, dtype=np.int8)
        # Union of target sequences; GameState.available_actions re-adds RESET.
        available: list[int] = sorted({a.value for seq in self.target_sequence_per_level for a in seq})
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=levels_completed,
            win_levels=len(self.target_sequence_per_level),
            available_actions=available,
        )
        raw.frame = [frame_arr]
        return GameState(raw=raw)
