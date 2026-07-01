"""Example ``Solver`` subclasses: ``SolverRandom`` (R12.03) plus
``SolverSequence`` for deterministic tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from dataclasses import dataclass, field
from html import escape

import arcengine
import numpy as np

import taaf.game
from taaf.solver import Solver


def _finish_remaining(games: list[taaf.game.Game]) -> None:
    """Call ``finish_game()`` on every game whose run isn't yet finalized.
    Safety net for cancel paths; one bad ``finish_game`` doesn't block the
    others.
    """
    for game in games:
        if game.game_run is not None and game.game_run.final_score is None:
            try:
                game.finish_game()
            except Exception:
                pass


@dataclass
class SolverRandom(Solver):
    """Plays a uniformly random non-RESET valid action each turn (R12.03).
    For complex actions (e.g. ACTION6 CLICK) picks a uniform pixel within
    the current frame.

    Fields:

    - ``max_actions_per_game``: optional per-game cap (default 1000).
      Reaching it calls ``finish_game()`` → state ``gave_up``. Set to
      ``None`` to rely on the game ending or the benchmark soft deadline.
    - ``delay_move``: ``asyncio.sleep`` before each action. ``0`` is a
      bare yield; setting e.g. ``0.05`` lets cancellation-path tests
      avoid races against a fast random win.
    - ``click_on_color``: when ``True``, a CLICK targets a uniformly
      chosen *color* present in the frame, then a uniform pixel of that
      color. Colors are weighted equally regardless of area, so a small
      interactable blob is as likely as the background — stronger CLICK
      signal than a uniform pixel. Default ``False`` keeps the
      uniform-pixel behavior.
    - ``seed``: each game in a ``run_games`` call gets its own RNG seeded
      ``seed + i`` (i = position in the list), so multi-pass copies get
      distinct sequences while the run is reproducible.
    - ``fake_generated_min/max`` / ``fake_uncached_min/max``: synthetic
      per-action token counts drawn from the same RNG as the action
      choice, so seed reproducibility extends to token totals.
    - ``fake_giveup_generated_tokens`` / ``fake_giveup_uncached_input_tokens``:
      synthetic tokens reported via ``finish_game(...)`` when the game ends
      without a win (give-up / cancel) — the no-move-turn cost (R11.03).
      Default 0 reports nothing.
    - ``stop_on_error``: when ``False`` (default), an exception in one
      game is logged to its ``solver_note``, ``finish_game()`` is called
      (state ``gave_up``) and the other games keep playing. ``True``
      re-raises, letting ``Benchmark.run`` mark every surviving game
      ``crashed`` at teardown.
    """

    label: str = "SolverRandom"
    max_actions_per_game: int | None = 1000
    delay_move: float = 0.0
    click_on_color: bool = False
    seed: int = 0
    fake_generated_min: int = 50
    fake_generated_max: int = 500
    fake_uncached_min: int = 40
    fake_uncached_max: int = 200
    fake_giveup_generated_tokens: int = 0
    fake_giveup_uncached_input_tokens: int = 0
    stop_on_error: bool = False

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        try:
            await asyncio.gather(
                *(self._play_one(game, random.Random(self.seed + i), self.seed + i) for i, game in enumerate(games))
            )
        except asyncio.CancelledError:
            _finish_remaining(games)
            raise

    async def _play_one(self, game: taaf.game.Game, rng: random.Random, seed_used: int) -> None:
        try:
            # solver_note is picked up by periodic in-flight saves too.
            if game.game_run is not None:
                game.game_run.solver_note = f"used seed {seed_used}"
            actions_taken = 0
            while True:
                await asyncio.sleep(self.delay_move)
                run = game.game_run
                if run is None or run.state != "playing":
                    break
                if self.max_actions_per_game is not None and actions_taken >= self.max_actions_per_game:
                    break
                action = self._pick_action(game, rng)
                game.execute_action(
                    action,
                    generated_tokens=rng.randint(self.fake_generated_min, self.fake_generated_max),
                    uncached_input_tokens=rng.randint(self.fake_uncached_min, self.fake_uncached_max),
                )
                actions_taken += 1
            self._finish(game)
        except asyncio.CancelledError:
            self._finish(game)
            raise
        except Exception as exc:
            if self.stop_on_error:
                raise
            if game.game_run is not None:
                game.game_run.solver_note = f"used seed {seed_used} — errored: {type(exc).__name__}: {exc}"
                # finish_game is documented idempotent; guard against subclass overrides that aren't.
                with contextlib.suppress(Exception):
                    game.finish_game()

    def _finish(self, game: taaf.game.Game) -> None:
        """``finish_game()`` reporting the synthetic give-up token cost when
        the game ends without a win (state still ``playing`` → give-up /
        cancel), else 0 (a win has no no-move decision cost)."""
        run = game.game_run
        if run is None:
            return
        giveup = run.state == "playing"
        game.finish_game(
            generated_tokens=self.fake_giveup_generated_tokens if giveup else 0,
            uncached_input_tokens=self.fake_giveup_uncached_input_tokens if giveup else 0,
        )

    def _pick_action(self, game: taaf.game.Game, rng: random.Random) -> arcengine.ActionInput:
        state = game.current_state
        # arcengine no-ops every non-RESET action after GAME_OVER; without
        # this short-circuit the walker burns ``max_actions_per_game`` going
        # nowhere. ``GameState.game_over`` is true only on WIN in this
        # framework, so check the raw arcengine state instead.
        if state.raw.state == arcengine.GameState.GAME_OVER:
            return arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        valid = state.available_actions
        non_reset = [a for a in valid if a != 0]
        action_id = rng.choice(non_reset) if non_reset else 0
        action = arcengine.GameAction.from_id(action_id)
        data: dict[str, object] = {}
        if action.is_complex():
            # ACTION6 (CLICK) needs (x, y) within grid bounds.
            if self.click_on_color:
                data = self._pick_color_pixel(state, rng)
            else:
                h, w = state.frame.data.shape
                data = {"x": rng.randint(0, w - 1), "y": rng.randint(0, h - 1)}
        return arcengine.ActionInput(id=action, data=data)

    @staticmethod
    def _pick_color_pixel(state: taaf.game.GameState, rng: random.Random) -> dict[str, object]:
        """Uniform color present in the frame, then a uniform pixel of it."""
        frame = state.frame.data
        color = rng.choice(np.unique(frame).tolist())
        ys, xs = np.where(frame == color)
        idx = rng.randrange(len(xs))
        return {"x": int(xs[idx]), "y": int(ys[idx])}


@dataclass
class SolverSequence(Solver):
    """Plays a fixed list of actions on each game. Useful for deterministic
    tests. Once the sequence is exhausted (or the game finishes early) the
    solver calls ``finish_game()``.

    Fields:

    - ``actions``: the sequence to play. Actions not currently in
      ``available_actions`` are skipped rather than raising.
    - ``fake_generated_per_move`` / ``fake_uncached_per_move``: constant
      synthetic per-action token cost so diagnostics has non-zero data.
    - ``fake_giveup_generated_tokens`` / ``fake_giveup_uncached_input_tokens``:
      synthetic tokens reported via ``finish_game(...)`` when the sequence
      ends without a win (give-up / cancel) — the no-move-turn cost (R11.03).
      Default 0 reports nothing.
    """

    label: str = "SolverSequence"
    actions: list[arcengine.ActionInput] = field(default_factory=lambda: list[arcengine.ActionInput]())
    fake_generated_per_move: int = 20
    fake_uncached_per_move: int = 10
    fake_giveup_generated_tokens: int = 0
    fake_giveup_uncached_input_tokens: int = 0

    _ANALYSIS_RELPATH = "solver_analysis/sequence.html"

    def _setup(self) -> None:
        if self.job_dir is None:
            return
        out_path = self.job_dir / self._ANALYSIS_RELPATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self._build_analysis_html())

    def _build_analysis_html(self) -> str:
        rows = "".join(
            f"<tr><td>{i}</td>"
            f"<td><code>{escape(action.id.name)}</code></td>"
            f"<td>{'&nbsp;' if not action.data else f'<code>{escape(json.dumps(dict(action.data)))}</code>'}</td>"
            f"</tr>"
            for i, action in enumerate(self.actions)
        )
        body = (
            "<h1>SolverSequence — action list</h1>"
            f"<p>Plays the same fixed sequence on every game ({len(self.actions)} actions). "
            "Skips actions not in the current <code>available_actions</code>; stops when the game finishes.</p>"
            "<table><tr><th>#</th><th>action</th><th>data</th></tr>"
            f"{rows}</table>"
        )
        # Self-contained CSS — don't import diagnostics' chain
        # (matplotlib / scipy / imageio) from this lightweight module.
        css = (
            "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
            "padding:20px;max-width:900px;margin:0 auto;line-height:1.4;}"
            "h1{color:#fff;}"
            "table{border-collapse:collapse;margin:12px 0;}"
            "th,td{border:1px solid #3a3a3a;padding:6px 10px;text-align:left;}"
            "th{background:#2a2a2a;}"
            "code{background:#2a2a2a;padding:1px 4px;border-radius:3px;}"
        )
        return (
            '<!doctype html>\n<html><head><meta charset="utf-8">'
            "<title>SolverSequence — sequence</title>"
            f"<style>{css}</style></head>\n<body>\n{body}\n</body></html>\n"
        )

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        try:
            await asyncio.gather(*(self._play_one(game) for game in games))
        except asyncio.CancelledError:
            _finish_remaining(games)
            raise

    async def _play_one(self, game: taaf.game.Game) -> None:
        try:
            run = game.game_run
            if run is not None and self.job_dir is not None:
                run.solver_analysis_html = self._ANALYSIS_RELPATH
            for action in self.actions:
                await asyncio.sleep(0)
                run = game.game_run
                if run is None or run.state != "playing":
                    break
                if action.id.value not in game.current_state.available_actions:
                    continue  # robust to available_actions narrowing mid-game
                game.execute_action(
                    action,
                    generated_tokens=self.fake_generated_per_move,
                    uncached_input_tokens=self.fake_uncached_per_move,
                )
            self._finish(game)
        except asyncio.CancelledError:
            self._finish(game)
            raise

    def _finish(self, game: taaf.game.Game) -> None:
        """``finish_game()`` reporting the synthetic give-up token cost when
        the sequence ends without a win (state still ``playing`` → give-up /
        cancel), else 0 (a win has no no-move decision cost)."""
        run = game.game_run
        if run is None:
            return
        giveup = run.state == "playing"
        game.finish_game(
            generated_tokens=self.fake_giveup_generated_tokens if giveup else 0,
            uncached_input_tokens=self.fake_giveup_uncached_input_tokens if giveup else 0,
        )
