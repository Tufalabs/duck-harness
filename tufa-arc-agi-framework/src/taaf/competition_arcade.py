"""Local competition-mode ARC-AGI Arcade simulator (R11.13).

The real Kaggle submission path talks to a gateway-backed Arcade with
competition constraints: one scorecard, hidden baselines, and no repeated
game IDs within that scorecard. This module starts the same ``arc_agi``
REST app on localhost against the bundled offline games so those failures
can be reproduced before submission.
"""

from __future__ import annotations

import copy
import json
import logging
import socketserver
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import arc_agi
from arc_agi.models import EnvironmentInfo

import taaf.game_api

_LOGGER = logging.getLogger("taaf.competition_arcade")
_LOGGER.setLevel(logging.WARNING)

DEFAULT_API_KEY = "test-key-123"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CLONE_PREFIX = "k"
OFFICIAL_110_RUN_COUNT = 110


class _ThreadingWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


_OFFICIAL_GAME_IDS: tuple[str, ...] = (
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940",
    "wa30-ee6fef47", "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b",
    "sp80-589a99af", "m0r0-492f87ba", "vc33-5430563c", "ar25-0c556536",
    "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
    "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab",
    "tr87-cd924810",
)


def official_game_ids() -> tuple[str, ...]:
    """Return the 25 public official game IDs used for Kaggle smoke runs.

    Hardcoded so the build carries no external dataset dependency.
    """
    if len(_OFFICIAL_GAME_IDS) != 25:
        raise RuntimeError(f"Expected 25 official games, got {len(_OFFICIAL_GAME_IDS)}.")
    return _OFFICIAL_GAME_IDS


def clone_game_ids(
    source_game_ids: Sequence[str],
    *,
    total_runs: int,
    clone_prefix: str = DEFAULT_CLONE_PREFIX,
) -> list[str]:
    """Return unique 4-character game IDs for a competition-compatible pass.

    ``arc_agi`` competition scorecards can only create one run per game ID.
    Repeating a 25-game list to 110 entries therefore needs cloned IDs on
    the simulator side. The cloned IDs are stable, compact, and hyphen-free
    so the engine treats each one as a distinct base game.
    """
    if not source_game_ids:
        raise ValueError("source_game_ids must be non-empty.")
    if total_runs <= 0:
        raise ValueError("total_runs must be positive.")
    if not clone_prefix or len(clone_prefix) > 1:
        raise ValueError("clone_prefix must be a single character.")
    if total_runs > 1000:
        raise ValueError("clone_game_ids supports at most 1000 runs for 4-character clone IDs.")
    return [f"{clone_prefix}{i:03d}" for i in range(total_runs)]


def cloned_environment_infos(
    available_environments: Sequence[EnvironmentInfo],
    source_game_ids: Sequence[str],
    *,
    total_runs: int,
    clone_prefix: str = DEFAULT_CLONE_PREFIX,
) -> list[EnvironmentInfo]:
    """Clone selected ``EnvironmentInfo`` entries with unique game IDs."""
    clone_ids = clone_game_ids(source_game_ids, total_runs=total_runs, clone_prefix=clone_prefix)
    source_infos = [_find_environment_info(available_environments, game_id) for game_id in source_game_ids]
    out: list[EnvironmentInfo] = []
    for i, clone_id in enumerate(clone_ids):
        source = source_infos[i % len(source_infos)]
        private_tags = list(source.private_tags or [])
        private_tags.append(f"taaf_source_game:{source.game_id}")
        out.append(
            source.model_copy(
                deep=True,
                update={
                    "game_id": clone_id,
                    "private_tags": private_tags,
                },
            )
        )
    return out


def make_competition_arcade_spec(base_url: str) -> taaf.game_api.ArcadeSpec:
    """Build a ``GameAPI`` spec that talks to a competition-mode REST Arcade."""
    return taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=base_url,
        environments_dir="",
    )


@dataclass
class CompetitionArcadeServer:
    """Context-managed localhost competition Arcade server.

    By default this exposes the 25 official games once each. Use
    ``official_110()`` to expose the R2.57 110-run clone set.
    """

    game_ids: Sequence[str] = field(default_factory=official_game_ids)
    total_runs: int | None = None
    clone_prefix: str = DEFAULT_CLONE_PREFIX
    host: str = DEFAULT_HOST
    port: int = 0
    api_key: str = DEFAULT_API_KEY
    environments_dir: str = taaf.game_api._AUTO_ENV_DIR
    include_frame_data: bool = True
    save_all_recordings: bool = False

    _arcade: arc_agi.Arcade | None = field(default=None, init=False, repr=False)
    _app: Any | None = field(default=None, init=False, repr=False)
    _api: Any | None = field(default=None, init=False, repr=False)
    _server: Any | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _base_url: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def official_110(
        cls,
        *,
        host: str = DEFAULT_HOST,
        port: int = 0,
        clone_prefix: str = DEFAULT_CLONE_PREFIX,
    ) -> CompetitionArcadeServer:
        """Expose 110 cloned official games for the submission-shaped benchmark."""
        return cls(
            game_ids=official_game_ids(),
            total_runs=OFFICIAL_110_RUN_COUNT,
            clone_prefix=clone_prefix,
            host=host,
            port=port,
        )

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("CompetitionArcadeServer has not been started.")
        return self._base_url

    @property
    def arcade_spec(self) -> taaf.game_api.ArcadeSpec:
        return make_competition_arcade_spec(self.base_url)

    @property
    def exposed_game_ids(self) -> list[str]:
        if self._arcade is None:
            raise RuntimeError("CompetitionArcadeServer has not been started.")
        return [env.game_id for env in self._arcade.available_environments]

    def start(self) -> CompetitionArcadeServer:
        if self._server is not None:
            return self

        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=taaf.game_api._resolve_environments_dir(self.environments_dir),
            logger=_LOGGER,
        )
        arcade.available_environments = self._build_exposed_environments(arcade.available_environments)

        from arc_agi import server as arc_agi_server

        app, api = arc_agi_server.create_app(
            arcade,
            competition_mode=True,
            save_all_recordings=self.save_all_recordings,
            include_frame_data=self.include_frame_data,
        )

        def _anon_key() -> tuple[str, int, dict[str, str]]:
            return json.dumps({"api_key": self.api_key}), 200, {"Content-Type": "application/json"}

        app.add_url_rule("/api/games/anonkey", methods=["GET"], view_func=_anon_key, endpoint="anon_key")
        server = make_server(
            self.host,
            self.port,
            app,
            server_class=_ThreadingWSGIServer,
            handler_class=_QuietWSGIRequestHandler,
        )
        self._arcade = arcade
        self._app = app
        self._api = api
        self._server = server
        self._base_url = f"http://{self.host}:{server.server_port}"
        self._thread = threading.Thread(target=server.serve_forever, name="taaf-competition-arcade", daemon=True)
        self._thread.start()
        self._wait_until_ready()
        return self

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> CompetitionArcadeServer:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def _build_exposed_environments(self, available: Sequence[EnvironmentInfo]) -> list[EnvironmentInfo]:
        game_ids = tuple(self.game_ids)
        if self.total_runs is not None:
            return cloned_environment_infos(
                available,
                game_ids,
                total_runs=self.total_runs,
                clone_prefix=self.clone_prefix,
            )
        return [copy.deepcopy(_find_environment_info(available, game_id)) for game_id in game_ids]

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 10.0
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{self.base_url}/api/healthcheck", timeout=1.0) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError) as exc:
                last_error = repr(exc)
            time.sleep(0.05)
        raise RuntimeError(f"Competition Arcade did not become ready: {last_error}")


def _find_environment_info(available: Sequence[EnvironmentInfo], game_id: str) -> EnvironmentInfo:
    by_id = {env.game_id: env for env in available}
    if game_id in by_id:
        return by_id[game_id]
    base_id = game_id.split("-", 1)[0]
    for env in available:
        if env.game_id.split("-", 1)[0] == base_id:
            return env
    raise ValueError(f"Game {game_id!r} is not available in the local ARC environment files.")
