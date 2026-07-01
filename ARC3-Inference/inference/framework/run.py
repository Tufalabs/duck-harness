"""Run the harness through TAAF's Benchmark/GameAPI stack."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import hashlib
import inspect
import json
import logging
import math
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import taaf.benchmark
import taaf.deploy
import taaf.deploy_inline
import taaf.deploy_kaggle
import taaf.deploy_slurm
import taaf.game
import taaf.game_api

from inference.framework.kaggle import DUCK_HARNESS_PUBLIC_GAME_IDS
from inference.framework.solver import HarnessSolver
from inference.utils.run_artifacts import save_git_info, setup_experiment_directory

log = logging.getLogger(__name__)
RUN_CONFIG_FILENAME = "run_config.json"
_DEPLOYMENT_MAX_RUNTIME_UNSET = object()

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
load_dotenv(dotenv_path=_project_root / ".env.example")
load_dotenv(dotenv_path=_project_root / ".env", override=True)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def _parse_optional_list(raw_value: Any, *, option_name: str) -> list[str]:
    if isinstance(raw_value, list):
        values = raw_value
    else:
        text = str(raw_value or "").strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{option_name} must be a game id, comma-separated list, or JSON list."
                ) from exc
            values = parsed if isinstance(parsed, list) else [parsed]
        else:
            values = [item.strip() for item in text.split(",")]

    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _resolve_requested_game(
    requested: str,
    available: list[str],
    *,
    fallback_available: list[str] | None = None,
) -> str:
    lowered = requested.lower()
    prefix_map = {gid.split("-")[0].lower(): gid for gid in available}
    game_id = prefix_map.get(lowered, requested)
    if game_id in available:
        return game_id
    if fallback_available is not None:
        fallback_prefix_map = {
            gid.split("-")[0].lower(): gid for gid in fallback_available
        }
        fallback_game_id = fallback_prefix_map.get(lowered, requested)
        if fallback_game_id in fallback_available:
            return fallback_game_id
    raise ValueError(f"Unknown ARC-AGI3 game: {requested}")


def _apply_share_version_overrides(args: argparse.Namespace) -> None:
    """The share notebook overrides the benchmark game list at runtime (it plays
    the full live/offline competition set), so the locally selected games are
    only a packaging placeholder. Note that rather than stripping the selection
    — which would leave a harness-only invocation with no games at all."""
    if getattr(args, "kaggle_make_share_version", False):
        log.info(
            "Share mode: the deployed notebook plays the full competition game "
            "set at runtime; the locally selected games are only a packaging "
            "placeholder."
        )


def _resolve_game_ids(args: argparse.Namespace) -> list[str]:
    requested_games = _parse_optional_list(args.game, option_name="--game")
    dataset_specs = _parse_optional_list(args.dataset, option_name="--dataset")
    include_tags = _parse_optional_list(args.include_tags, option_name="--include-tags")
    exclude_tags = _parse_optional_list(args.exclude_tags, option_name="--exclude-tags")
    official_game_ids = list(DUCK_HARNESS_PUBLIC_GAME_IDS)
    if bool(getattr(args, "kaggle_duck_public_harness", False)):
        if requested_games or dataset_specs or include_tags or exclude_tags:
            raise ValueError(
                "--kaggle-duck-public-harness cannot be combined with --game, "
                "--dataset, --include-tags, or --exclude-tags."
            )
        return official_game_ids
    # Local dataset/generator game enumeration is not available in this build, so
    # the only built-in game set is the official one (``--include-tags official``).
    # The live competition supplies its own game list at runtime.
    if dataset_specs or exclude_tags or any(tag.lower() != "official" for tag in include_tags):
        raise ValueError(
            "Dataset/tag-based game selection beyond '--include-tags official' is not "
            "available in this build. Use --game with explicit ids, --include-tags "
            "official, or --kaggle-duck-public-harness."
        )
    selected_game_ids = official_game_ids if include_tags else []
    if requested_games:
        resolved: list[str] = []
        for requested in requested_games:
            game_id = _resolve_requested_game(requested, official_game_ids)
            if game_id not in resolved:
                resolved.append(game_id)
        return resolved
    if not selected_game_ids:
        raise ValueError(
            "At least one of --game, --include-tags official, or "
            "--kaggle-duck-public-harness is required."
        )
    return selected_game_ids


def _pass_schedule(*, pass_offset: int, n_passes: int) -> list[int]:
    return [pass_offset + offset for offset in range(n_passes)]


def _make_games(
    game_ids: list[str],
    *,
    environments_dir: str | None = None,
    arcade_spec: taaf.game_api.ArcadeSpec | None = None,
) -> list[taaf.game.Game]:
    if arcade_spec is not None:
        return [
            taaf.game_api.GameAPI(env_name=game_id, arcade_spec=arcade_spec)
            for game_id in game_ids
        ]
    arcade_spec = (
        None
        if not environments_dir
        else taaf.game_api.ArcadeSpec(environments_dir=str(environments_dir))
    )
    return [
        taaf.game_api.GameAPI(env_name=game_id)
        if arcade_spec is None
        else taaf.game_api.GameAPI(env_name=game_id, arcade_spec=arcade_spec)
        for game_id in game_ids
    ]


def _resolve_repo_paths(raw_value: Any, *, option_name: str) -> list[Path]:
    repo_paths: list[Path] = []
    for raw_path in _parse_optional_list(raw_value, option_name=option_name):
        path = Path(raw_path)
        if not path.is_absolute():
            path = _project_root / path
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(
                f"{option_name} entry does not exist or is not a directory: {raw_path}"
            )
        if not (path / "pyproject.toml").is_file():
            raise ValueError(f"{option_name} entry is not a Python project: {path}")
        if path not in repo_paths:
            repo_paths.append(path)
    return repo_paths


def _project_name(repo_path: Path) -> str:
    with (repo_path / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)
    name = pyproject.get("project", {}).get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{repo_path / 'pyproject.toml'} has no [project].name")
    return name.strip()


def _write_dependency_overrides(run_dir: Path, source_repos: list[Path]) -> Path:
    if not source_repos:
        raise ValueError("--deployment-source-repos is required for Slurm deployment.")

    seen_names: set[str] = set()
    lines: list[str] = []
    for repo_path in source_repos:
        name = _project_name(repo_path)
        if name in seen_names:
            raise ValueError(
                f"Duplicate source repo project name in --deployment-source-repos: {name}"
            )
        seen_names.add(name)
        snapshot_path = (run_dir / "src" / repo_path.name).resolve()
        lines.append(f"{name} @ {snapshot_path.as_uri()}")

    override_path = (run_dir / "deployment-overrides.txt").resolve()
    override_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return override_path


def _format_slurm_export_flag(
    override_path: Path, extra_env: dict[str, str] | None = None
) -> str:
    values = ["ALL", f"UV_OVERRIDE={override_path}"]
    for name, value in (extra_env or {}).items():
        clean_name = str(name).strip()
        if not clean_name:
            continue
        clean_value = str(value)
        if "," in clean_value:
            raise ValueError(
                f"Slurm export value for {clean_name} must not contain commas."
            )
        values.append(f"{clean_name}={clean_value}")
    return "--export=" + ",".join(values)


def _default_kaggle_kernel_slug(args: argparse.Namespace) -> str:
    raw_slug = str(getattr(args, "kaggle_kernel_slug", "") or "").strip()
    if raw_slug:
        return taaf.deploy_kaggle.slugify(raw_slug)
    base = str(getattr(args, "run_name", "") or "").strip() or f"taaf-{args.agent}"
    # Share deployments (R2.61) -share-suffix the *default* slug; an explicit
    # --kaggle-kernel-slug above is left untouched (matches taaf's own rule).
    if bool(getattr(args, "kaggle_make_share_version", False)):
        base = f"{base}-share"
    return taaf.deploy_kaggle.slugify(base)


def _default_kaggle_kernel_title(args: argparse.Namespace, *, kernel_slug: str) -> str:
    raw_title = str(getattr(args, "kaggle_kernel_title", "") or "").strip()
    if raw_title:
        return raw_title
    return kernel_slug


def _make_deployment_target(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    max_runtime_s: float | None | object = _DEPLOYMENT_MAX_RUNTIME_UNSET,
) -> taaf.deploy.DeploymentTarget:
    target_name = str(args.deployment_target).strip().lower()
    if max_runtime_s is _DEPLOYMENT_MAX_RUNTIME_UNSET:
        max_experiment_runtime_minutes = _max_experiment_runtime_minutes(args)
        resolved_max_runtime_s = (
            None
            if max_experiment_runtime_minutes is None
            else max_experiment_runtime_minutes * 60.0
        )
    else:
        resolved_max_runtime_s = max_runtime_s
    if target_name == "inline":
        if resolved_max_runtime_s is None:
            resolved_max_runtime_s = taaf.deploy_inline.InlineTarget.max_runtime_s
        return taaf.deploy_inline.InlineTarget(
            max_runtime_s=float(resolved_max_runtime_s)
        )
    if target_name == "kaggle":
        source_repos = _resolve_repo_paths(
            getattr(args, "deployment_source_repos", ""),
            option_name="--deployment-source-repos",
        )
        kernel_slug = _default_kaggle_kernel_slug(args)
        target_kwargs: dict[str, Any] = {
            "username": str(getattr(args, "kaggle_username", "") or "").strip() or None,
            "kernel_slug": kernel_slug,
            "kernel_title": _default_kaggle_kernel_title(args, kernel_slug=kernel_slug),
            "dataset_ref": str(getattr(args, "kaggle_dataset_ref", "") or "").strip()
            or None,
            "additional_dataset_sources": _parse_optional_list(
                getattr(args, "kaggle_additional_dataset_sources", ""),
                option_name="--kaggle-additional-dataset-sources",
            ),
            "extra_source_repos": [_project_root, *source_repos],
            "run_as_submission": bool(getattr(args, "kaggle_run_as_submission", False)),
            "public": bool(getattr(args, "kaggle_public", False)),
            "enable_internet": bool(getattr(args, "kaggle_enable_internet", False)),
            "cpu_only": bool(getattr(args, "kaggle_cpu_only", False)),
            "accelerator": str(getattr(args, "kaggle_accelerator", "") or "").strip()
            or None,
            "dataset_version_message": (
                str(getattr(args, "kaggle_dataset_version_message", "") or "").strip()
                or f"Update {kernel_slug} TAAF source bundle."
            ),
            "dry_run": bool(getattr(args, "kaggle_dry_run", False)),
            "make_share_version": bool(
                getattr(args, "kaggle_make_share_version", False)
            ),
        }
        if resolved_max_runtime_s is not None:
            target_kwargs["max_runtime_s"] = float(resolved_max_runtime_s)
        return taaf.deploy_kaggle.KaggleTarget(**target_kwargs)
    if target_name != "slurm":
        raise ValueError(f"Unsupported deployment target: {args.deployment_target}")
    if not str(args.slurm_time or "").strip():
        raise ValueError("--slurm-time is required for Slurm deployment.")

    source_repos = _resolve_repo_paths(
        args.deployment_source_repos, option_name="--deployment-source-repos"
    )
    override_path = _write_dependency_overrides(run_dir, source_repos)
    extra_sbatch_flags = _parse_optional_list(
        args.slurm_extra_sbatch_flags, option_name="--slurm-extra-sbatch-flags"
    )
    if any(
        flag == "--export" or flag.startswith("--export=")
        for flag in extra_sbatch_flags
    ):
        raise ValueError(
            "--slurm-extra-sbatch-flags must not set --export; the harness owns worker exports."
        )
    extra_sbatch_flags.append(
        _format_slurm_export_flag(
            override_path,
            getattr(args, "slurm_export_env", None),
        )
    )

    target_kwargs: dict[str, Any] = {
        "gpu": args.slurm_gpu,
        "gpu_count": int(args.slurm_gpu_count),
        "time": args.slurm_time,
        "image": args.slurm_image or None,
        "partition": args.slurm_partition or None,
        "nodelist": args.slurm_nodelist or None,
        "extra_sbatch_flags": extra_sbatch_flags,
        "extra_source_repos": [_project_root, *source_repos],
        "main_project": _project_root.name,
    }
    if (
        resolved_max_runtime_s is not None
        and "max_runtime_s"
        in inspect.signature(taaf.deploy_slurm.TufaSlurmTarget).parameters
    ):
        target_kwargs["max_runtime_s"] = float(resolved_max_runtime_s)
    target = taaf.deploy_slurm.TufaSlurmTarget(**target_kwargs)
    if resolved_max_runtime_s is not None and not hasattr(target, "max_runtime_s"):
        setattr(target, "max_runtime_s", float(resolved_max_runtime_s))
    return target


def _validate_local_server_config(args: argparse.Namespace) -> None:
    config_path = str(args.slurm_local_server_config or "").strip()
    if not config_path:
        raise ValueError(
            "--slurm-local-server-config is required when --slurm-start-local-server is set."
        )


def _local_server_api_key_file(args: argparse.Namespace, *, run_dir: Path) -> str:
    configured = str(args.slurm_local_server_api_key_file or "").strip()
    if configured:
        return configured
    return str(run_dir.resolve() / "server-api-key")


def _bundled_project_root(run_dir: Path) -> Path:
    return (run_dir / "src" / _project_root.name).resolve()


def _local_server_config_path(args: argparse.Namespace, *, run_dir: Path) -> str:
    config_path = str(args.slurm_local_server_config or "").strip()
    if not config_path:
        return ""

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        return config_path

    try:
        repo_relative_path = path.resolve().relative_to(_project_root)
    except ValueError:
        return config_path
    return str(_bundled_project_root(run_dir) / repo_relative_path)


def _make_solver(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    max_runtime_minutes_per_game: float,
) -> HarnessSolver:
    start_local_server = str(
        args.deployment_target
    ).strip().lower() == "slurm" and bool(args.slurm_start_local_server)
    if start_local_server:
        _validate_local_server_config(args)
    local_server_count = max(1, int(args.slurm_gpu_count)) if start_local_server else 1
    effective_concurrency = _effective_concurrent_jobs(args)
    return HarnessSolver(
        label=args.agent,
        model=args.model,
        analyzer_timeout=getattr(args, "analyzer_timeout", 120),
        max_actions_per_game=args.max_actions,
        max_runtime_s_per_game=max_runtime_minutes_per_game * 60.0,
        concurrency=effective_concurrency,
        save_request_logs=bool(args.analyzer_save_request_logs),
        start_local_server=start_local_server,
        local_server_config=(
            _local_server_config_path(args, run_dir=run_dir)
            if start_local_server
            else ""
        ),
        local_server_api_key_file=(
            _local_server_api_key_file(args, run_dir=run_dir)
            if start_local_server
            else ""
        ),
        local_server_repo_dir=str(
            _bundled_project_root(run_dir) if start_local_server else _project_root
        ),
        local_server_port=getattr(args, "slurm_local_server_port", None),
        local_server_tensor_parallel_size=getattr(
            args, "slurm_local_server_tensor_parallel_size", None
        ),
        local_server_count=local_server_count,
    )


def _current_username() -> str:
    candidates = [os.environ.get("USER"), os.environ.get("LOGNAME")]
    try:
        candidates.append(getpass.getuser())
    except OSError:
        pass
    for candidate in candidates:
        username = str(candidate or "").strip()
        if username:
            return username
    raise ValueError(
        "Could not determine the current username. Set USER or LOGNAME, or use "
        "an experiments.root_dir without a username placeholder."
    )


def _expand_experiments_dir(raw_path: str | Path) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError(
            "--experiments-dir is required when --experiment-dir is not set. "
            "Set experiments.root_dir in configs/inference.json or pass --experiments-dir."
        )
    username_placeholders = ("{username}", "{user}", "<username>", "$USER", "${USER}")
    if any(placeholder in text for placeholder in username_placeholders):
        username = _current_username()
        for placeholder in username_placeholders:
            text = text.replace(placeholder, username)
    return Path(os.path.expandvars(os.path.expanduser(text)))


def _experiments_dir(args: argparse.Namespace, *, required: bool = True) -> Path | None:
    raw_path = str(getattr(args, "experiments_dir", "") or "").strip()
    if not raw_path and not required:
        return None
    return _expand_experiments_dir(raw_path)


def _experiment_dir(args: argparse.Namespace) -> Path:
    if str(args.experiment_dir or "").strip():
        path = Path(args.experiment_dir)
        path.mkdir(parents=True, exist_ok=True)
        if not (path / "git_info.txt").exists():
            save_git_info(path)
        return path
    path, _log_file = setup_experiment_directory(
        base_output_dir=_experiments_dir(args),
        run_name=args.run_name or None,
    )
    return path


def _optional_positive_float(raw_value: Any, *, option_name: str) -> float | None:
    if raw_value in (None, ""):
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{option_name} must be positive.")
    return value


def _max_experiment_runtime_minutes(args: argparse.Namespace) -> float | None:
    minutes = _optional_positive_float(
        args.max_experiment_runtime_minutes,
        option_name="--max-experiment-runtime-minutes",
    )
    hours = _optional_positive_float(
        args.max_experiment_runtime_hours,
        option_name="--max-experiment-runtime-hours",
    )
    if minutes is not None and hours is not None:
        raise ValueError(
            "Use only one of --max-experiment-runtime-minutes or --max-experiment-runtime-hours."
        )
    if hours is not None:
        return hours * 60.0
    if minutes is not None:
        return minutes
    return None


def _game_run_count(*, game_count: int, n_passes: int) -> int:
    if game_count <= 0:
        raise ValueError("At least one game is required.")
    if n_passes <= 0:
        raise ValueError("--n-passes must be positive.")
    return game_count * n_passes


def _wave_count(*, game_count: int, n_passes: int, concurrent_jobs: int) -> int:
    if concurrent_jobs <= 0:
        raise ValueError("--concurrent-jobs must be positive.")
    return math.ceil(
        _game_run_count(game_count=game_count, n_passes=n_passes) / concurrent_jobs
    )


def _concurrency_multiplier(args: argparse.Namespace) -> int:
    if str(getattr(args, "deployment_target", "")).strip().lower() != "slurm":
        return 1
    if not bool(getattr(args, "slurm_start_local_server", False)):
        return 1
    return max(1, int(getattr(args, "slurm_gpu_count", 1) or 1))


def _effective_concurrent_jobs(args: argparse.Namespace) -> int:
    return int(args.concurrent_jobs) * _concurrency_multiplier(args)


def _max_runtime_minutes_per_game(
    args: argparse.Namespace,
    *,
    game_count: int,
    max_experiment_runtime_minutes: float | None,
) -> tuple[float, str, int]:
    explicit = _optional_positive_float(
        args.max_runtime_minutes, option_name="--max-runtime-minutes"
    )
    waves = _wave_count(
        game_count=game_count,
        n_passes=int(args.n_passes),
        concurrent_jobs=_effective_concurrent_jobs(args),
    )
    if explicit is not None:
        return explicit, "explicit", waves
    if max_experiment_runtime_minutes is None:
        raise ValueError(
            "Set --max-runtime-minutes, or set --max-experiment-runtime-minutes/"
            "--max-experiment-runtime-hours so the per-game limit can be derived."
        )
    return max_experiment_runtime_minutes / waves, "auto", waves


def _deployment_max_runtime_s(
    *,
    deployment_target: str,
    max_experiment_runtime_minutes: float | None,
    max_runtime_minutes_per_game: float,
    wave_count: int,
) -> float | None:
    if max_experiment_runtime_minutes is not None:
        return max_experiment_runtime_minutes * 60.0
    if str(deployment_target).strip().lower() != "inline":
        return None
    return (max_runtime_minutes_per_game * max(1, wave_count) + 10.0) * 60.0


def _copy_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def _competition_clone_runs(args: argparse.Namespace) -> int | None:
    raw_value = getattr(args, "competition_clone_runs", 0) or 0
    clone_runs = int(raw_value)
    if clone_runs < 0:
        raise ValueError("--competition-clone-runs must be non-negative.")
    return clone_runs or None


def _competition_arcade_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "simulate_competition_arcade", False))


def _competition_arcade_module() -> Any:
    try:
        import taaf.competition_arcade as competition_arcade
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--simulate-competition-arcade requires a TAAF build with taaf.competition_arcade."
        ) from exc
    return competition_arcade


def _enter_competition_arcade(
    args: argparse.Namespace,
    *,
    game_ids: list[str],
    stack: contextlib.ExitStack,
) -> tuple[list[str], taaf.game_api.ArcadeSpec | None]:
    if not _competition_arcade_enabled(args):
        return game_ids, None
    if str(args.deployment_target).strip().lower() != "inline":
        raise ValueError(
            "--simulate-competition-arcade is only supported with --deployment-target inline."
        )
    if int(args.n_passes) != 1:
        raise ValueError("--simulate-competition-arcade requires --n-passes 1.")
    competition_arcade = _competition_arcade_module()
    server_kwargs: dict[str, Any] = {
        "game_ids": tuple(game_ids),
        "total_runs": _competition_clone_runs(args),
    }
    environments_dir = getattr(args, "environments_dir", None)
    if environments_dir:
        server_kwargs["environments_dir"] = environments_dir
    server = stack.enter_context(
        competition_arcade.CompetitionArcadeServer(**server_kwargs)
    )
    return server.exposed_game_ids, server.arcade_spec


def _slurm_local_server_job_count(args: argparse.Namespace) -> int:
    if str(args.deployment_target).strip().lower() != "slurm":
        return 1
    if not bool(args.slurm_start_local_server):
        return 1
    return max(1, int(args.slurm_gpu_count))


def _split_pass_ranges(total_passes: int, group_count: int) -> list[tuple[int, int]]:
    if total_passes <= 0:
        raise ValueError("--n-passes must be positive.")
    if group_count <= 1:
        return [(0, total_passes)]
    base = total_passes // group_count
    remainder = total_passes % group_count
    ranges: list[tuple[int, int]] = []
    start = 0
    for group_index in range(group_count):
        count = base + (1 if group_index < remainder else 0)
        if count <= 0:
            continue
        ranges.append((start, count))
        start += count
    return ranges


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if not path.is_absolute():
        path = _project_root / path
    return path.resolve()


def _run_scoped_server_port(*, run_dir: Path, group_count: int) -> int:
    floor = 24000
    span = 20000
    width = max(1, int(group_count))
    usable = max(1, span - width)
    digest = hashlib.blake2b(
        str(run_dir.resolve()).encode("utf-8"), digest_size=4
    ).digest()
    return floor + (int.from_bytes(digest, "big") % usable)


def _server_port_from_config(
    config_path: str, *, run_dir: Path | None = None, group_count: int = 1
) -> int:
    env_port = os.environ.get("SERVER_PORT", "").strip()
    explicit_env_port = os.environ.get(
        "ARC3_SERVER_PORT_EXPLICIT", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if env_port and (run_dir is None or explicit_env_port):
        return int(env_port)
    if run_dir is not None:
        return _run_scoped_server_port(run_dir=run_dir, group_count=group_count)
    path = _resolve_config_path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    port = data.get("server", {}).get("port", 1234)
    return int(port)


def _local_server_env_for_port(port: int) -> dict[str, str]:
    base_url = f"http://127.0.0.1:{port}/v1"
    return {
        "SERVER_PORT": str(port),
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm",
    }


def _solver_args_for_local_server_pool(
    args: argparse.Namespace, *, run_dir: Path
) -> argparse.Namespace:
    if (
        str(args.deployment_target).strip().lower() != "slurm"
        or not bool(args.slurm_start_local_server)
        or int(args.slurm_gpu_count) <= 1
    ):
        return args
    if getattr(args, "slurm_local_server_port", None) is not None:
        base_port = int(args.slurm_local_server_port)
    else:
        base_port = _server_port_from_config(
            args.slurm_local_server_config,
            run_dir=run_dir,
            group_count=int(args.slurm_gpu_count),
        )
    return _copy_args(
        args,
        slurm_local_server_port=base_port,
        slurm_local_server_tensor_parallel_size=1,
    )


def _write_run_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    game_ids: list[str],
    max_experiment_runtime_minutes: float | None,
    max_runtime_minutes_per_game: float,
    max_runtime_minutes_per_game_source: str,
    wave_count: int,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    game_run_count = _game_run_count(
        game_count=len(game_ids), n_passes=int(args.n_passes)
    )
    pass_offset = int(getattr(args, "pass_offset", 0) or 0)
    slurm_starts_local_server = str(
        args.deployment_target
    ).strip().lower() == "slurm" and bool(args.slurm_start_local_server)
    concurrency_multiplier = _concurrency_multiplier(args)
    effective_concurrent_jobs = _effective_concurrent_jobs(args)
    payload = {
        "version": 2,
        "runner": "taaf",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "agent": args.agent,
        "model": args.model,
        "dataset": args.dataset,
        "include_tags": _parse_optional_list(
            args.include_tags, option_name="--include-tags"
        ),
        "exclude_tags": _parse_optional_list(
            args.exclude_tags, option_name="--exclude-tags"
        ),
        "environments_dir": args.environments_dir,
        "experiments_dir": (
            str(experiments_dir)
            if (experiments_dir := _experiments_dir(args, required=False)) is not None
            else None
        ),
        "games": list(game_ids),
        "game_count": len(game_ids),
        "game_run_count": game_run_count,
        "n_passes": int(args.n_passes),
        "pass_schedule": _pass_schedule(
            pass_offset=pass_offset,
            n_passes=int(args.n_passes),
        ),
        "concurrent_jobs": int(args.concurrent_jobs),
        "concurrent_jobs_scope": "per_gpu" if concurrency_multiplier > 1 else "total",
        "effective_concurrent_jobs": effective_concurrent_jobs,
        "analyzer_timeout_seconds": getattr(args, "analyzer_timeout", 120),
        "wave_count": wave_count,
        "max_actions": args.max_actions,
        "max_runtime_minutes_per_game": max_runtime_minutes_per_game,
        "max_runtime_minutes_per_game_source": max_runtime_minutes_per_game_source,
        "max_experiment_runtime_minutes": max_experiment_runtime_minutes,
        "max_experiment_runtime_hours": (
            None
            if max_experiment_runtime_minutes is None
            else max_experiment_runtime_minutes / 60.0
        ),
        "deployment": {
            "target": args.deployment_target,
            "wait": bool(args.deployment_wait),
            "source_repos": _parse_optional_list(
                args.deployment_source_repos,
                option_name="--deployment-source-repos",
            ),
            "slurm": {
                "gpu": args.slurm_gpu,
                "gpu_count": int(args.slurm_gpu_count),
                "time": args.slurm_time,
                "image": args.slurm_image or None,
                "partition": args.slurm_partition or None,
                "nodelist": args.slurm_nodelist or None,
                "extra_sbatch_flags": _parse_optional_list(
                    args.slurm_extra_sbatch_flags,
                    option_name="--slurm-extra-sbatch-flags",
                ),
                "start_local_server": slurm_starts_local_server,
                "local_server_config": (
                    _local_server_config_path(args, run_dir=run_dir)
                    if slurm_starts_local_server
                    else None
                ),
                "local_server_count": (
                    max(1, int(args.slurm_gpu_count))
                    if slurm_starts_local_server
                    else 0
                ),
                "local_server_base_port": getattr(
                    args, "slurm_local_server_port", None
                ),
                "local_server_tensor_parallel_size": getattr(
                    args,
                    "slurm_local_server_tensor_parallel_size",
                    None,
                ),
                "local_server_api_key_file": (
                    _local_server_api_key_file(args, run_dir=run_dir)
                    if slurm_starts_local_server
                    else None
                ),
                "local_server_repo_dir": (
                    str(_bundled_project_root(run_dir))
                    if slurm_starts_local_server
                    else None
                ),
            },
            "kaggle": {
                "kernel_slug": _default_kaggle_kernel_slug(args),
                "kernel_title": _default_kaggle_kernel_title(
                    args,
                    kernel_slug=_default_kaggle_kernel_slug(args),
                ),
                "dataset_ref": str(
                    getattr(args, "kaggle_dataset_ref", "") or ""
                ).strip()
                or None,
                "additional_dataset_sources": _parse_optional_list(
                    getattr(args, "kaggle_additional_dataset_sources", ""),
                    option_name="--kaggle-additional-dataset-sources",
                ),
                "public": bool(getattr(args, "kaggle_public", False)),
                "enable_internet": bool(getattr(args, "kaggle_enable_internet", False)),
                "cpu_only": bool(getattr(args, "kaggle_cpu_only", False)),
                "accelerator": str(
                    getattr(args, "kaggle_accelerator", "") or ""
                ).strip()
                or None,
                "run_as_submission": bool(
                    getattr(args, "kaggle_run_as_submission", False)
                ),
                "dry_run": bool(getattr(args, "kaggle_dry_run", False)),
                "duck_public_harness": bool(
                    getattr(args, "kaggle_duck_public_harness", False)
                ),
                "make_share_version": bool(
                    getattr(args, "kaggle_make_share_version", False)
                ),
            },
        },
        "competition_arcade": {
            "simulated": _competition_arcade_enabled(args),
            "clone_runs": _competition_clone_runs(args),
        },
    }
    if extra_payload:
        payload.update(extra_payload)
    (run_dir / RUN_CONFIG_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _run_split_slurm_local_servers(
    args: argparse.Namespace, *, game_ids: list[str]
) -> None:
    group_count = _slurm_local_server_job_count(args)
    pass_ranges = _split_pass_ranges(int(args.n_passes), group_count)
    max_experiment_runtime_minutes = _max_experiment_runtime_minutes(args)
    total_gpu_count = int(args.slurm_gpu_count)

    child_runtimes: list[tuple[int, int, float, str, int]] = []
    for pass_start, pass_count in pass_ranges:
        child_args = _copy_args(args, n_passes=pass_count, slurm_gpu_count=1)
        runtime, source, waves = _max_runtime_minutes_per_game(
            child_args,
            game_count=len(game_ids),
            max_experiment_runtime_minutes=max_experiment_runtime_minutes,
        )
        child_runtimes.append((pass_start, pass_count, runtime, source, waves))

    parent_wave_count = max(
        waves for _pass_start, _pass_count, _runtime, _source, waves in child_runtimes
    )
    parent_runtime = max(
        runtime for _pass_start, _pass_count, runtime, _source, _waves in child_runtimes
    )
    parent_runtime_source = (
        "explicit"
        if all(child[3] == "explicit" for child in child_runtimes)
        else "split-auto"
    )
    deployment_max_runtime_s = _deployment_max_runtime_s(
        deployment_target=args.deployment_target,
        max_experiment_runtime_minutes=max_experiment_runtime_minutes,
        max_runtime_minutes_per_game=parent_runtime,
        wave_count=parent_wave_count,
    )
    run_dir = _experiment_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_port = _server_port_from_config(
        args.slurm_local_server_config,
        run_dir=run_dir,
        group_count=len(pass_ranges),
    )
    _write_run_config(
        args,
        run_dir=run_dir,
        game_ids=game_ids,
        max_experiment_runtime_minutes=max_experiment_runtime_minutes,
        max_runtime_minutes_per_game=parent_runtime,
        max_runtime_minutes_per_game_source=parent_runtime_source,
        wave_count=parent_wave_count,
        extra_payload={
            "split_slurm_local_servers": {
                "job_count": len(pass_ranges),
                "gpu_count_per_job": 1,
                "total_gpu_count": total_gpu_count,
                "base_port": base_port,
                "pass_ranges": [
                    {
                        "pass_offset": pass_start,
                        "n_passes": pass_count,
                        "pass_schedule": _pass_schedule(
                            pass_offset=pass_start,
                            n_passes=pass_count,
                        ),
                    }
                    for pass_start, pass_count, _runtime, _source, _waves in child_runtimes
                ],
            },
            "hardware": {
                "gpu_type": str(args.slurm_gpu).lower(),
                "gpu_count": total_gpu_count,
            },
        },
    )

    print(f"Run directory: {run_dir.absolute()}")
    print(f"Games: {', '.join(game_ids)}")
    print(
        "Split Slurm local-server run: "
        f"{len(pass_ranges)} jobs; 1 GPU/job; {total_gpu_count} GPUs total"
    )
    print(f"Total passes: {args.n_passes}; concurrency/job: {args.concurrent_jobs}")
    print(
        "Max runtime per game: "
        f"{parent_runtime:.2f} minutes ({parent_runtime_source}; largest child wave count {parent_wave_count})"
    )

    handles: list[Any] = []
    for group_index, (
        pass_start,
        pass_count,
        runtime,
        runtime_source,
        waves,
    ) in enumerate(child_runtimes):
        child_dir = run_dir / "passes" / str(pass_start)
        child_dir.mkdir(parents=True, exist_ok=True)
        port = base_port + group_index
        port_env = _local_server_env_for_port(port)
        child_label = f"{args.run_name or args.agent}-gpu{group_index}"
        child_args = _copy_args(
            args,
            n_passes=pass_count,
            run_name=child_label,
            pass_offset=pass_start,
            slurm_gpu_count=1,
            slurm_local_server_port=port,
            slurm_local_server_tensor_parallel_size=1,
            slurm_local_server_api_key_file=str(child_dir / "server-api-key"),
            slurm_export_env=port_env,
        )
        _write_run_config(
            child_args,
            run_dir=child_dir,
            game_ids=game_ids,
            max_experiment_runtime_minutes=max_experiment_runtime_minutes,
            max_runtime_minutes_per_game=runtime,
            max_runtime_minutes_per_game_source=runtime_source,
            wave_count=waves,
            extra_payload={
                "pass_offset": pass_start,
                "pass_count": pass_count,
                "split_group": group_index,
                "split_group_count": len(pass_ranges),
                "parent_run_dir": str(run_dir),
                "hardware": {
                    "gpu_type": str(args.slurm_gpu).lower(),
                    "gpu_count": total_gpu_count,
                },
                "slurm_job_gpu_count": 1,
            },
        )

        solver = _make_solver(
            child_args,
            run_dir=child_dir,
            max_runtime_minutes_per_game=runtime,
        )
        benchmark = taaf.benchmark.Benchmark(
            label=child_label,
            games=_make_games(
                game_ids,
                environments_dir=args.environments_dir,
            ),
            solver=solver,
            n_passes=pass_count,
            job_dir=child_dir,
        )
        print(
            f"Group {group_index}: passes {pass_start}-{pass_start + pass_count - 1}; "
            f"port {port}; waves {waves}; max {runtime:.2f} min/game"
        )
        handle = asyncio.run(
            benchmark.deploy(
                _make_deployment_target(
                    child_args,
                    run_dir=child_dir,
                    max_runtime_s=deployment_max_runtime_s,
                )
            )
        )
        handles.append(handle)
        job_id = getattr(handle, "job_id", None)
        if job_id:
            print(f"  Slurm job id: {job_id}")
            print(f"  Slurm stdout: {child_dir / 'stdout.log'}")
            print(f"  Slurm stderr: {child_dir / 'stderr.log'}")

    if args.deployment_wait:
        for handle in handles:
            handle.wait()
    else:
        print("Deployment submitted; not waiting for completion.")


def _run(args: argparse.Namespace) -> None:
    game_ids = _resolve_game_ids(args)
    with contextlib.ExitStack() as stack:
        game_ids, arcade_spec = _enter_competition_arcade(
            args, game_ids=game_ids, stack=stack
        )
        if args.list_games:
            for game_id in game_ids:
                print(game_id)
            return

        max_experiment_runtime_minutes = _max_experiment_runtime_minutes(args)
        run_dir = _experiment_dir(args)
        solver_args = _solver_args_for_local_server_pool(args, run_dir=run_dir)
        max_runtime_minutes_per_game, max_runtime_minutes_source, wave_count = (
            _max_runtime_minutes_per_game(
                solver_args,
                game_count=len(game_ids),
                max_experiment_runtime_minutes=max_experiment_runtime_minutes,
            )
        )
        deployment_max_runtime_s = _deployment_max_runtime_s(
            deployment_target=args.deployment_target,
            max_experiment_runtime_minutes=max_experiment_runtime_minutes,
            max_runtime_minutes_per_game=max_runtime_minutes_per_game,
            wave_count=wave_count,
        )
        _write_run_config(
            solver_args,
            run_dir=run_dir,
            game_ids=game_ids,
            max_experiment_runtime_minutes=max_experiment_runtime_minutes,
            max_runtime_minutes_per_game=max_runtime_minutes_per_game,
            max_runtime_minutes_per_game_source=max_runtime_minutes_source,
            wave_count=wave_count,
        )

        solver = _make_solver(
            solver_args,
            run_dir=run_dir,
            max_runtime_minutes_per_game=max_runtime_minutes_per_game,
        )
        benchmark = taaf.benchmark.Benchmark(
            label=args.run_name or args.agent,
            games=_make_games(
                game_ids,
                environments_dir=args.environments_dir,
                arcade_spec=arcade_spec,
            ),
            solver=solver,
            n_passes=int(args.n_passes),
            job_dir=run_dir,
        )
        print(f"Run directory: {run_dir.absolute()}")
        print(f"Games: {', '.join(game_ids)}")
        if solver.concurrency != int(solver_args.concurrent_jobs):
            concurrency_text = (
                f"{int(solver_args.concurrent_jobs)} per GPU/server "
                f"({solver.concurrency} total)"
            )
        else:
            concurrency_text = str(solver.concurrency)
        print(
            f"Passes: {benchmark.n_passes}; concurrency: {concurrency_text}; waves: {wave_count}"
        )
        print(
            "Max runtime per game: "
            f"{max_runtime_minutes_per_game:.2f} minutes ({max_runtime_minutes_source})"
        )
        print(f"Deployment target: {args.deployment_target}")
        if _competition_arcade_enabled(solver_args):
            print("Competition Arcade: simulated localhost")
        if (
            str(solver_args.deployment_target).strip().lower() == "slurm"
            and bool(solver_args.slurm_start_local_server)
            and int(solver_args.slurm_gpu_count) > 1
        ):
            print(
                "Solver local servers: "
                f"{solver.local_server_count} one-GPU servers starting at port {solver.local_server_port}"
            )
        handle = asyncio.run(
            benchmark.deploy(
                _make_deployment_target(
                    solver_args,
                    run_dir=run_dir,
                    max_runtime_s=deployment_max_runtime_s,
                )
            )
        )
        job_id = getattr(handle, "job_id", None)
        kernel_id = getattr(handle, "kernel_id", "")
        uploaded = bool(getattr(handle, "uploaded", True))
        if kernel_id:
            print(f"Kaggle kernel: {kernel_id}")
            print(f"Kaggle notebook: https://www.kaggle.com/code/{kernel_id}")
            dataset_ref = getattr(handle, "dataset_ref", "")
            if dataset_ref:
                print(f"Kaggle source dataset: {dataset_ref}")
            if not uploaded:
                print(f"Kaggle dry-run bundles: {run_dir / 'kaggle'}")
        elif job_id:
            print(f"Slurm job id: {job_id}")
            print(f"Slurm stdout: {run_dir / 'stdout.log'}")
            print(f"Slurm stderr: {run_dir / 'stderr.log'}")
        if args.deployment_wait and (not kernel_id or uploaded):
            handle.wait()
            if kernel_id:
                print(f"Kaggle output: {run_dir / 'kaggle-output'}")
        elif args.deployment_wait and kernel_id and not uploaded:
            print("Kaggle dry-run requested; nothing remote to wait for.")
        elif kernel_id and uploaded:
            print("Kaggle deployment submitted; not waiting for completion.")
        elif job_id:
            print("Deployment submitted; not waiting for completion.")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Run the inference harness through TAAF Benchmark/GameAPI."
    )
    parser.add_argument("--agent", "-a", default="inference")
    parser.add_argument(
        "--game",
        "-g",
        default="",
        help="Game id/env name, comma-separated list, or JSON list.",
    )
    parser.add_argument("--dataset", "--datasets", dest="dataset", default="")
    parser.add_argument("--include-tags", dest="include_tags", default="")
    parser.add_argument("--exclude-tags", dest="exclude_tags", default="")
    parser.add_argument(
        "--environments-dir", dest="environments_dir", default=None
    )
    parser.add_argument("--datasets-dir", dest="datasets_dir", default=None)
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="Print the resolved games without running them.",
    )
    parser.add_argument("--run-name", dest="run_name", default="")
    parser.add_argument("--experiments-dir", dest="experiments_dir", default="")
    parser.add_argument("--experiment-dir", dest="experiment_dir", default="")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument("--max-experiment-runtime-minutes", type=float, default=None)
    parser.add_argument("--max-experiment-runtime-hours", type=float, default=None)
    parser.add_argument("--n-passes", dest="n_passes", type=int, default=1)
    parser.add_argument("--concurrent-jobs", type=int, default=16)
    parser.add_argument(
        "--simulate-competition-arcade",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run against TAAF's localhost competition Arcade simulator. Inline target only.",
    )
    parser.add_argument(
        "--competition-clone-runs",
        type=int,
        default=0,
        help="When simulating the competition Arcade, clone the selected games to this many unique run IDs.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--timeout",
        "--analyzer-timeout",
        dest="analyzer_timeout",
        type=float,
        default=120,
    )
    parser.add_argument(
        "--deployment-target", choices=["inline", "slurm", "kaggle"], default="inline"
    )
    parser.add_argument(
        "--deployment-wait",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for non-inline deployments to finish before exiting.",
    )
    parser.add_argument(
        "--deployment-source-repos",
        default="",
        help="Repo path, comma-separated list, or JSON list bundled with Slurm/Kaggle deployments.",
    )
    parser.add_argument(
        "--kaggle-duck-public-harness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            f"Run the {len(DUCK_HARNESS_PUBLIC_GAME_IDS)} public games used by the "
            "duck Kaggle validation harness."
        ),
    )
    parser.add_argument(
        "--kaggle-make-share-version",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deploy the public 'share' Kaggle variant (R2.61): minimal notebook, "
        "minimal source bundle, and -share-suffixes default slugs.",
    )
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--kaggle-kernel-slug", default="")
    parser.add_argument("--kaggle-kernel-title", default="")
    parser.add_argument("--kaggle-dataset-ref", default="")
    parser.add_argument(
        "--kaggle-additional-dataset-sources",
        default="",
        help="Extra Kaggle datasets as owner/slug refs, comma-separated list, or JSON list.",
    )
    parser.add_argument(
        "--kaggle-public",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make the pushed Kaggle notebook public.",
    )
    parser.add_argument(
        "--kaggle-enable-internet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable internet for the pushed Kaggle notebook.",
    )
    parser.add_argument(
        "--kaggle-cpu-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable Kaggle GPU for a cheap smoke run.",
    )
    parser.add_argument("--kaggle-accelerator", default="NvidiaRtxPro6000")
    parser.add_argument(
        "--kaggle-run-as-submission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emulate Kaggle submission mode in the notebook.",
    )
    parser.add_argument(
        "--kaggle-dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only write the Kaggle source/kernel bundles locally; do not upload.",
    )
    parser.add_argument("--kaggle-dataset-version-message", default="")
    parser.add_argument("--slurm-gpu", choices=["B200", "B300", "RTX"], default="B200")
    parser.add_argument("--slurm-gpu-count", type=int, default=1)
    parser.add_argument("--slurm-time", default="06:00:00")
    parser.add_argument("--slurm-image", default="")
    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-nodelist", default="")
    parser.add_argument(
        "--slurm-start-local-server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start the local vLLM server from solver setup inside the Slurm allocation.",
    )
    parser.add_argument("--slurm-local-server-config", default="")
    parser.add_argument("--slurm-local-server-api-key-file", default="")
    parser.add_argument(
        "--slurm-extra-sbatch-flags",
        default="",
        help="Extra sbatch flags as a comma-separated list or JSON list. Do not set --export here.",
    )
    parser.add_argument(
        "--save-request-logs",
        dest="analyzer_save_request_logs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.max_actions is not None and args.max_actions <= 0:
        args.max_actions = None
    if args.max_runtime_minutes is not None and args.max_runtime_minutes <= 0:
        args.max_runtime_minutes = None
    if args.n_passes <= 0:
        log.error("--n-passes must be positive.")
        sys.exit(1)
    if args.concurrent_jobs <= 0:
        log.error("--concurrent-jobs must be positive.")
        sys.exit(1)
    if args.competition_clone_runs < 0:
        log.error("--competition-clone-runs must be non-negative.")
        sys.exit(1)
    if args.slurm_gpu_count <= 0:
        log.error("--slurm-gpu-count must be positive.")
        sys.exit(1)
    if not args.list_games and not args.model:
        log.error("--model is required unless --list-games is set.")
        sys.exit(1)
    _apply_share_version_overrides(args)
    try:
        _run(args)
    except Exception as exc:
        log.error("%s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
