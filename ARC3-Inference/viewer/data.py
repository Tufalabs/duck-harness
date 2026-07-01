"""Load viewer payloads from structured run data."""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.utils.run_artifacts import is_selectable_run_dir_name, run_dir_sort_key
from inference.utils.viewer_artifacts import load_raw_events


_SECTION_RE = re.compile(r"(?m)^\[(.+?)\]\s*$")
_KNOWN_SECTION_LABELS = {
    "ASSISTANT",
    "ACTION_RESPONSE",
    "ANALYZER STATUS",
    "MODEL CONTEXT",
    "MODEL RESPONSE META",
    "OUTPUT",
    "PROMPT LOG SNAPSHOT",
    "SYSTEM PROMPT",
    "THINKING",
    "USER PROMPT",
}
_KNOWN_SECTION_PREFIXES = ("ERROR", "TOOL CALL:", "TOOL RESULT:")
_HIDDEN_SECTION_LABELS = {
    "ACTION_RESPONSE",
    "MODEL CONTEXT",
    "MODEL RESPONSE META",
    "PROMPT LOG SNAPSHOT",
}
_ARC_COLOR_MAP = {
    0: "#FFFFFFFF",
    1: "#CCCCCCFF",
    2: "#999999FF",
    3: "#666666FF",
    4: "#333333FF",
    5: "#000000FF",
    6: "#E53AA3FF",
    7: "#FF7BCCFF",
    8: "#F93C31FF",
    9: "#1E93FFFF",
    10: "#88D8F1FF",
    11: "#FFDC00FF",
    12: "#FF851BFF",
    13: "#921231FF",
    14: "#4FCC30FF",
    15: "#A356D6FF",
}
_RUN_PAYLOAD_CACHE: dict[tuple[str, str, tuple[Any, ...]], dict[str, Any]] = {}
_PASS_INDEX_RE = re.compile(r"_p(?P<pass_index>\d+)_viewer_data\.json$")
_TOOL_CODE_PARAMETER_RE = re.compile(
    r"<parameter=code>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_RUNTIME_VALUE_SYMBOLS = {
    "current_frame",
    "current_frame.ascii",
    "current_frame.step",
    "current_frame.level",
    "current_frame.shape",
    "current_frame.segmentation",
    "history",
    "history.action",
    "history.frame",
    "history.frame.ascii",
    "history.frame.step",
    "history.frame.level",
    "history.frame.shape",
    "history.frame.segmentation",
    "valid_actions",
    "last_action_result",
}
_RUNTIME_CALL_SYMBOLS = {
    "action",
    "board_lines",
}
_RUNTIME_SYMBOLS = _RUNTIME_VALUE_SYMBOLS | _RUNTIME_CALL_SYMBOLS


@dataclass(frozen=True)
class _NormalizedGameArtifact:
    summary: dict[str, Any]
    compact: dict[str, Any]
    full: dict[str, Any]


def _direct_run_dir(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.is_dir() and is_selectable_run_dir_name(candidate.name):
        return candidate
    return None


def find_latest_run_dir(base_dir: str | Path = "runs") -> Path | None:
    """Return the newest timestamped run directory, if any."""
    runs_dir = Path(base_dir)
    if not runs_dir.exists():
        return None
    direct_run_dir = _direct_run_dir(runs_dir)
    if direct_run_dir is not None:
        return direct_run_dir
    candidates = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir() and is_selectable_run_dir_name(path.name)],
        key=run_dir_sort_key,
    )
    return candidates[-1] if candidates else None


def list_run_dirs(base_dir: str | Path = "runs") -> list[Path]:
    """Return timestamped run directories, newest first."""
    runs_dir = Path(base_dir)
    if not runs_dir.exists():
        return []
    direct_run_dir = _direct_run_dir(runs_dir)
    if direct_run_dir is not None:
        return [direct_run_dir]
    return sorted(
        [path for path in runs_dir.iterdir() if path.is_dir() and is_selectable_run_dir_name(path.name)],
        key=run_dir_sort_key,
        reverse=True,
    )


def _resolve_run_dir(*, runs_dir: str | Path, run_dir: str | Path | None) -> Path:
    resolved_run_dir = Path(run_dir) if run_dir is not None else find_latest_run_dir(runs_dir)
    if resolved_run_dir is None or not resolved_run_dir.exists():
        raise FileNotFoundError(f"No run directory found in {Path(runs_dir)}")
    return resolved_run_dir


def load_run_summary(*, runs_dir: str | Path = "runs", run_dir: str | Path | None = None) -> dict[str, Any]:
    """Load lightweight run metadata and per-game summaries for the viewer shell."""
    resolved_run_dir = _resolve_run_dir(runs_dir=runs_dir, run_dir=run_dir)
    available_runs = [path.name for path in list_run_dirs(runs_dir)]
    cache_key = (
        str(resolved_run_dir.resolve()),
        "summary",
        _run_dir_fingerprint(resolved_run_dir),
    )
    cached = _RUN_PAYLOAD_CACHE.get(cache_key)
    if cached is None:
        cached = {
            "source": "viewer_data",
            "arc_palette": _arc_palette(),
            "games": _load_game_summaries(resolved_run_dir),
        }
        for existing_key in [key for key in _RUN_PAYLOAD_CACHE if key[:2] == cache_key[:2] and key != cache_key]:
            _RUN_PAYLOAD_CACHE.pop(existing_key, None)
        _RUN_PAYLOAD_CACHE[cache_key] = cached

    return {
        "run_name": resolved_run_dir.name,
        "selected_run": resolved_run_dir.name,
        "available_runs": available_runs,
        "source": cached["source"],
        "arc_palette": cached["arc_palette"],
        "games": cached["games"],
    }


def load_run_payload(
    *,
    runs_dir: str | Path = "runs",
    run_dir: str | Path | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """Load a viewer payload for a specific run, or the latest run by default."""
    resolved_run_dir = _resolve_run_dir(runs_dir=runs_dir, run_dir=run_dir)
    available_runs = [path.name for path in list_run_dirs(runs_dir)]
    cache_key = (
        str(resolved_run_dir.resolve()),
        "compact" if compact else "full",
        _run_dir_fingerprint(resolved_run_dir),
    )
    cached = _RUN_PAYLOAD_CACHE.get(cache_key)
    if cached is None:
        artifacts = _load_game_artifacts(resolved_run_dir)
        cached = {
            "source": "viewer_data",
            "arc_palette": _arc_palette(),
            "games": [artifact.compact if compact else artifact.full for artifact in artifacts],
        }
        for existing_key in [key for key in _RUN_PAYLOAD_CACHE if key[:2] == cache_key[:2] and key != cache_key]:
            _RUN_PAYLOAD_CACHE.pop(existing_key, None)
        _RUN_PAYLOAD_CACHE[cache_key] = cached

    return {
        "run_name": resolved_run_dir.name,
        "selected_run": resolved_run_dir.name,
        "available_runs": available_runs,
        "source": cached["source"],
        "arc_palette": cached["arc_palette"],
        "games": cached["games"],
    }


def load_game_payload(
    *,
    runs_dir: str | Path = "runs",
    run_dir: str | Path | None = None,
    game_index: int,
) -> dict[str, Any]:
    """Load compact viewer data for one game inside a run."""
    resolved_run_dir = _resolve_run_dir(runs_dir=runs_dir, run_dir=run_dir)
    path = _viewer_data_path_for_index(resolved_run_dir, game_index)

    cache_key = (
        str(resolved_run_dir.resolve()),
        str(game_index),
        "compact-game",
        _run_dir_fingerprint(resolved_run_dir),
    )
    cached = _RUN_PAYLOAD_CACHE.get(cache_key)
    if cached is None:
        cached = _load_game_artifact(resolved_run_dir, path).compact
        for existing_key in [key for key in _RUN_PAYLOAD_CACHE if key[:2] == cache_key[:2] and key != cache_key]:
            _RUN_PAYLOAD_CACHE.pop(existing_key, None)
        _RUN_PAYLOAD_CACHE[cache_key] = cached
    return cached


def load_game_shell_payload(
    *,
    runs_dir: str | Path = "runs",
    run_dir: str | Path | None = None,
    game_index: int,
) -> dict[str, Any]:
    """Load one game with lightweight step summaries for the interactive viewer."""
    resolved_run_dir = _resolve_run_dir(runs_dir=runs_dir, run_dir=run_dir)
    path = _viewer_data_path_for_index(resolved_run_dir, game_index)
    cache_key = (
        str(resolved_run_dir.resolve()),
        str(game_index),
        "game-shell",
        _run_dir_fingerprint(resolved_run_dir),
    )
    cached = _RUN_PAYLOAD_CACHE.get(cache_key)
    if cached is None:
        cached = _load_game_shell(resolved_run_dir, path)
        for existing_key in [key for key in _RUN_PAYLOAD_CACHE if key[:2] == cache_key[:2] and key != cache_key]:
            _RUN_PAYLOAD_CACHE.pop(existing_key, None)
        _RUN_PAYLOAD_CACHE[cache_key] = cached
    return cached


def load_game_step_payload(
    *,
    runs_dir: str | Path = "runs",
    run_dir: str | Path | None = None,
    game_index: int,
    step_index: int,
) -> dict[str, Any]:
    """Load one hydrated viewer step for a selected game."""
    resolved_run_dir = _resolve_run_dir(runs_dir=runs_dir, run_dir=run_dir)
    path = _viewer_data_path_for_index(resolved_run_dir, game_index)
    if step_index < 0:
        raise FileNotFoundError(f"Step index {step_index} not found in game {game_index}")
    cache_key = (
        str(resolved_run_dir.resolve()),
        str(game_index),
        str(step_index),
        "game-step",
        _run_dir_fingerprint(resolved_run_dir),
    )
    cached = _RUN_PAYLOAD_CACHE.get(cache_key)
    if cached is None:
        cached = _load_game_step(resolved_run_dir, path, step_index)
        for existing_key in [key for key in _RUN_PAYLOAD_CACHE if key[:3] == cache_key[:3] and key != cache_key]:
            _RUN_PAYLOAD_CACHE.pop(existing_key, None)
        _RUN_PAYLOAD_CACHE[cache_key] = cached
    return cached


def _seed_sort_key(path: Path) -> tuple[int, int | str, str]:
    try:
        return (0, int(path.name), path.name)
    except ValueError:
        return (1, path.name, path.name)


def _seed_artifact_dirs(run_dir: Path) -> list[Path]:
    split_dirs = [path for name in ("passes", "seeds") for path in [(run_dir / name)] if path.exists()]
    return sorted(
        [
            path / "artifacts"
            for split_dir in split_dirs
            for path in split_dir.iterdir()
            if path.is_dir() and (path / "artifacts").exists()
        ],
        key=lambda path: _seed_sort_key(path.parent),
    )


def _artifact_run_dir(viewer_data_path: Path) -> Path:
    if viewer_data_path.parent.name == "artifacts":
        return viewer_data_path.parent.parent
    return viewer_data_path.parent


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _viewer_pass_index(viewer_data_path: Path) -> int | None:
    match = _PASS_INDEX_RE.search(viewer_data_path.name)
    if match is None:
        return None
    return int(match.group("pass_index"))


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pass_from_run_config(config: dict[str, Any], artifact_run_dir: Path, pass_index: int | None) -> int | None:
    raw_offset = config.get("pass_offset")
    pass_offset = _coerce_optional_int(raw_offset)
    if pass_offset is None and artifact_run_dir.parent.name == "passes":
        pass_offset = _coerce_optional_int(artifact_run_dir.name)
    if pass_index is not None:
        return (pass_offset or 0) + pass_index
    raw_schedule = config.get("pass_schedule")
    if isinstance(raw_schedule, list) and raw_schedule:
        return _coerce_optional_int(raw_schedule[0])
    return pass_offset


def _artifact_metadata(root_run_dir: Path, viewer_data_path: Path) -> dict[str, Any]:
    artifact_run_dir = _artifact_run_dir(viewer_data_path)
    is_split_child = artifact_run_dir.parent.name in {"passes", "seeds"} and artifact_run_dir.parent.parent == root_run_dir
    is_root_artifact = artifact_run_dir == root_run_dir
    if not (is_split_child or is_root_artifact):
        return {}

    config = _load_optional_json(artifact_run_dir / "run_config.json")
    pass_index = _viewer_pass_index(viewer_data_path)
    pass_label = _pass_from_run_config(config, artifact_run_dir, pass_index)
    raw_seed = config.get("seed")
    if pass_label is None and is_split_child and artifact_run_dir.parent.name == "seeds":
        raw_seed = raw_seed if raw_seed is not None else artifact_run_dir.name
    if pass_label is None and raw_seed is None:
        return {}
    metadata: dict[str, Any] = {"artifact_run_dir": str(artifact_run_dir)}
    if pass_label is not None:
        metadata["pass_label"] = str(pass_label)
        metadata["pass_index"] = pass_label
    if is_split_child:
        metadata["split_dir"] = artifact_run_dir.name
        metadata["split_dir_kind"] = artifact_run_dir.parent.name
    if raw_seed is not None and pass_label is None:
        metadata["seed_label"] = str(raw_seed)
        metadata["seed_dir"] = artifact_run_dir.name
        try:
            metadata["seed"] = int(raw_seed)
        except (TypeError, ValueError):
            metadata["seed"] = raw_seed
    return metadata


def _run_dir_fingerprint(run_dir: Path) -> tuple[Any, ...]:
    relevant_paths: list[Path] = []
    relevant_paths.extend(_viewer_data_paths(run_dir))
    relevant_paths.extend(sorted(run_dir.glob("*requests.jsonl")))
    relevant_paths.extend(sorted(run_dir.glob("seeds/*/*requests.jsonl")))
    relevant_paths.extend(sorted(run_dir.glob("seeds/*/run_config.json")))
    total_size = 0
    max_mtime_ns = 0
    for path in relevant_paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        total_size += stat.st_size
        max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
    return (
        len(relevant_paths),
        total_size,
        max_mtime_ns,
        bool(_viewer_data_paths(run_dir)),
    )


def _arc_palette() -> list[str]:
    max_index = max(_ARC_COLOR_MAP) if _ARC_COLOR_MAP else 15
    palette: list[str] = []
    for index in range(max_index + 1):
        r, g, b = _hex_to_rgb(_ARC_COLOR_MAP.get(index, "#000000FF"))
        palette.append(f"rgb({r}, {g}, {b})")
    return palette


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    if len(value) < 6:
        raise ValueError(f"Expected at least 6 hex chars, got {value!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _viewer_data_paths(run_dir: Path) -> list[Path]:
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        return sorted(artifacts_dir.glob("*viewer_data.json"))

    paths: list[Path] = []
    for seed_artifacts_dir in _seed_artifact_dirs(run_dir):
        paths.extend(sorted(seed_artifacts_dir.glob("*viewer_data.json")))
    return paths


def _viewer_data_path_for_index(run_dir: Path, game_index: int) -> Path:
    paths = _viewer_data_paths(run_dir)
    if game_index < 0 or game_index >= len(paths):
        raise FileNotFoundError(f"Game index {game_index} not found in {run_dir}")
    return paths[game_index]


def compact_saved_game_payload(
    payload: dict[str, Any],
    *,
    viewer_data_path: Path | None = None,
    request_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a compact viewer payload suitable for long-term on-disk storage."""
    if _is_compact_saved_game_payload(payload):
        compact = dict(payload)
        compact["game_id"] = str(compact.get("game_id") or "").strip()
        compact["status"] = str(compact.get("status") or "").strip()
        raw_events = [_normalize_event(event) for event in load_raw_events(payload, viewer_data_path=viewer_data_path)]
        if raw_events:
            for index, event in enumerate(raw_events):
                event["event_index"] = index
            compact["viewer_steps"] = [
                _compact_viewer_step(step)
                for step in _build_viewer_steps(raw_events, request_snapshots=request_snapshots)
            ]
            compact["eventCount"] = int(compact.get("eventCount") or len(raw_events))
        else:
            compact["viewer_steps"] = list(compact.get("viewer_steps") or [])
            compact["eventCount"] = int(compact.get("eventCount") or len(compact["viewer_steps"]))
        replay_url = str(compact.get("replay_url") or "").strip()
        if replay_url:
            compact["replay_url"] = replay_url
        else:
            compact.pop("replay_url", None)
        last_event = compact.get("lastEvent")
        if not isinstance(last_event, dict) or not last_event:
            compact.pop("lastEvent", None)
        runtime_usage = _stored_runtime_usage_summary(compact)
        if runtime_usage is None:
            runtime_usage = _runtime_usage_summary(
                viewer_steps=list(compact.get("viewer_steps") or []),
                normalized_events=raw_events,
            )
        compact.update(runtime_usage)
        return compact

    normalized_events = [_normalize_event(event) for event in payload.get("events", [])]
    for index, event in enumerate(normalized_events):
        event["event_index"] = index
    viewer_steps = _build_viewer_steps(normalized_events, request_snapshots=request_snapshots)
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"events", "viewer_steps", "eventCount", "lastEvent"}
    }
    compact.update(_compact_game_payload(payload, normalized_events=normalized_events, viewer_steps=viewer_steps))
    return compact


def _is_compact_saved_game_payload(payload: dict[str, Any]) -> bool:
    return "events" not in payload and isinstance(payload.get("viewer_steps"), list)


def _load_game_artifacts(run_dir: Path) -> list[_NormalizedGameArtifact]:
    return [_load_game_artifact(run_dir, path) for path in _viewer_data_paths(run_dir)]


def _load_game_summaries(run_dir: Path) -> list[dict[str, Any]]:
    return [_load_game_summary(run_dir, path) for path in _viewer_data_paths(run_dir)]


def _load_game_summary(run_dir: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = _artifact_metadata(run_dir, path)
    game_id = str(payload.get("game_id") or "").strip()
    pass_label = str(metadata.get("pass_label") or "").strip()
    if game_id and pass_label:
        metadata["display_name"] = f"{game_id} (pass {pass_label})"
    seed_label = str(metadata.get("seed_label") or "").strip()
    if game_id and seed_label:
        metadata["display_name"] = f"{game_id} (seed {seed_label})"

    if _is_compact_saved_game_payload(payload):
        compact_payload = _compact_saved_summary_payload(payload)
    else:
        compact_payload = _summary_payload_from_raw_events(payload)
    compact_payload.update(metadata)
    return _summary_from_compact_payload(compact_payload)


def _load_game_shell(run_dir: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = _artifact_metadata(run_dir, path)
    game_id = str(payload.get("game_id") or "").strip()
    pass_label = str(metadata.get("pass_label") or "").strip()
    if game_id and pass_label:
        metadata["display_name"] = f"{game_id} (pass {pass_label})"
    seed_label = str(metadata.get("seed_label") or "").strip()
    if game_id and seed_label:
        metadata["display_name"] = f"{game_id} (seed {seed_label})"

    if _is_compact_saved_game_payload(payload):
        compact_payload = _compact_saved_summary_payload(payload)
    else:
        compact_payload = _summary_payload_from_raw_events(payload)

    raw_events = _raw_event_dicts(payload, viewer_data_path=path)
    step_summaries = _build_lightweight_viewer_steps(raw_events)
    if not step_summaries and _is_compact_saved_game_payload(payload):
        step_summaries = [
            _lazy_step_summary_from_compact_step(step, index)
            for index, step in enumerate(payload.get("viewer_steps") or [])
            if isinstance(step, dict)
        ]

    compact_payload.update(metadata)
    compact_payload["viewer_steps"] = step_summaries
    compact_payload["stepCount"] = len(step_summaries)
    compact_payload["stepsAreLazy"] = True
    return compact_payload


def _load_game_step(run_dir: Path, path: Path, step_index: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = _raw_event_dicts(payload, viewer_data_path=path)
    step_summaries = _build_lightweight_viewer_steps(raw_events)
    if not step_summaries and _is_compact_saved_game_payload(payload):
        steps = [step for step in payload.get("viewer_steps") or [] if isinstance(step, dict)]
        if step_index >= len(steps):
            raise FileNotFoundError(f"Step index {step_index} not found in {path}")
        return {
            "stepIndex": step_index,
            "stepCount": len(steps),
            "step": {**_compact_viewer_step(steps[step_index]), "detailLoaded": True},
        }

    if step_index >= len(step_summaries):
        raise FileNotFoundError(f"Step index {step_index} not found in {path}")

    artifact_run_dir = _artifact_run_dir(path)
    game_id = str(payload.get("game_id") or "").strip()
    request_snapshots = _load_request_snapshots(
        _resolve_request_log_path(run_dir=artifact_run_dir, viewer_data_path=path, game_id=game_id)
    )
    hydrated_step = _hydrate_lightweight_step(
        step_summaries[step_index],
        raw_events,
        request_snapshots=request_snapshots,
        step_index=step_index,
    )
    return {
        "stepIndex": step_index,
        "stepCount": len(step_summaries),
        "step": hydrated_step,
    }


def _compact_saved_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in payload.items()
        if key not in {"events", "viewer_steps"}
    }
    compact["game_id"] = str(compact.get("game_id") or "").strip()
    compact["status"] = str(compact.get("status") or "").strip()
    viewer_steps = list(payload.get("viewer_steps") or [])
    compact["eventCount"] = int(compact.get("eventCount") or len(viewer_steps))
    if isinstance(payload.get("lastEvent"), dict) and payload.get("lastEvent"):
        compact["lastEvent"] = dict(payload["lastEvent"])
    runtime_usage = _stored_runtime_usage_summary(compact)
    if runtime_usage is None:
        runtime_usage = _runtime_usage_summary(viewer_steps=viewer_steps)
    compact.update(runtime_usage)
    return compact


def _summary_payload_from_raw_events(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_events = [_normalize_event(event) for event in payload.get("events", [])]
    compact = {
        "game_id": str(payload.get("game_id") or "").strip(),
        "status": str(payload.get("status") or "").strip(),
        "eventCount": len(normalized_events),
    }
    replay_url = str(payload.get("replay_url") or "").strip()
    if replay_url:
        compact["replay_url"] = replay_url
    last_event = _compact_last_event_summary(normalized_events[-1] if normalized_events else None)
    if last_event is not None:
        compact["lastEvent"] = last_event
    compact.update(_runtime_usage_summary(normalized_events=normalized_events))
    return compact


def _raw_event_dicts(payload: dict[str, Any], *, viewer_data_path: Path | None = None) -> list[dict[str, Any]]:
    raw_events = payload.get("events")
    if isinstance(raw_events, list):
        return [dict(event) for event in raw_events if isinstance(event, dict)]
    return [dict(event) for event in load_raw_events(payload, viewer_data_path=viewer_data_path)]


def _load_game_artifact(run_dir: Path, path: Path) -> _NormalizedGameArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = _artifact_metadata(run_dir, path)
    game_id = str(payload.get("game_id") or "").strip()
    pass_label = str(metadata.get("pass_label") or "").strip()
    if game_id and pass_label:
        metadata["display_name"] = f"{game_id} (pass {pass_label})"
    seed_label = str(metadata.get("seed_label") or "").strip()
    if game_id and seed_label:
        metadata["display_name"] = f"{game_id} (seed {seed_label})"
    artifact_run_dir = _artifact_run_dir(path)
    request_snapshots = _load_request_snapshots(
        _resolve_request_log_path(run_dir=artifact_run_dir, viewer_data_path=path, game_id=game_id)
    )
    if _is_compact_saved_game_payload(payload):
        compact_payload = compact_saved_game_payload(payload, viewer_data_path=path, request_snapshots=request_snapshots)
        compact_payload.update(metadata)
        return _NormalizedGameArtifact(
            summary=_summary_from_compact_payload(compact_payload),
            compact=compact_payload,
            full=compact_payload,
        )

    normalized_events = [_normalize_event(event) for event in payload.get("events", [])]
    for index, event in enumerate(normalized_events):
        event["event_index"] = index
    viewer_steps = _build_viewer_steps(normalized_events, request_snapshots=request_snapshots)
    compact_payload = _compact_game_payload(payload, normalized_events=normalized_events, viewer_steps=viewer_steps)
    compact_payload.update(metadata)
    full_payload = {
        **payload,
        **metadata,
        "events": normalized_events,
        "viewer_steps": viewer_steps,
        "runtimeSymbolUseCount": compact_payload["runtimeSymbolUseCount"],
        "runtimeSymbolUsage": compact_payload["runtimeSymbolUsage"],
    }
    return _NormalizedGameArtifact(
        summary=_summary_from_compact_payload(compact_payload),
        compact=compact_payload,
        full=full_payload,
    )


def _summary_from_compact_payload(compact_payload: dict[str, Any]) -> dict[str, Any]:
    game: dict[str, Any] = {
        "game_id": str(compact_payload.get("game_id") or "").strip(),
        "status": str(compact_payload.get("status") or "").strip(),
        "eventCount": int(compact_payload.get("eventCount") or 0),
        "runtimeSymbolUseCount": int(compact_payload.get("runtimeSymbolUseCount") or 0),
        "runtimeSymbolUsage": list(compact_payload.get("runtimeSymbolUsage") or []),
    }
    seed_label = str(compact_payload.get("seed_label") or "").strip()
    if seed_label:
        game["seed_label"] = seed_label
        game["display_name"] = f"{game['game_id']} (seed {seed_label})"
    pass_label = str(compact_payload.get("pass_label") or "").strip()
    if pass_label:
        game["pass_label"] = pass_label
        game["display_name"] = f"{game['game_id']} (pass {pass_label})"
    display_name = str(compact_payload.get("display_name") or "").strip()
    if display_name:
        game["display_name"] = display_name
    if compact_payload.get("pass_index") is not None:
        game["pass_index"] = compact_payload.get("pass_index")
    if compact_payload.get("seed") is not None:
        game["seed"] = compact_payload.get("seed")
    seed_dir = str(compact_payload.get("seed_dir") or "").strip()
    if seed_dir:
        game["seed_dir"] = seed_dir
    artifact_run_dir = str(compact_payload.get("artifact_run_dir") or "").strip()
    if artifact_run_dir:
        game["artifact_run_dir"] = artifact_run_dir
    replay_url = str(compact_payload.get("replay_url") or "").strip()
    if replay_url:
        game["replay_url"] = replay_url
    last_event = compact_payload.get("lastEvent")
    if isinstance(last_event, dict) and last_event:
        game["lastEvent"] = dict(last_event)
    return game


def _compact_game_payload(
    payload: dict[str, Any],
    *,
    normalized_events: list[dict[str, Any]],
    viewer_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    game: dict[str, Any] = {
        "game_id": str(payload.get("game_id") or "").strip(),
        "status": str(payload.get("status") or "").strip(),
        "viewer_steps": [_compact_viewer_step(step) for step in viewer_steps],
        "eventCount": len(normalized_events),
    }
    replay_url = str(payload.get("replay_url") or "").strip()
    if replay_url:
        game["replay_url"] = replay_url
    last_event = _compact_last_event_summary(normalized_events[-1] if normalized_events else None)
    if last_event is not None:
        game["lastEvent"] = last_event
    game.update(_runtime_usage_summary(viewer_steps=viewer_steps, normalized_events=normalized_events))
    return game


def _normalize_runtime_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            continue
        if isinstance(current, ast.Subscript):
            current = current.value
            continue
        if isinstance(current, ast.Name):
            parts.append(current.id)
            break
        return None
    if not parts:
        return None
    return ".".join(reversed(parts))


def _extract_python_code(tool_call_content: str) -> str:
    text = str(tool_call_content or "").strip()
    if not text:
        return ""
    match = _TOOL_CODE_PARAMETER_RE.search(text)
    if match:
        return str(match.group(1) or "").strip("\n")
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if isinstance(parsed, dict):
        code = parsed.get("code")
        if isinstance(code, str):
            return code.strip("\n")
    return ""


def _annotate_ast_parents(node: ast.AST) -> None:
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_viewer_parent", parent)


def _count_runtime_symbols_in_code(code: str) -> dict[str, int]:
    if not code.strip():
        return {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    _annotate_ast_parents(tree)

    counts: dict[str, int] = defaultdict(int)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _normalize_runtime_chain(node.func)
            if chain in _RUNTIME_CALL_SYMBOLS:
                counts[chain] += 1
            continue
        if isinstance(node, ast.Attribute):
            parent = getattr(node, "_viewer_parent", None)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                # Counted via the enclosing Subscript; avoid double-counting
                # subscripted access such as `current_frame.segmentation['nodes']`.
                continue
            chain = _normalize_runtime_chain(node)
            if chain in _RUNTIME_VALUE_SYMBOLS:
                counts[chain] += 1
            continue
        if isinstance(node, ast.Subscript):
            chain = _normalize_runtime_chain(node)
            if chain in _RUNTIME_VALUE_SYMBOLS:
                counts[chain] += 1
            continue
        if isinstance(node, ast.Name):
            parent = getattr(node, "_viewer_parent", None)
            if isinstance(parent, ast.Attribute):
                continue
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            if isinstance(parent, ast.Subscript) and parent.value is node:
                continue
            if node.id in _RUNTIME_VALUE_SYMBOLS:
                counts[node.id] += 1
    return counts



def _stored_runtime_usage_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw_usage = payload.get("runtimeSymbolUsage")
    raw_total = payload.get("runtimeSymbolUseCount")
    if raw_usage is None and raw_total is None:
        return None

    usage: list[dict[str, Any]] = []
    if isinstance(raw_usage, list):
        for item in raw_usage:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                count = int(item.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            usage.append({"name": name, "count": count})

    try:
        total = int(raw_total)
    except (TypeError, ValueError):
        total = sum(int(item["count"]) for item in usage)

    return {
        "runtimeSymbolUseCount": total,
        "runtimeSymbolUsage": usage,
    }


def _runtime_usage_summary(
    *,
    viewer_steps: list[dict[str, Any]] | None = None,
    normalized_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)

    for step in viewer_steps or []:
        context = step.get("localContext") or step.get("context") or {}
        for section in context.get("sections") or []:
            label = str(section.get("label") or "").strip()
            if label != "TOOL CALL: python":
                continue
            code = _extract_python_code(str(section.get("content") or ""))
            for name, count in _count_runtime_symbols_in_code(code).items():
                counts[name] += count

    if not counts:
        for event in normalized_events or []:
            for section in event.get("transcript_sections", []):
                label = str(section.get("label") or "").strip()
                if label != "TOOL CALL: python":
                    continue
                code = _extract_python_code(str(section.get("content") or ""))
                for name, count in _count_runtime_symbols_in_code(code).items():
                    counts[name] += count

    ranked = [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "runtimeSymbolUseCount": sum(counts.values()),
        "runtimeSymbolUsage": ranked,
    }


def _compact_last_event_summary(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    summary: dict[str, Any] = {}
    event_type = str(event.get("type") or "").strip()
    if event_type:
        summary["type"] = event_type
    status = str(event.get("status") or "").strip()
    if status:
        summary["status"] = status
    for key in ("action_num", "score", "reward"):
        if event.get(key) is not None:
            summary[key] = event.get(key)
    transcript = str(event.get("transcript") or "")
    if transcript:
        summary["transcriptLength"] = len(transcript)
    return summary or None


def _compact_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not context:
        return None
    sections: list[dict[str, Any]] = []
    for raw_section in context.get("sections", []):
        label = str(raw_section.get("label") or "").strip()
        content = str(raw_section.get("content") or "")
        kind = str(raw_section.get("kind") or "tool").strip() or "tool"
        section: dict[str, Any] = {
            "label": label,
            "content": content,
            "kind": kind,
        }
        if raw_section.get("inContext") is False:
            section["inContext"] = False
        sections.append(section)
    if not sections:
        return None
    return {"sections": sections}


def _compact_step_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    compact: dict[str, Any] = {}
    event_type = str(event.get("type") or "").strip()
    if event_type:
        compact["type"] = event_type
    for key in ("action_display", "action_name", "state"):
        value = event.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                compact[key] = text
    for key in ("reward", "score", "level"):
        if event.get(key) is not None:
            compact[key] = event.get(key)
    board = event.get("board")
    if isinstance(board, list) and board:
        compact["board"] = board
    else:
        board_ascii = str(event.get("board_ascii") or "")
        if board_ascii:
            compact["board_ascii"] = board_ascii
    return compact or None


def _compact_viewer_step(step: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "title": str(step.get("title") or "").strip(),
        "batchSize": int(step.get("batchSize") or 0),
        "stepKind": str(step.get("stepKind") or "event").strip() or "event",
    }
    if step.get("actionDisplay"):
        compact["actionDisplay"] = step.get("actionDisplay")
    if step.get("stepRangeLabel"):
        compact["stepRangeLabel"] = step.get("stepRangeLabel")
    if step.get("stateTransition"):
        compact["stateTransition"] = step.get("stateTransition")
    for key in ("reward", "score", "state", "level"):
        if step.get(key) is not None:
            compact[key] = step.get(key)
    context = _compact_context(step.get("context"))
    if context is not None:
        compact["context"] = context
    local_context = _compact_context(step.get("localContext"))
    if local_context is not None:
        compact["localContext"] = local_context
    board_event = _compact_step_event(step.get("boardEvent") or step.get("event"))
    if board_event is not None:
        compact["boardEvent"] = board_event
    elif step.get("event") is not None:
        fallback_event = _compact_step_event(step.get("event"))
        if fallback_event is not None:
            compact["event"] = fallback_event
    return compact


def _resolve_request_log_path(*, run_dir: Path, viewer_data_path: Path, game_id: str) -> Path | None:
    candidates: list[Path] = []
    if game_id:
        candidates.append(run_dir / f"{game_id}_requests.jsonl")
    suffix = "_viewer_data.json"
    name = viewer_data_path.name
    if name.endswith(suffix):
        stem = name[:-len(suffix)]
        if stem:
            candidates.append(run_dir / f"{stem}_requests.jsonl")
    candidates.append(run_dir / "requests.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_request_snapshots(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    snapshots: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        snapshots.append(
            {
                "messages": list(payload.get("messages") or []),
                "tools": list(payload.get("tools") or []),
                "tool_choice": str(payload.get("tool_choice") or "").strip() or None,
                "analysis_step": _normalize_positive_int(payload.get("analysis_step")),
                "action": _normalize_positive_int(payload.get("action")),
                "request_index_within_turn": _normalize_positive_int(payload.get("request_index_within_turn")),
            }
        )
    return snapshots


def _split_labeled_sections(text: str) -> list[dict[str, str]]:
    matches = [
        match
        for match in _SECTION_RE.finditer(text)
        if _is_known_section_label(match.group(1).strip())
    ]
    if not matches:
        return []

    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"label": label, "content": content, "kind": _classify_section(label)})
    return sections


def _is_known_section_label(label: str) -> bool:
    return label in _KNOWN_SECTION_LABELS or any(label.startswith(prefix) for prefix in _KNOWN_SECTION_PREFIXES)


def _classify_section(label: str) -> str:
    if label in {"ASSISTANT", "OUTPUT", "THINKING"}:
        return "reasoning"
    if label.startswith("TOOL CALL") or label.startswith("TOOL RESULT") or label.startswith("ERROR"):
        return "tool"
    return "meta"


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    normalized.setdefault("title", "Event")
    normalized.setdefault("type", "event")
    normalized.setdefault("board", None)
    normalized.setdefault("board_ascii", "")
    normalized.setdefault("transcript", "")
    normalized["transcript_sections"] = _split_labeled_sections(normalized.get("transcript", ""))
    normalized.pop("_sort_base", None)
    return normalized


def _normalized_number(value: Any) -> int | float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    return int(parsed) if parsed.is_integer() else parsed


def _normalize_analysis_step(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_step_number(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _format_step_range(first_step: int | None, last_step: int | None) -> str | None:
    if first_step is None or last_step is None:
        return None
    if first_step == last_step:
        return f"Step {first_step}"
    return f"Steps {first_step}-{last_step}"


def _build_step_title(step_number: int | None, fallback_title: str) -> str:
    if step_number is None:
        return fallback_title
    return f"Step {step_number}"


def _action_label(event: dict[str, Any]) -> str:
    return str(event.get("action_display") or event.get("action_name") or "No action")


def _state_transition(before_event: dict[str, Any] | None, after_event: dict[str, Any] | None) -> str | None:
    before_state = str((before_event or {}).get("state") or "").strip()
    after_state = str((after_event or {}).get("state") or "").strip()
    if not before_state and not after_state:
        return None
    if before_state and after_state and before_state != after_state:
        return f"{before_state} -> {after_state}"
    return after_state or before_state or None


def _action_display_with_prior_state(event: dict[str, Any], board_event: dict[str, Any] | None) -> str:
    action_label = _action_label(event)
    prior_state = str((board_event or {}).get("state") or "").strip()
    if action_label == "RESET" and prior_state == "GAME_OVER":
        return "GAME_OVER -> RESET"
    return action_label


def _normalize_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _render_tool_arguments(tool_name: str, arguments: Any) -> str:
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError, json.JSONDecodeError):
        rendered = str(arguments).strip()
        return rendered if rendered else "{}"
    if not isinstance(parsed, dict):
        return json.dumps(parsed, indent=2, ensure_ascii=True)

    lines = ["<tool_call>", f"<function={tool_name}>"]
    for parameter_name, parameter_value in parsed.items():
        lines.append(f"<parameter={parameter_name}>")
        if isinstance(parameter_value, str):
            rendered_value = parameter_value.rstrip("\n")
        elif isinstance(parameter_value, bool):
            rendered_value = "true" if parameter_value else "false"
        elif parameter_value is None:
            rendered_value = "null"
        else:
            rendered_value = json.dumps(parameter_value, indent=2, ensure_ascii=True)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_tool_result(content: Any) -> str:
    stripped = str(content or "").strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return stripped
    if not isinstance(parsed, dict):
        return json.dumps(parsed, indent=2, ensure_ascii=True)

    stdout = str(parsed.get("stdout", "") or "").rstrip("\n")
    error = str(parsed.get("error", "") or "").rstrip("\n")
    result = parsed.get("result")
    has_result = result not in (None, "", [], {})
    if stdout and not error and not has_result:
        return stdout

    blocks: list[str] = []
    if stdout:
        blocks.append(stdout)
    if has_result:
        blocks.append(
            json.dumps(result, indent=2, ensure_ascii=True) if isinstance(result, (dict, list)) else str(result)
        )
    if error:
        blocks.append(f"error:\n{error}" if (stdout or has_result) else error)
    if blocks:
        return "\n\n".join(block for block in blocks if block.strip())
    return json.dumps(parsed, indent=2, ensure_ascii=True)


def _request_snapshot_sections(snapshot: dict[str, Any], *, include_system_prompt: bool) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}
    for message in snapshot.get("messages", []):
        role = str(message.get("role", "")).strip().lower()
        content = _normalize_message_text(message.get("content", ""))
        if role == "system" and content:
            if not include_system_prompt:
                continue
            sections.append({"label": "SYSTEM PROMPT", "content": content, "kind": "meta", "inContext": True})
            continue
        if role == "user" and content:
            sections.append({"label": "USER PROMPT", "content": content, "kind": "meta", "inContext": True})
            continue
        if role == "assistant":
            if content:
                sections.append({"label": "ASSISTANT", "content": content, "kind": "reasoning", "inContext": True})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                name = str(function.get("name", "")).strip() or "unknown"
                tool_call_id = str(tool_call.get("id", "")).strip()
                if tool_call_id:
                    tool_names_by_id[tool_call_id] = name
                sections.append(
                    {
                        "label": f"TOOL CALL: {name}",
                        "content": _render_tool_arguments(name, function.get("arguments", "{}")),
                        "kind": "tool",
                        "inContext": True,
                    }
                )
            continue
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            tool_name = tool_names_by_id.get(tool_call_id, "unknown")
            if content:
                sections.append(
                    {
                        "label": f"TOOL RESULT: {tool_name}",
                        "content": _render_tool_result(content),
                        "kind": "tool",
                        "inContext": True,
                    }
                )
    return sections


def _transcript_sections_from_events(
    analysis_events: list[dict[str, Any]],
    *,
    include_system_prompt: bool,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    system_prompt_included = False
    for event in analysis_events:
        for section in event.get("transcript_sections", []):
            label = str(section.get("label", "")).strip()
            content = str(section.get("content", "")).strip()
            if not content or label == "ANALYZER STATUS" or label in _HIDDEN_SECTION_LABELS:
                continue
            if label == "SYSTEM PROMPT":
                if not include_system_prompt or system_prompt_included:
                    continue
                system_prompt_included = True
            kind = str(section.get("kind", "tool"))
            sections.append(
                {
                    "label": label,
                    "content": content,
                    "kind": "tool" if kind == "meta" else kind,
                }
            )
    return sections


def _latest_request_snapshot(
    request_snapshots: list[dict[str, Any]],
    *,
    analysis_step: int | None,
) -> dict[str, Any] | None:
    if analysis_step is None:
        return None
    matches = [snapshot for snapshot in request_snapshots if snapshot.get("analysis_step") == analysis_step]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda snapshot: (
            int(snapshot.get("request_index_within_turn") or 0),
            int(snapshot.get("action") or 0),
        ),
    )[-1]


def _section_signature(section: dict[str, Any]) -> tuple[str, str]:
    label = str(section.get("label", "")).strip()
    content = str(section.get("content", "")).strip()
    if label.startswith("TOOL CALL:") or label.startswith("TOOL RESULT:") or label.startswith("ERROR"):
        try:
            canonical = json.dumps(
                json.loads(content),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            canonical = re.sub(r"\s+", " ", content)
        return (label, canonical)
    return (
        label,
        content,
    )


def _default_in_context_for_transcript_section(section: dict[str, Any]) -> bool:
    return str(section.get("label", "")).strip() != "THINKING"


def _extract_context(
    analysis_events: list[dict[str, Any]] | None,
    *,
    include_system_prompt: bool,
    request_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not analysis_events:
        return None

    request_sections = _request_snapshot_sections(request_snapshot or {}, include_system_prompt=include_system_prompt)
    transcript_sections = _transcript_sections_from_events(
        list(analysis_events),
        include_system_prompt=include_system_prompt,
    )

    sections: list[dict[str, Any]] = []
    if request_sections:
        transcript_insertions: dict[int, list[dict[str, Any]]] = defaultdict(list)
        search_start = 0
        request_signatures = [_section_signature(section) for section in request_sections]
        for section in transcript_sections:
            signature = _section_signature(section)
            found_index = None
            for index in range(search_start, len(request_signatures)):
                if request_signatures[index] == signature:
                    found_index = index
                    break
            if found_index is None:
                transcript_insertions[search_start].append(
                    {**section, "inContext": _default_in_context_for_transcript_section(section)}
                )
                continue
            search_start = found_index + 1

        for index in range(len(request_sections) + 1):
            sections.extend(transcript_insertions.get(index, []))
            if index < len(request_sections):
                sections.append(request_sections[index])
    else:
        sections.extend(
            {**section, "inContext": _default_in_context_for_transcript_section(section)}
            for section in transcript_sections
        )

    transcript = "\n\n".join(
        str(event.get("transcript", "")).strip()
        for event in analysis_events
        if str(event.get("transcript", "")).strip()
    ).strip()
    if not sections and transcript:
        sections.append(
            {
                "label": "TRANSCRIPT",
                "content": transcript,
                "kind": "tool",
                "inContext": False,
            }
        )

    if not sections:
        return None
    return {
        "sections": sections,
        "hasExactModelContext": bool(request_sections),
        "requestIndexWithinTurn": request_snapshot.get("request_index_within_turn") if request_snapshot else None,
        "messageCount": len(request_snapshot.get("messages", [])) if request_snapshot else 0,
    }


def _new_analysis_group(analysis_step: int, source_event_index: int) -> dict[str, Any]:
    return {
        "analysis_step": analysis_step,
        "source_event_index": source_event_index,
        "analysis_event": None,
        "pre_board_event": None,
        "actions": [],
        "last_action_event": None,
    }


def _build_standalone_action_step(
    event: dict[str, Any],
    event_index: int,
    *,
    board_event: dict[str, Any] | None,
) -> dict[str, Any]:
    board_source = board_event or event
    action_num = _normalize_step_number(event.get("action_num"))
    return {
        "title": _build_step_title(action_num, f"Step {event_index + 1}"),
        "sourceEventIndex": event_index,
        "event": event,
        "boardEvent": board_source,
        "context": None,
        "actionDisplay": _action_display_with_prior_state(event, board_event),
        "stepRangeLabel": None,
        "batchSize": 1,
        "reward": event.get("reward"),
        "score": board_source.get("score"),
        "state": board_source.get("state"),
        "stateTransition": _state_transition(board_event, event),
        "level": board_source.get("level"),
        "stepKind": "action",
    }


def _build_action_summary(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not actions:
        return None

    first_action_num = _normalize_step_number(actions[0].get("action_num"))
    last_action_num = _normalize_step_number(actions[-1].get("action_num"))
    prior_event = actions[0].get("_viewer_pre_board_event")
    return {
        "actionDisplay": " -> ".join(
            _action_display_with_prior_state(action, prior_event if index == 0 else None)
            for index, action in enumerate(actions)
        ),
        "stepRangeLabel": _format_step_range(first_action_num, last_action_num),
        "batchSize": len(actions),
        "reward": sum(_normalized_number(action.get("reward")) for action in actions),
        "stateTransition": _state_transition(prior_event, actions[-1]),
    }


def _apply_action_summary(step: dict[str, Any], action_summary: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(step)
    payload["actionDisplay"] = action_summary.get("actionDisplay") if action_summary else None
    payload["stepRangeLabel"] = action_summary.get("stepRangeLabel") if action_summary else None
    payload["batchSize"] = int(action_summary.get("batchSize", 0)) if action_summary else 0
    payload["reward"] = action_summary.get("reward") if action_summary else None
    payload["stateTransition"] = action_summary.get("stateTransition") if action_summary else None
    return payload


def _build_analysis_frame_step(
    group: dict[str, Any],
    *,
    include_system_prompt: bool,
    action_summary: dict[str, Any] | None,
    request_snapshots: list[dict[str, Any]],
    prior_analysis_events: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis_event = group.get("analysis_event")
    board_event = group.get("pre_board_event") or analysis_event or group.get("last_action_event")
    summary_event = analysis_event or board_event or group.get("last_action_event")
    request_snapshot = _latest_request_snapshot(
        request_snapshots,
        analysis_step=_normalize_analysis_step(group.get("analysis_step")),
    )
    context_events = [analysis_event] if (request_snapshot is not None and analysis_event is not None) else prior_analysis_events

    return _apply_action_summary(
        {
            "title": _build_step_title(group.get("analysis_step"), f"Turn {group.get('analysis_step')}"),
            "sourceEventIndex": group.get("source_event_index", 0),
            "event": summary_event,
            "boardEvent": board_event,
            "context": _extract_context(
                context_events,
                include_system_prompt=include_system_prompt,
                request_snapshot=request_snapshot,
            ),
            "localContext": _extract_context(
                [analysis_event] if analysis_event is not None else [],
                include_system_prompt=include_system_prompt,
            ),
            "score": board_event.get("score") if board_event else None,
            "state": board_event.get("state") if board_event else None,
            "level": board_event.get("level") if board_event else None,
            "stepKind": "turn",
        },
        action_summary,
    )


def _build_latest_state_step(
    group: dict[str, Any],
    *,
    action_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_event = group.get("last_action_event") or group.get("analysis_event") or group.get("pre_board_event")
    return _apply_action_summary(
        {
            "title": "Latest State",
            "sourceEventIndex": int((latest_event or {}).get("event_index", group.get("source_event_index", 0))) + 1,
            "event": latest_event,
            "boardEvent": latest_event,
            "context": None,
            "score": latest_event.get("score") if latest_event else None,
            "state": latest_event.get("state") if latest_event else None,
            "level": latest_event.get("level") if latest_event else None,
            "stepKind": "result",
        },
        action_summary,
    )


def _lightweight_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    payload = dict(event)
    payload.setdefault("title", "Event")
    payload.setdefault("type", "event")
    payload.setdefault("board", None)
    payload.setdefault("board_ascii", "")
    payload.pop("transcript", None)
    payload.pop("transcript_sections", None)
    return payload


def _compact_lightweight_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    return _compact_step_event(_lightweight_event(event))


def _lazy_step_summary_from_compact_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    summary = _compact_viewer_step(step)
    summary["sourceEventIndex"] = int(step.get("sourceEventIndex") or index)
    summary["lazy"] = True
    summary["detailLoaded"] = False
    return summary


def _lightweight_analysis_step(
    group: dict[str, Any],
    *,
    index: int,
    action_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis_event = group.get("analysis_event")
    board_event = group.get("pre_board_event") or analysis_event or group.get("last_action_event")
    summary_event = analysis_event or board_event or group.get("last_action_event")
    step = _apply_action_summary(
        {
            "title": _build_step_title(group.get("analysis_step"), f"Turn {group.get('analysis_step')}"),
            "sourceEventIndex": group.get("source_event_index", 0),
            "event": _compact_lightweight_event(summary_event),
            "boardEvent": _compact_lightweight_event(board_event),
            "score": board_event.get("score") if board_event else None,
            "state": board_event.get("state") if board_event else None,
            "level": board_event.get("level") if board_event else None,
            "stepKind": "turn",
            "analysisStep": group.get("analysis_step"),
            "lazy": True,
            "detailLoaded": False,
        },
        action_summary,
    )
    step["stepIndex"] = index
    return step


def _lightweight_latest_state_step(
    group: dict[str, Any],
    *,
    index: int,
    action_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_event = group.get("last_action_event") or group.get("analysis_event") or group.get("pre_board_event")
    step = _apply_action_summary(
        {
            "title": "Latest State",
            "sourceEventIndex": int((latest_event or {}).get("event_index", group.get("source_event_index", 0))) + 1,
            "event": _compact_lightweight_event(latest_event),
            "boardEvent": _compact_lightweight_event(latest_event),
            "score": latest_event.get("score") if latest_event else None,
            "state": latest_event.get("state") if latest_event else None,
            "level": latest_event.get("level") if latest_event else None,
            "stepKind": "result",
            "analysisStep": group.get("analysis_step"),
            "lazy": True,
            "detailLoaded": False,
        },
        action_summary,
    )
    step["stepIndex"] = index
    return step


def _lightweight_standalone_action_step(
    event: dict[str, Any],
    event_index: int,
    *,
    board_event: dict[str, Any] | None,
    index: int,
) -> dict[str, Any]:
    board_source = board_event or event
    action_num = _normalize_step_number(event.get("action_num"))
    return {
        "title": _build_step_title(action_num, f"Step {event_index + 1}"),
        "sourceEventIndex": event_index,
        "event": _compact_lightweight_event(event),
        "boardEvent": _compact_lightweight_event(board_source),
        "actionDisplay": _action_display_with_prior_state(event, board_event),
        "stepRangeLabel": None,
        "batchSize": 1,
        "reward": event.get("reward"),
        "score": board_source.get("score"),
        "state": board_source.get("state"),
        "stateTransition": _state_transition(board_event, event),
        "level": board_source.get("level"),
        "stepKind": "action",
        "stepIndex": index,
        "lazy": True,
        "detailLoaded": False,
    }


def _build_lightweight_viewer_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []

    initial_event = next((event for event in events if event.get("type") == "initial"), None)
    current_board_event = initial_event
    analysis_groups: dict[int, dict[str, Any]] = {}
    standalone_steps: list[dict[str, Any]] = []

    for event_index, raw_event in enumerate(events):
        event = dict(raw_event)
        event["event_index"] = event_index
        event_type = event.get("type")
        if event_type == "initial":
            current_board_event = event
            continue

        if event_type == "action":
            analysis_step = _normalize_analysis_step(event.get("analysis_step"))
            board_before_action = current_board_event or event
            if analysis_step is None:
                standalone_steps.append(
                    _lightweight_standalone_action_step(
                        event,
                        event_index,
                        board_event=board_before_action,
                        index=0,
                    )
                )
            else:
                group = analysis_groups.setdefault(analysis_step, _new_analysis_group(analysis_step, event_index))
                group["source_event_index"] = min(group["source_event_index"], event_index)
                if group["pre_board_event"] is None:
                    group["pre_board_event"] = board_before_action
                event["_viewer_pre_board_event"] = board_before_action
                group["actions"].append(event)
                group["last_action_event"] = event
            current_board_event = event
            continue

        if event_type == "analysis":
            analysis_step = _normalize_analysis_step(event.get("analysis_step"))
            if analysis_step is None:
                standalone_steps.append(
                    {
                        "title": event.get("title") or f"Analysis {event_index + 1}",
                        "sourceEventIndex": event_index,
                        "event": _compact_lightweight_event(event),
                        "boardEvent": _compact_lightweight_event(current_board_event or event),
                        "actionDisplay": None,
                        "stepRangeLabel": None,
                        "batchSize": 0,
                        "reward": event.get("reward"),
                        "score": (current_board_event or event).get("score"),
                        "state": (current_board_event or event).get("state"),
                        "level": (current_board_event or event).get("level"),
                        "stepKind": "turn",
                        "lazy": True,
                        "detailLoaded": False,
                    }
                )
            else:
                group = analysis_groups.setdefault(analysis_step, _new_analysis_group(analysis_step, event_index))
                group["source_event_index"] = min(group["source_event_index"], event_index)
                group["analysis_event"] = event
                if group["pre_board_event"] is None:
                    group["pre_board_event"] = current_board_event or event
            continue

        if event.get("board"):
            current_board_event = event

    ordered_analysis_groups = sorted(
        analysis_groups.values(),
        key=lambda group: (group["source_event_index"], group["analysis_step"]),
    )
    analysis_steps: list[dict[str, Any]] = []
    previous_action_summary: dict[str, Any] | None = None
    for group_index, group in enumerate(ordered_analysis_groups):
        analysis_steps.append(
            _lightweight_analysis_step(
                group,
                index=group_index,
                action_summary=previous_action_summary,
            )
        )
        previous_action_summary = _build_action_summary(list(group.get("actions", [])))

    if ordered_analysis_groups and previous_action_summary is not None:
        analysis_steps.append(
            _lightweight_latest_state_step(
                ordered_analysis_groups[-1],
                index=len(analysis_steps),
                action_summary=previous_action_summary,
            )
        )

    steps = sorted(
        [*analysis_steps, *standalone_steps],
        key=lambda step: int(step.get("sourceEventIndex", 0)),
    )
    for index, step in enumerate(steps):
        step["stepIndex"] = index

    if steps:
        return steps

    if initial_event is None:
        return []

    return [
        {
            "title": "INITIAL",
            "sourceEventIndex": 0,
            "event": _compact_lightweight_event(initial_event),
            "boardEvent": _compact_lightweight_event(initial_event),
            "actionDisplay": initial_event.get("action_display") or "RESET",
            "stepRangeLabel": None,
            "batchSize": 0,
            "reward": initial_event.get("reward"),
            "score": initial_event.get("score"),
            "state": initial_event.get("state"),
            "level": initial_event.get("level"),
            "stepKind": "initial",
            "stepIndex": 0,
            "lazy": True,
            "detailLoaded": False,
        }
    ]


def _hydrate_lightweight_step(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    request_snapshots: list[dict[str, Any]],
    step_index: int,
) -> dict[str, Any]:
    step = dict(summary)
    step["detailLoaded"] = True
    analysis_step = _normalize_analysis_step(summary.get("analysisStep"))
    if analysis_step is None:
        return step

    analysis_event = next(
        (
            event
            for event in events
            if event.get("type") == "analysis" and _normalize_analysis_step(event.get("analysis_step")) == analysis_step
        ),
        None,
    )
    if analysis_event is None:
        return step

    normalized_event = _normalize_event(analysis_event)
    request_snapshot = _latest_request_snapshot(request_snapshots, analysis_step=analysis_step)
    context = _extract_context(
        [normalized_event],
        include_system_prompt=step_index == 0,
        request_snapshot=request_snapshot,
    )
    if context is not None:
        step["localContext"] = context
        step["context"] = context
    return step


def _build_viewer_steps(events: list[dict[str, Any]], *, request_snapshots: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not events:
        return []
    request_snapshots = list(request_snapshots or [])

    initial_event = next((event for event in events if event.get("type") == "initial"), None)
    current_board_event = initial_event
    analysis_groups: dict[int, dict[str, Any]] = {}
    standalone_steps: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "initial":
            current_board_event = event
            continue

        if event_type == "action":
            analysis_step = _normalize_analysis_step(event.get("analysis_step"))
            board_before_action = current_board_event or event
            if analysis_step is None:
                standalone_steps.append(
                    _build_standalone_action_step(event, event_index, board_event=board_before_action)
                )
            else:
                group = analysis_groups.setdefault(analysis_step, _new_analysis_group(analysis_step, event_index))
                group["source_event_index"] = min(group["source_event_index"], event_index)
                if group["pre_board_event"] is None:
                    group["pre_board_event"] = board_before_action
                event["_viewer_pre_board_event"] = board_before_action
                group["actions"].append(event)
                group["last_action_event"] = event
            current_board_event = event
            continue

        if event_type == "analysis":
            analysis_step = _normalize_analysis_step(event.get("analysis_step"))
            if analysis_step is None:
                standalone_steps.append(
                    {
                        "title": event.get("title") or f"Analysis {event_index + 1}",
                        "sourceEventIndex": event_index,
                        "event": event,
                        "boardEvent": current_board_event or event,
                        "context": _extract_context([event], include_system_prompt=True),
                        "localContext": _extract_context([event], include_system_prompt=True),
                        "actionDisplay": None,
                        "stepRangeLabel": None,
                        "batchSize": 0,
                        "reward": event.get("reward"),
                        "score": (current_board_event or event).get("score"),
                        "state": (current_board_event or event).get("state"),
                        "level": (current_board_event or event).get("level"),
                        "stepKind": "turn",
                    }
                )
            else:
                group = analysis_groups.setdefault(analysis_step, _new_analysis_group(analysis_step, event_index))
                group["source_event_index"] = min(group["source_event_index"], event_index)
                group["analysis_event"] = event
                if group["pre_board_event"] is None:
                    group["pre_board_event"] = current_board_event or event
            continue

        if event.get("board"):
            current_board_event = event

    ordered_analysis_groups = sorted(
        analysis_groups.values(),
        key=lambda group: (group["source_event_index"], group["analysis_step"]),
    )
    analysis_steps: list[dict[str, Any]] = []
    previous_action_summary: dict[str, Any] | None = None
    cumulative_analysis_events: list[dict[str, Any]] = []
    for index, group in enumerate(ordered_analysis_groups):
        analysis_event = group.get("analysis_event")
        if analysis_event is not None:
            cumulative_analysis_events.append(analysis_event)
        analysis_steps.append(
            _build_analysis_frame_step(
                group,
                include_system_prompt=index == 0,
                action_summary=previous_action_summary,
                request_snapshots=request_snapshots,
                prior_analysis_events=list(cumulative_analysis_events),
            )
        )
        previous_action_summary = _build_action_summary(list(group.get("actions", [])))

    if ordered_analysis_groups and previous_action_summary is not None:
        analysis_steps.append(
            _build_latest_state_step(
                ordered_analysis_groups[-1],
                action_summary=previous_action_summary,
            )
        )

    steps = sorted(
        [*analysis_steps, *standalone_steps],
        key=lambda step: int(step.get("sourceEventIndex", 0)),
    )

    if steps:
        return steps

    if initial_event is None:
        return []

    return [
        {
            "title": "INITIAL",
            "sourceEventIndex": 0,
            "event": initial_event,
            "boardEvent": initial_event,
            "context": None,
            "actionDisplay": initial_event.get("action_display") or "RESET",
            "stepRangeLabel": None,
            "batchSize": 0,
            "reward": initial_event.get("reward"),
            "score": initial_event.get("score"),
            "state": initial_event.get("state"),
            "level": initial_event.get("level"),
            "stepKind": "initial",
        }
    ]
