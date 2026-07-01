from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


_VIEWER_DATA_SUFFIX = "_viewer_data.json"
_RAW_EVENTS_SUFFIX = "_events.json.gz"
_RAW_EVENTS_JSONL_SUFFIX = "_events.jsonl"


def raw_events_sidecar_path(viewer_data_path: Path) -> Path:
    name = viewer_data_path.name
    if name.endswith(_VIEWER_DATA_SUFFIX):
        stem = name[: -len(_VIEWER_DATA_SUFFIX)]
        sidecar_name = f"{stem}{_RAW_EVENTS_SUFFIX}" if stem else f"viewer_data{_RAW_EVENTS_SUFFIX}"
        return viewer_data_path.with_name(sidecar_name)
    if name.endswith(".json"):
        return viewer_data_path.with_name(f"{viewer_data_path.stem}_events.json.gz")
    return viewer_data_path.with_name(f"{name}_events.json.gz")


def raw_events_jsonl_sidecar_path(viewer_data_path: Path) -> Path:
    name = viewer_data_path.name
    if name.endswith(_VIEWER_DATA_SUFFIX):
        stem = name[: -len(_VIEWER_DATA_SUFFIX)]
        sidecar_name = f"{stem}{_RAW_EVENTS_JSONL_SUFFIX}" if stem else f"viewer_data{_RAW_EVENTS_JSONL_SUFFIX}"
        return viewer_data_path.with_name(sidecar_name)
    if name.endswith(".json"):
        return viewer_data_path.with_name(f"{viewer_data_path.stem}_events.jsonl")
    return viewer_data_path.with_name(f"{name}_events.jsonl")


def reset_raw_events_sidecar(viewer_data_path: Path) -> None:
    raw_events_jsonl_sidecar_path(viewer_data_path).unlink(missing_ok=True)
    raw_events_sidecar_path(viewer_data_path).unlink(missing_ok=True)


def append_raw_events_sidecar(viewer_data_path: Path, events: list[dict[str, Any]]) -> Path:
    sidecar_path = raw_events_jsonl_sidecar_path(viewer_data_path)
    if not events:
        return sidecar_path
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("a", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, separators=(",", ":")) + "\n")
    return sidecar_path


def write_raw_events_sidecar(viewer_data_path: Path, events: list[dict[str, Any]]) -> Path:
    sidecar_path = raw_events_sidecar_path(viewer_data_path)
    payload = json.dumps(events, separators=(",", ":")).encode("utf-8")
    sidecar_path.write_bytes(gzip.compress(payload))
    return sidecar_path


def _load_jsonl_events(sidecar_path: Path) -> list[dict[str, Any]] | None:
    if not sidecar_path.exists():
        return None
    try:
        lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            return []
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def load_raw_events(
    payload: dict[str, Any],
    *,
    viewer_data_path: Path | None = None,
) -> list[dict[str, Any]]:
    raw_events = payload.get("events")
    if isinstance(raw_events, list):
        return [event for event in raw_events if isinstance(event, dict)]

    if viewer_data_path is None:
        return []

    jsonl_events = _load_jsonl_events(raw_events_jsonl_sidecar_path(viewer_data_path))
    if jsonl_events is not None:
        return jsonl_events

    sidecar_path = raw_events_sidecar_path(viewer_data_path)
    if not sidecar_path.exists():
        return []

    try:
        decoded = gzip.decompress(sidecar_path.read_bytes()).decode("utf-8")
        parsed = json.loads(decoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [event for event in parsed if isinstance(event, dict)]
