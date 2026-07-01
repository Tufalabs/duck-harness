from __future__ import annotations

import argparse
from collections import deque
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from inference.utils.viewer_artifacts import load_raw_events
from inference.utils.run_artifacts import is_selectable_run_dir_name, run_dir_sort_key, save_git_info
from viewer.data import _split_labeled_sections

MISSING_LEVEL_STATUS = "MISSING"
COMPLETED_LEVEL_STATUS = "COMPLETED"
PER_LEVEL_SCORE_CAP = 115.0
GAME_SCORE_CAP = 100.0


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    if not values or not weights:
        return 0.0
    if len(values) != len(weights):
        raise ValueError("Weighted mean requires the same number of values and weights.")
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight


def compute_level_score(*, baseline_actions: int, agent_steps: int | None, agent_status: str) -> float:
    if (
        agent_status != COMPLETED_LEVEL_STATUS
        or agent_steps is None
        or agent_steps <= 0
        or baseline_actions <= 0
    ):
        return 0.0
    return min(float((baseline_actions / agent_steps) ** 2) * 100.0, PER_LEVEL_SCORE_CAP)


def compute_game_score(level_scores: list[tuple[int, float]]) -> float:
    raw_score = _weighted_mean(
        [score for _level, score in level_scores],
        [level for level, _score in level_scores],
    )
    total_weight = sum(level for level, _score in level_scores)
    if total_weight <= 0:
        return 0.0
    completed_weight = sum(level for level, score in level_scores if score > 0.0)
    max_score = completed_weight / total_weight * GAME_SCORE_CAP
    return min(raw_score, max_score)


@dataclass(frozen=True)
class GameTraceExport:
    game_id: str
    output_path: Path
    analysis_event_count: int
    message_count: int


@dataclass(frozen=True)
class RunTraceExport:
    run_name: str
    games: list[GameTraceExport]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export machine-readable per-episode analyzer traces from saved run artifacts.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Specific run directory name to export. Repeat to export multiple runs.",
    )
    return parser.parse_args()


def _list_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        [path for path in runs_dir.iterdir() if path.is_dir() and is_selectable_run_dir_name(path.name)],
        key=run_dir_sort_key,
    )


def _resolve_run_dirs(runs_dir: Path, selected_runs: list[str]) -> list[Path]:
    all_runs = _list_run_dirs(runs_dir)
    if not selected_runs:
        return all_runs

    by_name = {path.name: path for path in all_runs}
    missing = [name for name in selected_runs if name not in by_name]
    if missing:
        available = ", ".join(path.name for path in all_runs) or "(none)"
        raise FileNotFoundError(
            f"Unknown run directory name(s): {', '.join(missing)}. Available runs: {available}"
        )
    return [by_name[name] for name in selected_runs]


def _analysis_events(payload: dict[str, Any], *, viewer_data_path: Path | None = None) -> list[dict[str, Any]]:
    raw_events = load_raw_events(payload, viewer_data_path=viewer_data_path)

    analysis_events: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        if str(raw_event.get("type") or "").strip() != "analysis":
            continue
        transcript = str(raw_event.get("transcript") or "").strip()
        if not transcript:
            continue
        try:
            analysis_step = int(raw_event.get("analysis_step"))
        except (TypeError, ValueError):
            analysis_step = None
        try:
            action_num = int(raw_event.get("action_num"))
        except (TypeError, ValueError):
            action_num = None
        analysis_event = {
            "analysis_step": analysis_step,
            "action_num": action_num,
            "event_index": raw_event.get("event_index"),
            "title": raw_event.get("title"),
            "status": raw_event.get("status"),
            "transcript": transcript,
        }
        board = raw_event.get("board")
        if isinstance(board, list):
            analysis_event["grid"] = board
        analysis_events.append(analysis_event)
    return analysis_events


def _normalize_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_level_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw_levels = payload.get("level_summaries")
    if not isinstance(raw_levels, list):
        return {}

    level_map: dict[int, dict[str, Any]] = {}
    for raw_level in raw_levels:
        if not isinstance(raw_level, dict):
            continue
        level_num = _normalize_int(raw_level.get("level"))
        if level_num is None or level_num <= 0:
            continue
        raw_actions = _normalize_int(raw_level.get("actions"))
        status = str(raw_level.get("status") or "UNKNOWN").strip() or "UNKNOWN"
        level_summary: dict[str, Any] = {
            "actions": None if raw_actions is None else max(0, raw_actions),
            "status": status,
        }
        if "baseline_actions" in raw_level:
            baseline_actions = _normalize_int(raw_level.get("baseline_actions"))
            if baseline_actions is None:
                raise ValueError(f"level {level_num}: invalid baseline_actions {raw_level.get('baseline_actions')!r}.")
            if baseline_actions <= 0:
                raise ValueError(f"level {level_num}: baseline_actions must be positive, got {baseline_actions}.")
            level_summary["baseline_actions"] = baseline_actions
        level_map[level_num] = level_summary
    return level_map


def _derive_level_transitions(
    payload: dict[str, Any],
    *,
    viewer_data_path: Path | None = None,
    level_annotations: dict[int, dict[str, Any]],
    total_levels: int | None,
    analysis_message_links: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_events = load_raw_events(payload, viewer_data_path=viewer_data_path)
    if not raw_events:
        return []

    initial_score = 0
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        if str(raw_event.get("type") or "").strip() != "initial":
            continue
        initial_score = max(0, _normalize_int(raw_event.get("score")) or 0)
        break

    previous_score = initial_score
    transitions: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        if str(raw_event.get("type") or "").strip() != "action":
            continue

        score_after = _normalize_int(raw_event.get("score"))
        if score_after is None or score_after <= previous_score:
            continue

        action_num = _normalize_int(raw_event.get("action_num"))
        analysis_step = _normalize_int(raw_event.get("analysis_step"))
        state_after = str(raw_event.get("state") or "").strip() or None
        action_display = str(raw_event.get("action_display") or "").strip() or None

        for completed_level in range(previous_score + 1, score_after + 1):
            next_level = completed_level + 1
            if total_levels is not None and completed_level >= total_levels:
                next_level = None
            if state_after == "WIN":
                next_level = None

            transition: dict[str, Any] = {
                "completed_level": completed_level,
                "next_level": next_level,
                "action_num": action_num,
                "analysis_step": analysis_step,
                "score_after_action": score_after,
                "state_after_action": state_after,
            }
            if action_display:
                transition["action"] = action_display

            level_summary = level_annotations.get(completed_level)
            if level_summary is not None:
                transition["actions"] = level_summary.get("actions")
                transition["status"] = level_summary.get("status")
                if "baseline_actions" in level_summary:
                    transition["baseline_actions"] = level_summary.get("baseline_actions")
                if "score" in level_summary:
                    transition["level_score"] = level_summary.get("score")

            if analysis_step is not None:
                message_link = analysis_message_links.get(analysis_step)
                if message_link is not None:
                    transition["message_index"] = message_link.get("message_index")
                    transition["message_indices"] = list(message_link.get("message_indices") or [])
                    transition["message_role"] = message_link.get("message_role")

            transitions.append(transition)

        previous_score = score_after

    return transitions


def _validate_level_map_baselines(
    level_map: dict[int, dict[str, Any]],
    *,
    baseline_actions_by_level: list[int],
) -> None:
    for level_num, raw_level in level_map.items():
        if "baseline_actions" not in raw_level or level_num > len(baseline_actions_by_level):
            continue
        expected = int(baseline_actions_by_level[level_num - 1])
        actual = int(raw_level["baseline_actions"])
        if actual != expected:
            raise ValueError(
                f"level {level_num}: artifact baseline_actions={actual} "
                f"does not match expected baseline_actions={expected}."
            )


def _artifact_baseline_actions_by_level(payload: dict[str, Any]) -> list[int] | None:
    """Per-level ``baseline_actions`` recorded in the run artifact.

    The harness persists these per level, so the artifact is a self-contained
    source. Returns ``None`` when any level is missing a positive value, which
    disables per-level re-scoring.
    """
    level_map = _extract_level_map(payload)
    if not level_map:
        return None
    total_levels = max(level_map, default=0)
    baselines: list[int] = []
    for level_num in range(1, total_levels + 1):
        raw_level = level_map.get(level_num) or {}
        value = _normalize_int(raw_level.get("baseline_actions"))
        if value is None or value <= 0:
            return None
        baselines.append(value)
    return baselines


def _build_scored_level_summaries(
    payload: dict[str, Any],
    *,
    baseline_actions_by_level: list[int] | None,
) -> tuple[list[dict[str, Any]], float | None, int | None]:
    level_map = _extract_level_map(payload)
    if not level_map:
        return [], None, None

    if baseline_actions_by_level:
        _validate_level_map_baselines(level_map, baseline_actions_by_level=baseline_actions_by_level)

    if not baseline_actions_by_level:
        raw_levels = []
        for level_num in sorted(level_map):
            raw_level = level_map[level_num]
            raw_levels.append(
                {
                    "level": level_num,
                    "actions": raw_level.get("actions"),
                    "status": raw_level.get("status"),
                }
            )
        return raw_levels, None, None

    total_levels = max(max(level_map, default=0), len(baseline_actions_by_level))
    level_summaries: list[dict[str, Any]] = []
    level_scores: list[tuple[int, float]] = []
    for level_num in range(1, total_levels + 1):
        raw_level = level_map.get(level_num)
        actions = raw_level.get("actions") if raw_level is not None else None
        status = str(raw_level.get("status") or MISSING_LEVEL_STATUS) if raw_level is not None else MISSING_LEVEL_STATUS

        summary: dict[str, Any] = {
            "level": level_num,
            "actions": actions,
            "status": status,
        }

        if level_num <= len(baseline_actions_by_level):
            baseline_actions = int(baseline_actions_by_level[level_num - 1])
            score = compute_level_score(
                baseline_actions=baseline_actions,
                agent_steps=actions,
                agent_status=status,
            )
            summary["baseline_actions"] = baseline_actions
            summary["score"] = score
            level_scores.append((level_num, score))

        level_summaries.append(summary)

    return level_summaries, compute_game_score(level_scores), len(baseline_actions_by_level)


def _split_transcript_header(transcript: str) -> tuple[str | None, str]:
    lines = transcript.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().startswith("--- ") and lines[0].strip().endswith(" ---"):
        return lines[0].strip(), "\n".join(lines[1:])
    return None, transcript


def _normalize_sections(transcript: str) -> list[dict[str, str]]:
    _, transcript_body = _split_transcript_header(transcript)
    sections = _split_labeled_sections(transcript_body)
    if not sections:
        stripped_transcript = transcript_body.strip()
        if not stripped_transcript:
            return []
        return [{"label": "TRANSCRIPT", "kind": "tool", "content": stripped_transcript}]
    normalized_sections: list[dict[str, str]] = []
    for section in sections:
        content = str(section.get("content") or "")
        stripped_content = content.strip()
        if not stripped_content or stripped_content == "(empty)":
            continue
        normalized_sections.append(
            {
                "label": str(section.get("label") or "").strip(),
                "kind": str(section.get("kind") or "tool").strip(),
                "content": content,
            }
        )
    return normalized_sections


def _tool_name_from_label(label: str, prefix: str) -> str:
    if label.startswith(prefix):
        name = label[len(prefix) :].strip()
        if name:
            return name
    return "unknown"


def _tool_arguments_string(tool_name: str, content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return "{}"
    if stripped.startswith("<tool_call>"):
        parameter_matches = re.findall(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            stripped,
            flags=re.DOTALL,
        )
        if parameter_matches:
            return json.dumps({name: value for name, value in parameter_matches}, ensure_ascii=True)
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        if tool_name == "python":
            return json.dumps({"code": stripped}, ensure_ascii=True)
        return json.dumps({"raw_content": stripped}, ensure_ascii=True)
    if tool_name == "python" and not isinstance(parsed, dict):
        return json.dumps({"code": stripped}, ensure_ascii=True)
    return json.dumps(parsed, ensure_ascii=True)


def _tool_result_content(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return "{}"
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return json.dumps({"stdout": stripped}, ensure_ascii=True, separators=(",", ":"))
    if not isinstance(parsed, (dict, list)):
        return json.dumps({"stdout": stripped}, ensure_ascii=True, separators=(",", ":"))
    return json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))


def _collect_tool_calls(
    sections: list[dict[str, str]],
    start_index: int,
    *,
    next_call_counter: int,
) -> tuple[list[dict[str, Any]], deque[str], int, int]:
    tool_calls: list[dict[str, Any]] = []
    pending_tool_call_ids: deque[str] = deque()
    index = start_index
    call_counter = next_call_counter

    while index < len(sections) and sections[index]["label"].startswith("TOOL CALL: "):
        tool_name = _tool_name_from_label(sections[index]["label"], "TOOL CALL: ")
        call_counter += 1
        tool_call_id = f"call_{call_counter:05d}"
        pending_tool_call_ids.append(tool_call_id)
        tool_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": _tool_arguments_string(tool_name, sections[index]["content"]),
                },
            }
        )
        index += 1

    return tool_calls, pending_tool_call_ids, index, call_counter


def _joined_reasoning(reasoning_parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in reasoning_parts if part.strip()).strip()


def _append_message(messages: list[dict[str, Any]], message: dict[str, Any]) -> int:
    messages.append(message)
    return len(messages) - 1


def _flush_reasoning_only_message(messages: list[dict[str, Any]], reasoning_parts: list[str]) -> int | None:
    reasoning = _joined_reasoning(reasoning_parts)
    reasoning_parts.clear()
    if reasoning:
        return _append_message(messages, {"role": "assistant", "reasoning": reasoning})
    return None


def _messages_from_sections(
    analysis_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    analysis_message_links: dict[int, dict[str, Any]] = {}
    tool_call_counter = 0
    system_prompt_emitted = False

    for event in analysis_events:
        analysis_step = _normalize_int(event.get("analysis_step"))
        event_message_indices: list[int] = []
        action_message_index: int | None = None

        def _record_message(index: int | None, *, action_message: bool = False) -> None:
            nonlocal action_message_index
            if index is None:
                return
            event_message_indices.append(index)
            if action_message:
                action_message_index = index

        normalized_sections = _normalize_sections(str(event["transcript"]))
        pending_tool_call_ids: deque[str] = deque()
        pending_reasoning_parts: list[str] = []
        grid = event.get("grid")
        grid_attached = False
        index = 0

        while index < len(normalized_sections):
            section = normalized_sections[index]
            label = section["label"]
            content = section["content"]

            if label == "SYSTEM PROMPT":
                _record_message(_flush_reasoning_only_message(messages, pending_reasoning_parts))
                if not system_prompt_emitted:
                    _record_message(_append_message(messages, {"role": "system", "content": content}))
                    system_prompt_emitted = True
                index += 1
                continue

            if label == "USER PROMPT":
                _record_message(_flush_reasoning_only_message(messages, pending_reasoning_parts))
                user_message: dict[str, Any] = {"role": "user", "content": content}
                if isinstance(grid, list) and not grid_attached:
                    user_message["grid"] = grid
                    grid_attached = True
                _record_message(_append_message(messages, user_message))
                index += 1
                continue

            if label == "THINKING":
                pending_reasoning_parts.append(content)
                index += 1
                continue

            if label == "ASSISTANT":
                assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
                reasoning = _joined_reasoning(pending_reasoning_parts)
                pending_reasoning_parts.clear()
                if reasoning:
                    assistant_message["reasoning"] = reasoning
                tool_calls, new_pending_ids, next_index, tool_call_counter = _collect_tool_calls(
                    normalized_sections,
                    index + 1,
                    next_call_counter=tool_call_counter,
                )
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                    pending_tool_call_ids.extend(new_pending_ids)
                    index = next_index
                else:
                    index += 1
                _record_message(
                    _append_message(messages, assistant_message),
                    action_message=bool(tool_calls),
                )
                continue

            if label.startswith("TOOL CALL: "):
                tool_calls, new_pending_ids, next_index, tool_call_counter = _collect_tool_calls(
                    normalized_sections,
                    index,
                    next_call_counter=tool_call_counter,
                )
                pending_tool_call_ids.extend(new_pending_ids)
                assistant_message: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
                reasoning = _joined_reasoning(pending_reasoning_parts)
                pending_reasoning_parts.clear()
                if reasoning:
                    assistant_message["reasoning"] = reasoning
                _record_message(_append_message(messages, assistant_message), action_message=True)
                index = next_index
                continue

            if label.startswith("TOOL RESULT: "):
                _record_message(_flush_reasoning_only_message(messages, pending_reasoning_parts))
                tool_result: dict[str, Any] = {"role": "tool", "content": _tool_result_content(content)}
                if pending_tool_call_ids:
                    tool_result["tool_call_id"] = pending_tool_call_ids.popleft()
                else:
                    tool_call_counter += 1
                    tool_result["tool_call_id"] = f"call_{tool_call_counter:05d}"
                _record_message(_append_message(messages, tool_result))
                index += 1
                continue

            # Transcript-only sections like analyzer status are not part of live model history.
            index += 1

        _record_message(_flush_reasoning_only_message(messages, pending_reasoning_parts))

        if analysis_step is not None and event_message_indices:
            linked_message_index = action_message_index if action_message_index is not None else event_message_indices[-1]
            analysis_message_links[analysis_step] = {
                "message_index": linked_message_index,
                "message_indices": list(event_message_indices),
                "message_role": str(messages[linked_message_index].get("role") or ""),
            }

    return messages, analysis_message_links


def export_game_traces(
    *,
    viewer_data_path: Path,
    run_name: str,
    output_dir: Path,
    environments_dir: str | None = None,
) -> GameTraceExport | None:
    payload = json.loads(viewer_data_path.read_text(encoding="utf-8"))
    game_id = str(payload.get("game_id") or "").strip()
    if not game_id:
        raise ValueError(f"{viewer_data_path}: missing game_id.")

    analysis_events = _analysis_events(payload, viewer_data_path=viewer_data_path)
    if not analysis_events:
        return None

    messages, analysis_message_links = _messages_from_sections(analysis_events)
    if not messages:
        return None

    output_path = output_dir / run_name / f"{game_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_actions_by_level = _artifact_baseline_actions_by_level(payload)
    level_summaries, game_score, max_game_level = _build_scored_level_summaries(
        payload,
        baseline_actions_by_level=baseline_actions_by_level,
    )
    level_annotations = {
        int(summary["level"]): summary
        for summary in level_summaries
        if _normalize_int(summary.get("level")) is not None
    }
    level_transitions = _derive_level_transitions(
        payload,
        viewer_data_path=viewer_data_path,
        level_annotations=level_annotations,
        total_levels=max_game_level,
        analysis_message_links=analysis_message_links,
    )
    trace_payload: dict[str, Any] = {
        "id": f"{run_name}/{game_id}",
        "messages": messages,
    }
    if max_game_level is not None:
        trace_payload["max_game_level"] = max_game_level
    if level_summaries:
        trace_payload["level_summaries"] = level_summaries
    if game_score is not None:
        trace_payload["game_score"] = game_score
    if level_transitions:
        trace_payload["level_transitions"] = level_transitions
    output_path.write_text(
        json.dumps(trace_payload, indent=2),
        encoding="utf-8",
    )
    return GameTraceExport(
        game_id=game_id,
        output_path=output_path,
        analysis_event_count=len(analysis_events),
        message_count=len(messages),
    )


def export_run_traces(
    *,
    run_dir: Path,
    output_dir: Path,
    environments_dir: str | None = None,
) -> RunTraceExport:
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return RunTraceExport(run_name=run_dir.name, games=[])

    run_output_dir = output_dir / run_dir.name
    if run_output_dir.exists():
        shutil.rmtree(run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    source_git_info_path = run_dir / "git_info.txt"
    if source_git_info_path.exists():
        shutil.copy2(source_git_info_path, run_output_dir / source_git_info_path.name)
    else:
        save_git_info(run_output_dir)

    exported_games: list[GameTraceExport] = []
    for viewer_data_path in sorted(artifacts_dir.glob("*viewer_data.json")):
        exported = export_game_traces(
            viewer_data_path=viewer_data_path,
            run_name=run_dir.name,
            output_dir=output_dir,
            environments_dir=environments_dir,
        )
        if exported is not None:
            exported_games.append(exported)

    return RunTraceExport(run_name=run_dir.name, games=exported_games)


def export_traces(
    *,
    runs_dir: Path,
    output_dir: Path,
    selected_runs: list[str],
    environments_dir: str | None = None,
) -> list[RunTraceExport]:
    run_dirs = _resolve_run_dirs(runs_dir, selected_runs)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        export_run_traces(run_dir=run_dir, output_dir=output_dir, environments_dir=environments_dir)
        for run_dir in run_dirs
    ]


def main() -> int:
    args = _parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir / "traces"

    try:
        exports = export_traces(
            runs_dir=runs_dir,
            output_dir=output_dir,
            selected_runs=list(args.run),
            environments_dir=(str(args.environments_dir) if args.environments_dir else None),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_games = 0
    total_events = 0
    total_messages = 0
    for export in exports:
        print(f"{export.run_name}:")
        if not export.games:
            print("  no transcript-bearing games found")
            continue
        for game in export.games:
            total_games += 1
            total_events += game.analysis_event_count
            total_messages += game.message_count
            print(
                f"  {game.game_id}: wrote {game.output_path.name} "
                f"({game.analysis_event_count} analysis turns, {game.message_count} messages)"
            )
    print(
        f"traces written to {output_dir} "
        f"({total_games} episodes, {total_events} analysis turns, {total_messages} messages)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
