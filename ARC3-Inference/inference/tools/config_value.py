"""Read a dotted key from a JSON-compatible config file."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _lookup(data: Any, dotted_key: str) -> Any:
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: config_value.py <config_path> <dotted_key> [default]")

    config_path = Path(sys.argv[1])
    dotted_key = sys.argv[2]
    default = sys.argv[3] if len(sys.argv) > 3 else ""

    if not config_path.exists():
        print(default)
        return

    data = json.loads(config_path.read_text(encoding="utf-8"))
    value = _lookup(data, dotted_key)
    if value is None:
        print(default)
        return
    if isinstance(value, bool):
        print("true" if value else "false")
        return
    if isinstance(value, (list, dict)):
        print(json.dumps(value, separators=(",", ":"), ensure_ascii=True))
        return
    print(value)


if __name__ == "__main__":
    main()
