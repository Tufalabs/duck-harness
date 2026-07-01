"""Generic support utilities for TAAF."""

import json
import os
import pickle
from pathlib import Path
from typing import Any


def atomic_json_dump(obj: Any, path: Path) -> None:
    """Write ``obj`` as JSON to ``path`` via tempfile + os.replace.

    A crash mid-write leaves either the previous file or nothing — never a
    partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    """Write ``obj`` as a pickle to ``path`` via tempfile + os.replace.

    Same crash-safety guarantee as ``atomic_json_dump``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
