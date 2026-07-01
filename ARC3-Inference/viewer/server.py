"""Simple HTTP server for browsing the latest ARC3 run."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from inference.utils.run_artifacts import is_selectable_run_dir_name
from viewer.data import load_game_payload, load_game_shell_payload, load_game_step_payload, load_run_summary


log = logging.getLogger(__name__)
_GZIP_MIN_BYTES = 1024


@dataclass(frozen=True)
class _ResponseBody:
    body: bytes
    is_gzipped: bool


def _index_html_path() -> Path:
    return Path(__file__).resolve().parent / "index.html"

def _index_html_path() -> Path:
    return Path(__file__).resolve().parent / "index.html"


def _load_index_html() -> str:
    return _index_html_path().read_text(encoding="utf-8")


def _index_html_version() -> int:
    return _index_html_path().stat().st_mtime_ns


def _requested_run_dir(*, runs_dir: Path, default_run_dir: Path | None, requested_run: str | None) -> Path | None:
    requested_name = str(requested_run or "").strip()
    if not requested_name:
        return default_run_dir

    candidate = Path(requested_name)
    if candidate.is_absolute():
        return candidate

    if runs_dir.is_dir() and is_selectable_run_dir_name(runs_dir.name) and requested_name == runs_dir.name:
        return runs_dir

    if default_run_dir is not None and requested_name == default_run_dir.name:
        return default_run_dir

    return runs_dir / requested_name


class _ViewerHandler(BaseHTTPRequestHandler):
    """Serve the viewer shell and run payload API."""

    runs_dir: Path
    run_dir: Path | None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(_load_index_html())
            return
        if parsed.path == "/api/viewer-version":
            self._send_json({"version": _index_html_version()})
            return
        if parsed.path == "/api/run":
            self._handle_run_api(parsed.query)
            return
        if parsed.path == "/api/game":
            self._handle_game_api(parsed.query)
            return
        if parsed.path == "/api/game-step":
            self._handle_game_step_api(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _handle_run_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        try:
            payload = load_run_summary(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc), "games": []}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}", "games": []}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _handle_game_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        raw_index = params.get("index", [None])[0]
        try:
            game_index = int(str(raw_index))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game index."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            full_payload = params.get("full", ["false"])[0].lower() in {"1", "true", "yes", "on"}
            loader = load_game_payload if full_payload else load_game_shell_payload
            payload = loader(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
                game_index=game_index,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _handle_game_step_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        raw_game_index = params.get("index", [None])[0]
        raw_step_index = params.get("step", [None])[0]
        try:
            game_index = int(str(raw_game_index))
            step_index = int(str(raw_step_index))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game/step index."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = load_game_step_payload(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
                game_index=game_index,
                step_index=step_index,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _send_html(self, html: str) -> None:
        content = html.encode("utf-8")
        content = self._maybe_gzip(content)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        content = self._maybe_gzip(content)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def _maybe_gzip(self, content: bytes) -> "_ResponseBody":
        accept_encoding = self.headers.get("Accept-Encoding", "")
        if len(content) < _GZIP_MIN_BYTES or "gzip" not in accept_encoding.lower():
            return _ResponseBody(body=content, is_gzipped=False)
        return _ResponseBody(body=gzip.compress(content), is_gzipped=True)


def build_handler(*, runs_dir: Path, run_dir: Path | None) -> type[_ViewerHandler]:
    """Bind configuration into the request handler class."""
    handler_cls = type("ViewerHandler", (_ViewerHandler,), {})
    handler_cls.runs_dir = runs_dir
    handler_cls.run_dir = run_dir
    return handler_cls


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ARC3 viewer for the latest run.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None, help="Optional explicit run directory to view.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    runs_dir = Path(args.runs_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None
    handler = build_handler(runs_dir=runs_dir, run_dir=run_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    target = run_dir if run_dir is not None else runs_dir
    log.info("Viewer serving %s at http://%s:%d", target, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Viewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
