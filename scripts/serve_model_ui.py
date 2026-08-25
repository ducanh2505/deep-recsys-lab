"""Serve the static model UI and proxy its requests to BentoML."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ModelUIHandler(BaseHTTPRequestHandler):
    """One-page static server with a narrow same-origin API proxy."""

    ui_file: Path
    backend_url: str

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/", "/api.html"}:
            self._serve_ui(head_only=False)
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if path.startswith("/api/"):
            self._proxy("GET")
            return
        self._json_error(HTTPStatus.NOT_FOUND, "route not found")

    def do_HEAD(self) -> None:  # noqa: N802
        if urlsplit(self.path).path in {"/", "/api.html"}:
            self._serve_ui(head_only=True)
            return
        self._json_error(HTTPStatus.NOT_FOUND, "route not found", head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.startswith("/api/"):
            self._proxy("POST")
            return
        self._json_error(HTTPStatus.NOT_FOUND, "route not found")

    def _serve_ui(self, *, head_only: bool) -> None:
        payload = self.ui_file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def _proxy(self, method: str) -> None:
        upstream_path = self.path[len("/api") :]
        target = f"{self.backend_url}{upstream_path}"
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() in {"accept", "content-type", "x-api-key", "x-request-id"}
        }
        request = Request(target, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                payload = response.read()
                self._relay(
                    response.status,
                    response.headers.get("Content-Type"),
                    payload,
                    response.headers.get("X-Request-ID"),
                )
        except HTTPError as error:
            payload = error.read()
            self._relay(
                error.code,
                error.headers.get("Content-Type"),
                payload,
                error.headers.get("X-Request-ID"),
            )
        except URLError as error:
            self._json_error(
                HTTPStatus.BAD_GATEWAY,
                f"recommendation API is unavailable: {error.reason}",
            )

    def _relay(
        self,
        status: int,
        content_type: str | None,
        payload: bytes,
        request_id: str | None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(payload)

    def _json_error(self, status: HTTPStatus, detail: str, *, head_only: bool = False) -> None:
        payload = json.dumps({"detail": detail}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-url", default="http://127.0.0.1:3000")
    parser.add_argument(
        "--ui-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "api.html",
    )
    args = parser.parse_args()
    ui_file = args.ui_file.resolve()
    if not ui_file.is_file():
        parser.error(f"UI file does not exist: {ui_file}")

    backend_url = args.backend_url.rstrip("/")
    handler = type(
        "ConfiguredModelUIHandler",
        (ModelUIHandler,),
        {"ui_file": ui_file, "backend_url": backend_url},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Model serving UI: http://{args.host}:{args.port}")
    print(f"Proxying API requests to: {backend_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
