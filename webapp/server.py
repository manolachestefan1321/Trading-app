"""Small backend for the chart-photo web app (Python standard library HTTP server).

    ANTHROPIC_API_KEY=sk-ant-... python -m webapp.server --port 8000

Serves the static PWA from webapp/static and one API endpoint:

    POST /api/analyze   {"image": "<base64>", "media_type": "image/jpeg"}
      200 -> {"ok": true, "result": {...}}
      4xx/5xx -> {"ok": false, "error": "...", "code": "..."}   (never a signal)

The API key stays on the server (environment variable / .env); the browser never sees it.
Optional protections for a server reachable from the internet:
    APP_ACCESS_TOKEN      clients must send it in the X-Access-Token header
    RATE_LIMIT_PER_HOUR   analyses per client IP per hour (default 30)
    TLS_CERT / TLS_KEY    serve HTTPS directly (needed for PWA install on a phone)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import logging
import mimetypes
import os
import ssl
import threading
import time
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from webapp.vision import AnalysisError, analyze_chart

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 8 * 1024 * 1024  # base64 of a 5 MB image plus JSON overhead
log = logging.getLogger("fxpa.webapp")

mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("application/javascript", ".js")


def _configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class RateLimiter:
    """Sliding one-hour window per client key."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.per_hour <= 0:
            return True
        now = time.time() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.per_hour:
                return False
            q.append(now)
            return True


class Handler(BaseHTTPRequestHandler):
    server_version = "fxpa-webapp"
    analyzer = staticmethod(analyze_chart)   # swapped out in tests
    limiter = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_HOUR", "30")))
    access_token = os.environ.get("APP_ACCESS_TOKEN", "")

    # ------------------------------------------------------------------ helpers
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, code: str) -> None:
        self._send_json(status, {"ok": False, "error": message, "code": code})

    def _client_ip(self) -> str:
        return self.client_address[0]

    def log_message(self, fmt: str, *args) -> None:  # route access logs through logging
        log.info("%s %s", self._client_ip(), fmt % args)

    # ------------------------------------------------------------------ static
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path == "/":
            path = "/index.html"
        if path == "/api/health":
            self._send_json(200, {"ok": True, "configured": _configured(),
                                  "token_required": bool(self.access_token)})
            return
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # The service worker handles offline caching; keep HTTP caching short so updates show up.
        self.send_header("Cache-Control", "no-cache")
        if target.name == "sw.js":
            self.send_header("Service-Worker-Allowed", "/")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ API
    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/analyze":
            self._error(404, "Not found.", "not_found")
            return
        if self.access_token and not hmac.compare_digest(
                self.headers.get("X-Access-Token", ""), self.access_token):
            self._error(401, "This app needs an access code.", "token_required")
            return
        if not self.limiter.allow(self._client_ip()):
            self._error(429, "Too many analyses in the last hour. Please wait and try again.", "rate_limited")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._error(413 if length > MAX_BODY else 400, "Missing or oversized upload.", "bad_request")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            image = base64.b64decode(payload["image"], validate=True)
            media_type = str(payload.get("media_type", "image/jpeg"))
        except (json.JSONDecodeError, KeyError, TypeError, binascii.Error, ValueError):
            self._error(400, "The upload was not a valid image.", "bad_request")
            return
        if not _configured():
            self._error(503, "The analysis service is not configured (no API key on the server).", "config")
            return
        try:
            result = self.analyzer(image, media_type)
        except AnalysisError as exc:
            self._error(exc.status, exc.message, exc.code)
            return
        except Exception:  # never leak internals, never return a signal on failure
            log.exception("unexpected analysis failure")
            self._error(500, "Something went wrong during analysis. No signal is shown.", "internal")
            return
        self._send_json(200, {"ok": True, "result": result.to_dict()})


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    p = argparse.ArgumentParser(description="Chart photo analysis web app")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:  # load ANTHROPIC_API_KEY etc. from the project .env
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    except ImportError:
        pass
    Handler.access_token = os.environ.get("APP_ACCESS_TOKEN", "")
    Handler.limiter = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_HOUR", "30")))
    srv = make_server(a.host, a.port)
    scheme = "http"
    cert, key = os.environ.get("TLS_CERT"), os.environ.get("TLS_KEY")
    if cert and key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    if not _configured():
        log.warning("ANTHROPIC_API_KEY is not set: every analysis will return an error (no signals)")
    log.info("serving on %s://%s:%d", scheme, a.host, a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
