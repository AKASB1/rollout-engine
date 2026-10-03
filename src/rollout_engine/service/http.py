"""HTTP/1.1 transport for the controller (standard library: asyncio streams), and a client.

JSON bodies, UTF-8, ``Connection: close`` per request. Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import asyncio
import json

REASONS = {
    200: "OK",
    202: "Accepted",
    204: "No Content",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    410: "Gone",
    413: "Payload Too Large",
    429: "Too Many Requests",
    500: "Internal Server Error",
}
MAX_BODY = 16 * 1024 * 1024


async def _handle_conn(
    controller, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        line = await reader.readline()
        if not line:
            return
        try:
            method, target, _ = line.decode("latin-1").rstrip("\r\n").split(" ", 2)
        except ValueError:
            await _respond(writer, 400, {"error": "bad request line"}, {})
            return
        headers = {}
        while True:
            h = await reader.readline()
            if h in (b"\r\n", b"\n", b""):
                break
            k, _, v = h.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", "0") or 0)
        if n > MAX_BODY:
            await _respond(writer, 413, {"error": "body too large"}, {})
            return
        raw = await reader.readexactly(n) if n else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            await _respond(writer, 400, {"error": "body is not UTF-8 JSON"}, {})
            return
        try:
            status, obj, extra = await controller.handle(method, target, body)
        except Exception as e:  # noqa: BLE001 - report, do not crash the server
            status, obj, extra = 500, {"error": repr(e)}, {}
        await _respond(writer, status, obj, extra)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def _respond(writer, status: int, obj, extra: dict) -> None:
    if isinstance(obj, str):
        data = obj.encode("utf-8")
        ctype = extra.get("Content-Type", "text/plain; charset=utf-8")
    elif status == 204:
        data, ctype = b"", "application/json"
    else:
        data = json.dumps(obj, sort_keys=True).encode("utf-8")
        ctype = "application/json"
    head = [
        f"HTTP/1.1 {status} {REASONS.get(status, 'Status')}",
        f"Content-Type: {ctype}",
        f"Content-Length: {len(data)}",
        "Connection: close",
    ]
    for k, v in extra.items():
        if k != "Content-Type":
            head.append(f"{k}: {v}")
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + data)
    await writer.drain()


async def serve(
    controller, host: str = "127.0.0.1", port: int = 18300
) -> asyncio.base_events.Server:
    return await asyncio.start_server(lambda r, w: _handle_conn(controller, r, w), host, port)


class HttpClient:
    """Minimal async JSON client (one connection per request)."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port

    async def _req(self, method: str, path: str, body: dict | None):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        head = f"{method} {path} HTTP/1.1\r\nHost: {self.host}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n"
        writer.write(head.encode("latin-1") + data)
        await writer.drain()
        raw = await reader.read()
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        head, _, payload = raw.partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        headers = {}
        for h in head.split(b"\r\n")[1:]:
            k, _, v = h.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        if headers.get("content-type", "").startswith("application/json") and payload:
            return status, json.loads(payload.decode("utf-8")), headers
        return status, payload.decode("utf-8") if payload else {}, headers

    async def post(self, path: str, body: dict | None = None):
        status, obj, _ = await self._req("POST", path, body or {})
        return status, obj

    async def get(self, path: str):
        status, obj, _ = await self._req("GET", path, None)
        return status, obj
