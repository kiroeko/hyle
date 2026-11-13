"""Local HTTP transport and cold route composition for text adapters."""
from __future__ import annotations

import io
import json
import queue
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

from .contracts import TokenEvent, TokenRequest
from .runtime import TextEngine


class ClientDisconnected(ConnectionError):
    pass


def _wait_socket(connection: socket.socket, closing: threading.Event, *, write: bool = False) -> None:
    while not closing.is_set():
        ready, writable, _ = select.select([] if write else [connection], [connection] if write else [], [], .25)
        if ready or writable:
            return
    raise ClientDisconnected('server closing')


class _SocketReader(io.RawIOBase):
    def __init__(self, connection: socket.socket, closing: threading.Event):
        super().__init__()
        self.connection, self.closing = connection, closing

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        while not self.closing.is_set():
            try:
                return self.connection.recv_into(target)
            except (BlockingIOError, InterruptedError):
                _wait_socket(self.connection, self.closing)
        raise ClientDisconnected('server closing')


class _SocketWriter(io.RawIOBase):
    def __init__(self, connection: socket.socket, closing: threading.Event):
        super().__init__()
        self.connection, self.closing = connection, closing

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        remaining = memoryview(data)
        total = len(remaining)
        while remaining:
            if self.closing.is_set():
                raise ClientDisconnected('server closing')
            try:
                sent = self.connection.send(remaining)
            except (BlockingIOError, InterruptedError):
                _wait_socket(self.connection, self.closing, write=True)
                continue
            if not sent:
                raise ClientDisconnected('client disconnected')
            remaining = remaining[sent:]
        return total


class HttpExchange:
    """One HTTP request; the selected adapter owns protocol payloads."""
    MAX_BODY = 1024 * 1024

    def __init__(self, handler: BaseHTTPRequestHandler):
        self._handler = handler
        self._next_peer_check = 0.0

    def read_json(self) -> Any:
        length = int(self._handler.headers.get('Content-Length', '-1'))
        if length <= 0 or length > self.MAX_BODY:
            raise ValueError('invalid or oversized request body')
        raw = self._handler.rfile.read(length)
        if len(raw) != length:
            raise ValueError('incomplete request body')
        return json.loads(raw.decode('utf-8'))

    def send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        handler = self._handler
        handler.send_response(status)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', str(len(data)))
        handler.send_header('Connection', 'close')
        handler.end_headers()
        handler.wfile.write(data)
        handler.close_connection = True

    def begin_stream(self) -> None:
        handler = self._handler
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        handler.send_header('Cache-Control', 'no-cache')
        handler.send_header('Connection', 'close')
        handler.end_headers()
        handler.close_connection = True

    def send_event(self, kind: str, event: Mapping[str, Any]) -> None:
        data = json.dumps(event, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self._handler.wfile.write(b'event: ' + kind.encode('ascii') + b'\ndata: ' + data + b'\n\n')
        self._handler.wfile.flush()

    def send_data(self, event: Mapping[str, Any] | str) -> None:
        data = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False, separators=(',', ':'))
        self._handler.wfile.write(b'data: ' + data.encode('utf-8') + b'\n\n')
        self._handler.wfile.flush()

    def next_event(self, request: TokenRequest) -> TokenEvent:
        while True:
            if self._handler.server.closing.is_set():  # type: ignore[attr-defined]
                self._handler.close_connection = True
                raise ClientDisconnected('server closing')
            now = time.monotonic()
            if now >= self._next_peer_check:
                try:
                    connection = self._handler.connection
                    readable, _, _ = select.select([connection], [], [], 0)
                    # EOF only closes the peer's sending direction. A client
                    # may still be waiting for this response after SHUT_WR.
                    # A reset raises OSError; a graceful full close cannot be
                    # distinguished from a half-close until a write fails.
                    if readable:
                        connection.recv(1, socket.MSG_PEEK)
                    disconnected = False
                except (OSError, ValueError):
                    disconnected = True
                if disconnected:
                    self._handler.close_connection = True
                    raise ClientDisconnected('client disconnected')
                self._next_peer_check = now + .25
            try:
                return request.read(timeout=max(0., self._next_peer_check - time.monotonic()))
            except queue.Empty:
                continue


class TextHttpAdapter(Protocol):
    def handle_post(self, exchange: HttpExchange, engine: TextEngine) -> None: ...


class TextHttpHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'Hyle/0.1'

    def setup(self) -> None:
        super().setup()
        self.rfile.close()
        self.wfile.close()
        self.connection.setblocking(False)
        closing = self.server.closing  # type: ignore[attr-defined]
        self.rfile = io.BufferedReader(_SocketReader(self.connection, closing))
        self.wfile = _SocketWriter(self.connection, closing)

    def do_GET(self) -> None:
        exchange = HttpExchange(self)
        server = self.server
        path = urlsplit(self.path).path
        if path == '/health':
            healthy = server.engine.healthy  # type: ignore[attr-defined]
            exchange.send_json(200 if healthy else 503, {'status': 'ok' if healthy else 'unavailable'})
        elif path == '/metrics' and server.metrics_enabled:  # type: ignore[attr-defined]
            data = server.engine.metrics.render()  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Connection', 'close')
            self.end_headers(); self.wfile.write(data); self.close_connection = True
        else:
            adapter, key = server.resolve_route(path)  # type: ignore[attr-defined]
            if adapter is not None and hasattr(adapter, 'handle_get'):
                adapter.handle_get(exchange, server.engine, key)  # type: ignore[attr-defined]
            else: exchange.send_json(404, {'error': 'endpoint not found'})

    def do_DELETE(self) -> None:
        exchange = HttpExchange(self)
        adapter, key = self.server.resolve_route(urlsplit(self.path).path)  # type: ignore[attr-defined]
        if adapter is not None and hasattr(adapter, 'handle_delete'):
            adapter.handle_delete(exchange, self.server.engine, key)  # type: ignore[attr-defined]
        else: exchange.send_json(404, {'error': 'endpoint not found'})

    def do_POST(self) -> None:
        exchange = HttpExchange(self)
        adapter = self.server.routes.get(urlsplit(self.path).path)  # type: ignore[attr-defined]
        if adapter is None:
            exchange.send_json(404, {'error': 'endpoint not found'})
            return
        adapter.handle_post(exchange, self.server.engine)  # type: ignore[attr-defined]


class TextHttpServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, engine: TextEngine, metrics_enabled: bool, routes: Mapping[str, TextHttpAdapter]):
        self.engine, self.metrics_enabled = engine, metrics_enabled
        self.routes = MappingProxyType(dict(routes))
        self.closing = threading.Event()
        self._connections_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        super().__init__(address, TextHttpHandler)

    def resolve_route(self, path):
        if path in self.routes: return self.routes[path], ''
        for prefix in sorted(self.routes, key=len, reverse=True):
            if prefix.endswith('/*') and path.startswith(prefix[:-1]):
                return self.routes[prefix], path[len(prefix) - 1:]
        return None, ''

    def process_request(self, request, client_address) -> None:
        # Register before starting the handler so close cannot miss a thread
        # that has accepted a socket but has not started executing yet.
        with self._connections_lock:
            if self.closing.is_set():
                self.shutdown_request(request)
                return
            self._connections.add(request)
            try:
                super().process_request(request, client_address)
            except BaseException:
                self._connections.discard(request)
                raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.discard(request)

    def handle_error(self, request, client_address) -> None:
        if self.closing.is_set() and isinstance(sys.exc_info()[1], OSError):
            return
        super().handle_error(request, client_address)

    def server_close(self) -> None:
        self.closing.set()
        self.engine.begin_close()
        with self._connections_lock:
            connections = tuple(self._connections)
        # Wake blocked reads and writes before ThreadingMixIn joins handlers.
        # Their request handles cancel through the original worker owner.
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().server_close()
        self.engine.close()
