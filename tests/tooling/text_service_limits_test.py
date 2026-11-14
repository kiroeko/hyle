"""Bounded ingress and IO must never substitute for native retirement."""
from dataclasses import replace
import http.client
import importlib.util
import io
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.text.app import TextApiServer, parse_arguments
from engine.text.configuration import PreparedChoices, Selection, ServiceLimits, SessionPolicy, read_configuration
from engine.text.contracts import WorkerError
from engine.text.runtime import TextEngine
from engine.text.server import _SocketReader
from engine.text.sessions import TextSessions
from tests.tooling.text_frontend_test import FakeCodec, FakeWorker
from tests.tooling.text_sessions_test import Native, Worker


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline: raise AssertionError('bounded operation did not finish')
        threading.Event().wait(.005)


class LimitsConfigurationTest(unittest.TestCase):
    def test_defaults_and_toml_are_validated_before_starting_a_server(self):
        args = parse_arguments(['serve', '--engine', 'hyle', '--inputs', 'inputs', '--tokenizer-dir', 'tokenizer'])
        self.assertEqual(read_configuration(args).service, ServiceLimits())
        def config(text):
            args.config = SimpleNamespace(read_text=lambda **_: text, resolve=lambda: Path('service.toml').resolve())
            return read_configuration(args)
        value = config('[service]\nmax_connections=5\nread_idle_seconds=2\nwrite_idle_seconds=3\n'
                       '[sessions]\nqueue_timeout_seconds=4\n')
        self.assertEqual(value.service, ServiceLimits(5, 2, 3))
        self.assertEqual(value.sessions.queue_timeout_seconds, 4)
        for text in ('[service]\nunknown=1', '[service]\nmax_connections=0', '[service]\nmax_connections=true',
                     '[service]\nread_idle_seconds=nan', '[service]\nwrite_idle_seconds=0',
                     '[sessions]\nqueue_timeout_seconds=inf', '[sessions]\nqueue_timeout_seconds=false'):
            with self.subTest(text=text), self.assertRaises(ValueError): config(text)


class QueueLimitsTest(unittest.TestCase):
    def test_deadline_survives_spurious_wakes_and_full_queue_has_no_native_admission(self):
        worker = Worker()
        owner = TextSessions(worker, SessionPolicy(kv_capacity=1, max_waiting=1, queue_timeout_seconds=.15))
        first = owner.start('first', [], [1], 1, Selection())
        errors, finished = [], threading.Event()
        def pending():
            try: owner.start('queued', [], [1], 1, Selection())
            except Exception as error: errors.append(error)
            finally: finished.set()
        thread = threading.Thread(target=pending)
        started = time.monotonic(); thread.start()
        try:
            wait_until(lambda: owner.snapshot()['waiting'] == 1)
            self.assertEqual(owner.snapshot(), {'waiting': 1, 'active': 1, 'resident': 1})
            with self.assertRaisesRegex(WorkerError, 'waiting capacity'): owner.start('overflow', [], [1], 1, Selection())
            while not finished.wait(.01) and time.monotonic() - started < 1:
                with owner._condition: owner._condition.notify_all()
            self.assertTrue(finished.wait(.5)); thread.join(timeout=1)
            self.assertGreaterEqual(time.monotonic() - started, .12)
            self.assertLess(time.monotonic() - started, .8)
            self.assertEqual(len(errors), 1); self.assertEqual(errors[0].kind, 'busy')
            self.assertIn('deadline', str(errors[0])); self.assertEqual(len(worker.requests), 1)
            self.assertEqual(owner.snapshot(), {'waiting': 0, 'active': 1, 'resident': 1})
            first.close()
            self.assertEqual(owner.snapshot(), {'waiting': 0, 'active': 0, 'resident': 0})
            owner.start('recovered', [], [1], 1, Selection()).close()
        finally:
            owner.begin_close(); thread.join(timeout=2); first.close(); owner.close()

    def test_failed_retirement_retains_residency_after_the_delivery_owner_leaves(self):
        worker = Worker(); owner = TextSessions(worker, SessionPolicy(kv_capacity=1))
        turn = owner.start('first', [], [1], 1, Selection())
        worker.retirement_fails = True
        with self.assertRaises(WorkerError): turn.close()
        self.assertEqual(owner.snapshot(), {'waiting': 0, 'active': 0, 'resident': 1})
        self.assertFalse(owner.healthy)
        with self.assertRaises(WorkerError): owner.close()


class ServiceHttpTest(unittest.TestCase):
    def serve(self, *, limits=ServiceLimits(), engine=None):
        engine = engine or TextEngine(FakeWorker(), FakeCodec(), 'hyle')
        engine.choices = replace(engine.choices, service=limits)
        # A deliberately small OS send buffer makes the real blocked-write
        # branch deterministic without adding any product debug switch.
        class Server(TextApiServer):
            def process_request(self, request, address):
                request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
                super().process_request(request, address)
        server = Server(('127.0.0.1', 0), engine, True)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}); thread.start()
        def stop():
            server.shutdown(); thread.join(timeout=3); server.server_close()
        self.addCleanup(stop)
        return server

    def request(self, server, path='/health', data=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        try:
            connection.request('GET' if data is None else 'POST', path,
                               body=None if data is None else json.dumps(data), headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally: connection.close()

    def test_connection_limit_rejects_before_creating_another_handler_and_recovers(self):
        server = self.serve(limits=ServiceLimits(max_connections=1))
        held = socket.create_connection(server.server_address, timeout=2)
        try:
            held.sendall(b'GET /health HTTP/1.1\r\nHost:')
            wait_until(lambda: len(server._connections) == 1)
            status, headers, body = self.request(server)
            self.assertEqual(status, 503); self.assertEqual(headers['Retry-After'], '1')
            self.assertEqual(json.loads(body)['error']['type'], 'overloaded_error')
            self.assertEqual(len(server._connections), 1)
        finally: held.close()
        wait_until(lambda: not server._connections)
        self.assertEqual(self.request(server)[0], 200)
        self.assertEqual(server.engine.metrics.requests, 0)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('openai', 'anthropic', 'httpx')), 'SDKs are optional')
    def test_official_clients_receive_retryable_busy_before_admission(self):
        import anthropic, httpx, openai
        server = self.serve(limits=ServiceLimits(max_connections=1))
        held = socket.create_connection(server.server_address, timeout=2)
        try:
            held.sendall(b'GET /health HTTP/1.1\r\nHost:')
            wait_until(lambda: len(server._connections) == 1)
            with openai.OpenAI(base_url=f'http://127.0.0.1:{server.server_port}/v1', api_key='test', max_retries=0,
                    http_client=httpx.Client(trust_env=False)) as client:
                with self.assertRaises(openai.InternalServerError) as error: client.models.list()
                self.assertEqual(error.exception.response.headers['retry-after'], '1')
            with anthropic.Anthropic(base_url=f'http://127.0.0.1:{server.server_port}', api_key='test', max_retries=0,
                    http_client=httpx.Client(trust_env=False)) as client:
                with self.assertRaises(anthropic.InternalServerError) as error: client.models.list()
                self.assertEqual(error.exception.response.headers['retry-after'], '1')
            self.assertEqual(server.engine.metrics.requests, 0)
        finally: held.close()

    def test_read_idle_closes_incomplete_headers_and_returns_408_for_incomplete_body(self):
        server = self.serve(limits=ServiceLimits(read_idle_seconds=.07))
        for body in (False, True):
            with socket.create_connection(server.server_address, timeout=2) as connection:
                payload = (b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{'
                           if body else b'GET /health HTTP/1.1\r\nHost:')
                connection.sendall(payload)
                data = b''
                while True:
                    chunk = connection.recv(4096)
                    if not chunk: break
                    data += chunk
                if body: self.assertTrue(data.startswith(b'HTTP/1.1 408 '), data)
            wait_until(lambda: not server._connections)
        self.assertEqual(self.request(server)[0], 200)
        self.assertFalse(hasattr(server.engine.worker, 'request'))

    def test_write_idle_cancels_and_holds_native_budget_until_confirmed_retirement(self):
        retiring, release = threading.Event(), threading.Event()
        class HeldNative(Native):
            def retire(self, **kwargs):
                retiring.set()
                if not release.wait(3): raise WorkerError('timeout', 'test retirement was not released')
                return super().retire(**kwargs)
        class LiveWorker(Worker):
            context, stop_ids, healthy = 16, frozenset(), True
            def open_session(self, ids, limit, **kwargs):
                request = HeldNative(self, ids)
                self.requests.append(request); return request
        class BigCodec(FakeCodec):
            stop_ids = frozenset()
            def new_decoder(self):
                return SimpleNamespace(push=lambda token: 'x' * (8 << 20), finish=lambda: '')
        worker = LiveWorker()
        engine = TextEngine(worker, BigCodec(), 'hyle',
            session_policy=SessionPolicy(kv_capacity=1, queue_timeout_seconds=.08))
        server = self.serve(limits=ServiceLimits(write_idle_seconds=.08), engine=engine)
        connection = socket.socket(); connection.settimeout(2)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        connection.connect(server.server_address)
        body = json.dumps({'model': 'hyle', 'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 1, 'stream': True}).encode()
        try:
            connection.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: ' +
                               str(len(body)).encode() + b'\r\n\r\n' + body)
            self.assertTrue(retiring.wait(2), 'blocked write never cancelled the native request')
            self.assertEqual(engine.sessions.snapshot(), {'waiting': 0, 'active': 1, 'resident': 1})
            status, _, metrics = self.request(server, '/metrics')
            self.assertEqual(status, 200)
            for line in (b'hyle_sessions_active 1', b'hyle_sessions_resident 1', b'hyle_sessions_waiting 0', b'hyle_http_connections 2'):
                self.assertIn(line, metrics)
            requests = [('/v1/responses', {'model': 'hyle', 'input': 'x', 'max_output_tokens': 1}),
                ('/v1/chat/completions', {'model': 'hyle', 'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 1}),
                ('/v1/messages', {'model': 'hyle', 'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 1})]
            for path, payload in requests:
                status, headers, _ = self.request(server, path, payload)
                self.assertEqual(status, 503); self.assertEqual(headers['Retry-After'], '1')
                self.assertEqual(len(worker.requests), 1)
            release.set()
            wait_until(lambda: engine.sessions.snapshot()['resident'] == 0)
            self.assertTrue(worker.requests[0].retired)
            self.assertEqual(self.request(server)[0], 200)
        finally:
            release.set(); connection.close()


class SocketProgressTest(unittest.TestCase):
    def test_read_idle_renews_only_after_real_bytes_without_a_total_request_deadline(self):
        # Assert progress/idle semantics without depending on sub-100ms cloud
        # runner scheduling. Real blocked socket IO is covered above.
        clock = [0.]
        class Socket:
            ready = False
            chunks = iter(b'abc')
            def recv_into(self, target):
                if not self.ready: raise BlockingIOError()
                self.ready = False; target[0] = next(self.chunks); return 1
        connection = Socket()
        def select_progress(readable, writable, exceptional, timeout):
            self.assertEqual(readable, [connection])
            clock[0] += min(.07, timeout)
            if timeout < .07: return [], [], []
            connection.ready = True
            return [connection], [], []
        with patch('engine.text.server.time.monotonic', lambda: clock[0]), \
             patch('engine.text.server.select.select', select_progress), \
             io.BufferedReader(_SocketReader(connection, threading.Event(), .12)) as reader:
            self.assertEqual(reader.read(3), b'abc')
        self.assertAlmostEqual(clock[0], .21)


if __name__ == '__main__': unittest.main()
