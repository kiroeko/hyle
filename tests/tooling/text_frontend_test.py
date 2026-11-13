import json
import contextlib
import io
import queue
import socket
import struct
import threading
import time
import unittest
import urllib.error
import urllib.request
from argparse import Namespace

from engine.text.app import ResponsesServer, make_worker_command, parse_arguments
from engine.text.contracts import WorkerError
from engine.text.responses import InvalidRequest, ResponsesAdapter, parse_responses_input
from engine.text.runtime import GenerationProgress, TextEngine
from engine.text.server import TextHttpServer
from engine.text.worker_client import TokenEvent


class FakeDecoder:
    def push(self, token):
        return chr(token)

    def finish(self):
        return ""


class FakeCodec:
    stop_ids = frozenset({3})
    def encode_messages(self, messages):
        return [1, 2]

    def new_decoder(self):
        return FakeDecoder()


class FakeRequest:
    def __init__(self):
        self.events = iter([TokenEvent("token", ord("a")), TokenEvent("token", 3), TokenEvent("completed")])
        self.acks = 0
        self.closed = False
        self.closed_event = threading.Event()

    def read(self, timeout=None):
        return next(self.events)

    def acknowledge(self):
        self.acks += 1

    def close(self):
        self.closed = True
        self.closed_event.set()


class FakeWorker:
    context = 16
    stop_ids = {3}
    healthy = True

    def submit(self, tokens, max_output_tokens, *, stop_ids=()):
        self.extra_stop_ids = stop_ids
        self.request = FakeRequest()
        return self.request


class TextFrontendTest(unittest.TestCase):
    def test_half_closed_request_still_receives_nonstream_response(self):
        class GatedRequest(FakeRequest):
            def __init__(self):
                super().__init__()
                self.started, self.resume = threading.Event(), threading.Event()
                self.first = True

            def read(self, timeout=None):
                if self.first:
                    self.first = False
                    self.started.set()
                    if not self.resume.wait(timeout=3):
                        raise RuntimeError('test did not resume generation')
                    # Ensure the next read checks the peer after the FIN.
                    time.sleep(.3)
                return super().read(timeout)

        request = GatedRequest()
        class GatedWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                return request

        server = ResponsesServer(('127.0.0.1', 0), TextEngine(GatedWorker(), FakeCodec(), 'hyle'), True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
        try:
            body = json.dumps({'model': 'hyle', 'input': 'x', 'max_output_tokens': 3}).encode()
            connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n'
                               + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
            self.assertTrue(request.started.wait(timeout=3))
            connection.shutdown(socket.SHUT_WR)
            request.resume.set()
            data = bytearray()
            while part := connection.recv(4096):
                data.extend(part)
            headers, response = bytes(data).split(b'\r\n\r\n', 1)
            self.assertTrue(headers.startswith(b'HTTP/1.1 200 '), headers)
            self.assertEqual(json.loads(response)['output'][0]['content'][0]['text'], 'a')
            self.assertTrue(request.closed_event.wait(timeout=3))
        finally:
            request.resume.set()
            connection.close()
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

    def test_nonstream_disconnect_cancels_while_tokens_continue(self):
        class BusyRequest(FakeRequest):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.produced = 0

            def read(self, timeout=None):
                time.sleep(.01)
                if self.produced == 100:
                    return TokenEvent('completed')
                self.produced += 1
                self.started.set()
                return TokenEvent('token', ord('a'))

        request = BusyRequest()
        class BusyWorker(FakeWorker):
            context = 256
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                return request

        server = ResponsesServer(('127.0.0.1', 0), TextEngine(BusyWorker(), FakeCodec(), 'hyle'), True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
        try:
            body = json.dumps({'model': 'hyle', 'input': 'x', 'max_output_tokens': 100}).encode()
            connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n'
                               + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
            self.assertTrue(request.started.wait(timeout=3))
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('hh', 1, 0))
            connection.close()
            self.assertTrue(request.closed_event.wait(timeout=.75), 'disconnect did not cancel ongoing generation')
            self.assertLess(request.produced, 100)
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

    def test_explicit_sampling_arguments_are_not_silently_ignored(self):
        base = ['generate', '--engine', 'hyle', '--inputs', 'model', '--tokenizer-dir', 'tokenizer', '--text', 'x']
        for option, value in (('--temperature', '1'), ('--top-k', '0'), ('--top-p', '1'), ('--seed', '0')):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as exit_code:
                    parse_arguments(base + [option, value])
                self.assertEqual(exit_code.exception.code, 2)
                self.assertIn('require --sampling random', error.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(['serve', '--engine', 'hyle', '--inputs', 'model', '--tokenizer-dir', 'tokenizer',
                             '--temperature', '.8'])
        defaults = parse_arguments(base + ['--sampling', 'random'])
        self.assertEqual((defaults.temperature, defaults.top_k, defaults.top_p, defaults.seed), (1., 0, 1., 0))
        command = make_worker_command(defaults)
        for option, value in (('--temperature', '1.0'), ('--top-k', '0'), ('--top-p', '1.0'), ('--seed', '0')):
            self.assertEqual(command[command.index(option) + 1], value)
        selected = parse_arguments(base + ['--sampling', 'random', '--temperature', '.8', '--top-k', '40',
                                           '--top-p', '.95', '--seed', '7'])
        self.assertEqual((selected.temperature, selected.top_k, selected.top_p, selected.seed), (.8, 40, .95, 7))

    def test_invalid_random_sampling_is_rejected_before_loading(self):
        base = ['generate', '--engine', 'hyle', '--inputs', 'model', '--tokenizer-dir', 'tokenizer',
                '--text', 'x', '--sampling', 'random']
        for option, value in (('--temperature', 'nan'), ('--temperature', '0'), ('--top-p', 'inf'),
                              ('--top-p', '0'), ('--top-p', '1.01'), ('--top-k', '-1'),
                              ('--seed', '-1'), ('--seed', '18446744073709551616')):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_arguments(base + [option, value])
                self.assertEqual(error.exception.code, 2)

    def test_server_close_cancels_inflight_handler_before_return(self):
        class StalledRequest(FakeRequest):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.first = True

            def read(self, timeout=None):
                if self.first:
                    self.first = False
                    self.started.set()
                    return TokenEvent('token', ord('a'))
                self.closed_event.wait(timeout=timeout)
                raise queue.Empty

        request = StalledRequest()
        class StalledWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                return request

        server = ResponsesServer(('127.0.0.1', 0), TextEngine(StalledWorker(), FakeCodec(), 'hyle'), True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
        try:
            body = json.dumps({'model': 'hyle', 'input': 'x', 'max_output_tokens': 3}).encode()
            connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n'
                               + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
            self.assertTrue(request.started.wait(timeout=3))
            server.shutdown()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            server.server_close()
            self.assertTrue(request.closed_event.wait(timeout=3), 'request handler outlived server close')
        finally:
            connection.close()
            request.closed_event.wait(timeout=3)
            if thread.is_alive():
                server.shutdown()
                thread.join(timeout=3)
            server.server_close()

    def test_server_close_wakes_incomplete_request_body(self):
        entered, finished = threading.Event(), threading.Event()

        class ReadingAdapter:
            def handle_post(self, exchange, engine):
                entered.set()
                try:
                    exchange.read_json()
                except (OSError, ValueError):
                    pass
                finally:
                    finished.set()

        server = TextHttpServer(('127.0.0.1', 0), TextEngine(FakeWorker(), FakeCodec(), 'hyle'),
                                True, {'/read': ReadingAdapter()})
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
        try:
            connection.sendall(b'POST /read HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{')
            self.assertTrue(entered.wait(timeout=3))
            server.shutdown()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            server.server_close()
            self.assertTrue(finished.is_set())
        finally:
            connection.close()
            if thread.is_alive():
                server.shutdown()
                thread.join(timeout=3)
            server.server_close()

    def test_transport_composes_independent_adapter_routes(self):
        class TestAdapter:
            def handle_post(self, exchange, engine):
                payload = exchange.read_json()
                prepared = engine.prepare([{'role': 'user', 'content': payload['text']}], 3)
                request = engine.submit(prepared)
                try:
                    text = engine.generate(request, lambda _: None, GenerationProgress())
                    exchange.send_json(200, {'output': text})
                finally:
                    request.close()
        worker = FakeWorker()
        engine = TextEngine(worker, FakeCodec(), 'hyle')
        routes = {'/v1/responses': ResponsesAdapter(), '/test-adapter': TestAdapter()}
        server = TextHttpServer(('127.0.0.1', 0), engine, True, routes)
        routes.clear()  # The running service retains its cold composition.
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            for path, body, expected in (
                ('/test-adapter', {'text': 'x'}, 'a'),
                ('/v1/responses', {'model': 'hyle', 'input': 'x', 'max_output_tokens': 3}, 'a'),
            ):
                request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                                 headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request) as response:
                    result = json.load(response)
                text = result['output'] if path == '/test-adapter' else result['output'][0]['content'][0]['text']
                self.assertEqual(text, expected)
                self.assertTrue(worker.request.closed_event.wait(timeout=3))
            with self.assertRaises(urllib.error.HTTPError) as missing:
                urllib.request.urlopen(urllib.request.Request(base + '/missing', data=b'{}'))
            self.assertEqual(missing.exception.code, 404)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_unspecified_overlap_is_selected_by_model_preparation(self):
        options = dict(engine="hyle", inputs="model", backend="cuda", capacity=128,
                       max_sessions=1, active_sequences=1, draft_count=0,
                       sampling="greedy", draft_vocabulary=None, draft_head=None)
        self.assertNotIn("--overlap", make_worker_command(Namespace(**options, overlap=None)))
        for setting, value in ((True, "on"), (False, "off")):
            command = make_worker_command(Namespace(**options, overlap=setting))
            self.assertEqual(command[command.index("--overlap") + 1], value)

    def test_unspecified_draft_count_uses_model_policy(self):
        options = dict(engine="hyle", inputs="model", backend="cuda", capacity=128,
                       max_sessions=1, active_sequences=1, overlap=None,
                       sampling="greedy", draft_vocabulary=None, draft_head=None)
        self.assertNotIn("--draft-count", make_worker_command(Namespace(**options, draft_count=None)))
        for count in (0, 1, 3):
            command = make_worker_command(Namespace(**options, draft_count=count))
            self.assertEqual(command[command.index("--draft-count") + 1], str(count))

    def test_reject_unsupported_request_instead_of_ignoring(self):
        for payload in ({"model": "hyle", "input": "x", "tools": []},
                        {"model": "hyle", "input": "x", "store": "true"},
                        {"model": "wrong", "input": "x"}):
            with self.subTest(payload=payload), self.assertRaises(InvalidRequest):
                parse_responses_input(payload, "hyle")

    def test_message_parts_and_context_limit(self):
        messages, _, _, _ = parse_responses_input({"model": "hyle", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]}, "hyle")
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])
        engine = TextEngine(FakeWorker(), FakeCodec(), "hyle")
        with self.assertRaises(ValueError):
            engine.prepare(messages, 16)

    def test_runtime_delivers_before_acknowledging(self):
        engine = TextEngine(FakeWorker(), FakeCodec(), "hyle")
        prepared = engine.prepare([{"role": "user", "content": "x"}], 3)
        request = engine.submit(prepared)
        progress = GenerationProgress()
        def refuse(_):
            raise ConnectionError("consumer stopped")
        with self.assertRaises(ConnectionError):
            engine.generate(request, refuse, progress)
        self.assertEqual((progress.count, progress.last, request.acks), (1, ord("a"), 0))

    def test_submit_errors_preserve_request_and_worker_failure_classification(self):
        class RejectingWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                raise self.failure

        worker = RejectingWorker()
        engine = TextEngine(worker, FakeCodec(), 'hyle')
        server = ResponsesServer(('127.0.0.1', 0), engine, True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f'http://127.0.0.1:{server.server_port}/v1/responses'
            cases = ((ValueError('invalid token'), 400, 'invalid_request_error'),
                     (WorkerError('invalid', 'invalid token'), 400, 'invalid_request_error'),
                     (WorkerError('closed', 'invalid token'), 503, 'server_error'))
            for failure, status, kind in cases:
                worker.failure = failure
                for streaming in (False, True):
                    body = {'model': 'hyle', 'input': 'x', 'max_output_tokens': 3, 'stream': streaming}
                    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                                     headers={'Content-Type': 'application/json'})
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request)
                    with caught.exception as response:
                        self.assertEqual(response.status, status)
                        self.assertEqual(json.loads(response.read())['error']['type'], kind)
            self.assertEqual((engine.metrics.requests, engine.metrics.failed), (6, 6))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_http_error_classification_uses_kind_not_message(self):
        class ErrorWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                request = super().submit(tokens, max_output_tokens, stop_ids=stop_ids)
                request.events = iter([TokenEvent("error", message="invalid: same wording", error_kind=self.error_kind)])
                return request
        worker = ErrorWorker()
        server = ResponsesServer(("127.0.0.1", 0), TextEngine(worker, FakeCodec(), "hyle"), True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/responses"
            for kind, status, code in (("invalid", 400, "invalid_request_error"), ("model", 503, "server_error")):
                worker.error_kind = kind
                for streaming in (False, True):
                    body = {"model": "hyle", "input": "x", "max_output_tokens": 3, "stream": streaming}
                    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                                     headers={"Content-Type": "application/json"})
                    try:
                        response = urllib.request.urlopen(request)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        data = response.read().decode()
                        self.assertEqual(response.status, 200 if streaming else status)
                    if streaming:
                        events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ")]
                        self.assertEqual(events[-1]["type"], "error")
                        self.assertEqual(events[-1]["code"], code)
                        self.assertIsNone(events[-1]["param"])
                        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
                    else:
                        self.assertEqual(json.loads(data)["error"]["type"], code)
                    self.assertTrue(worker.request.closed_event.wait(timeout=3))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_tokenizer_end_token_is_submitted_and_not_rendered(self):
        codec = FakeCodec()
        codec.stop_ids = frozenset({3, ord("a")})
        worker = FakeWorker()
        engine = TextEngine(worker, codec, "hyle")
        request = engine.submit(engine.prepare([{"role": "user", "content": "x"}], 3))
        chunks = []
        engine.generate(request, chunks.append, GenerationProgress())
        self.assertEqual(worker.extra_stop_ids, (ord("a"),))
        self.assertEqual(chunks, [])

    def test_tokenizer_stop_at_output_limit_is_completed(self):
        class StopWorker(FakeWorker):
            def submit(self, tokens, max_output_tokens, *, stop_ids=()):
                request = super().submit(tokens, max_output_tokens, stop_ids=stop_ids)
                request.events = iter([TokenEvent("token", ord("a")), TokenEvent("completed")])
                return request
        codec = FakeCodec()
        codec.stop_ids = frozenset({3, ord("a")})
        server = ResponsesServer(("127.0.0.1", 0), TextEngine(StopWorker(), codec, "hyle"), True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/responses"
            for streaming in (False, True):
                body = {"model": "hyle", "input": "x", "max_output_tokens": 1, "stream": streaming}
                request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request) as response:
                    data = response.read().decode()
                if streaming:
                    events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ")]
                    self.assertEqual(events[-1]["type"], "response.completed")
                    output = events[-1]["response"]
                else:
                    output = json.loads(data)
                self.assertEqual(output["status"], "completed")
                self.assertEqual(output["output"][0]["content"][0]["text"], "")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_http_response_and_stream_finish_and_acknowledge(self):
        worker = FakeWorker()
        server = ResponsesServer(("127.0.0.1", 0), TextEngine(worker, FakeCodec(), "hyle"), True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/responses"
            for streaming in (False, True):
                body = {"model": "hyle", "input": "x", "max_output_tokens": 3, "stream": streaming}
                request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request) as response:
                    data = response.read().decode()
                if streaming:
                    events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ")]
                    self.assertEqual(events[-1]["type"], "response.completed")
                    self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
                    output = events[-1]["response"]
                else:
                    output = json.loads(data)
                self.assertEqual(output["output"][0]["content"][0]["text"], "a")
                self.assertEqual(output["usage"]["output_tokens"], 2)
                self.assertEqual(worker.request.acks, 2)
                self.assertTrue(worker.request.closed_event.wait(timeout=3))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
