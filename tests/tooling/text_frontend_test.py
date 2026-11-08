import json
import threading
import unittest
import urllib.error
import urllib.request
from argparse import Namespace

from engine.text.app import InvalidRequest, ResponsesServer, TextEngine, make_worker_command, parse_responses_input
from engine.text.worker_client import TokenEvent


class FakeDecoder:
    def push(self, token):
        return chr(token)

    def finish(self):
        return ""


class FakeCodec:
    def encode_messages(self, messages):
        return [1, 2]

    def new_decoder(self):
        return FakeDecoder()


class FakeRequest:
    def __init__(self):
        self.events = iter([TokenEvent("token", ord("a")), TokenEvent("token", 3), TokenEvent("completed")])
        self.acks = 0
        self.closed = False

    def read(self, timeout=None):
        return next(self.events)

    def acknowledge(self):
        self.acks += 1

    def close(self):
        self.closed = True


class FakeWorker:
    context = 16
    stop_ids = {3}
    healthy = True

    def submit(self, tokens, max_output_tokens):
        self.request = FakeRequest()
        return self.request


class TextFrontendTest(unittest.TestCase):
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
                        {"model": "hyle", "input": "x", "store": True},
                        {"model": "wrong", "input": "x"}):
            with self.subTest(payload=payload), self.assertRaises(InvalidRequest):
                parse_responses_input(payload, "hyle")

    def test_message_parts_and_context_limit(self):
        messages, _, _, _ = parse_responses_input({"model": "hyle", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]}, "hyle")
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])
        engine = TextEngine(FakeWorker(), FakeCodec(), "hyle")
        with self.assertRaises(InvalidRequest):
            engine.prepare({"model": "hyle", "input": "x", "max_output_tokens": 16})

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
                self.assertTrue(worker.request.closed)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
