"""Exercise request-local errors through a real model worker and Responses adapter.

Only existing local inputs are used. The caller may choose another port when
2048 is already occupied; the test never stops an existing listener.
"""
import argparse
import json
import threading
import urllib.error
import urllib.request

from engine.text.app import ResponsesServer
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient


class Decoder:
    def push(self, token):
        return str(token)

    def finish(self):
        return ''


class FixtureCodec:
    stop_ids = frozenset()

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def encode_messages(self, messages):
        return [self.vocabulary] if messages[-1]['content'] == 'invalid' else [1, 2, 3]

    def new_decoder(self):
        return Decoder()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    worker = WorkerClient([args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
                           '--capacity', '128', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', '0'])
    server = None
    thread = None
    try:
        server = ResponsesServer(('127.0.0.1', args.port), TextEngine(worker, FixtureCodec(worker.vocabulary), 'hyle'),
                                 args.metrics)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        endpoint = f'http://127.0.0.1:{args.port}/v1/responses'
        for streaming in (False, True):
            body = {'model': 'hyle', 'input': 'invalid', 'max_output_tokens': 4, 'stream': streaming}
            request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
            try:
                response = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                text = response.read().decode()
                assert response.status == (200 if streaming else 400), (response.status, text)
            if streaming:
                events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
                assert events[-1]['type'] == 'error' and events[-1]['code'] == 'invalid_request_error', events[-1]
            else:
                assert json.loads(text)['error']['type'] == 'invalid_request_error'
            assert worker.healthy
        body = {'model': 'hyle', 'input': 'valid', 'max_output_tokens': 4}
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        assert result['status'] in ('completed', 'incomplete') and result['usage']['output_tokens'] > 0
        if args.metrics:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=10) as response:
                metrics = response.read().decode()
            assert 'hyle_requests_failed_total 2\n' in metrics and 'hyle_requests_completed_total 1\n' in metrics
        print('real worker: nonstream and stream invalid requests isolated; following request completed')
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
                thread.join()
            server.server_close()
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
