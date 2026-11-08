"""Exercise two real HTTP streams, cancellation, and a subsequent request.

Uses an existing local Qwen worker and input directory; never downloads models.
"""
import argparse
import json
import socket
import struct
import threading
import time
import urllib.request

from engine.text.app import ResponsesServer
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient
from tests.tooling.text_worker_error_integration_test import FixtureCodec


def connect(port):
    body = json.dumps({'model': 'hyle', 'input': 'valid', 'max_output_tokens': 100, 'stream': True}).encode()
    connection = socket.create_connection(('127.0.0.1', port), timeout=60)
    connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n'
                       b'Content-Type: application/json\r\nConnection: close\r\n'
                       + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
    return connection


def receive_first_delta(connection):
    data = b''
    while b'event: response.output_text.delta\n' not in data:
        part = connection.recv(65536)
        assert part, data
        data += part
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sampling', choices=('greedy', 'random'), default='greedy')
    args = parser.parse_args()
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', '256', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', '0',
               '--sampling', args.sampling]
    if args.sampling == 'random':
        command += ['--temperature', '0.8', '--top-k', '40', '--top-p', '0.95', '--seed', '123']
    worker = WorkerClient(command)
    server = thread = first = second = None
    try:
        server = ResponsesServer(('127.0.0.1', args.port), TextEngine(worker, FixtureCodec(worker.vocabulary), 'hyle'),
                                 args.metrics)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        first, second = connect(args.port), connect(args.port)
        receive_first_delta(first)
        data = receive_first_delta(second)
        assert b'event: response.completed\n' not in data and b'event: response.incomplete\n' not in data
        metrics = server.engine.metrics.render().decode()
        assert 'hyle_requests_total 2\n' in metrics and 'hyle_requests_completed_total 0\n' in metrics, metrics
        first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('hh', 1, 0))
        first.close()
        first = None
        while True:
            part = second.recv(65536)
            if not part:
                break
            data += part
        second.close()
        second = None
        events = [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith('data: ')]
        assert [item['sequence_number'] for item in events] == list(range(len(events)))
        assert events[-1]['type'] in ('response.completed', 'response.incomplete'), events[-1]
        assert events[-1]['response']['usage']['output_tokens'] > 1
        assert worker.healthy
        body = json.dumps({'model': 'hyle', 'input': 'valid', 'max_output_tokens': 4}).encode()
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/responses', data=body,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            followup = json.load(response)
        assert followup['status'] in ('completed', 'incomplete') and followup['usage']['output_tokens'] > 0
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            metrics = server.engine.metrics.render().decode()
            if 'hyle_requests_failed_total 1\n' in metrics and 'hyle_requests_completed_total 2\n' in metrics:
                break
            time.sleep(.05)
        else:
            raise AssertionError(metrics)
        if args.metrics:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=10) as response:
                assert response.read().decode() == metrics
        print(f'two live {args.sampling} HTTP streams: reset one; peer and subsequent request completed', flush=True)
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
                thread.join()
            server.server_close()
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
