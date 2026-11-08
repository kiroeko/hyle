"""Exercise two real HTTP streams, cancellation, and a subsequent request.

Uses an existing local Qwen worker and input directory; never downloads models.
"""
import argparse
import json
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

from engine.text.app import ResponsesServer
from engine.text.codec import HuggingFaceTextCodec
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
    parser.add_argument('--draft-count', type=int, choices=range(4), default=0,
                        help='explicit MTP draft depth; 0 tests ordinary generation')
    parser.add_argument('--tokenizer-dir', help='existing local tokenizer and model chat template')
    parser.add_argument('--require-batched-verifier', action='store_true',
                        help='also require an observed two-request recursive verifier batch')
    args = parser.parse_args()
    if args.require_batched_verifier:
        if args.draft_count < 2:
            parser.error('--require-batched-verifier requires at least two drafts')
        command = [sys.executable, '-B', '-m', 'tests.tooling.text_worker_http_concurrency_test',
                   '--engine', args.engine, '--inputs', args.inputs, '--port', str(args.port),
                   '--sampling', args.sampling, '--draft-count', str(args.draft_count),
                   '--metrics' if args.metrics else '--no-metrics']
        if args.tokenizer_dir:
            command += ['--tokenizer-dir', args.tokenizer_dir]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding='utf-8', errors='replace', timeout=180)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            raise RuntimeError(f'concurrent Responses check failed ({result.returncode})')
        pattern = rf'^generation_batch requests=2 T={args.draft_count + 1} role=1(?:\s|$)'
        count = len(re.findall(pattern, result.stderr, flags=re.MULTILINE))
        if not count:
            raise AssertionError('two-request recursive verifier batch was not observed')
        print(f'observed {count} two-request K{args.draft_count} verifier batch(es)', flush=True)
        return
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', '256', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', str(args.draft_count),
               '--sampling', args.sampling]
    if args.sampling == 'random':
        command += ['--temperature', '0.8', '--top-k', '40', '--top-p', '0.95', '--seed', '123']
    worker = WorkerClient(command)
    server = thread = first = second = None
    try:
        codec = (HuggingFaceTextCodec(args.tokenizer_dir, expected_vocabulary=worker.vocabulary)
                 if args.tokenizer_dir else FixtureCodec(worker.vocabulary))
        server = ResponsesServer(('127.0.0.1', args.port), TextEngine(worker, codec, 'hyle'), args.metrics)
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
        streamed = ''.join(item['delta'] for item in events if item['type'] == 'response.output_text.delta')
        assert streamed == events[-1]['response']['output'][0]['content'][0]['text']
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
        print(f'two live {args.sampling} HTTP streams (draft-count={args.draft_count}, '
              f'{"local tokenizer" if args.tokenizer_dir else "fixture codec"}): '
              'reset one; peer and subsequent request completed', flush=True)
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
