"""Local tokenizer/template and real model Responses integration; no downloads."""
import argparse
import http.client
import json
from pathlib import Path
import socket
import threading
import urllib.request

from engine.text.app import ResponsesServer
from engine.text.codec import HuggingFaceTextCodec
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient


def post(port, streaming, *, half_close=False):
    body = json.dumps({'model': 'hyle', 'input': '你好，what is 2 + 2?',
                       'max_output_tokens': 16, 'stream': streaming}).encode('utf-8')
    if half_close:
        with socket.create_connection(('127.0.0.1', port), timeout=60) as connection:
            connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n'
                               b'Content-Type: application/json\r\nConnection: close\r\n'
                               + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
            connection.shutdown(socket.SHUT_WR)
            response = http.client.HTTPResponse(connection)
            response.begin()
            assert response.status == 200, response.status
            raw = response.read().decode('utf-8')
    else:
        request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/responses', data=body,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode('utf-8')
    if not streaming:
        return json.loads(raw)
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ')]
    assert [event['sequence_number'] for event in events] == list(range(len(events)))
    assert events[-1]['type'] in ('response.completed', 'response.incomplete'), events[-1]
    output = events[-1]['response']
    text = ''.join(event['delta'] for event in events if event['type'] == 'response.output_text.delta')
    assert text == output['output'][0]['content'][0]['text']
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mtp-auto', action='store_true', help='use the model\'s automatic MTP policy')
    args = parser.parse_args()
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', '256', '--max-sessions', '1', '--active-sequences', '1']
    if not args.mtp_auto:
        command += ['--draft-count', '0']
    worker = WorkerClient(command)
    try:
        template = Path('tests/data/text-codec-test-template.jinja')
        default = HuggingFaceTextCodec(args.tokenizer_dir, expected_vocabulary=worker.vocabulary)
        explicit = HuggingFaceTextCodec(args.tokenizer_dir, template=template, expected_vocabulary=worker.vocabulary)
        messages = [{'role': 'system', 'content': 'Be concise'}, {'role': 'user', 'content': '你好, alma!'}]
        default_ids = default.encode_messages(messages)
        explicit_ids = explicit.encode_messages(messages)
        assert default_ids != explicit_ids
        assert explicit._tokenizer.decode(explicit_ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False) == 'system:Be concise;user:你好, alma!;assistant:'
        text = 'Hello, 世界! Árvíztűrő tükörfúrógép.'
        ids = explicit._tokenizer.encode(text, add_special_tokens=False)
        decoder = explicit.new_decoder()
        decoded = ''.join(decoder.push(token) for token in ids) + decoder.finish()
        assert decoded == text, (decoded, text)
        for name, codec in (('default', default), ('explicit', explicit)):
            server = ResponsesServer(('127.0.0.1', args.port), TextEngine(worker, codec, 'hyle'), args.metrics)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                ordinary, stream = post(args.port, False), post(args.port, True)
                assert ordinary['usage']['output_tokens'] > 0
                assert ordinary['usage'] == stream['usage']
                assert ordinary['output'][0]['content'][0]['text'] == stream['output'][0]['content'][0]['text']
                for streaming in (False, True):
                    half_closed = post(args.port, streaming, half_close=True)
                    assert half_closed['usage'] == ordinary['usage']
                    assert half_closed['output'][0]['content'][0]['text'] == ordinary['output'][0]['content'][0]['text']
                print(name, 'template: nonstream and stream matched', ordinary['usage']['output_tokens'], 'tokens', flush=True)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
        print('local tokenizer multilingual incremental decode matched', flush=True)
    finally:
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
