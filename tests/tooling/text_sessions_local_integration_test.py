"""Local CUDA Responses continuations and cold choices. No download/build."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import struct
import threading
import time
import urllib.request

from engine.text.app import TextApiServer, make_worker_command, parse_arguments, prepare_choices
from engine.text.codec import HuggingFaceTextCodec
from engine.text.configuration import read_configuration
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--asset-cache', required=True)
    parser.add_argument('--template')
    parser.add_argument('--port', type=int, default=2048)
    args = parser.parse_args()
    with socket.socket() as check:
        if check.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('requested inference port is already occupied')
    arguments = ['serve', '--engine', args.engine, '--inputs', args.inputs,
        '--tokenizer-dir', args.tokenizer_dir, '--backend', 'cuda', '--capacity', '4096',
        '--max-sessions', '3', '--active-sequences', '2', '--config', args.config,
        '--asset-cache', args.asset_cache, '--port', str(args.port), '--metrics']
    if args.template: arguments += ['--template', args.template]
    options = parse_arguments(arguments)
    choices = read_configuration(options)
    codec = HuggingFaceTextCodec(options.tokenizer_dir, template=options.template)
    choices = prepare_choices(options, codec, choices)
    worker = WorkerClient(make_worker_command(options, choices), startup_timeout=120)
    server = None
    try:
        codec.validate_vocabulary(worker.vocabulary)
        engine = TextEngine(worker, codec, 'hyle', choices=choices, session_policy=choices.sessions)
        server = TextApiServer(('127.0.0.1', args.port), engine, True)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05})
        thread.start()
        import openai
        client = openai.OpenAI(base_url=f'http://127.0.0.1:{args.port}/v1', api_key='local',
                               max_retries=0, timeout=180, _strict_response_validation=True)
        try:
            def create(profile, sampler='default'):
                return client.responses.create(model='hyle', input='Please answer with a short greeting.',
                    max_output_tokens=16, extra_body={'hyle': {'execution_profile': profile,
                                                             'sampling_profile': sampler, 'seed': 7}})
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(create, 'default'), pool.submit(create, 'subset'), pool.submit(create, 'plain', 'chat')]
                initial = [future.result() for future in futures]
            assert [response.metadata['hyle.execution_profile'] for response in initial] == ['default', 'subset', 'plain']
            assert all(response.store and response.usage.output_tokens > 0 for response in initial)
            first = initial[0]
            next_turn = client.responses.create(model='hyle', previous_response_id=first.id,
                input='Now give a second short greeting.', max_output_tokens=16,
                extra_body={'hyle': {'seed': 7}})
            assert next_turn.previous_response_id == first.id
            if args.template:
                assert next_turn.metadata['hyle.context'] == 'continued', 'append-stable template did not reuse original native KV'
            print('HTTP continuation:', next_turn.metadata, 'cached_tokens=', next_turn.usage.input_tokens_details.cached_tokens, flush=True)
            # A fork remains valid after the parent tip was consumed by a child.
            fork = client.responses.create(model='hyle', previous_response_id=first.id,
                input='Give a different short greeting.', max_output_tokens=16,
                extra_body={'hyle': {'execution_profile': 'plain', 'sampling_profile': 'chat', 'seed': 11}})
            assert fork.previous_response_id == first.id
            fetched = client.responses.retrieve(next_turn.id)
            assert fetched.output_text == next_turn.output_text
            deleted = client.responses.with_raw_response.delete(next_turn.id)
            assert deleted.http_response.json()['deleted']
            try:
                client.responses.retrieve(next_turn.id)
            except openai.NotFoundError:
                pass
            else:
                raise AssertionError('deleted continuation remained available')
            try:
                client.responses.create(model='hyle', input='x', max_output_tokens=16,
                    extra_body={'hyle': {'execution_profile': 'not-prepared'}})
            except openai.BadRequestError:
                pass
            else:
                raise AssertionError('unprepared profile was accepted')
            events = list(client.responses.create(model='hyle', input='Say hello briefly.',
                max_output_tokens=16, stream=True, store=False,
                extra_body={'hyle': {'execution_profile': 'subset'}}))
            assert events[-1].type in ('response.completed', 'response.incomplete')
            assert all(event.sequence_number == index for index, event in enumerate(events))
            for response in initial + [fork]:
                client.responses.delete(response.id)
            # Drop delivery during generation. The failed metric is written
            # after close() has awaited the original native terminal event.
            body = json.dumps(dict(model='hyle', input='Explain a long topic in detail.',
                                   max_output_tokens=256, stream=True)).encode()
            connection = socket.create_connection(('127.0.0.1', args.port), timeout=10)
            connection.sendall(b'POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                               + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
            prefix = b''
            while b'response.created' not in prefix:
                block = connection.recv(4096)
                assert block, 'stream closed before generation began'
                prefix += block
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('HH' if os.name == 'nt' else 'ii', 1, 0))
            connection.close()
            deadline = time.monotonic() + 30
            while True:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=10) as reply:
                    metrics = reply.read()
                if b'hyle_requests_failed_total 1' in metrics: break
                if time.monotonic() > deadline: raise AssertionError('disconnected request did not finish native retirement')
                time.sleep(.02)
            recovered = create('default')
            client.responses.delete(recovered.id)
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=10) as reply:
                assert json.load(reply)['status'] == 'ok'
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=10) as reply:
                assert b'hyle_requests_completed_total 7' in reply.read()
            print('PASS strict OpenAI SDK: mixed profiles, saved turns/fork, retrieve/delete, invalid isolation, SSE and disconnect retirement', flush=True)
        finally:
            client.close()
            server.shutdown(); server.server_close(); thread.join(timeout=10)
            assert not thread.is_alive()
    finally:
        worker.close(timeout=120)


if __name__ == '__main__': main()
