"""All three text protocols against one existing local native worker; no build/download."""
import argparse
import json
import os
import socket
import struct
import threading
import time
import urllib.request

from engine.text.app import TextApiServer
from engine.model_preparation import local_text_format, validate_local_tokenizer
from engine.text.codec import HuggingFaceTextCodec
from engine.text.configuration import SessionPolicy
from engine.text.runtime import TextEngine
from engine.text.worker_client import WorkerClient


def response_events(client, **parameters):
    # The documented incomplete terminal event is distinct from completed.
    # The SDK's completed-only accumulator cannot represent quota endings.
    with client.responses.create(**parameters, stream=True) as stream:
        events = list(stream)
    assert events and events[-1].type in ('response.completed', 'response.incomplete')
    assert [event.sequence_number for event in events] == list(range(len(events)))
    return events[-1].response


def disconnect_during_output(base, port, protocol):
    """Reset delivery after a real text delta, then await native retirement."""
    def failures():
        with urllib.request.urlopen(base + '/metrics', timeout=10) as reply:
            lines = reply.read().decode('ascii').splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith('hyle_requests_failed_total ')))
    before = failures()
    prompt = 'Count from 1 to 100, one number per line, without skipping any number.'
    payload = dict(model='hyle', stream=True)
    if protocol == 'responses':
        path = '/v1/responses'; payload.update(input=prompt, max_output_tokens=256, store=False)
    else:
        path = '/v1/chat/completions' if protocol == 'chat' else '/v1/messages'
        payload.update(messages=[dict(role='user', content=prompt)], max_tokens=256)
    body = json.dumps(payload).encode()
    connection = socket.create_connection(('127.0.0.1', port), timeout=120)
    try:
        connection.sendall(f'POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
        pending = b''; seen = False
        while not seen:
            block = connection.recv(4096)
            assert block, 'stream ended before a real generated text delta'
            pending += block
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                if not line.startswith(b'data: '): continue
                event = json.loads(line[6:])
                if protocol == 'responses': seen |= event.get('type') == 'response.output_text.delta' and bool(event.get('delta'))
                elif protocol == 'chat': seen |= any(choice.get('delta', {}).get('content') for choice in event.get('choices', []))
                else: seen |= event.get('type') == 'content_block_delta' and bool(event.get('delta', {}).get('text'))
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
            struct.pack('HH' if os.name == 'nt' else 'ii', 1, 0))
    finally:
        connection.close()
    deadline = time.monotonic() + 60
    while failures() < before + 1:
        if time.monotonic() >= deadline: raise AssertionError(protocol + ' disconnect did not retire native request')
        time.sleep(.02)


def main():
    import anthropic, httpx, openai
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'apple', 'vulkan'), default='cuda')
    parser.add_argument('--cpu-ffn-layers', type=int, default=0)
    parser.add_argument('--prefill-chunk-tokens', type=int)
    args = parser.parse_args()
    if not args.metrics: raise ValueError('native disconnect acceptance requires metrics')
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', args.backend,
        '--capacity', '2048', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', '0',
        '--cpu-ffn-layers', str(args.cpu_ffn_layers)]
    if args.prefill_chunk_tokens is not None: command += ['--prefill-chunk-tokens', str(args.prefill_chunk_tokens)]
    worker = WorkerClient(command, startup_timeout=120)
    try:
        assert worker.placement.primary_backend == args.backend
        assert worker.placement.cpu_ffn_layers == args.cpu_ffn_layers
        assert worker.execution_profiles[0].algorithm == 'ordinary'
        assert worker.execution_profiles[0].max_draft_tokens == 0
        assert worker.source_identity == (Path(args.inputs) / 'weights.tsv').read_text().splitlines()[1]
        codec = HuggingFaceTextCodec(args.tokenizer_dir, expected_vocabulary=worker.vocabulary,
            model_format=local_text_format(args.inputs))
        source, _ = validate_local_tokenizer(args.inputs, codec.preparation_tokenizer)
        assert source == worker.source_identity
        codec.validate_vocabulary(worker.vocabulary)
        engine = TextEngine(worker, codec, 'hyle', session_policy=SessionPolicy(kv_capacity=2))
        server = TextApiServer(('127.0.0.1', args.port), engine, args.metrics)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            base = f'http://127.0.0.1:{args.port}'
            with openai.OpenAI(base_url=base + '/v1', api_key='local-test', max_retries=0,
                    http_client=httpx.Client(trust_env=False), _strict_response_validation=True) as oa, \
                 anthropic.Anthropic(base_url=base, api_key='local-test', max_retries=0,
                    http_client=httpx.Client(trust_env=False), _strict_response_validation=True) as ant:
                prompt = 'Reply with one English greeting.'
                messages = [{'role': 'user', 'content': prompt}]
                responses = oa.responses.create(model='hyle', input=prompt, max_output_tokens=8, store=False)
                chat = oa.chat.completions.create(model='hyle', messages=messages, max_completion_tokens=8)
                message = ant.messages.create(model='hyle', messages=messages, max_tokens=8)
                expected = responses.output_text
                assert expected == chat.choices[0].message.content == message.content[0].text
                assert responses.usage.output_tokens == chat.usage.completion_tokens == message.usage.output_tokens
                chunks = list(oa.chat.completions.create(model='hyle', messages=messages, max_tokens=8,
                    stream=True, stream_options={'include_usage': True}))
                assert ''.join(choice.delta.content or '' for chunk in chunks for choice in chunk.choices) == expected
                assert chunks[-1].usage.completion_tokens == chat.usage.completion_tokens
                with ant.messages.stream(model='hyle', messages=messages, max_tokens=8) as stream:
                    assert ''.join(stream.text_stream) == expected
                    assert stream.get_final_message().usage.output_tokens == message.usage.output_tokens
                final = response_events(oa, model='hyle', input=prompt, max_output_tokens=8, store=False)
                assert final.output_text == expected and final.usage.output_tokens == responses.usage.output_tokens
                print('PASS local', args.backend, 'Responses, Chat and Messages text/usage; strict SDK stream helpers', flush=True)
                for streaming in (False, True):
                    if streaming:
                        quota = response_events(oa, model='hyle', input=prompt, max_output_tokens=1, store=False)
                        # The parsing accumulator raises on a length finish.
                        # Inspect the SDK's typed wire chunks and final usage.
                        with oa.chat.completions.create(model='hyle', messages=messages, max_tokens=1,
                                stream=True, stream_options={'include_usage': True}) as stream:
                            quota_chunks = list(stream)
                        reasons = [choice.finish_reason for chunk in quota_chunks for choice in chunk.choices if choice.finish_reason]
                        assert reasons == ['length']
                        chat_quota_reason = reasons[0]
                        chat_quota_tokens = quota_chunks[-1].usage.completion_tokens
                        with ant.messages.stream(model='hyle', messages=messages, max_tokens=1) as stream:
                            list(stream); ant_quota = stream.get_final_message()
                    else:
                        quota = oa.responses.create(model='hyle', input=prompt, max_output_tokens=1, store=False)
                        chat_quota = oa.chat.completions.create(model='hyle', messages=messages, max_tokens=1)
                        ant_quota = ant.messages.create(model='hyle', messages=messages, max_tokens=1)
                        chat_quota_reason = chat_quota.choices[0].finish_reason
                        chat_quota_tokens = chat_quota.usage.completion_tokens
                    assert quota.status == 'incomplete' and quota.incomplete_details.reason == 'max_output_tokens'
                    assert chat_quota_reason == 'length' and ant_quota.stop_reason == 'max_tokens'
                    assert quota.usage.output_tokens == chat_quota_tokens == ant_quota.usage.output_tokens == 1
                print('PASS all3 actual native quota/usage/finish reasons in normal and SSE', flush=True)
                prefilled = ant.messages.create(model='hyle', messages=[*messages, {'role': 'assistant', 'content': 'Hello'}], max_tokens=4)
                assert prefilled.usage.output_tokens > 0 and prefilled.stop_reason in ('end_turn', 'max_tokens')
                print('PASS local tokenizer/native assistant prefill', flush=True)
                if expected:
                    for streaming in (False, True):
                        if streaming:
                            with oa.chat.completions.stream(model='hyle', messages=messages, max_tokens=8, stop=expected[0]) as stream:
                                list(stream); stopped = stream.get_final_completion()
                            with ant.messages.stream(model='hyle', messages=messages, max_tokens=8, stop_sequences=[expected[0]]) as stream:
                                list(stream); ant_stopped = stream.get_final_message()
                        else:
                            stopped = oa.chat.completions.create(model='hyle', messages=messages, max_tokens=8, stop=expected[0])
                            ant_stopped = ant.messages.create(model='hyle', messages=messages, max_tokens=8, stop_sequences=[expected[0]])
                        assert stopped.choices[0].message.content == '' and stopped.choices[0].finish_reason == 'stop'
                        assert not any(part.text for part in ant_stopped.content if part.type == 'text')
                        assert ant_stopped.stop_reason == 'stop_sequence' and ant_stopped.stop_sequence == expected[0]
                    assert worker.healthy
                    assert oa.chat.completions.create(model='hyle', messages=messages, max_tokens=8).choices[0].message.content == expected
                    print('PASS local native stop cancellation/retirement and subsequent request', flush=True)
                else: raise AssertionError('native greeting must contain text to test stopping')
                # Responses has no text-stop parameter. Its normal stop is the
                # original model EOS; quota completion was checked separately.
                natural = oa.responses.create(model='hyle', input=prompt, max_output_tokens=64, store=False)
                assert natural.status == 'completed' and natural.usage.output_tokens < 64
                with oa.responses.stream(model='hyle', input=prompt, max_output_tokens=64, store=False) as stream:
                    list(stream); natural_stream = stream.get_final_response()
                assert natural_stream.status == 'completed' and natural_stream.output_text == natural.output_text
                for protocol in ('responses', 'chat', 'anthropic'):
                    disconnect_during_output(base, args.port, protocol)
                    assert worker.healthy
                    recovered = oa.responses.create(model='hyle', input=prompt, max_output_tokens=8, store=False)
                    assert recovered.output_text == expected
                    snapshot = engine.sessions.snapshot()
                    assert snapshot['waiting'] == snapshot['active'] == 0
                    print('PASS actual native disconnect/retirement/reuse', protocol, flush=True)
                for protocol in ('responses', 'chat', 'anthropic'):
                    print('PASS_NATIVE_PROTOCOL_BASE', args.backend, args.cpu_ffn_layers, protocol,
                        'normal/SSE stop quota disconnect terminal-retirement reuse', flush=True)
                print('SDK versions', openai.__version__, anthropic.__version__, 'tokens', message.usage.output_tokens, flush=True)
        finally:
            server.shutdown(); thread.join(timeout=3); server.server_close()
    finally: worker.close(timeout=120)


if __name__ == '__main__': main()
