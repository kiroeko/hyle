"""Run an SDK programming tool loop against an explicitly supplied local model.

No downloads and no server-side tool execution. The driver only reads/replaces
its own calc.py fixture; it never executes generated code or commands.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


FUNCTIONS = [
    {'name': 'read_file', 'description': 'Read the exact contents of a relative project file.',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}},
    {'name': 'edit_file', 'description': 'Replace exactly one occurrence of old_text with new_text in a project file.',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_text': {'type': 'string'},
        'new_text': {'type': 'string'}}, 'required': ['path', 'old_text', 'new_text']}},
]
INSTRUCTIONS = 'You are a coding assistant. Inspect files with read_file and modify them with edit_file. Never invent file contents. After successful editing, briefly report success.'
TASK = 'Read calc.py. Then replace exactly ANSWER = 41 with ANSWER = 42 using edit_file. Report success after the tool result.'


def run(args):
    import anthropic
    import httpx
    import openai
    from engine.model_preparation import local_text_format, validate_local_tokenizer
    from engine.text.app import TextApiServer
    from engine.text.codec import HuggingFaceTextCodec
    from engine.text.configuration import SessionPolicy
    from engine.text.runtime import TextEngine
    from engine.text.worker_client import WorkerClient

    args.fixture.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture / 'calc.py'
    codec = HuggingFaceTextCodec(args.tokenizer, model_format=local_text_format(args.inputs))
    source, _ = validate_local_tokenizer(args.inputs, codec.preparation_tokenizer)
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', args.backend,
        '--capacity', '4096', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', '0', '--overlap', 'off',
        '--cpu-ffn-layers', str(args.cpu_ffn_layers)]
    if args.prefill_chunk_tokens is not None: command += ['--prefill-chunk-tokens', str(args.prefill_chunk_tokens)]
    worker = WorkerClient(command, startup_timeout=120)
    engine = server = thread = oa = ant = None
    try:
        assert worker.source_identity == source
        engine = TextEngine(worker, codec, 'hyle', session_policy=SessionPolicy(capacity=16, kv_capacity=2, max_waiting=4))
        server = TextApiServer(('127.0.0.1', args.port), engine, True)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        base = f'http://127.0.0.1:{args.port}'
        oa = openai.OpenAI(base_url=base+'/v1', api_key='local-test', max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=90), _strict_response_validation=True)
        ant = anthropic.Anthropic(base_url=base, api_key='local-test', max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=90), _strict_response_validation=True)

        oa_model = oa.models.retrieve('hyle')
        ant_model = ant.models.retrieve('hyle')
        capabilities = oa_model.model_dump()['hyle']
        assert capabilities == ant_model.model_dump()['hyle'] == engine.describe_capabilities()
        assert [m.id for m in oa.models.list().data] == [m.id for m in ant.models.list().data] == ['hyle']
        assert capabilities['context_tokens'] == worker.context == 4096
        assert capabilities['placement'] == {'primary_backend': args.backend, 'cpu_ffn_layers': args.cpu_ffn_layers}
        assert capabilities['execution_profiles'][0]['algorithm'] == 'ordinary'
        assert capabilities['execution_profiles'][0]['max_draft_tokens'] == 0
        assert capabilities['codec']['tools']['supported'] and not capabilities['codec']['tools']['strict']
        assert not capabilities['structured_output']['strict_schema']
        print('PASS actual model discovery; both official SDK formats and actual READY capabilities', flush=True)

        def execute(name, arguments):
            assert arguments.get('path') == 'calc.py', arguments
            if name == 'read_file':
                assert set(arguments) == {'path'}, arguments
                return fixture.read_text(encoding='utf-8')
            assert name == 'edit_file' and set(arguments) == {'path', 'old_text', 'new_text'}, (name, arguments)
            old, new = arguments['old_text'], arguments['new_text']
            assert isinstance(old, str) and old and isinstance(new, str)
            text = fixture.read_text(encoding='utf-8'); assert text.count(old) == 1, arguments
            updated = text.replace(old, new, 1)
            assert updated.strip() == 'ANSWER = 42', (arguments, updated)
            fixture.write_text(updated, encoding='utf-8')
            return 'Successfully updated calc.py: ANSWER = 42.'

        for protocol in ('responses', 'chat', 'anthropic'):
            for streaming in (False, True):
                fixture.write_text('ANSWER = 41\n', encoding='utf-8')
                history = [{'role': 'user', 'content': TASK}]
                previous = None; ids = set(); native_contexts = []
                for expected in ('read_file', 'edit_file', None):
                    if protocol == 'responses':
                        parameters = dict(model='hyle', input=history, instructions=INSTRUCTIONS,
                            tools=[{'type': 'function', **f, 'strict': False} for f in FUNCTIONS],
                            tool_choice=args.first_tool_choice if expected == 'read_file' else {'type': 'function', 'name': expected} if expected else 'none',
                            parallel_tool_calls=False, max_output_tokens=192, store=True,
                            metadata={'case': 'coding-tool-loop'})
                        if previous: parameters['previous_response_id'] = previous
                        if streaming:
                            with oa.responses.stream(**parameters) as events:
                                event_list = list(events); result = events.get_final_response()
                            assert [e.sequence_number for e in event_list] == list(range(len(event_list)))
                        else: result = oa.responses.create(**parameters)
                        assert result.status == 'completed', result
                        native_contexts.append(result.metadata['hyle.context'])
                        calls = [item for item in result.output if item.type == 'function_call']
                        if expected:
                            assert len(calls) == 1 and calls[0].name == expected, result
                            call = calls[0]; identity = call.call_id
                            output = execute(call.name, json.loads(call.arguments))
                            history = [{'type': 'function_call_output', 'call_id': identity, 'output': output}]
                        else:
                            assert not calls and result.output_text.strip(), result
                        previous = result.id
                    elif protocol == 'chat':
                        parameters = dict(model='hyle', messages=[{'role': 'developer', 'content': INSTRUCTIONS}, *history],
                            tools=[{'type': 'function', 'function': {**f, 'strict': False}} for f in FUNCTIONS],
                            tool_choice=args.first_tool_choice if expected == 'read_file' else {'type': 'function', 'function': {'name': expected}} if expected else 'none',
                            parallel_tool_calls=False, max_tokens=192)
                        if streaming:
                            with oa.chat.completions.stream(**parameters) as events:
                                list(events); result = events.get_final_completion()
                        else: result = oa.chat.completions.create(**parameters)
                        choice = result.choices[0]; message = choice.message
                        history.append(message.model_dump(exclude_none=True))
                        if expected:
                            assert choice.finish_reason == 'tool_calls' and len(message.tool_calls) == 1, result
                            call = message.tool_calls[0]; identity = call.id
                            assert call.function.name == expected, result
                            output = execute(call.function.name, json.loads(call.function.arguments))
                            history.append({'role': 'tool', 'tool_call_id': identity, 'content': output})
                        else: assert choice.finish_reason == 'stop' and message.content.strip() and not message.tool_calls, result
                    else:
                        parameters = dict(model='hyle', messages=history, system=INSTRUCTIONS,
                            tools=[{'name': f['name'], 'description': f['description'], 'input_schema': f['parameters']} for f in FUNCTIONS],
                            tool_choice={'type': 'any' if args.first_tool_choice=='required' else 'auto', 'disable_parallel_tool_use': True} if expected == 'read_file' else
                                {'type': 'tool', 'name': expected, 'disable_parallel_tool_use': True} if expected else {'type': 'none'},
                            max_tokens=192)
                        if streaming:
                            with ant.messages.stream(**parameters) as events:
                                list(events); result = events.get_final_message()
                        else: result = ant.messages.create(**parameters)
                        history.append({'role': 'assistant', 'content': [b.model_dump(exclude_none=True) for b in result.content]})
                        calls = [b for b in result.content if b.type == 'tool_use']
                        if expected:
                            assert result.stop_reason == 'tool_use' and len(calls) == 1 and calls[0].name == expected, result
                            call = calls[0]; identity = call.id; output = execute(call.name, call.input)
                            history.append({'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': identity, 'content': output}]})
                        else:
                            assert result.stop_reason == 'end_turn' and not calls and any(b.type=='text' and b.text.strip() for b in result.content), result
                    if expected:
                        assert identity and identity not in ids; ids.add(identity)
                assert len(ids) == 2 and fixture.read_text(encoding='utf-8').strip() == 'ANSWER = 42'
                print('PASS actual model SDK coding loop', protocol, 'stream', streaming,
                    'first choice',args.first_tool_choice,'two client tools+final answer;stable unique IDs;native contexts', native_contexts, flush=True)
        with httpx.Client(trust_env=False) as client:
            metrics = client.get(base+'/metrics'); assert metrics.status_code == 200
        snapshot = engine.sessions.snapshot()
        assert snapshot['waiting'] == snapshot['active'] == 0 and 0 <= snapshot['resident'] <= 2
        for name, value in snapshot.items(): assert f'hyle_sessions_{name} {value}\n' in metrics.text
        assert 'hyle_http_connections ' in metrics.text
        print('PASS all3 protocols with official SDK sync/SSE local coding tool loops; SDKs', openai.__version__, anthropic.__version__, flush=True)
    finally:
        if oa: oa.close()
        if ant: ant.close()
        if server:
            server.shutdown()
            if thread: thread.join(timeout=5)
            server.server_close()
        elif engine: engine.close()
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'apple', 'vulkan'), default='cuda')
    parser.add_argument('--cpu-ffn-layers', type=int, default=0)
    parser.add_argument('--prefill-chunk-tokens', type=int)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--first-tool-choice', choices=('auto','required'), default='auto')
    args = parser.parse_args()
    if not all(importlib.util.find_spec(name) for name in ('openai', 'anthropic', 'transformers', 'tokenizers', 'httpx')):
        print('SKIP: local text API dependencies and official SDKs are required'); return 77
    run(args); return 0


if __name__ == '__main__': raise SystemExit(main())
