"""Text protocol contracts, real sockets and optional strict official SDKs."""
import importlib.util
import json
import socket
import struct
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from engine.text.anthropic import parse_messages
from engine.text.app import TextApiServer
from engine.text.chat_completions import parse_chat_completion
from engine.text.codec import HuggingFaceTextCodec
from engine.text.configuration import ExecutionChoice, PreparedChoices, SamplingChoice, SessionPolicy
from engine.text.contracts import TokenEvent, WorkerError
from engine.text.runtime import GenerationProgress, TextEngine, _TextStops
from tests.tooling.text_sessions_test import Worker as LiveWorker


class Decoder:
    def __init__(self, chunks, tail=''): self.chunks, self.tail = chunks, tail
    def push(self, token): return self.chunks[token - 10]
    def finish(self): return self.tail


class Codec:
    stop_ids = frozenset({3})
    def __init__(self):
        self.chunks, self.tail = ['a'], ''
        self.messages = []
    def encode_messages(self, messages, *, continue_final_message=False):
        self.messages.append((messages, continue_final_message))
        return [1, 2]
    def new_decoder(self): return Decoder(list(self.chunks), self.tail)


class Request:
    def __init__(self, events):
        self.events = iter(events)
        self.acks = self.reads = self.retirements = 0
        self.closed = threading.Event()
        self.retirement_error = None
    def read(self, timeout=None): self.reads += 1; return next(self.events)
    def acknowledge(self): self.acks += 1
    def retire(self, *, timeout=120):
        self.retirements += 1
        if self.retirement_error: raise self.retirement_error
        return TokenEvent('cancelled')
    def close(self): self.closed.set()


class Worker:
    context = 1024
    stop_ids = frozenset({3})
    healthy = True
    def __init__(self):
        self.events = [TokenEvent('token', 10), TokenEvent('token', 3), TokenEvent('completed')]
        self.requests = []
        self.options = []
        self.failure = self.retirement_error = None
    def submit(self, ids, limit, **options):
        if self.failure: raise self.failure
        request = Request(list(self.events)); request.retirement_error = self.retirement_error
        self.requests.append(request); self.options.append(options)
        return request


class CompletionParsingTest(unittest.TestCase):
    def test_chat_preserves_roles_and_text_parts_and_maps_signed_seed(self):
        inputs, _ = parse_chat_completion({'model': 'hyle', 'messages': [
            {'role': 'developer', 'content': 'Be concise'},
            {'role': 'user', 'content': [{'type': 'text', 'text': '你好'}, {'type': 'text', 'text': '!'}]}],
            'max_completion_tokens': 12, 'seed': -1}, 'hyle')
        self.assertEqual((inputs.limit, inputs.selection['seed']), (12, (1 << 64) - 1))
        self.assertEqual(inputs.messages, [{'role': 'developer', 'content': 'Be concise'}, {'role': 'user', 'content': '你好!'}])

    def test_messages_system_blocks_and_assistant_prefill(self):
        inputs, _ = parse_messages({'model': 'hyle', 'max_tokens': 8,
            'system': [{'type': 'text', 'text': 'Be concise'}], 'messages': [
                {'role': 'user', 'content': 'The answer?'}, {'role': 'assistant', 'content': 'It is '}],
            'top_k': 40, 'stop_sequences': ['END']}, 'hyle')
        self.assertTrue(inputs.continue_final)
        self.assertEqual((inputs.messages[0]['role'], inputs.stops, inputs.top_k), ('system', ('END',), 40))

    def test_rejects_unimplemented_or_invalid_semantics(self):
        chat = {'model': 'hyle', 'messages': [{'role': 'user', 'content': 'x'}]}
        messages = {**chat, 'max_tokens': 3}
        for parse, base, changes in (
            (parse_chat_completion, chat, [dict(n=2), dict(n=True), dict(tools=[]), dict(stop=['x'] * 5),
                dict(max_tokens=3, max_completion_tokens=3), dict(seed=True), dict(seed=1 << 63),
                dict(seed=1, hyle={'seed': True}), dict(stream_options={'include_usage': True}),
                dict(stream=True, stream_options={'include_usage': 1}), dict(store=True)]),
            (parse_messages, messages, [dict(max_tokens=None), dict(max_tokens=True), dict(stop_sequences='x'),
                dict(system={'text': 'x'}), dict(tools=[]), dict(thinking={'type': 'enabled'}), dict(stream='yes')])):
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError): parse({**base, **change}, 'hyle')
        for parse, base in ((parse_chat_completion, chat), (parse_messages, messages)):
            for change in (dict(model='wrong'), dict(messages=[]), dict(messages=[{'role': 'tool', 'content': 'x'}]),
                dict(messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}])):
                with self.subTest(change=change), self.assertRaises(ValueError): parse({**base, **change}, 'hyle')
        with self.assertRaises(ValueError): parse_messages(chat, 'hyle')

    def test_nullable_chat_options_use_defaults(self):
        inputs, _ = parse_chat_completion({'model': 'hyle', 'messages': [{'role': 'user', 'content': 'x'}],
            'n': None, 'stream': None, 'store': None, 'seed': None, 'max_tokens': None}, 'hyle')
        self.assertEqual((inputs.limit, inputs.stream), (256, False))

    def test_top_k_selects_existing_sampler_without_preparing_one(self):
        choices = PreparedChoices(samplers=(SamplingChoice(), SamplingChoice('chat', True, .8, 40, .95)))
        self.assertEqual(choices.select(top_k=40, temperature=.8).sampling_index, 1)
        for value in (True, -1, 1.5, 100):
            with self.subTest(value=value), self.assertRaises(ValueError): choices.select(top_k=value)

    def test_template_prefill_does_not_add_an_assistant_generation_prompt(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): self.options = kwargs; return {'input_ids': [1, 2]}
        codec = object.__new__(HuggingFaceTextCodec); codec._tokenizer = Tokenizer(); codec._template = None
        codec.encode_messages([{'role': 'assistant', 'content': 'prefix'}], continue_final_message=True)
        self.assertTrue(codec._tokenizer.options['continue_final_message'])
        self.assertFalse(codec._tokenizer.options['add_generation_prompt'])


class TextStopsTest(unittest.TestCase):
    def test_delimiters_never_leak_across_chunk_boundaries(self):
        value = 'hello<END>ignored'
        for boundary in range(1, len(value)):
            stops = _TextStops(['<END>']); chunks = []
            for chunk in (value[:boundary], value[boundary:]):
                visible, matched = stops.push(chunk); chunks.append(visible)
                if matched: break
            self.assertEqual(''.join(chunks), 'hello')
            self.assertEqual(matched, '<END>')

    def test_overlapping_stops_are_independent_of_chunking(self):
        for pieces in (['abcd-tail'], ['a', 'b', 'cd-tail'], list('abcd-tail')):
            stops = _TextStops(['abcd', 'b']); output = []
            for chunk in pieces:
                visible, matched = stops.push(chunk); output.append(visible)
                if matched: break
            self.assertEqual((''.join(output), matched), ('a', 'b'))

    def test_partial_prefix_flushes_at_real_end(self):
        stops = _TextStops(['<END>'])
        self.assertEqual(stops.push('hello<EN'), ('hello', None))
        self.assertEqual(stops.push('', final=True), ('<EN', None))

    def test_generation_counts_consumed_tokens_and_holds_final_credit(self):
        worker, codec = Worker(), Codec(); codec.chunks = ['before<EN', 'D>ignored', 'after']
        worker.events = [TokenEvent('token', 10 + i) for i in range(3)] + [TokenEvent('completed')]
        engine = TextEngine(worker, codec, 'hyle'); request = engine.submit(engine.prepare([{'role': 'user', 'content': 'x'}], 8))
        progress = GenerationProgress(); chunks = []
        self.assertEqual(engine.generate(request, chunks.append, progress, stop_sequences=['<END>']), 'before')
        self.assertEqual((''.join(chunks), progress.count, progress.stop_sequence, request.acks, request.reads), ('before', 2, '<END>', 1, 2))
        request.close()


SDK_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('openai', 'anthropic', 'httpx'))


@unittest.skipUnless(SDK_AVAILABLE, 'OpenAI and Anthropic SDKs are optional')
class CompletionSdkTest(unittest.TestCase):
    def setUp(self):
        import anthropic, httpx, openai
        self.worker, self.codec = Worker(), Codec()
        self.engine = TextEngine(self.worker, self.codec, 'hyle')
        self.server = TextApiServer(('127.0.0.1', 0), self.engine, True)
        self.thread = threading.Thread(target=self.server.serve_forever); self.thread.start()
        base_url = f'http://127.0.0.1:{self.server.server_port}'
        self.openai = openai.OpenAI(base_url=base_url + '/v1', api_key='local-test', max_retries=0,
            http_client=httpx.Client(trust_env=False), _strict_response_validation=True)
        self.anthropic = anthropic.Anthropic(base_url=base_url, api_key='local-test', max_retries=0,
            http_client=httpx.Client(trust_env=False), _strict_response_validation=True)
        self.messages = [{'role': 'user', 'content': 'x'}]

    def tearDown(self):
        self.openai.close(); self.anthropic.close()
        self.server.shutdown(); self.thread.join(timeout=3); self.server.server_close()

    def chat(self, **kwargs):
        return self.openai.chat.completions.create(model='hyle', messages=self.messages, max_tokens=3, **kwargs)

    def anthropic_message(self, **kwargs):
        return self.anthropic.messages.create(model='hyle', messages=self.messages, max_tokens=3, **kwargs)

    def raw(self, path, payload):
        request = urllib.request.Request(f'http://127.0.0.1:{self.server.server_port}' + path,
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=3)

    def test_strict_nonstream_objects_and_usage(self):
        chat, message = self.chat(), self.anthropic_message()
        self.assertEqual((chat.object, chat.choices[0].message.content, chat.choices[0].finish_reason), ('chat.completion', 'a', 'stop'))
        self.assertEqual((chat.usage.prompt_tokens, chat.usage.completion_tokens, chat.usage.total_tokens), (2, 2, 4))
        self.assertEqual((message.type, message.content[0].text, message.stop_reason, message.stop_sequence), ('message', 'a', 'end_turn', None))
        self.assertEqual((message.usage.input_tokens, message.usage.output_tokens), (2, 2))
        self.assertTrue(all(request.closed.is_set() and request.retirements == 1 for request in self.worker.requests))

    def test_chat_stream_data_only_done_and_optional_usage(self):
        chunks = list(self.chat(stream=True, stream_options={'include_usage': True}))
        self.assertEqual(chunks[0].choices[0].delta.role, 'assistant')
        self.assertEqual(len({chunk.id for chunk in chunks}), 1)
        self.assertEqual(''.join(choice.delta.content or '' for chunk in chunks for choice in chunk.choices), 'a')
        self.assertEqual(chunks[-2].choices[0].finish_reason, 'stop')
        self.assertEqual((chunks[-1].choices, chunks[-1].usage.completion_tokens), ([], 2))
        with self.raw('/v1/chat/completions', {'model': 'hyle', 'messages': self.messages, 'max_tokens': 3, 'stream': True}) as response:
            raw = response.read().decode()
        self.assertTrue(raw.rstrip().endswith('data: [DONE]'))
        self.assertNotIn('event:', raw)
        self.assertTrue(all(chunk.usage is None for chunk in self.chat(stream=True)))
        with self.openai.chat.completions.stream(model='hyle', messages=self.messages, max_completion_tokens=3) as stream:
            list(stream)
            self.assertEqual(stream.get_final_completion().choices[0].message.content, 'a')

    def test_anthropic_stream_helper_reconstructs_final_message(self):
        with self.anthropic.messages.stream(model='hyle', messages=self.messages, max_tokens=3) as stream:
            events = list(stream); final = stream.get_final_message()
        self.assertEqual([event.type for event in events if event.type != 'text'], ['message_start', 'content_block_start',
            'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop'])
        self.assertEqual([event.text for event in events if event.type == 'text'], ['a'])
        self.assertEqual((final.content[0].text, final.stop_reason, final.usage.output_tokens), ('a', 'end_turn', 2))

    def test_truncation_and_eos_at_quota_have_distinct_reasons(self):
        self.codec.chunks = ['a', 'b', 'c']
        self.worker.events = [TokenEvent('token', 10 + i) for i in range(3)] + [TokenEvent('completed')]
        self.assertEqual(self.chat().choices[0].finish_reason, 'length')
        self.assertEqual(self.anthropic_message().stop_reason, 'max_tokens')
        self.worker.events[-2] = TokenEvent('token', 3)
        self.assertEqual(self.chat().choices[0].finish_reason, 'stop')
        self.assertEqual(self.anthropic_message().stop_reason, 'end_turn')

    def test_stop_sequences_cancel_native_work_and_match_streaming(self):
        self.codec.chunks = ['ab<EN', 'D>ignored', 'later']
        self.worker.events = [TokenEvent('token', 10 + i) for i in range(3)] + [TokenEvent('completed')]
        chat = self.chat(stop='<END>')
        self.assertEqual((chat.choices[0].message.content, chat.choices[0].finish_reason), ('ab', 'stop'))
        message = self.anthropic_message(stop_sequences=['<END>'])
        self.assertEqual((message.content[0].text, message.stop_reason, message.stop_sequence), ('ab', 'stop_sequence', '<END>'))
        chunks = list(self.chat(stop='<END>', stream=True))
        self.assertEqual(''.join(c.choices[0].delta.content or '' for c in chunks), 'ab')
        with self.anthropic.messages.stream(model='hyle', messages=self.messages, max_tokens=3, stop_sequences=['<END>']) as stream:
            text = ''.join(stream.text_stream); message = stream.get_final_message()
        self.assertEqual((text, message.stop_reason), ('ab', 'stop_sequence'))
        self.assertTrue(all(r.reads == 2 and r.acks == 1 and r.retirements == 1 and r.closed.is_set() for r in self.worker.requests))

    def test_sampling_selection_is_shared_with_responses(self):
        self.engine.choices = PreparedChoices(executions=(ExecutionChoice(), ExecutionChoice('plain', 0)),
            samplers=(SamplingChoice(), SamplingChoice('chat', True, .8, 40, .95)))
        values = {'execution_profile': 'plain', 'sampling_profile': 'chat', 'seed': 7}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.chat, temperature=.8, extra_body={'hyle': values}),
                pool.submit(self.anthropic_message, temperature=.8, top_k=40, extra_body={'hyle': values}),
                pool.submit(self.openai.responses.create, model='hyle', input='x', max_output_tokens=3,
                    store=False, extra_body={'hyle': values})]
            for future in futures: future.result(timeout=3)
        for options in self.worker.options:
            self.assertEqual((options['execution_profile'], options['sampling_index'], options['seed']), (1, 1, 7))

    def test_invalid_sampling_and_tools_reject_before_any_work(self):
        import openai, anthropic
        for call, error in ((self.chat, openai.BadRequestError), (self.anthropic_message, anthropic.BadRequestError)):
            with self.assertRaises(error): call(temperature=.123)
            with self.assertRaises(error): call(extra_body={'tools': []})
        self.assertEqual(self.worker.requests, [])
        self.assertEqual(self.chat().choices[0].message.content, 'a')

    def test_native_generation_and_retirement_failures_do_not_end_in_success(self):
        import openai, anthropic
        for retirement in (False, True):
            self.worker.events = [TokenEvent('token', 10), TokenEvent('completed')] if retirement else [
                TokenEvent('token', 10), TokenEvent('error', message='model failed', error_kind='model')]
            self.worker.retirement_error = WorkerError('retirement', 'not retired') if retirement else None
            for call, error in ((self.chat, openai.APIError), (self.anthropic_message, anthropic.APIError)):
                with self.assertRaises(error): call()
                with self.assertRaises(error): list(call(stream=True))
        self.assertTrue(all(r.closed.is_set() for r in self.worker.requests))

    def test_anthropic_assistant_prefix_reaches_template_as_continuation(self):
        self.anthropic.messages.create(model='hyle', max_tokens=3, messages=[*self.messages, {'role': 'assistant', 'content': 'It is '}])
        self.assertTrue(self.codec.messages[-1][1])

    def test_decoder_failure_is_a_server_error_after_admission(self):
        import openai, anthropic
        class BrokenDecoder(Decoder):
            def finish(self): raise ValueError('invalid internal decoder state')
        self.codec.new_decoder = lambda: BrokenDecoder(self.codec.chunks)
        for call, error in ((self.chat, openai.InternalServerError), (self.anthropic_message, anthropic.InternalServerError)):
            with self.assertRaises(error) as caught: call()
            self.assertEqual(caught.exception.status_code, 503)

    def test_stateless_protocols_share_session_budget_but_not_private_domains(self):
        native = LiveWorker()
        native.context = 1024
        self.engine.worker = native
        from engine.text.sessions import TextSessions
        owner = TextSessions(native, SessionPolicy(kv_capacity=1))
        self.engine.sessions = owner
        self.chat(); self.anthropic_message()
        self.assertNotEqual(native.domains[0], native.domains[1])
        self.assertTrue(all(request.retired for request in native.requests))
        self.assertEqual((owner._native_count, len(owner._records), len(owner._active)), (0, 0, 0))

    def test_disconnect_cancels_both_protocols(self):
        class Busy(Request):
            def __init__(self): super().__init__([]); self.started = threading.Event()
            def read(self, timeout=None): self.started.set(); time.sleep(.02); return TokenEvent('token', 10)
        for path in ('/v1/chat/completions', '/v1/messages'):
            request = Busy()
            self.worker.submit = lambda *args, **kwargs: request
            connection = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=3)
            try:
                payload = json.dumps({'model': 'hyle', 'messages': self.messages, 'max_tokens': 100, 'stream': True}).encode()
                connection.sendall(f'POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n'.encode() + payload)
                self.assertTrue(request.started.wait(timeout=3))
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('hh', 1, 0)); connection.close()
                self.assertTrue(request.closed.wait(timeout=1))
                self.assertEqual(request.retirements, 1)
            finally: connection.close()


if __name__ == '__main__': unittest.main()
