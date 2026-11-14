"""Tool grammar/history contracts and real-socket official SDK consumers.

The worker here is synthetic. Actual model tool-loop qualification is separate.
"""
from __future__ import annotations

import importlib.util
import json
import threading
import unittest

from engine.models.qwen35.tool_format import ToolFormat
from engine.text.app import TextApiServer
from engine.text.codec import _ToolDecodeSession
from engine.text.contracts import TokenEvent
from engine.text.runtime import TextEngine
from engine.text.tools import history_call, json_object, tool_options, validate_tool_history
from tests.tooling.text_completion_protocols_test import Request


FUNCTION = {'name': 'read_file', 'description': 'Read the named project file.', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'line': {'type': 'integer'},
        'flag': {'type': 'boolean'}, 'options': {'type': 'object'}}, 'required': ['path']}}
XML = '<tool_call>\n<function=read_file>\n<parameter=path>\n程序.py\n</parameter>\n</function>\n</tool_call>'


class ToolContractsTest(unittest.TestCase):
    def test_definitions_are_owned_and_strict_is_never_claimed(self):
        source = json.loads(json.dumps(FUNCTION))
        value = tool_options([source], choice='named', name='read_file', parallel=False)
        source['parameters']['properties'].clear()
        self.assertIn('path', value.definitions[0]['function']['parameters']['properties'])
        self.assertTrue(value.enabled)
        self.assertFalse(tool_options([FUNCTION], choice='none').enabled)
        self.assertIsNone(tool_options([]))
        for functions, options in (([FUNCTION, FUNCTION], {}), ([{**FUNCTION, 'strict': True}], {}),
                ([{**FUNCTION, 'description': False}], {}), ([FUNCTION], {'choice': 'named', 'name': 'missing'}),
                ([], {'choice': 'required'}), ([FUNCTION], {'parallel': 1}), ([FUNCTION], {'choice': []})):
            with self.subTest(options=options), self.assertRaises(ValueError): tool_options(functions, **options)

    def test_arguments_cannot_hide_duplicates_or_nonfinite_values(self):
        for value in ('[]', '{"path":1,"path":2}', '{"path":NaN}', '{"path":1e400}'):
            with self.subTest(value=value), self.assertRaises(ValueError): json_object(value)

    def test_history_ids_are_unique_and_results_are_ordered(self):
        call = history_call('call_1', 'read_file', '{"path":"程序.py"}')
        assistant = {'role': 'assistant', 'content': '', 'tool_calls': [call]}
        result = {'role': 'tool', 'content': 'ok', 'tool_call_id': 'call_1'}
        validate_tool_history([assistant, result, {'role': 'user', 'content': 'continue'}])
        for messages in ([assistant], [result], [assistant, result, result],
                         [assistant, {'role': 'user', 'content': 'skip'}], [assistant, result, assistant, result]):
            with self.subTest(messages=messages), self.assertRaises(ValueError): validate_tool_history(messages)

    def test_grammar_preserves_order_values_and_every_fragment_boundary(self):
        xml = 'Checking.\n' + XML.replace('程序.py', ' leading\nbody\n ') .replace('</function>',
            '<parameter=line>\n3\n</parameter>\n<parameter=flag>\nTrue\n</parameter>\n'
            '<parameter=options>\n{"x":[1,"<&>"]}\n</parameter>\n</function>')
        definitions = tool_options([FUNCTION]).definitions
        for width in (1, 2, 7, len(xml)):
            events = []
            parser = ToolFormat.new_parser(definitions, lambda name, args: events.append(('call', name, args)),
                on_text=lambda text: events.append(('text', text)))
            for at in range(0, len(xml), width): parser.push(xml[at:at+width])
            parser.finish()
            self.assertEqual(''.join(event[1] for event in events if event[0]=='text'), 'Checking.\n')
            self.assertEqual(events[-1], ('call', 'read_file', {'path': ' leading\nbody\n ',
                'line': 3, 'flag': True, 'options': {'x': [1, '<&>']}}))

    def test_prefilled_choice_truncation_and_invalid_model_output(self):
        definitions = tool_options([FUNCTION]).definitions
        for choice, name in (('required', None), ('named', 'read_file')):
            prefix = ToolFormat.generation_prefix(choice, name); calls = []
            parser = ToolFormat.new_parser(definitions, lambda *args: calls.append(args), prefix=prefix, required=True)
            parser.push(XML[len(prefix):]); parser.finish()
            self.assertEqual(calls, [('read_file', {'path': '程序.py'})])
        parser = ToolFormat.new_parser(definitions, lambda *_: self.fail('unfinished call published'))
        parser.push(XML[:50]); self.assertEqual(parser.finish(truncated=True), '')
        for text in (XML[:50], XML.replace('read_file', 'undeclared'), XML + 'suffix', XML + XML):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parser = ToolFormat.new_parser(definitions, lambda *_: None, max_calls=1)
                parser.push(text); parser.finish()

    def test_source_instruction_roles_are_normalized_once(self):
        messages = ToolFormat.normalize_messages([{'role': 'system', 'content': 'first'},
            {'role': 'developer', 'content': 'second'}, {'role': 'user', 'content': 'question'}])
        self.assertEqual(messages[0], {'role': 'system', 'content': 'first\n\nsecond'})

    def test_source_template_orders_parallel_results_by_call_identity(self):
        calls = [history_call('a','read_file',{'path':'a.py'}),history_call('b','read_file',{'path':'b.py'})]
        messages = [{'role':'assistant','content':'','tool_calls':calls},
            {'role':'tool','tool_call_id':'b','content':'B'}, {'role':'tool','tool_call_id':'a','content':'A'},
            {'role':'user','content':'continue'}]
        validate_tool_history(messages)
        normalized = ToolFormat.normalize_messages(messages)
        self.assertEqual([m['content'] for m in normalized],['','A','B','continue'])
        self.assertEqual([m['content'] for m in messages],['','B','A','continue'])

    def test_text_only_codec_needs_no_tool_interface(self):
        from engine.text.contracts import ExecutionProfileFacts,ImageInput,PreparedPrompt
        class Codec:
            stop_ids = frozenset()
            def encode_messages(self,messages): return [1,2]
        class Worker:
            stop_ids = frozenset()
            context = 512
        engine = TextEngine(Worker(),Codec(),'text-only')
        messages = [{'role':'user','content':'plain'}]
        self.assertEqual(engine.prepare(messages,8,tools=tool_options([FUNCTION],choice='none')).ids,[1,2])
        with self.assertRaisesRegex(ValueError,'does not support tool calls'):
            engine.prepare(messages,8,tools=tool_options([FUNCTION]))
        class ImageCodec(Codec):
            def prepare_images(self,messages,*,continue_final_message,max_tokens):
                return PreparedPrompt([1,2],())
        worker=Worker()
        worker.execution_profiles=(ExecutionProfileFacts('default','ordinary',0,('text','image')),)
        image=[{'role':'user','content':[{'type':'image','image':ImageInput('image/png',b'fixture')}]}]
        engine=TextEngine(worker,ImageCodec(),'image-only')
        self.assertEqual(engine.prepare(image,8,tools=tool_options([FUNCTION],choice='none')).ids,[1,2])

    def test_large_fragmented_code_argument_is_preserved(self):
        value = '  def őr():\n    return "<&>"\n' * 4096
        calls = []
        parser = ToolFormat.new_parser(tool_options([FUNCTION]).definitions, lambda *args: calls.append(args))
        xml = XML.replace('程序.py', value)
        for at in range(0,len(xml),17): parser.push(xml[at:at+17])
        parser.finish()
        self.assertEqual(calls,[('read_file',{'path':value})])

    def test_xml_value_types_follow_local_refs_and_nullable_unions(self):
        function = {'name':'read_file','parameters':{'type':'object','$defs':{'Path':{'type':'string'}},
            'properties':{'path':{'$ref':'#/$defs/Path'},'line':{'anyOf':[{'type':'integer'},{'type':'null'}]}}}}
        definitions = tool_options([function]).definitions
        ToolFormat.validate_tools(definitions)
        for raw,expected in (('3',3),('null',None)):
            calls=[]
            parser=ToolFormat.new_parser(definitions,lambda *args:calls.append(args))
            parser.push(XML.replace('程序.py','{"keep":"as text"}').replace('</function>',
                '<parameter=line>\n'+raw+'\n</parameter>\n</function>'));parser.finish()
            self.assertEqual(calls,[('read_file',{'path':'{"keep":"as text"}','line':expected})])
        function['parameters']['properties']['line']={'$ref':'https://invalid.example/schema'}
        with self.assertRaisesRegex(ValueError,'local schema references'):
            ToolFormat.validate_tools(tool_options([function]).definitions)
        function={'name':'read_file','parameters':{'type':'object','properties':{'path':{'enum':['true','false']}}}}
        calls=[]
        parser=ToolFormat.new_parser(tool_options([function]).definitions,lambda *args:calls.append(args))
        parser.push(XML.replace('程序.py','true'));parser.finish()
        self.assertEqual(calls,[('read_file',{'path':'true'})])


class _Raw:
    def push(self, token): return chr(token)
    def finish(self): return ''


class _Codec:
    stop_ids = frozenset()
    def __init__(self): self.messages = []; self.prefix = ''
    def encode_messages(self, messages, *, tools=None, **_):
        self.messages = messages
        self.prefix = ToolFormat.generation_prefix(tools.choice, tools.name) if tools else ''
        return [1, 2]
    encode_tool_messages = encode_messages
    def new_decoder(self): return _Raw()
    def new_tool_decoder(self, tools, on_call, incomplete, on_text, filter_text):
        parser = ToolFormat.new_parser(tools.definitions, on_call, prefix=self.prefix, on_text=on_text,
            required=tools.choice in ('required', 'named'), max_calls=None if tools.parallel else 1)
        return _ToolDecodeSession(_Raw(), parser, incomplete, filter_text)


class _Worker:
    context = 4096
    stop_ids = frozenset({3})
    healthy = True
    def __init__(self, codec): self.codec, self.text, self.requests = codec, XML, []
    def submit(self, ids, limit, **options):
        text = self.text
        if self.codec.prefix:
            assert text.startswith(self.codec.prefix)
            text = text[len(self.codec.prefix):]
        events = [TokenEvent('token', ord(c)) for c in text[:limit]]
        if len(text)<limit: events.append(TokenEvent('token', 3))
        events.append(TokenEvent('completed'))
        request = Request(events); self.requests.append(request)
        return request


SDK_AVAILABLE = all(importlib.util.find_spec(name) for name in ('openai', 'anthropic', 'httpx'))


@unittest.skipUnless(SDK_AVAILABLE, 'official SDKs are optional; grammar contracts still run')
class ToolSdkTest(unittest.TestCase):
    def setUp(self):
        import anthropic, httpx, openai
        self.codec = _Codec(); self.worker = _Worker(self.codec)
        self.engine = TextEngine(self.worker, self.codec, 'hyle')
        self.server = TextApiServer(('127.0.0.1', 0), self.engine, True)
        self.thread = threading.Thread(target=self.server.serve_forever); self.thread.start()
        base = f'http://127.0.0.1:{self.server.server_port}'
        self.openai = openai.OpenAI(base_url=base+'/v1', api_key='local-test', max_retries=0,
            http_client=httpx.Client(trust_env=False), _strict_response_validation=True)
        self.anthropic = anthropic.Anthropic(base_url=base, api_key='local-test', max_retries=0,
            http_client=httpx.Client(trust_env=False), _strict_response_validation=True)
        self.user = {'role': 'user', 'content': 'read 程序.py'}
        self.chat_tools = [{'type': 'function', 'function': FUNCTION}]
        self.response_tools = [{'type': 'function', **FUNCTION}]
        self.anthropic_tools = [{'name': FUNCTION['name'], 'description': FUNCTION['description'], 'input_schema': FUNCTION['parameters']}]

    def tearDown(self):
        self.openai.close(); self.anthropic.close()
        self.server.shutdown(); self.thread.join(timeout=3); self.server.server_close()

    def response(self, stream=False, **options):
        parameters = dict(model='hyle', input=[self.user], max_output_tokens=256, store=False, tools=self.response_tools)
        parameters.update(options)
        if not stream: return self.openai.responses.create(**parameters)
        with self.openai.responses.stream(**parameters) as events:
            list(events); return events.get_final_response()

    def chat(self, stream=False, **options):
        parameters = dict(model='hyle', messages=[self.user], max_tokens=256, tools=self.chat_tools)
        parameters.update(options)
        if not stream: return self.openai.chat.completions.create(**parameters)
        with self.openai.chat.completions.stream(**parameters) as events:
            list(events); return events.get_final_completion()

    def message(self, stream=False, **options):
        parameters = dict(model='hyle', messages=[self.user], max_tokens=256, tools=self.anthropic_tools)
        parameters.update(options)
        if not stream: return self.anthropic.messages.create(**parameters)
        with self.anthropic.messages.stream(**parameters) as events:
            list(events); return events.get_final_message()

    def test_all_three_sdk_helpers_preserve_call_ids_arguments_and_results(self):
        for stream in (False, True):
            with self.subTest(api='responses', stream=stream):
                self.worker.text = XML
                result = self.response(stream)
                call = next(item for item in result.output if item.type=='function_call')
                self.assertEqual((call.name, json.loads(call.arguments)), ('read_file', {'path':'程序.py'}))
                self.worker.text = 'File contains a function.'
                reply = self.response(stream, input=[self.user,
                    *(item.model_dump(exclude_none=True) for item in result.output),
                    {'type':'function_call_output','call_id':call.call_id,'output':'def greet(): pass'}])
                self.assertEqual(reply.output_text, self.worker.text)
            with self.subTest(api='chat', stream=stream):
                self.worker.text = XML
                result = self.chat(stream); message = result.choices[0].message
                self.assertEqual(result.choices[0].finish_reason, 'tool_calls')
                call = message.tool_calls[0]
                self.assertEqual(json.loads(call.function.arguments), {'path':'程序.py'})
                self.worker.text = 'File contains a function.'
                reply = self.chat(stream, messages=[self.user, message.model_dump(exclude_none=True),
                    {'role':'tool','tool_call_id':call.id,'content':'def greet(): pass'}])
                self.assertEqual(reply.choices[0].message.content, self.worker.text)
            with self.subTest(api='anthropic', stream=stream):
                self.worker.text = XML
                result = self.message(stream)
                self.assertEqual(result.stop_reason, 'tool_use')
                call = next(block for block in result.content if block.type=='tool_use')
                self.assertEqual(call.input, {'path':'程序.py'})
                self.worker.text = 'File contains a function.'
                reply = self.message(stream, messages=[self.user,
                    {'role':'assistant','content':[b.model_dump(exclude_none=True) for b in result.content]},
                    {'role':'user','content':[{'type':'tool_result','tool_use_id':call.id,'content':'def greet(): pass'}]}])
                self.assertEqual(reply.content[0].text, self.worker.text)
        self.assertTrue(all(r.closed.is_set() for r in self.worker.requests))

    def test_named_and_required_choices_use_the_prepared_prompt_prefix(self):
        for stream in (False, True):
            self.worker.text = XML
            for choice in ('required', {'type':'function','name':'read_file'}):
                result = self.response(stream, tool_choice=choice)
                self.assertEqual(next(i for i in result.output if i.type=='function_call').name, 'read_file')
            for choice in ('required', {'type':'function','function':{'name':'read_file'}}):
                self.assertEqual(self.chat(stream, tool_choice=choice).choices[0].message.tool_calls[0].function.name,'read_file')
            for choice in ({'type':'any'}, {'type':'tool','name':'read_file','disable_parallel_tool_use':True}):
                self.assertEqual(next(b for b in self.message(stream,tool_choice=choice).content if b.type=='tool_use').name,'read_file')

    def test_truncated_arguments_never_become_a_successful_tool(self):
        self.assertEqual(self.chat(max_tokens=24).choices[0].finish_reason, 'length')
        self.assertIsNone(self.chat(max_tokens=24).choices[0].message.tool_calls)
        self.assertEqual(self.response(max_output_tokens=24).status, 'incomplete')
        self.assertEqual(self.message(max_tokens=24).stop_reason, 'max_tokens')

    def test_multiple_calls_have_distinct_ids_and_ordered_stream_indices(self):
        self.worker.text = XML + '\n' + XML
        for stream in (False, True):
            response = self.response(stream)
            calls = [i for i in response.output if i.type=='function_call']
            self.assertEqual(len(calls), 2); self.assertNotEqual(calls[0].call_id,calls[1].call_id)
            calls = self.chat(stream).choices[0].message.tool_calls
            self.assertEqual(len(calls), 2); self.assertNotEqual(calls[0].id,calls[1].id)
            calls = [b for b in self.message(stream).content if b.type=='tool_use']
            self.assertEqual(len(calls), 2); self.assertNotEqual(calls[0].id,calls[1].id)

    def test_stop_in_arguments_retires_without_publishing_a_call(self):
        result = self.chat(stop='程序')
        self.assertEqual(result.choices[0].finish_reason,'stop')
        self.assertIsNone(result.choices[0].message.tool_calls)
        result = self.message(stop_sequences=['程序'])
        self.assertEqual((result.stop_reason,result.stop_sequence),('stop_sequence','程序'))
        self.assertFalse(any(b.type=='tool_use' for b in result.content))

    def test_strict_and_orphan_results_reject_without_native_work(self):
        import openai, anthropic
        with self.assertRaises(openai.BadRequestError):
            self.response(tools=[{'type':'function',**FUNCTION,'strict':True}])
        with self.assertRaises(openai.BadRequestError):
            self.chat(messages=[{'role':'tool','tool_call_id':'missing','content':'x'}])
        with self.assertRaises(anthropic.BadRequestError):
            self.message(messages=[{'role':'user','content':[{'type':'tool_result','tool_use_id':'missing','content':'x'}]}])
        self.assertEqual(self.worker.requests, [])


if __name__ == '__main__': unittest.main()
