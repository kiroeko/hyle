"""Anthropic Messages text request, response and named-event SSE adapter."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .completions import (CompletionInput, finish_reason, output_limit, request_object,
                          serve_completion, stop_sequences, streaming, text_content)
from .media import content_parts
from .tools import history_call, tool_options, validate_tool_history


@dataclass(frozen=True)
class MessagesPresentation:
    stream: bool
    id: str
    model: str

    def message(self, text, input_tokens, output_tokens, reason=None, sequence=None, calls=()):
        import json
        return {'id': self.id, 'type': 'message', 'role': 'assistant', 'model': self.model,
                'content': ([{'type': 'text', 'text': text}] + [{'type': 'tool_use', 'id': call.id,
                    'name': call.name, 'input': json.loads(call.arguments)} for call in calls]) if reason is not None else [],
                'stop_reason': reason, 'stop_sequence': sequence,
                'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}}

    def start(self, exchange, engine, prepared):
        exchange.send_event('message_start', {'type': 'message_start', 'message': self.message('', len(prepared.ids), 0)})
        exchange.send_event('content_block_start', {'type': 'content_block_start', 'index': 0,
                                                   'content_block': {'type': 'text', 'text': ''}})

    def delta(self, exchange, text):
        exchange.send_event('content_block_delta', {'type': 'content_block_delta', 'index': 0,
                                                   'delta': {'type': 'text_delta', 'text': text}})

    def tool_call(self, exchange, call, index):
        if index == 0: exchange.send_event('content_block_stop', {'type': 'content_block_stop', 'index': 0})
        index += 1
        exchange.send_event('content_block_start', {'type': 'content_block_start', 'index': index,
            'content_block': {'type': 'tool_use', 'id': call.id, 'name': call.name, 'input': {}}})
        exchange.send_event('content_block_delta', {'type': 'content_block_delta', 'index': index,
            'delta': {'type': 'input_json_delta', 'partial_json': call.arguments}})
        exchange.send_event('content_block_stop', {'type': 'content_block_stop', 'index': index})

    def finish(self, exchange, engine, prepared, text, progress):
        reason = 'stop_sequence' if progress.stop_sequence is not None else (
            'max_tokens' if finish_reason(engine, prepared, progress) == 'length' else 'end_turn')
        if progress.tool_calls and reason == 'end_turn': reason = 'tool_use'
        if self.stream:
            if not progress.tool_calls: exchange.send_event('content_block_stop', {'type': 'content_block_stop', 'index': 0})
            exchange.send_event('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': reason,
                'stop_sequence': progress.stop_sequence}, 'usage': {'output_tokens': progress.count}})
            exchange.send_event('message_stop', {'type': 'message_stop'})
        else:
            exchange.send_json(200, self.message(text, len(prepared.ids), progress.count, reason,
                progress.stop_sequence, progress.tool_calls))


def _messages(value):
    if not isinstance(value, list) or not value: raise ValueError('messages must be a nonempty list')
    messages = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {'role', 'content'} or item['role'] not in ('user', 'assistant'):
            raise ValueError('messages require user or assistant role and content')
        role, content = item['role'], item['content']
        if isinstance(content, str): messages.append({'role': role, 'content': content}); continue
        if not isinstance(content, list) or not content: raise ValueError('message content must be text or blocks')
        ordinary, calls, saw_ordinary = [], [], False
        for block in content:
            if not isinstance(block, dict): raise ValueError('content blocks must be objects')
            kind = block.get('type')
            if kind == 'tool_use':
                if role != 'assistant' or set(block) - {'type', 'id', 'name', 'input', 'cache_control'}:
                    raise ValueError('invalid tool_use block')
                calls.append(history_call(block.get('id'), block.get('name'), block.get('input')))
            elif kind == 'tool_result':
                if role != 'user' or saw_ordinary or set(block) - {'type', 'tool_use_id', 'content', 'is_error', 'cache_control'}:
                    raise ValueError('tool results must precede user text')
                error = block.get('is_error', False)
                if type(error) is not bool: raise ValueError('tool is_error must be a boolean')
                result = text_content(block.get('content', ''))
                messages.append({'role': 'tool', 'tool_call_id': block.get('tool_use_id'),
                    'content': ('Tool error: ' if error else '') + result})
            else:
                if calls: raise ValueError('this model format requires assistant text before tool calls')
                ordinary.append(block); saw_ordinary = True
        if ordinary or calls:
            normalized = {'role': role, 'content': content_parts(ordinary, protocol='anthropic', role=role) if ordinary else ''}
            if calls: normalized['tool_calls'] = calls
            messages.append(normalized)
    validate_tool_history(messages)
    return messages


def _tools(payload):
    source = payload.get('tools')
    definitions = None
    if source is not None:
        if not isinstance(source, list): raise ValueError('tools must be a list')
        definitions = []
        for tool in source:
            if not isinstance(tool, dict) or set(tool) - {'name', 'description', 'input_schema', 'strict', 'cache_control'}:
                raise ValueError('only client function tools are supported')
            definitions.append({'name': tool.get('name'), 'description': tool.get('description'),
                'parameters': tool.get('input_schema'), 'strict': tool.get('strict')})
    choice = payload.get('tool_choice')
    if choice is None: choice = {'type': 'auto'}
    if not isinstance(choice, dict) or set(choice) - {'type', 'name', 'disable_parallel_tool_use'}:
        raise ValueError('invalid tool_choice')
    kind = choice.get('type')
    if not isinstance(kind, str): raise ValueError('tool choice type must be a string')
    mode = {'auto': 'auto', 'none': 'none', 'any': 'required', 'tool': 'named'}.get(kind)
    if mode is None: raise ValueError('unsupported tool choice')
    disable = choice.get('disable_parallel_tool_use', False)
    if type(disable) is not bool: raise ValueError('disable_parallel_tool_use must be a boolean')
    return tool_options(definitions, choice=mode, name=choice.get('name'), parallel=not disable)


def parse_messages(payload, model_name):
    request_object(payload, model_name, {'model', 'messages', 'system', 'max_tokens', 'stream',
        'temperature', 'top_p', 'top_k', 'stop_sequences', 'hyle', 'tools', 'tool_choice'})
    if 'max_tokens' not in payload: raise ValueError('max_tokens is required')
    messages = _messages(payload.get('messages'))
    if 'system' in payload: messages.insert(0, {'role': 'system', 'content': text_content(payload['system'])})
    stops = payload.get('stop_sequences')
    if stops is not None and not isinstance(stops, list): raise ValueError('stop_sequences must be a list')
    inputs = CompletionInput(messages, output_limit(payload['max_tokens']), streaming(payload.get('stream', False)),
        payload.get('hyle'), payload.get('temperature'), payload.get('top_p'), payload.get('top_k'),
        stop_sequences(stops), continue_final=messages[-1]['role'] == 'assistant', tools=_tools(payload))
    return inputs, MessagesPresentation(inputs.stream, 'msg_' + uuid.uuid4().hex, model_name)


class AnthropicMessagesAdapter:
    @staticmethod
    def error(exchange, status, message, started):
        value = {'type': 'error', 'error': {'type': 'invalid_request_error' if status < 500 else 'api_error', 'message': message}}
        if started: exchange.send_event('error', value)
        else: exchange.send_json(status, value)

    def handle_post(self, exchange, engine):
        serve_completion(exchange, engine, parse_messages, self.error)
