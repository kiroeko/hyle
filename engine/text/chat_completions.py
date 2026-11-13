"""OpenAI Chat Completions text request, response and data-only SSE adapter."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from .completions import (CompletionInput, finish_reason, message_list, output_limit,
                          request_object, serve_completion, stop_sequences, streaming)


@dataclass(frozen=True)
class ChatPresentation:
    stream: bool
    include_usage: bool
    id: str
    created: int
    model: str

    def chunk(self, delta=None, *, reason=None, usage=None, usage_only=False):
        value = {'id': self.id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model,
                 'choices': [] if usage_only else [{'index': 0, 'delta': delta or {}, 'finish_reason': reason, 'logprobs': None}]}
        if self.include_usage: value['usage'] = usage
        return value

    def start(self, exchange, engine, prepared):
        exchange.send_data(self.chunk({'role': 'assistant', 'content': ''}))

    def delta(self, exchange, text):
        exchange.send_data(self.chunk({'content': text}))

    def finish(self, exchange, engine, prepared, text, progress):
        reason = finish_reason(engine, prepared, progress)
        usage = {'prompt_tokens': len(prepared.ids), 'completion_tokens': progress.count,
                 'total_tokens': len(prepared.ids) + progress.count}
        if self.stream:
            exchange.send_data(self.chunk(reason=reason))
            if self.include_usage: exchange.send_data(self.chunk(usage=usage, usage_only=True))
            exchange.send_data('[DONE]')
        else:
            exchange.send_json(200, {'id': self.id, 'object': 'chat.completion', 'created': self.created,
                'model': self.model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text,
                'refusal': None}, 'finish_reason': reason, 'logprobs': None}], 'usage': usage})


def parse_chat_completion(payload, model_name):
    request_object(payload, model_name, {'model', 'messages', 'max_tokens', 'max_completion_tokens', 'stream',
        'stream_options', 'temperature', 'top_p', 'seed', 'stop', 'n', 'store', 'hyle'})
    if payload.get('max_tokens') is not None and payload.get('max_completion_tokens') is not None:
        raise ValueError('choose max_tokens or max_completion_tokens')
    limit = output_limit(payload.get('max_completion_tokens') if payload.get('max_completion_tokens') is not None else
                         payload.get('max_tokens') if payload.get('max_tokens') is not None else 256)
    stream = streaming(False if payload.get('stream') is None else payload['stream'])
    if payload.get('n') is not None and (type(payload['n']) is not int or payload['n'] != 1):
        raise ValueError('only n=1 is supported')
    if payload.get('store') is not None and payload['store'] is not False:
        raise ValueError('Chat Completions storage is not supported')
    options = payload.get('stream_options')
    if options is not None and (not stream or not isinstance(options, dict) or set(options) - {'include_usage', 'include_obfuscation'}):
        raise ValueError('stream_options requires streaming and supported options')
    options = options or {}
    if type(options.get('include_usage', False)) is not bool:
        raise ValueError('include_usage must be a boolean')
    if 'include_obfuscation' in options and options['include_obfuscation'] is not False:
        raise ValueError('stream obfuscation is not supported')
    values = payload.get('hyle')
    if 'seed' in payload and payload['seed'] is not None:
        seed = payload['seed']
        if type(seed) is not int or not -(1 << 63) <= seed < 1 << 63:
            raise ValueError('seed must be a signed 64-bit integer')
        if values is not None and not isinstance(values, dict): raise ValueError('hyle must be an object')
        values = dict(values or {})
        seed &= (1 << 64) - 1
        if values.get('seed') is not None and (type(values['seed']) is not int or values['seed'] != seed):
            raise ValueError('seed conflicts with hyle.seed')
        values['seed'] = seed
    inputs = CompletionInput(message_list(payload.get('messages'), ('developer', 'system', 'user', 'assistant')),
        limit, stream, values, payload.get('temperature'), payload.get('top_p'), stops=stop_sequences(payload.get('stop'), maximum=4))
    return inputs, ChatPresentation(stream, options.get('include_usage', False), 'chatcmpl-' + uuid.uuid4().hex,
                                    int(time.time()), model_name)


class ChatCompletionsAdapter:
    @staticmethod
    def error(exchange, status, message, started):
        value = {'error': {'message': message, 'type': 'invalid_request_error' if status < 500 else 'server_error',
                           'param': None, 'code': None}}
        if started: exchange.send_data(value)
        else: exchange.send_json(status, value)

    def handle_post(self, exchange, engine):
        serve_completion(exchange, engine, parse_chat_completion, self.error)
