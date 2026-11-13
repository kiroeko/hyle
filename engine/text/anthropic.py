"""Anthropic Messages text request, response and named-event SSE adapter."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .completions import (CompletionInput, finish_reason, message_list, output_limit, request_object,
                          serve_completion, stop_sequences, streaming, text_content)


@dataclass(frozen=True)
class MessagesPresentation:
    stream: bool
    id: str
    model: str

    def message(self, text, input_tokens, output_tokens, reason=None, sequence=None):
        return {'id': self.id, 'type': 'message', 'role': 'assistant', 'model': self.model,
                'content': [{'type': 'text', 'text': text}] if reason is not None else [],
                'stop_reason': reason, 'stop_sequence': sequence,
                'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}}

    def start(self, exchange, engine, prepared):
        exchange.send_event('message_start', {'type': 'message_start', 'message': self.message('', len(prepared.ids), 0)})
        exchange.send_event('content_block_start', {'type': 'content_block_start', 'index': 0,
                                                   'content_block': {'type': 'text', 'text': ''}})

    def delta(self, exchange, text):
        exchange.send_event('content_block_delta', {'type': 'content_block_delta', 'index': 0,
                                                   'delta': {'type': 'text_delta', 'text': text}})

    def finish(self, exchange, engine, prepared, text, progress):
        reason = 'stop_sequence' if progress.stop_sequence is not None else (
            'max_tokens' if finish_reason(engine, prepared, progress) == 'length' else 'end_turn')
        if self.stream:
            exchange.send_event('content_block_stop', {'type': 'content_block_stop', 'index': 0})
            exchange.send_event('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': reason,
                'stop_sequence': progress.stop_sequence}, 'usage': {'output_tokens': progress.count}})
            exchange.send_event('message_stop', {'type': 'message_stop'})
        else:
            exchange.send_json(200, self.message(text, len(prepared.ids), progress.count, reason, progress.stop_sequence))


def parse_messages(payload, model_name):
    request_object(payload, model_name, {'model', 'messages', 'system', 'max_tokens', 'stream',
        'temperature', 'top_p', 'top_k', 'stop_sequences', 'hyle'})
    if 'max_tokens' not in payload: raise ValueError('max_tokens is required')
    messages = message_list(payload.get('messages'), ('user', 'assistant'), protocol='anthropic')
    if 'system' in payload: messages.insert(0, {'role': 'system', 'content': text_content(payload['system'])})
    stops = payload.get('stop_sequences')
    if stops is not None and not isinstance(stops, list): raise ValueError('stop_sequences must be a list')
    inputs = CompletionInput(messages, output_limit(payload['max_tokens']), streaming(payload.get('stream', False)),
        payload.get('hyle'), payload.get('temperature'), payload.get('top_p'), payload.get('top_k'),
        stop_sequences(stops), continue_final=messages[-1]['role'] == 'assistant')
    return inputs, MessagesPresentation(inputs.stream, 'msg_' + uuid.uuid4().hex, model_name)


class AnthropicMessagesAdapter:
    @staticmethod
    def error(exchange, status, message, started):
        value = {'type': 'error', 'error': {'type': 'invalid_request_error' if status < 500 else 'api_error', 'message': message}}
        if started: exchange.send_event('error', value)
        else: exchange.send_json(status, value)

    def handle_post(self, exchange, engine):
        serve_completion(exchange, engine, parse_messages, self.error)
