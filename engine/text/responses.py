"""Text-only OpenAI Responses request and event adapter."""
from __future__ import annotations

import time
import uuid
from typing import Any

from .contracts import TokenRequest, WorkerError
from .runtime import GenerationProgress, PreparedText, TextEngine
from .server import ClientDisconnected, HttpExchange


class InvalidRequest(ValueError):
    pass


def response_error_kind(error: Exception) -> str:
    return 'invalid_request_error' if isinstance(error, WorkerError) and error.kind == 'invalid' else 'server_error'


def parse_responses_input(payload: Any, model_name: str) -> tuple[list[dict[str, str]], int, bool, str | None]:
    if not isinstance(payload, dict):
        raise InvalidRequest('request body must be a JSON object')
    supported = {'model', 'input', 'instructions', 'max_output_tokens', 'stream', 'store'}
    unknown = set(payload) - supported
    if unknown:
        raise InvalidRequest('unsupported request fields: ' + ', '.join(sorted(unknown)))
    if payload.get('model') != model_name:
        raise InvalidRequest('model does not match the loaded model')
    if 'input' not in payload:
        raise InvalidRequest('input is required')
    stream = payload.get('stream', False)
    if type(stream) is not bool:
        raise InvalidRequest('stream must be a boolean')
    if 'store' in payload and payload['store'] is not False:
        raise InvalidRequest('stored responses are not supported; store must be false')
    limit = payload.get('max_output_tokens', 256)
    if type(limit) is not int or limit <= 0:
        raise InvalidRequest('max_output_tokens must be a positive integer')
    instructions = payload.get('instructions')
    if instructions is not None and not isinstance(instructions, str):
        raise InvalidRequest('instructions must be a string')
    messages: list[dict[str, str]] = []
    if instructions is not None:
        messages.append({'role': 'system', 'content': instructions})
    source = payload['input']
    if isinstance(source, str):
        messages.append({'role': 'user', 'content': source})
    elif isinstance(source, list) and source:
        for item in source:
            if not isinstance(item, dict) or set(item) - {'type', 'role', 'content'}:
                raise InvalidRequest('input items must be text messages')
            if item.get('type', 'message') != 'message' or item.get('role') not in ('system', 'user', 'assistant'):
                raise InvalidRequest('only system, user and assistant text messages are supported')
            content = item.get('content')
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if (not isinstance(part, dict) or set(part) != {'type', 'text'} or
                            part['type'] not in ('input_text', 'output_text') or not isinstance(part['text'], str)):
                        raise InvalidRequest('only text content parts are supported')
                    parts.append(part['text'])
                text = ''.join(parts)
            else:
                raise InvalidRequest('message content must be text')
            messages.append({'role': item['role'], 'content': text})
    else:
        raise InvalidRequest('input must be a string or a nonempty message list')
    return messages, limit, stream, instructions


def response_object(response_id: str, item_id: str, model: str, created: int, prepared: PreparedText,
                    status: str, text: str, output_tokens: int, instructions: str | None) -> dict[str, Any]:
    part = {'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}
    item = {'id': item_id, 'type': 'message', 'status': status, 'role': 'assistant', 'content': [part]}
    usage = {'input_tokens': len(prepared.ids), 'input_tokens_details': {'cached_tokens': 0},
             'output_tokens': output_tokens, 'output_tokens_details': {'reasoning_tokens': 0},
             'total_tokens': len(prepared.ids) + output_tokens}
    return {'id': response_id, 'object': 'response', 'created_at': created, 'status': status,
            'error': None, 'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
            'instructions': instructions, 'max_output_tokens': prepared.limit, 'model': model,
            'output': [item] if status != 'in_progress' else [], 'parallel_tool_calls': False,
            'previous_response_id': None, 'reasoning': {'effort': None, 'summary': None},
            'store': False, 'text': {'format': {'type': 'text'}}, 'tool_choice': 'none', 'tools': [],
            'truncation': 'disabled', 'usage': usage if status != 'in_progress' else None, 'metadata': {}}


class ResponsesAdapter:
    """Convert Responses JSON and SSE to the shared text-request contract."""
    @staticmethod
    def _error(exchange: HttpExchange, status: int, message: str) -> None:
        exchange.send_json(status, {'error': {'message': message,
            'type': 'invalid_request_error' if status < 500 else 'server_error', 'param': None, 'code': None}})

    def handle_post(self, exchange: HttpExchange, engine: TextEngine) -> None:
        try:
            messages, limit, streaming, instructions = parse_responses_input(exchange.read_json(), engine.model_name)
            prepared = engine.prepare(messages, limit)
        except (ValueError, UnicodeError) as error:
            self._error(exchange, 400, str(error)); return
        engine.metrics.begin()
        try:
            request = engine.submit(prepared)
        except Exception as error:
            engine.metrics.end(False, 0)
            invalid = isinstance(error, ValueError) or response_error_kind(error) == 'invalid_request_error'
            self._error(exchange, 400 if invalid else 503, str(error)); return
        if streaming:
            self._stream(exchange, engine, prepared, request, instructions)
        else:
            self._complete(exchange, engine, prepared, request, instructions)

    def _complete(self, exchange: HttpExchange, engine: TextEngine, prepared: PreparedText,
                  request: TokenRequest, instructions: str | None) -> None:
        progress = GenerationProgress(); success = False
        try:
            text = engine.generate(request, lambda _: None, progress, lambda: exchange.next_event(request))
            count, last = progress.count, progress.last
            status = 'incomplete' if count == prepared.limit and last not in engine.stop_ids else 'completed'
            result = response_object('resp_' + uuid.uuid4().hex, 'msg_' + uuid.uuid4().hex,
                                     engine.model_name, int(time.time()), prepared, status, text, count, instructions)
            exchange.send_json(200, result); success = True
        except ClientDisconnected:
            pass
        except Exception as error:
            try: self._error(exchange, 400 if response_error_kind(error) == 'invalid_request_error' else 503, str(error))
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally:
            request.close(); engine.metrics.end(success, progress.count)

    def _stream(self, exchange: HttpExchange, engine: TextEngine, prepared: PreparedText,
                request: TokenRequest, instructions: str | None) -> None:
        response_id, item_id, created = 'resp_' + uuid.uuid4().hex, 'msg_' + uuid.uuid4().hex, int(time.time())
        sequence = 0; progress = GenerationProgress(); success = False
        def send(kind: str, **values: Any) -> None:
            nonlocal sequence
            event = {'type': kind, 'sequence_number': sequence, **values}; sequence += 1
            exchange.send_event(kind, event)
        try:
            exchange.begin_stream()
            empty = response_object(response_id, item_id, engine.model_name, created, prepared,
                                    'in_progress', '', 0, instructions)
            send('response.created', response=empty)
            send('response.in_progress', response=empty)
            send('response.output_item.added', output_index=0,
                 item={'id': item_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []})
            send('response.content_part.added', item_id=item_id, output_index=0, content_index=0,
                 part={'type': 'output_text', 'text': '', 'annotations': [], 'logprobs': []})
            def on_text(chunk: str) -> None:
                send('response.output_text.delta', item_id=item_id, output_index=0, content_index=0,
                     delta=chunk, logprobs=[])
            text = engine.generate(request, on_text, progress, lambda: exchange.next_event(request))
            count, last = progress.count, progress.last
            status = 'incomplete' if count == prepared.limit and last not in engine.stop_ids else 'completed'
            result = response_object(response_id, item_id, engine.model_name, created, prepared,
                                     status, text, count, instructions)
            send('response.output_text.done', item_id=item_id, output_index=0, content_index=0, text=text, logprobs=[])
            send('response.content_part.done', item_id=item_id, output_index=0, content_index=0,
                 part=result['output'][0]['content'][0])
            send('response.output_item.done', output_index=0, item=result['output'][0])
            send('response.' + status, response=result); success = True
        except (ClientDisconnected, BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as error:
            try: send('error', code=response_error_kind(error), message=str(error), param=None)
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally:
            request.close(); engine.metrics.end(success, progress.count)
