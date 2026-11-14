"""Bounded text completion lifecycle shared by stateless HTTP protocols."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .contracts import TokenRequest, WorkerError, ToolOptions
from .runtime import GenerationProgress, PreparedText, TextEngine
from .server import ClientDisconnected, HttpExchange
from .media import content_parts
from .tools import history_call, validate_tool_history


@dataclass(frozen=True)
class CompletionInput:
    messages: list[dict[str, Any]]
    limit: int
    stream: bool
    selection: dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stops: tuple[str, ...] = ()
    continue_final: bool = False
    tools: ToolOptions | None = None


class CompletionPresentation(Protocol):
    def start(self, exchange: HttpExchange, engine: TextEngine, prepared: PreparedText) -> None: ...
    def delta(self, exchange: HttpExchange, text: str) -> None: ...
    def finish(self, exchange: HttpExchange, engine: TextEngine, prepared: PreparedText,
               text: str, progress: GenerationProgress) -> None: ...


def request_object(payload: Any, model_name: str, supported: set[str]) -> dict:
    if not isinstance(payload, dict): raise ValueError('request body must be a JSON object')
    unknown = set(payload) - supported
    if unknown: raise ValueError('unsupported request fields: ' + ', '.join(sorted(unknown)))
    if payload.get('model') != model_name: raise ValueError('model does not match the loaded model')
    return payload


def text_content(value: Any) -> str:
    if isinstance(value, str): return value
    if not isinstance(value, list): raise ValueError('content must be text or a text block list')
    parts = []
    for part in value:
        if (not isinstance(part, dict) or set(part) != {'type', 'text'} or
                part['type'] != 'text' or not isinstance(part['text'], str)):
            raise ValueError('only text content blocks are supported')
        parts.append(part['text'])
    return ''.join(parts)


def message_list(value: Any, roles: tuple[str, ...], *, protocol='chat') -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value: raise ValueError('messages must be a nonempty list')
    messages = []
    for message in value:
        allowed = {'role', 'content'} | ({'tool_calls', 'tool_call_id', 'name', 'refusal'} if protocol == 'chat' else set())
        if not isinstance(message, dict) or set(message) - allowed or message.get('role') not in roles:
            raise ValueError('messages require a supported role and content')
        role, calls = message['role'], message.get('tool_calls')
        content = message.get('content')
        if content is None and role == 'assistant' and calls: content = ''
        normalized = {'role': role, 'content': content_parts(content, protocol=protocol, role=role)}
        if calls is not None:
            if role != 'assistant' or not isinstance(calls, list) or not calls:
                raise ValueError('assistant tool_calls must be a nonempty list')
            normalized['tool_calls'] = []
            for index, call in enumerate(calls):
                if (not isinstance(call, dict) or not {'id', 'type', 'function'} <= call.keys() or
                        set(call) - {'id', 'type', 'function', 'index'} or call.get('type') != 'function' or
                        not isinstance(call['function'], dict) or set(call['function']) != {'name', 'arguments'}):
                    raise ValueError('invalid assistant tool call')
                if 'index' in call and (type(call['index']) is not int or call['index'] != index):
                    raise ValueError('tool call index does not match its history position')
                normalized['tool_calls'].append(history_call(call['id'], call['function']['name'], call['function']['arguments']))
        if role == 'tool': normalized['tool_call_id'] = message.get('tool_call_id')
        elif 'tool_call_id' in message: raise ValueError('only tool results accept tool_call_id')
        if message.get('name') is not None and not isinstance(message['name'], str): raise ValueError('message name must be a string')
        if message.get('refusal') is not None: raise ValueError('refusal history is not supported')
        messages.append(normalized)
    if protocol == 'chat': validate_tool_history(messages)
    return messages


def output_limit(value: Any) -> int:
    if type(value) is not int or value <= 0: raise ValueError('output token limit must be a positive integer')
    return value


def streaming(value: Any) -> bool:
    if type(value) is not bool: raise ValueError('stream must be a boolean')
    return value


def stop_sequences(value: Any, *, maximum: int | None = None) -> tuple[str, ...]:
    if value is None: return ()
    if isinstance(value, str): value = [value]
    if not isinstance(value, list) or any(not isinstance(stop, str) or not stop for stop in value):
        raise ValueError('stop sequences must be nonempty strings')
    if maximum is not None and len(value) > maximum: raise ValueError('too many stop sequences')
    return tuple(value)


def finish_reason(engine: TextEngine, prepared: PreparedText, progress: GenerationProgress) -> str:
    return 'stop' if progress.stop_sequence is not None or progress.last in engine.stop_ids or progress.count < prepared.limit else 'length'


def invalid_request(error: Exception, *, validation: bool = True) -> bool:
    return (validation and isinstance(error, (ValueError, UnicodeError))) or (
        isinstance(error, WorkerError) and error.kind in ('invalid', 'admission'))


def _close_request(request: TokenRequest) -> None:
    try:
        retire = getattr(request, 'retire', None)
        if retire is not None:
            event = retire(timeout=120)
            if event.kind not in ('completed', 'cancelled'):
                raise WorkerError(event.error_kind or 'retirement', event.message or 'native request did not retire successfully')
    finally: request.close()


def serve_completion(exchange: HttpExchange, engine: TextEngine,
    parse: Callable[[Any, str], tuple[CompletionInput, CompletionPresentation]],
    send_error: Callable[[HttpExchange, int, str, bool], None]) -> None:
    """Protocol formatters own JSON/SSE; the original engine owns all work."""
    try:
        inputs, presentation = parse(exchange.read_json(), engine.model_name)
        selection = engine.choices.select(inputs.selection, temperature=inputs.temperature,
                                          top_p=inputs.top_p, top_k=inputs.top_k)
        prepared = engine.prepare(inputs.messages, inputs.limit, selection, continue_final_message=inputs.continue_final,
            **({'tools': inputs.tools} if inputs.tools is not None else {}))
        selection = prepared.selection
    except (ValueError, UnicodeError, WorkerError) as error:
        send_error(exchange, 400 if invalid_request(error) else 503, str(error), False)
        return
    request = None
    progress = GenerationProgress()
    success = started = closed = False
    engine.metrics.begin()
    try:
        # Use the same admission/KV budget and trusted prefix-sharing policy as
        # Responses, without creating stored protocol records or session IDs.
        request = engine.sessions.start('req_' + uuid.uuid4().hex, inputs.messages, prepared.ids, prepared.limit,
            selection, save=False, **({'attachments': prepared.attachments} if prepared.attachments else {})) if engine.sessions else engine.submit(prepared)
        if inputs.stream:
            exchange.begin_stream(); started = True
            presentation.start(exchange, engine, prepared)
        text = engine.generate(request, (lambda chunk: presentation.delta(exchange, chunk)) if inputs.stream else (lambda _: None),
            progress, lambda: exchange.next_event(request), stop_sequences=inputs.stops, prepared=prepared,
            on_tool_call=(lambda call: presentation.tool_call(exchange, call, len(progress.tool_calls)-1)) if inputs.stream else None)
        # Custom text stops may precede the native terminal. Confirm retirement
        # before announcing protocol success; no unused KV or RNG is saved.
        closed = True
        _close_request(request)
        presentation.finish(exchange, engine, prepared, text, progress)
        success = True
    except (ClientDisconnected, BrokenPipeError, ConnectionResetError, OSError):
        pass
    except Exception as error:
        try: send_error(exchange, 400 if invalid_request(error, validation=request is None) else 503, str(error), started)
        except (BrokenPipeError, ConnectionResetError, OSError): pass
    finally:
        try:
            if request is not None and not closed: _close_request(request)
        finally: engine.metrics.end(success, progress.count)
