"""OpenAI Responses image/text input and text event adapter."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .contracts import TokenRequest, WorkerError
from .runtime import GenerationProgress, PreparedText, TextEngine
from .sessions import SessionNotFound, SessionTurn
from .server import ClientDisconnected, HttpExchange
from .tools import history_call, tool_options, validate_tool_history


class InvalidRequest(ValueError):
    pass


def response_error_kind(error: Exception) -> str:
    return 'invalid_request_error' if isinstance(error, WorkerError) and error.kind == 'invalid' else 'server_error'


def parse_responses_input(payload: Any, model_name: str) -> tuple[list[dict[str, Any]], int, bool, str | None]:
    if not isinstance(payload, dict):
        raise InvalidRequest('request body must be a JSON object')
    supported = {'model', 'input', 'instructions', 'max_output_tokens', 'stream', 'store',
                 'previous_response_id', 'temperature', 'top_p', 'hyle',
                 'tools', 'tool_choice', 'parallel_tool_calls', 'metadata'}
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
    if 'store' in payload and type(payload['store']) is not bool:
        raise InvalidRequest('store must be a boolean')
    previous = payload.get('previous_response_id')
    if previous is not None and (not isinstance(previous, str) or not previous or len(previous) > 128):
        raise InvalidRequest('previous_response_id must name a stored response')
    limit = payload.get('max_output_tokens', 256)
    if type(limit) is not int or limit <= 0:
        raise InvalidRequest('max_output_tokens must be a positive integer')
    instructions = payload.get('instructions')
    if instructions is not None and not isinstance(instructions, str):
        raise InvalidRequest('instructions must be a string')
    messages: list[dict[str, Any]] = []
    if instructions is not None:
        messages.append({'role': 'system', 'content': instructions})
    source = payload['input']
    if isinstance(source, str):
        messages.append({'role': 'user', 'content': source})
    elif isinstance(source, list) and source:
        for item in source:
            if not isinstance(item, dict): raise InvalidRequest('input items must be objects')
            if item.get('type') == 'function_call':
                if set(item) - {'id', 'type', 'call_id', 'name', 'arguments', 'status'}:
                    raise InvalidRequest('unsupported function call fields')
                call = history_call(item.get('call_id'), item.get('name'), item.get('arguments'))
                if messages and messages[-1]['role'] == 'assistant':
                    messages[-1].setdefault('tool_calls', []).append(call)
                else: messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
                continue
            if item.get('type') == 'function_call_output':
                if set(item) - {'id', 'type', 'call_id', 'output', 'status'} or not isinstance(item.get('output'), str):
                    raise InvalidRequest('function_call_output requires a text output')
                messages.append({'role': 'tool', 'tool_call_id': item.get('call_id'), 'content': item['output']})
                continue
            if set(item) - {'type', 'role', 'content', 'id', 'status'}:
                raise InvalidRequest('input items must be message objects')
            if item.get('type', 'message') != 'message' or item.get('role') not in ('developer', 'system', 'user', 'assistant'):
                raise InvalidRequest('unsupported message role')
            from .media import content_parts
            text = content_parts(item.get('content'), protocol='responses', role=item['role'])
            messages.append({'role': item['role'], 'content': text})
    else:
        raise InvalidRequest('input must be a string or a nonempty message list')
    if previous is None: validate_tool_history(messages)
    return messages, limit, stream, instructions


def parse_response_tools(payload):
    definitions = payload.get('tools')
    if definitions is not None:
        if not isinstance(definitions, list): raise InvalidRequest('tools must be a list')
        for definition in definitions:
            if not isinstance(definition, dict) or definition.get('type') != 'function':
                raise InvalidRequest('only client function tools are supported')
        definitions = [{key: value for key, value in definition.items() if key != 'type'} for definition in definitions]
    choice = payload.get('tool_choice')
    if choice is None: choice = 'auto'
    name = None
    if isinstance(choice, dict):
        if set(choice) != {'type', 'name'} or choice['type'] != 'function':
            raise InvalidRequest('invalid named function tool choice')
        choice, name = 'named', choice['name']
    return tool_options(definitions, choice=choice, name=name, parallel=payload.get('parallel_tool_calls', True))


def _call_item(call, status='completed'):
    return {'id': 'fc_' + call.id, 'type': 'function_call', 'status': status,
            'call_id': call.id, 'name': call.name, 'arguments': call.arguments if status == 'completed' else ''}


def _history(text, progress):
    return {'role': 'assistant', 'content': text, 'tool_calls': [call.history() for call in progress.tool_calls]}


def response_object(response_id: str, item_id: str, model: str, created: int, prepared: PreparedText,
                    status: str, text: str, output_tokens: int, instructions: str | None, *,
                    previous: str | None = None, store: bool = False, cached_tokens: int = 0,
                    metadata: dict | None = None, calls=()) -> dict[str, Any]:
    part = {'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}
    item = {'id': item_id, 'type': 'message', 'status': status, 'role': 'assistant', 'content': [part]}
    usage = {'input_tokens': len(prepared.ids), 'input_tokens_details': {'cached_tokens': cached_tokens},
             'output_tokens': output_tokens, 'output_tokens_details': {'reasoning_tokens': 0},
             'total_tokens': len(prepared.ids) + output_tokens}
    options = prepared.tools
    return {'id': response_id, 'object': 'response', 'created_at': created, 'status': status,
            'error': None, 'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
            'instructions': instructions, 'max_output_tokens': prepared.limit, 'model': model,
            'output': [item, *(_call_item(call) for call in calls)] if status != 'in_progress' else [],
            'parallel_tool_calls': options.parallel if options else False,
            'previous_response_id': previous, 'reasoning': {'effort': None, 'summary': None},
            'store': store, 'text': {'format': {'type': 'text'}},
            'tool_choice': ({'type': 'function', 'name': options.name} if options.choice == 'named' else options.choice) if options else 'none',
            'tools': [{'type': 'function', **d['function'], 'strict': False} for d in options.definitions] if options else [],
            'truncation': 'disabled', 'usage': usage if status != 'in_progress' else None, 'metadata': metadata or {}}


@dataclass(frozen=True)
class ResponseContext:
    response_id: str
    item_id: str
    created: int
    instructions: str | None
    previous: str | None
    store: bool
    metadata: dict

    def object(self, engine, prepared, request, status, text, count, calls=()):
        return response_object(self.response_id, self.item_id, engine.model_name, self.created, prepared,
            status, text, count, self.instructions, previous=self.previous, store=self.store,
            cached_tokens=getattr(request, 'cached_tokens', 0), metadata=self.metadata, calls=calls)


class ResponsesAdapter:
    """Convert Responses JSON and SSE to the shared text-request contract."""
    @staticmethod
    def _error(exchange: HttpExchange, status: int, message: str) -> None:
        exchange.send_json(status, {'error': {'message': message,
            'type': 'invalid_request_error' if status < 500 else 'server_error', 'param': None, 'code': None}})

    def handle_post(self, exchange: HttpExchange, engine: TextEngine) -> None:
        try:
            payload = exchange.read_json()
            messages, limit, streaming, instructions = parse_responses_input(payload, engine.model_name)
            tools = parse_response_tools(payload)
            user_metadata = payload.get('metadata')
            if user_metadata is None: user_metadata = {}
            if (not isinstance(user_metadata, dict) or len(user_metadata) > 16 or
                    any(not isinstance(k, str) or not isinstance(v, str) or len(k)>64 or len(v)>512 or k.startswith('hyle.')
                        for k,v in user_metadata.items())):
                raise InvalidRequest('metadata requires bounded string values and nonreserved keys')
            selection = engine.choices.select(payload.get('hyle'), temperature=payload.get('temperature'), top_p=payload.get('top_p'))
            previous = payload.get('previous_response_id')
            save = payload.get('store', engine.sessions is not None)
            if (save or previous is not None) and engine.sessions is None:
                raise InvalidRequest('stored continuations are not configured for this text owner')
            parent = engine.sessions.parent(previous) if previous is not None else None
            dialog = list(parent.messages) if parent else []
            dialog.extend(messages[1:] if instructions is not None else messages)
            validate_tool_history(dialog)
            prepared = engine.prepare(([messages[0]] if instructions is not None else []) + dialog, limit, selection,
                                      parent.tape if parent else None, **({'tools': tools} if tools is not None else {}))
            selection = prepared.selection
        except SessionNotFound as error:
            self._error(exchange, 404, str(error)); return
        except WorkerError as error:
            self._error(exchange, 503, str(error)); return
        except (ValueError, UnicodeError) as error:
            self._error(exchange, 400, str(error)); return
        engine.metrics.begin()
        try:
            response_id, item_id = 'resp_' + uuid.uuid4().hex, 'msg_' + uuid.uuid4().hex
            request = engine.sessions.start(response_id, dialog, prepared.ids, prepared.limit, selection,
                parent=parent, save=save, **({'attachments': prepared.attachments} if prepared.attachments else {})) if engine.sessions else engine.submit(prepared)
            metadata = {**user_metadata, 'hyle.execution_profile': engine.choices.executions[selection.execution_profile].name,
                        'hyle.sampling_profile': engine.choices.samplers[selection.sampling_index].name,
                        'hyle.context': 'continued' if getattr(request, 'cached_tokens', 0) else 'replayed' if parent else 'new'}
            context = ResponseContext(response_id, item_id, int(time.time()), instructions, previous, save, metadata)
        except Exception as error:
            engine.metrics.end(False, 0)
            invalid = isinstance(error, ValueError) or response_error_kind(error) == 'invalid_request_error'
            self._error(exchange, 400 if invalid else 503, str(error)); return
        if streaming:
            self._stream(exchange, engine, prepared, request, context)
        else:
            self._complete(exchange, engine, prepared, request, context)

    def handle_get(self, exchange: HttpExchange, engine: TextEngine, key: str) -> None:
        try:
            if engine.sessions is None: raise SessionNotFound('stored responses are not configured')
            exchange.send_json(200, engine.sessions.retrieve(key))
        except SessionNotFound as error: self._error(exchange, 404, str(error))
        except WorkerError as error: self._error(exchange, 503, str(error))

    def handle_delete(self, exchange: HttpExchange, engine: TextEngine, key: str) -> None:
        try:
            if engine.sessions is None: raise SessionNotFound('stored responses are not configured')
            engine.sessions.delete(key)
            exchange.send_json(200, {'id': key, 'object': 'response.deleted', 'deleted': True})
        except SessionNotFound as error: self._error(exchange, 404, str(error))
        except WorkerError as error: self._error(exchange, 503, str(error))

    def _complete(self, exchange: HttpExchange, engine: TextEngine, prepared: PreparedText,
                  request: TokenRequest, context: ResponseContext) -> None:
        progress = GenerationProgress(); success = False
        try:
            text = engine.generate(request, lambda _: None, progress, lambda: exchange.next_event(request), prepared=prepared)
            count, last = progress.count, progress.last
            status = 'incomplete' if count == prepared.limit and last not in engine.stop_ids else 'completed'
            result = context.object(engine, prepared, request, status, text, count, progress.tool_calls)
            if isinstance(request, SessionTurn): request.commit(text, result,
                **({'assistant_message': _history(text, progress)} if progress.tool_calls else {}))
            exchange.send_json(200, result); success = True
        except ClientDisconnected:
            pass
        except Exception as error:
            try: self._error(exchange, 400 if response_error_kind(error) == 'invalid_request_error' else 503, str(error))
            except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally:
            try: request.close()
            finally: engine.metrics.end(success, progress.count)

    def _stream(self, exchange: HttpExchange, engine: TextEngine, prepared: PreparedText,
                request: TokenRequest, context: ResponseContext) -> None:
        response_id, item_id = context.response_id, context.item_id
        sequence = 0; progress = GenerationProgress(); success = False
        def send(kind: str, **values: Any) -> None:
            nonlocal sequence
            event = {'type': kind, 'sequence_number': sequence, **values}; sequence += 1
            exchange.send_event(kind, event)
        try:
            exchange.begin_stream()
            empty = context.object(engine, prepared, request, 'in_progress', '', 0)
            send('response.created', response=empty)
            send('response.in_progress', response=empty)
            send('response.output_item.added', output_index=0,
                 item={'id': item_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []})
            send('response.content_part.added', item_id=item_id, output_index=0, content_index=0,
                 part={'type': 'output_text', 'text': '', 'annotations': [], 'logprobs': []})
            def on_text(chunk: str) -> None:
                send('response.output_text.delta', item_id=item_id, output_index=0, content_index=0,
                     delta=chunk, logprobs=[])
            def on_call(call):
                index, identity = len(progress.tool_calls), 'fc_' + call.id
                send('response.output_item.added', output_index=index, item=_call_item(call, 'in_progress'))
                send('response.function_call_arguments.delta', item_id=identity, output_index=index, delta=call.arguments)
                send('response.function_call_arguments.done', item_id=identity, output_index=index, name=call.name, arguments=call.arguments)
                send('response.output_item.done', output_index=index, item=_call_item(call))
            text = engine.generate(request, on_text, progress, lambda: exchange.next_event(request), prepared=prepared, on_tool_call=on_call)
            count, last = progress.count, progress.last
            status = 'incomplete' if count == prepared.limit and last not in engine.stop_ids else 'completed'
            result = context.object(engine, prepared, request, status, text, count, progress.tool_calls)
            if isinstance(request, SessionTurn): request.commit(text, result,
                **({'assistant_message': _history(text, progress)} if progress.tool_calls else {}))
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
            try: request.close()
            finally: engine.metrics.end(success, progress.count)
