"""Protocol-neutral validation of client tools and ordered tool history."""
from __future__ import annotations

import copy
import json
import re

from .contracts import ToolOptions


_NAME = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


def tool_name(value) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError('tool names require 1-64 letters, digits, underscores or hyphens')
    return value


def call_id(value) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError('tool call IDs must be nonempty bounded strings')
    return value


def json_object(value) -> dict:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result: raise ValueError('duplicate JSON tool argument: ' + key)
            result[key] = item
        return result
    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite tool arguments')))
    if not isinstance(value, dict): raise ValueError('tool arguments must be a JSON object')
    # Also rejects infinities produced by an overflowing JSON numeric literal.
    json.dumps(value, allow_nan=False, ensure_ascii=False).encode('utf-8')
    return value


def tool_options(definitions, *, choice='auto', name=None, parallel=True) -> ToolOptions | None:
    if definitions is None: definitions = []
    if not isinstance(definitions, list): raise ValueError('tools must be a list')
    if choice not in ('auto', 'none', 'required', 'named'): raise ValueError('unsupported tool choice')
    if type(parallel) is not bool: raise ValueError('parallel tool calls must be a boolean')
    normalized, names = [], set()
    for function in definitions:
        if not isinstance(function, dict) or set(function) - {'name', 'description', 'parameters', 'strict'}:
            raise ValueError('unsupported function tool definition')
        function_name = tool_name(function.get('name'))
        if function_name in names: raise ValueError('duplicate tool name: ' + function_name)
        names.add(function_name)
        description = function.get('description')
        if description is None: description = ''
        if not isinstance(description, str): raise ValueError('tool description must be a string')
        strict = function.get('strict')
        if strict is not None and type(strict) is not bool: raise ValueError('strict must be a boolean')
        if strict: raise ValueError('strict tool schemas require generation constraints and are not prepared')
        schema = function.get('parameters')
        if schema is None: schema = {'type': 'object', 'properties': {}}
        if not isinstance(schema, dict) or schema.get('type', 'object') != 'object':
            raise ValueError('tool parameters must describe an object')
        if 'properties' in schema and (not isinstance(schema['properties'], dict) or
                any(not isinstance(s, dict) for s in schema['properties'].values())):
            raise ValueError('tool properties must contain schemas')
        json_object(schema)
        normalized.append({'type': 'function', 'function': {
            'name': function_name, 'description': description, 'parameters': copy.deepcopy(schema)}})
    if choice == 'named':
        if tool_name(name) not in names: raise ValueError('chosen tool was not declared')
    elif name is not None: raise ValueError('only named tool choice accepts a name')
    if not normalized:
        if choice not in ('auto', 'none'): raise ValueError('required tool choice needs declared tools')
        return None
    return ToolOptions(tuple(normalized), choice, name, parallel)


def history_call(identity, name, arguments) -> dict:
    return {'id': call_id(identity), 'type': 'function',
            'function': {'name': tool_name(name), 'arguments': json_object(arguments)}}


def validate_tool_history(messages) -> None:
    """Calls must have exactly one following result before ordinary dialogue."""
    seen, pending = set(), set()
    for message in messages:
        if message['role'] == 'tool':
            identity = call_id(message.get('tool_call_id'))
            if identity not in pending: raise ValueError('tool result does not match an outstanding call')
            pending.remove(identity)
            continue
        if pending: raise ValueError('tool results must follow the assistant tool calls')
        calls = message.get('tool_calls', ())
        if calls and message['role'] != 'assistant': raise ValueError('only assistant messages may call tools')
        for call in calls:
            identity = call_id(call.get('id'))
            if identity in seen: raise ValueError('duplicate tool call ID')
            seen.add(identity); pending.add(identity)
    if pending: raise ValueError('tool results are missing for assistant calls')
