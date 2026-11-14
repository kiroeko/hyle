"""Qwen's template-owned XML tool grammar, independent of HTTP protocols."""
from __future__ import annotations

import json
import re


class ToolFormat:
    @staticmethod
    def validate_tools(definitions):
        for definition in definitions:
            root = definition['function']['parameters']
            schema = _resolve(root, root)
            if any(key in schema for key in ('allOf', 'anyOf', 'oneOf')):
                raise ValueError('this tool grammar requires an object parameter root or local object reference')
            for name, field in schema.get('properties', {}).items():
                if any(c in name for c in '>\r\n'): raise ValueError('parameter name cannot be represented by this tool grammar')
                _types(field, root)

    @staticmethod
    def normalize_messages(messages):
        # This source template has a single leading system role. Preserve the
        # supplied instruction order while accepting the developer API role.
        instructions, dialogue, results = [], [], []
        pending = ()
        def flush_results():
            nonlocal pending
            if not pending:
                if results: raise ValueError('tool result has no preceding calls')
                return
            by_id = {message.get('tool_call_id'): message for message in results}
            if len(by_id) != len(results) or set(by_id) != set(pending) or len(set(pending)) != len(pending):
                raise ValueError('tool results do not match the preceding calls')
            # This source template omits IDs in <tool_response>. Associate
            # results by the calls' order even when the API returns them out
            # of order; never silently bind a result to the wrong function.
            dialogue.extend(by_id[identity] for identity in pending)
            results.clear(); pending = ()
        for message in messages:
            if message['role'] == 'tool': results.append(message); continue
            flush_results()
            if message['role'] in ('system', 'developer'):
                if dialogue: raise ValueError('this template requires instructions before dialogue')
                if not isinstance(message['content'], str): raise ValueError('instructions must contain text')
                instructions.append(message['content'])
            else:
                dialogue.append(message)
                pending = tuple(call.get('id') for call in message.get('tool_calls', ()))
        flush_results()
        return ([{'role': 'system', 'content': '\n\n'.join(instructions)}] if instructions else []) + dialogue

    @staticmethod
    def generation_prefix(choice, name):
        if choice == 'named': return '<tool_call>\n<function=' + name + '>\n'
        if choice == 'required': return '<tool_call>\n<function='
        return ''

    @staticmethod
    def new_parser(definitions, on_call, *, prefix='', required=False, max_calls=None, on_text=None):
        return _ToolParser(definitions, on_call, prefix, required, max_calls, on_text)


def _resolve(schema, root, seen=()):
    if not isinstance(schema, dict): return {}
    while '$ref' in schema:
        reference = schema['$ref']
        if not isinstance(reference, str) or not (reference=='#' or reference.startswith('#/')) or reference in seen or len(seen)>=32:
            raise ValueError('tool value types require finite local schema references')
        seen = (*seen, reference); target = root
        for part in reference[2:].split('/') if reference!='#' else ():
            key = part.replace('~1','/').replace('~0','~')
            if not isinstance(target, dict) or key not in target: raise ValueError('tool schema reference was not found')
            target = target[key]
        if not isinstance(target, dict): raise ValueError('tool schema reference must name a schema')
        schema = {**target, **{key:value for key,value in schema.items() if key!='$ref'}}
    return schema


def _types(schema, root, depth=0):
    if depth>=32: raise ValueError('tool value type composition is too deep')
    schema = _resolve(schema, root)
    kind = schema.get('type')
    if kind is not None:
        kinds = {kind} if isinstance(kind,str) else set(kind) if isinstance(kind,list) and all(isinstance(k,str) for k in kind) else set()
        if not kinds or kinds-{'string','object','array','integer','number','boolean','null'}:
            raise ValueError('unsupported tool parameter type')
        if 'number' in kinds: kinds.add('integer')
        return kinds
    for keyword in ('anyOf','oneOf','allOf'):
        if keyword in schema:
            branches = schema[keyword]
            if not isinstance(branches,list) or not branches: raise ValueError('tool type composition must contain schemas')
            values = [_types(branch,root,depth+1) for branch in branches]
            if keyword=='allOf':
                known = [value for value in values if value is not None]
                return set.intersection(*known) if known else None
            return None if any(value is None for value in values) else set.union(*values)
    if 'const' in schema: return {_kind(schema['const'])}
    if isinstance(schema.get('enum'),list) and schema['enum']: return {_kind(value) for value in schema['enum']}
    return None


def _kind(value):
    return ('null' if value is None else 'boolean' if type(value) is bool else 'integer' if type(value) is int else
        'number' if type(value) is float else 'object' if isinstance(value,dict) else 'array' if isinstance(value,list) else 'string')


def _parameter_value(raw, schema, root):
    # Remove exactly the template's framing newlines, never user whitespace.
    if raw.startswith('\r\n'): raw = raw[2:]
    elif raw.startswith('\n'): raw = raw[1:]
    if raw.endswith('\r\n'): raw = raw[:-2]
    elif raw.endswith('\n'): raw = raw[:-1]
    kinds = _types(schema, root)
    if kinds=={'string'}: return raw
    if (kinds is None or 'null' in kinds) and raw in ('null','None'): return None
    if (kinds is None or 'boolean' in kinds) and raw in ('True','False'): return raw=='True'
    try: value = json.loads(raw)
    except ValueError:
        if kinds is None or 'string' in kinds: return raw
        raise ValueError('model produced an invalid typed tool parameter') from None
    json.dumps(value, allow_nan=False)
    if isinstance(value,str):
        if kinds is None or 'string' in kinds: return raw
        raise ValueError('model produced an incompatible tool parameter type')
    kind = _kind(value)
    if kinds is not None and kind not in kinds and not (kind=='integer' and 'number' in kinds):
        if 'string' in kinds: return raw
        raise ValueError('model produced an incompatible tool parameter type')
    return value


class _ToolParser:
    def __init__(self, definitions, on_call, prefix, required, max_calls, on_text):
        self.functions = {d['function']['name']: d['function'] for d in definitions}
        self.on_call, self.required, self.max_calls = on_call, required, max_calls
        if prefix and not prefix.startswith('<tool_call>'): raise ValueError('invalid tool prefix')
        self.pending, self.inside, self.calls = '', bool(prefix), 0
        self.parts = [prefix[len('<tool_call>'):]] if prefix else []
        self.on_text, self.stopped = on_text, False

    def _text(self, text, visible):
        if self.calls:
            if text.strip(): raise ValueError('model produced text after tool calls')
        elif text and self.on_text is not None:
            if self.on_text(text) is False: self.stopped = True
        else: visible.append(text)

    def _call(self, text):
        match = re.fullmatch(r'\s*<function=([^>\r\n]+)>\s*(.*?)\s*</function>\s*', text, re.S)
        if not match or match[1] not in self.functions: raise ValueError('model produced an invalid or undeclared tool call')
        name, body = match[1], match[2]
        root = self.functions[name].get('parameters', {})
        schema = _resolve(root, root)
        properties = schema.get('properties', {})
        if not isinstance(properties, dict): raise ValueError('tool properties must be an object')
        arguments, position = {}, 0
        while position < len(body):
            start = re.match(r'\s*<parameter=([^>\r\n]+)>', body[position:])
            if not start:
                if not body[position:].strip(): break
                raise ValueError('model produced invalid tool parameters')
            key = start[1]
            if key in arguments: raise ValueError('model repeated a tool parameter')
            begin = position + start.end(); end = body.find('</parameter>', begin)
            if end < 0: raise ValueError('model produced an unfinished tool parameter')
            parameter_schema = properties.get(key, schema.get('additionalProperties', {}))
            if not isinstance(parameter_schema, dict): parameter_schema = {}
            arguments[key] = _parameter_value(body[begin:end], parameter_schema, root)
            position = end + len('</parameter>')
        self.calls += 1
        if self.max_calls is not None and self.calls > self.max_calls:
            raise ValueError('model exceeded the selected tool call count')
        self.on_call(name, arguments)

    def push(self, chunk):
        if self.stopped: return ''
        self.pending += chunk
        visible = []
        while self.pending:
            if self.inside:
                end = self.pending.find('</tool_call>')
                if end < 0:
                    keep = min(len(self.pending), len('</tool_call>')-1)
                    if len(self.pending)>keep: self.parts.append(self.pending[:-keep])
                    self.pending = self.pending[-keep:] if keep else ''
                    break
                self.parts.append(self.pending[:end]); self._call(''.join(self.parts)); self.parts.clear()
                self.pending = self.pending[end + len('</tool_call>'):]; self.inside = False
            else:
                begin = self.pending.find('<tool_call>')
                if begin >= 0:
                    self._text(self.pending[:begin], visible)
                    if self.stopped: self.pending = ''; break
                    self.pending = self.pending[begin+len('<tool_call>'):]; self.inside = True
                else:
                    marker = '<tool_call>'
                    keep = next((n for n in range(min(len(marker)-1,len(self.pending)),0,-1)
                                 if self.pending.endswith(marker[:n])), 0)
                    self._text(self.pending[:-keep] if keep else self.pending, visible)
                    self.pending = self.pending[-keep:] if keep else ''
                    break
        return ''.join(visible)

    def finish(self, *, truncated=False):
        if self.stopped: return ''
        if self.inside:
            if not truncated: raise ValueError('model ended inside a tool call')
            self.pending = ''; self.parts.clear(); return ''
        if self.required and not self.calls and not truncated: raise ValueError('model did not produce the required tool call')
        remaining, self.pending = self.pending, ''
        visible = []; self._text(remaining, visible)
        return ''.join(visible)
