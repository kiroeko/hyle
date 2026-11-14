"""Model discovery projections over the actual loaded engine's facts."""
from __future__ import annotations

from urllib.parse import unquote


class ModelsAdapter:
    @staticmethod
    def _error(exchange, status, message, anthropic):
        kind = 'not_found_error' if status == 404 else 'invalid_request_error'
        error = {'type': kind, 'message': message}
        if anthropic: exchange.send_json(status, {'type': 'error', 'error': error})
        else: exchange.send_json(status, {'error': {**error, 'param': None, 'code': 'model_not_found' if status == 404 else None}})

    def handle_get(self, exchange, engine, key):
        anthropic = exchange.header('anthropic-version') is not None
        try:
            query = exchange.query_parameters()
            allowed = {'after_id', 'before_id', 'limit'} if anthropic and not key else set()
            if set(query) - allowed or any(len(values) != 1 for values in query.values()):
                raise ValueError('unsupported or repeated model discovery query')
            if 'after_id' in query and 'before_id' in query:
                raise ValueError('choose after_id or before_id')
            if 'limit' in query:
                value = query['limit'][0]
                if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 1000:
                    raise ValueError('model list limit must be between 1 and 1000')
            cursor = query.get('after_id', query.get('before_id'))
            if cursor is not None and cursor != [engine.model_name]: raise ValueError('unknown model list cursor')
        except ValueError as error:
            self._error(exchange, 400, str(error), anthropic); return
        if key and unquote(key) != engine.model_name:
            self._error(exchange, 404, 'model was not found', anthropic); return
        capabilities = engine.describe_capabilities()
        if anthropic:
            item = {'id': engine.model_name, 'type': 'model', 'display_name': engine.model_name,
                'created_at': '1970-01-01T00:00:00Z', 'max_input_tokens': engine.worker.context,
                'max_tokens': engine.worker.context, 'hyle': capabilities}
            data = [] if cursor is not None else [item]
            result = item if key else {'data': data, 'has_more': False,
                'first_id': engine.model_name if data else None, 'last_id': engine.model_name if data else None}
        else:
            item = {'id': engine.model_name, 'object': 'model', 'created': 0, 'owned_by': 'hyle', 'hyle': capabilities}
            result = item if key else {'object': 'list', 'data': [item]}
        exchange.send_json(200, result)
