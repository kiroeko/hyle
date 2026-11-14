"""Protocol-neutral image content and bounded inline image transport.

The model processor owns patch order and placeholder expansion. This module
does not import a model, fetch a URL or interpret projection mathematics.
"""
from __future__ import annotations

import base64
import binascii
from typing import Any
from .contracts import ImageInput, PromptAttachment, PreparedPrompt

MAX_IMAGE_BYTES = 16 << 20
IMAGE_TYPES = frozenset(('image/png', 'image/jpeg', 'image/webp', 'image/gif'))


def inline_image(value: str, *, detail='auto') -> ImageInput:
    if not isinstance(value, str) or not value.startswith('data:'):
        raise ValueError('images require inline base64 data URLs; remote URL fetching is not enabled')
    header, separator, payload = value.partition(',')
    if not separator or not header.endswith(';base64'):
        raise ValueError('image data URL must use base64 encoding')
    image=base64_image(header[5:-7], payload)
    return ImageInput(image.media_type,image.data,detail)


def base64_image(media_type: str, payload: str) -> ImageInput:
    if media_type not in IMAGE_TYPES or not isinstance(payload, str) or not payload or len(payload) > ((MAX_IMAGE_BYTES + 2)//3)*4:
        raise ValueError('unsupported image type or image byte limit exceeded')
    try: data = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as error: raise ValueError('invalid image base64') from error
    if not data or len(data) > MAX_IMAGE_BYTES: raise ValueError('image byte limit exceeded')
    return ImageInput(media_type, data)


def content_parts(value: Any, *, protocol: str, role: str):
    if isinstance(value, str): return value
    if not isinstance(value, list): raise ValueError('content must be text or a content block list')
    result, images = [], False
    for part in value:
        if not isinstance(part, dict): raise ValueError('invalid message content block')
        kind = part.get('type')
        if kind in (('input_text', 'output_text') if protocol == 'responses' else ('text',)):
            allowed = {'type', 'text'} | ({'annotations', 'logprobs'} if protocol == 'responses' and kind == 'output_text' else set())
            if set(part) - allowed or not isinstance(part.get('text'), str):
                raise ValueError('invalid text content block')
            result.append({'type': 'text', 'text': part['text']})
            continue
        if role != 'user': raise ValueError('image inputs require a user message')
        if protocol == 'chat' and kind == 'image_url' and set(part) == {'type', 'image_url'}:
            source = part['image_url']
            if (not isinstance(source, dict) or set(source) - {'url', 'detail'} or
                    source.get('detail', 'auto') not in ('auto', 'high', 'low')):
                raise ValueError('invalid image_url content block')
            image = inline_image(source.get('url'), detail=source.get('detail','auto'))
        elif protocol == 'responses' and kind == 'input_image' and not set(part) - {'type', 'image_url', 'detail'}:
            if part.get('detail', 'auto') not in ('auto', 'high', 'low'): raise ValueError('unsupported image detail')
            image = inline_image(part.get('image_url'), detail=part.get('detail','auto'))
        elif protocol == 'anthropic' and kind == 'image' and set(part) == {'type', 'source'}:
            source = part['source']
            if not isinstance(source, dict) or set(source) != {'type', 'media_type', 'data'} or source['type'] != 'base64':
                raise ValueError('images require a base64 source')
            image = base64_image(source['media_type'], source['data'])
        else:
            raise ValueError('unsupported message content block')
        result.append({'type': 'image', 'image': image}); images = True
    return result if images else ''.join(part['text'] for part in result)
