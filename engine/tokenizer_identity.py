"""Cold source binding for token semantics, independent of chat templates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

_SIGNATURE = 'HYLE_TOKENIZER_BINDING_V1'


def semantic_identity(tokenizer) -> str:
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        raise ValueError('source binding requires the effective fast tokenizer')
    actual = json.loads(backend.to_str())
    # Encode-call state and presentation templates do not change token meanings.
    actual.pop('padding', None)
    actual.pop('truncation', None)
    special = {name: getattr(tokenizer, name, None) for name in
               ('bos_token_id', 'eos_token_id', 'unk_token_id', 'pad_token_id')}
    value = dict(backend=actual, special=special, special_ids=sorted(tokenizer.all_special_ids))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def _record(source: str, vocabulary: int, tokenizer) -> str:
    if not re.fullmatch('[0-9a-f]{64}', source) or type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError('invalid source tokenizer binding')
    if any(type(token) is not int or not 0 <= token < vocabulary
           for token in (*tokenizer.get_vocab().values(), *tokenizer.all_special_ids)):
        raise ValueError('source tokenizer IDs exceed the original weight vocabulary')
    return f'{_SIGNATURE}\n{source}\n{vocabulary}\n{semantic_identity(tokenizer)}\n'


def bind_source_tokenizer(path: Path, source: str, vocabulary: int, reference, selected=None) -> None:
    """Source preparation supplies the reference; loading never self-enrolls it."""
    expected = _record(source, vocabulary, reference)
    if selected is not None and _record(source, vocabulary, selected) != expected:
        raise ValueError('selected tokenizer differs from the original source tokenizer')
    if path.exists():
        if path.read_text(encoding='utf-8') != expected:
            raise ValueError('existing source tokenizer binding differs')
        return
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(expected)


def validate_source_binding(path: Path, source: str, vocabulary: int, tokenizer) -> None:
    if not path.is_file():
        raise ValueError('model has no source tokenizer binding; rerun source preparation with its original tokenizer')
    if path.read_text(encoding='utf-8') != _record(source, vocabulary, tokenizer):
        raise ValueError('tokenizer identity does not match the prepared weight source')
