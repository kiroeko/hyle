"""Local Qwen assets: tokenizer vocabulary, draft IDs and unchanged stored rows."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct

from ...asset_store import AssetStore, file_digest
from .draft_vocabulary_preparation import language_preferences, tokenizer_default_ids, write_vocabulary


@dataclass(frozen=True)
class PreparedAssets:
    directory: Path
    vocabulary: Path
    draft_vocabulary: Path
    draft_head: Path
    created: bool

    @property
    def execution_assets(self) -> tuple[tuple[str, Path], ...]:
        return (('draft.vocabulary', self.draft_vocabulary), ('draft.head', self.draft_head))

    @property
    def artifacts(self) -> tuple[tuple[str, Path], ...]:
        return (('vocabulary', self.vocabulary), *self.execution_assets)


def matches_source(inputs: Path) -> bool:
    path = Path(inputs) / 'weights.tsv'
    if not path.is_file():
        return False
    with path.open('r', encoding='utf-8') as stream:
        return stream.readline(129).strip() in ('HYLE_QWEN_ROLE_FILES_V2', 'HYLE_QWEN_GGUF_ROLE_FILES_V2')


def source_facts(inputs: Path) -> dict:
    lines = (inputs / 'weights.tsv').read_text(encoding='utf-8').splitlines()
    if len(lines) < 4 or lines[0] not in ('HYLE_QWEN_ROLE_FILES_V2', 'HYLE_QWEN_GGUF_ROLE_FILES_V2'):
        raise ValueError('cold asset preparation requires a local Qwen V2 source')
    identity = lines[1]
    if not re.fullmatch('[0-9a-f]{64}', identity):
        raise ValueError('invalid prepared weight identity')
    geometry = shlex.split(lines[2]); hidden, vocabulary = int(geometry[1]), int(geometry[3])
    if geometry[0] != 'geometry' or hidden <= 0 or not 1 < vocabulary <= 2**31 - 1:
        raise ValueError('invalid source projection geometry')
    roles = {}
    for line in lines[3:]:
        row = shlex.split(line)
        if row and row[0] in ('lm_head.weight', 'model.language_model.embed_tokens.weight'):
            if row[0] in roles: raise ValueError('duplicate original projection role')
            roles[row[0]] = row
    row = roles.get('lm_head.weight', roles.get('model.language_model.embed_tokens.weight'))
    if row is None:
        raise ValueError('source has no complete target projection')
    if lines[0] == 'HYLE_QWEN_ROLE_FILES_V2':
        if len(row) != 8:
            raise ValueError('stored row preparation requires an independent original matrix')
        dtype, rank, shape, path, offset, size = row[1], int(row[2]), row[3:5], row[5], int(row[6]), int(row[7])
    else:
        if len(row) != 11:
            raise ValueError('stored row preparation requires canonical original GGUF rows')
        dtype, rank, shape, path, offset, size = row[2], int(row[4]), row[5:7], row[7], int(row[8]), int(row[9])
    supported = {'BF16', 'F16', 'F32', 'Q8_0', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'IQ3_S', 'IQ4_NL', 'IQ4_XS'}
    if dtype not in supported or rank != 2 or list(map(int, shape)) != [vocabulary, hidden] or size % vocabulary:
        raise ValueError('source projection is not an independently stored row matrix')
    path = Path(path)
    if not path.is_absolute():
        path = inputs / path
    path = path.resolve(strict=True)
    if offset < 0 or size <= 0 or offset + size > path.stat().st_size:
        raise ValueError('source projection byte range differs from the local file')
    return dict(identity=identity, hidden=hidden, vocabulary=vocabulary, dtype=dtype,
                path=path, offset=offset, size=size, row_bytes=size // vocabulary)


def tokenizer_identity(directory: Path, template: Path | None = None, tokenizer=None) -> str:
    names = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
             'chat_template.jinja', 'config.json', 'generation_config.json', 'vocab.json', 'merges.txt',
             'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model', 'vocab.txt')
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        if path.is_file():
            digest.update(name.encode()); digest.update(bytes.fromhex(file_digest(path)))
    if template is not None:
        digest.update(b'explicit-template'); digest.update(bytes.fromhex(file_digest(template)))
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        actual = json.loads(backend.to_str())
        # These encode-call settings are transient, not tokenizer identity.
        actual['padding'] = actual['truncation'] = None
        digest.update(json.dumps(actual, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
        digest.update(json.dumps(sorted(tokenizer.all_special_ids)).encode('ascii'))
    return digest.hexdigest()


def prepare_assets(inputs: Path, tokenizer_directory: Path, tokenizer, store: AssetStore, *,
                   rows: int | None = None, languages: str | None = None, template: Path | None = None) -> PreparedAssets:
    facts = source_facts(Path(inputs))
    language_set = language_preferences(languages) if languages is not None else None
    selected_rows = (facts['vocabulary'] + 1) // 2 if rows is None else rows
    if type(selected_rows) is not int or not 0 < selected_rows < facts['vocabulary']:
        raise ValueError('draft rows must be a proper nonempty subset of the target vocabulary')
    tokenizer_id = tokenizer_identity(Path(tokenizer_directory), template, tokenizer)
    recipe = json.dumps(dict(provider='qwen35-stored-rows-v1', source=facts['identity'], tokenizer=tokenizer_id,
        rows=selected_rows, languages=sorted(language_set) if language_set is not None else None,
        dtype=facts['dtype']), sort_keys=True, separators=(',', ':'))

    def build(output: Path) -> None:
        ids, _ = tokenizer_default_ids(tokenizer, facts['vocabulary'], rows=selected_rows, languages=language_set,
                                      template=template.read_text(encoding='utf-8') if template else None)
        with (output / 'mtp-draft-vocabulary.ids').open('x', encoding='utf-8', newline='\n') as stream:
            write_vocabulary(stream, facts['identity'], facts['vocabulary'], ids, None)
        with (output / 'vocabulary.tsv').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(f'HYLE_TOKENIZER_VOCABULARY_V1\n{facts["identity"]}\n{tokenizer_id}\n{facts["vocabulary"]}\n')
            for index in range(facts['vocabulary']):
                token = tokenizer.convert_ids_to_tokens(index)
                stream.write(f'{index}\t{json.dumps(token, ensure_ascii=False)}\n')
        digest = hashlib.sha256()
        before = facts['path'].stat()
        with facts['path'].open('rb') as source, (output / 'draft-head.bin').open('xb') as target:
            # Coalesce adjacent original IDs; bounded reads also cover a large
            # contiguous subset without allocating another head-sized buffer.
            first = 0
            while first < len(ids):
                end = first + 1
                while end < len(ids) and ids[end] == ids[end - 1] + 1:
                    end += 1
                source.seek(facts['offset'] + ids[first] * facts['row_bytes'])
                remaining = (end - first) * facts['row_bytes']
                while remaining:
                    block = source.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError('source projection ended during asset preparation')
                    target.write(block); digest.update(block); remaining -= len(block)
                first = end
        after = facts['path'].stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('source projection changed during preparation')
        mapping = hashlib.sha256(struct.pack('<' + 'i' * len(ids), *ids)).hexdigest()
        metadata = (f'HYLE_QWEN_DRAFT_HEAD_STORED_SUBSET_V1\n{facts["identity"]}\n{digest.hexdigest()}\n'
                    f'stored-rows-v1:{facts["dtype"]}\n{len(ids)} {facts["hidden"]}\n'
                    f'"draft-head.bin" {len(ids) * facts["row_bytes"]}\n{mapping}\n')
        (output / 'mtp-draft-head.tsv').write_text(metadata, encoding='utf-8')

    directory, created = store.prepare(recipe, build)
    return PreparedAssets(directory, directory / 'vocabulary.tsv', directory / 'mtp-draft-vocabulary.ids',
                          directory / 'mtp-draft-head.tsv', created)
