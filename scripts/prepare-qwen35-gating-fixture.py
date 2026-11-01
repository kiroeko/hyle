#!/usr/bin/env python3
"""Render the fixed M1-3 eight-turn workload with a local Qwen tokenizer.

This script only loads tokenizer files. It never loads model weights, starts
inference, downloads files, or changes an existing output directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/qwen35-mtp-gating'
PROMPTS = (
    '01-chat-en.txt', '02-code-python.txt', '03-chat-zh.txt', '04-review-en.txt',
    '05-summary-en.txt', '06-summary-zh.txt', '07-writing-en.txt', '08-code-sql.txt',
)
CAPACITY, QUOTA = 2048, 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = args.tokenizer_dir.resolve()
    output = args.output.resolve()
    if not model.is_dir():
        parser.error('--tokenizer-dir must be an existing local directory')
    if output.exists():
        parser.error('output already exists; preserve the fixed workload')
    required = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
    for name in required:
        if not (model / name).is_file():
            parser.error('missing local tokenizer file: ' + str(model / name))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if tokenizer.chat_template != (model / 'chat_template.jinja').read_text(encoding='utf-8'):
        raise ValueError('selected template differs from the local model template')
    prompts = [(FIXTURE / name).read_text(encoding='utf-8').strip() for name in PROMPTS]
    if len(set(prompts)) != 8 or any(not p for p in prompts):
        raise ValueError('fixture requires eight distinct nonempty prompts')
    options = dict(add_generation_prompt=True, enable_thinking=False)
    marker = 'HYLE_ONLY_LOCATE_ASSISTANT_TEMPLATE_BOUNDARY'
    rendered, ids = [], []
    closing = None
    for i, prompt in enumerate(prompts):
        messages = [{'role': 'user', 'content': prompt}]
        single = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        direct = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **options)['input_ids']
        if tokenizer.encode(single, add_special_tokens=False) != direct:
            raise ValueError('rendered/tokenized chat-template paths differ')
        text = single
        if i:
            # Derive the prior assistant closing delimiter from this model's
            # template. The marker and fake messages never enter the fixture.
            pair = tokenizer.apply_chat_template([
                {'role': 'user', 'content': 'Previous user'},
                {'role': 'assistant', 'content': marker}, *messages], tokenize=False, **options)
            if pair.count(marker) != 1:
                raise ValueError('template does not preserve the boundary marker')
            text = pair.split(marker, 1)[1]
            if not text.endswith(single):
                raise ValueError('template has no content-independent incremental user suffix')
            boundary = text[:-len(single)]
            if not boundary or (closing is not None and boundary != closing):
                raise ValueError('inconsistent assistant closing delimiter')
            closing = boundary
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(encoded, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
            raise ValueError('tokenizer roundtrip changed the rendered prompt')
        if any(token < 0 or token >= 248320 for token in encoded):
            raise ValueError('fixture token is outside Qwen3.5-2B model vocabulary')
        rendered.append(text)
        ids.append(encoded)
    total = sum(map(len, ids))
    maximum_committed = total + len(ids) * (QUOTA - 1) + len(ids) - 1
    if total >= 1500 or maximum_committed > CAPACITY or len({tuple(v) for v in ids}) != 8:
        raise ValueError(f'fixture exceeds bounds or repeats IDs: input={total}, context={maximum_committed}')
    output.mkdir(parents=True, exist_ok=False)
    turns = output / 'turns'
    turns.mkdir()
    schedule = []
    counts = []
    committed = 0
    for i, (name, text, values) in enumerate(zip(PROMPTS, rendered, ids), 1):
        stem = Path(name).stem
        (turns / (stem + '.ids')).write_text(' '.join(map(str, values)) + '\n', encoding='ascii')
        (turns / (stem + '.rendered.txt')).write_text(text, encoding='utf-8', newline='\n')
        schedule.extend([f'prompt turns/{stem}.ids', f'generate {QUOTA}'])
        committed += len(values) + QUOTA - 1 + (1 if i > 1 else 0)
        counts.append(dict(turn=i, source=name, input_tokens=len(values), output_quota=QUOTA,
                           maximum_committed_after_turn=committed))
    (output / 'real-eight-turn.txt').write_text('\n'.join(schedule) + '\n', encoding='utf-8', newline='\n')
    with (output / 'token-counts.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    info = [
        'model_tokenizer_directory=' + str(model),
        'python=' + platform.python_version(),
        'transformers=' + importlib.metadata.version('transformers'),
        'tokenizers=' + importlib.metadata.version('tokenizers'),
        'jinja2=' + importlib.metadata.version('jinja2'),
        'tokenizer_class=' + type(tokenizer).__name__,
        'local_files_only=true; trust_remote_code=false; model_weights_loaded=false; downloads=false',
        'chat_template=local chat_template.jinja; add_generation_prompt=true; enable_thinking=false',
        'assistant_closing_delimiter=' + repr(closing),
        'first_turn=one user message plus native assistant generation prefix',
        'later_turns=native closing delimiter for prior quota-truncated assistant, then new user and generation prefix',
        'history=append actual generated content; no fabricated assistant response or replayed conversation',
        'EOS=consumer honors EOS; if earlier assistant already emitted im_end, fixed suffix still adds its explicit boundary',
        'quota=64 maximum outputs per turn; not a forced token tape or guarantee of64 actual outputs',
        'content_fixed_before_model_measurement=true; do not tune after observing performance or acceptance',
        'total_input_tokens=' + str(total),
        'maximum_committed_context=' + str(maximum_committed),
        'capacity=' + str(CAPACITY),
        'command=' + subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()),
                                             '--tokenizer-dir', str(model), '--output', str(output)]),
    ]
    info.extend('local_file=' + str(model / name) + '; bytes=' + str((model / name).stat().st_size)
                for name in required)
    (output / 'tokenizer-info.txt').write_text('\n'.join(info) + '\n', encoding='utf-8')
    print('\n'.join(info[-7:]))
    print('per_turn_input_tokens=' + ','.join(str(len(v)) for v in ids))
    print('schedule=' + str(output / 'real-eight-turn.txt'))


if __name__ == '__main__':
    main()
