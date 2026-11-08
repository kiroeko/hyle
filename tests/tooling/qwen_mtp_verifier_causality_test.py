"""Check that K1 target logits do not depend on a later draft token.

This local-device integration test requires existing prepared Qwen inputs and
the quality worker. It neither downloads weights nor measures output quality.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np


def load_quality(root: Path):
    spec = importlib.util.spec_from_file_location('hyle_quality', root / 'scripts/inference-quality.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('quality worker controller is unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_ids(path: Path) -> list[int]:
    ids = [int(value) for value in path.read_text(encoding='ascii').split()]
    if not ids or any(value < 0 for value in ids):
        raise ValueError(f'invalid token IDs in {path}')
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-worker', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--draft-vocabulary', type=Path, required=True,
                        help='existing vocabulary artifact bound to the supplied weights')
    parser.add_argument('--prompt-ids', type=Path, required=True)
    parser.add_argument('--prefix-ids', type=Path, required=True)
    parser.add_argument('--draft-ids', type=int, nargs=2, default=(0, 1))
    parser.add_argument('--capacity', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True, help='new local text-log directory')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    quality = load_quality(root)
    prompt = read_ids(args.prompt_ids)
    prefix = read_ids(args.prefix_ids)
    if len(prefix) > 255 or args.capacity <= 512 or len(prompt) + len(prefix) >= args.capacity:
        raise ValueError('prefix or context exceeds the bounded quality worker configuration')
    with args.draft_vocabulary.open(encoding='utf-8') as source:
        header = [source.readline().rstrip('\r\n') for _ in range(5)]
    if header[0] != 'hyle-draft-vocabulary-v2' or header[4] != 'head original':
        raise ValueError('expected a source-bound original-head vocabulary artifact')
    vocabulary = int(header[2])
    if len(set(args.draft_ids)) != 2 or any(token < 0 or token >= vocabulary for token in args.draft_ids):
        raise ValueError('draft IDs must be distinct and within the full vocabulary')
    if any(token >= vocabulary for token in prompt + prefix):
        raise ValueError('prompt or prefix token is outside the full vocabulary')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, token in enumerate(args.draft_ids):
        vocabulary_path = (args.output / f'draft-{index}.ids').resolve()
        vocabulary_path.write_text('\n'.join(header + [str(token)]) + '\n', encoding='ascii')
        command = [args.quality_worker.resolve(), args.inputs.resolve(), str(args.capacity), str(len(prefix) + 1),
                   '--mtp', '1', '--mtp-draft-vocabulary', vocabulary_path]
        worker = None
        clean = False
        try:
            worker = quality.Worker(command, args.quality_worker.resolve().parent, args.output / f'run-{index}')
            if worker.vocabulary != vocabulary:
                raise ValueError('worker and draft artifact vocabulary differ')
            worker.send(f'P 1 {len(prompt)} ' + ' '.join(map(str, prompt)))
            frame = worker.next(2)
            quality.check_row(frame, 1, 0, len(prompt), vocabulary)
            captured = [frame.values.copy()]
            for position, forced in enumerate(prefix, 1):
                worker.send(f'D 1 {forced}')
                frame = worker.next(2)
                quality.check_row(frame, 1, position, len(prompt) + position, vocabulary)
                captured.append(frame.values.copy())
            rows.append(captured)
            clean = True
        finally:
            if worker is not None:
                worker.close(clean)
        receipt = (args.output / f'run-{index}' / 'stderr.log').read_text(encoding='utf-8', errors='replace')
        expected = (rf'quality_verified_rows={len(prefix)} total_positions={len(prefix) + 1} '
                    r'mtp=1 max_drafts=1 draft_vocabulary_rows=1 draft_head_rows=1(?:\s|$)')
        if not re.search(expected, receipt):
            raise AssertionError('worker did not execute the expected single-token-draft verifier')
    for position, (first, second) in enumerate(zip(*rows)):
        if not np.array_equal(first, second):
            raise AssertionError(f'verifier first-row logits depend on draft token at position {position}')
    print(f'PASS: {len(prefix) + 1} same-prefix rows with draft IDs {args.draft_ids[0]} and {args.draft_ids[1]}')


if __name__ == '__main__':
    main()
