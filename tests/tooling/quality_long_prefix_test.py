"""Check chunked quality-worker prefill against existing production routes.

Uses only caller-provided local workers, models and token IDs; no network access.
Worker logs are text under --output. GPU calls are serialized.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyle-worker', type=Path, required=True)
    parser.add_argument('--hyle-runtime', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--peer-worker', type=Path, required=True)
    parser.add_argument('--peer-runtime', type=Path, required=True)
    parser.add_argument('--peer-model', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--capacity', type=int, default=2048)
    parser.add_argument('--lengths', type=int, nargs='+', default=[513, 1025])
    parser.add_argument('--mtp-depth', type=int, action='append', default=[])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('hyle_quality', Path('scripts/inference-quality.py'))
    quality = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = quality
    spec.loader.exec_module(quality)
    seed = [int(value) for value in args.prompt.read_text().split()]
    if not seed:
        raise ValueError('empty prompt')
    if not args.lengths or any(count < 513 or count + 1 >= args.capacity for count in args.lengths):
        raise ValueError('lengths must exceed 512 and leave room for a decode token')
    maximum = max(args.lengths)
    tokens = (seed * (maximum // len(seed) + 1))[:maximum]
    args.output.mkdir()
    args.trace.mkdir()
    cases = [
        ('hyle', [args.hyle_worker, args.inputs, str(args.capacity), '32'], args.hyle_runtime),
        ('peer', [args.peer_worker, args.peer_runtime, args.peer_model, str(args.capacity)], args.peer_runtime),
    ]
    hyle_rows = {}
    for name, command, runtime in cases:
        worker = None
        clean = False
        try:
            worker = quality.Worker(command, runtime, args.output / name)
            for count in args.lengths:
                comparable_append = name == 'peer' and count % 512 == 1
                if comparable_append:
                    before = count - 1
                    worker.send(f'P 1 {before} ' + ' '.join(map(str, tokens[:before])))
                    first = worker.next(2)
                    quality.check_row(first, 1, 0, before, worker.vocabulary)
                    worker.send(f'D 1 {tokens[before]}')
                    appended = worker.next(2)
                    quality.check_row(appended, 1, 1, count, worker.vocabulary)
                    worker.reset()
                worker.send(f'P 2 {count} ' + ' '.join(map(str, tokens[:count])))
                chunked = worker.next(2)
                quality.check_row(chunked, 2, 0, count, worker.vocabulary)
                if name == 'hyle':
                    hyle_rows[count] = chunked.values.copy()
                elif comparable_append and not np.array_equal(appended.values, chunked.values):
                    raise AssertionError((name, count, 'max difference',
                                          float(np.max(np.abs(appended.values - chunked.values)))))
                worker.reset()
                if comparable_append:
                    print(name, count, 'chunked vs append: exact', flush=True)
                elif name == 'peer':
                    print(name, count, 'chunked prefill: retired', flush=True)
            clean = True
        finally:
            if worker is not None:
                worker.close(clean)
    for depth in args.mtp_depth:
        worker = None
        clean = False
        try:
            worker = quality.Worker([args.hyle_worker, args.inputs, str(args.capacity), '8', '--mtp', str(depth)],
                                    args.hyle_runtime, args.output / f'mtp-{depth}')
            worker.send(f'P 1 {maximum} ' + ' '.join(map(str, tokens)))
            prefill = worker.next(2)
            quality.check_row(prefill, 1, 0, maximum, worker.vocabulary)
            worker.send(f'D 1 {tokens[-1]}')
            verification = worker.next(2)
            quality.check_row(verification, 1, 1, maximum + 1, worker.vocabulary)
            print('hyle MTP', depth, maximum, 'prefill and verifier: retired', flush=True)
            clean = True
        finally:
            if worker is not None:
                worker.close(clean)
    for count in args.lengths:
        prompt = args.trace / f'prompt-{count}.ids'
        prompt.write_text(' '.join(map(str, tokens[:count])) + '\n', encoding='ascii')
        schedule = args.trace / f'schedule-{count}.txt'
        schedule.write_text(f'prompt "{prompt.resolve().as_posix()}"\ngenerate 8\n', encoding='utf-8')
        dest = args.trace / str(count)
        chunk = (count - 1) // 512
        subprocess.run([args.probe, '--sampling-logits-range', args.inputs, schedule, 'bounded',
                        str(chunk), str(chunk), dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=180, check=True)
        observed = np.fromfile(dest / f'prefill_chunk{chunk * 512}' / 'logits.f32', dtype='<f4')
        if not np.array_equal(hyle_rows[count], observed):
            raise AssertionError(('hyle', count, 'quality worker differs from production prefill'))
        print('hyle', count, 'chunked vs production prefill: exact', flush=True)


if __name__ == '__main__':
    main()
