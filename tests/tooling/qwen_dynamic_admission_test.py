"""Explicit local CUDA regression for request admission and slot retirement.

This is a real-model diagnostic; callers supply an existing prepared model and
prompt files. It does not download weights or change the production build.
Run from the repository root with ``python -B -m tests.tooling.qwen_dynamic_admission_test``.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys

import numpy as np

from engine.text.worker_client import WorkerClient


def drain(request, initial=()):
    ids, boundaries = list(initial), []
    try:
        while True:
            event = request.read(timeout=60)
            if event.kind == 'token':
                ids.append(event.token)
                request.acknowledge()
            elif event.kind == 'turn_end':
                boundaries.append((event.turn, len(ids)))
                request.acknowledge()
            elif event.kind == 'completed':
                return ids, boundaries
            else:
                raise AssertionError((event.kind, event.message))
    finally:
        request.close()


def first_turn(request):
    ids = []
    while True:
        event = request.read(timeout=60)
        if event.kind == 'token':
            ids.append(event.token)
            request.acknowledge()
        elif event.kind == 'turn_end':
            assert event.turn == 0
            request.acknowledge()
            return ids
        else:
            raise AssertionError((event.kind, event.message))


def check_teacher(args, prompt, actual):
    if not (args.teacher_worker and args.teacher_inputs and args.teacher_output):
        raise AssertionError('different batch outputs require explicit same-prefix BF16 teacher validation')
    spec = importlib.util.spec_from_file_location('hyle_quality', 'scripts/inference-quality.py')
    quality = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = quality
    spec.loader.exec_module(quality)
    worker = quality.Worker([args.teacher_worker, args.teacher_inputs, str(args.capacity), str(len(prompt) + len(actual))],
                            Path(args.teacher_worker).parent, Path(args.teacher_output))
    success = False
    alternatives = []
    try:
        # The quality worker's initial prefill accepts at most 512 tokens.
        # Feed the remaining known prompt through its ordinary teacher step.
        initial = min(512, len(prompt))
        worker.send('P 1 ' + str(initial) + ' ' + ' '.join(map(str, prompt[:initial])))
        frame = worker.next(2)
        for token in prompt[initial:]:
            worker.send(f'D 1 {token}')
            frame = worker.next(2)
        for position, token in enumerate(actual):
            if (frame.case != 1 or frame.position != len(prompt) - initial + position or
                    frame.length != len(prompt) + position):
                raise AssertionError('teacher prefix metadata differs')
            rank = int(np.count_nonzero(frame.values > frame.values[token]) + 1)
            gap = float(frame.values[frame.top1] - frame.values[token])
            if rank != 1:
                alternatives.append((position, token, rank, gap))
                if rank != 2 or gap >= 0.1:
                    raise AssertionError(('teacher mismatch', position, token, rank, gap))
            if position + 1 < len(actual):
                worker.send(f'D 1 {token}')
                frame = worker.next(2)
        success = True
    finally:
        worker.close(success)
    print('same-prefix BF16 teacher alternatives:', alternatives, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--next-prompt', required=True)
    parser.add_argument('--sampling', choices=('greedy', 'random'), default='greedy')
    parser.add_argument('--draft-count', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=1024)
    parser.add_argument('--max-output-tokens', type=int, default=64)
    parser.add_argument('--batch-reference-engine', help='optional pre-change engine for the same paired batch')
    parser.add_argument('--teacher-worker')
    parser.add_argument('--teacher-inputs')
    parser.add_argument('--teacher-output')
    parser.add_argument('--verify-teacher', action='store_true', help='check even when greedy outputs match')
    args = parser.parse_args()
    if args.verify_teacher and args.sampling != 'greedy':
        parser.error('--verify-teacher requires greedy sampling')
    turns = [([int(value) for value in Path(path).read_text().split()], args.max_output_tokens)
             for path in (args.prompt, args.next_prompt)]
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', str(args.capacity), '--max-sessions', '2', '--active-sequences', '2',
               '--draft-count', str(args.draft_count)]
    if args.sampling == 'random':
        command += ['--sampling', 'random', '--temperature', '0.8', '--top-k', '40',
                    '--top-p', '0.95', '--seed', '123']
    batch_references = []
    if args.batch_reference_engine and args.sampling == 'greedy':
        reference_command = [args.batch_reference_engine, *command[1:]]
        reference = WorkerClient(reference_command)
        try:
            reference_single, _ = drain(reference.submit_turns([turns[0]]))
            anchor = reference.open_session(turns[0][0], 1)
            first_turn(anchor)
            pair = [reference.submit_turns([turns[0]]) for _ in range(2)]
            anchor.finish_input()
            drain(anchor)
            with ThreadPoolExecutor(max_workers=2) as pool:
                batch_references = [result[0] for result in pool.map(drain, pair)]
        finally:
            reference.close(timeout=60)
    worker = WorkerClient(command)
    try:
        expected, boundaries = drain(worker.submit_turns(turns))
        count = args.max_output_tokens
        assert boundaries == [(0, count), (1, 2 * count)]
        if batch_references:
            assert reference_single == expected[:count]
        live = worker.open_session(*turns[0])
        head = first_turn(live)
        assert head == expected[:count]
        for _ in range(2):
            ids, boundaries = drain(worker.submit_turns([turns[0]]))
            assert len(ids) == count and boundaries == [(0, count)]
            if args.sampling == 'greedy':
                assert ids == expected[:count]
        live.append(*turns[1])
        live.finish_input()
        tail, boundaries = drain(live)
        assert head + tail == expected and boundaries == [(1, count)]
        print(f'live session: {2 * count} IDs match finite baseline; two peers completed and reused a slot')

        blocked = worker.submit(*turns[0])
        first = blocked.read(timeout=60)
        assert first.kind == 'token'
        peer_ids, boundaries = drain(worker.submit_turns([turns[0]]))
        assert len(peer_ids) == count and boundaries == [(0, count)]
        blocked.acknowledge()
        remaining, _ = drain(blocked, [first.token])
        assert len(remaining) == count
        if args.sampling == 'greedy':
            assert remaining == expected[:count]
            if peer_ids not in [expected[:count], *batch_references]:
                for index, reference_ids in enumerate([expected[:count], *batch_references]):
                    first = next((i for i, (a, b) in enumerate(zip(reference_ids, peer_ids)) if a != b), None)
                    print('peer differs from reference', index, 'at', first,
                          'tokens', (reference_ids[first], peer_ids[first]) if first is not None else None,
                          flush=True)
                check_teacher(args, turns[0][0], peer_ids)
            elif args.verify_teacher:
                check_teacher(args, turns[0][0], peer_ids)
        print('finite request: peer completed while original output was backpressured')

        anchor = worker.open_session(*turns[0])
        first_turn(anchor)
        peer = worker.open_session(*turns[0])
        first_turn(peer)
        assert peer.cancel()
        terminal = peer.read(timeout=60)
        assert terminal.kind == 'cancelled', (terminal.kind, terminal.message)
        peer.close()
        ids, boundaries = drain(worker.submit_turns([turns[0]]))
        assert len(ids) == count and boundaries == [(0, count)]
        anchor.finish_input()
        _, boundaries = drain(anchor)
        assert not boundaries
        print('cancelled live peer retired; new peer reused slot while anchor remained open')
    finally:
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
