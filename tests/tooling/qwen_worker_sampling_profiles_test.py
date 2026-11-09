"""Check cold-prepared sampling profiles with an existing Qwen CUDA model."""
import argparse
from pathlib import Path
import subprocess
import threading

from engine.text.worker_client import WorkerClient


def consume(request):
    tokens = []
    try:
        while True:
            event = request.read(timeout=120)
            if event.kind == 'token':
                tokens.append(event.token)
                request.acknowledge()
            elif event.kind == 'completed':
                return tokens
            else:
                raise RuntimeError((event.kind, event.error_kind, event.message))
    finally:
        request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--draft-head', type=Path)
    args = parser.parse_args()
    prompt = [int(value) for value in args.prompt.read_text(encoding='utf-8').split()]
    if not prompt:
        raise ValueError('prompt must contain token IDs')
    base = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
            '--capacity', '256', '--max-sessions', '2', '--active-sequences', '2', '--draft-count', '3']
    subset = ['--draft-vocabulary', args.inputs / 'mtp-draft-vocabulary.ids']
    if args.draft_head:
        subset += ['--draft-head', args.draft_head]
    random = ['--sampling', 'random', '--temperature', '0.8', '--top-k', '40', '--top-p', '0.95']

    for invalid in ('other', 'random,0.8,40', 'random,0.8,40,0.95,extra'):
        result = subprocess.run([args.engine, '--engine-worker', '--additional-sampling', invalid],
                                capture_output=True, text=True, timeout=30)
        if not result.returncode or 'additional' not in result.stderr:
            raise AssertionError(f'invalid profile was accepted: {invalid}')
    for valid in ('greedy', 'random', 'random,0.8,40,0.95'):
        result = subprocess.run([args.engine, '--engine-worker', '--additional-sampling', valid],
                                capture_output=True, text=True, timeout=30)
        if not result.returncode or 'model input path is required' not in result.stderr:
            raise AssertionError(f'valid profile did not reach model admission: {valid}')

    def single(command, **options):
        worker = WorkerClient(command, startup_timeout=120)
        try:
            return consume(worker.submit(prompt, 16, **options))
        finally:
            worker.close(timeout=120)

    greedy_reference = single([*base, *subset], max_draft_tokens=3)
    random_reference = single([*base, *random], seed=123, max_draft_tokens=1)
    mixed = [*base, *subset, '--additional-sampling', 'random,0.8,40,0.95']
    worker = WorkerClient(mixed, startup_timeout=120)
    try:
        greedy = consume(worker.submit(prompt, 16, sampling_index=0, max_draft_tokens=3))
        sampled = consume(worker.submit(prompt, 16, sampling_index=1, seed=123, max_draft_tokens=1))
        if greedy != greedy_reference or sampled != random_reference:
            raise AssertionError('prepared profile changed the corresponding single-profile output')
        requests = [worker.submit(prompt, 16, sampling_index=0, max_draft_tokens=3),
                    worker.submit(prompt, 16, sampling_index=1, seed=123, max_draft_tokens=1)]
        results = [None, None]
        errors = []
        def read(index):
            try:
                results[index] = consume(requests[index])
            except BaseException as error:
                errors.append(error)
        threads = [threading.Thread(target=read, args=(index,)) for index in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=180)
        if errors or any(thread.is_alive() for thread in threads) or any(len(value or []) != 16 for value in results):
            raise AssertionError(f'concurrent mixed profiles failed: {errors}')
        request = worker.submit(prompt, 1, sampling_index=2)
        try:
            if request.read(timeout=120).kind != 'error':
                raise AssertionError('unprepared sampling profile was not rejected')
        finally:
            request.close()
        if len(consume(worker.submit(prompt, 2, sampling_index=0))) != 2 or not worker.healthy:
            raise AssertionError('rejected profile poisoned the worker')
    finally:
        worker.close(timeout=120)

    # Preparing greedy after the default random profile must be equally usable.
    inverse = [*base, *random, *subset, '--additional-sampling', 'greedy']
    if len(single(inverse, sampling_index=1, max_draft_tokens=3)) != 16:
        raise AssertionError('additional greedy profile did not complete')
    if len(single([*base, '--additional-sampling', 'random'],
                  sampling_index=1, seed=0, max_draft_tokens=1)) != 16:
        raise AssertionError('default random profile did not complete')
    print('PASS prepared greedy/subset K3 and random/full K1: single outputs match, concurrent requests complete; '
          'unprepared profile rejected; inverse profile order accepted', flush=True)


if __name__ == '__main__':
    main()
