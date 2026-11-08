"""Check cancellation and slot reuse with real single-session random relay.

Run from the repository root with an existing Qwen3.5-2B preparation and token
prompt. This test uses CUDA and does not download weights or build the engine.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from engine.text.worker_client import WorkerClient


def tokens(request, count):
    values = []
    while len(values) < count:
        event = request.read(timeout=60)
        if event.kind != 'token':
            raise AssertionError((event.kind, event.error_kind, event.message, len(values), count))
        values.append(event.token)
        request.acknowledge()
    return values


def complete(worker, prompt, count):
    request = worker.submit(prompt, count)
    try:
        values = tokens(request, count)
        terminal = request.read(timeout=60)
        if terminal.kind != 'completed':
            raise AssertionError((terminal.kind, terminal.error_kind, terminal.message))
        return values
    finally:
        request.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args._child:
        command = [sys.executable, '-B', '-m', 'tests.tooling.qwen_random_relay_retirement_test',
                   '--engine', str(args.engine), '--inputs', str(args.inputs),
                   '--prompt', str(args.prompt), '--_child']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace', timeout=180)
        if result.returncode:
            raise RuntimeError(f'retirement check failed: {result.stderr[-8000:]}')
        if 'planned_replay T=1 slot_count=3 ' not in result.stderr:
            raise AssertionError('single-session decode relay was not prepared')
        print(result.stdout.strip())
        print('decode relay prepared; native in-flight state at cancellation was not observed', flush=True)
        return
    prompt = [int(value) for value in args.prompt.read_text().split()]
    if not prompt or len(prompt) + 64 > 512:
        raise ValueError('prompt must leave room for the fixed output limit')
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', '512', '--max-sessions', '1', '--active-sequences', '1',
               '--draft-count', '0', '--sampling', 'random', '--temperature', '0.8',
               '--top-k', '40', '--top-p', '0.95', '--seed', '7']
    worker = WorkerClient(command, startup_timeout=120)
    try:
        baseline = complete(worker, prompt, 48)
        for attempt in range(2):
            request = worker.submit(prompt, 64)
            try:
                prefix = tokens(request, 24)
                if prefix != baseline[:24]:
                    raise AssertionError(f'cancelled request {attempt} changed its RNG prefix')
            finally:
                # Close requests cancellation and abandons output; the model
                # must observe real retirement before reusing the only slot.
                request.close()
            followup = complete(worker, prompt, 48)
            if followup != baseline:
                raise AssertionError(f'followup {attempt} changed after cancellation')
        if not worker.healthy:
            raise AssertionError('model worker became unhealthy')
        print('PASS: random single-session cancellation and slot reuse; '
              'two 24-token prefixes and two 48-token followups match', flush=True)
    finally:
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
