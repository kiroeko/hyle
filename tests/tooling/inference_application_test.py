"""Real-model CLI/worker equivalence using existing local assets; no build/download."""
import argparse
from pathlib import Path
import subprocess

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
                raise AssertionError((event.kind, event.message))
    finally:
        request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--diagnostic')
    parser.add_argument('--diagnostic-schedule', type=Path)
    args = parser.parse_args()
    prompt = [int(value) for value in args.prompt.read_text(encoding='utf-8').split()]
    for depth, sampling in ((0, 'greedy'), (1, 'greedy'), (0, 'random')):
        preparation = ['--inputs', str(args.inputs), '--backend', 'cuda', '--capacity', '256',
                       '--max-sessions', '1', '--active-sequences', '1', '--draft-count', str(depth),
                       '--sampling', sampling, '--seed', '123']
        if sampling == 'random':
            preparation += ['--temperature', '0.8', '--top-k', '40', '--top-p', '0.95']
        result = subprocess.run([args.engine, 'generate', *preparation, '--tokens', '-',
                                 '--max-output-tokens', '16', '--execution-profile', '0', '--sampling-index', '0'],
                                input=' '.join(map(str, prompt)), text=True, encoding='utf-8',
                                capture_output=True, timeout=180)
        if result.returncode:
            raise AssertionError(result.stderr[-3000:])
        actual = [int(value) for value in result.stdout.split()]
        worker = WorkerClient([args.engine, '--engine-worker', *preparation], startup_timeout=120)
        try:
            expected = consume(worker.submit(prompt, 16, execution_profile=0, seed=123))
        finally:
            worker.close(timeout=120)
        if actual != expected or len(actual) != 16:
            raise AssertionError(f'CLI/worker output differs: K={depth}, {sampling}')
        print(f'PASS CLI/worker: {sampling}, K={depth}, 16 identical token IDs', flush=True)
    if args.diagnostic:
        if not args.diagnostic_schedule:
            raise ValueError('--diagnostic requires an existing --diagnostic-schedule')
        help = subprocess.run([args.diagnostic, '--help'], capture_output=True, text=True, timeout=30)
        if help.returncode or 'Qwen diagnostic' not in help.stdout:
            raise AssertionError('explicit diagnostic entry did not start')
        report = subprocess.run([args.diagnostic, '--inputs', str(args.inputs), '--backend', 'cuda',
                                 '--mtp', 'off', '--capacity', '256', '--schedule', str(args.diagnostic_schedule)],
                                capture_output=True, text=True, timeout=180)
        if report.returncode or '"generation"' not in report.stdout:
            raise AssertionError('explicit diagnostic report failed: ' + report.stderr[-2000:])
        print('PASS explicit diagnostic consumer: help and completed generation report', flush=True)


if __name__ == '__main__':
    main()
