#!/usr/bin/env python3
"""Measure the local Qwen3.5-2B worker under pure Metal request concurrency."""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil
from pathlib import Path
import argparse
import re
from statistics import median
import sys
from tempfile import TemporaryFile
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/qwen35-metal-concurrency'
sys.path.insert(0, str(ROOT))
from engine.text.worker_client import WorkerClient, WorkerRequest


@dataclass
class RequestSample:
    request: WorkerRequest
    submitted: float
    tokens: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def read(self, timeout, until=None):
        while until is None or len(self.tokens) < until:
            event = self.request.read(timeout)
            if event.kind == 'completed':
                if not self.tokens or until is not None:
                    raise RuntimeError('request finished before the benchmark observed the required output')
                return
            if event.kind != 'token':
                raise RuntimeError(f'request ended during the benchmark: {event.kind} {event.message}')
            self.tokens.append(event.token)
            self.times.append(monotonic())
            self.request.acknowledge()


def submit(worker, prompt, quota):
    submitted = monotonic()
    return RequestSample(worker.submit(prompt, quota, seed=42), submitted)


def p95(values):
    return sorted(values)[ceil(len(values) * .95) - 1]


def intervals(sample):
    return zip(sample.times, sample.times[1:])


def work(worker, pool, width, prompts, long_prompt, quota, timeout):
    if width is None:
        anchor = submit(worker, prompts[0], 256)
        anchor.read(timeout, until=16)
        newcomer = submit(worker, long_prompt, 32)
        active = pool.submit(anchor.read, timeout)
        newcomer.read(timeout)
        active.result()
        gaps = [end - start for start, end in intervals(anchor)
                if start <= newcomer.times[0] and end >= newcomer.submitted]
        if not gaps or not any(newcomer.submitted < t < newcomer.times[0] for t in anchor.times):
            raise RuntimeError('long prefill did not overlap the anchor decode')
        return [anchor, newcomer], {'active_itl_p95_ms': p95(gaps) * 1000,
                                   'active_itl_max_ms': max(gaps) * 1000,
                                   'new_request_ttft_ms': (newcomer.times[0] - newcomer.submitted) * 1000}

    streams = []
    decoded, duration = 0, 0.0
    for start in range(0, len(prompts), width):
        burst = [submit(worker, prompt, quota) for prompt in prompts[start:start + width]]
        readers = [pool.submit(sample.read, timeout) for sample in burst]
        for reader in readers:
            reader.result()
        streams.extend(burst)
        decoded += sum(len(sample.tokens) - 1 for sample in burst)
        duration += max(s.times[-1] for s in burst) - min(s.times[0] for s in burst)
    ttft = [s.times[0] - s.submitted for s in streams]
    gaps = [end - start for sample in streams for start, end in intervals(sample)]
    if not gaps or duration <= 0:
        raise RuntimeError('benchmark emitted too few tokens to measure decode')
    return streams, {'decode_tps': decoded / duration, 'ttft_p50_ms': median(ttft) * 1000,
                     'ttft_p95_ms': p95(ttft) * 1000, 'itl_p95_ms': p95(gaps) * 1000}


def run(engine, args, mode, width, prompts, long_prompt):
    slots, capacity = (2, 8192) if width is None else (width, 1024)
    command = [engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'apple',
               '--active-sequences', str(slots), '--max-sessions', str(slots),
               '--capacity', str(capacity), '--draft-count', str(mode)]
    with TemporaryFile(mode='w+t', dir=ROOT / '.scratch', encoding='utf-8', errors='replace') as diagnostics:
        try:
            started = monotonic()
            worker = WorkerClient(command, startup_timeout=args.timeout, stderr=diagnostics)
            ready = monotonic()
            try:
                placement = worker.placement
                profile, = worker.execution_profiles
                if (not placement or placement.primary_backend != 'apple' or placement.cpu_ffn_layers
                        or profile.algorithm != ('mtp' if mode else 'ordinary')
                        or profile.max_draft_tokens != mode or worker.context != capacity
                        or worker.vocabulary != 248320):
                    raise RuntimeError('prepared worker differs from the requested Metal or drafting configuration')
                with ThreadPoolExecutor(max_workers=slots) as pool:
                    streams, measurements = work(worker, pool, width, prompts, long_prompt,
                                                 args.output_tokens, args.timeout)
            finally:
                worker.close()
            elapsed = monotonic() - started
        except Exception:
            diagnostics.seek(0)
            sys.stderr.writelines(deque(diagnostics, maxlen=35))
            raise
        diagnostics.seek(0)
        batches = defaultdict(int)
        for row in diagnostics:
            match = re.match(r'generation_batch requests=(\d+) T=(\d+).*\bfull_hit=0\b', row)
            if match:
                batches[tuple(map(int, match.groups()))] += 1
    largest = max((shape[0] for shape in batches), default=0)
    if largest < 1 or (width is not None and width > 1 and largest < 2):
        raise RuntimeError(f'benchmark did not observe the intended native batch: {dict(batches)}')
    tokens = [len(sample.tokens) for sample in streams]
    measurements.update(full_tps=sum(tokens) / elapsed, ready_ms=(ready - started) * 1000)
    ids = sha256(repr([s.tokens for s in streams]).encode('ascii')).hexdigest()[:16]
    return measurements, largest, tokens, ids


def token_prompts(path, count, maximum):
    prompts = tuple(tuple(map(int, row.split())) for row in path.read_text(encoding='utf-8-sig').splitlines() if row.strip())
    if len(prompts) != count or any(len(row) > maximum or min(row) < 0 for row in prompts):
        raise ValueError(f'{path} must contain {count} prompt rows of 1 to {maximum} nonnegative token IDs')
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--engine-version', required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--baseline-version')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=FIXTURES / 'batch.ids',
                        help='eight prepared prompts, space-separated token IDs per line')
    parser.add_argument('--arrival-prompt', type=Path, default=FIXTURES / 'arrival.ids',
                        help='one long prompt in the same token format')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--gpu-lock', type=Path, help='existing Mac validation GPU lock')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('this benchmark requires native Metal on macOS')
    if (args.repeats < 1 or not 2 <= args.output_tokens < 1024 or args.timeout < 1
            or bool(args.baseline) != bool(args.baseline_version)):
        parser.error('invalid timing configuration or baseline without its version')
    args.inputs = args.inputs.resolve(strict=True)
    current = args.engine.resolve(strict=True)
    arms = [('candidate', current)]
    if args.baseline:
        base = args.baseline.resolve(strict=True)
        arms.extend((('baseline', base), ('A/A', base)))
    prompts = token_prompts(args.prompts, 8, 1024 - args.output_tokens)
    long_prompt, = token_prompts(args.arrival_prompt, 1, 8192 - 32)
    if len(long_prompt) <= 512:
        parser.error('the arrival case needs a prompt longer than one default prefill chunk')
    (ROOT / '.scratch').mkdir(exist_ok=True)
    print('versions', args.engine_version, args.baseline_version or '-', 'backend apple',
          'input_lengths', list(map(len, prompts)), 'arrival_length', len(long_prompt), 'n', args.repeats, flush=True)
    print('prompts_sha256', sha256(repr((prompts, long_prompt)).encode('ascii')).hexdigest(), flush=True)
    cases = [(mode, width) for mode in (0, 1) for width in (1, 4, 8, None)]
    samples = defaultdict(list)
    with ExitStack() as stack:
        if args.gpu_lock:
            import fcntl
            lease = stack.enter_context(args.gpu_lock.open('r+b'))
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for block in range(args.repeats):
            indices = range(len(cases))
            for index in indices if block % 2 == 0 else reversed(indices):
                mode, width = cases[index]
                offset = (index + block) % len(arms)
                case = f'width{width}' if width else 'arrival'
                for arm, executable in arms[offset:] + arms[:offset]:
                    timings, largest, count, ids = run(executable, args, mode, width, prompts, long_prompt)
                    samples[mode, case, arm].append(timings)
                    print('sample', block + 1, f'K{mode}', case, arm, f'max_native_batch={largest}',
                          f'tokens={count}', f'ids={ids}',
                          ' '.join(f'{key}={value:.3f}' for key, value in timings.items()), flush=True)
    for (mode, case, arm), values in samples.items():
        summary = []
        for metric in values[0]:
            observations = [sample[metric] for sample in values]
            summary.append(f'{metric}={median(observations):.3f}[{min(observations):.3f},{max(observations):.3f}]')
        print('median[min,max]', f'K{mode}', case, arm, f'n={len(values)}', *summary)
    if args.baseline:
        for mode, width in cases:
            case = f'width{width}' if width else 'arrival'
            blocks = list(zip(*(samples[mode, case, arm] for arm in ('candidate', 'baseline', 'A/A'))))
            for metric in blocks[0][0]:
                changes = [100 * (new[metric] / (base[metric] * again[metric]) ** .5 - 1)
                           for new, base, again in blocks]
                repeats = [100 * (again[metric] / base[metric] - 1) for _, base, again in blocks]
                print('paired_percent', f'K{mode}', case, metric,
                      f'candidate={median(changes):.3f}[{min(changes):.3f},{max(changes):.3f}]',
                      f'A/A={median(repeats):.3f}[{min(repeats):.3f},{max(repeats):.3f}]')


if __name__ == '__main__':
    main()
