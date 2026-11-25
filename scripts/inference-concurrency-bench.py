#!/usr/bin/env python3
"""Measure concurrent multi-turn coding reviews on the Qwen3.5-2B Metal worker."""

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

    def read_until(self, count, timeout):
        while len(self.tokens) < count:
            event = self.request.read(timeout)
            if event.kind != 'token':
                raise RuntimeError(f'anchor ended before other users arrived: {event.kind} {event.message}')
            self.tokens.append(event.token)
            self.times.append(monotonic())
            self.request.acknowledge()


def p95(values):
    return sorted(values)[ceil(len(values) * .95) - 1]


def intervals(sample):
    return zip(sample.times, sample.times[1:])


def decode_by_width(streams):
    """Count delivery time and tokens while exactly N turns are decoding."""
    events = defaultdict(lambda: [0, 0])
    for stream in streams:
        events[stream.times[0]][0] += 1
        events[stream.times[-1]][0] -= 1
        for timestamp in stream.times[1:]:
            events[timestamp][1] += 1
    durations, tokens = defaultdict(float), defaultdict(int)
    active = 0
    previous = min(events)
    for timestamp, (change, delivered) in sorted(events.items()):
        durations[active] += timestamp - previous
        tokens[active] += delivered
        active += change
        previous = timestamp
    return durations, tokens


def coding_review(worker, pool, prompts, quota, timeout):
    """Replay independent live conversations after the worker is ready."""
    def start(prompt):
        submitted = monotonic()
        return RequestSample(worker.open_session(prompt, quota, seed=42), submitted)

    def conversation(sample, followups):
        request = sample.request
        samples = []
        for turn in (*followups, None):
            while True:
                event = request.read(timeout)
                if event.kind == 'turn_end':
                    request.acknowledge()
                    if len(sample.tokens) < 2:
                        raise RuntimeError('review turn produced too few tokens for throughput')
                    samples.append(sample)
                    break
                if event.kind != 'token':
                    raise RuntimeError(f'coding review ended during a turn: {event.kind} {event.message}')
                sample.tokens.append(event.token)
                sample.times.append(monotonic())
                request.acknowledge()
            if turn is None:
                request.finish_input(timeout=timeout)
            else:
                sample = RequestSample(request, monotonic())
                request.append(turn, quota, timeout=timeout)
        event = request.read(timeout)
        if event.kind != 'completed':
            raise RuntimeError(f'coding review did not complete: {event.kind} {event.message}')
        return samples

    anchor = start(prompts[0][0])
    anchor.read_until(48, timeout)
    active = [anchor, *(start(turns[0]) for turns in prompts[1:])]
    readers = [pool.submit(conversation, sample, turns[1:]) for sample, turns in zip(active, prompts)]
    streams = [sample for reader in readers for sample in reader.result()]
    durations, decoded = decode_by_width(streams)
    depth = len(prompts)
    overlap = durations[depth]
    if overlap <= 0 or not decoded[depth]:
        raise RuntimeError(f'the {depth} conversations did not all decode concurrently')
    latency = [stream.times[0] - stream.submitted for stream in streams]
    gaps = [end - begin for sample in streams for begin, end in intervals(sample)]
    emitted = sum(len(stream.tokens) for stream in streams)
    elapsed = max(stream.times[-1] for stream in streams) - anchor.submitted
    return streams, {'combined_tps': emitted / elapsed, 'concurrent_decode_tps': decoded[depth] / overlap,
                     'overlap_s': overlap, 'ttft_p50_ms': median(latency) * 1000,
                     'ttft_p95_ms': p95(latency) * 1000, 'itl_p95_ms': p95(gaps) * 1000}


def run(engine, args, mode, prompts, vocabulary):
    slots, capacity = len(prompts), 16384
    command = [engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'apple',
               '--active-sequences', str(slots), '--max-sessions', str(slots),
               '--capacity', str(capacity), '--draft-count', str(mode)]
    if vocabulary:
        command.extend(('--draft-vocabulary', vocabulary))
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
                    streams, measurements = coding_review(worker, pool, prompts, args.output_tokens, args.timeout)
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
    if not largest:
        raise RuntimeError('worker emitted no native generation batch diagnostics')
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
    parser.add_argument('--sessions', type=int, choices=(2, 3, 4), default=2,
                        help='number of concurrent coding conversations')
    parser.add_argument('--agent-turns', type=Path,
                        help='four consecutive token inputs per conversation')
    parser.add_argument('--draft-vocabulary', type=Path,
                        help='add a K2 comparison using a prepared draft-vocabulary file')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=192,
                        help='per-turn output ceiling, at least 192')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--gpu-lock', type=Path, help='existing Mac validation GPU lock')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('this benchmark requires native Metal on macOS')
    if args.repeats < 3:
        parser.error('the comparison requires at least three independent processes per arm')
    if (not 192 <= args.output_tokens < 1024 or args.timeout < 1
            or bool(args.baseline) != bool(args.baseline_version)):
        parser.error('invalid timing configuration or baseline without its version')
    args.inputs = args.inputs.resolve(strict=True)
    current = args.engine.resolve(strict=True)
    arms = [('candidate', current)]
    if args.baseline:
        base = args.baseline.resolve(strict=True)
        arms.extend((('baseline', base), ('A/A', base)))
    count = 4 * args.sessions
    rows = token_prompts(args.agent_turns or FIXTURES / 'agent-review.ids',
                         count if args.agent_turns else 16, 16384 - args.output_tokens)[:count]
    prompts = tuple(rows[i:i + 4] for i in range(0, count, 4))
    if (min(map(len, rows)) < 1500 or
            any(sum(map(len, request)) + 4 * args.output_tokens > 16384 for request in prompts)):
        parser.error('coding review requires inputs of at least 1500 tokens with each conversation fitting 16384')
    label = f'coding-review-{args.sessions}'
    cases = [(0, label, None), (2, label, None)]
    if args.draft_vocabulary:
        cases.append((2, f'{label}-subset', args.draft_vocabulary.resolve(strict=True)))
    (ROOT / '.scratch').mkdir(exist_ok=True)
    print('versions', args.engine_version, args.baseline_version or '-', 'backend apple',
          'input_lengths', list(map(len, rows)), 'n', args.repeats, flush=True)
    print('prompts_sha256', sha256(repr(prompts).encode('ascii')).hexdigest(), flush=True)
    samples = defaultdict(list)
    with ExitStack() as stack:
        if args.gpu_lock:
            import fcntl
            lease = stack.enter_context(args.gpu_lock.open('r+b'))
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for block in range(args.repeats):
            indices = range(len(cases))
            for index in indices if block % 2 == 0 else reversed(indices):
                mode, case, vocabulary = cases[index]
                offset = (index + block) % len(arms)
                for arm, executable in arms[offset:] + arms[:offset]:
                    timings, largest, count, ids = run(executable, args, mode, prompts, vocabulary)
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
        for mode, case, _ in cases:
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
