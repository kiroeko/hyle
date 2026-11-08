"""Exercise sampled MTP worker reuse and compare matching finite CLI routes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from engine.text.worker_client import WorkerClient


def worker_run(args):
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', '128', '--max-sessions', '1', '--active-sequences', '1',
               '--draft-count', str(args.drafts), '--overlap', 'off', '--sampling', 'random',
               '--temperature', '0.8', '--top-k', '40', '--top-p', '0.95', '--seed', '7']
    worker = WorkerClient(command, startup_timeout=120)
    outputs = []
    try:
        for _ in range(2):
            request = worker.submit([1, 2, 3], 32)
            ids = []
            try:
                while True:
                    event = request.read(timeout=120)
                    if event.kind == 'token':
                        ids.append(event.token)
                        request.acknowledge()
                    elif event.kind == 'completed':
                        break
                    else:
                        raise RuntimeError((event.kind, event.error_kind, event.message))
            finally:
                request.close()
            if len(ids) != 32:
                raise AssertionError(f'expected 32 outputs, got {len(ids)}')
            outputs.append(ids)
        if outputs[0] != outputs[1] or not worker.healthy:
            raise AssertionError('reused worker changed its sampled output or became unhealthy')
        return outputs[0]
    finally:
        worker.close(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--drafts', type=int, choices=(1, 3), required=True)
    parser.add_argument('--reference', help='compare the candidate finite CLI with this existing finite CLI')
    parser.add_argument('--expect-worker-ids', action='store_true',
                        help='also require exact worker/finite IDs for a known identical numerical plan')
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.expect_worker_ids and not args.reference:
        parser.error('--expect-worker-ids requires --reference')
    if args._child:
        print(json.dumps(worker_run(args)), flush=True)
        return
    case = args.case.resolve()
    case.mkdir(parents=True, exist_ok=False)
    (case / 'prompt.ids').write_text('1 2 3\n', encoding='ascii')
    (case / 'schedule.txt').write_text('prompt prompt.ids\ngenerate 32\n', encoding='ascii')
    command = [sys.executable, '-B', '-m', 'tests.tooling.qwen_sampled_mtp_worker_test', '--engine', args.engine, '--inputs', args.inputs,
               '--case', str(case), '--drafts', str(args.drafts), '--_child']
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
    if result.returncode:
        raise RuntimeError(f'resident worker failed: {result.stderr[-6000:]}')
    ids = json.loads(result.stdout)
    if args.reference:
        command = [args.engine, '--inputs', args.inputs, '--backend', 'cuda', '--capacity', '128',
                   '--schedule', str(case / 'schedule.txt'), '--mtp', 'on', '--mtp-max-drafts', str(args.drafts),
                   '--decode-overlap', 'off', '--sampling', 'random', '--temperature', '0.8', '--top-k', '40',
                   '--top-p', '0.95', '--seed', '7']
        reports = []
        for executable in (args.engine, args.reference):
            command[0] = executable
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
            if result.returncode:
                raise RuntimeError(f'finite CLI failed: {result.stderr[-6000:]}')
            report = json.loads(result.stdout.lstrip('\ufeff'))
            if not report['mtp'] or report['sampling'] != 'random' or report['mtp_max_drafts'] != args.drafts:
                raise AssertionError('finite CLI did not select sampled MTP')
            reports.append(report['sessions'][0])
        candidate, reference = reports
        finite_ids = candidate['result']['generated_ids']
        if len(finite_ids) != 32:
            raise AssertionError(f'expected 32 finite outputs, got {len(finite_ids)}')
        if finite_ids != reference['result']['generated_ids'] or candidate['ledger'] != reference['ledger']:
            raise AssertionError('matching finite routes changed IDs or the acceptance/state/RNG ledger')
        if args.expect_worker_ids and ids != finite_ids:
            raise AssertionError('worker IDs differ from the finite CLI for this case')
        first = next((i for i, pair in enumerate(zip(ids, finite_ids)) if pair[0] != pair[1]), None)
        print(f'worker/finite first differing token: {first}', flush=True)
    print(f'PASS sampled K{args.drafts}: two 32-token worker requests, slot reused'
          + ('; matching finite CLI IDs and ledger match' if args.reference else ''), flush=True)


if __name__ == '__main__':
    main()
