"""Verify two-session ordinary relay against serial CUDA generation.

Run with an existing Qwen3.5-2B image and preparation. The case directory holds
only tiny input schedules; the test neither builds nor downloads a model.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--sampling', choices=('greedy', 'random'), required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    case.mkdir(parents=True, exist_ok=True)
    (case / 'prompt.ids').write_text('1 2 3\n', encoding='utf-8')
    for name in ('A', 'B'):
        (case / f'{name}.schedule').write_text('prompt prompt.ids\ngenerate 32\n', encoding='utf-8')
    (case / 'sessions.txt').write_text('A A.schedule\nB B.schedule\n', encoding='utf-8')
    common = [str(args.engine), '--inputs', str(args.inputs), '--session-schedules', str(case / 'sessions.txt'),
              '--backend', 'cuda', '--capacity', '128', '--max-sessions', '2',
              '--active-sequences', '2', '--prefix-cache', 'off', '--mtp', 'off', '--sampling', args.sampling]
    if args.sampling == 'random':
        common += ['--temperature', '0.8', '--top-k', '40', '--top-p', '0.95', '--seed', '7']
    reports = {}
    for mode in ('off', 'auto'):
        result = subprocess.run(common + ['--decode-overlap', mode], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=180)
        if result.returncode:
            raise RuntimeError(f'{mode} exited {result.returncode}: {result.stderr[-8000:]}')
        report = json.loads(result.stdout.lstrip('\ufeff'))
        assert report['session_count'] == 2 and report['active_sequences'] == 2
        assert report['live_committed_versions'] == 0
        assert 'generation_batch requests=2' in result.stderr, 'no real two-member batch'
        reports[mode] = report
    serial, relay = reports['off'], reports['auto']
    assert not serial['decode_overlap'] and serial['overlap_successor_publications'] == 0
    assert relay['decode_overlap'] and relay['decode_overlap_available']
    assert relay['overlap_successor_submissions'] > 0 and relay['overlap_successor_publications'] > 0
    assert relay['overlap_successor_refusals'] == 0
    def published(report):
        return {entry['session_id']: entry['result']['generated_ids'] for entry in report['sessions']}
    expected, actual = published(serial), published(relay)
    assert set(expected) == {'A', 'B'} and all(len(ids) == 32 for ids in expected.values())
    assert actual == expected, 'two-session published output changed'
    print(f"PASS {args.sampling}: 2 x 32 matching token IDs, "
          f"{relay['overlap_successor_submissions']} successor submissions, "
          f"{relay['overlap_successor_publications']} publications, "
          f"{relay['overlap_successor_discards']} discards", flush=True)


if __name__ == '__main__':
    main()
