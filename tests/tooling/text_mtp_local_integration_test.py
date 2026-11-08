"""Verify automatic MTP is used by the text frontend and delivered consistently.

This is a transport check; it does not compare generated tokens with a teacher.
"""
import argparse
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--expected-drafts', type=int, choices=(1, 3), required=True)
    parser.add_argument('--port', type=int, default=2048)
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    command = [sys.executable, '-B', '-m', 'tests.tooling.text_codec_local_integration_test',
               '--engine', args.engine, '--inputs', args.inputs, '--tokenizer-dir', args.tokenizer_dir,
               '--port', str(args.port), '--metrics' if args.metrics else '--no-metrics', '--mtp-auto']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace', timeout=180)
    if result.returncode:
        raise RuntimeError(f'text frontend failed ({result.returncode}):\n{result.stderr[-6000:]}')
    rows = [int(count) for count in re.findall(r'^mtp1 tokens=(\d+)', result.stderr, flags=re.MULTILINE)]
    assert len(rows) == 4 and all(value == args.expected_drafts + 1 for value in rows), rows
    print(result.stdout.strip())
    print(f'actual automatic K{args.expected_drafts}: four initial verifier batches observed')


if __name__ == '__main__':
    main()
