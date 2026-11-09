"""Verify prepared sampling profiles on one local Qwen CUDA model."""
import argparse
from collections import Counter
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, help='existing qwen_generation_batch_relay_probe executable')
    parser.add_argument('--inputs', required=True, help='existing local Qwen3.5-2B prepared inputs')
    parser.add_argument('--mode', choices=('ordinary', 'k1', 'k3'), required=True)
    args = parser.parse_args()
    result = subprocess.run([args.probe, '--mixed-sampling', args.inputs, args.mode],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
    if result.returncode:
        raise RuntimeError(f'mixed sampling failed: {result.stderr[-4000:]}')
    roles = Counter(int(role) for role in re.findall(
        r'generation_batch requests=\d+ T=\d+ role=(\d+)', result.stderr))
    if args.mode != 'ordinary' and not any(roles[role] for role in (1, 3, 4)):
        raise AssertionError('MTP configuration never executed a draft or verification role')
    if 'mixed sampling ' + args.mode + ' PASS:' not in result.stdout:
        raise AssertionError('probe did not complete its mixed-sampling checks')
    print(f'PASS {args.mode}: initial mixed cohort, three profiles, both admission orders, RNG replay, isolated rejection; executed roles {dict(roles)}')


if __name__ == '__main__':
    main()
