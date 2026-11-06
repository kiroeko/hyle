"""Check automatic full-head MTP against explicit K1 on existing GGUF inputs."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hyle', type=Path, required=True)
    parser.add_argument('--q8-inputs', type=Path, required=True)
    parser.add_argument('--q4-inputs', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    for label, inputs in (('q8', args.q8_inputs), ('q4', args.q4_inputs)):
        assert not (inputs / 'mtp-draft-vocabulary.ids').exists()
        assert not (inputs / 'mtp-draft-vocabulary-k3.ids').exists()
        reports = {}
        for mode, flags in (
            ('auto', []),
            ('k1', ['--mtp', 'on', '--mtp-max-drafts', '1',
                    '--mtp-draft-head', 'original', '--mtp-draft-vocabulary', 'full']),
            ('off', ['--mtp', 'off']),
            ('random-auto', ['--sampling', 'random', '--seed', '42']),
            ('random-off', ['--sampling', 'random', '--seed', '42', '--mtp', 'off']),
        ):
            folder = args.output / label / mode
            folder.mkdir(parents=True)
            command = [str(args.hyle), '--inputs', str(inputs), '--backend', 'cuda',
                       '--capacity', '32768', '--schedule', str(args.schedule), *flags]
            result = subprocess.run(command, capture_output=True, timeout=180)
            (folder / 'stdout.log').write_bytes(result.stdout)
            (folder / 'stderr.log').write_bytes(result.stderr)
            if result.returncode:
                raise RuntimeError(f'{label}/{mode}: see {folder}')
            raw = json.loads(result.stdout.decode('utf-8-sig'))
            expected = mode in ('auto', 'k1')
            assert raw['mtp'] == expected and raw['mtp_max_drafts'] == int(expected), (label, mode)
            assert raw['draft_vocabulary_rows'] == 0, (label, mode)
            assert raw['live_committed_versions'] == 0, (label, mode)
            reports[mode] = raw['sessions'][0]
        for actual, reference in (('auto', 'k1'), ('random-auto', 'random-off')):
            assert reports[actual]['result']['generated_ids'] == reports[reference]['result']['generated_ids'], (label, actual)
            assert reports[actual]['ledger'] == reports[reference]['ledger'], (label, actual)
        print('PASS', label, 'automatic full K1, explicit off and random fallback', flush=True)


if __name__ == '__main__':
    main()
