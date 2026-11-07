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
        reports = {}
        for mode, flags in (
            ('auto', []),
            ('auto-manual', ['--mtp-draft-head', 'original', '--mtp-draft-vocabulary', 'full']),
            ('auto-full', ['--mtp-draft-vocabulary', 'full']),
            ('k1', ['--mtp', 'on', '--mtp-max-drafts', '1',
                    '--mtp-draft-head', 'original', '--mtp-draft-vocabulary', 'full']),
            ('off', ['--mtp', 'off']),
            ('auto-zero', ['--mtp-max-drafts', '0']),
            ('off-zero', ['--mtp', 'off', '--mtp-max-drafts', '0']),
            ('on-zero', ['--mtp', 'on', '--mtp-max-drafts', '0']),
            ('commands-auto', ['--replay', 'commands']),
            ('commands-k1', ['--replay', 'commands', '--mtp', 'on', '--mtp-max-drafts', '1',
                             '--mtp-draft-head', 'original', '--mtp-draft-vocabulary', 'full']),
            ('auto-k4', ['--mtp-max-drafts', '4']),
            ('explicit-k4', ['--mtp', 'on', '--mtp-max-drafts', '4',
                             '--mtp-draft-head', 'original', '--mtp-draft-vocabulary', 'full']),
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
            expected = mode in ('auto', 'auto-manual', 'auto-full', 'k1', 'on-zero', 'commands-auto', 'commands-k1', 'auto-k4', 'explicit-k4')
            reasons = {'auto': ('original_projection', 'configured_default'),
                       'auto-manual': ('explicit_draft_asset', 'configured_default'),
                       'auto-full': ('explicit_draft_asset', 'configured_default'),
                       'k1': ('explicit_on', 'explicit'), 'off': ('explicit_off', 'disabled'),
                       'auto-zero': ('drafting_disabled', 'disabled'),
                       'off-zero': ('explicit_off', 'disabled'),
                       'on-zero': ('explicit_on', 'explicit'),
                       'commands-auto': ('original_projection', 'configured_default'),
                       'commands-k1': ('explicit_on', 'explicit'),
                       'auto-k4': ('original_projection', 'explicit'),
                       'explicit-k4': ('explicit_on', 'explicit'),
                       'random-auto': ('sampling', 'disabled'),
                       'random-off': ('explicit_off', 'disabled')}
            assert (raw['mtp_selection_reason'], raw['mtp_depth_source']) == reasons[mode], (label, mode)
            expected_depth = 4 if mode in ('auto-k4', 'explicit-k4') else int(
                mode in ('auto', 'auto-manual', 'auto-full', 'k1', 'commands-auto', 'commands-k1'))
            assert raw['mtp'] == expected and raw['mtp_max_drafts'] == expected_depth, (label, mode)
            if mode in ('auto-k4', 'explicit-k4'):
                assert raw['future_shape_hints'] == 0, (label, mode)
            if mode == 'on-zero': assert raw['draft_rounds'] == 0, (label, mode)
            assert raw['draft_vocabulary_rows'] == 0, (label, mode)
            assert raw['live_committed_versions'] == 0, (label, mode)
            reports[mode] = raw['sessions'][0]
        for actual, reference in (('auto', 'k1'), ('auto-manual', 'k1'), ('auto-full', 'k1'),
                                  ('commands-auto', 'commands-k1'), ('auto-k4', 'explicit-k4'),
                                  ('auto-zero', 'off'), ('off-zero', 'off'),
                                  ('random-auto', 'random-off')):
            assert reports[actual]['result']['generated_ids'] == reports[reference]['result']['generated_ids'], (label, actual)
            assert reports[actual]['ledger'] == reports[reference]['ledger'], (label, actual)
        conflict = subprocess.run([str(args.hyle), '--inputs', str(inputs), '--backend', 'cuda',
            '--capacity', '32768', '--schedule', str(args.schedule), '--mtp', 'off', '--mtp-max-drafts', '1'],
            capture_output=True, timeout=180)
        assert conflict.returncode != 0 and b'--mtp-max-drafts requires' in conflict.stderr, label
        print('PASS', label, 'automatic full K1, explicit off and random fallback', flush=True)


if __name__ == '__main__':
    main()
