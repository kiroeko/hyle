"""Check K3 selection from a source-bound vocabulary and original GGUF head."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess


def check(args):
    args.output.mkdir(parents=True)
    (args.output / 'prompt.ids').write_text('1 2 3\n', encoding='utf-8')
    schedule = args.output / 'schedule.txt'
    schedule.write_text('prompt prompt.ids\ngenerate 17\n', encoding='utf-8')
    for label, source in (('Q8', args.q8_inputs), ('Q4', args.q4_inputs)):
        fixture = args.output / label
        fixture.mkdir()
        shutil.copyfile(source / 'weights.tsv', fixture / 'weights.tsv')
        marker = fixture / 'mtp-draft-vocabulary-k3.ids'
        shutil.copyfile(source / 'mtp-draft-vocabulary.ids', marker)

        def run(name, flags=(), failure=None):
            log = fixture / name
            log.mkdir()
            command = [str(args.hyle), '--inputs', str(fixture), '--backend', 'cuda',
                       '--capacity', '2048', '--schedule', str(schedule), *flags]
            (log / 'command.txt').write_text(subprocess.list2cmdline(command) + '\n', encoding='utf-8')
            with (log / 'stdout.log').open('wb') as stdout, (log / 'stderr.log').open('wb') as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180)
            error = (log / 'stderr.log').read_text(encoding='utf-8')
            if failure:
                assert result.returncode and failure in error, (label, name, error)
                assert 'startup_to_ready_seconds=' not in error
                return None
            assert result.returncode == 0, (label, name, error)
            report = json.loads((log / 'stdout.log').read_text(encoding='utf-8'))
            assert report['live_committed_versions'] == report['future_shape_hints'] == 0
            return report

        auto = run('auto')
        assert auto['mtp'] and auto['mtp_max_drafts'] == 3
        assert (auto['mtp_selection_reason'], auto['mtp_depth_source']) == ('prepared_vocabulary', 'k3_artifact')
        assert auto['draft_vocabulary_rows'] > 0
        assert auto['draft_head_rows'] == auto['draft_vocabulary_rows']
        manual = run('manual', ['--mtp', 'on', '--mtp-max-drafts', '3', '--mtp-draft-head', 'original',
                                '--mtp-draft-vocabulary', str(marker)])
        assert auto['sessions'][0]['result']['generated_ids'] == manual['sessions'][0]['result']['generated_ids']
        assert auto['sessions'][0]['ledger'] == manual['sessions'][0]['ledger']
        k1 = run('explicit-k1', ['--mtp-max-drafts', '1'])
        assert k1['mtp'] and k1['mtp_max_drafts'] == 1 and k1['mtp_depth_source'] == 'explicit'
        original = marker.read_text(encoding='utf-8').splitlines()
        original[1] = '0' * 64
        marker.write_text('\n'.join(original) + '\n', encoding='utf-8')
        run('stale', failure='source identity differs')
        off = run('disabled', ['--mtp', 'off'])
        assert not off['mtp'] and not off['draft_vocabulary_rows']
        print(f'PASS {label}: source-bound K3 original head, explicit depth, stale refusal and retirement', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hyle', type=Path, required=True)
    p.add_argument('--q8-inputs', type=Path, required=True)
    p.add_argument('--q4-inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    check(p.parse_args())
