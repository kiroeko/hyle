"""Check that explicit model-worker asset paths are not policy keywords.

Uses existing local Qwen3.5-2B assets. The caller owns and removes --fixture;
the test refuses an existing directory and does not download anything.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--q8-inputs', type=Path, required=True)
    parser.add_argument('--bf16-inputs', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    args = parser.parse_args()
    engine = args.engine.resolve(strict=True)
    q8 = args.q8_inputs.resolve(strict=True)
    bf16 = args.bf16_inputs.resolve(strict=True)
    schedule = args.schedule.resolve(strict=True)
    fixture = args.fixture.resolve()
    fixture.mkdir()
    shutil.copyfile(q8 / 'mtp-draft-vocabulary.ids', fixture / 'full')
    shutil.copyfile(bf16 / 'mtp-draft-head.tsv', fixture / 'original')

    def run(inputs, options, expected_error=None):
        command = [str(engine), '--engine-worker', '--inputs', str(inputs),
                   '--backend', 'cuda', '--capacity', '128', '--max-sessions', '1',
                   '--active-sequences', '1', '--draft-count', '1', *options]
        result = subprocess.run(command, cwd=fixture, input='Q\n', text=True,
                                capture_output=True, timeout=180)
        if expected_error:
            assert result.returncode and expected_error in result.stderr, (command, result.stdout, result.stderr)
        else:
            assert result.returncode == 0 and 'READY' in result.stdout and 'BYE' in result.stdout, (
                command, result.stdout, result.stderr)
        print('PASS', *options, flush=True)

    # These existing files have names formerly interpreted as policy keywords.
    run(q8, ['--draft-vocabulary', 'full'])
    run(bf16, ['--draft-vocabulary', str(bf16 / 'mtp-draft-vocabulary.ids'),
               '--draft-head', 'original'])
    # Verify the successful runs consumed the files rather than ignoring them.
    (fixture / 'full').write_text('invalid\n', encoding='utf-8')
    run(q8, ['--draft-vocabulary', 'full'], 'draft vocabulary')
    (fixture / 'original').write_text('invalid\n', encoding='utf-8')
    run(bf16, ['--draft-vocabulary', str(bf16 / 'mtp-draft-vocabulary.ids'),
               '--draft-head', 'original'], 'draft head')
    # A nonexistent file named auto must not silently turn on automatic selection.
    run(q8, ['--draft-vocabulary', 'auto'], 'draft vocabulary')
    run(bf16, ['--draft-vocabulary', str(bf16 / 'mtp-draft-vocabulary.ids'),
               '--draft-head', 'auto'], 'draft head')

    # The original Qwen CLI resolves its policy keywords at the entry point.
    # A relative path still selects the actual file on that entry point.
    shutil.copyfile(q8 / 'mtp-draft-vocabulary.ids', fixture / 'full')
    for vocabulary, is_file in [('full', False), ('./full', True)]:
        command = [str(engine), '--inputs', str(q8), '--backend', 'cuda',
                   '--capacity', '2048', '--schedule', str(schedule), '--mtp', 'on',
                   '--mtp-max-drafts', '1', '--mtp-draft-vocabulary', vocabulary,
                   '--mtp-draft-head', 'original']
        result = subprocess.run(command, cwd=fixture, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, (command, result.stderr)
        report = json.loads(result.stdout)
        assert report['mtp'] and (report['draft_vocabulary_rows'] > 0) == is_file, report
        assert report['sessions'][0]['result']['generated_ids'], report
        print('PASS legacy --mtp-draft-vocabulary', vocabulary, flush=True)


if __name__ == '__main__':
    main()
