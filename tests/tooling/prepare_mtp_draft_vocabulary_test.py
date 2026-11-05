from collections import Counter
from contextlib import redirect_stderr
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('recorded_draft_vocabulary', ROOT / 'scripts/prepare-mtp-draft-vocabulary.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)


class RecordedCalibration(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='recorded-vocabulary-', dir=ROOT / '.scratch'))
        self.identity = 'a' * 64
        self.inputs = self.work / 'inputs'; self.inputs.mkdir()
        (self.inputs / 'weights.tsv').write_text('HYLE_QWEN_ROLE_FILES_V2\n' + self.identity + '\ngeometry 256 512 8\n')

    def output(self, folder, ids, identity=None):
        folder.mkdir(parents=True)
        raw = dict(source_id=identity or self.identity, sampling='greedy', mtp_max_drafts=0,
                   live_committed_versions=0, sessions=[dict(result=dict(generated_ids=ids))])
        (folder / 'stdout.log').write_text(json.dumps(raw), encoding='utf-8')
        (folder / 'generated.ids').write_text(' '.join(map(str, ids)))

    def legacy(self, name='legacy'):
        folder = self.work / name; folder.mkdir()
        (folder / 'calibration.txt').write_text('source=' + self.identity + '\n')
        self.output(folder / 'prompt01', [5, 5, 2])
        return folder

    def test_frequency_order_and_original_id_fill_are_unchanged(self):
        self.assertEqual(V.selected_ids(Counter({5:2, 2:1}), 8, 4), [0,1,2,5])
        self.assertEqual(V.selected_ids(Counter({5:2, 2:2}), 8, 1), [2])
        self.assertEqual(V.selected_ids(Counter({5:2, 2:1}), 8, 8), list(range(8)))
        for counts in (Counter(), Counter({8:1}), Counter({2:0}), Counter({2:-1})):
            with self.assertRaises(ValueError): V.selected_ids(counts, 8, 4)

    def test_mixed_corpus_never_reads_validation_outputs(self):
        folder = self.work / 'mixed'; folder.mkdir()
        (folder / 'prompts.tsv').write_text('split\tindex\tgroup\tprompt\ntrain\t1\ten\tIndependent training\n'
                                           'validation\t1\ten\tPoisoned heldout\n')
        self.output(folder / 'train-01-en', [1, 1, 3])
        (folder / 'train-01-en/prompt.txt').write_text('Independent training\n')
        poisoned = folder / 'validation-01-en'; poisoned.mkdir()
        (poisoned / 'stdout.log').write_text('THIS MUST NOT BE PARSED')
        self.assertEqual(V.recorded_counts(folder, self.identity, 8), (Counter({1:2, 3:1}), 1))

    def test_source_identity_and_id_file_must_both_match(self):
        folder = self.legacy()
        raw_path = folder / 'prompt01/stdout.log'
        original = json.loads(raw_path.read_text())
        changed = dict(original, source_id='b' * 64); raw_path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError): V.recorded_counts(folder, self.identity, 8)
        raw_path.write_text(json.dumps(original)); (folder / 'prompt01/generated.ids').write_text('5 5 7')
        with self.assertRaises(ValueError): V.recorded_counts(folder, self.identity, 8)

    def test_invalid_model_records_are_not_imported(self):
        folder = self.legacy(); path = folder / 'prompt01/stdout.log'; original = json.loads(path.read_text())
        for field, value in [('sampling','random'), ('mtp_max_drafts',1), ('live_committed_versions',1)]:
            changed = dict(original); changed[field] = value; path.write_text(json.dumps(changed))
            with self.subTest(field=field):
                with self.assertRaises(ValueError): V.recorded_counts(folder, self.identity, 8)

    def test_cli_reuse_has_no_executable_tokenizer_or_subprocess_dependency(self):
        first = self.legacy('first'); second = self.legacy('second')
        output = self.work / 'selected.ids'; report = self.work / 'report'
        argv = ['prepare', '--inputs', str(self.inputs), '--work-dir', str(report), '--output', str(output),
                '--rows', '4', '--calibration-run', str(first), '--calibration-run', str(second)]
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'transformers':None}), \
             patch.object(V.subprocess, 'run', side_effect=AssertionError('offline mode launched a subprocess')):
            V.main()
        self.assertEqual(output.read_text().split()[3:], ['0','1','2','5'])
        self.assertIn('generated=6', (report / 'calibration.txt').read_text())
        with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): V.main()

    def test_duplicate_run_cannot_double_count_or_publish(self):
        folder = self.legacy(); output = self.work / 'selected.ids'
        argv = ['prepare', '--inputs', str(self.inputs), '--work-dir', str(self.work / 'report'), '--output', str(output),
                '--calibration-run', str(folder), '--calibration-run', str(folder)]
        with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): V.main()
        self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
