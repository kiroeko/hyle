from collections import Counter
from contextlib import redirect_stderr
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('recorded_draft_vocabulary', ROOT / 'scripts/prepare-mtp-draft-vocabulary.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
import draft_vocabulary_preparation as P


class PreparationTokenizer:
    all_special_ids = [7]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return [5, 5, 2] if text else []

    def decode(self, ids):
        return ' '.join(map(str, ids))


class DefaultPreparation(unittest.TestCase):
    def test_tokenizer_selection_is_deterministic_and_keeps_special_ids(self):
        ids, fingerprint = P.tokenizer_default_ids(PreparationTokenizer(), 8)
        self.assertEqual(ids, [0, 2, 5, 7])
        self.assertEqual((ids, fingerprint), P.tokenizer_default_ids(PreparationTokenizer(), 8))
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(P.tokenizer_default_ids(PreparationTokenizer(), 8, 1)[0], [7])
        self.assertEqual(P.tokenizer_default_ids(PreparationTokenizer(), 8, 5)[0], [0,1,2,5,7])

    def test_invalid_token_ids_and_vocabulary_are_rejected(self):
        tokenizer = PreparationTokenizer()
        tokenizer.all_special_ids = [8]
        with self.assertRaises(ValueError): P.tokenizer_default_ids(tokenizer, 8)
        tokenizer.all_special_ids = [7]
        with self.assertRaises(ValueError): P.tokenizer_default_ids(tokenizer, 2)
        with self.assertRaises(ValueError): P.tokenizer_default_ids(tokenizer, 0)
        with self.assertRaises(ValueError): P.tokenizer_default_ids(PreparationTokenizer(), 8, 9)

    def test_multiple_languages_keep_shared_code_and_special_tokens(self):
        class ProfileTokenizer(PreparationTokenizer):
            def encode(self, text, **kwargs):
                return {'english': [1], 'chinese': [2], 'code': [3], 'arabic': [4]}[text]
        with patch.object(P, 'PROMPTS', ('english', 'chinese', 'code', 'arabic')), \
             patch.object(P, '_PROMPT_GROUPS', ('en', 'zh', 'code', 'ar')):
            selected = P.language_preferences(' ZH, en,zh ')
            self.assertEqual(P.calibration_prompts(selected), ('english', 'chinese', 'code'))
            self.assertEqual(P.tokenizer_default_ids(ProfileTokenizer(), 8, 4, selected)[0], [1,2,3,7])
        for value in ('', 'en,', 'zh,123', 'zh,bad!'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                P.language_preferences(value)
        self.assertEqual(P.language_preferences('en,xx'), frozenset({'en','xx'}))
        hungarian = P.calibration_prompts(P.language_preferences('hu,en'))
        self.assertTrue(any('magyarul' in prompt for prompt in hungarian))
        self.assertTrue(any('C++20' in prompt for prompt in hungarian))
        with self.assertRaises(ValueError): P.calibration_prompts(frozenset({'xx'}))

    def test_local_gguf_prepare_writes_only_quantized_mtp_default(self):
        spec = importlib.util.spec_from_file_location('qwen_prepare_test', ROOT/'engine/models/qwen35/prepare.py')
        prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)
        (ROOT/'.scratch').mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='tokenizer-default-', dir=ROOT/'.scratch'))
        model = work/'model'; model.mkdir(); (model/'config.json').write_text('{}')
        tokenizer_dir=work/'separate-tokenizer';tokenizer_dir.mkdir()
        class Tokenizer(PreparationTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return {'input_ids':[1, 2]} if kwargs['tokenize'] else messages[0]['content']
        tokenizer = Tokenizer()
        seen=[]
        def from_pretrained(path, local_files_only):
            self.assertTrue(local_files_only);seen.append(path);return tokenizer
        auto = types.SimpleNamespace(from_pretrained=from_pretrained)
        fake_transformers = types.SimpleNamespace(AutoTokenizer=auto)
        identity = 'a'*64
        cases=(('Q5_K',True,'auto',True,None),('BF16',True,'auto',False,None),('Q5_K',False,'auto',False,None),
               ('Q5_K',True,'none',False,None),('BF16',True,'tokenizer',True,None),
               ('BF16',True,'auto',True,'zh,en,hu'))
        for dtype, mtp, mode, expect_artifact, languages in cases:
            with self.subTest(dtype=dtype, mtp=mtp, mode=mode):
                out = work/f'{dtype}-{mtp}-{mode}-{languages}'
                geometry = {'tied_head':True, 'vocabulary':8}
                receipt = dict(source_id=identity, source_representation='gguf_converted',
                               geometry=geometry, physical_roles=1)
                lines = ['signature', identity, 'geometry', 'source',
                         f'model.language_model.embed_tokens.weight\ttoken_embd.weight\t{dtype}']
                gguf = types.SimpleNamespace(prepare_role_files=lambda path, enabled: (lines, receipt))
                argv = ['prepare', '--model', str(model), '--output', str(out),
                        '--gguf-role-files', str(work/'unused'),'--tokenizer-dir',str(tokenizer_dir)]
                if mtp: argv.append('--mtp')
                if mode != 'auto': argv.extend(('--draft-vocabulary',mode))
                if languages: argv.extend(('--draft-languages',languages))
                with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'transformers':fake_transformers, 'gguf_roles':gguf}), \
                     patch.object(prepare.geometry, 'role_shapes', return_value={'mtp.test':(1,)}):
                    prepare.main()
                artifact = out/'mtp-draft-vocabulary.ids'
                self.assertEqual(seen[-1],tokenizer_dir)
                self.assertEqual(json.loads((out/'input.json').read_text())['tokenizer_directory'],str(tokenizer_dir.resolve()))
                self.assertEqual(artifact.exists(), expect_artifact)
                if artifact.exists():
                    self.assertEqual(artifact.read_text().splitlines(),
                        ['hyle-draft-vocabulary-v2', identity, '8', 'recommended_drafts none', 'head original', '0 2 5 7'])
                    self.assertEqual(json.loads((out/'input.json').read_text())['draft_vocabulary']['rows'], 4)
                    if languages:
                        self.assertEqual(json.loads((out/'input.json').read_text())['draft_vocabulary']['languages'], ['en','hu','zh'])


class RecordedCalibration(unittest.TestCase):
    def setUp(self):
        (ROOT/'.scratch').mkdir(exist_ok=True)
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
        self.assertEqual(V.selected_ids(Counter({5:2, 2:1}), 8, 4, frozenset({7})), [0,2,5,7])
        with self.assertRaises(ValueError): V.selected_ids(Counter({5:2}), 8, 1, frozenset({6,7}))
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

    def test_language_filter_keeps_code_and_never_reads_other_groups(self):
        folder = self.work / 'languages'; folder.mkdir()
        (folder / 'prompts.tsv').write_text('split\tindex\tgroup\tprompt\n'
            'train\t1\ten\tEnglish\ntrain\t2\tzh\tChinese\ntrain\t3\tcode\tCode\n'
            'train\t4\tar\tNot requested\nvalidation\t5\ten\tHeld out\n')
        for index, group, text, ids in ((1,'en','English',[1,1]), (2,'zh','Chinese',[2]),
                                         (3,'code','Code',[3])):
            location=folder/f'train-{index:02}-{group}'
            self.output(location, ids)
            (location/'prompt.txt').write_text(text+'\n')
        self.assertEqual(V.recorded_counts(folder, self.identity, 8, frozenset({'en','zh'})),
                         (Counter({1:2,2:1,3:1}), 3))
        with self.assertRaises(ValueError):
            V.recorded_counts(self.legacy(), self.identity, 8, frozenset({'en'}))

    def test_cli_generates_source_bound_multilanguage_artifact(self):
        folder = self.work / 'languages'; folder.mkdir()
        (folder/'prompts.tsv').write_text('split\tindex\tgroup\tprompt\n'
            'train\t1\ten\tEnglish\ntrain\t2\tzh\tChinese\ntrain\t3\tcode\tCode\n'
            'train\t4\tar\tNot requested\n')
        for index, group, text, ids in ((1,'en','English',[1,1]), (2,'zh','Chinese',[2,2]),
                                         (3,'code','Code',[3])):
            location=folder/f'train-{index:02}-{group}'; self.output(location,ids)
            (location/'prompt.txt').write_text(text+'\n')
        output=self.work/'selected.ids'; report=self.work/'report'
        argv=['prepare','--inputs',str(self.inputs),'--work-dir',str(report),'--output',str(output),
              '--rows','4','--languages','zh,en','--calibration-run',str(folder)]
        with patch.object(sys,'argv',argv), patch.object(V.subprocess,'run',side_effect=AssertionError('unexpected subprocess')):
            V.main()
        self.assertEqual(output.read_text().splitlines()[0:3],
                         ['hyle-draft-vocabulary-v2',self.identity,'8'])
        self.assertEqual(output.read_text().splitlines()[-1], '0 1 2 3')
        self.assertIn('prompts=3', (report/'calibration.txt').read_text())
        self.assertIn('languages=en,zh; shared=code', (report/'calibration.txt').read_text())

    def test_selected_language_without_training_data_fails_before_output(self):
        folder=self.work/'languages';folder.mkdir()
        (folder/'prompts.tsv').write_text('split\tindex\tgroup\tprompt\ntrain\t1\ten\tEnglish\n')
        self.output(folder/'train-01-en',[1,1]);(folder/'train-01-en/prompt.txt').write_text('English\n')
        output=self.work/'selected.ids'
        argv=['prepare','--inputs',str(self.inputs),'--work-dir',str(self.work/'report'),
              '--output',str(output),'--rows','4','--languages','en,zh','--calibration-run',str(folder)]
        with patch.object(sys,'argv',argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): V.main()
        self.assertFalse(output.exists())

    def test_custom_corpus_can_supply_other_language_tags(self):
        folder=self.work/'custom';folder.mkdir()
        (folder/'prompts.tsv').write_text('split\tindex\tgroup\tprompt\ntrain\t1\txx\tCustom\n')
        self.output(folder/'train-01-xx',[4,4]);(folder/'train-01-xx/prompt.txt').write_text('Custom\n')
        self.assertEqual(V.recorded_counts(folder, self.identity, 8, P.language_preferences('xx')),
                         (Counter({4:2}), 1))

    def test_live_language_calibration_reserves_special_ids_and_checks_source(self):
        model=self.work/'hyle.exe';model.write_bytes(b'existing')
        tokenizer_dir=self.work/'tokenizer';tokenizer_dir.mkdir()
        tokenizer=PreparationTokenizer()
        fake_transformers=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer))
        output=self.work/'live.ids'; report=self.work/'live-report'
        argv=['prepare','--inputs',str(self.inputs),'--work-dir',str(report),'--output',str(output),
              '--hyle',str(model),'--tokenizer-dir',str(tokenizer_dir),'--rows','1',
              '--tokens','2','--languages','hu,en']
        calls=[]
        def run(command, *, stdout, stderr, timeout):
            calls.append(command)
            raw=dict(source_id=self.identity,sampling='greedy',mtp_max_drafts=0,
                     live_committed_versions=0,sessions=[dict(result=dict(generated_ids=[2,3]))])
            stdout.write(json.dumps(raw).encode('utf8'))
            return types.SimpleNamespace(returncode=0)
        with patch.object(sys,'argv',argv), patch.dict(sys.modules,{'transformers':fake_transformers}), \
             patch.object(V.subprocess,'run',side_effect=run):
            V.main()
        self.assertEqual(len(calls), len(P.calibration_prompts(P.language_preferences('hu,en'))))
        self.assertEqual(output.read_text().splitlines()[-1], '7')
        self.assertIn('languages=en,hu; shared=code', (report/'calibration.txt').read_text())

    def test_live_calibration_rejects_wrong_model_before_publishing(self):
        model=self.work/'hyle.exe';model.write_bytes(b'existing')
        tokenizer_dir=self.work/'tokenizer';tokenizer_dir.mkdir()
        fake_transformers=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: PreparationTokenizer()))
        output=self.work/'bad.ids'
        argv=['prepare','--inputs',str(self.inputs),'--work-dir',str(self.work/'bad-report'),
              '--output',str(output),'--hyle',str(model),'--tokenizer-dir',str(tokenizer_dir),
              '--rows','4','--tokens','2','--languages','hu']
        def run(command, *, stdout, stderr, timeout):
            raw=dict(source_id='b'*64,sampling='greedy',mtp_max_drafts=0,
                     live_committed_versions=0,sessions=[dict(result=dict(generated_ids=[2,3]))])
            stdout.write(json.dumps(raw).encode('utf8'))
            return types.SimpleNamespace(returncode=0)
        with patch.object(sys,'argv',argv), patch.dict(sys.modules,{'transformers':fake_transformers}), \
             patch.object(V.subprocess,'run',side_effect=run):
            with self.assertRaises(ValueError): V.main()
        self.assertFalse(output.exists())

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
        self.assertEqual(output.read_text().splitlines()[3:],
                         ['recommended_drafts none', 'head original', '0 1 2 5'])
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

    def test_measured_recommendation_is_bound_to_the_source(self):
        folder = self.legacy(); output = self.work / 'selected.ids'
        argv = ['prepare', '--inputs', str(self.inputs), '--work-dir', str(self.work / 'report'),
                '--output', str(output), '--rows', '4', '--recommended-drafts', '3',
                '--calibration-run', str(folder)]
        with patch.object(sys, 'argv', argv): V.main()
        self.assertEqual(output.read_text().splitlines(),
                         ['hyle-draft-vocabulary-v2', self.identity, '8',
                          'recommended_drafts 3', 'head original', '0 1 2 5'])

    def test_invalid_recommendation_does_not_publish_an_artifact(self):
        folder = self.legacy(); output = self.work / 'selected.ids'
        argv = ['prepare', '--inputs', str(self.inputs), '--work-dir', str(self.work / 'report'),
                '--output', str(output), '--rows', '4', '--recommended-drafts', '0',
                '--calibration-run', str(folder)]
        with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): V.main()
        self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
