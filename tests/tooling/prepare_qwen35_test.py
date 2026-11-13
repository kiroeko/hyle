"""Offline preparation entrypoint checks; no model download or execution."""
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('prepare_qwen35_tool',ROOT/'scripts/prepare-qwen35.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
from engine.tokenizer_identity import bind_source_tokenizer


class Tokenizer:
    all_special_ids=[7]
    def __init__(self):
        self.backend_tokenizer=types.SimpleNamespace(to_str=lambda:json.dumps(
            dict(model=dict(type='WordLevel',vocab=self.get_vocab()))))
    def get_vocab(self): return {f't{i}':i for i in range(8)}
    def apply_chat_template(self,messages,**kwargs): return messages[0]['content']
    def encode(self,text,**kwargs): return [5,5,2]


class PrepareQwen35Test(unittest.TestCase):
    def setUp(self):
        (ROOT/'.scratch').mkdir(exist_ok=True)
        self.work=Path(tempfile.mkdtemp(prefix='prepare-qwen35-',dir=ROOT/'.scratch'))
        self.inputs=self.work/'inputs';self.inputs.mkdir()
        self.identity='a'*64
        (self.inputs/'weights.tsv').write_text('HYLE_QWEN_ROLE_FILES_V2\n'+self.identity+
            '\ngeometry 4 8 8\nmtp.fc.weight\tBF16\n',encoding='utf-8')
        self.tokenizer=self.work/'tokenizer';self.tokenizer.mkdir()
        bind_source_tokenizer(self.inputs/'tokenizer-binding.tsv',self.identity,8,Tokenizer())
        self.out=self.work/'assets'

    def fake_transformers(self):
        def load(path, *, local_files_only, trust_remote_code, use_fast):
            self.assertTrue(local_files_only)
            self.assertFalse(trust_remote_code)
            self.assertTrue(use_fast)
            return Tokenizer()
        return types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=load))

    def test_existing_inputs_create_reusable_subset_without_execution(self):
        args=['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
              '--tokenizer-dir',str(self.tokenizer),'--draft-rows','5']
        with patch.dict(sys.modules,{'transformers':self.fake_transformers()}), \
             patch.object(tool.subprocess,'run',side_effect=AssertionError('unexpected model execution')):
            tool.main(args)
        self.assertEqual((self.out/'mtp-draft-vocabulary.ids').read_text().splitlines(),
            ['hyle-draft-vocabulary-v2',self.identity,'8','recommended_drafts none','head original','0 1 2 5 7'])
        self.assertFalse((self.out/'weights.tsv').exists())

    def test_optional_head_publishes_reusable_metadata_and_preference(self):
        library=self.work/'ggml.dll';library.write_bytes(b'local')
        version='llama.cpp@'+'b'*40
        def make_head(inputs,lib,actual_version,output,*,compact_vocabulary):
            self.assertEqual((inputs,lib,actual_version),(self.inputs.resolve(),library.resolve(),version))
            self.assertEqual(compact_vocabulary,self.out/'mtp-draft-vocabulary.ids')
            output.mkdir();(output/'draft-head-q4_k.bin').write_bytes(b'head')
            (output/'mtp-draft-head.tsv').write_text('HYLE_QWEN_DRAFT_HEAD_Q4_K_V1\n'+self.identity+
                '\n'+'c'*64+'\n'+version+':Q4_K:reference:no-imatrix\n8 4\n"draft-head-q4_k.bin" 4\n')
        head=types.SimpleNamespace(source_head=lambda path:None,prepare=make_head)
        args=['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
              '--tokenizer-dir',str(self.tokenizer),'--draft-head-q4',str(library),'--quantizer-version',version]
        with patch.dict(sys.modules,{'transformers':self.fake_transformers()}),patch.object(tool,'head_preparer',return_value=head):
            tool.main(args)
        self.assertEqual((self.out/'mtp-draft-vocabulary.ids').read_text().splitlines()[4],'head prepared')
        self.assertEqual((self.out/'mtp-draft-head.tsv').read_text().splitlines()[5],
                         '"draft-head/draft-head-q4_k.bin" 4')
        self.assertEqual((self.out/'draft-head/draft-head-q4_k.bin').read_bytes(),b'head')

    def test_model_mode_forwards_direct_local_gguf(self):
        model=self.work/'model';model.mkdir()
        gguf=self.work/'local.gguf';gguf.write_bytes(b'local')
        def fake_prepare(argv):
            self.assertIn('--gguf',argv);self.assertIn(str(gguf.resolve()),argv)
            self.assertIn('--mtp',argv);self.assertIn('--draft-rows',argv)
            output=Path(argv[argv.index('--output')+1]);output.mkdir()
            (output/'weights.tsv').write_text((self.inputs/'weights.tsv').read_text())
        with patch.object(tool.prepare,'main',side_effect=fake_prepare):
            tool.main(['--model',str(model),'--gguf',str(gguf),'--output',str(self.out),'--mtp',
                       '--draft-vocabulary','tokenizer','--draft-rows','5'])
        self.assertTrue((self.out/'weights.tsv').is_file())

    def test_calibration_reuse_delegates_without_tokenizer_or_hyle(self):
        run=self.work/'calibration';run.mkdir()
        def fake_run(command,cwd,check):
            self.assertNotIn('--hyle',command);self.assertNotIn('--tokenizer-dir',command)
            self.assertIn(str(run.resolve()),command)
            output=Path(command[command.index('--output')+1])
            output.write_text('hyle-draft-vocabulary-v2\n'+self.identity+
                              '\n8\nrecommended_drafts none\nhead original\n0 1 2 3\n')
            return subprocess.CompletedProcess(command,0)
        with patch.object(tool.subprocess,'run',side_effect=fake_run),patch.dict(sys.modules,{'transformers':None}):
            tool.main(['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
                       '--draft-vocabulary','calibrated','--calibration-run',str(run)])
        self.assertEqual((self.out/'mtp-draft-vocabulary.ids').read_text().splitlines()[-1],'0 1 2 3')

    def test_model_calibration_is_only_started_when_requested(self):
        hyle=self.work/'hyle.exe';hyle.write_bytes(b'local')
        def fake_run(command,cwd,check):
            self.assertEqual(command[command.index('--hyle')+1],str(hyle.resolve()))
            self.assertEqual(command[command.index('--tokenizer-dir')+1],str(self.tokenizer.resolve()))
            self.assertEqual(command[command.index('--rows')+1],'5')
            output=Path(command[command.index('--output')+1])
            output.write_text('hyle-draft-vocabulary-v2\n'+self.identity+
                              '\n8\nrecommended_drafts none\nhead original\n0 1 2 3 4\n')
            return subprocess.CompletedProcess(command,0)
        with patch.object(tool.subprocess,'run',side_effect=fake_run):
            tool.main(['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
                       '--draft-vocabulary','calibrated','--hyle',str(hyle),
                       '--tokenizer-dir',str(self.tokenizer),'--draft-rows','5'])
        self.assertEqual((self.out/'mtp-draft-vocabulary.ids').read_text().splitlines()[-1],'0 1 2 3 4')

    def test_invalid_combinations_do_not_publish_output(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                tool.main(['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
                           '--draft-vocabulary','none'])
        self.assertFalse(self.out.exists())
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                tool.main(['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
                           '--draft-vocabulary','calibrated'])
        self.assertFalse(self.out.exists())
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                tool.main(['--inputs',str(self.inputs),'--output',str(self.out),'--mtp',
                           '--tokenizer-dir',str(self.tokenizer),'--draft-rows','9'])
        self.assertFalse(self.out.exists())


if __name__=='__main__':
    unittest.main()
