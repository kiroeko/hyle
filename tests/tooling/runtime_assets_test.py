from pathlib import Path
import struct
import tempfile
import threading
import unittest

from engine.asset_store import AssetStore
from engine.models.qwen35.runtime_assets import prepare_assets

ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    all_special_ids = [7]
    calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        self.template = kwargs.get('chat_template')
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return [5, 2, 5]

    def convert_ids_to_tokens(self, index):
        return 'token\t' + str(index)


class RuntimeAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.scratch')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'inputs'; self.inputs.mkdir()
        self.tokenizer = self.root / 'tokenizer'; self.tokenizer.mkdir()
        (self.tokenizer / 'tokenizer.json').write_text('{}')
        self.payload = b''.join(bytes([j]) * 512 for j in range(8))
        (self.inputs / 'head.bin').write_bytes(self.payload)
        (self.inputs / 'weights.tsv').write_text('HYLE_QWEN_ROLE_FILES_V2\n' + 'a' * 64 + '\n'
            'geometry 256 512 8\nmodel.language_model.embed_tokens.weight BF16 2 8 256 "head.bin" 0 4096\n')
        self.store = AssetStore(self.root / 'cache')

    def prepare(self, tokenizer=None, **kwargs):
        return prepare_assets(self.inputs, self.tokenizer, tokenizer or Tokenizer(), self.store, rows=4, **kwargs)

    def test_publish_and_reuse_preserves_original_row_bytes(self):
        tokenizer = Tokenizer(); first = self.prepare(tokenizer)
        calls = tokenizer.calls
        second = self.prepare(tokenizer)
        self.assertTrue(first.created); self.assertFalse(second.created)
        self.assertEqual(first.directory, second.directory)
        self.assertEqual(tokenizer.calls, calls)
        self.assertEqual((first.directory / 'draft-head.bin').read_bytes(),
                         b''.join(self.payload[j * 512:(j + 1) * 512] for j in (0, 2, 5, 7)))
        self.assertIn('HYLE_QWEN_DRAFT_HEAD_STORED_SUBSET_V1', first.draft_head.read_text())
        self.assertEqual(len(first.vocabulary.read_text().splitlines()), 12)

    def test_language_tokenizer_and_template_are_immutable_recipe_inputs(self):
        first = self.prepare(languages='zh,en')
        hu = self.prepare(languages='hu')
        self.assertNotEqual(first.directory, hu.directory)
        template = self.root / 'template.jinja'; template.write_text('custom')
        tokenizer = Tokenizer(); custom = self.prepare(tokenizer, languages='hu', template=template)
        self.assertEqual(tokenizer.template, 'custom')
        self.assertNotEqual(hu.directory, custom.directory)
        (self.tokenizer / 'tokenizer.json').write_text('{"changed":true}')
        self.assertNotEqual(self.prepare(languages='zh,en').directory, first.directory)
        self.assertTrue(first.draft_head.exists())

    def test_corruption_is_rejected_without_replacing_a_live_entry(self):
        result = self.prepare()
        payload = result.directory / 'draft-head.bin'
        payload.write_bytes(b'x' * payload.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.prepare()
        self.assertEqual(payload.read_bytes(), b'x' * len(payload.read_bytes()))

    def test_failed_preparation_does_not_publish_and_can_be_retried(self):
        def fail(path):
            (path / 'partial').write_bytes(b'unfinished')
            raise ValueError('interrupted')
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            self.store.prepare('failed-recipe', fail)
        ready, created = self.store.prepare('failed-recipe', lambda p: (p / 'complete').write_bytes(b'ok'))
        self.assertTrue(created); self.assertEqual((ready / 'complete').read_bytes(), b'ok')

    def test_concurrent_preparers_publish_once(self):
        results, errors = [], []
        gate = threading.Barrier(2)
        def run():
            try:
                gate.wait(timeout=5); results.append(self.prepare())
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=10)
        self.assertFalse(errors); self.assertEqual(len(results), 2)
        self.assertEqual(results[0].directory, results[1].directory)
        self.assertEqual(sum(result.created for result in results), 1)


if __name__ == '__main__':
    unittest.main()
