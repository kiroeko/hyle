import json
from pathlib import Path
import tempfile
import unittest

from engine.tokenizer_identity import bind_source_tokenizer, semantic_identity, validate_source_binding


class Tokenizer:
    all_special_ids = [2]
    eos_token_id = 2
    vocab = {'a': 0, 'b': 1, '<eos>': 2}
    merges = ['a b']
    padding = None
    chat_template = 'presentation'

    @property
    def backend_tokenizer(self): return self
    def get_vocab(self): return self.vocab
    def to_str(self):
        return json.dumps(dict(model=dict(vocab=self.vocab, merges=self.merges),
                               normalizer=None, padding=self.padding, truncation=None))


class TokenizerIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.scratch')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tokenizer-binding.tsv'
        self.original = Tokenizer()
        bind_source_tokenizer(self.path, 'a' * 64, 3, self.original)

    def test_swapped_ids_same_size_and_changed_segmentation_are_rejected(self):
        for attribute, value in (('vocab', {'a': 1, 'b': 0, '<eos>': 2}), ('merges', ['b a']),
                                 ('eos_token_id', 1)):
            changed = Tokenizer(); setattr(changed, attribute, value)
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate_source_binding(self.path, 'a' * 64, 3, changed)
            with self.assertRaisesRegex(ValueError, 'original source'):
                bind_source_tokenizer(self.path, 'a' * 64, 3, self.original, changed)

    def test_source_and_vocabulary_are_bound_and_loading_never_enrolls(self):
        for source, vocabulary in (('b' * 64, 3), ('a' * 64, 4)):
            with self.assertRaises(ValueError):
                validate_source_binding(self.path, source, vocabulary, self.original)
        missing = self.path.with_name('missing.tsv')
        with self.assertRaisesRegex(ValueError, 'rerun source preparation'):
            validate_source_binding(missing, 'a' * 64, 3, self.original)
        self.assertFalse(missing.exists())

    def test_template_and_transient_encode_options_do_not_change_semantics(self):
        changed = Tokenizer(); changed.padding = {'length': 32}; changed.chat_template = 'new template'
        self.assertEqual(semantic_identity(self.original), semantic_identity(changed))
        validate_source_binding(self.path, 'a' * 64, 3, changed)
        bind_source_tokenizer(self.path, 'a' * 64, 3, self.original, changed)


if __name__ == '__main__': unittest.main()
