"""Independent optional preparation capabilities; no weights or backend needed."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from engine import model_preparation as catalog


class PreparationCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.scratch')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def provider(self, **capabilities):
        return catalog.LocalPreparationProvider(matches=lambda path: path == self.root,
            validate_tokenizer=lambda path, tokenizer: ('a' * 64, 100),
            source_identity=lambda path: ('a' * 64, 100), **capabilities)

    def test_text_only_model_does_not_require_draft_or_visual_preparers(self):
        with patch.object(catalog, '_PREPARERS', (self.provider(),)):
            self.assertEqual(catalog.prepared_source_identity(self.root), ('a' * 64, 100))
            self.assertEqual(catalog.validate_local_tokenizer(self.root, object()), ('a' * 64, 100))
            operations = (
                ('derived assets', lambda: catalog.prepare_local_assets(self.root, self.root, object(), self.root / 'cache')),
                ('vision assets', lambda: catalog.prepare_local_vision(self.root, self.root, self.root / 'vision.tsv')),
                ('image processing', lambda: catalog.local_image_processor(self.root, self.root, object(), context=512)),
                ('vocabulary writing', lambda: catalog.write_prepared_vocabulary(self.root, object(), [1], None,
                    expected_source=('a' * 64, 100))),
            )
            for capability, operation in operations:
                with self.subTest(capability=capability), self.assertRaisesRegex(ValueError, capability):
                    operation()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_visual_asset_and_image_processing_capabilities_are_independent(self):
        prepare = Mock(return_value='visual-identity')
        provider = self.provider(prepare_vision=prepare)
        output, original = self.root / 'vision.tsv', self.root / 'visual.gguf'
        with patch.object(catalog, '_PREPARERS', (provider,)):
            self.assertEqual(catalog.prepare_local_vision(self.root, self.root, output, mmproj=original), 'visual-identity')
            prepare.assert_called_once_with(self.root, self.root, output, mmproj=original)
            with self.assertRaisesRegex(ValueError, 'image processing'):
                catalog.local_image_processor(self.root, self.root, object(), context=512)
        process = Mock(return_value='processor')
        with patch.object(catalog, '_PREPARERS', (replace(provider, prepare_vision=None, image_processor=process),)):
            tokenizer = object()
            self.assertEqual(catalog.local_image_processor(self.root, self.root, tokenizer, context=512), 'processor')
            process.assert_called_once_with(self.root, tokenizer, context=512)
            with self.assertRaisesRegex(ValueError, 'vision assets'):
                catalog.prepare_local_vision(self.root, self.root, output)

    def test_vocabulary_writer_still_requires_the_actual_source_identity(self):
        write = Mock()
        provider = self.provider(write_vocabulary=write)
        with patch.object(catalog, '_PREPARERS', (provider,)):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                catalog.write_prepared_vocabulary(self.root, object(), [1], 1, expected_source=('b' * 64, 100))
            write.assert_not_called()
            output = object()
            catalog.write_prepared_vocabulary(self.root, output, [1, 2], 1, expected_source=('a' * 64, 100))
            write.assert_called_once_with(output, 'a' * 64, 100, [1, 2], 1)


if __name__ == '__main__':
    unittest.main()
