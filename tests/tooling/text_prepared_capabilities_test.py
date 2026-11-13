"""Routing consumes prepared facts independently of model asset conventions."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from engine.text.configuration import ExecutionChoice, PreparedChoices, Selection
from engine.text.contracts import ExecutionProfileFacts, PreparedPrompt, PromptAttachment
from engine.text.runtime import TextEngine


class Codec:
    stop_ids = frozenset()
    image_calls = 0

    def encode_messages(self, messages):
        return [1, 2]

    def prepare_images(self, messages, **options):
        self.image_calls += 1
        return PreparedPrompt([1, 2], (PromptAttachment('other-model.image', 1, 1, (1,), b'pixels'),))


class PreparedCapabilitiesTest(unittest.TestCase):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': object()}]}]

    def engine(self, choices, facts):
        worker = SimpleNamespace(context=128, stop_ids=frozenset(), execution_profiles=facts)
        return TextEngine(worker, Codec(), 'model', choices=choices)

    def test_other_asset_roles_and_embedded_weights_do_not_change_image_routing(self):
        for assets in ((), (('other-model.encoder', Path('some.asset')),)):
            choices = PreparedChoices(executions=(ExecutionChoice(), ExecutionChoice('pictures', assets=assets)))
            facts = (ExecutionProfileFacts('default', 'ordinary', 0),
                     ExecutionProfileFacts('pictures', 'other-proposal', 2, ('text', 'image')))
            engine = self.engine(choices, facts)
            request = engine.prepare(self.messages, 8, Selection(max_draft_tokens=2))
            self.assertEqual(request.selection.execution_profile, 1)
            self.assertEqual(request.selection.max_draft_tokens, 2)
            self.assertEqual(request.attachments[0].format, 'other-model.image')

    def test_an_asset_name_never_claims_a_capability_the_executor_did_not_prepare(self):
        choices = PreparedChoices(executions=(ExecutionChoice(assets=(('vision.encoder', Path('image.asset')),)),))
        engine = self.engine(choices, (ExecutionProfileFacts('default', 'ordinary', 0),))
        with self.assertRaisesRegex(ValueError, 'image execution profile'):
            engine.prepare(self.messages, 8)
        self.assertEqual(engine.codec.image_calls, 0)

    def test_actual_depth_refuses_unknown_auto_or_stale_config_before_rendering(self):
        for declared in (None, 3):
            engine = self.engine(PreparedChoices(executions=(ExecutionChoice(drafts=declared),)),
                (ExecutionProfileFacts('default', 'ordinary', 0),))
            with self.assertRaisesRegex(ValueError, 'actual prepared'):
                engine.prepare([{'role': 'user', 'content': 'hello'}], 8, Selection(max_draft_tokens=1))

    def test_ambiguous_or_explicit_incompatible_image_choices_are_not_silently_replaced(self):
        choices = PreparedChoices(executions=(ExecutionChoice(), ExecutionChoice('second'), ExecutionChoice('third')))
        facts = (ExecutionProfileFacts('default', 'ordinary', 0),
                 ExecutionProfileFacts('second', 'ordinary', 0, ('text', 'image')),
                 ExecutionProfileFacts('third', 'ordinary', 0, ('text', 'image')))
        engine = self.engine(choices, facts)
        with self.assertRaisesRegex(ValueError, 'select a cold-prepared'):
            engine.prepare(self.messages, 8)
        self.assertEqual(engine.prepare(self.messages, 8, Selection(execution_profile=2)).selection.execution_profile, 2)
        facts = (facts[0], ExecutionProfileFacts('second', 'ordinary', 0), facts[2])
        engine = self.engine(choices, facts)
        with self.assertRaisesRegex(ValueError, 'select a cold-prepared'):
            engine.prepare(self.messages, 8, Selection(execution_profile=1))

    def test_profile_names_must_match_the_actual_native_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'differ from'):
            self.engine(PreparedChoices(), (ExecutionProfileFacts('another-model', 'ordinary', 0),))

    def test_declared_input_kinds_are_checked_for_text_too(self):
        engine = self.engine(PreparedChoices(), (ExecutionProfileFacts('default', 'image-only', 0, ('image',)),))
        with self.assertRaisesRegex(ValueError, 'text execution profile'):
            engine.prepare([{'role': 'user', 'content': 'hello'}], 8)


if __name__ == '__main__':
    unittest.main()
