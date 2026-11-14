from pathlib import Path
import tempfile
import unittest

from engine.text.app import make_worker_command, parse_arguments
from engine.text.configuration import read_configuration

ROOT = Path(__file__).resolve().parents[2]


class TextConfigurationTest(unittest.TestCase):
    def test_hybrid_placement_reaches_the_same_worker_for_serve_and_generate(self):
        base = ['--engine', 'hyle', '--inputs', 'prepared', '--tokenizer-dir', 'tokenizer',
                '--backend', 'vulkan', '--cpu-ffn-layers', '2']
        for suffix in (['serve'], ['generate', '--text', 'hello']):
            args = parse_arguments([suffix[0], *base, *suffix[1:]])
            for choices in (None, read_configuration(args)):
                command = make_worker_command(args, choices)
                self.assertEqual(command[command.index('--cpu-ffn-layers') + 1], '2')
                self.assertEqual(command[command.index('--backend') + 1], 'vulkan')
        for value in ('-1', '18446744073709551616'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_arguments(['serve', *base[:-1], value])
        with self.assertRaises(SystemExit):
            parse_arguments(['serve', *base, '--backend', 'cpu'])

    def config(self, text):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / '.scratch')
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'service.toml'; path.write_text(text)
        args = parse_arguments(['serve', '--engine', 'hyle.exe', '--inputs', 'model',
                               '--tokenizer-dir', 'tokenizer', '--max-sessions', '3', '--config', str(path)])
        return args, read_configuration(args)

    def test_named_choices_preserve_independent_execution_sampling_and_resource_axes(self):
        args, choices = self.config('''
[default_profile]
draft_count=1
[[profiles]]
name="subset"
draft_count=3
generate={languages="hu,en"}
[[profiles]]
name="plain"
draft_count=0
[[samplers]]
name="chat"
temperature=0.8
top_k=40
top_p=0.95
[sessions]
capacity=32
kv_capacity=2
[prefix_cache]
entries=4
shared=false
''')
        self.assertEqual(len(choices.executions), 3)
        selected = choices.select({'execution_profile': 'plain', 'sampling_profile': 'chat', 'seed': 7})
        self.assertEqual((selected.execution_profile, selected.sampling_index, selected.seed), (2, 1, 7))
        self.assertEqual(choices.sessions.capacity, 32)
        self.assertEqual(choices.sessions.kv_capacity, 2)
        self.assertEqual(choices.prefix_entries, 4)
        self.assertEqual(choices.select(temperature=.8, top_p=.95).sampling_index, 1)
        command = make_worker_command(args, choices)
        self.assertIn('plain', command)
        self.assertIn('random,0.8,40,0.95', command)
        self.assertEqual(command[command.index('--prefix-cache-entries') + 1], '4')

    def test_unprepared_requests_and_client_cache_domains_are_rejected(self):
        _, choices = self.config('')
        for value in ({'execution_profile': 'missing'}, {'cache_sharing_domain': 1}, {'seed': True},
                      {'max_draft_tokens': -1}, {'sampling_profile': 'missing'}):
            with self.subTest(value=value), self.assertRaises(ValueError): choices.select(value)
        with self.assertRaisesRegex(ValueError, 'cold prepared'): choices.select(temperature=.123)

    def test_conflicting_prepared_depth_and_sampling_are_rejected_before_model_work(self):
        _, choices = self.config('''
[default_profile]
draft_count=1
[[samplers]]
name="chat"
temperature=0.8
top_p=0.95
''')
        with self.assertRaises(ValueError): choices.select({'max_draft_tokens': 2})
        with self.assertRaises(ValueError): choices.select({'sampling_profile': 'chat'}, temperature=1.)

    def test_invalid_cold_configuration_has_no_silent_default(self):
        for text in ('[unrecognized]\nx=1', '[[profiles]]\nname="default"\ndraft_count=1',
                     '[sessions]\nkv_capacity=4', '[sessions]\nttl_seconds=0',
                     '[prefix_cache]\nshared="yes"', '[[samplers]]\nname="bad"\ntop_k=-1'):
            with self.subTest(text=text), self.assertRaises(ValueError): self.config(text)


if __name__ == '__main__': unittest.main()
