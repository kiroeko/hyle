from concurrent.futures import Future
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
from itertools import count
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'inference_concurrency_bench', ROOT / 'scripts/inference-concurrency-bench.py')
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


class Request:
    def __init__(self, first_token):
        self.events = []
        self.acknowledged = 0
        self.append([first_token], 320)

    def read(self, timeout):
        return self.events.pop(0)

    def acknowledge(self):
        self.acknowledged += 1

    def append(self, prompt, quota, timeout=None):
        self.events.extend((SimpleNamespace(kind='token', token=prompt[0]), SimpleNamespace(kind='turn_end')))

    def finish_input(self, timeout):
        self.events.append(SimpleNamespace(kind='completed'))


class SequentialReaders:
    def submit(self, call, *args):
        result = Future()
        result.set_result(call(*args))
        return result


class ConcurrencyBenchmarkTest(unittest.TestCase):
    def test_early_eos_admits_peers_and_consumes_the_boundary_once(self):
        opened = []

        def open_session(prompt, quota, seed):
            request = Request(prompt[0])
            opened.append(request)
            return request

        with patch.object(bench, 'monotonic', side_effect=count().__next__):
            streams, values = bench.coding_review(SimpleNamespace(open_session=open_session), SequentialReaders(),
                                                  (((1,), (2,)), ((3,), (4,))), 320, 1, 83)
        self.assertEqual([s.tokens for s in streams], [[1], [2], [3], [4]])
        self.assertEqual([r.acknowledged for r in opened], [4, 4])
        self.assertEqual(values['overlap_s'], 0)
        self.assertIsNone(values['concurrent_decode_tps'])
        self.assertIsNone(values['itl_p95_ms'])
        self.assertGreater(values['combined_tps'], 0)

    def test_custom_turns_fit_larger_context_and_rotate_within_matched_blocks(self):
        rows = ((1,) * 9400, (2,) * 90, (3,) * 5800, (4,) * 4400,
                (5,) * 4600, (6,) * 2000, (7,) * 2200, (8,) * 300)
        arguments = ['bench', '--engine', str(ROOT), '--engine-version', 'trial',
                     '--baseline', str(ROOT), '--baseline-version', 'main',
                     '--inputs', str(ROOT), '--sessions', '2', '--agent-turns', 'custom.ids',
                     '--capacity', '24576', '--output-tokens', '320', '--arrival-token', '83']
        seen = []

        def run(_, args, mode, prompts, vocabulary):
            self.assertEqual((args.capacity, args.arrival_token), (24576, 83))
            seen.append((mode, tuple(user[0][0] for user in prompts)))
            return {'combined_tps': 1.0, 'concurrent_decode_tps': None}, 2, [320] * 8, 'ids'

        with patch.object(sys, 'platform', 'darwin'), patch.object(sys, 'argv', arguments), \
             patch.object(bench, 'token_prompts', return_value=rows), patch.object(bench, 'run', side_effect=run), \
             redirect_stdout(StringIO()) as log:
            bench.main()
        self.assertEqual(len(seen), 18)
        for block in range(3):
            expected = (5, 1) if block == 1 else (1, 5)
            self.assertEqual({order for _, order in seen[block*6:block*6+6]}, {expected})
        self.assertIn('insufficient paired coverage', log.getvalue())

        arguments[arguments.index('24576')] = '16384'
        with patch.object(sys, 'platform', 'darwin'), patch.object(sys, 'argv', arguments), \
             patch.object(bench, 'token_prompts', return_value=rows), patch.object(bench, 'run') as invoke, \
             redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.main()
        invoke.assert_not_called()


if __name__ == '__main__':
    unittest.main()
