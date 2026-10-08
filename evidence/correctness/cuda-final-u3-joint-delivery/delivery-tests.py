"""Lightweight process/control checks, never CUDA qualification."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('delivery', Path(__file__).with_name('delivery.py'))
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryControl(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hyle-delivery-control-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / 'fixed-state'
        (self.root / 'a.cc').write_text('one')
        (self.root / 'b.cc').write_text('one')
        self.jobs = [self.job('a', 'a.cc'), self.job('b', 'b.cc')]
        self.runner = delivery.Runner(self.root, self.jobs, self.state)

    def job(self, name, source, code=None):
        code = code or "from pathlib import Path; p=Path('starts'); p.write_text(p.read_text()+'1' if p.exists() else '1')"
        return {'id': name, 'inputs': [source], 'argv': [sys.executable, '-c', code], 'timeout': 5}

    def starts(self):
        path = self.root / 'starts'
        return len(path.read_text()) if path.exists() else 0

    def test_duplicate_and_changed_log_directory_do_not_start_process(self):
        self.assertEqual(self.runner.perform('a', self.root / 'logs-1'), 'RUN')
        second = delivery.Runner(self.root, self.jobs, self.state)
        self.assertEqual(second.perform('a', self.root / 'logs-2'), 'REUSE')
        self.assertEqual(self.starts(), 1)

    def test_document_change_does_not_invalidate(self):
        self.runner.perform('a', self.root / 'logs')
        (self.root / 'README.md').write_text('documentation change')
        self.assertEqual(self.runner.perform('a', self.root / 'new-logs'), 'REUSE')
        self.assertEqual(self.starts(), 1)

    def test_changed_input_only_invalidates_its_consumer(self):
        self.runner.perform('a', self.root / 'logs')
        self.runner.perform('b', self.root / 'logs')
        (self.root / 'a.cc').write_text('two')
        self.assertEqual(self.runner.verdict('a')[0], 'STOP')
        self.assertEqual(self.runner.verdict('b')[0], 'REUSE')
        self.assertEqual(self.runner.perform('a', self.root / 'logs', 'a.cc changed'), 'RUN')
        (self.root / 'a.cc').write_text('three')
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.runner.perform('a', self.root / 'new-logs', 'another change')
        self.assertEqual(self.starts(), 3)

    def test_failed_task_needs_changed_input_and_reason(self):
        script = self.root / 'program.py'
        script.write_text('raise SystemExit(2)')
        job = {'id': 'fail', 'inputs': ['program.py'], 'argv': [sys.executable, str(script)], 'timeout': 5}
        runner = delivery.Runner(self.root, [job], self.state)
        with self.assertRaisesRegex(ValueError, 'failed'):
            runner.perform('fail', self.root / 'first')
        with self.assertRaises(ValueError):
            runner.perform('fail', self.root / 'other', 'unchanged retry')
        script.write_text('raise SystemExit(0)')
        with self.assertRaises(ValueError):
            runner.perform('fail', self.root / 'other')
        runner.perform('fail', self.root / 'other', 'corrected declared program.py')
        self.assertEqual(len(runner.history('fail')), 2)

    def test_unknown_task_is_rejected_before_subprocess(self):
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            self.runner.perform('unlisted', self.root / 'logs')
        self.assertEqual(self.starts(), 0)

    def test_interrupted_command_is_persisted_and_stopped(self):
        with patch.object(subprocess, 'run', side_effect=KeyboardInterrupt):
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                self.runner.perform('a', self.root / 'logs')
        self.assertEqual(self.runner.history('a')[-1]['status'], 'FAIL')
        self.assertEqual(self.runner.verdict('a')[0], 'STOP')

    def test_changed_or_missing_proof_does_not_restart(self):
        self.runner.perform('a', self.root / 'logs')
        (self.root / 'logs/a-1.log').write_text('changed')
        self.assertEqual(self.runner.verdict('a')[0], 'STOP')
        with self.assertRaises(ValueError):
            self.runner.perform('a', self.root / 'other')
        self.assertEqual(self.starts(), 1)

    def test_one_writer(self):
        with self.runner.locked():
            with self.assertRaises(OSError):
                with delivery.Runner(self.root, self.jobs, self.state).locked():
                    self.fail('A second writer acquired the same task state')

    def test_empty_or_skipped_xml_is_not_pass(self):
        job = self.job('xml', 'a.cc', "from pathlib import Path; Path('empty.xml').write_text('<testsuites/>')")
        job['xml'] = 'empty.xml'
        runner = delivery.Runner(self.root, [job], self.state)
        with self.assertRaisesRegex(ValueError, 'failed'):
            runner.perform('xml', self.root / 'logs')

    def test_fixed_manifest_connects_all_four_remaining_recipes(self):
        manifest = delivery.read(Path(__file__).with_name('delivery.json'))
        jobs = delivery.make_jobs(manifest)
        self.assertEqual(len(manifest['profiles']), 4)
        self.assertEqual(len({j['id'] for j in jobs}), len(jobs))
        for profile in manifest['profiles']:
            selected = [j for j in jobs if j['profile'] == profile['id']]
            self.assertTrue(any(j['id'].endswith('.sdk-build') for j in selected))
            self.assertEqual(sum('.direct-current-' in j['id'] for j in selected), 21)
            self.assertEqual(sum('.graph-current-' in j['id'] for j in selected), 21)
            self.assertTrue(all('.scratch/cuda-u3-' not in ' '.join(j.get('argv', [])) for j in selected))
            self.assertTrue(all(j.get('timeout', 0) > 0 for j in selected))

    def test_paused_product_goal_cannot_launch_registered_build(self):
        private = self.root / 'isolated/evidence/correctness/batch'
        private.mkdir(parents=True)
        script = private / 'delivery.py'
        script.write_bytes(Path(__file__).with_name('delivery.py').read_bytes())
        (private / 'delivery.json').write_text(json.dumps({'run_enabled': False}))
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
        result = subprocess.run([sys.executable, str(script),
            'run', '--task', 'win-mixed-static.build'], capture_output=True, text=True, **flags)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('goal is paused', result.stderr)

    def test_existing_configure_imports_once_without_launching(self):
        job = self.jobs[0]
        receipt = self.root / 'old.json'
        receipt.write_text(json.dumps([{'name': 'configure', 'returncode': 0, 'seconds': 2.0}]))
        job['legacy'] = {'receipt': 'old.json', 'name': 'configure',
            'sha256': self.runner.digest(receipt), 'inputs': self.runner.inputs(job)}
        self.runner.seed(); self.runner.seed()
        self.assertEqual(self.runner.verdict('a')[0], 'REUSE')
        self.assertEqual(len(self.runner.history('a')), 1)
        self.assertEqual(self.starts(), 0)

    def test_old_batch_starters_refuse_before_work(self):
        retired = delivery.read(Path(__file__).with_name('retired-entrypoints.json'))
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
        for entry in retired:
            path = delivery.ROOT / entry
            if path.exists():
                result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, **flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Retired CUDA delivery starter', result.stderr)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows to WSL argument mapping')
    def test_real_wsl_child_writes_the_canonical_xml_path(self):
        code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('<testsuites><testsuite><testcase classname=\"Probe\" name=\"Output\" status=\"run\"/></testsuite></testsuites>')"
        job = {'id': 'wsl-proof', 'inputs': ['a.cc'], 'linux_output': True,
            'argv': ['wsl.exe', '-d', 'Ubuntu-24.04', '--', 'python3', '-c', code, '{logs}/probe.xml'],
            'xml': '{logs}/probe.xml', 'timeout': 20}
        runner = delivery.Runner(self.root, [job], self.state)
        self.assertTrue(delivery.child_logs(job, self.root).startswith('/mnt/'))
        self.assertEqual(delivery.child_logs({}, self.root), self.root.as_posix())
        self.assertEqual(runner.perform('wsl-proof', self.root / 'logs'), 'RUN')
        self.assertEqual(runner.history('wsl-proof')[-1]['tests'], 1)


if __name__ == '__main__':
    unittest.main()
