"""Offline CMake identity checks for original HF in-tree source programs."""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')).hexdigest()


class HuggingFaceSource(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hf-source-', dir=ROOT/'.scratch')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.vendor = self.path/'third_party/fixture'
        self.vendor.mkdir(parents=True)
        (self.vendor/'original.py').write_bytes(b'original = 1\n')
        (self.vendor/'LICENSE').write_bytes(b'Fixture license\n')
        self.entry = {
            'name': 'hf_fixture', 'class': 'in_tree',
            'repository': 'https://huggingface.co/fixture/model',
            'version': 'fixture-v1', 'revision': 'a'*40,
            'source_file': {'path': 'inference/kernel.py', 'vendored_file': 'original.py',
                            'sha256': sha((self.vendor/'original.py').read_bytes())},
            'vendored': {'path': 'third_party/fixture', 'tree_sha256': self.tree()},
            'license': {'expression': 'MIT', 'files': [
                {'root': 'hyle', 'path': 'third_party/fixture/LICENSE',
                 'sha256': sha((self.vendor/'LICENSE').read_bytes())}]} }

    def tree(self):
        manifest = ''.join(p.name+'|'+sha(p.read_bytes())+'\n' for p in sorted(self.vendor.iterdir()))
        return hashlib.sha256(manifest.encode()).hexdigest()

    def run_case(self, entry=None, audit=True):
        lock = json.loads((ROOT/'cmake/HyleDependencyLock.json').read_text(encoding='utf-8'))
        lock['dependencies'].append(entry or self.entry)
        lock_path = self.path/'lock.json'
        lock_path.write_text(json.dumps(lock), encoding='utf-8')
        script = self.path/'check.cmake'
        script.write_text(f'''cmake_minimum_required(VERSION 3.25)
set(HYLE_DEPENDENCY_LOCK_FILE "{lock_path.as_posix()}")
set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE "{lock_path.as_posix()}")
include("{(ROOT/'cmake/HyleDependencyVersions.cmake').as_posix()}")
set(HYLE_DEPENDENCY_SOURCE_ROOT "{self.path.as_posix()}")
'''+('hyle_audit_in_tree_dependencies(hf_fixture)\n' if audit else '')+'''
if(NOT HYLE_DEP_HF_FIXTURE_SOURCE_URL STREQUAL "https://huggingface.co/fixture/model/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/inference/kernel.py")
  message(FATAL_ERROR "source URL projection differs")
endif()
if(NOT HYLE_DEP_HF_FIXTURE_ARCHIVE_URL STREQUAL "")
  message(FATAL_ERROR "HF snapshot acquired a fake archive")
endif()
''', encoding='utf-8')
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        return subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True, **options)

    def test_hf_and_existing_github_projection(self):
        result = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unselected_source_is_not_required(self):
        (self.vendor/'original.py').unlink()
        result = self.run_case(audit=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_identity_cannot_be_replaced_by_new_tree_hash(self):
        (self.vendor/'original.py').write_bytes(b'original = 2\n')
        self.entry['vendored']['tree_sha256'] = self.tree()
        result = self.run_case()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('original source hash mismatch', result.stderr)

    def test_bad_repository_revision_class_and_paths(self):
        mutations = [
            ('repository', 'https://huggingface.co.evil/fixture/model'),
            ('repository', 'http://huggingface.co/fixture/model'),
            ('repository', 'https://huggingface.co/fixture/model?revision=main'),
            ('revision', 'main'), ('class', 'fetchcontent'),
            ('source_file.path', '../kernel.py'), ('source_file.path', '/kernel.py'),
            ('source_file.path', 'inference/./kernel.py'),
            ('source_file.vendored_file', '../original.py'),
            ('source_file.sha256', 'bad')]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                entry = copy.deepcopy(self.entry)
                if '.' in field:
                    first, second = field.split('.')
                    entry[first][second] = value
                else:
                    entry[field] = value
                result = self.run_case(entry)
                self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
