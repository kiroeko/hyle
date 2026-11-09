"""Guard direct source dependencies between inference layers."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)
SOURCE_SUFFIXES = {'.h', '.hpp', '.c', '.cc', '.cpp', '.cu', '.cuh', '.mm'}


def under(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def resolved_include(source: Path, name: str) -> Path | None:
    # Relative headers and public Hyle headers; system and external headers
    # remain the compiler/build system's responsibility.
    for candidate in (source.parent / name, ROOT / 'include' / name, ROOT / name):
        if candidate.is_file():
            return candidate.resolve()
    return None


class InferenceDependencyTest(unittest.TestCase):
    def test_resolves_relative_and_public_headers(self):
        source = ROOT / 'engine/reporting/example.cc'
        self.assertEqual(resolved_include(source, '../models/qwen35/model_core.h'),
                         (ROOT / 'engine/models/qwen35/model_core.h').resolve())
        self.assertEqual(resolved_include(source, 'hyle/graph/graph.h'),
                         (ROOT / 'include/hyle/graph/graph.h').resolve())

    def test_shared_layers_do_not_include_model_or_higher_layers(self):
        rules = (
            ('engine/inference', ('engine/models', 'engine/text', 'engine/reporting')),
            ('engine/reporting', ('engine/models', 'engine/text')),
            ('include/hyle/compute', ('engine', 'include/hyle/graph', 'src/graph')),
            ('src/compute', ('engine', 'include/hyle/graph', 'src/graph')),
            ('include/hyle/graph', ('engine',)),
            ('src/graph', ('engine',)),
        )
        violations = []
        for source_root, forbidden_roots in rules:
            forbidden = [(ROOT / path).resolve() for path in forbidden_roots]
            for source in (ROOT / source_root).rglob('*'):
                if not source.is_file() or source.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                for name in INCLUDE.findall(source.read_text(encoding='utf-8')):
                    target = resolved_include(source, name)
                    if target is not None and any(under(target, layer) for layer in forbidden):
                        violations.append(f'{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}')
        self.assertFalse(violations, '\n'.join(violations))


if __name__ == '__main__':
    unittest.main()
