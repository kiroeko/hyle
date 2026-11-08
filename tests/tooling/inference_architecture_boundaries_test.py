"""Static dependency boundaries for the inference kernel and text runtime.

This checks direct source imports/includes, not runtime correctness or whether
an untested model/backend is supported. No model, SDK configuration or build is
needed. The existing engine-components CTest imports this test class.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
BACKEND_GATE = re.compile(r'^\s*#\s*(?:if|ifdef|ifndef|elif)\b.*\bHYLE_COMPUTE_(?:CPU|CUDA|APPLE|VULKAN)\b')
CPP_SUFFIXES = {'.h', '.hpp', '.c', '.cc', '.cpp', '.cu', '.cuh', '.m', '.mm', '.inl', '.ipp', '.inc', '.glsl', '.comp'}


def source_files(roots):
    for root in roots:
        if not root.is_dir():
            raise AssertionError(f'missing architecture root: {root}')
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in CPP_SUFFIXES:
                yield path


def include_targets(source: Path, name: str):
    """Check local, repository and public-include resolution, even for new files."""
    name = name.replace('\\', '/')
    return tuple(Path(os.path.normpath(os.path.join(base, name)))
                 for base in (source.parent, ROOT, ROOT / 'include'))


def forbidden_include(source: Path, name: str, forbidden):
    return any(target == root or target.is_relative_to(root)
               for target in include_targets(source, name) for root in forbidden)


def inspect_includes(roots, forbidden):
    violations = []
    count = 0
    for path in source_files(roots):
        count += 1
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            match = INCLUDE.match(line)
            if match and forbidden_include(path, match.group(1), forbidden):
                violations.append(f'{path.relative_to(ROOT)}:{line_number}: {match.group(1)}')
    if not count:
        raise AssertionError('architecture source set is empty')
    return violations


class ArchitectureBoundariesTest(unittest.TestCase):
    def test_compute_and_graph_do_not_depend_on_engine(self):
        roots = [ROOT / 'include/hyle/compute', ROOT / 'include/hyle/graph']
        roots += [p for p in (ROOT / 'src').iterdir()
                  if p.is_dir() and (p.name.startswith('compute') or p.name.startswith('graph'))]
        self.assertEqual(inspect_includes(roots, [ROOT / 'engine']), [])

    def test_compute_does_not_depend_on_graph(self):
        compute = [ROOT / 'include/hyle/compute']
        compute += [p for p in (ROOT / 'src').iterdir() if p.is_dir() and p.name.startswith('compute')]
        graph = [ROOT / 'include/hyle/graph']
        graph += [p for p in (ROOT / 'src').iterdir() if p.is_dir() and p.name.startswith('graph')]
        self.assertEqual(inspect_includes(compute, graph), [])

    def test_common_inference_does_not_depend_on_models_or_frontends(self):
        self.assertEqual(inspect_includes([ROOT / 'engine/inference'],
            [ROOT / 'engine/models', ROOT / 'engine/text', ROOT / 'engine/reporting']), [])

    def test_compute_and_graph_generators_do_not_import_engine(self):
        violations = []
        roots = [p for p in (ROOT / 'src').iterdir()
                 if p.is_dir() and (p.name.startswith('compute') or p.name.startswith('graph'))]
        for root in roots:
            for path in root.rglob('*.py'):
                tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imports = [node.module or '']
                    else:
                        continue
                    for module in imports:
                        if module == 'engine' or module.startswith('engine.'):
                            violations.append(f'{path.relative_to(ROOT)}:{node.lineno}: {module}')
        self.assertEqual(violations, [])

    def test_checker_detects_reverse_edges_before_files_exist(self):
        common = ROOT / 'engine/inference/future.h'
        compute = ROOT / 'src/compute/future.cc'
        self.assertTrue(forbidden_include(common, '../models/example/model.h', [ROOT / 'engine/models']))
        self.assertTrue(forbidden_include(common, '../text/frontend.h', [ROOT / 'engine/text']))
        self.assertTrue(forbidden_include(compute, '../../engine/inference/request.h', [ROOT / 'engine']))
        self.assertTrue(forbidden_include(compute, 'hyle/graph/future.h', [ROOT / 'include/hyle/graph']))
        self.assertFalse(forbidden_include(compute, 'hyle/compute/future.h', [ROOT / 'include/hyle/graph']))

    def test_common_request_and_route_policies_do_not_compile_in_backend_choices(self):
        files = ('request_admission.h', 'request_loop.h', 'generation_dispatcher.h', 'token_worker.h',
                 'mtp_policy.h', 'decode_route_policy.h', 'model_provider.h', 'model_executor.h')
        violations = []
        for name in files:
            path = ROOT / 'engine/inference' / name
            for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if BACKEND_GATE.match(line):
                    violations.append(f'{name}:{line_number}: {line.strip()}')
        self.assertEqual(violations, [])

    def test_text_core_does_not_import_protocol_or_model_implementations(self):
        allowed = {'__init__': set(), 'contracts': set(), 'codec': set(), 'worker_client': {'contracts'},
                   'runtime': {'contracts', 'codec'}, 'server': {'contracts', 'runtime'}}
        violations = []
        for name, local_allowed in allowed.items():
            path = ROOT / 'engine/text' / (name + '.py')
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 1:
                        violations.append(f'{name}:{node.lineno}: parent package import')
                        continue
                    imports = ([node.module] if node.module else [alias.name for alias in node.names])
                    if node.level == 1:
                        for module in imports:
                            if module.split('.')[0] not in local_allowed:
                                violations.append(f'{name}:{node.lineno}: .{module}')
                        continue
                else:
                    continue
                for module in imports:
                    if ((name != 'server' and module.split('.')[0] in {'http', 'fastapi', 'flask', 'starlette'}) or
                            module.startswith('engine.')):
                        violations.append(f'{name}:{node.lineno}: {module}')
        self.assertEqual(violations, [])


if __name__ == '__main__':
    unittest.main()
