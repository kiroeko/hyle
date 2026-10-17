#!/usr/bin/env python3
"""Check generated visibility and preprocess a tiny fixture; no product build."""

import argparse
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
COMPILER = shutil.which("cl") or shutil.which("clang++") or shutil.which("c++")
spec = importlib.util.spec_from_file_location("hyle_visibility_generator", ROOT / "scripts/generate-visibility.py")
generator = importlib.util.module_from_spec(spec)
sys.dont_write_bytecode = True
spec.loader.exec_module(generator)


class VisibilityGenerationTest(unittest.TestCase):
    def test_checked_in_header(self):
        result, _ = generator.expected(ROOT)
        self.assertEqual(result, (ROOT / "include/hyle/visibility.h").read_text(encoding="utf-8"))

    def test_legacy_io_and_header_only_names(self):
        owners = generator.owners(ROOT)
        self.assertEqual(owners["HYLE_IO_API"], "HYLE_IO_CORE_BUILD")
        self.assertEqual(owners["HYLE_CONTAINER_CONCURRENT_API"], "HYLE_CONTAINER_CONCURRENT_BUILD")
        self.assertEqual(owners["HYLE_CONTAINER_QUEUE_API"], "HYLE_CONTAINER_QUEUE_BUILD")

    def test_unknown_owner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no owning factor"):
            generator.build_macro("HYLE_UNKNOWN_FACTOR_API", {"core"})


class VisibilityPreprocessorTest(unittest.TestCase):
    def test_static_build_and_import(self):
        if not COMPILER:
            self.skipTest("supply --compiler to check the preprocessor fixture")
        fixture = ROOT / "tests/tooling/visibility_fixture.cc"
        msvc = Path(COMPILER).stem.lower() in {"cl", "clang-cl"}
        base = [COMPILER, "/nologo", "/EP", "/Zc:preprocessor", f"/I{ROOT / 'include'}", "/D_WIN32"] if msvc else [
            COMPILER, "-E", "-P", "-x", "c++", "-D_WIN32", f"-I{ROOT / 'include'}"]
        define = "/D" if msvc else "-D"
        names = {"compute", "graph", "io", "container", "concurrent", "queue"}
        cases = [([], set()), (["HYLE_COMPUTE_BUILD"], {"compute"}),
                 (["HYLE_IO_CORE_BUILD"], {"io"}),
                 (["HYLE_CONTAINER_CONCURRENT_BUILD"], {"concurrent"}),
                 (["HYLE_STATIC", "HYLE_COMPUTE_BUILD", "HYLE_IO_CORE_BUILD"], None)]
        for definitions, exported in cases:
            with self.subTest(definitions=definitions):
                result = subprocess.run(base + [define + item for item in definitions] + [str(fixture)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = dict(re.findall(r'visibility_(\w+)\s*=\s*"([^"]*)"\s*;', result.stdout))
                expected = {name: "" if exported is None else
                            "__declspec(dllexport)" if name in exported else "__declspec(dllimport)"
                            for name in names}
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--compiler", default=COMPILER)
    args, remaining = parser.parse_known_args()
    COMPILER = args.compiler
    unittest.main(argv=[sys.argv[0], *remaining])
