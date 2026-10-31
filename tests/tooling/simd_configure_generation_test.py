"""Exercise real configure-time SIMD outputs without a compiler or SDK build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".scratch/validation/simd-configure-generation"
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
ARGUMENTS = argparse.ArgumentParser(add_help=False)
ARGUMENTS.add_argument("--cmake", default=shutil.which("cmake"))
OPTIONS, TEST_ARGUMENTS = ARGUMENTS.parse_known_args()


class SimdConfigureGenerationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(OPTIONS.cmake, "pass --cmake with the actual CMake executable")
        SCRATCH.mkdir(parents=True, exist_ok=True)
        # Keep the small configure fixture/receipts. Repository cache cleanup
        # is explicit; this test never recursively deletes a build directory.
        self.directory = Path(tempfile.mkdtemp(prefix="configure-", dir=SCRATCH))
        self.source = self.directory / "source"
        self.build = self.directory / "build"
        self.source.mkdir()
        (self.source / "impl.h").write_text("// Configure-only kernel body.\n", encoding="utf-8")
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            'project(SimdConfigureGeneration LANGUAGES NONE)\n'
            'set(HYLE_TARGET_PROCESSOR x86_64)\n'
            'function(hyle_set_isa source tier)\nendfunction()\n'
            f'include("{ROOT.as_posix()}/cmake/HyleDispatch.cmake")\n'
            'hyle_simd_target_manifest(OUT_VARIANTS variants OUT_HEADER manifest)\n'
            'add_custom_target(first)\n'
            'add_custom_target(second)\n'
            'if(MODE STREQUAL "union")\n'
            '  hyle_add_dispatched_kernel(OWN_TARGET_SET TARGET first NAME shared\n'
            '    IMPL impl.h TARGETS scalar sse4 avx2 avx512)\n'
            '  hyle_add_dispatched_kernel(OWN_TARGET_SET TARGET second NAME shared\n'
            '    IMPL impl.h TARGETS scalar avx2)\n'
            'elseif(MODE STREQUAL "own-narrow")\n'
            '  hyle_add_dispatched_kernel(OWN_TARGET_SET TARGET first NAME shared\n'
            '    IMPL impl.h TARGETS scalar avx2)\n'
            'else()\n'
            '  hyle_add_dispatched_kernel(TARGET first NAME kernel\n'
            '    IMPL impl.h TARGETS scalar sse4 avx2 avx512)\n'
            'endif()\n'
            'if(EMIT_OBSOLETE)\n'
            '  hyle_add_dispatched_kernel(OWN_TARGET_SET TARGET first NAME obsolete\n'
            '    IMPL impl.h TARGETS scalar sse4 avx2 avx512)\n'
            'endif()\n'
            'get_target_property(first_sources first SOURCES)\n'
            'get_target_property(second_sources second SOURCES)\n'
            'file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/target-sources.txt"\n'
            '  CONTENT "${first_sources}\n${second_sources}\n" @ONLY)\n',
            encoding="utf-8",
        )
        self.attempt = 0

    def configure(self, *, exact="", mode="normal", obsolete=False):
        self.attempt += 1
        receipt = self.directory / f"configure-{self.attempt}"
        receipt.mkdir()
        command = [
            OPTIONS.cmake, "-S", str(self.source), "-B", str(self.build),
            f"-DHYLE_SIMD_EXACT_TARGET:STRING={exact}", f"-DMODE:STRING={mode}",
            f"-DEMIT_OBSOLETE:BOOL={'ON' if obsolete else 'OFF'}",
        ]
        (receipt / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run(command, text=True, capture_output=True, **NO_WINDOW)
        (receipt / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (receipt / "stderr.log").write_text(result.stderr, encoding="utf-8")
        (receipt / "result.json").write_text(
            json.dumps({"exit_code": result.returncode, "CXX": 0, "SDK_build": 0}) + "\n", encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        outputs = self.outputs()
        (receipt / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
        return outputs

    @property
    def generated(self):
        return self.build / "generated/simd-dispatch"

    def outputs(self):
        return {
            path.relative_to(self.generated).as_posix(): (
                hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in self.generated.rglob("*") if path.is_file()
        }

    def wrappers(self, name):
        return {path.name for path in (self.generated / name).glob("*.cc")}

    def test_reconfigure_preserves_selected_outputs_and_reclaims_only_obsolete_wrappers(self):
        fat = self.configure()
        self.assertEqual(self.wrappers("kernel"),
                         {f"kernel_{variant}.cc" for variant in ("scalar", "sse4", "avx2", "avx512")})
        # An unrelated file, even one with a generated-looking basename, is
        # not owned without this helper's exact first-line marker.
        foreign = self.generated / "foreign/foreign_scalar.cc"
        foreign.parent.mkdir()
        foreign.write_text("// This file belongs to another generator.\n", encoding="utf-8")
        sentinel = self.generated / "keep.txt"
        sentinel.write_text("Unrelated configure output.\n", encoding="utf-8")
        before = self.outputs()
        self.assertEqual(self.configure(), before)

        exact = self.configure(exact="AVX2")
        self.assertEqual(self.wrappers("kernel"), {"kernel_avx2.cc"})
        self.assertEqual(exact["kernel/kernel_avx2.cc"], fat["kernel/kernel_avx2.cc"])
        self.assertEqual(self.configure(exact="AVX2"), exact)
        self.assertEqual(exact["foreign/foreign_scalar.cc"], before["foreign/foreign_scalar.cc"])
        self.assertEqual(exact["keep.txt"], before["keep.txt"])
        fat_again = self.configure()
        self.assertEqual(self.wrappers("kernel"),
                         {f"kernel_{variant}.cc" for variant in ("scalar", "sse4", "avx2", "avx512")})
        self.assertEqual(fat_again["kernel/kernel_avx2.cc"], fat["kernel/kernel_avx2.cc"])

        # The later narrow call must not delete the first target's wider set.
        # OWN_TARGET_SET deliberately ignores the scalar product manifest.
        union = self.configure(exact="SCALAR", mode="union", obsolete=True)
        all_shared = {f"shared_{variant}.cc" for variant in ("scalar", "sse4", "avx2", "avx512")}
        self.assertEqual(self.wrappers("shared"), all_shared)
        self.assertEqual(len(self.wrappers("obsolete")), 4)
        self.assertFalse(self.wrappers("kernel"))
        self.assertEqual(self.configure(exact="SCALAR", mode="union", obsolete=True), union)
        sources = (self.build / "target-sources.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sources[1].split(";")), 2)
        self.assertTrue(all((self.build / path).exists() for line in sources for path in line.split(";") if path))

        # Removing the entire obsolete family and changing the own target set
        # reclaim only its unselected wrappers, retaining the surviving bytes.
        narrow = self.configure(exact="SCALAR", mode="own-narrow")
        self.assertEqual(self.wrappers("shared"), {"shared_scalar.cc", "shared_avx2.cc"})
        self.assertFalse(self.wrappers("obsolete"))
        for name in ("shared_scalar.cc", "shared_avx2.cc"):
            self.assertEqual(narrow[f"shared/{name}"], union[f"shared/{name}"])
        self.assertEqual(self.configure(exact="SCALAR", mode="own-narrow"), narrow)
        self.assertIn("foreign/foreign_scalar.cc", narrow)
        self.assertIn("keep.txt", narrow)
        (self.directory / "verification.json").write_text(json.dumps({
            "status": "PASS", "configure_processes": self.attempt, "CXX": 0, "SDK_build": 0,
            "unchanged_content_and_mtime": True, "fat_exact_round_trip": True,
            "multi_target_union": True, "own_target_set_change": True,
            "removed_kernel_reclaimed": True, "unowned_files_preserved": True,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *TEST_ARGUMENTS])
